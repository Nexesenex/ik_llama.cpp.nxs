#include "nvapi_poller.h"

#ifdef _WIN32

#define NOMINMAX
#include <windows.h>
#include <nvapi.h>

#ifdef GGML_USE_CUDA
#include "ggml-cuda.h"
#endif

#include <chrono>
#include <cstdio>

// Hysteresis below the temperature limit at which a paused (too-hot) GPU
// resumes polling, avoiding flapping right at the limit (default 85°C -> resume < 80°C).
static constexpr int HEAT_RESUME_HYSTERESIS_C = 5;

static bool get_nvapi_handles(const std::vector<int>& cuda_devices,
                              std::vector<NvPhysicalGpuHandle>& out_handles)
{
    NvAPI_Status status = NvAPI_Initialize();
    if (status != NVAPI_OK) {
        fprintf(stderr, "[NvapiPoller] NvAPI_Initialize failed\n");
        return false;
    }

    NvPhysicalGpuHandle gpu_handles[NVAPI_MAX_PHYSICAL_GPUS] = {0};
    NvU32 gpu_count = 0;
    status = NvAPI_EnumPhysicalGPUs(gpu_handles, &gpu_count);
    if (status != NVAPI_OK) {
        fprintf(stderr, "[NvapiPoller] NvAPI_EnumPhysicalGPUs failed\n");
        return false;
    }

    // Resolve the WDDM-only CUDA ordinal list: explicit list if given (TCC
    // ordinals are skipped), otherwise auto-detect all non-TCC GPUs in ggml
    // logical order — same order the orca heartbeat uses.
    std::vector<int> ordinals;
#ifdef GGML_USE_CUDA
    if (cuda_devices.empty()) {
        const int n_dev = ggml_backend_cuda_get_device_count();
        for (int i = 0; i < n_dev; ++i) {
            if (ggml_backend_cuda_device_is_tcc(i)) continue;
            ordinals.push_back(ggml_backend_cuda_get_device_ordinal(i));
        }
    }
#endif
    if (ordinals.empty()) {
        for (int ord : cuda_devices) {
            ordinals.push_back(ord);
        }
    }

    // Match each CUDA ordinal to a physical GPU by PCI bus ID (the positional
    // mapping breaks when TCC and WDDM GPUs interleave in NVAPI enumeration).
    for (int ord : ordinals) {
        char pci_bus_id[16] = {0};
#ifdef GGML_USE_CUDA
        // find which ggml logical device carries this ordinal, then its bus id
        bool found = false;
        const int n_dev = ggml_backend_cuda_get_device_count();
        for (int i = 0; i < n_dev; ++i) {
            if (ggml_backend_cuda_get_device_ordinal(i) == ord) {
                ggml_backend_cuda_get_device_pci_bus_id(i, pci_bus_id, sizeof(pci_bus_id));
                found = true;
                break;
            }
        }
        if (!found) {
            fprintf(stderr, "[NvapiPoller] WARN: CUDA device %d not present in ggml\n", ord);
            continue;
        }
#endif
        // CUDA bus id format: "0000:01:00.0" → bus is the 2nd field, hex
        unsigned int bus_hex = 0;
        int consumed = 0;
        if (sscanf(pci_bus_id, "%*x:%x:%*x.%*x%n", &bus_hex, &consumed) != 1 || consumed == 0) {
            fprintf(stderr, "[NvapiPoller] WARN: cannot parse PCI bus id '%s'\n", pci_bus_id);
            continue;
        }

        NvPhysicalGpuHandle match = nullptr;
        for (NvU32 k = 0; k < gpu_count; ++k) {
            NvU32 nv_bus = 0;
            if (NvAPI_GPU_GetBusId(gpu_handles[k], &nv_bus) == NVAPI_OK && nv_bus == bus_hex) {
                match = gpu_handles[k];
                break;
            }
        }
        if (match) {
            NvAPI_ShortString gpu_name = {0};
            NvAPI_GPU_GetFullName(match, gpu_name);
            fprintf(stderr, "[NvapiPoller] WDDM[%zu] <- CUDA %d (%s, PCI %s)\n",
                    out_handles.size(), ord, gpu_name, pci_bus_id);
            out_handles.push_back(match);
        } else {
            fprintf(stderr, "[NvapiPoller] WARN: no NVAPI physical GPU for CUDA %d (PCI %s)\n", ord, pci_bus_id);
        }
    }

    return !out_handles.empty();
}

bool nvapi_gpu_temp_ok(int limit_celsius) {
    NvAPI_Status status = NvAPI_Initialize();
    if (status != NVAPI_OK) {
        fprintf(stderr, "[NvapiPoller] NvAPI_Initialize failed\n");
        return true; // fail-open
    }

    NvPhysicalGpuHandle gpu_handles[NVAPI_MAX_PHYSICAL_GPUS] = {0};
    NvU32 gpu_count = 0;
    status = NvAPI_EnumPhysicalGPUs(gpu_handles, &gpu_count);
    if (status != NVAPI_OK) {
        fprintf(stderr, "[NvapiPoller] NvAPI_EnumPhysicalGPUs failed\n");
        NvAPI_Unload();
        return true; // fail-open
    }

    bool ok = true;
    for (NvU32 k = 0; k < gpu_count; ++k) {
        NV_GPU_THERMAL_SETTINGS thermal = {};
        thermal.version = NV_GPU_THERMAL_SETTINGS_VER;
        if (NvAPI_GPU_GetThermalSettings(gpu_handles[k], NVAPI_THERMAL_TARGET_ALL, &thermal) != NVAPI_OK) {
            continue; // ignore GPUs that report no thermal data
        }
        for (int i = 0; i < NVAPI_MAX_THERMAL_SENSORS_PER_GPU; ++i) {
            if (thermal.sensor[i].controller != NVAPI_THERMAL_CONTROLLER_NONE &&
                    thermal.sensor[i].currentTemp >= limit_celsius) {
                fprintf(stderr, "shark: GPU %u temperature %d°C exceeds %d°C limit\n",
                        k, thermal.sensor[i].currentTemp, limit_celsius);
                ok = false;
            }
        }
    }

    NvAPI_Unload();
    return ok;
}

NvapiPoller::NvapiPoller(const std::vector<int>& devices, int interval_ms, int rounds)
    : devices(devices), interval_ms(interval_ms), rounds(rounds) {
}

NvapiPoller::~NvapiPoller() {
    // Blocking teardown: request stop and wait for the thread to exit.
    should_run = false;
    if (worker.joinable()) {
        worker.join();
    }
    running = false;
}

void NvapiPoller::start() {
    if (running.load()) return;
    // Reap a previous thread that already exited after a non-blocking stop().
    // running == false here, so this join is immediate (thread has finished).
    if (worker.joinable()) {
        worker.join();
    }
    should_run = true;
    running = true;
    worker = std::thread(&NvapiPoller::thread_func, this);
}

// Non-blocking stop: only request the thread to exit. The decode thread never
// waits; the worker finishes its current cycle and clears running itself.
// The next start() (or the destructor) reaps the finished thread.
void NvapiPoller::stop() {
    if (!running.load()) return;
    should_run = false;
}

void NvapiPoller::get_hot_state(std::vector<uint8_t> & out) const {
    std::lock_guard<std::mutex> lk(hot_mtx);
    out = hot_state;
}

// No-op callback for the periodic utilization sampling subscription. The real
// "work" is done by the NVIDIA driver continuously computing GPU/FB/VID/BUS
// utilization samples on its backend; this callback merely holds the client alive.
static void __cdecl s_util_cb(NvPhysicalGpuHandle, NV_GPU_CLIENT_CALLBACK_UTILIZATION_DATA_V1 *) {
    // no-op: the periodic sampling itself keeps the backend active
}

void NvapiPoller::thread_func() {
    std::vector<NvPhysicalGpuHandle> handles;
    if (!get_nvapi_handles(devices, handles)) {
        fprintf(stderr, "[NvapiPoller] Failed to get NVAPI handles\n");
        return;
    }

    fprintf(stderr, "[NvapiPoller] Started on %zu device(s), interval=%dms\n",
            handles.size(), interval_ms);

    // Periodic utilization sampling: registering makes the driver continuously
    // compute per-domain (GPU/FB/VID/BUS) utilization on its own backend rather
    // than on a one-shot query. The callback is intentionally a no-op — the work
    // is done by the driver computing the samples, which keeps the GPU busy.
    // One subscription per physical GPU; we must unregister before NvAPI_Unload.
    // Skipped in monitor-only mode: it adds driver load, and --orca only needs
    // the temperature tracking.
    if (!monitor_only.load()) {
        std::vector<NV_GPU_CLIENT_UTILIZATION_PERIODIC_CALLBACK_SETTINGS> util_settings(handles.size());
        for (size_t k = 0; k < handles.size(); ++k) {
            util_settings[k].version = NV_GPU_CLIENT_UTILIZATION_PERIODIC_CALLBACK_SETTINGS_VER;
            util_settings[k].super.super.pCallbackParam = nullptr;
            util_settings[k].super.callbackPeriodms    = (NvU32) interval_ms;
            util_settings[k].callback = s_util_cb;
            NvAPI_Status status = NvAPI_GPU_ClientRegisterForUtilizationSampleUpdates(handles[k], &util_settings[k]);
            if (status != NVAPI_OK) {
                fprintf(stderr, "[NvapiPoller] utilization subscription on GPU %zu failed (0x%x)\n", k, (unsigned) status);
            }
        }
    }

    // Per-card heat state: local_hot[k] = GPU k is too hot to poll. Updated
    // each cycle; a paused GPU resumes once it cools below limit - hysteresis.
    // Published to the member hot_state so llama can feed the heartbeat skip.
    std::vector<uint8_t> local_hot(handles.size(), 0);

    while (should_run.load()) {
        // Per-card temperature protection. Checked first so the ping and burst
        // below respect it for this cycle. A hot card is skipped (no queries,
        // no warmup ping) while the other GPUs keep being polled.
        if (temp_limit.load() > 0) {
            for (size_t k = 0; k < handles.size(); ++k) {
                NV_GPU_THERMAL_SETTINGS thermal = {};
                thermal.version = NV_GPU_THERMAL_SETTINGS_VER;
                if (NvAPI_GPU_GetThermalSettings(handles[k], NVAPI_THERMAL_TARGET_ALL, &thermal) != NVAPI_OK) {
                    continue;
                }
                const int limit = temp_limit.load();
                for (int i = 0; i < NVAPI_MAX_THERMAL_SENSORS_PER_GPU; ++i) {
                    if (thermal.sensor[i].controller == NVAPI_THERMAL_CONTROLLER_NONE) {
                        continue;
                    }
                    const int t = thermal.sensor[i].currentTemp;
                    if (t >= limit && !local_hot[k]) {
                        fprintf(stderr, "[NvapiPoller] GPU %zu at %d°C >= %d°C, pausing polling to protect card\n",
                                k, t, limit);
                        local_hot[k] = true;
                    } else if (t < limit - HEAT_RESUME_HYSTERESIS_C && local_hot[k]) {
                        fprintf(stderr, "[NvapiPoller] GPU %zu cooled to %d°C, resuming polling\n", k, t);
                        local_hot[k] = false;
                    }
                }
            }
        }

        // Publish the per-card heat state so llama can feed the heartbeat skip.
        {
            std::lock_guard<std::mutex> lk(hot_mtx);
            hot_state = local_hot;
        }

        if (monitor_only.load()) {
            // --orca only: monitor temperatures, no NVAPI burst / CUDA ping.
            std::this_thread::sleep_for(std::chrono::milliseconds(interval_ms));
            continue;
        }

#ifdef GGML_USE_CUDA
        // Turn the driver-query chatter into real GPC activity: a full-residency
        // warmup burst per WDDM GPU on each cycle, with per-GPU FMA length
        // (--orca-ping, default ~1 ms). Fire-and-forget, complements the FMA-length
        // tuning of the heartbeat and leaves idle gaps between cycles. Too-hot
        // GPUs are skipped so a paused card is not warmed up.
        ggml_backend_cuda_ping(reinterpret_cast<const bool *>(local_hot.data()), (int) local_hot.size());
#endif

        for (size_t k = 0; k < handles.size(); ++k) {
            if (local_hot[k]) {
                continue; // too hot: skip this card this cycle
            }
            auto handle = handles[k];
            for (int round = 0; round < rounds; ++round) {
                // Aggressive NVAPI burst — forces high P-states via driver queries

                // 1. Dynamic P-states
                NV_GPU_DYNAMIC_PSTATES_INFO_EX dynPstates = {};
                dynPstates.version = NV_GPU_DYNAMIC_PSTATES_INFO_EX_VER;
                NvAPI_GPU_GetDynamicPstatesInfoEx(handle, &dynPstates);

                // 2. P-states 2.0
                NV_GPU_PERF_PSTATES20_INFO pstates20 = {};
                pstates20.version = NV_GPU_PERF_PSTATES20_INFO_VER;
                NvAPI_GPU_GetPstates20(handle, &pstates20);

                // 3. Clocks - query multiple types (current, base, boost)
                NV_GPU_CLOCK_FREQUENCIES clocks = {};
                clocks.version = NV_GPU_CLOCK_FREQUENCIES_VER;

                clocks.ClockType = NV_GPU_CLOCK_FREQUENCIES_CURRENT_FREQ;
                NvAPI_GPU_GetAllClockFrequencies(handle, &clocks);

                clocks.ClockType = NV_GPU_CLOCK_FREQUENCIES_BASE_CLOCK;
                NvAPI_GPU_GetAllClockFrequencies(handle, &clocks);

                clocks.ClockType = NV_GPU_CLOCK_FREQUENCIES_BOOST_CLOCK;
                NvAPI_GPU_GetAllClockFrequencies(handle, &clocks);

                // 4. Thermal settings
                NV_GPU_THERMAL_SETTINGS thermal = {};
                thermal.version = NV_GPU_THERMAL_SETTINGS_VER;
                NvAPI_GPU_GetThermalSettings(handle, NVAPI_THERMAL_TARGET_ALL, &thermal);

                // 5. Memory info
                NV_GPU_MEMORY_INFO_EX memInfo = {};
                memInfo.version = NV_GPU_MEMORY_INFO_EX_VER;
                NvAPI_GPU_GetMemoryInfoEx(handle, &memInfo);

                // 6. Fan tachometer
                NvU32 tach = 0;
                NvAPI_GPU_GetTachReading(handle, &tach);

                // 7. Performance decrease info
                NvU32 perfDec = 0;
                NvAPI_GPU_GetPerfDecreaseInfo(handle, &perfDec);

                // 8. Older P-states info (legacy, still useful)
                NV_GPU_PERF_PSTATES_INFO pstatesInfo = {};
                pstatesInfo.version = NV_GPU_PERF_PSTATES_INFO_VER;
                NvAPI_GPU_GetPstatesInfoEx(handle, &pstatesInfo, 0);

                // 9. GPU core count
                NvU32 coreCount = 0;
                NvAPI_GPU_GetGpuCoreCount(handle, &coreCount);

                // 10. PCI identifiers
                NvU32 deviceId = 0, subSystemId = 0, revisionId = 0, extDeviceId = 0;
                NvAPI_GPU_GetPCIIdentifiers(handle, &deviceId, &subSystemId, &revisionId, &extDeviceId);

                // 11. Bus type / bus ID
                NV_GPU_BUS_TYPE busType = NVAPI_GPU_BUS_TYPE_UNDEFINED;
                NvAPI_GPU_GetBusType(handle, &busType);

                NvU32 busId = 0;
                NvAPI_GPU_GetBusId(handle, &busId);

                // 12. Current P-state (forces P-state reporting path)
                NV_GPU_PERF_PSTATE_ID currentPstate = NVAPI_GPU_PERF_PSTATE_UNDEFINED;
                NvAPI_GPU_GetCurrentPstate(handle, &currentPstate);

                // 13. Current PCIE downstream width (forces PCIE transaction)
                NvU32 pcieWidth = 0;
                NvAPI_GPU_GetCurrentPCIEDownstreamWidth(handle, &pcieWidth);

                // 14. Encoder statistics (engages video encode engine path)
                NV_ENCODER_STATISTICS encStats = {};
                encStats.version = NV_ENCODER_STATISTICS_VER1;
                NvAPI_GPU_GetEncoderStatistics(handle, &encStats);

                // 15. GPU type / system type / SLI state (info paths)
                NV_GPU_TYPE gpuType = NV_SYSTEM_TYPE_GPU_UNKNOWN;
                NvAPI_GPU_GetGPUType(handle, &gpuType);

                NV_SYSTEM_TYPE sysType = NV_SYSTEM_TYPE_UNKNOWN;
                NvAPI_GPU_GetSystemType(handle, &sysType);

                // 16. PCIe slot / IRQ (bus plumbing)
                NvU32 busSlotId = 0;
                NvAPI_GPU_GetBusSlotId(handle, &busSlotId);

                NvU32 irq = 0;
                NvAPI_GPU_GetIRQ(handle, &irq);

                // 17. VBIOS reads (firmware path)
                NvU32 vbiosRev = 0, vbiosOemRev = 0;
                NvAPI_GPU_GetVbiosRevision(handle, &vbiosRev);
                NvAPI_GPU_GetVbiosOEMRevision(handle, &vbiosOemRev);
                NvAPI_ShortString vbiosStr = {0};
                NvAPI_GPU_GetVbiosVersionString(handle, vbiosStr);

                // 18. Frame buffer reads (memory path)
                NvU32 physFb = 0, virtFb = 0;
                NvAPI_GPU_GetPhysicalFrameBufferSize(handle, &physFb);
                NvAPI_GPU_GetVirtualFrameBufferSize(handle, &virtFb);

                // 19. Board info / memory bus width (board EEPROM + memory paths)
                NV_BOARD_INFO boardInfo = {};
                boardInfo.version = NV_BOARD_INFO_VER;
                NvAPI_GPU_GetBoardInfo(handle, &boardInfo);

                NvU32 ramBusWidth = 0;
                NvAPI_GPU_GetRamBusWidth(handle, &ramBusWidth);

                // 20. Architecture info (heavy chip-config read)
                NV_GPU_ARCH_INFO archInfo = {};
                archInfo.version = NV_GPU_ARCH_INFO_VER;
                NvAPI_GPU_GetArchInfo(handle, &archInfo);

                // 21. Virtualization mode
                NV_GPU_VIRTUALIZATION_INFO virtInfo = {};
                virtInfo.version = NV_GPU_VIRTUALIZATION_INFO_VER;
                NvAPI_GPU_GetVirtualizationInfo(handle, &virtInfo);

                // 22. Encoder sessions (engages video encode path)
                NV_ENCODER_SESSIONS_INFO encSessions = {};
                encSessions.version = NV_ENCODER_SESSIONS_INFO_VER;
                NvAPI_GPU_GetEncoderSessionsInfo(handle, &encSessions);

                // 23. GSP features (engages system processor firmware path)
                NV_GPU_GSP_INFO gspInfo = {};
                gspInfo.version = NV_GPU_GSP_INFO_VER;
                NvAPI_GPU_GetGspFeatures(handle, &gspInfo);

                // 24. GPU UUID (dedicated identifier path)
                NV_GPU_UUID gpuUuid = {};
                gpuUuid.version = NV_GPU_UUID_VER;
                NvAPI_GPU_GetUUID(handle, &gpuUuid);

                // 25. Overclock status (engages OC/perf clock control path)
                NV_GPU_OVERCLOCK_STATUS ocStatus = {};
                ocStatus.version = NV_GPU_OVERCLOCK_STATUS_VER;
                NvAPI_GPU_GetOverclockStatus(handle, &ocStatus);

                // 26. HDCP support status (display-protection path)
                NV_GPU_GET_HDCP_SUPPORT_STATUS hdcpStatus = {};
                hdcpStatus.version = NV_GPU_GET_HDCP_SUPPORT_STATUS_VER;
                NvAPI_GPU_GetHDCPSupportStatus(handle, &hdcpStatus);
            }
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(interval_ms));
    }

    // Unregister utilization subscriptions (callback == nullptr means unregister).
    if (!monitor_only.load()) {
        for (size_t k = 0; k < handles.size(); ++k) {
            NV_GPU_CLIENT_UTILIZATION_PERIODIC_CALLBACK_SETTINGS unreg = {};
            unreg.version   = NV_GPU_CLIENT_UTILIZATION_PERIODIC_CALLBACK_SETTINGS_VER;
            unreg.callback  = nullptr;
            NvAPI_GPU_ClientRegisterForUtilizationSampleUpdates(handles[k], &unreg);
        }
    }

    NvAPI_Unload();
    fprintf(stderr, "[NvapiPoller] Stopped\n");
    running = false; // non-blocking stop: mark idle so the next start() can relaunch
}

#endif // _WIN32
