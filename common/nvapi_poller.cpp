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

    // Simple mapping: assume CUDA device order matches physical order
    for (int cuda_idx : cuda_devices) {
        if (cuda_idx >= 0 && cuda_idx < (int)gpu_count) {
            out_handles.push_back(gpu_handles[cuda_idx]);
        }
    }

    return !out_handles.empty();
}

NvapiPoller::NvapiPoller(const std::vector<int>& devices, int interval_ms, int rounds)
    : devices(devices), interval_ms(interval_ms), rounds(rounds) {
}

NvapiPoller::~NvapiPoller() {
    stop();
}

void NvapiPoller::start() {
    if (running.load()) return;
    should_run = true;
    worker = std::thread(&NvapiPoller::thread_func, this);
    running = true;
}

void NvapiPoller::stop() {
    if (!running.load()) return;
    should_run = false;
    if (worker.joinable()) {
        worker.join();
    }
    running = false;
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

    while (should_run.load()) {
#ifdef GGML_USE_CUDA
        // Turn the driver-query chatter into real GPC activity: a tiny warmup
        // kernel per WDDM GPU on each cycle. Cheap (one 4096-FMA block/SM),
        // fire-and-forget, complements the FMA-length tuning of the heartbeat.
        ggml_backend_cuda_ping();
#endif

        for (auto handle : handles) {
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
            }
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(interval_ms));
    }

    // Unregister utilization subscriptions (callback == nullptr means unregister).
    for (size_t k = 0; k < handles.size(); ++k) {
        NV_GPU_CLIENT_UTILIZATION_PERIODIC_CALLBACK_SETTINGS unreg = {};
        unreg.version   = NV_GPU_CLIENT_UTILIZATION_PERIODIC_CALLBACK_SETTINGS_VER;
        unreg.callback  = nullptr;
        NvAPI_GPU_ClientRegisterForUtilizationSampleUpdates(handles[k], &unreg);
    }

    NvAPI_Unload();
    fprintf(stderr, "[NvapiPoller] Stopped\n");
}

#endif // _WIN32
