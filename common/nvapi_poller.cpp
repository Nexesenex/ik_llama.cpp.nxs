#include "nvapi_poller.h"

#ifdef _WIN32

#define NOMINMAX
#include <windows.h>
#include <nvapi.h>

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

NvapiPoller::NvapiPoller(const std::vector<int>& devices, int interval_ms)
    : devices(devices), interval_ms(interval_ms) {
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

void NvapiPoller::thread_func() {
    std::vector<NvPhysicalGpuHandle> handles;
    if (!get_nvapi_handles(devices, handles)) {
        fprintf(stderr, "[NvapiPoller] Failed to get NVAPI handles\n");
        return;
    }

    fprintf(stderr, "[NvapiPoller] Started on %zu device(s), interval=%dms\n",
            handles.size(), interval_ms);

    while (should_run.load()) {
        for (auto handle : handles) {
            // Aggressive metric hammering — forces high P-states

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
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(interval_ms));
    }

    NvAPI_Unload();
    fprintf(stderr, "[NvapiPoller] Stopped\n");
}

#endif // _WIN32
