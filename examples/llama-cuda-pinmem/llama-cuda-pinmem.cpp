#include "common.h"
#include "ggml-cuda.h"
#include "llama.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif

#include <cuda_runtime.h>

static bool g_verbose = false;
static bool g_raw_mode = false;

#define LOG_INFO(...)  fprintf(stdout, __VA_ARGS__)
#define LOG_ERR(...)   fprintf(stderr, "error: " __VA_ARGS__)
#define LOG_WARN(...)  fprintf(stdout, "warning: " __VA_ARGS__)
#define LOG_DBG(...)   do { if (g_verbose) fprintf(stdout, "  [debug] " __VA_ARGS__); } while (0)

static void print_usage(const char * prog) {
    LOG_INFO("Usage: %s [options]\n\n", prog);
    LOG_INFO("Options:\n");
    LOG_INFO("  -h, --help       Show this help message\n");
    LOG_INFO("  -v, --verbose    Enable verbose logging\n");
    LOG_INFO("  -m, --max SIZE   Maximum amount to test per device (e.g. 128G, 64G, default: total physical RAM)\n");
    LOG_INFO("  -p, --pinmem N   Set pinmem mode (0-3, default: current setting)\n");
    LOG_INFO("  -a, --all        Test all NVIDIA ordinal devices (default)\n");
    LOG_INFO("  -d, --dev N      Test a specific device by raw CUDA ordinal (repeatable)\n");
    LOG_INFO("  --raw            Use raw CUDA API (no llama backend init).\n");
    LOG_INFO("                   Avoids opening WDDM contexts on other GPUs.\n");
    LOG_INFO("\n");
    LOG_INFO("Pinmem modes:\n");
    LOG_INFO("  0 = No pinned memory\n");
    LOG_INFO("  1 = token_embd only\n");
    LOG_INFO("  2 = Try all, stop on fail (halving approach)\n");
    LOG_INFO("  3 = Pin all (default)\n");
}

static bool parse_size(const char * str, size_t & size) {
    char * end = nullptr;
    double val = std::strtod(str, &end);
    if (end == str) {
        return false;
    }
    switch (*end) {
        case 'G': case 'g': size = (size_t)(val * 1024LL * 1024LL * 1024LL); break;
        case 'M': case 'm': size = (size_t)(val * 1024LL * 1024LL); break;
        case 'K': case 'k': size = (size_t)(val * 1024LL); break;
        case '\0':          size = (size_t)val; break;
        default: return false;
    }
    return true;
}

static size_t get_total_physical_ram() {
#ifdef _WIN32
    MEMORYSTATUSEX status;
    status.dwLength = sizeof(status);
    GlobalMemoryStatusEx(&status);
    return (size_t)status.ullTotalPhys;
#else
    long pages = sysconf(_SC_PHYS_PAGES);
    long page_size = sysconf(_SC_PAGE_SIZE);
    return (size_t)pages * (size_t)page_size;
#endif
}

static size_t get_free_physical_ram() {
#ifdef _WIN32
    MEMORYSTATUSEX status;
    status.dwLength = sizeof(status);
    GlobalMemoryStatusEx(&status);
    return (size_t)status.ullAvailPhys;
#else
    return get_total_physical_ram();
#endif
}

static void print_size(const char * label, size_t bytes, bool newline = true) {
    if (bytes >= (1024LL * 1024LL * 1024LL)) {
        LOG_INFO("%s%.2f GiB", label, bytes / (1024.0 * 1024.0 * 1024.0));
    } else if (bytes >= (1024LL * 1024LL)) {
        LOG_INFO("%s%.2f MiB", label, bytes / (1024.0 * 1024.0));
    } else if (bytes >= 1024LL) {
        LOG_INFO("%s%.2f KiB", label, bytes / 1024.0);
    } else {
        LOG_INFO("%s%zu B", label, bytes);
    }
    if (newline) {
        LOG_INFO("\n");
    }
}

struct TestResult {
    size_t bytes;
    int    iterations;
    double total_ms;
    double alloc_ms;
    double register_ms;
};

static TestResult test_device_pinned_max(int cuda_ordinal, size_t max_test_size) {
    TestResult result = { 0, 0, 0.0, 0.0, 0.0 };

    LOG_INFO("  [1] Setting CUDA device %d (raw ordinal)...\n", cuda_ordinal);
    cudaError_t err = cudaSetDevice(cuda_ordinal);
    if (err != cudaSuccess) {
        LOG_INFO("  [FAIL] cudaSetDevice(%d): %s\n", cuda_ordinal, cudaGetErrorString(err));
        return result;
    }
    int cur = -1;
    cudaGetDevice(&cur);
    LOG_INFO("  [OK]  Device set, cudaGetDevice() = %d\n", cur);

    // Determine if TCC or WDDM via driver version query
    cudaDeviceProp prop;
    err = cudaGetDeviceProperties(&prop, cur);
    if (err == cudaSuccess) {
        LOG_INFO("  [OK]  %s, compute %d.%d, %s\n",
                 prop.name,
                 prop.major, prop.minor,
                 prop.kernelExecTimeoutEnabled ? "WDDM (TCC disabled)" : "TCC (no timeout)");
    }

    LOG_INFO("  [2] Initialising primary context (cudaFree(0))...\n");
    cudaFree(0);
    LOG_INFO("  [OK]  Context initialised\n");

    size_t free_mem = 0, total_mem = 0;
    err = cudaMemGetInfo(&free_mem, &total_mem);
    if (err == cudaSuccess) {
        LOG_INFO("  [3] cudaMemGetInfo: ");
        print_size("", total_mem, false);
        LOG_INFO(" total, ");
        print_size("", free_mem, false);
        LOG_INFO(" free\n");
    } else {
        LOG_INFO("  [3] cudaMemGetInfo: %s\n", cudaGetErrorString(err));
    }

    LOG_INFO("  [4] Starting pinning test (halving from ");
    print_size("", max_test_size, false);
    LOG_INFO(" down to 1 MiB)\n");

    const size_t min_chunk = 1ULL << 20;
    size_t try_size = max_test_size;
    int iteration = 0;

    while (try_size >= min_chunk) {
        iteration++;
        auto t0 = ggml_time_us();

        LOG_INFO("  [%d] Trying ", iteration);
        print_size("", try_size, false);
        LOG_INFO("...");

        void * ptr = VirtualAlloc(NULL, try_size, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
        auto t1 = ggml_time_us();

        if (ptr == nullptr) {
            auto dt = (t1 - t0) / 1000.0;
            LOG_INFO(" VirtualAlloc FAILED (%.1f ms)\n", dt);
            try_size /= 2;
            continue;
        }

        LOG_INFO(" alloc'd (%.1f ms), registering...", (t1 - t0) / 1000.0);

        err = cudaHostRegister(ptr, try_size, cudaHostRegisterPortable);
        auto t2 = ggml_time_us();
        double register_ms = (t2 - t1) / 1000.0;

        if (err == cudaSuccess) {
            result.bytes = try_size;
            result.iterations = iteration;
            result.total_ms = (t2 - t0) / 1000.0;
            result.alloc_ms = (t1 - t0) / 1000.0;
            result.register_ms = register_ms;

            LOG_INFO(" SUCCESS (%.1f ms)\n", register_ms);
            LOG_INFO("  [OK]  Pinned ");
            print_size("", try_size, false);
            LOG_INFO(" on raw CUDA ordinal %d\n", cuda_ordinal);

            cudaHostUnregister(ptr);
            VirtualFree(ptr, 0, MEM_RELEASE);
            break;
        }

        LOG_INFO(" FAILED: %s (%.1f ms)\n", cudaGetErrorString(err), register_ms);
        cudaGetLastError();

        VirtualFree(ptr, 0, MEM_RELEASE);
        try_size /= 2;
    }

    if (result.bytes == 0) {
        LOG_INFO("  [FAIL] Could not pin any host memory on raw CUDA %d after %d iterations\n",
                 cuda_ordinal, iteration);
    }

    return result;
}

int main(int argc, char ** argv) {
    size_t max_size = 0;
    int set_pinmem = -1;
    bool test_all = true;
    std::vector<int> test_devices;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            print_usage(argv[0]);
            return 0;
        } else if (strcmp(argv[i], "-v") == 0 || strcmp(argv[i], "--verbose") == 0) {
            g_verbose = true;
        } else if (strcmp(argv[i], "--raw") == 0) {
            g_raw_mode = true;
        } else if (strcmp(argv[i], "-m") == 0 || strcmp(argv[i], "--max") == 0) {
            if (i + 1 >= argc) { LOG_ERR("-m requires an argument\n"); return 1; }
            if (!parse_size(argv[++i], max_size)) { LOG_ERR("invalid size '%s'\n", argv[i]); return 1; }
        } else if (strcmp(argv[i], "-p") == 0 || strcmp(argv[i], "--pinmem") == 0) {
            if (i + 1 >= argc) { LOG_ERR("-p requires an argument\n"); return 1; }
            set_pinmem = atoi(argv[++i]);
            if (set_pinmem < 0 || set_pinmem > 3) { LOG_ERR("pinmem must be 0-3\n"); return 1; }
        } else if (strcmp(argv[i], "-a") == 0 || strcmp(argv[i], "--all") == 0) {
            test_all = true;
            test_devices.clear();
        } else if (strcmp(argv[i], "-d") == 0 || strcmp(argv[i], "--dev") == 0) {
            if (i + 1 >= argc) { LOG_ERR("-d requires a device number\n"); return 1; }
            test_all = false;
            test_devices.push_back(atoi(argv[++i]));
        } else {
            LOG_ERR("unknown option '%s'\n", argv[i]);
            LOG_INFO("try '%s --help' for more info\n", argv[0]);
            return 1;
        }
    }

    if (max_size == 0) {
        max_size = get_total_physical_ram();
    }

    LOG_INFO("=== CUDA Pinned Memory Test ===\n\n");

    LOG_INFO("System:\n");
    print_size("  Physical RAM total: ", get_total_physical_ram());
    print_size("  Physical RAM free:  ", get_free_physical_ram());
    LOG_INFO("  Mode:               %s\n", g_raw_mode ? "raw CUDA API (no llama backend init)" : "normal (via llama backend)");
    LOG_INFO("\n");

    // -------------------------------------------------------------------
    // Gather device list
    // -------------------------------------------------------------------
    struct DeviceInfo {
        int  raw_ordinal;
        char name[256];
        int  compute_major;
        int  compute_minor;
        bool is_tcc;  // true = kernelExecTimeoutEnabled == false
        size_t free_vram;
        size_t total_vram;
    };
    std::vector<DeviceInfo> devices;
    std::vector<TestResult> results;

    int device_count = 0;

    if (g_raw_mode) {
        // --- RAW MODE: use CUDA API directly, no llama_backend_init ---
        LOG_INFO("Mode: raw CUDA API -- no llama backend contexts opened\n\n");

        cudaError_t err = cudaGetDeviceCount(&device_count);
        if (err != cudaSuccess) {
            LOG_ERR("cudaGetDeviceCount: %s\n", cudaGetErrorString(err));
            return 1;
        }

        LOG_INFO("=== Device enumeration (raw CUDA ordinal order) ===\n");
        for (int i = 0; i < device_count; i++) {
            DeviceInfo info;
            info.raw_ordinal = i;

            cudaDeviceProp prop;
            err = cudaGetDeviceProperties(&prop, i);
            if (err == cudaSuccess) {
                strncpy(info.name, prop.name, sizeof(info.name) - 1);
                info.name[sizeof(info.name) - 1] = '\0';
                info.compute_major = prop.major;
                info.compute_minor = prop.minor;
                info.is_tcc = !prop.kernelExecTimeoutEnabled;
            } else {
                snprintf(info.name, sizeof(info.name), "Unknown (cudaGetDeviceProperties failed)");
                info.compute_major = 0;
                info.compute_minor = 0;
                info.is_tcc = false;
            }

            // Don't call cudaSetDevice/cudaFree(0) here to avoid opening unwanted WDDM contexts
            // cudaMemGetInfo needs an active context; skip it in raw mode until the test
            info.total_vram = 0;
            info.free_vram = 0;

            devices.push_back(info);

            LOG_INFO("  Raw CUDA %d: %s", i, info.name);
            if (err == cudaSuccess) {
                LOG_INFO(" (compute %d.%d, %s)", info.compute_major, info.compute_minor,
                         info.is_tcc ? "TCC" : "WDDM");
            }
            LOG_INFO("\n");
        }
        LOG_INFO("\n");

    } else {
        // --- NORMAL MODE: via llama backend ---
        LOG_INFO("Initialising llama backend...\n");
        llama_backend_init();

        if (set_pinmem >= 0) {
            ggml_backend_cuda_set_pinmem(set_pinmem);
        }
        int pinmem = ggml_backend_cuda_get_pinmem();
        device_count = ggml_backend_cuda_get_device_count();

        if (device_count == 0) {
            LOG_ERR("no CUDA devices found\n");
            llama_backend_free();
            return 1;
        }

        LOG_INFO("  CUDA devices:  %d\n", device_count);
        LOG_INFO("  Pinmem mode:   %d", pinmem);
        if (pinmem == 0)      LOG_INFO(" (disabled)");
        else if (pinmem == 1) LOG_INFO(" (token_embd only)");
        else if (pinmem == 2) LOG_INFO(" (try all, stop on fail)");
        else if (pinmem == 3) LOG_INFO(" (pin all - default)");
        LOG_INFO("\n\n");

        LOG_INFO("=== Device enumeration (ggml logical order = nvidia-smi order) ===\n");
        for (int i = 0; i < device_count; i++) {
            DeviceInfo info;
            info.raw_ordinal = ggml_backend_cuda_get_device_ordinal(i);
            ggml_backend_cuda_get_device_description(i, info.name, sizeof(info.name));
            ggml_backend_cuda_get_device_memory(i, &info.free_vram, &info.total_vram);

            // Query TCC/WDDM via raw CUDA
            cudaDeviceProp prop;
            info.is_tcc = false;
            info.compute_major = 0;
            info.compute_minor = 0;
            if (cudaGetDeviceProperties(&prop, info.raw_ordinal) == cudaSuccess) {
                info.is_tcc = !prop.kernelExecTimeoutEnabled;
                info.compute_major = prop.major;
                info.compute_minor = prop.minor;
            }

            devices.push_back(info);

            LOG_INFO("  CUDA%d (raw ordinal %d): %s", i, info.raw_ordinal, info.name);
            if (info.is_tcc) LOG_INFO(" [TCC]");
            LOG_INFO("\n");
            LOG_INFO("    Total VRAM: ");
            print_size("", info.total_vram);
            LOG_INFO("    Free VRAM:  ");
            print_size("", info.free_vram);
        }
        LOG_INFO("\n");
    }

    // -------------------------------------------------------------------
    // Resolve test device list
    // -------------------------------------------------------------------
    std::vector<int> to_test;
    int effective_count = g_raw_mode ? device_count : device_count;

    if (test_all) {
        if (effective_count == 0) {
            LOG_ERR("no CUDA devices found\n");
            if (!g_raw_mode) llama_backend_free();
            return 1;
        }
        for (int i = 0; i < effective_count; i++) {
            to_test.push_back(i);
        }
    } else {
        for (int d : test_devices) {
            if (d < 0 || d >= effective_count) {
                LOG_WARN("device %d out of range (0-%d), skipping\n", d, effective_count - 1);
                continue;
            }
            to_test.push_back(d);
        }
        if (to_test.empty()) {
            LOG_ERR("no valid devices specified\n");
            if (!g_raw_mode) llama_backend_free();
            return 1;
        }
    }

    LOG_INFO("=== Pinned memory testing ===\n");
    print_size("  Max test size per device: ", max_size);
    LOG_INFO("  Devices under test: ");
    for (size_t i = 0; i < to_test.size(); i++) {
        if (i > 0) LOG_INFO(", ");
        int idx = to_test[i];
        if (g_raw_mode) {
            LOG_INFO("raw CUDA %d", devices[idx].raw_ordinal);
        } else {
            LOG_INFO("CUDA%d (ordinal %d)", idx, devices[idx].raw_ordinal);
        }
    }
    LOG_INFO("\n\n");

    // -------------------------------------------------------------------
    // Run tests
    // -------------------------------------------------------------------
    int64_t t_start = ggml_time_us();
    for (size_t idx = 0; idx < to_test.size(); idx++) {
        int list_idx = to_test[idx];
        int raw_ord = devices[list_idx].raw_ordinal;

        LOG_INFO("+%.*s+\n", 60, "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++");
        if (g_raw_mode) {
            LOG_INFO("| Device %d/%d: raw CUDA %d - %-40s |\n",
                     (int)(idx + 1), (int)to_test.size(), raw_ord, devices[list_idx].name);
        } else {
            LOG_INFO("| Device %d/%d: CUDA%d (raw ordinal %d) - %-27s |\n",
                     (int)(idx + 1), (int)to_test.size(),
                     list_idx, raw_ord, devices[list_idx].name);
        }
        LOG_INFO("+%.*s+\n", 60, "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++");

        TestResult r = test_device_pinned_max(raw_ord, max_size);

        if (r.bytes > 0) {
            LOG_INFO("\n  >>> raw CUDA %d: max pinned host memory = ", raw_ord);
            print_size("", r.bytes, false);
            LOG_INFO(" (found in %d iterations, %.1f ms total)\n",
                     r.iterations, r.total_ms);
        } else {
            LOG_INFO("\n  >>> raw CUDA %d: COULD NOT PIN any host memory\n", raw_ord);
        }
        LOG_INFO("\n");

        results.push_back(r);
    }
    int64_t t_end = ggml_time_us();

    // -------------------------------------------------------------------
    // Summary
    // -------------------------------------------------------------------
    LOG_INFO("=== Summary ===\n");
    LOG_INFO("Total test time: %.1f s\n\n", (t_end - t_start) / 1e6);

    LOG_INFO("%-7s %-8s %-5s %-40s  %-22s  %s\n",
             "Test#", "Device", "Mode", "Name", "Max Pinned Host", "Time");
    LOG_INFO("%-7s %-8s %-5s %-40s  %-22s  %s\n",
             "-----", "------", "----", "----", "----------------", "----");

    for (size_t idx = 0; idx < to_test.size(); idx++) {
        int list_idx = to_test[idx];
        const auto & dev = devices[list_idx];
        const auto & r = results[idx];

        char dev_str[16];
        if (g_raw_mode) {
            snprintf(dev_str, sizeof(dev_str), "raw %d", dev.raw_ordinal);
        } else {
            snprintf(dev_str, sizeof(dev_str), "CUDA%d", list_idx);
        }

        char mode_str[8];
        snprintf(mode_str, sizeof(mode_str), "%s", dev.is_tcc ? "TCC" : "WDDM");

        char pinmem_str[24];
        if (r.bytes >= (1024LL * 1024LL * 1024LL)) {
            snprintf(pinmem_str, sizeof(pinmem_str), "%.2f GiB", r.bytes / (1024.0 * 1024.0 * 1024.0));
        } else if (r.bytes > 0) {
            snprintf(pinmem_str, sizeof(pinmem_str), "%.0f MiB", r.bytes / (1024.0 * 1024.0));
        } else {
            snprintf(pinmem_str, sizeof(pinmem_str), "FAILED");
        }

        char time_str[16];
        if (r.iterations > 0) {
            snprintf(time_str, sizeof(time_str), "%.1f s", r.total_ms / 1000.0);
        } else {
            snprintf(time_str, sizeof(time_str), "-");
        }

        LOG_INFO("%-7zu %-8s %-5s %-40s  %-22s  %s\n",
                 idx + 1, dev_str, mode_str, dev.name, pinmem_str, time_str);
    }

    LOG_INFO("\n");
    bool any_success = false;
    for (const auto & r : results) {
        if (r.bytes > 0) { any_success = true; break; }
    }
    if (!any_success) {
        LOG_INFO("NOTE: No device could pin host memory.\n");
        LOG_INFO("  - Try pinmem=2 for the backend halving approach.\n");
        LOG_INFO("  - Check if GGML_CUDA_NO_PINNED env var is set.\n");
        LOG_INFO("  - WDDM driver may limit to ~32 GiB per process.\n");
        LOG_INFO("  - A TCC driver bypasses this limit.\n");
    }

    if (!g_raw_mode) {
        llama_backend_free();
    }
    return any_success ? 0 : 1;
}
