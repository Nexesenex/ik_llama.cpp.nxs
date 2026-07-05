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
static bool g_bare_mode = false;

#define LOG_INFO(...)  fprintf(stdout, __VA_ARGS__)
#define LOG_ERR(...)   fprintf(stderr, "error: " __VA_ARGS__)
#define LOG_WARN(...)  fprintf(stdout, "warning: " __VA_ARGS__)
#define LOG_DBG(...)   do { if (g_verbose) fprintf(stdout, "  [debug] " __VA_ARGS__); } while (0)

// Unit for the default ascending test list (--unit parameter)
// Default is GiB (1024^3).  The same list of numbers {1,2,3,4,6,8, ...}
// is multiplied by this unit's byte multiplier.
enum test_unit {
    TESTUNIT_MIB,
    TESTUNIT_MB,
    TESTUNIT_GIB,
    TESTUNIT_GB,
};
static enum test_unit g_test_unit = TESTUNIT_GIB;

static size_t unit_mult(enum test_unit u) {
    switch (u) {
        case TESTUNIT_MIB: return 1024ULL * 1024ULL;
        case TESTUNIT_MB:  return 1000ULL * 1000ULL;
        case TESTUNIT_GIB: return 1024ULL * 1024ULL * 1024ULL;
        case TESTUNIT_GB:  return 1000ULL * 1000ULL * 1000ULL;
    }
    return 1024ULL * 1024ULL * 1024ULL;
}
static const char * unit_name(enum test_unit u) {
    switch (u) {
        case TESTUNIT_MIB: return "MiB";
        case TESTUNIT_MB:  return "MB";
        case TESTUNIT_GIB: return "GiB";
        case TESTUNIT_GB:  return "GB";
    }
    return "GiB";
}

static std::vector<size_t> g_seq_sizes; // custom --seq size list, empty = use default

// -----------------------------------------------------------------------
// Pin method specification
// -----------------------------------------------------------------------
enum pin_method_id {
    PIN_METHOD_VA,           // VirtualAlloc + cudaHostRegister(portable)
    PIN_METHOD_MALLOC_P,     // malloc + cudaHostRegister(portable)
    PIN_METHOD_MALLOC_NP,    // malloc + cudaHostRegister(non-portable)
    PIN_METHOD_AMALLOC_P,    // _aligned_malloc(size,32) + cudaHostRegister(portable)
    PIN_METHOD_CHOST,        // cudaHostAlloc (direct pinned)
    PIN_METHOD_COUNT,
    PIN_METHOD_ALL = PIN_METHOD_COUNT,
};

static const char * method_name(enum pin_method_id m) {
    switch (m) {
        case PIN_METHOD_VA:        return "va";
        case PIN_METHOD_MALLOC_P:  return "malloc_p";
        case PIN_METHOD_MALLOC_NP: return "malloc_np";
        case PIN_METHOD_AMALLOC_P: return "amalloc_p";
        case PIN_METHOD_CHOST:     return "chost";
        default:                   return "?";
    }
}
static const char * method_desc(enum pin_method_id m) {
    switch (m) {
        case PIN_METHOD_VA:        return "VirtualAlloc + cudaHostRegister(portable)";
        case PIN_METHOD_MALLOC_P:  return "malloc + cudaHostRegister(portable)";
        case PIN_METHOD_MALLOC_NP: return "malloc + cudaHostRegister(non-portable)";
        case PIN_METHOD_AMALLOC_P: return "_aligned_malloc + cudaHostRegister(portable)";
        case PIN_METHOD_CHOST:     return "cudaHostAlloc (direct pinned)";
        default:                   return "unknown";
    }
}
static enum pin_method_id parse_method(const char * s) {
    if (strcmp(s, "va")        == 0) return PIN_METHOD_VA;
    if (strcmp(s, "malloc_p")  == 0) return PIN_METHOD_MALLOC_P;
    if (strcmp(s, "malloc_np") == 0) return PIN_METHOD_MALLOC_NP;
    if (strcmp(s, "amalloc_p") == 0) return PIN_METHOD_AMALLOC_P;
    if (strcmp(s, "chost")     == 0) return PIN_METHOD_CHOST;
    if (strcmp(s, "all")       == 0) return PIN_METHOD_ALL;
    return (enum pin_method_id)-1;
}

// -----------------------------------------------------------------------
// Method-specific alloc / pin / free helpers
// -----------------------------------------------------------------------
static void * alloc_va(size_t size) {
#ifdef _WIN32
    return VirtualAlloc(NULL, size, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
#else
    (void)size; return nullptr;
#endif
}
static void free_va(void * ptr) {
#ifdef _WIN32
    VirtualFree(ptr, 0, MEM_RELEASE);
#else
    (void)ptr;
#endif
}

static void * alloc_malloc_m(size_t size) { return malloc(size); }
static void free_malloc_m(void * ptr) { free(ptr); }

#ifdef _WIN32
static void * alloc_amalloc(size_t size) { return _aligned_malloc(size, 32); }
static void free_amalloc(void * ptr) { _aligned_free(ptr); }
#else
static void * alloc_amalloc(size_t size) { (void)size; return nullptr; }
static void free_amalloc(void * ptr) { (void)ptr; }
#endif

static cudaError_t pin_portable(void * ptr, size_t size) {
    return cudaHostRegister(ptr, size, cudaHostRegisterPortable);
}
static cudaError_t pin_nonportable(void * ptr, size_t size) {
    return cudaHostRegister(ptr, size, cudaHostRegisterDefault);
}
static void unpin_std(void * ptr) { cudaHostUnregister(ptr); }

static cudaError_t allocpin_chost(void ** ptr, size_t size) {
    return cudaHostAlloc(ptr, size, cudaHostAllocDefault);
}
static void free_chost(void * ptr) { cudaFreeHost(ptr); }

static void print_usage(const char * prog) {
    LOG_INFO("Usage: %s [options]\n\n", prog);
    LOG_INFO("Options:\n");
    LOG_INFO("  -h, --help       Show this help message\n");
    LOG_INFO("  -v, --verbose    Enable verbose logging\n");
    LOG_INFO("  -m, --max SIZE   Maximum amount to test per device (e.g. 128G, 64G, default: total physical RAM)\n");
    LOG_INFO("  -p, --pinmem N   Set pinmem mode (0-3, default: current setting)\n");
    LOG_INFO("  -a, --all        Test all logical CUDA devices (default)\n");
    LOG_INFO("  -d, --dev N      Test a specific device only\n");
    LOG_INFO("  -r, --raw-ord    Interpret -d N as raw CUDA ordinal\n");
    LOG_INFO("  -M, --method M   Allocation method: va | malloc_p | malloc_np | amalloc_p | chost | all\n");
    LOG_INFO("                   (default: va = VirtualAlloc + cudaHostRegister)\n");
    LOG_INFO("  --list-methods   List available allocation methods and exit\n");
    LOG_INFO("  --unit U         Unit for the default ascending sequence: MiB, MB, GiB, GB (default: GiB)\n");
    LOG_INFO("  --list-units     List available test units and exit\n");
    LOG_INFO("  --seq ARG        Custom size sequence: \"start unit,end unit,step unit\"\n");
    LOG_INFO("                   e.g. --seq \"1 GiB,192 GiB,8 GiB\"\n");
    LOG_INFO("                   Units: B, KiB, MiB, GiB (binary), KB, MB, GB (decimal)\n");
    LOG_INFO("  --bare           Minimal init: llama_backend_init() only, then raw CUDA API.\n");
    LOG_INFO("                   Avoids ggml backend context state on all devices.\n");
    LOG_INFO("\n");
    LOG_INFO("Pinmem modes:\n");
    LOG_INFO("  0 = No pinned memory\n");
    LOG_INFO("  1 = token_embd only\n");
    LOG_INFO("  2 = Try all, stop on fail (halving approach)\n");
    LOG_INFO("  3 = Pin all (default)\n");
    LOG_INFO("\n");
    LOG_INFO("Allocation methods:\n");
    for (int i = 0; i < PIN_METHOD_COUNT; i++) {
        LOG_INFO("  %-10s %s\n", method_name((enum pin_method_id)i), method_desc((enum pin_method_id)i));
    }
    LOG_INFO("  %-10s Run all methods on each device (comparison table)\n", "all");
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

// Parse a size with explicit unit suffix: "1 GiB", "192 MB", "8 MiB", etc.
// Accepted units: B, KiB, MiB, GiB (binary), KB, MB, GB (decimal).
static bool parse_explicit_size(const char * str, size_t & bytes) {
    char * end = nullptr;
    double val = std::strtod(str, &end);
    if (end == str) return false;
    // skip spaces before unit
    while (*end == ' ') end++;
    if (strcmp(end, "GiB") == 0 || strcmp(end, "G") == 0 || strcmp(end, "g") == 0) {
        bytes = (size_t)(val * 1024LL * 1024LL * 1024LL);
    } else if (strcmp(end, "GB") == 0) {
        bytes = (size_t)(val * 1000LL * 1000LL * 1000LL);
    } else if (strcmp(end, "MiB") == 0 || strcmp(end, "M") == 0 || strcmp(end, "m") == 0) {
        bytes = (size_t)(val * 1024LL * 1024LL);
    } else if (strcmp(end, "MB") == 0) {
        bytes = (size_t)(val * 1000LL * 1000LL);
    } else if (strcmp(end, "KiB") == 0 || strcmp(end, "K") == 0 || strcmp(end, "k") == 0) {
        bytes = (size_t)(val * 1024LL);
    } else if (strcmp(end, "KB") == 0) {
        bytes = (size_t)(val * 1000LL);
    } else if (strcmp(end, "B") == 0 || *end == '\0') {
        bytes = (size_t)val;
    } else {
        return false;
    }
    return true;
}

// Parse --seq argument: "start unit,end unit,step unit"
// e.g. "--seq 1 GiB,192 GiB,8 GiB"
static bool parse_seq_arg(const char * arg) {
    g_seq_sizes.clear();
    std::string s(arg);
    // Split on commas
    size_t c1 = s.find(',');
    if (c1 == std::string::npos) return false;
    size_t c2 = s.find(',', c1 + 1);
    if (c2 == std::string::npos) return false;

    std::string start_str = s.substr(0, c1);
    std::string end_str   = s.substr(c1 + 1, c2 - c1 - 1);
    std::string step_str  = s.substr(c2 + 1);

    size_t start = 0, end = 0, step = 0;
    if (!parse_explicit_size(start_str.c_str(), start)) return false;
    if (!parse_explicit_size(end_str.c_str(),   end))   return false;
    if (!parse_explicit_size(step_str.c_str(),  step))  return false;
    if (step == 0 || start > end) return false;

    for (size_t sz = start; sz <= end; sz += step) {
        g_seq_sizes.push_back(sz);
    }
    return !g_seq_sizes.empty();
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

static TestResult test_device_pinned_max_method(int cuda_ordinal, size_t max_test_size, const char * label, enum pin_method_id method) {
    TestResult result = { 0, 0, 0.0, 0.0, 0.0 };

    LOG_INFO("  [1] cudaSetDevice(%d) (%s)...\n", cuda_ordinal, label);
    cudaError_t err = cudaSetDevice(cuda_ordinal);
    if (err != cudaSuccess) {
        LOG_INFO("  [FAIL] cudaSetDevice(%d): %s\n", cuda_ordinal, cudaGetErrorString(err));
        return result;
    }
    int cur = -1;
    cudaGetDevice(&cur);
    LOG_INFO("  [OK]  cudaGetDevice() = %d\n", cur);

    cudaDeviceProp prop;
    err = cudaGetDeviceProperties(&prop, cur);
    if (err == cudaSuccess) {
    #if CUDART_VERSION >= 12000
    int _kernel_exec_timeout = 1;
    cudaDeviceGetAttribute(&_kernel_exec_timeout, cudaDevAttrKernelExecTimeout, cur);
    const bool _is_wddm = _kernel_exec_timeout != 0;
#else
    const bool _is_wddm = prop.kernelExecTimeoutEnabled;
#endif
    LOG_INFO("  [OK]  %s, compute %d.%d, driver mode: %s\n",
                 prop.name, prop.major, prop.minor,
                 _is_wddm ? "WDDM" : "TCC");
    }

    LOG_INFO("  [2] cudaFree(0) to init primary context...\n");
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
    }

    LOG_INFO("  [4] Pinning test (method=%s)\n", method_name(method));

    // Build the list of sizes to test
    std::vector<size_t> sizes;
    if (!g_seq_sizes.empty()) {
        sizes = g_seq_sizes;
        LOG_INFO("       custom sequence: %zu sizes\n", sizes.size());
    } else {
        // Predefined ascending size sequence (unit controlled by --unit)
        static const size_t test_nums[] = {
            1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 1536
        };
        size_t mult = unit_mult(g_test_unit);
        LOG_INFO("       unit=%s, %zu sizes\n", unit_name(g_test_unit), sizeof(test_nums)/sizeof(test_nums[0]));
        for (size_t n : test_nums) {
            sizes.push_back(n * mult);
        }
    }

    int64_t t_start_total = ggml_time_us();
    int iteration = 0;

    for (size_t si = 0; si < sizes.size(); si++) {
        size_t try_size = sizes[si];

        // Only try sizes that fit within the user's RAM or limit
        if (try_size > max_test_size) {
            continue;
        }

        iteration++;
        auto t0 = ggml_time_us();
        LOG_INFO("  [%d] Trying ", iteration);
        print_size("", try_size, false);
        LOG_INFO("... ");

        bool alloc_ok = false;
        void * ptr = nullptr;
        double alloc_ms = 0.0, register_ms = 0.0;

        if (method == PIN_METHOD_CHOST) {
            auto t1 = ggml_time_us();
            err = allocpin_chost(&ptr, try_size);
            alloc_ms = (ggml_time_us() - t1) / 1000.0;

            if (err == cudaSuccess) {
                alloc_ok = true;
                register_ms = 0.0;
                LOG_INFO("cudaHostAlloc OK (%.1f ms)\n", alloc_ms);
            } else {
                LOG_INFO("cudaHostAlloc FAILED: %s (%.1f ms)\n",
                         cudaGetErrorString(err), alloc_ms);
                cudaGetLastError();
            }
        } else {
            auto t1 = ggml_time_us();
            switch (method) {
                case PIN_METHOD_VA:        ptr = alloc_va(try_size);        break;
                case PIN_METHOD_MALLOC_P:
                case PIN_METHOD_MALLOC_NP: ptr = alloc_malloc_m(try_size);  break;
                case PIN_METHOD_AMALLOC_P: ptr = alloc_amalloc(try_size);   break;
                default: break;
            }
            alloc_ms = (ggml_time_us() - t1) / 1000.0;

            if (ptr == nullptr) {
                LOG_INFO("alloc FAILED (%.1f ms)\n", alloc_ms);
                goto done;
            }

            auto t2 = ggml_time_us();
            if (method == PIN_METHOD_MALLOC_NP) {
                err = pin_nonportable(ptr, try_size);
            } else {
                err = pin_portable(ptr, try_size);
            }
            register_ms = (ggml_time_us() - t2) / 1000.0;

            if (err == cudaSuccess) {
                alloc_ok = true;
                LOG_INFO("alloc OK (%.1f ms), register OK (%.1f ms)\n", alloc_ms, register_ms);
            } else {
                LOG_INFO("alloc OK (%.1f ms), register FAILED: %s (%.1f ms)\n",
                         alloc_ms, cudaGetErrorString(err), register_ms);
                cudaGetLastError();

                switch (method) {
                    case PIN_METHOD_VA:        free_va(ptr);        break;
                    case PIN_METHOD_MALLOC_P:
                    case PIN_METHOD_MALLOC_NP: free_malloc_m(ptr);  break;
                    case PIN_METHOD_AMALLOC_P: free_amalloc(ptr);   break;
                    default: break;
                }
                goto done;
            }
        }

        if (alloc_ok) {
            result.bytes = try_size;
            result.iterations = iteration;
            result.total_ms = (ggml_time_us() - t_start_total) / 1000.0;
            result.alloc_ms = alloc_ms;
            result.register_ms = register_ms;

            if (method == PIN_METHOD_CHOST) {
                free_chost(ptr);
            } else {
                unpin_std(ptr);
                switch (method) {
                    case PIN_METHOD_VA:        free_va(ptr);        break;
                    case PIN_METHOD_MALLOC_P:
                    case PIN_METHOD_MALLOC_NP: free_malloc_m(ptr);  break;
                    case PIN_METHOD_AMALLOC_P: free_amalloc(ptr);   break;
                    default: break;
                }
            }
        }
    }

done:
    // Determine the status:
    // If no sizes were eligible (all > max_test_size), max_test_size is the cap
    if (iteration == 0) {
        LOG_INFO("  [SKIP] All test sizes exceed the %.2f GiB limit\n",
                 max_test_size / (1024.0 * 1024.0 * 1024.0));
    }

    if (result.bytes > 0) {
        LOG_INFO("  [OK]  Max pinned ");
        print_size("", result.bytes, false);
        LOG_INFO(" on %s\n", label);
    } else if (iteration > 0) {
        LOG_INFO("  [FAIL] Could not pin any memory on %s after %d attempts\n",
                 label, iteration);
    }

    return result;
}

static TestResult test_device_pinned_max(int cuda_ordinal, size_t max_test_size, const char * label) {
    return test_device_pinned_max_method(cuda_ordinal, max_test_size, label, PIN_METHOD_VA);
}

int main(int argc, char ** argv) {
    size_t max_size = 0;
    int set_pinmem = -1;
    bool test_all = true;
    bool raw_ordinal = false;
    bool list_methods = false;
    bool list_units = false;
    enum pin_method_id test_method = PIN_METHOD_VA;
    std::vector<int> test_devices;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            print_usage(argv[0]);
            return 0;
        } else if (strcmp(argv[i], "-v") == 0 || strcmp(argv[i], "--verbose") == 0) {
            g_verbose = true;
        } else if (strcmp(argv[i], "--bare") == 0) {
            g_bare_mode = true;
        } else if (strcmp(argv[i], "-m") == 0 || strcmp(argv[i], "--max") == 0) {
            if (i + 1 >= argc) { LOG_ERR("-m requires an argument\n"); return 1; }
            if (!parse_size(argv[++i], max_size)) { LOG_ERR("invalid size '%s'\n", argv[i]); return 1; }
        } else if (strcmp(argv[i], "-p") == 0 || strcmp(argv[i], "--pinmem") == 0) {
            if (i + 1 >= argc) { LOG_ERR("-p requires an argument\n"); return 1; }
            set_pinmem = atoi(argv[++i]);
            if (set_pinmem < 0 || set_pinmem > 3) { LOG_ERR("pinmem must be 0-3\n"); return 1; }
        } else if (strcmp(argv[i], "-M") == 0 || strcmp(argv[i], "--method") == 0) {
            if (i + 1 >= argc) { LOG_ERR("-M requires an argument\n"); return 1; }
            i++;
            enum pin_method_id m = parse_method(argv[i]);
            if ((int)m < 0) { LOG_ERR("unknown method '%s'\n", argv[i]); return 1; }
            test_method = m;
        } else if (strcmp(argv[i], "--list-methods") == 0) {
            list_methods = true;
        } else if (strcmp(argv[i], "--unit") == 0 || strcmp(argv[i], "--units") == 0) {
            if (i + 1 >= argc) { LOG_ERR("--unit requires an argument (MiB, MB, GiB, GB)\n"); return 1; }
            i++;
            if (strcmp(argv[i], "MiB") == 0 || strcmp(argv[i], "mib") == 0) {
                g_test_unit = TESTUNIT_MIB;
            } else if (strcmp(argv[i], "MB") == 0 || strcmp(argv[i], "mb") == 0) {
                g_test_unit = TESTUNIT_MB;
            } else if (strcmp(argv[i], "GiB") == 0 || strcmp(argv[i], "gib") == 0) {
                g_test_unit = TESTUNIT_GIB;
            } else if (strcmp(argv[i], "GB") == 0 || strcmp(argv[i], "gb") == 0) {
                g_test_unit = TESTUNIT_GB;
            } else {
                LOG_ERR("unknown unit '%s' (try: MiB, MB, GiB, GB)\n", argv[i]);
                return 1;
            }
        } else if (strcmp(argv[i], "--list-units") == 0) {
            list_units = true;
        } else if (strcmp(argv[i], "--seq") == 0) {
            // parsed separately after the loop (needs g_seq_sizes populated)
            if (i + 1 >= argc) { LOG_ERR("--seq requires an argument\n"); return 1; }
            i++; // skip the value
        } else if (strcmp(argv[i], "-a") == 0 || strcmp(argv[i], "--all") == 0) {
            test_all = true;
            test_devices.clear();
        } else if (strcmp(argv[i], "-d") == 0 || strcmp(argv[i], "--dev") == 0) {
            if (i + 1 >= argc) { LOG_ERR("-d requires a device number\n"); return 1; }
            test_all = false;
            test_devices.push_back(atoi(argv[++i]));
        } else if (strcmp(argv[i], "-r") == 0 || strcmp(argv[i], "--raw-ord") == 0) {
            raw_ordinal = true;
        } else {
            LOG_ERR("unknown option '%s'\n", argv[i]);
            LOG_INFO("try '%s --help' for more info\n", argv[0]);
            return 1;
        }
    }

    if (list_methods) {
        LOG_INFO("Available allocation methods:\n");
        for (int i = 0; i < PIN_METHOD_COUNT; i++) {
            LOG_INFO("  %-10s %s\n", method_name((enum pin_method_id)i), method_desc((enum pin_method_id)i));
        }
        LOG_INFO("  %-10s Run all methods\n", "all");
        return 0;
    }

    if (list_units) {
        LOG_INFO("Available test units for the default ascending sequence (--unit):\n");
        LOG_INFO("  MiB    Mebibytes (1024^2)\n");
        LOG_INFO("  MB     Megabytes (1000^2)\n");
        LOG_INFO("  GiB    Gibibytes (1024^3, default)\n");
        LOG_INFO("  GB     Gigabytes (1000^3)\n");
        LOG_INFO("The same numeric sequence {1,2,3,4,6,...} is multiplied by this unit.\n");
        return 0;
    }

    // Find --seq in args (must parse before max_size auto-fallback)
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--seq") == 0) {
            if (i + 1 >= argc) { LOG_ERR("--seq requires an argument\n"); return 1; }
            if (!parse_seq_arg(argv[i + 1])) {
                LOG_ERR("invalid --seq format, expected \"start unit,end unit,step unit\"\n");
                return 1;
            }
            LOG_INFO("Custom test sequence: %zu sizes\n", g_seq_sizes.size());
            break;
        }
    }

    if (max_size == 0) {
        max_size = get_total_physical_ram();
    }

    LOG_INFO("=== CUDA Pinned Memory Test ===\n\n");

    LOG_INFO("System:\n");
    print_size("  Physical RAM total: ", get_total_physical_ram());
    print_size("  Physical RAM free:  ", get_free_physical_ram());
    LOG_INFO("  Mode:               %s\n",
             g_bare_mode ? "bare (llama init + raw CUDA API)" : "normal (via ggml backend)");
    LOG_INFO("\n");

    // -------------------------------------------------------------------
    // Init phase
    // -------------------------------------------------------------------
    if (g_bare_mode) {
        LOG_INFO("Initialising llama backend for CUDA runtime loading only...\n");
        LOG_INFO("  (bare mode: enumeration and testing use raw CUDA API only)\n");
        llama_backend_init();
        LOG_INFO("  OK\n\n");

        if (set_pinmem >= 0) {
            ggml_backend_cuda_set_pinmem(set_pinmem);
        }
        int pinmem = ggml_backend_cuda_get_pinmem();
        LOG_INFO("  Pinmem mode:   %d", pinmem);
        if (pinmem == 0)      LOG_INFO(" (disabled)");
        else if (pinmem == 1) LOG_INFO(" (token_embd only)");
        else if (pinmem == 2) LOG_INFO(" (try all, stop on fail)");
        else if (pinmem == 3) LOG_INFO(" (pin all - default)");
        LOG_INFO("\n\n");
    } else {
        LOG_INFO("Initialising llama backend...\n");
        llama_backend_init();

        if (set_pinmem >= 0) {
            ggml_backend_cuda_set_pinmem(set_pinmem);
        }
        int pinmem = ggml_backend_cuda_get_pinmem();
        int device_count = ggml_backend_cuda_get_device_count();

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
    }

    // -------------------------------------------------------------------
    // Device enumeration
    // -------------------------------------------------------------------
    struct DeviceInfo {
        int  id;
        int  cuda_ordinal;
        char name[256];
        char pci_bus_id[16];
        const char * attach; // "CPU" or "PCH"
        int  compute_major;
        int  compute_minor;
        bool is_tcc;
        size_t free_vram;
        size_t total_vram;
    };
    std::vector<DeviceInfo> devices;
    std::vector<TestResult> results;

    int device_count = 0;

    if (g_bare_mode) {
        // --- BARE MODE: use raw CUDA API for enumeration ---
        cudaError_t err = cudaGetDeviceCount(&device_count);
        if (err != cudaSuccess) {
            LOG_ERR("cudaGetDeviceCount: %s\n", cudaGetErrorString(err));
            llama_backend_free();
            return 1;
        }

        LOG_INFO("=== Device enumeration (raw CUDA ordinal order) ===\n");
        for (int i = 0; i < device_count; i++) {
            DeviceInfo info;
            info.id = i;
            info.cuda_ordinal = i;

            cudaDeviceProp prop;
            err = cudaGetDeviceProperties(&prop, i);
            if (err == cudaSuccess) {
                strncpy(info.name, prop.name, sizeof(info.name) - 1);
                info.name[sizeof(info.name) - 1] = '\0';
                info.compute_major = prop.major;
                info.compute_minor = prop.minor;
#if CUDART_VERSION >= 12000
                int _kernel_exec_timeout = 1;
                cudaDeviceGetAttribute(&_kernel_exec_timeout, cudaDevAttrKernelExecTimeout, i);
                info.is_tcc = !_kernel_exec_timeout;
#else
                info.is_tcc = !prop.kernelExecTimeoutEnabled;
#endif
                info.attach = (prop.pciBusID < 0x80) ? "CPU" : "PCH";
            } else {
                snprintf(info.name, sizeof(info.name), "cudaGetDeviceProperties failed");
                info.compute_major = 0;
                info.compute_minor = 0;
                info.is_tcc = false;
                info.attach = "?";
            }
            cudaDeviceGetPCIBusId(info.pci_bus_id, sizeof(info.pci_bus_id), i);
            info.total_vram = 0;
            info.free_vram = 0;

            devices.push_back(info);

            LOG_INFO("  Device %d: %s (PCIE %s, %s), compute capability %d.%d, VRAM: %zu MiB\n",
                     i, info.name, info.pci_bus_id, info.attach,
                     info.compute_major, info.compute_minor,
                     prop.totalGlobalMem / (1024 * 1024));
        }
        LOG_INFO("\n");

    } else {
        // --- NORMAL MODE: via ggml backend ---
        device_count = ggml_backend_cuda_get_device_count();

        if (device_count == 0) {
            LOG_ERR("no CUDA devices found\n");
            llama_backend_free();
            return 1;
        }

        LOG_INFO("=== Device enumeration (ggml logical order = PCIe bus order) ===\n");
        for (int i = 0; i < device_count; i++) {
            DeviceInfo info;
            info.id = i;
            info.cuda_ordinal = ggml_backend_cuda_get_device_ordinal(i);
            ggml_backend_cuda_get_device_description(i, info.name, sizeof(info.name));
            ggml_backend_cuda_get_device_memory(i, &info.free_vram, &info.total_vram);

            cudaDeviceProp prop;
            info.is_tcc = false;
            info.compute_major = 0;
            info.compute_minor = 0;
            info.attach = "?";
            if (cudaGetDeviceProperties(&prop, info.cuda_ordinal) == cudaSuccess) {
#if CUDART_VERSION >= 12000
                int _kernel_exec_timeout = 1;
                cudaDeviceGetAttribute(&_kernel_exec_timeout, cudaDevAttrKernelExecTimeout, info.cuda_ordinal);
                info.is_tcc = !_kernel_exec_timeout;
#else
                info.is_tcc = !prop.kernelExecTimeoutEnabled;
#endif
                info.compute_major = prop.major;
                info.compute_minor = prop.minor;
                info.attach = (prop.pciBusID < 0x80) ? "CPU" : "PCH";
            }
            cudaDeviceGetPCIBusId(info.pci_bus_id, sizeof(info.pci_bus_id), info.cuda_ordinal);

            devices.push_back(info);

            LOG_INFO("  Device %d: %s (PCIE %s, %s), compute capability %d.%d, VRAM: %zu MiB\n",
                     i, info.name, info.pci_bus_id, info.attach,
                     info.compute_major, info.compute_minor,
                     info.total_vram / (1024 * 1024));
            print_size("    Free VRAM:  ", info.free_vram);
        }
        LOG_INFO("\n");
    }

    // -------------------------------------------------------------------
    // Resolve test device list
    // -------------------------------------------------------------------
    std::vector<int> to_test;

    if (test_all) {
        if (device_count == 0) {
            LOG_ERR("no CUDA devices found\n");
            if (!g_bare_mode) llama_backend_free();
            return 1;
        }
        for (int i = 0; i < device_count; i++) {
            to_test.push_back(i);
        }
    } else {
        for (int d : test_devices) {
            if (raw_ordinal) {
                bool found = false;
                for (int i = 0; i < device_count; i++) {
                    if (devices[i].cuda_ordinal == d) {
                        to_test.push_back(i);
                        found = true;
                        break;
                    }
                }
                if (!found) {
                    LOG_WARN("raw CUDA ordinal %d not found, skipping\n", d);
                }
            } else {
                if (d < 0 || d >= device_count) {
                    LOG_WARN("device %d out of range (0-%d), skipping\n", d, device_count - 1);
                    continue;
                }
                to_test.push_back(d);
            }
        }
        if (to_test.empty()) {
            LOG_ERR("no valid devices specified\n");
            if (!g_bare_mode) llama_backend_free();
            return 1;
        }
    }

    // -------------------------------------------------------------------
    // Run tests
    // -------------------------------------------------------------------
    LOG_INFO("=== Pinned memory testing ===\n");
    print_size("  Max test size per device: ", max_size);
    LOG_INFO("  Method:              %s\n", test_method == PIN_METHOD_ALL ? "all" : method_name(test_method));
    LOG_INFO("  Devices under test: ");
    for (size_t i = 0; i < to_test.size(); i++) {
        if (i > 0) LOG_INFO(", ");
        int idx = to_test[i];
        LOG_INFO("raw CUDA %d", devices[idx].cuda_ordinal);
    }
    LOG_INFO("\n\n");

    int64_t t_start = ggml_time_us();

    if (test_method == PIN_METHOD_ALL) {
        // Run all methods per device, show comparison per device
        for (size_t idx = 0; idx < to_test.size(); idx++) {
            int list_idx = to_test[idx];
            int raw_ord = devices[list_idx].cuda_ordinal;

            char label[64];
            snprintf(label, sizeof(label), "raw %d (%s)",
                     raw_ord, devices[list_idx].is_tcc ? "TCC" : "WDDM");

            LOG_INFO("+%.*s+\n", 70, "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++");
            LOG_INFO("| Device %d/%d: raw CUDA %d - %-46s |\n",
                     (int)(idx + 1), (int)to_test.size(), raw_ord, devices[list_idx].name);
            LOG_INFO("+%.*s+\n", 70, "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++");

            // Test each method
            std::vector<TestResult> method_results;
            for (int m = 0; m < PIN_METHOD_COUNT; m++) {
                LOG_INFO("\n--- Method: %s (%s) ---\n", method_name((enum pin_method_id)m), method_desc((enum pin_method_id)m));
                TestResult r = test_device_pinned_max_method(raw_ord, max_size, label, (enum pin_method_id)m);
                method_results.push_back(r);
                LOG_INFO("\n");
            }

            // Per-device comparison table
            LOG_INFO("--- Comparison for raw CUDA %d ---\n", raw_ord);
            LOG_INFO("%-12s %-22s %-10s %-12s\n", "Method", "Max Pinned", "Iter", "Time");
            LOG_INFO("%-12s %-22s %-10s %-12s\n", "------", "----------", "----", "----");

            for (int m = 0; m < PIN_METHOD_COUNT; m++) {
                const auto & r = method_results[m];
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
                LOG_INFO("%-12s %-22s %-10d %-12s\n",
                         method_name((enum pin_method_id)m), pinmem_str, r.iterations, time_str);
            }
            LOG_INFO("\n");
        }
    } else {
        // Single method test
        for (size_t idx = 0; idx < to_test.size(); idx++) {
            int list_idx = to_test[idx];
            int raw_ord = devices[list_idx].cuda_ordinal;

            char label[64];
            snprintf(label, sizeof(label), "raw %d (%s)",
                     raw_ord, devices[list_idx].is_tcc ? "TCC" : "WDDM");

            LOG_INFO("+%.*s+\n", 60, "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++");
            LOG_INFO("| Device %d/%d: raw CUDA %d - %-36s |\n",
                     (int)(idx + 1), (int)to_test.size(), raw_ord, devices[list_idx].name);
            LOG_INFO("+%.*s+\n", 60, "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++");

            TestResult r = test_device_pinned_max_method(raw_ord, max_size, label, test_method);

            if (r.bytes > 0) {
                LOG_INFO("\n  >>> raw CUDA %d: max pinned = ", raw_ord);
                print_size("", r.bytes, false);
                LOG_INFO(" (%d iterations, %.1f ms)\n", r.iterations, r.total_ms);
            } else {
                LOG_INFO("\n  >>> raw CUDA %d: COULD NOT PIN\n", raw_ord);
            }
            LOG_INFO("\n");

            results.push_back(r);
        }
    }
    int64_t t_end = ggml_time_us();

    // -------------------------------------------------------------------
    // Summary
    // -------------------------------------------------------------------
    LOG_INFO("=== Summary ===\n");
    LOG_INFO("Total test time: %.1f s\n\n", (t_end - t_start) / 1e6);

    if (test_method == PIN_METHOD_ALL) {
        LOG_INFO("See comparison tables per device above.\n");
        LOG_INFO("Tip: run with a specific method (-M <method>) for detailed per-iteration logs.\n\n");
    } else {
        LOG_INFO("%-5s %-5s %-40s  %-22s  %s\n",
                 "Raw", "Mode", "Name", "Max Pinned Host", "Time");
        LOG_INFO("%-5s %-5s %-40s  %-22s  %s\n",
                 "---", "----", "----", "----------------", "----");

        for (size_t idx = 0; idx < to_test.size(); idx++) {
            int list_idx = to_test[idx];
            const auto & dev = devices[list_idx];
            const auto & r = results[idx];

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

            LOG_INFO("%-5d %-5s %-40s  %-22s  %s\n",
                     dev.cuda_ordinal,
                     dev.is_tcc ? "TCC" : "WDDM",
                     dev.name, pinmem_str, time_str);
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
    }

    if (!g_bare_mode) {
        llama_backend_free();
    }
    // For single-method mode, return 1 if no device succeeded
    if (test_method != PIN_METHOD_ALL) {
        bool any_success = false;
        for (const auto & r : results) {
            if (r.bytes > 0) { any_success = true; break; }
        }
        return any_success ? 0 : 1;
    }
    return 0;
}
