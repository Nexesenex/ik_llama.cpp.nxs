#include <cuda_runtime.h>
#include <cuda.h>
#include <iostream>
#include <thread>
#include <chrono>
#include <vector>
#include <string>
#include <csignal>
#include <cstdlib>
#include <cstdio>
#include <algorithm>

volatile bool running = true;

void signal_handler(int) {
    running = false;
}

#ifdef _WIN32
#define popen  _popen
#define pclose _pclose
#endif

bool is_tcc_device(int dev_id) {
    cudaDeviceProp prop;
    cudaError_t err = cudaGetDeviceProperties(&prop, dev_id);
    if (err != cudaSuccess) return false;
    return prop.tccDriver;
}

int get_gpu_temperature(int dev_id) {
    // Use nvidia-smi to get temperature
    std::string cmd = "nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits -i " + std::to_string(dev_id);
    FILE* pipe = popen(cmd.c_str(), "r");
    if (!pipe) return -1; // fail-open: assume OK
    char buffer[256];
    if (fgets(buffer, sizeof(buffer), pipe)) {
        int temp = std::atoi(buffer);
        pclose(pipe);
        return temp;
    }
    pclose(pipe);
    return -1;
}

std::vector<int> filter_devices(const std::vector<int>& devices, int temp_limit) {
    std::vector<int> result;
    for (int dev : devices) {
        // Skip TCC devices (they don't need polling)
        if (is_tcc_device(dev)) {
            std::cout << "[GPU Poller] Device " << dev << " is TCC, skipping\n";
            continue;
        }
        // Check temperature
        int temp = get_gpu_temperature(dev);
        if (temp >= 0 && temp >= temp_limit) {
            std::cout << "[GPU Poller] Device " << dev << " at " << temp << "°C >= " << temp_limit << "°C, skipping\n";
            continue;
        }
        if (temp >= 0) {
            std::cout << "[GPU Poller] Device " << dev << " at " << temp << "°C, polling\n";
        } else {
            std::cout << "[GPU Poller] Device " << dev << " temp unknown, polling (fail-open)\n";
        }
        result.push_back(dev);
    }
    return result;
}

// Light heartbeat: touch each non-TCC (WDDM) device with a CUDA event
// record+sync every cycle - the cheapest API-only keep-alive. No kernel
// launch on purpose: this example is compiled as plain C++ against the CUDA
// runtime (see CMakeLists.txt), so CUDA launch syntax is not available.
void light_heartbeat(const std::vector<int>& devices) {
    for (int dev : devices) {
        cudaError_t err = cudaSetDevice(dev);
        if (err != cudaSuccess) {
            std::cerr << "[GPU Poller] cudaSetDevice(" << dev << ") failed: "
                      << cudaGetErrorString(err) << "\n";
            continue;
        }

        cudaEvent_t event;
        cudaError_t err2 = cudaEventCreate(&event);
        if (err2 != cudaSuccess) {
            std::cerr << "[GPU Poller] cudaEventCreate failed: "
                      << cudaGetErrorString(err2) << "\n";
            continue;
        }

        cudaEventRecord(event);
        cudaEventSynchronize(event);
        cudaEventDestroy(event);
    }
}

void print_usage(const char* prog) {
    std::cerr << "Usage: " << prog << " --devices 0,1 [--interval 15] [--temp-limit 85]\n"
              << "  --devices     Comma-separated GPU indices (e.g. 0,1)\n"
              << "  --interval    Polling interval(s) in ms (default: 15). One value\n"
              << "                applies to every device; a comma list maps\n"
              << "                positionally, e.g. --interval 20,40\n"
              << "  --temp-limit  Temperature limit in Celsius (default: 85)\n";
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " --devices 0,1 [--interval 15] [--temp-limit 85]\n";
        return 1;
    }

    std::vector<int> devices;
    std::vector<int> intervals = {5}; // one value = applies to every device
    int temp_limit = 85;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--devices" && i + 1 < argc) {
            std::string devs = argv[++i];
            size_t pos = 0;
            while ((pos = devs.find(',')) != std::string::npos) {
                devices.push_back(std::stoi(devs.substr(0, pos)));
                devs.erase(0, pos + 1);
            }
            devices.push_back(std::stoi(devs));
        } else if (arg == "--interval" && i + 1 < argc) {
            std::string vals = argv[++i];
            size_t pos = 0;
            while ((pos = vals.find(',')) != std::string::npos) {
                intervals.push_back(std::stoi(vals.substr(0, pos)));
                vals.erase(0, pos + 1);
            }
            intervals.push_back(std::stoi(vals));
        } else if (arg == "--temp-limit" && i + 1 < argc) {
            temp_limit = std::stoi(argv[++i]);
        } else {
            print_usage(argv[0]);
            return 1;
        }
    }

    if (devices.empty()) {
        std::cerr << "Error: no devices specified\n";
        print_usage(argv[0]);
        return 1;
    }

    // Filter devices based on TCC and temperature
    std::vector<int> active_devices = filter_devices(devices, temp_limit);
    if (active_devices.empty()) {
        std::cout << "[GPU Poller] No devices to poll, exiting\n";
        return 0;
    }

    // Resolve per-device intervals: one value applies to every active device; a
    // comma list maps positionally (device order). Devices past the list are
    // dropped (interval 0 = off).
    std::vector<int> per_dev_interval(active_devices.size(), 0);
    if (intervals.size() == 1) {
        std::fill(per_dev_interval.begin(), per_dev_interval.end(), intervals[0]);
    } else {
        const size_t n = std::min(intervals.size(), active_devices.size());
        for (size_t k = 0; k < n; ++k) per_dev_interval[k] = intervals[k];
    }
    for (size_t k = 0; k < active_devices.size();) {
        if (per_dev_interval[k] <= 0) {
            active_devices.erase(active_devices.begin() + k);
            per_dev_interval.erase(per_dev_interval.begin() + k);
        } else {
            ++k;
        }
    }
    if (active_devices.empty()) {
        std::cout << "[GPU Poller] No devices with a positive interval, exiting\n";
        return 0;
    }

    signal(SIGTERM, [](int) { running = false; });
    signal(SIGINT, [](int) { running = false; });

    std::cout << "[GPU Poller] Starting on devices: ";
    for (size_t k = 0; k < active_devices.size(); ++k) {
        std::cout << active_devices[k] << "(" << per_dev_interval[k] << "ms) ";
    }
    std::cout << "\n";

    // Re-check temperature periodically so a card that heats up during a long
    // generation is paused on its own while the others keep being polled
    // (the initial filter only covers launch). A paused card resumes once it
    // cools below the limit minus a small hysteresis.
    const int resume_hysteresis_c = 5;
    int shortest_interval = *std::min_element(per_dev_interval.begin(), per_dev_interval.end());
    const auto temp_check_period = std::chrono::milliseconds(std::max(50 * shortest_interval, 250));
    std::vector<int> hot_devices; // devices paused because they are too hot
    std::vector<std::chrono::steady_clock::time_point> next_due(active_devices.size(), std::chrono::steady_clock::time_point{});
    auto next_temp_check = std::chrono::steady_clock::now();

    while (running) {
        const auto now = std::chrono::steady_clock::now();
        // Heartbeat each device at its own cadence, skipping anything paused for heat.
        for (size_t k = 0; k < active_devices.size(); ++k) {
            const int d = active_devices[k];
            if (std::find(hot_devices.begin(), hot_devices.end(), d) != hot_devices.end()) {
                continue; // too hot
            }
            if (next_due[k] != std::chrono::steady_clock::time_point{} && now < next_due[k]) {
                continue; // not yet due
            }
            next_due[k] = now + std::chrono::milliseconds(per_dev_interval[k]);
            light_heartbeat({d});
        }
        if (now >= next_temp_check) {
            next_temp_check = now + temp_check_period;
            for (int d : active_devices) {
                int temp = get_gpu_temperature(d);
                if (temp < 0) continue; // fail-open: leave state unchanged
                bool is_hot = std::find(hot_devices.begin(), hot_devices.end(), d) != hot_devices.end();
                if (temp >= temp_limit && !is_hot) {
                    std::cout << "[GPU Poller] Device " << d << " at " << temp << "°C >= "
                              << temp_limit << "°C, pausing polling to protect card\n";
                    hot_devices.push_back(d);
                } else if (temp < temp_limit - resume_hysteresis_c && is_hot) {
                    std::cout << "[GPU Poller] Device " << d << " cooled to " << temp
                              << "°C, resuming polling\n";
                    hot_devices.erase(std::remove(hot_devices.begin(), hot_devices.end(), d), hot_devices.end());
                }
            }
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    std::cout << "[GPU Poller] Stopping.\n";
    return 0;
}