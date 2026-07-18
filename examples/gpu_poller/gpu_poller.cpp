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

bool is_tcc_device(int dev_id) {
    cudaDeviceProp prop;
    cudaError_t err = cudaGetDeviceProperties(&prop, dev_id);
    if (err != cudaSuccess) return false;
    return prop.tccDriver;
}

int get_gpu_temperature(int dev_id) {
    // Use nvidia-smi to get temperature
    std::string cmd = "nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits -i " + std::to_string(dev_id);
    FILE* pipe = _popen(cmd.c_str(), "r");
    if (!pipe) return -1; // fail-open: assume OK
    char buffer[256];
    if (fgets(buffer, sizeof(buffer), pipe)) {
        int temp = std::atoi(buffer);
        _pclose(pipe);
        return temp;
    }
    _pclose(pipe);
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
              << "  --interval    Polling interval in milliseconds (default: 15)\n"
              << "  --temp-limit  Temperature limit in Celsius (default: 85)\n";
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " --devices 0,1 [--interval 15] [--temp-limit 85]\n";
        return 1;
    }

    std::vector<int> devices;
    int interval_ms = 10;
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
            interval_ms = std::stoi(argv[++i]);
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

    signal(SIGTERM, [](int) { running = false; });
    signal(SIGINT, [](int) { running = false; });

    std::cout << "[GPU Poller] Starting on devices: ";
    for (int d : active_devices) std::cout << d << " ";
    std::cout << "(interval=" << interval_ms << "ms)\n";

    while (running) {
        light_heartbeat(active_devices);
        std::this_thread::sleep_for(std::chrono::milliseconds(interval_ms));
    }

    std::cout << "[GPU Poller] Stopping.\n";
    return 0;
}