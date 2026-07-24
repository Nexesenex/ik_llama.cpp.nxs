#pragma once

#ifdef _WIN32

#include <vector>
#include <atomic>
#include <thread>

// NVAPI-based GPU poller that forces high P-states via driver-level queries.
// Much more aggressive than CUDA events — mirrors what GPU Shark does.
// Only active during TG (controlled via start/stop).

class NvapiPoller {
public:
    // devices: CUDA device indices to keep awake
    // interval_ms: polling interval (default 5)
    explicit NvapiPoller(const std::vector<int>& devices, int interval_ms = 5);
    ~NvapiPoller();

    // Call from TG start / TG end (same places as shark_callback / heartbeat)
    void start();
    void stop();

    bool is_running() const { return running.load(); }

private:
    void thread_func();

    std::vector<int> devices;
    int interval_ms;
    std::atomic<bool> running{false};
    std::atomic<bool> should_run{false};
    std::thread worker;
};

#endif // _WIN32
