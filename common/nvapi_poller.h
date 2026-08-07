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
    // devices: CUDA device ordinals to keep awake (WDDM only; empty = auto-detect
    // non-TCC GPUs via ggml). Pass raw CUDA ordinals obtained from
    // ggml_backend_cuda_get_device_ordinal(), TCC devices are skipped.
    // interval_ms: polling interval (default 5)
    // rounds: number of NVAPI query bursts per poll cycle (default 3, try 3-5)
    explicit NvapiPoller(const std::vector<int>& devices, int interval_ms = 10, int rounds = 1);
    ~NvapiPoller();

    // Call from TG start / TG end (same places as shark_callback / heartbeat)
    void start();
    void stop();

    // Change the polling interval (ms). Applies to the next cycle; safe to call
    // while running since the loop re-reads it each iteration.
    void set_interval(int ms) { interval_ms = ms; }

    bool is_running() const { return running.load(); }

private:
    void thread_func();

    std::vector<int> devices;
    int interval_ms;
    int rounds;
    std::atomic<bool> running{false};
    std::atomic<bool> should_run{false};
    std::thread worker;
};

#endif // _WIN32
