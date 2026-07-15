#pragma once

#ifdef _WIN32

#include <vector>
#include <atomic>
#include <thread>
#include <mutex>
#include <cstdint>

// NVAPI-based GPU poller that forces high P-states via driver-level queries.
// Much more aggressive than CUDA events — mirrors what GPU Shark does.
// Only active during TG (controlled via start/stop).

// Returns true if no physical NVIDIA GPU exceeds the given temperature limit
// in Celsius. Fail-open: returns true when NVAPI is unavailable.
bool nvapi_gpu_temp_ok(int limit_celsius);

class NvapiPoller {
public:
    // devices: CUDA device ordinals to keep awake (WDDM only; empty = auto-detect
    // non-TCC GPUs via ggml). Pass raw CUDA ordinals obtained from
    // ggml_backend_cuda_get_device_ordinal(), TCC devices are skipped.
    // interval_ms: polling interval (default 100)
    // rounds: number of NVAPI query bursts per poll cycle (default 3, try 3-5)
    explicit NvapiPoller(const std::vector<int>& devices, int interval_ms = 100, int rounds = 1);
    ~NvapiPoller();

    // Call from TG start / TG end (same places as shark_callback / heartbeat).
    // start() launches the worker; it reaps a previous thread that is still
    // winding down from a non-blocking stop().
    void start();
    // Non-blocking: requests the worker to exit; the decode thread never waits.
    // The worker clears running on its next cycle; the destructor joins it.
    void stop();

    // Change the polling interval (ms). Applies to the next cycle; safe to call
    // while running since the loop re-reads it each iteration.
    void set_interval(int ms) { interval_ms = ms; }

    // Set the temperature limit in Celsius (0 = disabled). Per-card protection:
    // when a polled GPU reaches this temperature, that GPU is skipped (no NVAPI
    // queries, no warmup ping) until it cools below limit - 5°C; the other GPUs
    // keep being polled.
    void set_temp_limit(int limit) { temp_limit = limit; }

    // Monitor-only mode: the worker only does the per-card temperature check
    // and publishes hot_state (no NVAPI burst, no CUDA ping). Used by --orca,
    // whose heartbeat warmup consumes the published hot_state as its skip mask.
    void set_monitor_only(bool val) { monitor_only = val; }

    // Copy the latest per-card heat state into out (out[k] = GPU k too hot),
    // aligned with the WDDM slot order used by the ggml heartbeat.
    void get_hot_state(std::vector<uint8_t> & out) const;

    bool is_running() const { return running.load(); }

private:
    void thread_func();

    std::vector<int> devices;
    int interval_ms;
    int rounds;
    std::atomic<int> temp_limit{0};        // 0 = heat protection disabled
    std::atomic<bool> monitor_only{false};
    std::atomic<bool> running{false};
    std::atomic<bool> should_run{false};
    std::thread worker;
    mutable std::mutex hot_mtx;
    std::vector<uint8_t> hot_state;        // published per cycle (WDDM slot order)
};

#endif // _WIN32
