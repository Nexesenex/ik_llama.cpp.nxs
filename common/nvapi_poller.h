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

    // Change the polling interval(s) (ms). One value applies to every WDDM GPU;
    // a comma list maps positionally (WDDM slot order). Applies to the next
    // start; safe to call before start() since the worker resolves it once.
    void set_intervals(const std::vector<int>& ms) { intervals = ms; }

    // Set the per-card heat protection thresholds in Celsius (0 = disabled).
    // pause_celsius: when a polled GPU reaches this temperature, that GPU is
    // skipped (no NVAPI queries, no warmup ping) until it cools below
    // resume_celsius; the other GPUs keep being polled. resume_celsius <= 0 with
    // pause > 0 falls back to pause - HEAT_RESUME_HYSTERESIS_C.
    void set_temp_limits(int pause_celsius, int resume_celsius) {
        pause_temp = pause_celsius;
        resume_temp = resume_celsius;
    }

    // Monitor-only mode: the worker only does the per-card temperature check
    // and publishes hot_state (no NVAPI burst, no CUDA ping). Used by --poller-warmup-fma,
    // whose heartbeat warmup consumes the published hot_state as its skip mask.
    void set_monitor_only(bool val) { monitor_only = val; }

    // Copy the latest per-card heat state into out (out[k] = GPU k too hot),
    // aligned with the WDDM slot order used by the ggml heartbeat.
    void get_hot_state(std::vector<uint8_t> & out) const;

    // Copy the latest per-card permanent FMA heat penalty into out (out[k] =
    // penalty in 1/256ths for GPU k), aligned with the WDDM slot order.
    void get_penalties(std::vector<int> & out) const;

    bool is_running() const { return running.load(); }

private:
    void thread_func();

    std::vector<int> devices;
    std::vector<int> intervals;           // per-WDDM-GPU polling period (0 = off)
    int rounds;
    std::atomic<int> pause_temp{0};        // 0 = heat protection disabled
    std::atomic<int> resume_temp{0};       // resume below this after a pause
    std::atomic<bool> monitor_only{false};
    std::atomic<bool> running{false};
    std::atomic<bool> should_run{false};
    std::thread worker;
    mutable std::mutex hot_mtx;
    std::vector<uint8_t> hot_state;        // published per cycle (WDDM slot order)
    std::vector<int> penalties;            // permanent FMA penalty per card (1/256ths)
};

#endif // _WIN32
