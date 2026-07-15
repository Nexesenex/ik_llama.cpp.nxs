//
// CUDA watchdog: monitors GPU progress and detects hangs. Split out of ggml-cuda.cu;
// the cuda_watchdog struct stays nested in ggml_backend_cuda_context (common.cuh), and
// ggml-cuda.cu drives this TU through the ggml_cuda_watchdog_* helpers below.
//

#include "cuda-watchdog.cuh"

#include <chrono>
#include <thread>

static void ggml_cuda_watchdog_thread_proc(ggml_backend_cuda_context * ctx) {
    auto & wd = ctx->watchdog;
    ggml_cuda_set_device(ctx->device);

    while (true) {
        std::unique_lock<std::mutex> lock(wd.mtx);
        wd.cv.wait(lock, [&wd]() { return wd.armed || wd.stop; });
        if (wd.stop) break;
        wd.armed = false;
        lock.unlock();

        auto start = ggml_time_us();
        const int64_t timeout_us = 16000000;
        while (true) {
            cudaError_t err = cudaEventQuery(wd.event);
            if (err == cudaSuccess) break;
            if (err != cudaErrorNotReady) break;
            auto elapsed = ggml_time_us() - start;
            if (elapsed > timeout_us) {
                wd.hung = true;
                GGML_CUDA_LOG_ERROR("CUDA watchdog: CUDA%d (Device %d) appears hung (kernel did not complete within 16s)\n", ctx->device, ggml_backend_cuda_get_device_ordinal(ctx->device));
                break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    }
}

void ggml_cuda_watchdog_init(ggml_backend_cuda_context * ctx) {
    ggml_cuda_set_device(ctx->device);
    CUDA_CHECK(cudaEventCreate(&ctx->watchdog.event));
    ctx->watchdog.thread = std::thread(ggml_cuda_watchdog_thread_proc, ctx);
}

void ggml_cuda_watchdog_cleanup(ggml_backend_cuda_context * ctx) {
    // Stop watchdog thread first
    {
        std::lock_guard<std::mutex> lk(ctx->watchdog.mtx);
        ctx->watchdog.stop = true;
    }
    ctx->watchdog.cv.notify_one();
    if (ctx->watchdog.thread.joinable()) {
        ctx->watchdog.thread.join();
    }
    if (ctx->watchdog.event) {
        CUDA_CHECK(cudaEventDestroy(ctx->watchdog.event));
    }
}

void ggml_cuda_watchdog_arm(ggml_backend_cuda_context * ctx) {
    // Arm the watchdog: record event on the compute stream. The watchdog thread
    // polls this event and detects hangs.
    CUDA_CHECK(cudaEventRecord(ctx->watchdog.event, ctx->stream()));
    {
        std::lock_guard<std::mutex> lk(ctx->watchdog.mtx);
        ctx->watchdog.armed = true;
    }
    ctx->watchdog.cv.notify_one();
}

bool ggml_cuda_watchdog_is_hung(const ggml_backend_cuda_context * ctx) {
    return ctx->watchdog.hung;
}
