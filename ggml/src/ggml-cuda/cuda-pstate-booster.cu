//
// Clock-boosting / p-state machinery for WDDM GPUs (--poller-warmup-fma, --poller-warmup-mem, --poller-activity-fma,
// --poller-warmup-mma, --poller-activity-mma, --poller-ping-mem, --poller-ping-fma, --poller-ping-mma) and the GGML_CALL setters. Split out of
// ggml-cuda.cu; declarations in cuda-pstate-booster.cuh (internal API) and ggml-cuda.h
// (public GGML_CALL API). Logging macros and ggml_cuda_log come from common.cuh.
//

#include "cuda-pstate-booster.cuh"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstring>
#include <functional>
#include <thread>

static bool ggml_cuda_poller_warmup_fma = false;
static bool ggml_cuda_poller_active = false; // active: true during TG, false during PP
// warmup-mem: decode-gated mem-clock companion (--poller-warmup-mem). Per-GPU (WDDM-positional)
// burst count: the number of 2 MiB passes over the companion buffer per TG batch, fired in
// the same fashion as --poller-warmup-fma fires the FMA warmup. Set via ggml_backend_cuda_set_poller_warmup_mem.
static int  ggml_cuda_poller_warmup_mem[GGML_CUDA_MAX_DEVICES] = {0};
// Marks WDDM slots that set_poller_warmup_mem assigned an explicit value (incl. 0 => disabled).
// !override means "use default".
static bool ggml_cuda_poller_warmup_mem_override[GGML_CUDA_MAX_DEVICES] = {false};
// Any WDDM slot enabled at all (set_poller_warmup_mem assigned at least one positive burst count).
static bool ggml_cuda_poller_warmup_mem_any = false;
// activity-fma: decode-solicited FMA probe (--poller-activity-fma). Per-GPU (WDDM-positional)
// FMA chain length. Unlike --poller-warmup-fma (fires on every TG batch) this fires only on a
// GPU that actually received compute nodes in the current TG batch - the launch
// lives in ggml_backend_cuda_graph_compute, which the scheduler invokes for a
// device exactly when that device has work. Default 8192 (small, ~µs of work).
static int  ggml_cuda_poller_activity_fma[GGML_CUDA_MAX_DEVICES] = {0};
// Marks WDDM slots that set_poller_activity_fma assigned an explicit value (incl. 0 =>
// disabled). !override means "use default".
static bool ggml_cuda_poller_activity_fma_override[GGML_CUDA_MAX_DEVICES] = {false};
// Any WDDM slot enabled at all (set_poller_activity_fma assigned at least one positive value).
static bool ggml_cuda_poller_activity_fma_any = false;

// activity-fma probe default (bare --poller-activity-fma / missing list values). Much smaller
// than the 32768-FMA warmup: it accompanies real decode work (the GPU is already
// busy), so a short pulse is enough to drag the clock governor to boost without
// stealing the SM's budget from the actual kernels.
static constexpr int GGML_CUDA_POLLER_ACTIVITY_FMA_DEFAULT = 8192;

// Occupancy percentage of the poller FMA kernels (--poller-fma-occupancy N[,N,...],
// aliases -p-fma-o / -fishpit). Per-WDDM-GPU (positional) float, 0..100 = direct
// percentage: 0 = disabled (no burst on that GPU), 100 = full grid (16 blocks/SM),
// and the burst grid is scaled to occ% of that residency so fewer SMs are engaged
// during the pulse, leaving more room for the real TG compute. Single value broadcasts
// to every WDDM GPU; more values map positionally; missing values use the default 50.
// The default also applies whenever any FMA poller (warmup / activity / ping) is used
// without the flag (applied by set_poller_fma_occupancy, which fills every slot).
static float ggml_cuda_poller_fma_occupancy[GGML_CUDA_MAX_DEVICES] = {0.0f};

// warmup-mma: tensor-core MMA warmup (--poller-warmup-mma). Like --poller-warmup-fma but issues HMMA
// instructions instead of scalar FFMA: mma.sync.m16n8k16 (fp16) = 2048 FLOPs per
// instruction vs 2 for an FMA, so a much shorter wall-clock burst produces the
// same (or stronger) load on the clock governor - "same compute work, shorter
// pulse". Fires on every TG batch like --poller-warmup-fma (decode-gated warmup), honors the
// shared skip mask and the prompt-length scale. Per-GPU (WDDM-positional) HMMA
// chain length; a 0 set via set_poller_warmup_mma disables it on that GPU.
static int  ggml_cuda_poller_warmup_mma[GGML_CUDA_MAX_DEVICES] = {0};
// Marks WDDM slots that set_poller_warmup_mma assigned an explicit value (incl. 0 =>
// disabled). !override means "use default".
static bool ggml_cuda_poller_warmup_mma_override[GGML_CUDA_MAX_DEVICES] = {false};
// Any WDDM slot enabled at all (set_poller_warmup_mma assigned at least one positive value).
static bool ggml_cuda_poller_warmup_mma_any = false;
// Token interval between warmup bursts (--poller-warmup-interval N[,N,...], aliases
// -p-warm-i / -warmstream). Per-WDDM-GPU (positional) interval: the three warmup
// functions (mma, fma, mem) fire on that GPU every N-th TG batch (token) instead of
// every batch. Single value broadcasts to every WDDM GPU; more values map
// positionally; missing values use the default 1 (fire every batch, historical behavior).
static int  ggml_cuda_poller_warmup_interval[GGML_CUDA_MAX_DEVICES] = {0};
// Marks WDDM slots that set_poller_warmup_interval assigned an explicit value.
// !override means "use default" (fire every batch).
static bool ggml_cuda_poller_warmup_interval_override[GGML_CUDA_MAX_DEVICES] = {false};
// Per-WDDM-slot countdown to the next warmup burst; decremented per TG batch in set_poller_active.
static int  ggml_cuda_poller_warmup_countdown[GGML_CUDA_MAX_DEVICES] = {1};
// Per-WDDM-slot "burst due this batch" flag computed in set_poller_active and honored
// by the three launch functions.
static bool ggml_cuda_poller_warmup_due[GGML_CUDA_MAX_DEVICES] = {false};
// warmup-interval default (bare --poller-warmup-interval / missing list values): fire on
// every TG batch (the historical cadence).
static constexpr int GGML_CUDA_POLLER_WARMUP_INTERVAL_DEFAULT = 1;
// First TG token at which the warmup fires (--poller-warmup-start N[,N,...], aliases
// -p-warm-s / -streamsource). Per-WDDM-GPU (positional): the burst first fires on the
// N-th TG batch (token) of a phase; single value broadcasts to every WDDM GPU; more
// values map positionally; missing values use the default 2 (fire on the second token,
// the historical skip-first-batch behavior); 0 = never fire on that GPU.
static int  ggml_cuda_poller_warmup_start[GGML_CUDA_MAX_DEVICES] = {0};
// Marks WDDM slots that set_poller_warmup_start assigned an explicit value.
// !override means "use default" (fire on the second TG token).
static bool ggml_cuda_poller_warmup_start_override[GGML_CUDA_MAX_DEVICES] = {false};
static constexpr int GGML_CUDA_POLLER_WARMUP_START_DEFAULT = 2;
// warmup-mma default (bare --poller-warmup-mma / missing list values). 8192 HMMA is ~256x the
// FLOPs of the 32768-FMA warmup in a far shorter pulse - a dense boost
// without the sustained power draw of the full chain. 32768 (same duration as
// the warmup's 32768 FMA, ~1000x the FLOPs) is available as an explicit max-intensity
// setting; the default stays conservative to avoid burning the card.
static constexpr int GGML_CUDA_POLLER_WARMUP_MMA_DEFAULT = 8192;

// activity-mma: decode-solicited HMMA probe (--poller-activity-mma). Like --poller-activity-fma but for the
// tensor cores: a short HMMA burst fired exactly when a GPU actually receives
// compute nodes in the current TG batch (rides along with real kernels). Much
// denser than the FMA probe: 8192 HMMA is ~256x the FLOPs of the FMA probe's 8192
// FMA at a fraction of the wall time. Per-GPU (WDDM-positional) HMMA chain
// length; a 0 set via set_poller_activity_mma disables it on that GPU.
static int  ggml_cuda_poller_activity_mma[GGML_CUDA_MAX_DEVICES] = {0};
// Marks WDDM slots that set_poller_activity_mma assigned an explicit value (incl. 0 =>
// disabled). !override means "use default".
static bool ggml_cuda_poller_activity_mma_override[GGML_CUDA_MAX_DEVICES] = {false};
// Any WDDM slot enabled at all (set_poller_activity_mma assigned at least one positive value).
static bool ggml_cuda_poller_activity_mma_any = false;
// activity-mma default (bare --poller-activity-mma / missing list values). 8192 HMMA rides along
// with real decode kernels, so the pulse just tops up the clock governor; smaller
// than the 32768-FMA warmup because the GPU is already busy.
static constexpr int GGML_CUDA_POLLER_ACTIVITY_MMA_DEFAULT = 8192;

// Occupancy percentage of the poller MMA kernels (--poller-mma-occupancy N[,N,...],
// aliases -p-mma-o / -abyss). Per-WDDM-GPU (positional) float, 0..100 = direct
// percentage: 0 = disabled (no burst on that GPU), 100 = full grid (16 blocks/SM),
// and the burst grid is scaled to occ% of that residency so fewer SMs (and thus fewer
// tensor-core warps) are engaged during the pulse, leaving more room for the real TG
// compute. Single value broadcasts to every WDDM GPU; more values map positionally;
// missing values use the default 50. The default also applies whenever any MMA poller
// (warmup / activity / ping) is used without the flag.
static float ggml_cuda_poller_mma_occupancy[GGML_CUDA_MAX_DEVICES] = {0.0f};

// activity-mem: decode-solicited mem burst (--poller-activity-mem). Like --poller-activity-fma but for
// the mem-clock companion: a short stream of 2 MiB passes fired exactly when a GPU
// actually receives compute nodes in the current TG batch (rides along with real
// kernels). Per-GPU (WDDM-positional) burst count; a 0 set via set_poller_activity_mem
// disables it on that GPU.
static int  ggml_cuda_poller_activity_mem[GGML_CUDA_MAX_DEVICES] = {0};
// Marks WDDM slots that set_poller_activity_mem assigned an explicit value (incl. 0 => disabled).
// !override means "use default".
static bool ggml_cuda_poller_activity_mem_override[GGML_CUDA_MAX_DEVICES] = {false};
// Any WDDM slot enabled at all (set_poller_activity_mem assigned at least one positive burst count).
static bool ggml_cuda_poller_activity_mem_any = false;
// activity-mem probe default (bare --poller-activity-mem / missing list values). A single 2 MiB
// pass accompanies real decode work, so the pulse just tops up the mem clock governor.
static constexpr int GGML_CUDA_POLLER_ACTIVITY_MEM_BURSTS_DEFAULT = 1;

// L2 occupancy percentage of the poller mem companion (--poller-mem-occupancy N[,N,...],
// aliases -p-mem-o / -snakepit). Per-WDDM-GPU (positional) float, 0..100 = direct
// percentage: 0 = disabled (no burst on that GPU), 100 = full 2 MiB buffer per pass,
// and the number of buffer slots streamed per pass is scaled to occ% of the full
// buffer, so the burst's L2 footprint (then multiplied by the burst count) is capped
// and leaves more L2 for the real TG compute. Single value broadcasts to every WDDM
// GPU; more values map positionally; missing values use the default 25. The default
// also applies whenever any mem poller (warmup / activity / ping) is used without the
// flag.
static float ggml_cuda_poller_mem_occupancy[GGML_CUDA_MAX_DEVICES] = {0.0f};

// Decode-phase gate shared by the autonomous ping halves (--poller-ping-fma/--poller-ping-mma/--poller-ping-mem):
// true only while llama.cpp is in the TG phase (kept in sync with the warmup
// state, but set unconditionally so it works even when the warmup is disabled).
static std::atomic<bool> ggml_cuda_poller_gate = false;

// Spin kernel: single-block FMA chain burst, launched many times per SM to
// produce real activity and keep clocks elevated during TG (~us per launch).
// Per-device chain length is passed in so each WDDM GPU gets its own pulse.
static __device__ float ggml_cuda_poller_scratch = 0.0f;

// Tensor-core (HMMA) spin kernel: same full-residency launch shape as
// k_poller_warmup_fma but issues warp-convergent matrix-multiply-accumulate
// instructions on the tensor cores instead of scalar FMAs on the FP32 pipe.
// Each mma.sync.m16n8k16 fp16 instruction is 16x8x16 MACs = 2048 FLOPs (vs 2 for
// a scalar FFMA), so the same number of issued instructions delivers ~1000x the
// work per cycle - a much denser power pulse for the same wall-clock duration.
// Operands are fabricated fp16 fragments held in registers (constant values), so
// there is no memory traffic and no source tensor: the loop is pure tensor-core
// issue. A dependent accumulator chain (c = mma(a, b, c)) keeps each HMMA
// latency-bound rather than issue-bound, and the unconditional global write below
// defeats dead-code elimination. Fragment registers: c = 4x f32 (m16n8k16
// c-fragment), a = 4x b32, b = 2x b32 (packed f16x2 each).
static __global__ void k_poller_warmup_mma(const int n_mma) {
#if !(defined(GGML_USE_HIPBLAS) && defined(__HIP_PLATFORM_AMD__)) && !defined(GGML_USE_MUSA) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= CC_AMPERE)
    // Fragment init (m16n8k16): c (4x f32), a (4x b32 = 4x f16x2), b (2x b32).
    float c0 = 0.0f, c1 = 0.0f, c2 = 0.0f, c3 = 0.0f;
    unsigned a0 = 0x3c003c00u, a1 = 0x3c003c00u, a2 = 0x3c003c00u, a3 = 0x3c003c00u; // 1.0h, 1.0h packed
    unsigned b0 = 0x3c003c00u, b1 = 0x3c003c00u;
    // unroll 4: each loop check covers 4 HMMA = 8192 FLOPs, so counter/branch
    // overhead drops to ~1 per 4096 FLOPs.
    #pragma unroll 4
    for (int i = 0; i < n_mma; ++i) {
        asm volatile(
            "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
            : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
            : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
    }
    ggml_cuda_poller_scratch = c0 + c1 + c2 + c3;
#elif !(defined(GGML_USE_HIPBLAS) && defined(__HIP_PLATFORM_AMD__)) && !defined(GGML_USE_MUSA) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= CC_VOLTA)
    // Pre-Ampere: m16n8k8 fp16 (Volta/Turing) = 1024 FLOPs per instruction.
    // c-fragment is 4x f32, a is 4x b32 (4 f16x2), b is 2x b32 (2 f16x2).
    float c0 = 0.0f, c1 = 0.0f, c2 = 0.0f, c3 = 0.0f;
    unsigned a0 = 0x3c003c00u, a1 = 0x3c003c00u, a2 = 0x3c003c00u, a3 = 0x3c003c00u;
    unsigned b0 = 0x3c003c00u, b1 = 0x3c003c00u;
    #pragma unroll 4
    for (int i = 0; i < n_mma; ++i) {
        asm volatile(
            "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 "
            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
            : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
            : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
    }
    ggml_cuda_poller_scratch = c0 + c1 + c2 + c3;
#else
    // Pre-Volta / HIP / MUSA (no NVIDIA tensor cores): fall back to scalar FFMA
    // so the launch still produces a load pulse (just ~1000x lighter, same as the
    // warmup-fma).
    float acc[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 2.5f, 3.0f, 3.5f};
    const int n_rounds = n_mma >> 3;
    const int n_tail   = n_mma & 7;
    #pragma unroll 8
    for (int i = 0; i < n_rounds; ++i) {
        #pragma unroll
        for (int j = 0; j < 8; ++j) acc[j] = fmaf(acc[j], 1.0000001f, 1e-6f);
    }
    for (int j = 0; j < n_tail; ++j) acc[0] = fmaf(acc[0], 1.0000001f, 1e-6f);
    float sum = 0.0f;
    #pragma unroll
    for (int j = 0; j < 8; ++j) sum += acc[j];
    ggml_cuda_poller_scratch = sum;
#endif
}

static __global__ void k_poller_warmup_fma(const int n_fma) {
    if (n_fma <= 0) return;
    // 8 independent accumulator chains (ILP = 8): the FMA body has no serial
    // dependency, so each warp issues back-to-back FMAs and the schedulers
    // don't stall waiting for the previous result even if occupancy dips.
    // Constant-indexed => kept entirely in registers, no local memory.
    float acc[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 2.5f, 3.0f, 3.5f};
    const int n_rounds = n_fma >> 3;   // 8 FMAs per chain per round
    const int n_tail   = n_fma & 7;    // leftover FMAs folded into acc[0]
    // unroll 8: each loop check covers 8 chains x 8 unrolled steps = 64 raw
    // FFMA, so counter/branch overhead drops from ~1 per 4 FMAs to ~1 per 64.
    #pragma unroll 8
    for (int i = 0; i < n_rounds; ++i) {
        #pragma unroll
        for (int j = 0; j < 8; ++j) acc[j] = fmaf(acc[j], 1.0000001f, 1e-6f);
    }
    for (int j = 0; j < n_tail; ++j) acc[0] = fmaf(acc[0], 1.0000001f, 1e-6f);
    float sum = 0.0f;
    #pragma unroll
    for (int j = 0; j < 8; ++j) sum += acc[j];
    // Unconditional global write: guarantees the FMA chains are observable and
    // cannot be removed by dead-code elimination (fast-math builds included).
    ggml_cuda_poller_scratch = sum;
}

// Mem-clock companion kernel: streams 16-byte read-modify-writes through the small
// per-GPU buffer. The buffer is L2-resident, so the streaming reads (__ldcs)
// hit L2 and cost nothing; the write-through stores (__stwt) go to DRAM on
// every iteration, so the mem-clock governor sees sustained DRAM writes with
// no large working set, no cross-GPU copies and no PCIe traffic.
static __global__ void k_poller_warmup_mem(int4 * __restrict__ mem, const int n_mem, const int n_pass) {
    if (mem == nullptr || n_mem <= 0 || n_pass <= 0) return;
    const int stride = gridDim.x;
    for (int p = 0; p < n_pass; ++p) {
        for (int i = blockIdx.x; i < n_mem; i += stride) {
            int4 v = __ldcs(mem + i);
            v.x ^= p; v.y ^= p; v.z ^= p; v.w ^= p; // keep lines dirty, ~0 FLOP cost
            __stwt(mem + i, v);
        }
    }
}

// Launch one warmup burst per non-TCC device on a cached non-blocking stream.
// No dedicated thread: llama's decode hot path calls this through
// ggml_backend_cuda_set_poller_active() at every TG batch, which keeps the clocks
// elevated with real kernels. Launch is fire & forget (async).
static cudaStream_t ggml_cuda_poller_compute_streams[GGML_CUDA_MAX_DEVICES] = {nullptr};
// Serializes every poller kernel launch and the lazy per-device stream / mem-buffer
// creation, which are reached from two threads at once: the autonomous ping thread
// and the decode hot path (ggml_backend_cuda_graph_compute). Without it the
// check-then-create of the compute stream can race into two streams for one device,
// letting a mem burst run concurrently with an FMA/MMA burst on that device. With
// every launch under the same mutex and on the same per-device stream, kernels
// execute strictly in launch order: a mem kernel can never run while the FMA/MMA
// kernels ahead of it are still busy.
static std::mutex ggml_cuda_poller_launch_mtx;
// FMA chain length per non-TCC (WDDM) device, in ggml device order (index = the
// k-th WDDM GPU). A 0 set via set_poller_warmup_fma_strength disables the warmup on that GPU;
// 0 with no explicit value keeps the prior default behaviour.
static int ggml_cuda_poller_warmup_fma_strength[GGML_CUDA_MAX_DEVICES] = {0};
// Marks WDDM slots that set_poller_warmup_fma_strength() assigned an explicit value (incl. 0 =>
// disabled). !override means "use default".
static bool ggml_cuda_poller_warmup_fma_strength_override[GGML_CUDA_MAX_DEVICES] = {false};

static constexpr int GGML_CUDA_POLLER_WARMUP_FMA_DEFAULT = 32768;

// Prompt-length FMA scale (--poller-warmup-fma / --poller-ping-fma-amplitude). The longer the prompt (i.e. the
// more tokens already in context), the less FMA is needed to keep the clock up:
// the GPU is already warm and busy with real attention work over a long KV cache.
// The context is split into 256 brackets; when the prompt crosses bracket k
// (k = floor(256 * prompt_len / n_ctx), clamped to 255), the effective FMA of both
// the warmup and the ping is scaled by (256 - k) / 256. Set per decode by llama.cpp
// via ggml_backend_cuda_set_poller_prompt_len; 0 = no scaling (prompt unknown).
static std::atomic<int> ggml_cuda_poller_prompt_bracket = 0; // 0..255

// Per-GPU FMA chain length for the poller ping (index = k-th WDDM GPU).
// Stronger than a tickle but lighter than the warmup: ~1 ms of full-residency
// load per poller cycle, so the GPU still gets idle gaps (no constant wear).
// A 0 set via set_poller_ping_fma_amplitude disables the FMA ping on that GPU.
static int ggml_cuda_poller_ping_fma_amplitude[GGML_CUDA_MAX_DEVICES] = {0};
// Marks WDDM slots that set_poller_ping_fma_amplitude() assigned an explicit value (incl. 0 =>
// disabled). !override means "use default".
static bool ggml_cuda_poller_ping_fma_amplitude_override[GGML_CUDA_MAX_DEVICES] = {false};

static constexpr int GGML_CUDA_POLLER_PING_FMA_AMPLITUDE_DEFAULT = 8192;

// Per-GPU HMMA chain length for the tensor-core half of the poller ping (--poller-ping-mma-amplitude).
// Shares the single autonomous ping thread: each ping cycle launches both the FMA chain
// (k_poller_warmup_fma) and, when enabled, the MMA chain (k_poller_warmup_mma). ~1000x the FLOPs
// per instruction of the FMA half, so a far denser boost for the same wall-clock pulse.
// A 0 set via set_poller_ping_mma_amplitude disables the MMA ping on that GPU.
static int  ggml_cuda_poller_ping_mma_amplitude[GGML_CUDA_MAX_DEVICES] = {0};
// Marks WDDM slots that set_poller_ping_mma_amplitude() assigned an explicit value (incl. 0 => disabled).
// !override means "use default".
static bool ggml_cuda_poller_ping_mma_amplitude_override[GGML_CUDA_MAX_DEVICES] = {false};

static constexpr int GGML_CUDA_POLLER_PING_MMA_AMPLITUDE_DEFAULT = 8192;

// Per-GPU number of 2 MiB passes per autonomous mem-clock ping (--poller-ping-mem-amplitude). The
// mem half of the ping load, driven by the --poller-ping-mem thread: each ping cycle streams
// bursts 2 MiB passes through the companion buffer. A 0 set via set_poller_ping_mem_amplitude
// disables the mem ping on that GPU.
static int  ggml_cuda_poller_ping_mem_amplitude[GGML_CUDA_MAX_DEVICES] = {0};
// Marks WDDM slots that set_poller_ping_mem_amplitude() assigned an explicit value (incl. 0 => disabled).
// !override means "use default".
static bool ggml_cuda_poller_ping_mem_amplitude_override[GGML_CUDA_MAX_DEVICES] = {false};

static constexpr int GGML_CUDA_POLLER_PING_MEM_AMPLITUDE_DEFAULT = 1;

// Per-card skip mask for the heartbeat warmup (index = k-th WDDM GPU). Set from
// the NVAPI temp monitor (--poller-nvapi / --poller-warmup-fma): a too-hot card is not warmed up.
static bool ggml_cuda_poller_skip[GGML_CUDA_MAX_DEVICES] = {false};

// Per-card permanent FMA heat penalty (index = k-th WDDM GPU), in 1/256ths of
// the full budget. Fed by the NVAPI temp monitor: when a card hits the pause
// temp twice in a row, its scale_256 is permanently lowered by 16 (accumulates
// per double-hot event, never recovers). Applied in ggml_cuda_scale_fma_prompt.
static std::atomic<int> ggml_cuda_poller_penalty[GGML_CUDA_MAX_DEVICES] = {}; // NOLINT(cppcoreguidelines-avoid-c-arrays)

// Per-GPU mem-clock companion buffer (index = same as ggml_cuda_poller_compute_streams).
// The kernel streams 16-byte read-modify-writes through it with write-through
// stores (__stwt), so every store reaches DRAM even though the buffer is far
// smaller than L2: the mem-clock governor sees sustained DRAM writes without a
// large working set. 2 MiB (fully L2-resident on GPUs with >= 2 MiB L2, e.g. the
// 3090's 6 MiB) still spreads accesses across all memory channels/controllers at
// full residency, costs ~nothing in VRAM, and is device-local (no PCIe).
static int4 * ggml_cuda_poller_mem[GGML_CUDA_MAX_DEVICES] = {nullptr};
static const int   GGML_CUDA_POLLER_MEM_SLOTS  = (2 << 20) / (int) sizeof(int4); // 2 MiB
// Default number of 2 MiB passes per mem burst when a per-GPU burst count is not set
// (--poller-warmup-mem / --poller-activity-mem / --poller-ping-mem-amplitude). A single pass is
// 2 MiB of write-through stores: a short pulse that holds the mem clock up without
// idle-wasting bandwidth. Explicit per-GPU burst counts override this.
static const int   GGML_CUDA_POLLER_MEM_BURSTS_DEFAULT = 1;

// 16 blocks per SM (resident rating), 256 threads per block — full 2048-thread
// residency with no over-subscription. The per-GPU FMA chain makes the
// resident threads issue back-to-back for a deep, sustained load pulse at flat
// FLOP rate, which the clock governor needs to reach boost. Repeated every TG.
// The mem-clock companion is a separate kernel: k_poller_warmup_mem.

// Effective occupancy percentage for a WDDM slot (w): the direct percentage the setter
// stored for it (0..100). 0 = disabled (no burst on that GPU); 100 = full grid / full
// buffer. The setter fills every slot (default 50/50/25 first, then the explicit values),
// so a slot that was never explicitly assigned still carries its family default.
static float ggml_cuda_poller_occ_pct(const float * occ, int w) {
    if (w >= 0 && w < GGML_CUDA_MAX_DEVICES) {
        return occ[w];
    }
    return 100.0f;
}

// Grid size for a compute burst (FMA / MMA) at the given occupancy percentage:
// occ% of the full 16-blocks-per-SM residency. 0% = disabled (returns 0, the caller
// skips the burst); any positive percentage yields at least 1 block.
static int ggml_cuda_poller_occ_grid(int n_nsm, float occ_pct) {
    if (occ_pct <= 0.0f) return 0;
    int n_blocks = (int) ((n_nsm * 16 * occ_pct) / 100.0f);
    return n_blocks < 1 ? 1 : n_blocks;
}

// Number of 2 MiB-buffer slots for a mem burst at the given occupancy percentage
// (the burst's L2 footprint per pass). 0% = disabled (returns 0, the caller skips the
// burst); any positive percentage yields at least 1 slot.
static int ggml_cuda_poller_occ_mem_slots(float occ_pct) {
    if (occ_pct <= 0.0f) return 0;
    int n_slots = (int) ((GGML_CUDA_POLLER_MEM_SLOTS * occ_pct) / 100.0f);
    return n_slots < 1 ? 1 : n_slots;
}

// Apply the prompt-length scale to an FMA budget. The 256 brackets are split in
// 8 slices of 32 brackets each. Each slice is its own branch below, expressed as
// "start value - drop rate * j" where j is the bracket position within the slice
// (0..31) and the start value is where the previous slice leaves off, so the
// curve is continuous (scale is in 1/256ths of the full budget):
//   - slice 1 (k 0..31):    256 - 2*j          2x faster than baseline  (256 -> 194)
//   - slice 2 (k 32..63):   192 - 2*j          2x faster than baseline  (192 -> 130)
//   - slice 3 (k 64..95):   128 - j            baseline pace            (128 -> 97)
//   - slice 4 (k 96..127):  96 - j             baseline pace            (96 -> 65)
//   - slice 5 (k 128..159): 64 - j/2           half pace                (64 -> 49)
//   - slice 6 (k 160..191): 48 - j/2           half pace                (48 -> 33)
//   - slice 7 (k 192..223): 32 - j/2           half pace                (32 -> 17)
//   - slice 8 (k 224..255): no FMA at all (returns 0 -> callers skip the launch)
// The last slice is FMA-free so a near-full KV cache (where the GPU is already
// busy with real attention work) is never disturbed by an artificial pulse.
// On top of the curve, a per-card permanent heat penalty (fed by the NVAPI temp
// monitor when a card hits the pause temp twice in a row) is subtracted from
// scale_256. A budget (scale_256) below 32 is too thin to matter: FMA is disabled
// entirely (returns 0 -> callers skip the launch) rather than emitting a short
// pulse that adds stutter without any clock-elevation benefit. As an alternative
// absolute floor, a result shorter than 1024 FMA is also zeroed, so a small base
// cannot produce a sub-1024 pulse even if its budget is fine. prompt_bracket
// of 0 = full FMA. w = WDDM position of the card.
static int ggml_cuda_scale_fma_prompt(int fma, int w) {
    const int k = ggml_cuda_poller_prompt_bracket.load();
    if (k <= 0 || fma <= 0) return fma;
    // j = bracket position within the current 32-bracket slice (0..31).
    const int j = k % 32;
    int scale_256;
    if (k < 32) {
        scale_256 = 256 - 2*j;
    } else if (k < 64) {
        scale_256 = 192 - 2*j;
    } else if (k < 96) {
        scale_256 = 128 - j;
    } else if (k < 128) {
        scale_256 = 96 - j;
    } else if (k < 160) {
        scale_256 = 64 - j/2;
    } else if (k < 192) {
        scale_256 = 48 - j/2;
    } else if (k < 224) {
        scale_256 = 32 - j/2;
    } else {
        return 0;                 // slice 8 (k 224..255): no FMA at all
    }
    // Permanent heat penalty (in 1/256ths) applied on top of the curve.
    if (w >= 0 && w < GGML_CUDA_MAX_DEVICES) {
        scale_256 -= ggml_cuda_poller_penalty[w].load();
    }
    // A budget under 32 (1/256ths of the full budget) is too thin to matter:
    // disable FMA entirely so the launch is skipped instead of emitting a
    // short pulse that adds stutter without any clock-elevation benefit.
    if (scale_256 < 32) return 0;
    const int scaled = (int) (((int64_t) fma * scale_256) >> 8);
    // Alternative absolute floor: never emit a pulse shorter than 1024 FMA.
    // A smaller base (e.g. --poller-warmup-fma 2048) could otherwise pass the budget check
    // yet still produce a sub-1024 result.
    return scaled >= 1024 ? scaled : 0;
}

static void ggml_cuda_poller_warmup_fma_launch() {
    std::lock_guard<std::mutex> lk(ggml_cuda_poller_launch_mtx);
    int w = 0;
    for (int i = 0; i < ggml_cuda_info().device_count; ++i) {
        if (ggml_cuda_info().devices[i].is_tcc) continue;
        // skip: too-hot card (temp monitor), explicitly disabled via --poller-warmup-fma 0, or
        // not due this TG batch (--poller-warmup-interval cadence).
        const bool skip_this = (w < GGML_CUDA_MAX_DEVICES && ggml_cuda_poller_skip[w])
                            || (w < GGML_CUDA_MAX_DEVICES && ggml_cuda_poller_warmup_fma_strength_override[w] && ggml_cuda_poller_warmup_fma_strength[w] == 0)
                            || (w < GGML_CUDA_MAX_DEVICES && !ggml_cuda_poller_warmup_due[w]);
        w++;  // w = WDDM position of this device (TCC devices do not consume a slot)
        if (skip_this) continue;
        int cuda_id = ggml_cuda_info().cuda_device_id[i];
        if (cuda_id < 0) continue;
        if (ggml_cuda_poller_compute_streams[i] == nullptr) {
            cudaSetDevice(cuda_id);
            cudaStreamCreateWithFlags(&ggml_cuda_poller_compute_streams[i], cudaStreamNonBlocking);
        }
        int n_nsm = std::max(ggml_cuda_info().devices[i].nsm, 1);
        const int n_fma = ggml_cuda_scale_fma_prompt(
            ggml_cuda_poller_warmup_fma_strength[w - 1] > 0 ? ggml_cuda_poller_warmup_fma_strength[w - 1] : GGML_CUDA_POLLER_WARMUP_FMA_DEFAULT,
            w - 1);
        if (n_fma <= 0) continue; // prompt in the last brackets or heat penalty: no warmup
        cudaSetDevice(cuda_id);
        // --poller-fma-occupancy: scale the grid so fewer SMs are engaged during the
        // burst, leaving more for the real TG compute. 0% = disabled (skip the burst).
        // mem = nullptr: pure FMA kernel, the mem burst is --poller-warmup-mem's job.
        const int n_blocks = ggml_cuda_poller_occ_grid(n_nsm,
            ggml_cuda_poller_occ_pct(ggml_cuda_poller_fma_occupancy, w - 1));
        if (n_blocks <= 0) continue;
        k_poller_warmup_fma<<<n_blocks, 256, 0, ggml_cuda_poller_compute_streams[i]>>>(n_fma);
    }
}

// Launch one HMMA burst per WDDM GPU that is enabled in the warmup-mma mask. Same
// cadence as ggml_cuda_poller_warmup_fma_launch: fired from set_poller_active on every TG batch.
// The HMMA chain makes the resident warps hammer the tensor cores back-to-back for
// a dense power pulse; the prompt-length scale shortens it as the context fills.
static void ggml_cuda_poller_warmup_mma_launch() {
    std::lock_guard<std::mutex> lk(ggml_cuda_poller_launch_mtx);
    int w = 0;
    for (int i = 0; i < ggml_cuda_info().device_count; ++i) {
        if (ggml_cuda_info().devices[i].is_tcc) continue;
        // skip: too-hot card (temp monitor), explicitly disabled via --poller-warmup-mma 0, or
        // not due this TG batch (--poller-warmup-interval cadence).
        const bool skip_this = (w < GGML_CUDA_MAX_DEVICES && ggml_cuda_poller_skip[w])
                            || (w < GGML_CUDA_MAX_DEVICES && ggml_cuda_poller_warmup_mma_override[w] && ggml_cuda_poller_warmup_mma[w] == 0)
                            || (w < GGML_CUDA_MAX_DEVICES && !ggml_cuda_poller_warmup_due[w]);
        w++;  // w = WDDM position of this device (TCC devices do not consume a slot)
        if (skip_this) continue;
        int cuda_id = ggml_cuda_info().cuda_device_id[i];
        if (cuda_id < 0) continue;
        if (ggml_cuda_poller_compute_streams[i] == nullptr) {
            cudaSetDevice(cuda_id);
            cudaStreamCreateWithFlags(&ggml_cuda_poller_compute_streams[i], cudaStreamNonBlocking);
        }
        int n_nsm = std::max(ggml_cuda_info().devices[i].nsm, 1);
        const int n_mma = ggml_cuda_scale_fma_prompt(
            ggml_cuda_poller_warmup_mma[w - 1] > 0 ? ggml_cuda_poller_warmup_mma[w - 1] : GGML_CUDA_POLLER_WARMUP_MMA_DEFAULT,
            w - 1);
        if (n_mma <= 0) continue; // prompt in the last brackets or heat penalty: no warmup
        cudaSetDevice(cuda_id);
        // --poller-mma-occupancy: scale the grid so fewer SMs (and thus fewer tensor-core
        // warps) are engaged during the burst, leaving more for the real TG compute.
        // 0% = disabled (skip the burst).
        const int n_blocks = ggml_cuda_poller_occ_grid(n_nsm,
            ggml_cuda_poller_occ_pct(ggml_cuda_poller_mma_occupancy, w - 1));
        if (n_blocks <= 0) continue;
        k_poller_warmup_mma<<<n_blocks, 256, 0, ggml_cuda_poller_compute_streams[i]>>>(n_mma);
    }
}

// Launch one mem burst per WDDM GPU that is enabled in the warmup-mem mask.
// Same cadence as ggml_cuda_poller_warmup_fma_launch: fired from set_poller_active on every TG
// batch, so the mem-clock companion runs alongside (or instead of) the FMA
// warmup based on --poller-warmup-mem / --poller-warmup-fma. Dedicated mem kernel
// (k_poller_warmup_mem), independent of the FMA chain: each enabled GPU streams
// its own burst count (number of 2 MiB passes, 0 = off) via --poller-warmup-mem.
static void ggml_cuda_poller_warmup_mem_launch() {
    std::lock_guard<std::mutex> lk(ggml_cuda_poller_launch_mtx);
    int w = 0;
    for (int i = 0; i < ggml_cuda_info().device_count; ++i) {
        if (ggml_cuda_info().devices[i].is_tcc) continue;
        // skip: too-hot card (temp monitor), explicitly disabled via --poller-warmup-mem 0, or
        // not due this TG batch (--poller-warmup-interval cadence).
        const bool skip_this = (w < GGML_CUDA_MAX_DEVICES && ggml_cuda_poller_skip[w])
                            || (w < GGML_CUDA_MAX_DEVICES && ggml_cuda_poller_warmup_mem_override[w] && ggml_cuda_poller_warmup_mem[w] == 0)
                            || (w < GGML_CUDA_MAX_DEVICES && !ggml_cuda_poller_warmup_due[w]);
        w++;  // w = WDDM position of this device (TCC devices do not consume a slot)
        if (skip_this) continue;
        int cuda_id = ggml_cuda_info().cuda_device_id[i];
        if (cuda_id < 0) continue;
        cudaSetDevice(cuda_id);
        if (ggml_cuda_poller_compute_streams[i] == nullptr) {
            cudaStreamCreateWithFlags(&ggml_cuda_poller_compute_streams[i], cudaStreamNonBlocking);
        }
        if (ggml_cuda_poller_mem[i] == nullptr) {
            CUDA_CHECK(cudaMalloc(&ggml_cuda_poller_mem[i], (size_t) GGML_CUDA_POLLER_MEM_SLOTS * sizeof(int4)));
        }
        int n_nsm = std::max(ggml_cuda_info().devices[i].nsm, 1);
        const int n_bursts = ggml_cuda_poller_warmup_mem[w - 1] > 0 ? ggml_cuda_poller_warmup_mem[w - 1] : GGML_CUDA_POLLER_MEM_BURSTS_DEFAULT;
        // --poller-mem-occupancy: cap the number of slots streamed per pass (the burst's
        // L2 footprint, multiplied by n_bursts) so the mem companion leaves more L2 for
        // the real TG compute. 0% = disabled (skip the burst). The grid stays full so
        // every pass still spreads across all channels/controllers.
        const int n_mem = ggml_cuda_poller_occ_mem_slots(
            ggml_cuda_poller_occ_pct(ggml_cuda_poller_mem_occupancy, w - 1));
        if (n_mem <= 0) continue;
        k_poller_warmup_mem<<<n_nsm * 16, 256, 0, ggml_cuda_poller_compute_streams[i]>>>(
            ggml_cuda_poller_mem[i],
            n_mem,
            n_bursts);
    }
}

// Fire one activity-fma FMA burst on a single WDDM GPU (w = WDDM position, the same
// indexing as the warmup-fma/nvapi arrays). Called from ggml_backend_cuda_graph_compute:
// the scheduler invokes that function per device exactly when the device has compute
// nodes in the current batch's split graph, so this is the "GPU actually solicited"
// signal. Unlike --poller-warmup-fma (fires on every TG batch) and the autonomous ping threads,
// the probe accompanies real work on that specific card. FMA-only: the mem-clock side is
// --poller-warmup-mem's / --poller-ping-mem's job.
static void ggml_cuda_poller_activity_fma_ping_w(int w) {
    std::lock_guard<std::mutex> lk(ggml_cuda_poller_launch_mtx);
    const auto & info = ggml_cuda_info();
    int ww = 0;
    for (int i = 0; i < info.device_count; ++i) {
        if (info.devices[i].is_tcc) continue;
        if (ww == w) {
            // skip: too-hot card (temp monitor), or explicitly disabled via --poller-activity-fma 0.
            if (ggml_cuda_poller_skip[w]) return;
            if (ggml_cuda_poller_activity_fma_override[w] && ggml_cuda_poller_activity_fma[w] == 0) return;
            int cuda_id = info.cuda_device_id[i];
            if (cuda_id < 0) return;
            if (ggml_cuda_poller_compute_streams[i] == nullptr) {
                cudaSetDevice(cuda_id);
                cudaStreamCreateWithFlags(&ggml_cuda_poller_compute_streams[i], cudaStreamNonBlocking);
            }
            int n_nsm = std::max(info.devices[i].nsm, 1);
            int n_fma = ggml_cuda_poller_activity_fma[w] > 0 ? ggml_cuda_poller_activity_fma[w] : GGML_CUDA_POLLER_ACTIVITY_FMA_DEFAULT;
            // Permanent heat penalty (in 1/256ths of the full budget) on top of
            // the skip mask: a card that has hit the pause temp twice in a row
            // permanently loses 16/256 of its FMA per double-hot episode. A
            // penalty large enough to zero the budget disables the probe.
            if (w >= 0 && w < GGML_CUDA_MAX_DEVICES) {
                const int scale_256 = 256 - ggml_cuda_poller_penalty[w].load();
                // A budget under 32 (1/256ths) is too thin to matter: disable the
                // probe entirely (a short pulse adds stutter without any benefit).
                if (scale_256 < 32) return;
                n_fma = (int) (((int64_t) n_fma * scale_256) >> 8);
                // Alternative absolute floor: never emit a pulse shorter than
                // 1024 FMA, even for a small --poller-activity-fma base.
                if (n_fma < 1024) return;
            }
            cudaSetDevice(cuda_id);
            // --poller-fma-occupancy: scale the grid so fewer SMs are engaged during the
            // burst, leaving more for the real TG compute. 0% = disabled (no probe).
            const int n_blocks = ggml_cuda_poller_occ_grid(n_nsm,
                ggml_cuda_poller_occ_pct(ggml_cuda_poller_fma_occupancy, w));
            if (n_blocks <= 0) return;
            k_poller_warmup_fma<<<n_blocks, 256, 0, ggml_cuda_poller_compute_streams[i]>>>(n_fma);
            return;
        }
        ww++;
    }
}

// Map a ggml device index (i) to its WDDM position w (count of non-TCC devices
// before it). Returns -1 for TCC devices (never solicited by the warmup).
static int ggml_cuda_wddm_pos(int i) {
    const auto & info = ggml_cuda_info();
    if (i < 0 || i >= info.device_count || info.devices[i].is_tcc) return -1;
    int w = 0;
    for (int j = 0; j < i; ++j) {
        if (!info.devices[j].is_tcc) w++;
    }
    return w;
}

// Fire one activity-fma burst for the device currently being computed. The scheduler
// calls ggml_backend_cuda_graph_compute per device exactly when that device has
// compute nodes in the current graph, so this is the "actually solicited" signal.
// Gated to the TG phase via ggml_cuda_poller_gate (PP graphs don't trigger it).
// Fire one activity-fma burst for the device currently being computed (solicited
// from ggml_backend_cuda_graph_compute on a GPU that actually received work in
// this TG batch). Gated to the TG phase via ggml_cuda_poller_gate.
void ggml_cuda_poller_activity_fma_ping_device(int i) {
    if (!ggml_cuda_poller_activity_fma_any) return;
    if (!ggml_cuda_poller_gate.load()) return;
    const int w = ggml_cuda_wddm_pos(i);
    if (w < 0) return;
    ggml_cuda_poller_activity_fma_ping_w(w);
}

// Fire one activity-mma HMMA burst on a single WDDM GPU (w = WDDM position). Same
// decode-solicited semantics as ggml_cuda_poller_activity_fma_ping_w but launches the
// tensor-core kernel: called from ggml_backend_cuda_graph_compute exactly when
// the device has compute nodes in the current batch's split graph, so the probe
// rides along with real work on that specific card. Honors the shared skip mask
// and the permanent heat penalty.
static void ggml_cuda_poller_activity_mma_ping_w(int w) {
    std::lock_guard<std::mutex> lk(ggml_cuda_poller_launch_mtx);
    const auto & info = ggml_cuda_info();
    int ww = 0;
    for (int i = 0; i < info.device_count; ++i) {
        if (info.devices[i].is_tcc) continue;
        if (ww == w) {
            // skip: too-hot card (temp monitor), or explicitly disabled via --poller-activity-mma 0.
            if (ggml_cuda_poller_skip[w]) return;
            if (ggml_cuda_poller_activity_mma_override[w] && ggml_cuda_poller_activity_mma[w] == 0) return;
            int cuda_id = info.cuda_device_id[i];
            if (cuda_id < 0) return;
            if (ggml_cuda_poller_compute_streams[i] == nullptr) {
                cudaSetDevice(cuda_id);
                cudaStreamCreateWithFlags(&ggml_cuda_poller_compute_streams[i], cudaStreamNonBlocking);
            }
            int n_nsm = std::max(info.devices[i].nsm, 1);
            int n_mma = ggml_cuda_poller_activity_mma[w] > 0 ? ggml_cuda_poller_activity_mma[w] : GGML_CUDA_POLLER_ACTIVITY_MMA_DEFAULT;
            // Permanent heat penalty (in 1/256ths of the full budget) on top of
            // the skip mask: a card that has hit the pause temp twice in a row
            // permanently loses 16/256 of its HMMA per double-hot episode. A
            // penalty large enough to zero the budget disables the probe.
            if (w >= 0 && w < GGML_CUDA_MAX_DEVICES) {
                const int scale_256 = 256 - ggml_cuda_poller_penalty[w].load();
                // A budget under 32 (1/256ths) is too thin to matter: disable the
                // probe entirely (a short pulse adds stutter without any benefit).
                if (scale_256 < 32) return;
                n_mma = (int) (((int64_t) n_mma * scale_256) >> 8);
                // Alternative absolute floor: never emit a pulse shorter than
                // 1024 HMMA, even for a small --poller-activity-mma base.
                if (n_mma < 1024) return;
            }
            cudaSetDevice(cuda_id);
            // --poller-mma-occupancy: scale the grid so fewer SMs (and thus fewer tensor-core
            // warps) are engaged during the burst, leaving more for the real TG compute.
            // 0% = disabled (no probe).
            const int n_blocks = ggml_cuda_poller_occ_grid(n_nsm,
                ggml_cuda_poller_occ_pct(ggml_cuda_poller_mma_occupancy, w));
            if (n_blocks <= 0) return;
            k_poller_warmup_mma<<<n_blocks, 256, 0, ggml_cuda_poller_compute_streams[i]>>>(n_mma);
            return;
        }
        ww++;
    }
}

// Fire one activity-mma burst for the device currently being computed (same
// decode-solicited trigger as the FMA probe). Gated to the TG phase via
// ggml_cuda_poller_gate (PP graphs don't trigger it).
// Fire one activity-mma burst for the device currently being computed (solicited
// from ggml_backend_cuda_graph_compute on a GPU that actually received work in
// this TG batch). Gated to the TG phase via ggml_cuda_poller_gate.
void ggml_cuda_poller_activity_mma_ping_device(int i) {
    if (!ggml_cuda_poller_activity_mma_any) return;
    if (!ggml_cuda_poller_gate.load()) return;
    const int w = ggml_cuda_wddm_pos(i);
    if (w < 0) return;
    ggml_cuda_poller_activity_mma_ping_w(w);
}

// Fire one activity-mem mem burst on a single WDDM GPU (w = WDDM position). Same
// decode-solicited semantics as ggml_cuda_poller_activity_fma_ping_w but streams the mem-clock
// companion burst: called from ggml_backend_cuda_graph_compute exactly when the device has
// compute nodes in the current batch's split graph, so the burst rides along with real
// work on that specific card. Honors the shared skip mask and the permanent heat penalty.
static void ggml_cuda_poller_activity_mem_ping_w(int w) {
    std::lock_guard<std::mutex> lk(ggml_cuda_poller_launch_mtx);
    const auto & info = ggml_cuda_info();
    int ww = 0;
    for (int i = 0; i < info.device_count; ++i) {
        if (info.devices[i].is_tcc) continue;
        if (ww == w) {
            // skip: too-hot card (temp monitor), or explicitly disabled via --poller-activity-mem 0.
            if (ggml_cuda_poller_skip[w]) return;
            if (ggml_cuda_poller_activity_mem_override[w] && ggml_cuda_poller_activity_mem[w] == 0) return;
            int cuda_id = info.cuda_device_id[i];
            if (cuda_id < 0) return;
            cudaSetDevice(cuda_id);
            if (ggml_cuda_poller_compute_streams[i] == nullptr) {
                cudaStreamCreateWithFlags(&ggml_cuda_poller_compute_streams[i], cudaStreamNonBlocking);
            }
            if (ggml_cuda_poller_mem[i] == nullptr) {
                CUDA_CHECK(cudaMalloc(&ggml_cuda_poller_mem[i], (size_t) GGML_CUDA_POLLER_MEM_SLOTS * sizeof(int4)));
            }
            int n_nsm = std::max(info.devices[i].nsm, 1);
            int n_bursts = ggml_cuda_poller_activity_mem[w] > 0 ? ggml_cuda_poller_activity_mem[w] : GGML_CUDA_POLLER_ACTIVITY_MEM_BURSTS_DEFAULT;
            // Permanent heat penalty (in 1/256ths of the full budget) on top of
            // the skip mask: a card that has hit the pause temp twice in a row
            // permanently loses 16/256 of its mem burst count per double-hot
            // episode. A penalty large enough to zero the budget disables the probe.
            if (w >= 0 && w < GGML_CUDA_MAX_DEVICES) {
                const int scale_256 = 256 - ggml_cuda_poller_penalty[w].load();
                // A budget under 32 (1/256ths) is too thin to matter: disable the
                // probe entirely (a short burst adds stutter without any benefit).
                if (scale_256 < 32) return;
                n_bursts = (int) (((int64_t) n_bursts * scale_256) >> 8);
                // Alternative absolute floor: never emit a pulse shorter than one
                // 2 MiB pass, even for a small --poller-activity-mem base.
                if (n_bursts < 1) return;
            }
            cudaSetDevice(cuda_id);
            // --poller-mem-occupancy: cap the number of slots streamed per pass (the burst's
            // L2 footprint, multiplied by n_bursts) so the mem companion leaves more L2 for
            // the real TG compute. 0% = disabled (no probe).
            const int n_mem = ggml_cuda_poller_occ_mem_slots(
                ggml_cuda_poller_occ_pct(ggml_cuda_poller_mem_occupancy, w));
            if (n_mem <= 0) return;
            k_poller_warmup_mem<<<n_nsm * 16, 256, 0, ggml_cuda_poller_compute_streams[i]>>>(
                ggml_cuda_poller_mem[i],
                n_mem,
                n_bursts);
            return;
        }
        ww++;
    }
}

// Fire one activity-mem burst for the device currently being computed (same
// decode-solicited trigger as the FMA probe). Gated to the TG phase via
// ggml_cuda_poller_gate (PP graphs don't trigger it).
void ggml_cuda_poller_activity_mem_ping_device(int i) {
    if (!ggml_cuda_poller_activity_mem_any) return;
    if (!ggml_cuda_poller_gate.load()) return;
    const int w = ggml_cuda_wddm_pos(i);
    if (w < 0) return;
    ggml_cuda_poller_activity_mem_ping_w(w);
}

GGML_CALL void ggml_backend_cuda_set_poller_warmup_fma(bool val) {
    if (val == ggml_cuda_poller_warmup_fma) return;
    ggml_cuda_poller_warmup_fma = val;
    if (val) {
        const auto & info = ggml_cuda_info();
        int w = 0;
        for (int i = 0; i < info.device_count; ++i) {
            if (info.devices[i].is_tcc) continue;
            int cuda_id = info.cuda_device_id[i];
            if (cuda_id < 0) continue;
            char name[128] = {0};
            cudaSetDevice(cuda_id);
            cudaDeviceProp prop;
            CUDA_CHECK(cudaGetDeviceProperties(&prop, cuda_id));
            snprintf(name, sizeof(name), "%s", prop.name);
            char pci_bus_id[16] = {0};
            cudaDeviceGetPCIBusId(pci_bus_id, sizeof(pci_bus_id), cuda_id);
            if (ggml_cuda_poller_warmup_fma_strength_override[w] && ggml_cuda_poller_warmup_fma_strength[w] == 0) {
                GGML_CUDA_LOG_INFO("ggml_backend_cuda_set_poller_warmup_fma: GPU %d (%s, PCI %s), WDDM[%d]: warmup disabled (0 in --poller-warmup-fma list)\n",
                    cuda_id, name, pci_bus_id, w);
                w++;
                continue;
            }
            const int fma = ggml_cuda_poller_warmup_fma_strength[w] > 0 ? ggml_cuda_poller_warmup_fma_strength[w] : GGML_CUDA_POLLER_WARMUP_FMA_DEFAULT;
            const bool ping_off = ggml_cuda_poller_ping_fma_amplitude_override[w] && ggml_cuda_poller_ping_fma_amplitude[w] == 0;
            char ping_s[32] = "off";
            if (!ping_off) {
                const int ping_fma = ggml_cuda_poller_ping_fma_amplitude[w] > 0 ? ggml_cuda_poller_ping_fma_amplitude[w] : GGML_CUDA_POLLER_PING_FMA_AMPLITUDE_DEFAULT;
                snprintf(ping_s, sizeof(ping_s), "%d", ping_fma);
            }
            GGML_CUDA_LOG_INFO("ggml_backend_cuda_set_poller_warmup_fma: enabling heartbeat on GPU %d (%s, PCI %s), WDDM[%d]: %d FMA (ping %s)\n",
                cuda_id, name, pci_bus_id, w, fma, ping_s);
            w++;
        }
        // Prime the warmup now, at enable time (before any decode): creates the
        // per-GPU streams, loads the kernel, and warms the WDDM context, so the
        // first TG-phase burst is not a cold first launch on the token path.
        for (int i = 0; i < GGML_CUDA_MAX_DEVICES; ++i) {
            ggml_cuda_poller_warmup_due[i] = true; // prime bypasses the interval cadence
        }
        ggml_cuda_poller_warmup_fma_launch();
    } else {
        GGML_CUDA_LOG_INFO("ggml_backend_cuda_set_poller_warmup_fma: warmup disabled\n");
    }
}

// Per-device event record+sync tickle: record and synchronize a CUDA event on
// each non-TCC (WDDM) GPU. No kernel, no FMA - the cheapest way to touch a
// WDDM card so it does not fully idle between tokens. Mirrors the old
// gpu_poller light heartbeat, but in-process and decoupled from shark/warmup-fma.
// Gated to the TG phase via poller_gate (like the ping threads) and honors the
// shared skip mask, so a too-hot card is not tickled during PP.
static cudaStream_t ggml_cuda_poller_sync_streams[GGML_CUDA_MAX_DEVICES] = {nullptr};
static cudaEvent_t  ggml_cuda_poller_sync_events[GGML_CUDA_MAX_DEVICES]  = {nullptr};

// Record and synchronize a CUDA event on device index i (a non-TCC WDDM GPU).
// Persistent dedicated non-blocking stream + event per device (created once,
// reused every tick - no per-cycle event allocation). Recording on the legacy
// default (NULL) stream would be ordered after all other in-flight work (implicit
// stream sync), turning each tick into a pipeline-drain barrier; a dedicated
// stream records instantly and never perturbs decode.
static void ggml_cuda_poller_sync_tickle(int i) {
    if (ggml_cuda_poller_sync_streams[i] == nullptr) {
        if (cudaStreamCreateWithFlags(&ggml_cuda_poller_sync_streams[i], cudaStreamNonBlocking) != cudaSuccess) return;
    }
    if (ggml_cuda_poller_sync_events[i] == nullptr) {
        if (cudaEventCreate(&ggml_cuda_poller_sync_events[i]) != cudaSuccess) return;
    }
    cudaEventRecord(ggml_cuda_poller_sync_events[i], ggml_cuda_poller_sync_streams[i]);
    cudaEventSynchronize(ggml_cuda_poller_sync_events[i]);
}

static std::atomic<bool> ggml_cuda_poller_sync_stop = false;
// Per-GPU (WDDM-positional) tickle interval in ms; 0 = off for that GPU. Written
// by set_poller_sync, read by the thread each tick.
static std::atomic<int>  ggml_cuda_poller_sync_ms[GGML_CUDA_MAX_DEVICES] = {}; // NOLINT(cppcoreguidelines-avoid-c-arrays)
static std::thread       ggml_cuda_poller_sync_thread;

// Stop and join the tickle thread. Called by set_poller_sync(all-off) and by the guard below
// at process exit: a joinable std::thread would otherwise call std::terminate
// in its destructor, aborting a normal exit whenever --poller-sync is active.
static void ggml_cuda_poller_sync_stop_thread() {
    if (!ggml_cuda_poller_sync_thread.joinable()) return;
    ggml_cuda_poller_sync_stop = true;
    ggml_cuda_poller_sync_thread.join();
    ggml_cuda_poller_sync_thread = std::thread();
}

struct ggml_cuda_poller_sync_thread_guard {
    ~ggml_cuda_poller_sync_thread_guard() { ggml_cuda_poller_sync_stop_thread(); }
};
static ggml_cuda_poller_sync_thread_guard ggml_cuda_poller_sync_thread_guard_instance;

static void ggml_cuda_poller_sync_thread_proc() {
    // Per-GPU cadence (WDDM-positional), 10 ms tick so interval changes and a
    // stop are noticed promptly - same scheduling pattern as the ping threads.
    std::chrono::steady_clock::time_point next_due[GGML_CUDA_MAX_DEVICES] = {}; // NOLINT(cppcoreguidelines-avoid-c-arrays)
    bool was_active = false;
    while (!ggml_cuda_poller_sync_stop.load()) {
        const bool active = ggml_cuda_poller_gate.load();
        if (active && !was_active) {
            // Fresh TG entry: clear the per-GPU cadence so every enabled GPU
            // fires immediately instead of on a stale schedule from an earlier
            // TG phase.
            for (auto & d : next_due) {
                d = std::chrono::steady_clock::time_point();
            }
        }
        was_active = active;
        if (!active) {
            // Outside the TG phase (PP / idle / not decoding): hold off, same as
            // the ping threads. The tickle exists to keep a WDDM card from fully
            // idling between tokens; during PP the GPU is already busy.
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            continue;
        }
        const auto now = std::chrono::steady_clock::now();
        const auto & info = ggml_cuda_info();
        int w = 0;
        for (int i = 0; i < info.device_count; ++i) {
            if (info.devices[i].is_tcc) continue;
            const int ms = ggml_cuda_poller_sync_ms[w].load();
            if (ms > 0 && ggml_cuda_poller_skip[w]) {
                // too-hot card (fed by --poller-nvapi/--poller-warmup-fma): hold off this tick
            } else if (ms > 0 && (next_due[w] == std::chrono::steady_clock::time_point{} || now >= next_due[w])) {
                int cuda_id = info.cuda_device_id[i];
                if (cuda_id >= 0) {
                    cudaSetDevice(cuda_id);
                    ggml_cuda_poller_sync_tickle(i);
                    next_due[w] = now + std::chrono::milliseconds(ms);
                }
            }
            w++;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
}

// poller-sync=N[,N,...]: per-device event record+sync interval(s) in ms (0 = off).
// Single value broadcasts to every WDDM GPU; a comma list maps positionally;
// all zeros or n <= 0 disables. Decoupled from shark/warmup-fma/nvapi: this only
// runs the lightweight tickle thread.
GGML_CALL void ggml_backend_cuda_set_poller_sync(const int * intervals, int n) {
    for (int i = 0; i < GGML_CUDA_MAX_DEVICES; ++i) {
        ggml_cuda_poller_sync_ms[i].store(0);
    }
    bool any = false;
    if (n > 1) {
        const int m = std::min(n, GGML_CUDA_MAX_DEVICES);
        for (int i = 0; i < m; ++i) {
            const int ms = intervals[i] > 0 ? intervals[i] : 0;
            ggml_cuda_poller_sync_ms[i].store(ms);
            if (ms > 0) any = true;
        }
    } else if (n == 1) {
        const int ms = intervals[0] > 0 ? intervals[0] : 0;
        if (ms > 0) {
            for (int i = 0; i < GGML_CUDA_MAX_DEVICES; ++i) {
                ggml_cuda_poller_sync_ms[i].store(ms);
            }
            any = true;
        }
    }
    if (any && !ggml_cuda_poller_sync_thread.joinable()) {
        ggml_cuda_poller_sync_stop = false;
        ggml_cuda_poller_sync_thread = std::thread(ggml_cuda_poller_sync_thread_proc);
        GGML_CUDA_LOG_INFO("ggml_backend_cuda_set_poller_sync: event tickle on WDDM GPUs (per-GPU ms):");
        for (int i = 0; i < GGML_CUDA_MAX_DEVICES; ++i) {
            const int ms = ggml_cuda_poller_sync_ms[i].load();
            if (ms > 0) GGML_CUDA_LOG_INFO(" %d", ms);
        }
        GGML_CUDA_LOG_INFO("\n");
    } else if (!any && ggml_cuda_poller_sync_thread.joinable()) {
        ggml_cuda_poller_sync_stop_thread();
        GGML_CUDA_LOG_INFO("ggml_backend_cuda_set_poller_sync: disabled\n");
    }
}

// Launch the ping load on the w-th non-TCC (WDDM) GPU only (w = WDDM position;
// TCC devices do not consume a slot). Used by the autonomous ping threads for
// per-GPU cadence. Fire-and-forget, async. The core-clock ping fires its FMA half
// (k_poller_warmup_fma) for --poller-ping-fma, its MMA half (k_poller_warmup_mma) for
// --poller-ping-mma, and the mem-clock ping (--poller-ping-mem) fires the mem burst with
// its per-GPU burst count. Returns false when no such GPU or nothing to launch.
static bool ggml_cuda_poller_ping_launch_w(int w, bool do_fma, bool do_mma, bool do_mem) {
    std::lock_guard<std::mutex> lk(ggml_cuda_poller_launch_mtx);
    const auto & info = ggml_cuda_info();
    int ww = 0;
    for (int i = 0; i < info.device_count; ++i) {
        if (info.devices[i].is_tcc) continue;
        if (ww == w) {
            int cuda_id = info.cuda_device_id[i];
            if (cuda_id < 0) return false;
            // Per-half disable checks: FMA ping (--poller-ping-fma-amplitude 0), MMA ping
            // (--poller-ping-mma-amplitude 0), mem ping (--poller-ping-mem-amplitude 0). An
            // explicitly-disabled half is skipped; the other halves are unaffected.
            const bool fma_off = do_fma && ggml_cuda_poller_ping_fma_amplitude_override[w] && ggml_cuda_poller_ping_fma_amplitude[w] == 0;
            const bool mma_off = do_mma && ggml_cuda_poller_ping_mma_amplitude_override[w] && ggml_cuda_poller_ping_mma_amplitude[w] == 0;
            const bool mem_off = do_mem && ggml_cuda_poller_ping_mem_amplitude_override[w] && ggml_cuda_poller_ping_mem_amplitude[w] == 0;
            if (ggml_cuda_poller_compute_streams[i] == nullptr) {
                cudaSetDevice(cuda_id);
                cudaStreamCreateWithFlags(&ggml_cuda_poller_compute_streams[i], cudaStreamNonBlocking);
            }
            int n_nsm = std::max(info.devices[i].nsm, 1);
            const int n_fma = (do_fma && !fma_off)
                ? ggml_cuda_scale_fma_prompt(ggml_cuda_poller_ping_fma_amplitude[w] > 0 ? ggml_cuda_poller_ping_fma_amplitude[w] : GGML_CUDA_POLLER_PING_FMA_AMPLITUDE_DEFAULT, w)
                : 0;
            const int n_mma = (do_mma && !mma_off)
                ? ggml_cuda_scale_fma_prompt(ggml_cuda_poller_ping_mma_amplitude[w] > 0 ? ggml_cuda_poller_ping_mma_amplitude[w] : GGML_CUDA_POLLER_PING_MMA_AMPLITUDE_DEFAULT, w)
                : 0;
            const int n_bursts = (do_mem && !mem_off)
                ? (ggml_cuda_poller_ping_mem_amplitude[w] > 0 ? ggml_cuda_poller_ping_mem_amplitude[w] : GGML_CUDA_POLLER_MEM_BURSTS_DEFAULT)
                : 0;
            // Prompt in the last brackets or heat penalty zeroed every enabled half:
            // with no FMA, no MMA and no mem burst there is nothing to launch for this GPU.
            if (n_fma <= 0 && n_mma <= 0 && n_bursts <= 0) return false;
            cudaSetDevice(cuda_id);
            if (n_bursts > 0 && ggml_cuda_poller_mem[i] == nullptr) {
                CUDA_CHECK(cudaMalloc(&ggml_cuda_poller_mem[i], (size_t) GGML_CUDA_POLLER_MEM_SLOTS * sizeof(int4)));
            }
            // Each ping half launches its own kernel on the same stream: the FMA chain
            // (k_poller_warmup_fma) for --poller-ping-fma, the tensor-core HMMA chain
            // (k_poller_warmup_mma) for --poller-ping-mma, and the mem-clock stream
            // (k_poller_warmup_mem) for --poller-ping-mem. Same stream + mutex keeps
            // them strictly ordered per device. The occupancy limiters (--poller-fma-occupancy,
            // --poller-mma-occupancy, --poller-mem-occupancy) scale the grids / L2 footprint
            // so the pings leave more room for the real TG compute.
            if (n_fma > 0) {
                const int n_blocks = ggml_cuda_poller_occ_grid(n_nsm,
                    ggml_cuda_poller_occ_pct(ggml_cuda_poller_fma_occupancy, w));
                if (n_blocks > 0) {
                    k_poller_warmup_fma<<<n_blocks, 256, 0, ggml_cuda_poller_compute_streams[i]>>>(n_fma);
                }
            }
            if (n_mma > 0) {
                const int n_blocks = ggml_cuda_poller_occ_grid(n_nsm,
                    ggml_cuda_poller_occ_pct(ggml_cuda_poller_mma_occupancy, w));
                if (n_blocks > 0) {
                    k_poller_warmup_mma<<<n_blocks, 256, 0, ggml_cuda_poller_compute_streams[i]>>>(n_mma);
                }
            }
            if (n_bursts > 0) {
                const int n_mem = ggml_cuda_poller_occ_mem_slots(
                    ggml_cuda_poller_occ_pct(ggml_cuda_poller_mem_occupancy, w));
                if (n_mem > 0) {
                    k_poller_warmup_mem<<<n_nsm * 16, 256, 0, ggml_cuda_poller_compute_streams[i]>>>(
                        ggml_cuda_poller_mem[i],
                        n_mem,
                        n_bursts);
                }
            }
            return true;
        }
        ww++;
    }
    return false;
}

// Fire a ping burst on every non-TCC GPU, optionally with the FMA chain, the MMA
// chain and/or the mem-clock stream: --poller-ping-fma fires the FMA half, --poller-ping-mma
// the MMA half, --poller-ping-mem the mem half.
// skip[] is indexed by WDDM position (0 = first non-TCC GPU); skip == nullptr
// falls back to the shared skip mask (set_poller_skip), so a poller-nvapi-fed
// too-hot card is skipped. Fire-and-forget, async.
static void ggml_cuda_poller_ping_launch(bool do_fma, bool do_mma, bool do_mem, const bool * skip, int n_skip) {
    if (skip == nullptr) {
        // Autonomous ping threads: honor the shared skip mask fed by the NVAPI
        // temp monitor (--poller-nvapi / --poller-warmup-fma) so a too-hot card is skipped.
        skip = ggml_cuda_poller_skip;
        n_skip = GGML_CUDA_MAX_DEVICES;
    }
    const auto & info = ggml_cuda_info();
    int w = 0;
    for (int i = 0; i < info.device_count; ++i) {
        if (info.devices[i].is_tcc) continue;
        const bool skip_this = (skip != nullptr) && (w < n_skip) && skip[w];
        w++;  // w = WDDM position of this device (TCC devices do not consume a slot)
        if (skip_this) continue;
        ggml_cuda_poller_ping_launch_w(w - 1, do_fma, do_mma, do_mem);
    }
}

// Shared control for the single autonomous ping thread: owns the stop flag, three
// independent per-GPU interval sets (one per pulse half: FMA, MMA, mem), and the thread.
// The FMA (--poller-ping-fma) and MMA (--poller-ping-mma) halves are the core-clock
// compute ping; the mem half (--poller-ping-mem) is the mem-clock stream. All three fire
// from the same thread at their own per-GPU cadence. Joining happens in stop_thread()
// and in the destructor (process exit), so the thread is never left joinable and never leaks.
struct ggml_cuda_poller_ping_thread_control {
    std::atomic<bool> stop = false;
    // Per-GPU (WDDM-positional) interval in ms per half; 0 = off for that GPU. Written
    // by the setters, read by the thread each tick.
    std::atomic<int>  fma_ms[GGML_CUDA_MAX_DEVICES] = {}; // NOLINT(cppcoreguidelines-avoid-c-arrays)
    std::atomic<int>  mma_ms[GGML_CUDA_MAX_DEVICES] = {}; // NOLINT(cppcoreguidelines-avoid-c-arrays)
    std::atomic<int>  mem_ms[GGML_CUDA_MAX_DEVICES] = {}; // NOLINT(cppcoreguidelines-avoid-c-arrays)
    std::thread       t;

    void stop_thread() {
        if (!t.joinable()) return;
        stop = true;
        t.join();
        t = std::thread();
    }
    ~ggml_cuda_poller_ping_thread_control() { stop_thread(); }
};
static ggml_cuda_poller_ping_thread_control ggml_cuda_poller_ping_ctrl;

static void ggml_cuda_poller_ping_thread_proc(std::atomic<bool> & stop, ggml_cuda_poller_ping_thread_control & c) {
    std::chrono::steady_clock::time_point next_due_fma[GGML_CUDA_MAX_DEVICES] = {}; // NOLINT(cppcoreguidelines-avoid-c-arrays)
    std::chrono::steady_clock::time_point next_due_mma[GGML_CUDA_MAX_DEVICES] = {}; // NOLINT(cppcoreguidelines-avoid-c-arrays)
    std::chrono::steady_clock::time_point next_due_mem[GGML_CUDA_MAX_DEVICES] = {}; // NOLINT(cppcoreguidelines-avoid-c-arrays)
    bool was_active = false;
    while (!stop.load()) {
        const bool active = ggml_cuda_poller_gate.load();
        if (active && !was_active) {
            // Fresh TG entry: clear the per-GPU cadence so every enabled GPU
            // fires immediately instead of on a stale schedule from an earlier
            // TG phase.
            for (auto & d : next_due_fma) {
                d = std::chrono::steady_clock::time_point();
            }
            for (auto & d : next_due_mma) {
                d = std::chrono::steady_clock::time_point();
            }
            for (auto & d : next_due_mem) {
                d = std::chrono::steady_clock::time_point();
            }
        }
        was_active = active;
        if (!active) {
            // Outside the TG phase (PP / idle / not decoding): hold off. Poll the
            // gate at a short cadence so TG start is picked up quickly without
            // loading the GPU.
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            continue;
        }
        const auto now = std::chrono::steady_clock::now();
        for (int w = 0; w < GGML_CUDA_MAX_DEVICES; ++w) {
            if (ggml_cuda_poller_skip[w]) continue; // too-hot card (fed by --poller-nvapi/--poller-warmup-fma)
            // Each half keeps its own cadence independently (a not-yet-due FMA must
            // not gate the MMA/mem halves of the same GPU): each fires only when its
            // own interval elapsed since its own last ping.
            // FMA half (--poller-ping-fma): core-clock FMA chain.
            const int fma_interval = c.fma_ms[w].load();
            if (fma_interval > 0 &&
                (next_due_fma[w] == std::chrono::steady_clock::time_point{} || now >= next_due_fma[w])) {
                ggml_cuda_poller_ping_launch_w(w, true, false, false);
                next_due_fma[w] = now + std::chrono::milliseconds(fma_interval);
            }
            // MMA half (--poller-ping-mma): tensor-core HMMA chain, same thread.
            const int mma_interval = c.mma_ms[w].load();
            if (mma_interval > 0 &&
                (next_due_mma[w] == std::chrono::steady_clock::time_point{} || now >= next_due_mma[w])) {
                ggml_cuda_poller_ping_launch_w(w, false, true, false);
                next_due_mma[w] = now + std::chrono::milliseconds(mma_interval);
            }
            // Mem half (--poller-ping-mem): mem-clock burst stream.
            const int mem_interval = c.mem_ms[w].load();
            if (mem_interval > 0 &&
                (next_due_mem[w] == std::chrono::steady_clock::time_point{} || now >= next_due_mem[w])) {
                ggml_cuda_poller_ping_launch_w(w, false, false, true);
                next_due_mem[w] = now + std::chrono::milliseconds(mem_interval);
            }
        }
        // 10 ms tick: keeps per-GPU cadence within one interval of due while
        // noticing a stop or a TG->PP transition promptly.
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
}

// Apply a per-GPU interval list to one half of the ping thread control: a single
// value broadcasts to every WDDM GPU; a comma list maps positionally (values past the
// WDDM GPU count are ignored, GPUs past the list get 0/off). Returns true when
// any GPU has a positive interval.
static bool ggml_cuda_poller_ping_set_intervals(std::atomic<int> (&ms)[GGML_CUDA_MAX_DEVICES], const int * intervals, int n_intervals) {
    for (int i = 0; i < GGML_CUDA_MAX_DEVICES; ++i) {
        ms[i].store(0);
    }
    bool any = false;
    if (n_intervals > 1) {
        const int n = std::min(n_intervals, GGML_CUDA_MAX_DEVICES);
        for (int i = 0; i < n; ++i) {
            const int ms_i = intervals[i] > 0 ? intervals[i] : 0;
            ms[i].store(ms_i);
            if (ms_i > 0) any = true;
        }
    } else {
        const int ms_i = (n_intervals > 0 && intervals[0] > 0) ? intervals[0] : 0;
        if (ms_i > 0) {
            for (int i = 0; i < GGML_CUDA_MAX_DEVICES; ++i) {
                ms[i].store(ms_i);
            }
            any = true;
        }
    }
    return any;
}

// True when at least one half has a positive per-GPU interval on any GPU, i.e. the
// shared ping thread should be running.
static bool ggml_cuda_poller_ping_any_active(const ggml_cuda_poller_ping_thread_control & c) {
    for (int i = 0; i < GGML_CUDA_MAX_DEVICES; ++i) {
        if (c.fma_ms[i].load() > 0 || c.mma_ms[i].load() > 0 || c.mem_ms[i].load() > 0) {
            return true;
        }
    }
    return false;
}

// ping-fma=N[,N,...]: autonomous FMA ping per WDDM GPU (0 = off for a GPU). The
// core-clock FMA half of the ping load; per-GPU chain length comes from
// set_poller_ping_fma_amplitude() (default 8192). Shares the single ping thread with
// the MMA (--poller-ping-mma) and mem (--poller-ping-mem) halves.
GGML_CALL void ggml_backend_cuda_set_poller_ping_fma(const int * intervals, int n_intervals) {
    auto & c = ggml_cuda_poller_ping_ctrl;
    ggml_cuda_poller_ping_set_intervals(c.fma_ms, intervals, n_intervals);
    if (ggml_cuda_poller_ping_any_active(c) && !c.t.joinable()) {
        c.stop = false;
        c.t = std::thread(ggml_cuda_poller_ping_thread_proc, std::ref(c.stop), std::ref(c));
        GGML_CUDA_LOG_INFO("ggml_backend_cuda_set_poller_ping_fma: FMA ping on WDDM GPUs (per-GPU ms):");
        for (int i = 0; i < GGML_CUDA_MAX_DEVICES; ++i) {
            const int ms = c.fma_ms[i].load();
            if (ms > 0) GGML_CUDA_LOG_INFO(" %d", ms);
        }
        GGML_CUDA_LOG_INFO("\n");
    } else if (!ggml_cuda_poller_ping_any_active(c) && c.t.joinable()) {
        c.stop_thread();
        GGML_CUDA_LOG_INFO("ggml_backend_cuda_set_poller_ping_fma: disabled\n");
    }
}

// ping-mma=N[,N,...]: autonomous tensor-core (HMMA) ping per WDDM GPU (0 = off for a
// GPU). The core-clock MMA half of the ping load; per-GPU chain length comes from
// set_poller_ping_mma_amplitude() (default 8192). Shares the single ping thread with
// the FMA (--poller-ping-fma) and mem (--poller-ping-mem) halves.
GGML_CALL void ggml_backend_cuda_set_poller_ping_mma(const int * intervals, int n_intervals) {
    auto & c = ggml_cuda_poller_ping_ctrl;
    ggml_cuda_poller_ping_set_intervals(c.mma_ms, intervals, n_intervals);
    if (ggml_cuda_poller_ping_any_active(c) && !c.t.joinable()) {
        c.stop = false;
        c.t = std::thread(ggml_cuda_poller_ping_thread_proc, std::ref(c.stop), std::ref(c));
        GGML_CUDA_LOG_INFO("ggml_backend_cuda_set_poller_ping_mma: MMA ping on WDDM GPUs (per-GPU ms):");
        for (int i = 0; i < GGML_CUDA_MAX_DEVICES; ++i) {
            const int ms = c.mma_ms[i].load();
            if (ms > 0) GGML_CUDA_LOG_INFO(" %d", ms);
        }
        GGML_CUDA_LOG_INFO("\n");
    } else if (!ggml_cuda_poller_ping_any_active(c) && c.t.joinable()) {
        c.stop_thread();
        GGML_CUDA_LOG_INFO("ggml_backend_cuda_set_poller_ping_mma: disabled\n");
    }
}

// ping-mem=N[,N,...]: autonomous mem-clock stream per WDDM GPU (0 = off for a
// GPU). Single value broadcasts to every WDDM GPU; more values map positionally.
// The mem half of the ping load, decoupled from poller-nvapi/warmup-fma - no NVAPI, no
// temperature read. Each ping cycle streams the per-GPU burst count from
// set_poller_ping_mem_amplitude() (default 1 x 2 MiB). Shares the single ping thread
// with the FMA (--poller-ping-fma) and MMA (--poller-ping-mma) halves.
GGML_CALL void ggml_backend_cuda_set_poller_ping_mem(const int * intervals, int n_intervals) {
    auto & c = ggml_cuda_poller_ping_ctrl;
    ggml_cuda_poller_ping_set_intervals(c.mem_ms, intervals, n_intervals);
    if (ggml_cuda_poller_ping_any_active(c) && !c.t.joinable()) {
        c.stop = false;
        c.t = std::thread(ggml_cuda_poller_ping_thread_proc, std::ref(c.stop), std::ref(c));
        GGML_CUDA_LOG_INFO("ggml_backend_cuda_set_poller_ping_mem: mem stream on WDDM GPUs (per-GPU ms):");
        for (int i = 0; i < GGML_CUDA_MAX_DEVICES; ++i) {
            const int ms = c.mem_ms[i].load();
            if (ms > 0) GGML_CUDA_LOG_INFO(" %d", ms);
        }
        GGML_CUDA_LOG_INFO("\n");
    } else if (!ggml_cuda_poller_ping_any_active(c) && c.t.joinable()) {
        c.stop_thread();
        GGML_CUDA_LOG_INFO("ggml_backend_cuda_set_poller_ping_mem: disabled\n");
    }
}

GGML_CALL void ggml_backend_cuda_set_poller_warmup_fma_strength(const int * fmas, int n) {
    // Same per-WDDM-GPU mapping as the other clock flags: a single value
    // broadcasts to every WDDM GPU (bare --poller-warmup-fma = all GPUs at the default), a
    // comma list maps positionally in ggml device order (TCC devices don't
    // consume a slot), 0 in the list disables the warmup on that GPU
    // (e.g. --poller-warmup-fma 0,32768), and missing values keep the default. Negative
    // values are replaced by the default.
    for (int i = 0; i < GGML_CUDA_MAX_DEVICES; ++i) {
        ggml_cuda_poller_warmup_fma_strength_override[i] = false;
    }
    if (n > 1) {
        const int m = std::min(n, GGML_CUDA_MAX_DEVICES);
        for (int i = 0; i < m; ++i) {
            const int fma = fmas[i];
            if (fma < 0) {
                GGML_CUDA_LOG_WARN("%s: ignoring negative FMA length %d for WDDM[%d]\n", __func__, fma, i);
                continue;
            }
            ggml_cuda_poller_warmup_fma_strength[i] = fma;
            ggml_cuda_poller_warmup_fma_strength_override[i] = true;
        }
    } else if (n == 1) {
        // Single value (bare --poller-warmup-fma = 32768): broadcast to every WDDM GPU.
        // A lone 0 disables the warmup on every GPU (matches the "one value
        // broadcasts" rule and the 0 = off semantics of the list form).
        const int fma = fmas[0];
        if (fma < 0) {
            GGML_CUDA_LOG_WARN("%s: ignoring negative FMA length %d (bare --poller-warmup-fma = default)\n", __func__, fma);
        } else if (fma >= 0) {
            for (int i = 0; i < GGML_CUDA_MAX_DEVICES; ++i) {
                ggml_cuda_poller_warmup_fma_strength[i] = fma;
                ggml_cuda_poller_warmup_fma_strength_override[i] = true;
            }
        }
    }
}

GGML_CALL void ggml_backend_cuda_set_poller_ping_fma_amplitude(const int * fmas, int n) {
    // Same per-WDDM-GPU mapping as set_poller_warmup_fma_strength (single value broadcasts,
    // comma list positional, 0 = off for a GPU, negative -> default).
    for (int i = 0; i < GGML_CUDA_MAX_DEVICES; ++i) {
        ggml_cuda_poller_ping_fma_amplitude_override[i] = false;
    }
    if (n > 1) {
        const int m = std::min(n, GGML_CUDA_MAX_DEVICES);
        for (int i = 0; i < m; ++i) {
            const int fma = fmas[i];
            if (fma < 0) {
                GGML_CUDA_LOG_WARN("%s: ignoring negative ping FMA length %d for WDDM[%d]\n", __func__, fma, i);
                continue;
            }
            ggml_cuda_poller_ping_fma_amplitude[i] = fma;
            ggml_cuda_poller_ping_fma_amplitude_override[i] = true;
        }
    } else if (n == 1) {
        // Single value (bare --poller-ping-fma-amplitude = 8192): broadcast to every WDDM GPU.
        // A lone 0 disables the ping on every GPU.
        const int fma = fmas[0];
        if (fma < 0) {
            GGML_CUDA_LOG_WARN("%s: ignoring negative ping FMA length %d (bare --poller-ping-fma-amplitude = default)\n", __func__, fma);
        } else if (fma >= 0) {
            for (int i = 0; i < GGML_CUDA_MAX_DEVICES; ++i) {
                ggml_cuda_poller_ping_fma_amplitude[i] = fma;
                ggml_cuda_poller_ping_fma_amplitude_override[i] = true;
            }
        }
    }
}

GGML_CALL void ggml_backend_cuda_set_poller_ping_mma_amplitude(const int * mmas, int n) {
    // Same per-WDDM-GPU mapping as set_poller_ping_fma_amplitude: single value broadcasts,
    // comma list positional, 0 = off for a GPU, negative -> default. Feeds the tensor-core
    // (HMMA) half of the autonomous --poller-ping-mma thread.
    for (int i = 0; i < GGML_CUDA_MAX_DEVICES; ++i) {
        ggml_cuda_poller_ping_mma_amplitude_override[i] = false;
    }
    if (n > 1) {
        const int m = std::min(n, GGML_CUDA_MAX_DEVICES);
        for (int i = 0; i < m; ++i) {
            const int mma = mmas[i];
            if (mma < 0) {
                GGML_CUDA_LOG_WARN("%s: ignoring negative ping MMA length %d for WDDM[%d]\n", __func__, mma, i);
                continue;
            }
            ggml_cuda_poller_ping_mma_amplitude[i] = mma;
            ggml_cuda_poller_ping_mma_amplitude_override[i] = true;
        }
    } else if (n == 1) {
        // Single value (bare --poller-ping-mma-amplitude = 8192): broadcast to every WDDM GPU.
        // A lone 0 disables the MMA ping on every GPU.
        const int mma = mmas[0];
        if (mma < 0) {
            GGML_CUDA_LOG_WARN("%s: ignoring negative ping MMA length %d (bare --poller-ping-mma-amplitude = default)\n", __func__, mma);
        } else if (mma >= 0) {
            for (int i = 0; i < GGML_CUDA_MAX_DEVICES; ++i) {
                ggml_cuda_poller_ping_mma_amplitude[i] = mma;
                ggml_cuda_poller_ping_mma_amplitude_override[i] = true;
            }
        }
    }
}

GGML_CALL void ggml_backend_cuda_set_poller_ping_mem_amplitude(const int * bursts, int n) {
    // Same per-WDDM-GPU mapping as set_poller_ping_fma_amplitude: single value broadcasts,
    // comma list positional, 0 = off for a GPU, negative -> default. Feeds the mem burst
    // count (number of 2 MiB passes) of the autonomous --poller-ping-mem thread.
    for (int i = 0; i < GGML_CUDA_MAX_DEVICES; ++i) {
        ggml_cuda_poller_ping_mem_amplitude_override[i] = false;
    }
    if (n > 1) {
        const int m = std::min(n, GGML_CUDA_MAX_DEVICES);
        for (int i = 0; i < m; ++i) {
            const int bursts_i = bursts[i];
            if (bursts_i < 0) {
                GGML_CUDA_LOG_WARN("%s: ignoring negative ping mem burst count %d for WDDM[%d]\n", __func__, bursts_i, i);
                continue;
            }
            ggml_cuda_poller_ping_mem_amplitude[i] = bursts_i;
            ggml_cuda_poller_ping_mem_amplitude_override[i] = true;
        }
    } else if (n == 1) {
        // Single value (bare --poller-ping-mem-amplitude = 1): broadcast to every WDDM GPU.
        // A lone 0 disables the mem ping on every GPU.
        const int bursts_i = bursts[0];
        if (bursts_i < 0) {
            GGML_CUDA_LOG_WARN("%s: ignoring negative ping mem burst count %d (bare --poller-ping-mem-amplitude = default)\n", __func__, bursts_i);
        } else if (bursts_i >= 0) {
            for (int i = 0; i < GGML_CUDA_MAX_DEVICES; ++i) {
                ggml_cuda_poller_ping_mem_amplitude[i] = bursts_i;
                ggml_cuda_poller_ping_mem_amplitude_override[i] = true;
            }
        }
    }
}

GGML_CALL void ggml_backend_cuda_set_poller_skip(const bool * skip, int n) {
    // Same positional WDDM mapping as set_poller_warmup_fma_strength: skip[0] => first WDDM GPU.
    for (int i = 0; i < GGML_CUDA_MAX_DEVICES; ++i) {
        ggml_cuda_poller_skip[i] = (skip != nullptr) && (i < n) && skip[i];
    }
}

// Set the per-card permanent FMA heat penalty (in 1/256ths of the full budget).
// Same positional WDDM mapping as set_poller_skip: penalty[0] => first WDDM GPU.
// Fed by the NVAPI temp monitor (--poller-nvapi / --poller-warmup-fma): each time a card hits the
// pause temp twice in a row, the monitor adds 16 to that card's penalty here.
// Accumulates, never recovers. Applied to every FMA budget that is subject to
// the prompt scale - --poller-warmup-fma and --poller-ping-fma-amplitude via ggml_cuda_scale_fma_prompt, and
// --poller-activity-fma via ggml_cuda_poller_activity_fma_ping_w - so a penalty that zeroes the
// budget disables that card's FMA entirely.
GGML_CALL void ggml_backend_cuda_set_poller_penalty(const int * penalty, int n) {
    for (int i = 0; i < GGML_CUDA_MAX_DEVICES; ++i) {
        ggml_cuda_poller_penalty[i].store((penalty != nullptr) && (i < n) ? penalty[i] : 0);
    }
}

GGML_CALL bool ggml_backend_cuda_get_poller_warmup_fma(void) {
    return ggml_cuda_poller_warmup_fma;
}

// activity-fma=N[,N,...]: per-WDDM-GPU FMA chain length for the decode-solicited
// probe (--poller-activity-fma). Unlike --poller-warmup-fma (fires on every TG batch), the burst only
// fires on a GPU that actually received compute nodes in the current TG batch -
// the launch lives in ggml_backend_cuda_graph_compute, which the scheduler calls
// per device exactly when that device has work in the split graph. A single value
// broadcasts to every WDDM GPU (bare --poller-activity-fma = all GPUs at the default);
// more values map positionally (0 = off for that GPU); missing values use the
// default. All zeros or n <= 0 disables.
GGML_CALL void ggml_backend_cuda_set_poller_activity_fma(const int * fmas, int n) {
    for (int i = 0; i < GGML_CUDA_MAX_DEVICES; ++i) {
        ggml_cuda_poller_activity_fma[i] = 0;
        ggml_cuda_poller_activity_fma_override[i] = false;
    }
    bool any = false;
    if (n > 1) {
        const int m = std::min(n, GGML_CUDA_MAX_DEVICES);
        for (int i = 0; i < m; ++i) {
            const int fma = fmas[i];
            if (fma < 0) {
                GGML_CUDA_LOG_WARN("%s: ignoring negative FMA length %d for WDDM[%d]\n", __func__, fma, i);
                continue;
            }
            ggml_cuda_poller_activity_fma[i] = fma;
            ggml_cuda_poller_activity_fma_override[i] = true;
            any |= fma > 0;
        }
    } else if (n == 1) {
        // Single value (bare --poller-activity-fma = 8192): broadcast to every WDDM GPU.
        const int fma = fmas[0];
        if (fma < 0) {
            GGML_CUDA_LOG_WARN("%s: ignoring negative FMA length %d (bare --poller-activity-fma = default)\n", __func__, fma);
        } else if (fma > 0) {
            for (int i = 0; i < GGML_CUDA_MAX_DEVICES; ++i) {
                ggml_cuda_poller_activity_fma[i] = fma;
                ggml_cuda_poller_activity_fma_override[i] = true;
            }
            any = true;
        }
    }
    ggml_cuda_poller_activity_fma_any = any;
    if (any) {
        // Prime the stream at enable time so the first decode-phase burst is not
        // a cold allocation on the token path (mirrors set_poller_warmup_fma / set_poller_warmup_mem).
        for (int i = 0; i < ggml_cuda_info().device_count; ++i) {
            const int w = ggml_cuda_wddm_pos(i);
            if (w < 0 || !ggml_cuda_poller_activity_fma_override[w] || ggml_cuda_poller_activity_fma[w] == 0) continue;
            std::lock_guard<std::mutex> lk(ggml_cuda_poller_launch_mtx);
            if (ggml_cuda_poller_compute_streams[i] == nullptr) {
                cudaSetDevice(ggml_cuda_info().cuda_device_id[i]);
                cudaStreamCreateWithFlags(&ggml_cuda_poller_compute_streams[i], cudaStreamNonBlocking);
            }
        }
        GGML_CUDA_LOG_INFO("ggml_backend_cuda_set_poller_activity_fma: enabling decode-solicited FMA probe on WDDM GPUs:");
        for (int i = 0; i < GGML_CUDA_MAX_DEVICES; ++i) {
            if (ggml_cuda_poller_activity_fma_override[i] && ggml_cuda_poller_activity_fma[i] > 0) {
                GGML_CUDA_LOG_INFO(" %d:%d", i, ggml_cuda_poller_activity_fma[i]);
            }
        }
        GGML_CUDA_LOG_INFO("\n");
    } else {
        GGML_CUDA_LOG_INFO("ggml_backend_cuda_set_poller_activity_fma: disabled\n");
    }
}

// activity-mma=N[,N,...]: per-WDDM-GPU HMMA chain length for the decode-solicited
// tensor-core probe (--poller-activity-mma). Like --poller-activity-fma (fires exactly when a GPU
// actually receives compute nodes in the current TG batch) but launches the
// HMMA kernel instead of the scalar-FMA one, so the same wall-clock burst
// delivers ~1000x the compute work. A single value broadcasts to every WDDM GPU
// (bare --poller-activity-mma = all GPUs at the default); more values map positionally
// (0 = off for that GPU); missing values use the default. All zeros or n <= 0
// disables. Honors the shared skip mask and the permanent heat penalty.
GGML_CALL void ggml_backend_cuda_set_poller_activity_mma(const int * mmas, int n) {
    for (int i = 0; i < GGML_CUDA_MAX_DEVICES; ++i) {
        ggml_cuda_poller_activity_mma[i] = 0;
        ggml_cuda_poller_activity_mma_override[i] = false;
    }
    bool any = false;
    if (n > 1) {
        const int m = std::min(n, GGML_CUDA_MAX_DEVICES);
        for (int i = 0; i < m; ++i) {
            const int mma = mmas[i];
            if (mma < 0) {
                GGML_CUDA_LOG_WARN("%s: ignoring negative MMA length %d for WDDM[%d]\n", __func__, mma, i);
                continue;
            }
            ggml_cuda_poller_activity_mma[i] = mma;
            ggml_cuda_poller_activity_mma_override[i] = true;
            any |= mma > 0;
        }
    } else if (n == 1) {
        // Single value (bare --poller-activity-mma = 8192): broadcast to every WDDM GPU.
        const int mma = mmas[0];
        if (mma < 0) {
            GGML_CUDA_LOG_WARN("%s: ignoring negative MMA length %d (bare --poller-activity-mma = default)\n", __func__, mma);
        } else if (mma > 0) {
            for (int i = 0; i < GGML_CUDA_MAX_DEVICES; ++i) {
                ggml_cuda_poller_activity_mma[i] = mma;
                ggml_cuda_poller_activity_mma_override[i] = true;
            }
            any = true;
        }
    }
    ggml_cuda_poller_activity_mma_any = any;
    if (any) {
        // Prime the stream at enable time so the first decode-phase burst is not
        // a cold allocation on the token path (mirrors set_poller_activity_fma).
        for (int i = 0; i < ggml_cuda_info().device_count; ++i) {
            const int w = ggml_cuda_wddm_pos(i);
            if (w < 0 || !ggml_cuda_poller_activity_mma_override[w] || ggml_cuda_poller_activity_mma[w] == 0) continue;
            std::lock_guard<std::mutex> lk(ggml_cuda_poller_launch_mtx);
            if (ggml_cuda_poller_compute_streams[i] == nullptr) {
                cudaSetDevice(ggml_cuda_info().cuda_device_id[i]);
                cudaStreamCreateWithFlags(&ggml_cuda_poller_compute_streams[i], cudaStreamNonBlocking);
            }
        }
        GGML_CUDA_LOG_INFO("ggml_backend_cuda_set_poller_activity_mma: enabling decode-solicited HMMA probe on WDDM GPUs:");
        for (int i = 0; i < GGML_CUDA_MAX_DEVICES; ++i) {
            if (ggml_cuda_poller_activity_mma_override[i] && ggml_cuda_poller_activity_mma[i] > 0) {
                GGML_CUDA_LOG_INFO(" %d:%d", i, ggml_cuda_poller_activity_mma[i]);
            }
        }
        GGML_CUDA_LOG_INFO("\n");
    } else {
        GGML_CUDA_LOG_INFO("ggml_backend_cuda_set_poller_activity_mma: disabled\n");
    }
}

// activity-mem=N[,N,...]: per-WDDM-GPU mem burst count (number of 2 MiB passes) for the
// decode-solicited mem probe (--poller-activity-mem). Like --poller-activity-fma (fires exactly
// when a GPU actually receives compute nodes in the current TG batch) but streams the
// mem-clock companion burst instead of the FMA chain, so it tops up the mem clock
// governor. A single value broadcasts to every WDDM GPU (bare --poller-activity-mem = all
// GPUs at 1 burst); more values map positionally (0 = off for that GPU); missing values
// use the default. All zeros or n <= 0 disables. Honors the shared skip mask and the
// permanent heat penalty.
GGML_CALL void ggml_backend_cuda_set_poller_activity_mem(const int * bursts, int n) {
    for (int i = 0; i < GGML_CUDA_MAX_DEVICES; ++i) {
        ggml_cuda_poller_activity_mem[i] = 0;
        ggml_cuda_poller_activity_mem_override[i] = false;
    }
    bool any = false;
    if (n > 1) {
        const int m = std::min(n, GGML_CUDA_MAX_DEVICES);
        for (int i = 0; i < m; ++i) {
            const int b = bursts[i];
            if (b < 0) {
                GGML_CUDA_LOG_WARN("%s: ignoring negative mem burst count %d for WDDM[%d]\n", __func__, b, i);
                continue;
            }
            ggml_cuda_poller_activity_mem[i] = b;
            ggml_cuda_poller_activity_mem_override[i] = true;
            any |= b > 0;
        }
    } else if (n == 1) {
        // Single value (bare --poller-activity-mem = 1): broadcast to every WDDM GPU.
        const int b = bursts[0];
        if (b < 0) {
            GGML_CUDA_LOG_WARN("%s: ignoring negative mem burst count %d (bare --poller-activity-mem = default)\n", __func__, b);
        } else if (b > 0) {
            for (int i = 0; i < GGML_CUDA_MAX_DEVICES; ++i) {
                ggml_cuda_poller_activity_mem[i] = b;
                ggml_cuda_poller_activity_mem_override[i] = true;
            }
            any = true;
        }
    }
    ggml_cuda_poller_activity_mem_any = any;
    if (any) {
        // Prime the stream + mem buffer at enable time so the first decode-phase burst is not
        // a cold allocation on the token path (mirrors set_poller_activity_fma).
        for (int i = 0; i < ggml_cuda_info().device_count; ++i) {
            const int w = ggml_cuda_wddm_pos(i);
            if (w < 0 || !ggml_cuda_poller_activity_mem_override[w] || ggml_cuda_poller_activity_mem[w] == 0) continue;
            std::lock_guard<std::mutex> lk(ggml_cuda_poller_launch_mtx);
            cudaSetDevice(ggml_cuda_info().cuda_device_id[i]);
            if (ggml_cuda_poller_compute_streams[i] == nullptr) {
                cudaStreamCreateWithFlags(&ggml_cuda_poller_compute_streams[i], cudaStreamNonBlocking);
            }
            if (ggml_cuda_poller_mem[i] == nullptr) {
                CUDA_CHECK(cudaMalloc(&ggml_cuda_poller_mem[i], (size_t) GGML_CUDA_POLLER_MEM_SLOTS * sizeof(int4)));
            }
        }
        GGML_CUDA_LOG_INFO("ggml_backend_cuda_set_poller_activity_mem: enabling decode-solicited mem probe on WDDM GPUs:");
        for (int i = 0; i < GGML_CUDA_MAX_DEVICES; ++i) {
            if (ggml_cuda_poller_activity_mem_override[i] && ggml_cuda_poller_activity_mem[i] > 0) {
                GGML_CUDA_LOG_INFO(" %d:%d", i, ggml_cuda_poller_activity_mem[i]);
            }
        }
        GGML_CUDA_LOG_INFO("\n");
    } else {
        GGML_CUDA_LOG_INFO("ggml_backend_cuda_set_poller_activity_mem: disabled\n");
    }
}

// warmup-mma=N[,N,...]: per-WDDM-GPU HMMA chain length for the tensor-core warmup
// (--poller-warmup-mma). Like --poller-warmup-fma (fires on every TG batch) but issues tensor-core MMA
// instructions instead of scalar FMAs, so the same wall-clock burst delivers
// ~1000x the compute work - a much denser power pulse for the clock governor.
// A single value broadcasts to every WDDM GPU (bare --poller-warmup-mma = all GPUs at the
// default); more values map positionally (0 = off for that GPU); missing values
// use the default. All zeros or n <= 0 disables. Honors the shared skip mask and
// the prompt-length scale. Requires Volta+ tensor cores (m16n8k8) or Ampere+
// (m16n8k16); pre-Volta cards fall back to scalar FFMA inside the kernel.
GGML_CALL void ggml_backend_cuda_set_poller_warmup_mma(const int * mmas, int n) {
    for (int i = 0; i < GGML_CUDA_MAX_DEVICES; ++i) {
        ggml_cuda_poller_warmup_mma[i] = 0;
        ggml_cuda_poller_warmup_mma_override[i] = false;
    }
    bool any = false;
    if (n > 1) {
        const int m = std::min(n, GGML_CUDA_MAX_DEVICES);
        for (int i = 0; i < m; ++i) {
            const int mma = mmas[i];
            if (mma < 0) {
                GGML_CUDA_LOG_WARN("%s: ignoring negative MMA length %d for WDDM[%d]\n", __func__, mma, i);
                continue;
            }
            ggml_cuda_poller_warmup_mma[i] = mma;
            ggml_cuda_poller_warmup_mma_override[i] = true;
            any |= mma > 0;
        }
    } else if (n == 1) {
        // Single value (bare --poller-warmup-mma = 8192): broadcast to every WDDM GPU.
        const int mma = mmas[0];
        if (mma < 0) {
            GGML_CUDA_LOG_WARN("%s: ignoring negative MMA length %d (bare --poller-warmup-mma = default)\n", __func__, mma);
        } else if (mma > 0) {
            for (int i = 0; i < GGML_CUDA_MAX_DEVICES; ++i) {
                ggml_cuda_poller_warmup_mma[i] = mma;
                ggml_cuda_poller_warmup_mma_override[i] = true;
            }
            any = true;
        }
    }
    ggml_cuda_poller_warmup_mma_any = any;
    if (any) {
        // Prime the warmup now, at enable time (before any decode): creates the
        // per-GPU streams, loads the kernel, and warms the WDDM context, so the
        // first TG-phase burst is not a cold first launch on the token path
        // (mirrors set_poller_warmup_fma / set_poller_warmup_mem).
        for (int i = 0; i < GGML_CUDA_MAX_DEVICES; ++i) {
            ggml_cuda_poller_warmup_due[i] = true; // prime bypasses the interval cadence
        }
        ggml_cuda_poller_warmup_mma_launch();
        GGML_CUDA_LOG_INFO("ggml_backend_cuda_set_poller_warmup_mma: enabling tensor-core warmup on WDDM GPUs:");
        for (int i = 0; i < GGML_CUDA_MAX_DEVICES; ++i) {
            if (ggml_cuda_poller_warmup_mma_override[i] && ggml_cuda_poller_warmup_mma[i] > 0) {
                GGML_CUDA_LOG_INFO(" %d:%d", i, ggml_cuda_poller_warmup_mma[i]);
            }
        }
        GGML_CUDA_LOG_INFO("\n");
    } else {
        GGML_CUDA_LOG_INFO("ggml_backend_cuda_set_poller_warmup_mma: disabled\n");
    }
}

// warmup-mem=N[,N,...]: per-WDDM-GPU mem burst count (number of 2 MiB passes) for the
// decode-gated mem burst (--poller-warmup-mem). The burst fires on that GPU at every TG batch
// alongside the (optional) warmup-fma; 0 = off for that GPU. A single value broadcasts to
// every WDDM GPU (bare --poller-warmup-mem = 1 burst); more values map positionally; missing
// values use the default. All zeros or n <= 0 disables.
GGML_CALL void ggml_backend_cuda_set_poller_warmup_mem(const int * bursts, int n) {
    for (int i = 0; i < GGML_CUDA_MAX_DEVICES; ++i) {
        ggml_cuda_poller_warmup_mem[i] = 0;
        ggml_cuda_poller_warmup_mem_override[i] = false;
    }
    bool any = false;
    if (n > 1) {
        const int m = std::min(n, GGML_CUDA_MAX_DEVICES);
        for (int i = 0; i < m; ++i) {
            const int b = bursts[i];
            if (b < 0) {
                GGML_CUDA_LOG_WARN("%s: ignoring negative mem burst count %d for WDDM[%d]\n", __func__, b, i);
                continue;
            }
            ggml_cuda_poller_warmup_mem[i] = b;
            ggml_cuda_poller_warmup_mem_override[i] = true;
            any |= b > 0;
        }
    } else if (n == 1) {
        // Single value (bare --poller-warmup-mem = 1): broadcast to every WDDM GPU.
        // A lone 0 disables the mem burst on every GPU.
        const int b = bursts[0];
        if (b < 0) {
            GGML_CUDA_LOG_WARN("%s: ignoring negative mem burst count %d (bare --poller-warmup-mem = default)\n", __func__, b);
        } else if (b > 0) {
            for (int i = 0; i < GGML_CUDA_MAX_DEVICES; ++i) {
                ggml_cuda_poller_warmup_mem[i] = b;
                ggml_cuda_poller_warmup_mem_override[i] = true;
            }
            any = true;
        }
    }
    ggml_cuda_poller_warmup_mem_any = any;
    if (any) {
        // Prime the streams + mem buffers at enable time (mirrors set_poller_warmup_fma), so
        // the first TG-phase burst is not a cold allocation on the token path.
        for (int i = 0; i < GGML_CUDA_MAX_DEVICES; ++i) {
            ggml_cuda_poller_warmup_due[i] = true; // prime bypasses the interval cadence
        }
        ggml_cuda_poller_warmup_mem_launch();
        GGML_CUDA_LOG_INFO("ggml_backend_cuda_set_poller_warmup_mem: enabling mem burst on WDDM GPUs:");
        for (int i = 0; i < GGML_CUDA_MAX_DEVICES; ++i) {
            if (ggml_cuda_poller_warmup_mem_override[i] && ggml_cuda_poller_warmup_mem[i] > 0) {
                GGML_CUDA_LOG_INFO(" %d:%d", i, ggml_cuda_poller_warmup_mem[i]);
            }
        }
        GGML_CUDA_LOG_INFO("\n");
    } else {
        GGML_CUDA_LOG_INFO("ggml_backend_cuda_set_poller_warmup_mem: disabled\n");
    }
}

// Set the current prompt length in tokens (called per decode by llama.cpp).
// n_ctx = the context size; k = floor(256 * prompt / n_ctx) clamped to 255
// selects the scaling bracket (see ggml_cuda_scale_fma_prompt for the piecewise
// curve). Both the decode-gated warmup and the autonomous FMA ping honor this.
// n_ctx <= 0 or n_prompt <= 0 resets to no scaling (full FMA).
GGML_CALL void ggml_backend_cuda_set_poller_prompt_len(int n_prompt, int n_ctx) {
    int k = 0;
    if (n_ctx > 0 && n_prompt > 0) {
        k = (int) (((int64_t) n_prompt * 256) / n_ctx);
        if (k > 255) k = 255;
    }
    ggml_cuda_poller_prompt_bracket.store(k);
}

// Per-WDDM-GPU token interval between warmup bursts (--poller-warmup-interval N[,N,...],
// aliases -p-warm-i / -warmstream). Applies to all three warmup functions (mma, fma, mem):
// on each WDDM GPU the burst fires on every N-th TG batch instead of every batch.
// A single value (bare --poller-warmup-interval = default 1) broadcasts to every WDDM GPU;
// more values map positionally; missing values use the default 1 (fire every batch).
GGML_CALL void ggml_backend_cuda_set_poller_warmup_interval(const int * intervals, int n) {
    for (int i = 0; i < GGML_CUDA_MAX_DEVICES; ++i) {
        ggml_cuda_poller_warmup_interval[i] = 0;
        ggml_cuda_poller_warmup_interval_override[i] = false;
    }
    if (n > 1) {
        const int m = std::min(n, GGML_CUDA_MAX_DEVICES);
        for (int i = 0; i < m; ++i) {
            const int interval = intervals[i];
            if (interval < 0) {
                GGML_CUDA_LOG_WARN("%s: ignoring interval %d for WDDM[%d] (must be >= 0, 0 = never fire)\n", __func__, interval, i);
                continue;
            }
            ggml_cuda_poller_warmup_interval[i] = interval;
            ggml_cuda_poller_warmup_interval_override[i] = true;
        }
    } else if (n == 1) {
        // Single value (bare --poller-warmup-interval = default): broadcast to every WDDM GPU.
        // A lone 0 disables the warmup burst on every GPU.
        const int interval = intervals[0];
        if (interval < 0) {
            GGML_CUDA_LOG_WARN("%s: ignoring interval %d (must be >= 0, 0 = never fire)\n", __func__, interval);
        } else {
            for (int i = 0; i < GGML_CUDA_MAX_DEVICES; ++i) {
                ggml_cuda_poller_warmup_interval[i] = interval;
                ggml_cuda_poller_warmup_interval_override[i] = true;
            }
        }
    }
    // Reset every countdown so the new cadence starts immediately at the next TG batch.
    for (int i = 0; i < GGML_CUDA_MAX_DEVICES; ++i) {
        ggml_cuda_poller_warmup_countdown[i] = ggml_cuda_poller_warmup_interval_override[i] && ggml_cuda_poller_warmup_interval[i] > 0
            ? ggml_cuda_poller_warmup_interval[i] : GGML_CUDA_POLLER_WARMUP_INTERVAL_DEFAULT;
    }
    GGML_CUDA_LOG_INFO("ggml_backend_cuda_set_poller_warmup_interval: warmup interval per WDDM GPU:");
    for (int i = 0; i < GGML_CUDA_MAX_DEVICES; ++i) {
        if (ggml_cuda_poller_warmup_interval_override[i]) {
            GGML_CUDA_LOG_INFO(" %d:%d", i, ggml_cuda_poller_warmup_interval[i]);
        }
    }
    GGML_CUDA_LOG_INFO("\n");
}

// First TG token at which the warmup burst fires (--poller-warmup-start N[,N,...],
// aliases -p-warm-s / -streamsource). Applies to all three warmup functions (mma, fma, mem):
// on each WDDM GPU the burst first fires on the N-th TG batch (token) of a phase instead of
// the historical default second token. A single value (bare --poller-warmup-start = default 2)
// broadcasts to every WDDM GPU; more values map positionally; missing values use the default 2
// (fire on the second TG token, the historical skip-first-batch behavior); 0 = never fire.
GGML_CALL void ggml_backend_cuda_set_poller_warmup_start(const int * starts, int n) {
    for (int i = 0; i < GGML_CUDA_MAX_DEVICES; ++i) {
        ggml_cuda_poller_warmup_start[i] = 0;
        ggml_cuda_poller_warmup_start_override[i] = false;
    }
    if (n > 1) {
        const int m = std::min(n, GGML_CUDA_MAX_DEVICES);
        for (int i = 0; i < m; ++i) {
            const int start = starts[i];
            if (start < 0) {
                GGML_CUDA_LOG_WARN("%s: ignoring start %d for WDDM[%d] (must be >= 0, 0 = never fire)\n", __func__, start, i);
                continue;
            }
            ggml_cuda_poller_warmup_start[i] = start;
            ggml_cuda_poller_warmup_start_override[i] = true;
        }
    } else if (n == 1) {
        // Single value (bare --poller-warmup-start = default): broadcast to every WDDM GPU.
        // A lone 0 disables the warmup burst on every GPU.
        const int start = starts[0];
        if (start < 0) {
            GGML_CUDA_LOG_WARN("%s: ignoring start %d (must be >= 0, 0 = never fire)\n", __func__, start);
        } else {
            for (int i = 0; i < GGML_CUDA_MAX_DEVICES; ++i) {
                ggml_cuda_poller_warmup_start[i] = start;
                ggml_cuda_poller_warmup_start_override[i] = true;
            }
        }
    }
    GGML_CUDA_LOG_INFO("ggml_backend_cuda_set_poller_warmup_start: warmup start token per WDDM GPU:");
    for (int i = 0; i < GGML_CUDA_MAX_DEVICES; ++i) {
        if (ggml_cuda_poller_warmup_start_override[i]) {
            GGML_CUDA_LOG_INFO(" %d:%d", i, ggml_cuda_poller_warmup_start[i]);
        }
    }
    GGML_CUDA_LOG_INFO("\n");
}
// Occupancy percentage of the poller FMA kernels (--poller-fma-occupancy N[,N,...], aliases
// -p-fma-o / -fishpit). Per-WDDM-GPU (positional) float, 0..100 = direct percentage:
// 0 = disabled (no burst on that GPU), 100 = full grid (16 blocks/SM). Single value
// broadcasts to every WDDM GPU; more values map positionally; missing values use the
// given default (50, applied whenever any FMA poller is used without the flag). Applies
// to the warmup, the decode-solicited activity probes and the autonomous ping thread.
GGML_CALL void ggml_backend_cuda_set_poller_fma_occupancy(const float * occ, int n, float default_occ) {
    for (int i = 0; i < GGML_CUDA_MAX_DEVICES; ++i) {
        ggml_cuda_poller_fma_occupancy[i] = default_occ;
    }
    if (n > 1) {
        const int m = std::min(n, GGML_CUDA_MAX_DEVICES);
        for (int i = 0; i < m; ++i) {
            const float pct = occ[i];
            if (pct < 0.0f || pct > 100.0f) {
                GGML_CUDA_LOG_WARN("%s: ignoring occupancy %.1f%% for WDDM[%d] (must be 0..100, 0 = disabled)\n", __func__, pct, i);
                continue;
            }
            ggml_cuda_poller_fma_occupancy[i] = pct;
        }
    } else if (n == 1) {
        // Single value (bare --poller-fma-occupancy = default): broadcast to every WDDM GPU.
        const float pct = occ[0];
        if (pct < 0.0f || pct > 100.0f) {
            GGML_CUDA_LOG_WARN("%s: ignoring occupancy %.1f%% (must be 0..100, 0 = disabled)\n", __func__, pct);
        } else {
            for (int i = 0; i < GGML_CUDA_MAX_DEVICES; ++i) {
                ggml_cuda_poller_fma_occupancy[i] = pct;
            }
        }
    }
    GGML_CUDA_LOG_INFO("ggml_backend_cuda_set_poller_fma_occupancy: occupancy %% per WDDM GPU:");
    for (int i = 0; i < GGML_CUDA_MAX_DEVICES; ++i) {
        GGML_CUDA_LOG_INFO(" %d:%.1f", i, ggml_cuda_poller_fma_occupancy[i]);
    }
    GGML_CUDA_LOG_INFO("\n");
}

// Occupancy percentage of the poller MMA kernels (--poller-mma-occupancy N[,N,...], aliases
// -p-mma-o / -abyss). Per-WDDM-GPU (positional) float, 0..100 = direct percentage:
// 0 = disabled (no burst on that GPU), 100 = full grid (16 blocks/SM). Single value
// broadcasts to every WDDM GPU; more values map positionally; missing values use the
// given default (50, applied whenever any MMA poller is used without the flag). Applies
// to the warmup, the decode-solicited activity probes and the autonomous ping thread.
GGML_CALL void ggml_backend_cuda_set_poller_mma_occupancy(const float * occ, int n, float default_occ) {
    for (int i = 0; i < GGML_CUDA_MAX_DEVICES; ++i) {
        ggml_cuda_poller_mma_occupancy[i] = default_occ;
    }
    if (n > 1) {
        const int m = std::min(n, GGML_CUDA_MAX_DEVICES);
        for (int i = 0; i < m; ++i) {
            const float pct = occ[i];
            if (pct < 0.0f || pct > 100.0f) {
                GGML_CUDA_LOG_WARN("%s: ignoring occupancy %.1f%% for WDDM[%d] (must be 0..100, 0 = disabled)\n", __func__, pct, i);
                continue;
            }
            ggml_cuda_poller_mma_occupancy[i] = pct;
        }
    } else if (n == 1) {
        // Single value (bare --poller-mma-occupancy = default): broadcast to every WDDM GPU.
        const float pct = occ[0];
        if (pct < 0.0f || pct > 100.0f) {
            GGML_CUDA_LOG_WARN("%s: ignoring occupancy %.1f%% (must be 0..100, 0 = disabled)\n", __func__, pct);
        } else {
            for (int i = 0; i < GGML_CUDA_MAX_DEVICES; ++i) {
                ggml_cuda_poller_mma_occupancy[i] = pct;
            }
        }
    }
    GGML_CUDA_LOG_INFO("ggml_backend_cuda_set_poller_mma_occupancy: occupancy %% per WDDM GPU:");
    for (int i = 0; i < GGML_CUDA_MAX_DEVICES; ++i) {
        GGML_CUDA_LOG_INFO(" %d:%.1f", i, ggml_cuda_poller_mma_occupancy[i]);
    }
    GGML_CUDA_LOG_INFO("\n");
}

// L2 occupancy percentage of the poller mem companion (--poller-mem-occupancy N[,N,...],
// aliases -p-mem-o / -snakepit). Per-WDDM-GPU (positional) float, 0..100 = direct
// percentage: 0 = disabled (no burst on that GPU), 100 = full 2 MiB buffer per pass.
// Single value broadcasts to every WDDM GPU; more values map positionally; missing
// values use the given default (25, applied whenever any mem poller is used without the
// flag). Applies to the warmup, the decode-solicited activity probes and the ping thread.
GGML_CALL void ggml_backend_cuda_set_poller_mem_occupancy(const float * occ, int n, float default_occ) {
    for (int i = 0; i < GGML_CUDA_MAX_DEVICES; ++i) {
        ggml_cuda_poller_mem_occupancy[i] = default_occ;
    }
    if (n > 1) {
        const int m = std::min(n, GGML_CUDA_MAX_DEVICES);
        for (int i = 0; i < m; ++i) {
            const float pct = occ[i];
            if (pct < 0.0f || pct > 100.0f) {
                GGML_CUDA_LOG_WARN("%s: ignoring occupancy %.1f%% for WDDM[%d] (must be 0..100, 0 = disabled)\n", __func__, pct, i);
                continue;
            }
            ggml_cuda_poller_mem_occupancy[i] = pct;
        }
    } else if (n == 1) {
        // Single value (bare --poller-mem-occupancy = default): broadcast to every WDDM GPU.
        const float pct = occ[0];
        if (pct < 0.0f || pct > 100.0f) {
            GGML_CUDA_LOG_WARN("%s: ignoring occupancy %.1f%% (must be 0..100, 0 = disabled)\n", __func__, pct);
        } else {
            for (int i = 0; i < GGML_CUDA_MAX_DEVICES; ++i) {
                ggml_cuda_poller_mem_occupancy[i] = pct;
            }
        }
    }
    GGML_CUDA_LOG_INFO("ggml_backend_cuda_set_poller_mem_occupancy: occupancy %% per WDDM GPU:");
    for (int i = 0; i < GGML_CUDA_MAX_DEVICES; ++i) {
        GGML_CUDA_LOG_INFO(" %d:%.1f", i, ggml_cuda_poller_mem_occupancy[i]);
    }
    GGML_CUDA_LOG_INFO("\n");
}

GGML_CALL void ggml_backend_cuda_set_poller_active(bool val) {
    ggml_cuda_poller_gate.store(val);
    if (!ggml_cuda_poller_warmup_fma && !ggml_cuda_poller_warmup_mem_any && !ggml_cuda_poller_warmup_mma_any) {
        // all of warmup-fma/warmup-mem/warmup-mma disabled: ignore TG/PP transitions entirely
        ggml_cuda_poller_active = false;
        return;
    }
    const bool changed = (val != ggml_cuda_poller_active);
    ggml_cuda_poller_active = val;
    if (val) {
        // A TG phase just started (changed): arm each slot's countdown from its start
        // token (--poller-warmup-start, default 2 = fire on the second TG token, i.e. the
        // historical skip-first-batch behavior: the trailing PP kernels left the GPU at
        // full load and the first decode provides its own activity, so the transition
        // burst would only contend with the first token).
        if (changed) {
            for (int w = 0; w < GGML_CUDA_MAX_DEVICES; ++w) {
                ggml_cuda_poller_warmup_countdown[w] = ggml_cuda_poller_warmup_start_override[w] ? ggml_cuda_poller_warmup_start[w] : GGML_CUDA_POLLER_WARMUP_START_DEFAULT;
            }
        }
        // Decrement each slot's countdown; when it hits 0 the burst is due this batch and
        // the countdown re-arms from the slot's interval (--poller-warmup-interval).
        // Combined with the real decode kernels this sustains SM activity during the whole
        // TG phase, keeping the clock boost alive. ~us fire & forget.
        for (int w = 0; w < GGML_CUDA_MAX_DEVICES; ++w) {
            const int start = ggml_cuda_poller_warmup_start_override[w] ? ggml_cuda_poller_warmup_start[w] : GGML_CUDA_POLLER_WARMUP_START_DEFAULT;
            const int interval = ggml_cuda_poller_warmup_interval_override[w] ? ggml_cuda_poller_warmup_interval[w] : GGML_CUDA_POLLER_WARMUP_INTERVAL_DEFAULT;
            if (start <= 0 || interval <= 0 || --ggml_cuda_poller_warmup_countdown[w] > 0) {
                ggml_cuda_poller_warmup_due[w] = false;
            } else {
                ggml_cuda_poller_warmup_countdown[w] = interval;
                ggml_cuda_poller_warmup_due[w] = true;
            }
        }
        // MMA first: the densest power pulse (2048 FLOPs/HMMA) gives the clock
        // governor the sharpest leading edge, then the lighter FMA chain sustains
        // the SM load, and the mem burst tails it keeping DRAM writes active up to
        // the next real kernel.
        if (ggml_cuda_poller_warmup_mma_any) {
            ggml_cuda_poller_warmup_mma_launch();
        }
        if (ggml_cuda_poller_warmup_fma) {
            ggml_cuda_poller_warmup_fma_launch();
        }
        if (ggml_cuda_poller_warmup_mem_any) {
            ggml_cuda_poller_warmup_mem_launch();
        }
    }
    if (changed) {
        GGML_CUDA_LOG_INFO("ggml_backend_cuda_set_poller_active: %s\n",
            val ? "heartbeat active during TG (GPU clocks elevated)"
                : "heartbeat inactive during PP (GPU clocks may drop)");
    }
}
