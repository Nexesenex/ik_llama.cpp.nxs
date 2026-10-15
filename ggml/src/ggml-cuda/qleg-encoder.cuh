#pragma once

#include "common.cuh"


// --- Q6_0 OLS kept (make_qx_quants+fudge; plain max/-32 unused) ---

// --- make_qx_quants device port ---

// Bit-twiddle FP16 matching ggml_compute_fp32_to_fp16; NaN payload is vendor-specific (harness nan_d_equal).
static __device__ __forceinline__ uint16_t fp32_to_fp16_ggml(float f) {
    const float scale_to_inf  = __int_as_float(0x77800000u);
    const float scale_to_zero = __int_as_float(0x08800000u);
    float base = __fmul_rn(__fmul_rn(fabsf(f), scale_to_inf), scale_to_zero);

    const uint32_t w      = __float_as_uint(f);
    const uint32_t shl1_w = w + w;
    const uint32_t sign   = w & 0x80000000u;
    uint32_t bias = shl1_w & 0xFF000000u;
    if (bias < 0x71000000u) {
        bias = 0x71000000u;
    }

    base = __fadd_rn(__int_as_float((bias >> 1) + 0x07800000u), base);
    const uint32_t bits          = __float_as_uint(base);
    const uint32_t exp_bits      = (bits >> 13) & 0x00007C00u;
    const uint32_t mantissa_bits = bits & 0x00000FFFu;
    const uint32_t nonsign       = exp_bits + mantissa_bits;
    return (uint16_t)((sign >> 16) | (shl1_w > 0xFF000000u ? 0x7E00u : nonsign));
}


// round-half-to-even via the 2^23 + 2^22 magic constant (ggml-quants.c:1779)
static __device__ int nearest_int_device(float fval) {
    const unsigned int u = __float_as_uint(__fadd_rn(fval, 12582912.0f));
    return (int)((u & 0x007fffffu) - 0x00400000u);
}


// clamp_l_device(l, nmax) as in make_qx_quants
static __device__ int clamp_l_device(int l, int nmax) {
    return l > nmax-1 ? nmax-1 : (l < -nmax ? -nmax : l);
}


// Q4 nibble pack for one-thread-per-block kernels (L in [0,15]).
static __device__ void pack_q4_nibbles_device(const uint8_t * L, uint8_t * qs) {
    for (int j = 0; j < 16; ++j) {
        qs[j] = (uint8_t)(L[j] | (L[j + 16] << 4));
    }
}


// make_qx_quants port (rmse_type==1 only); RN intrinsics, exact CPU order, no FMA.
static __device__ float make_qx_quants_device(int n, int nmax, const float * x, int8_t * L, const float * qw) {
    float max  = 0.0f;
    float amax = 0.0f;
    for (int i = 0; i < n; ++i) {
        const float ax = fabsf(x[i]);
        if (ax > amax) { amax = ax; max = x[i]; }
    }
    if (amax < 1e-15f) { // GROUP_MAX_EPS: all zero
        for (int i = 0; i < n; ++i) L[i] = 0;
        return 0.0f;
    }
    float iscale = __fdiv_rn(-(float)nmax, max);

    float sumlx = 0.0f;
    float suml2 = 0.0f;
    for (int i = 0; i < n; ++i) {
        int l = nearest_int_device(__fmul_rn(iscale, x[i]));
        l = clamp_l_device(l, nmax);
        L[i] = (int8_t)(l + nmax);
        const float w = qw[i]; // rmse_type == 1, qw always provided
        sumlx = __fadd_rn(sumlx, __fmul_rn(__fmul_rn(w, x[i]), (float)l));
        suml2 = __fadd_rn(suml2, __fmul_rn(__fmul_rn(w, (float)l), (float)l));
    }
    float scale = suml2 != 0.0f ? __fdiv_rn(sumlx, suml2) : 0.0f;
    float best = __fmul_rn(scale, sumlx);
    float best_sumlx = sumlx, best_suml2 = suml2;

    for (int is = -9; is <= 9; ++is) {
        iscale = __fdiv_rn(-__fadd_rn((float)nmax, __fmul_rn(0.1f, (float)is)), max);
        sumlx = suml2 = 0.0f;
        for (int i = 0; i < n; ++i) {
            int l = nearest_int_device(__fmul_rn(iscale, x[i]));
            l = clamp_l_device(l, nmax);
            const float w = qw[i];
            sumlx = __fadd_rn(sumlx, __fmul_rn(__fmul_rn(w, x[i]), (float)l));
            suml2 = __fadd_rn(suml2, __fmul_rn(__fmul_rn(w, (float)l), (float)l));
        }
        if (suml2 > 0.0f && __fmul_rn(sumlx, sumlx) > __fmul_rn(best, suml2)) {
            for (int i = 0; i < n; ++i) {
                int l = nearest_int_device(__fmul_rn(iscale, x[i]));
                L[i] = (int8_t)(nmax + clamp_l_device(l, nmax));
            }
            scale = __fdiv_rn(sumlx, suml2);
            best = __fmul_rn(scale, sumlx);
            best_sumlx = sumlx; best_suml2 = suml2;
        }
        iscale = __fdiv_rn(__fadd_rn((float)(nmax-1), __fmul_rn(0.1f, (float)is)), max);
        sumlx = suml2 = 0.0f;
        for (int i = 0; i < n; ++i) {
            int l = nearest_int_device(__fmul_rn(iscale, x[i]));
            l = clamp_l_device(l, nmax);
            const float w = qw[i];
            sumlx = __fadd_rn(sumlx, __fmul_rn(__fmul_rn(w, x[i]), (float)l));
            suml2 = __fadd_rn(suml2, __fmul_rn(__fmul_rn(w, (float)l), (float)l));
        }
        if (suml2 > 0.0f && __fmul_rn(sumlx, sumlx) > __fmul_rn(best, suml2)) {
            for (int i = 0; i < n; ++i) {
                int l = nearest_int_device(__fmul_rn(iscale, x[i]));
                L[i] = (int8_t)(nmax + clamp_l_device(l, nmax));
            }
            scale = __fdiv_rn(sumlx, suml2);
            best = __fmul_rn(scale, sumlx);
            best_sumlx = sumlx; best_suml2 = suml2;
        }
    }

    // coordinate descent; identical step sequence to the reference
    sumlx = best_sumlx; suml2 = best_suml2;
    for (int iter = 0; iter < n*(2*nmax-1); ++iter) {
        float abs_gmax = 0.0f, gmax = 0.0f;
        int best_j = -1;
        for (int j = 0; j < n; ++j) {
            const float w = qw[j];
            const int l = (int)L[j] - nmax;
            // g = scale*w*(x[j] - scale*l), each op rounded separately
            const float g = __fmul_rn(__fmul_rn(scale, w),
                    __fadd_rn(x[j], -__fmul_rn(scale, (float)l)));
            if ((g > 0.0f && l < nmax-1) || (g < 0.0f && l > -nmax)) {
                const float ag = fabsf(g);
                if (ag > abs_gmax) { abs_gmax = ag; gmax = g; best_j = j; }
            }
        }
        if (best_j < 0) break;

        float new_sumlx = sumlx, new_suml2 = suml2;
        const float w = qw[best_j];
        int l = (int)L[best_j] - nmax;
        if (gmax > 0.0f) {
            new_sumlx = __fadd_rn(new_sumlx, __fmul_rn(w, x[best_j]));
            new_suml2 = __fadd_rn(new_suml2, __fmul_rn(w, (float)(2*l + 1)));
            l += 1;
        } else {
            new_sumlx = __fsub_rn(new_sumlx, __fmul_rn(w, x[best_j]));
            new_suml2 = __fsub_rn(new_suml2, __fmul_rn(w, (float)(2*l - 1)));
            l -= 1;
        }
        if (new_suml2 > 0.0f && __fmul_rn(new_sumlx, new_sumlx) > __fmul_rn(best, new_suml2)) {
            sumlx = new_sumlx; suml2 = new_suml2;
            scale = __fdiv_rn(sumlx, suml2);
            best = __fmul_rn(scale, sumlx);
            L[best_j] = (int8_t)(l + nmax);
        } else {
            break;
        }
    }
    return scale;
}


// --- make_qkx3_quants device port (Q4_1/Q5_1; double RN intrinsics, CPU order) ---
static __device__ float make_qkx3_quants_device(int n, int nmax, const float * x, const float * weights,
        uint8_t * L, float * the_min, uint8_t * Laux,
        float rmin, float rdelta, int nstep, bool use_mad) {
    float min = x[0];
    float max = x[0];
    double sum_w = weights ? (double)weights[0] : (double)(x[0]*x[0]);
    double sum_x = sum_w * (double)x[0];
    double sum_x2 = sum_w * (double)x[0] * (double)x[0];
    for (int i = 1; i < n; ++i) {
        if (x[i] < min) min = x[i];
        if (x[i] > max) max = x[i];
        float w = weights ? weights[i] : x[i]*x[i];
        sum_w = __dadd_rn(sum_w, (double)w);
        sum_x = __dadd_rn(sum_x, __dmul_rn((double)w, (double)x[i]));
        sum_x2 = __dadd_rn(sum_x2, __dmul_rn(__dmul_rn((double)w, (double)x[i]), (double)x[i]));
    }
    if (min > 0) {
        min = 0;
    }
    if (max - min < 1e-10f) {
        for (int i = 0; i < n; ++i) L[i] = 0;
        *the_min = -min;
        return 0.f;
    }
    float iscale = __fdiv_rn((float)nmax, __fsub_rn(max, min));
    float scale = __fdiv_rn(1.0f, iscale);
    double best_mad = 0;
    for (int i = 0; i < n; ++i) {
        int l = nearest_int_device(__fmul_rn(iscale, __fsub_rn(x[i], min)));
        l = l > nmax ? nmax : (l < 0 ? 0 : l);
        L[i] = (uint8_t)l;
        double diff = __dadd_rn(__dmul_rn((double)scale, (double)L[i]), (double)min);
        diff = __dsub_rn(diff, (double)x[i]);
        diff = use_mad ? fabs(diff) : __dmul_rn(diff, diff);
        double w = weights ? (double)weights[i] : (double)(x[i]*x[i]);
        best_mad = __dadd_rn(best_mad, __dmul_rn(w, diff));
    }
    if (nstep < 1) {
        *the_min = -min;
        return scale;
    }
    for (int is = 0; is <= nstep; ++is) {
        iscale = __fdiv_rn(__fadd_rn(__fadd_rn(rmin, __fmul_rn(rdelta, (float)is)), (float)nmax), __fsub_rn(max, min));
        double sum_l = 0, sum_l2 = 0, sum_xl = 0;
        for (int i = 0; i < n; ++i) {
            int l = nearest_int_device(__fmul_rn(iscale, __fsub_rn(x[i], min)));
            l = l > nmax ? nmax : (l < 0 ? 0 : l);
            Laux[i] = (uint8_t)l;
            float w = weights ? weights[i] : x[i]*x[i];
            sum_l  = __dadd_rn(sum_l, __dmul_rn((double)w, (double)l));
            sum_l2 = __dadd_rn(sum_l2, __dmul_rn(__dmul_rn((double)w, (double)l), (double)l));
            sum_xl = __dadd_rn(sum_xl, __dmul_rn(__dmul_rn((double)w, (double)l), (double)x[i]));
        }
        double D = __dsub_rn(__dmul_rn(sum_w, sum_l2), __dmul_rn(sum_l, sum_l));
        if (D > 0) {
            double this_scale = __ddiv_rn(__dsub_rn(__dmul_rn(sum_w, sum_xl), __dmul_rn(sum_x, sum_l)), D);
            double this_min   = __ddiv_rn(__dsub_rn(__dmul_rn(sum_l2, sum_x), __dmul_rn(sum_l, sum_xl)), D);
            if (this_min > 0) {
                this_min = 0;
                this_scale = __ddiv_rn(sum_xl, sum_l2);
            }
            double mad = 0;
            if (use_mad) {
                for (int i = 0; i < n; ++i) {
                    double diff = __dadd_rn(__dmul_rn((double)this_scale, (double)Laux[i]), (double)this_min);
                    diff = __dsub_rn(diff, (double)x[i]);
                    diff = fabs(diff);
                    double w = weights ? (double)weights[i] : (double)(x[i]*x[i]);
                    mad = __dadd_rn(mad, __dmul_rn(w, diff));
                }
            } else {
                mad = __dsub_rn(sum_x2, __dmul_rn(2*this_scale, sum_xl));
                mad = __dsub_rn(mad, __dmul_rn(2*this_min, sum_x));
                mad = __dadd_rn(mad, __dmul_rn(__dmul_rn(2*this_scale, this_min), sum_l));
                mad = __dadd_rn(mad, __dmul_rn(__dmul_rn(this_scale, this_scale), sum_l2));
                mad = __dadd_rn(mad, __dmul_rn(__dmul_rn(this_min, this_min), sum_w));
            }
            if (mad < best_mad) {
                for (int i = 0; i < n; ++i) {
                    L[i] = Laux[i];
                }
                best_mad = mad;
                scale = (float)this_scale;
                min = (float)this_min;
            }
        }
    }
    if (use_mad) {
        *the_min = -min;
        return scale;
    }

    double sum_l = 0, sum_l2 = 0, sum_xl = 0;
    for (int i = 0; i < n; ++i) {
        int l = L[i];
        double w = weights ? (double)weights[i] : (double)(x[i]*x[i]);
        sum_l  = __dadd_rn(sum_l, __dmul_rn(w, (double)l));
        sum_l2 = __dadd_rn(sum_l2, __dmul_rn(__dmul_rn(w, (double)l), (double)l));
        sum_xl = __dadd_rn(sum_xl, __dmul_rn(__dmul_rn(w, (double)l), (double)x[i]));
    }
    double best = __dsub_rn(
            __dadd_rn(__dmul_rn(__dmul_rn(2.0, (double)scale), sum_xl),
                      __dmul_rn(__dmul_rn(2.0, (double)min), sum_x)),
            __dmul_rn(__dmul_rn(__dmul_rn(2.0, (double)scale), (double)min), sum_l));
    best = __dsub_rn(best, __dmul_rn(__dmul_rn((double)scale, (double)scale), sum_l2));
    best = __dsub_rn(best, __dmul_rn(__dmul_rn((double)min, (double)min), sum_w));
    int last_j = -1, last_dir = 0;
    for (int itry = 0; itry < nmax*n; ++itry) {
        float gmax = 0;
        int best_j = -1, dir = 0;
        for (int j = 0; j < n; ++j) {
            // NOTE float (not double): CPU computes g in float32.
            const float g = __fsub_rn(__fsub_rn(x[j], __fmul_rn(scale, (float)L[j])), min);
            if (g > 0 && L[j] < nmax && g > gmax) {
                gmax = g; best_j = j; dir = 1;
            }
            else if (g < 0 && L[j] > 0 && -g > gmax) {
                gmax = -g; best_j = j; dir = -1;
            }
        }
        if (best_j < 0 || (best_j == last_j && dir == -last_dir)) break;
        double w = weights ? (double)weights[best_j] : (double)(x[best_j]*x[best_j]);
        sum_l  = __dadd_rn(sum_l, __dmul_rn(w, (double)dir));
        sum_l2 = __dadd_rn(sum_l2, __dmul_rn(w, (double)(2*L[best_j]*dir + 1)));
        sum_xl = __dadd_rn(sum_xl, __dmul_rn(__dmul_rn(w, (double)x[best_j]), (double)dir));
        double D = __dsub_rn(__dmul_rn(sum_w, sum_l2), __dmul_rn(sum_l, sum_l));
        if (D <= 0) break;
        double this_scale = __ddiv_rn(__dsub_rn(__dmul_rn(sum_w, sum_xl), __dmul_rn(sum_x, sum_l)), D);
        double this_min   = __ddiv_rn(__dsub_rn(__dmul_rn(sum_l2, sum_x), __dmul_rn(sum_l, sum_xl)), D);
        if (this_min > 0) {
            this_min = 0;
            this_scale = __ddiv_rn(sum_xl, sum_l2);
        }
        if (this_scale < 0) break;
        double score = __dadd_rn(__dmul_rn(2*this_scale, sum_xl), __dmul_rn(2*this_min, (double)sum_x));
        score = __dsub_rn(score, __dmul_rn(__dmul_rn(2*this_scale, this_min), sum_l));
        score = __dsub_rn(score, __dmul_rn(__dmul_rn(this_scale, this_scale), sum_l2));
        score = __dsub_rn(score, __dmul_rn(__dmul_rn(this_min, this_min), sum_w));
        if (score <= best) break;
        best = score;
        scale = (float)this_scale;
        min = (float)this_min;
        L[best_j] += (uint8_t)dir;
        last_j = best_j; last_dir = dir;
    }
    *the_min = -min;
    return scale;
}


// make_qp_quants port (ggml-quants.c:2361); uint8 wrap + MIN-only clamps kept bit-exact.
static __device__ float make_qp_quants_device(int n, int nmax, const float * x, uint8_t * L, const float * qw) {
    float max = 0.0f;
    for (int i = 0; i < n; ++i) {
        max = max > x[i] ? max : x[i];
    }
    if (max < 1e-16f) {
        for (int i = 0; i < n; ++i) L[i] = 0;
        return 0.0f;
    }
    float iscale = __fdiv_rn((float)nmax, max);
    for (int i = 0; i < n; ++i) {
        L[i] = (uint8_t)nearest_int_device(__fmul_rn(iscale, x[i]));
    }
    float scale = __fdiv_rn(1.0f, iscale);
    float best_mse = 0.0f;
    for (int i = 0; i < n; ++i) {
        const float diff = __fsub_rn(x[i], __fmul_rn(scale, (float)L[i]));
        const float w = qw[i];
        best_mse = __fadd_rn(best_mse, __fmul_rn(__fmul_rn(w, diff), diff));
    }
    for (int is = -4; is <= 4; ++is) {
        if (is == 0) continue;
        const float iscale_is = __fdiv_rn(__fadd_rn(__fmul_rn(0.1f, (float)is), (float)nmax), max);
        const float scale_is = __fdiv_rn(1.0f, iscale_is);
        float mse = 0.0f;
        for (int i = 0; i < n; ++i) {
            int l = nearest_int_device(__fmul_rn(iscale_is, x[i]));
            l = l > nmax ? nmax : l;
            const float diff = __fsub_rn(x[i], __fmul_rn(scale_is, (float)l));
            const float w = qw[i];
            mse = __fadd_rn(mse, __fmul_rn(__fmul_rn(w, diff), diff));
        }
        if (mse < best_mse) {
            best_mse = mse;
            iscale = iscale_is;
        }
    }
    float sumlx = 0.0f, suml2 = 0.0f;
    for (int i = 0; i < n; ++i) {
        int l = nearest_int_device(__fmul_rn(iscale, x[i]));
        l = l > nmax ? nmax : l;
        L[i] = (uint8_t)l;
        const float w = qw[i];
        sumlx = __fadd_rn(sumlx, __fmul_rn(__fmul_rn(w, x[i]), (float)l));
        suml2 = __fadd_rn(suml2, __fmul_rn(__fmul_rn(w, (float)l), (float)l));
    }
    for (int itry = 0; itry < 5; ++itry) {
        int n_changed = 0;
        for (int i = 0; i < n; ++i) {
            const float w = qw[i];
            const float slx = __fsub_rn(sumlx, __fmul_rn(__fmul_rn(w, x[i]), (float)L[i]));
            const float sl2 = __fsub_rn(suml2, __fmul_rn(__fmul_rn(w, (float)L[i]), (float)L[i]));
            if (slx > 0.0f && sl2 > 0.0f) {
                int new_l = nearest_int_device(__fdiv_rn(__fmul_rn(x[i], sl2), slx));
                new_l = new_l > nmax ? nmax : new_l;
                if (new_l != (int)L[i]) {
                    const float nslx = __fadd_rn(slx, __fmul_rn(__fmul_rn(w, x[i]), (float)new_l));
                    const float nsl2 = __fadd_rn(sl2, __fmul_rn(__fmul_rn(w, (float)new_l), (float)new_l));
                    if (__fmul_rn(__fmul_rn(nslx, nslx), suml2) > __fmul_rn(__fmul_rn(sumlx, sumlx), nsl2)) {
                        L[i] = (uint8_t)new_l;
                        sumlx = nslx;
                        suml2 = nsl2;
                        ++n_changed;
                    }
                }
            }
        }
        if (!n_changed) break;
    }
    return __fdiv_rn(sumlx, suml2);
}


// Per-block imatrix weights, shared by all imatrix kernels.
static __device__ void imatrix_weights_device(const float * xb, const float * qb, float s2, float * weight, int n) {
    for (int j = 0; j < n; ++j) {
        // __fsqrt_rn keeps sqrt exact under -use_fast_math, no FMA.
        weight[j] = __fmul_rn(qb[j], __fsqrt_rn(__fadd_rn(s2, __fmul_rn(xb[j], xb[j]))));
    }
}


// On-device superblock sigma: 2x (S/XXXS/S) vs 1x (XS/XXS), exact powers of 2.
static __device__ float sigma2_2x_device(const float * xbl) {
    float sum = 0.0f;
    for (int j = 0; j < QK_K; ++j) sum = __fadd_rn(sum, __fmul_rn(xbl[j], xbl[j]));
    return __fmul_rn(sum, 2.0f/(float)QK_K);
}


// --- generic chunked host drivers (~128 MiB, shared) ---

using quantize_kernel_t = void (*)(const float *, void *, int64_t, float);

// Streaming chunk executor shared by the plain/OLS and imatrix drivers.
// Double-buffered overlap across chunks: while chunk i runs (kernel + D2H)
// on the compute stream, chunk i+1 uploads on the transfer stream and the
// host stages/copies through pinned memory. launch(dx, dy, base, nblocks, st)
// issues the type-specific kernel launch on the given stream. Falls back to
// synchronous copies when streams or pinned staging are unavailable.
// Returns bytes written, or 0 on failure (caller falls back to CPU).
template <typename LaunchFn>
static size_t quantize_exec_chunked(const float * src, void * dst, int64_t nblocks_total,
        int64_t qk, size_t blk_size, int64_t chunk_blocks, const char * name, LaunchFn launch) {
    const int64_t chunk_x = chunk_blocks*qk;
    const int64_t chunk_y = chunk_blocks*(int64_t)blk_size;

    float   * x_dev[2] = {nullptr, nullptr};
    uint8_t * y_dev[2] = {nullptr, nullptr};
    float   * x_pin[2] = {nullptr, nullptr};
    uint8_t * y_pin[2] = {nullptr, nullptr};
    cudaStream_t st_h2d = nullptr, st_exe = nullptr;
    cudaEvent_t ev_h2d[2] = {nullptr, nullptr}, ev_done[2] = {nullptr, nullptr};

    bool ok = true;
    cudaError_t err = cudaSuccess;
    // Single-chunk tensors gain nothing from pipelining; use the synchronous
    // path directly (also avoids pinned staging).
    if (nblocks_total <= chunk_blocks) {
        ok = false;
    }
    for (int b = 0; b < 2 && ok; ++b) {
        if ((err = cudaMalloc(&x_dev[b], chunk_x*sizeof(float))) != cudaSuccess) ok = false;
        if (ok && (err = cudaMalloc(&y_dev[b], chunk_y)) != cudaSuccess) ok = false;
        if (ok && (err = cudaMallocHost(&x_pin[b], chunk_x*sizeof(float))) != cudaSuccess) ok = false;
        if (ok && (err = cudaMallocHost(&y_pin[b], chunk_y)) != cudaSuccess) ok = false;
        if (ok && (err = cudaEventCreateWithFlags(&ev_h2d[b], cudaEventDisableTiming)) != cudaSuccess) ok = false;
        if (ok && (err = cudaEventCreateWithFlags(&ev_done[b], cudaEventDisableTiming)) != cudaSuccess) ok = false;
    }
    if (ok && (err = cudaStreamCreate(&st_h2d)) != cudaSuccess) ok = false;
    if (ok && (err = cudaStreamCreate(&st_exe)) != cudaSuccess) ok = false;

    auto release = [&]() {
        for (int b = 0; b < 2; ++b) {
            if (x_dev[b]) cudaFree(x_dev[b]);
            if (y_dev[b]) cudaFree(y_dev[b]);
            if (x_pin[b]) cudaFreeHost(x_pin[b]);
            if (y_pin[b]) cudaFreeHost(y_pin[b]);
            if (ev_h2d[b]) cudaEventDestroy(ev_h2d[b]);
            if (ev_done[b]) cudaEventDestroy(ev_done[b]);
        }
        if (st_h2d) cudaStreamDestroy(st_h2d);
        if (st_exe) cudaStreamDestroy(st_exe);
    };

    if (!ok) {
        // Synchronous fallback: single buffers, blocking copies (old behavior).
        release();
        float   * xs = nullptr;
        uint8_t * ys = nullptr;
        if ((err = cudaMalloc(&xs, chunk_x*sizeof(float))) != cudaSuccess) {
            fprintf(stderr, "%s: %s: cudaMalloc(x_dev): %s\n", __func__, name, cudaGetErrorString(err));
            return 0;
        }
        if ((err = cudaMalloc(&ys, chunk_y)) != cudaSuccess) {
            fprintf(stderr, "%s: %s: cudaMalloc(y_dev): %s\n", __func__, name, cudaGetErrorString(err));
            cudaFree(xs);
            return 0;
        }
        for (int64_t base = 0; base < nblocks_total; base += chunk_blocks) {
            const int64_t nblocks = std::min(chunk_blocks, nblocks_total - base);
            err = cudaMemcpy(xs, src + base*qk, nblocks*qk*sizeof(float), cudaMemcpyHostToDevice);
            if (err != cudaSuccess) {
                fprintf(stderr, "%s: %s: cudaMemcpy H2D: %s\n", __func__, name, cudaGetErrorString(err));
                break;
            }
            launch(xs, ys, base, nblocks, nullptr);
            if ((err = cudaGetLastError()) != cudaSuccess) {
                fprintf(stderr, "%s: %s: kernel launch: %s\n", __func__, name, cudaGetErrorString(err));
                break;
            }
            err = cudaMemcpy((char *)dst + base*blk_size, ys, nblocks*blk_size, cudaMemcpyDeviceToHost);
            if (err != cudaSuccess) {
                fprintf(stderr, "%s: %s: cudaMemcpy D2H: %s\n", __func__, name, cudaGetErrorString(err));
                break;
            }
        }
        cudaFree(xs);
        cudaFree(ys);
        if (err != cudaSuccess) {
            return 0;
        }
        return nblocks_total*blk_size;
    }

    for (int64_t base = 0, i = 0; base < nblocks_total; base += chunk_blocks, ++i) {
        const int b = (int)(i & 1);
        const int64_t nblocks = std::min(chunk_blocks, nblocks_total - base);

        // Buffer b must be fully done (H2D + kernel + D2H of iteration i-2)
        // before its staging and device buffers are reused.
        if (i >= 2) {
            err = cudaStreamWaitEvent(st_h2d, ev_done[b], 0);
            if (err != cudaSuccess) {
                fprintf(stderr, "%s: %s: stream wait: %s\n", __func__, name, cudaGetErrorString(err));
                break;
            }
        }
        memcpy(x_pin[b], src + base*qk, nblocks*qk*sizeof(float));
        err = cudaMemcpyAsync(x_dev[b], x_pin[b], nblocks*qk*sizeof(float), cudaMemcpyHostToDevice, st_h2d);
        if (err != cudaSuccess) {
            fprintf(stderr, "%s: %s: cudaMemcpyAsync H2D: %s\n", __func__, name, cudaGetErrorString(err));
            break;
        }
        err = cudaEventRecord(ev_h2d[b], st_h2d);
        if (err != cudaSuccess) {
            fprintf(stderr, "%s: %s: event record: %s\n", __func__, name, cudaGetErrorString(err));
            break;
        }
        err = cudaStreamWaitEvent(st_exe, ev_h2d[b], 0);
        if (err != cudaSuccess) {
            fprintf(stderr, "%s: %s: stream wait: %s\n", __func__, name, cudaGetErrorString(err));
            break;
        }
        launch(x_dev[b], y_dev[b], base, nblocks, st_exe);
        if ((err = cudaGetLastError()) != cudaSuccess) {
            fprintf(stderr, "%s: %s: kernel launch: %s\n", __func__, name, cudaGetErrorString(err));
            break;
        }
        err = cudaMemcpyAsync(y_pin[b], y_dev[b], nblocks*blk_size, cudaMemcpyDeviceToHost, st_exe);
        if (err != cudaSuccess) {
            fprintf(stderr, "%s: %s: cudaMemcpyAsync D2H: %s\n", __func__, name, cudaGetErrorString(err));
            break;
        }
        err = cudaEventRecord(ev_done[b], st_exe);
        if (err != cudaSuccess) {
            fprintf(stderr, "%s: %s: event record: %s\n", __func__, name, cudaGetErrorString(err));
            break;
        }
        err = cudaEventSynchronize(ev_done[b]);
        if (err != cudaSuccess) {
            fprintf(stderr, "%s: %s: event sync: %s\n", __func__, name, cudaGetErrorString(err));
            break;
        }
        memcpy((char *)dst + base*blk_size, y_pin[b], nblocks*blk_size);
    }

    release();
    if (err != cudaSuccess) {
        return 0;
    }

    return nblocks_total*blk_size;
}

using imatrix_kernel_t = void (*)(const float *, const float *, const float *, void *, int64_t, int64_t, int32_t, float);

size_t ggml_cuda_quantize_generic(const float * src, void * dst, int64_t nrows, int64_t n_per_row,
        int64_t qk, size_t blk_size, quantize_kernel_t kernel, const char * name, float fudge, int block_size);
size_t ggml_cuda_quantize_imatrix_generic(const float * src, void * dst, int64_t nrows, int64_t n_per_row,
        const float * imatrix, int64_t qk, size_t blk_size, imatrix_kernel_t kernel, const char * name, float fudge,
        bool per_block_sigma);

size_t ggml_cuda_quantize_qleg(int device, ggml_type type, const float * src, void * dst, int64_t nrows, int64_t n_per_row, int64_t nslice,
        const float * imatrix);
