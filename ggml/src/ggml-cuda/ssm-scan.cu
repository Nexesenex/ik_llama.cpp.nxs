#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA) && CUDART_VERSION >= 11070
#define USE_CUB
#endif

#ifdef USE_CUB
#include <cub/cub.cuh>
using namespace cub;
#endif

#include "ssm-scan.cuh"

#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wpass-failed"
#endif
template <size_t splitD, size_t N, size_t L_template>
__global__ void __launch_bounds__(splitD, 1)
    ssm_scan_f32(const float *__restrict__ src0, const float *__restrict__ src1,
                 const float *__restrict__ src2,
                 const float *__restrict__ src3, const float *__restrict__ src4,
                 const float *__restrict__ src5,
                 float *__restrict__ dst,
                 const int src0_nb1, const int src0_nb2,
                 const int src1_nb1, const int src1_nb2, const int src1_nb3,
                 const int src2_nb1, const int src2_nb2, const int src3_nb1,
                 const int src4_nb1, const int src4_nb2, const int src5_nb1, const int src5_nb2,
                 const int D, const int L_param, const int B) {
    const size_t L = L_template == 0 ? L_param : L_template;
    const int bidx = blockIdx.x;
    const int bidy = blockIdx.y;

    const float * s0_block = (const float *) ((const char *) src0 + bidx * src0_nb2 + bidy * splitD * src0_nb1);
    const float * x_block  = (const float *) ((const char *) src1 + (bidx * src1_nb2) + bidy * splitD * sizeof(float));
    const float * dt_block = (const float *) ((const char *) src2 + (bidx * src2_nb2) + bidy * splitD * sizeof(float));
    const float * A_block  = (const float *) ((const char *) src3 + bidy * splitD * src3_nb1);
    const float * B_block  = (const float *) ((const char *) src4 + (bidx * src4_nb2));
    const float * C_block  = (const float *) ((const char *) src5 + (bidx * src5_nb2));
    float *       y_block  = (float *) ((char *) dst + (bidx * src1_nb2) + bidy * splitD * sizeof(float));
    float *       s_block  = (float *) ((char *) dst + src1_nb3 + bidx * src0_nb2 + bidy * splitD * src0_nb1);

    const int stride_x  = src1_nb1 / sizeof(float);
    const int stride_dt = src2_nb1 / sizeof(float);
    const int stride_B  = src4_nb1 / sizeof(float);
    const int stride_C  = src5_nb1 / sizeof(float);
    const int stride_y  = stride_x;

    float regA[N];
    float regs0[N];

    __shared__ float smemB[N];
    __shared__ float smemC[N];

#ifdef USE_CUB
    using BlockLoad = cub::BlockLoad<float, splitD, N, cub::BLOCK_LOAD_WARP_TRANSPOSE>;
    using BlockStore = cub::BlockStore<float, splitD, N, cub::BLOCK_STORE_WARP_TRANSPOSE>;

    union CubTempStorage {
        typename BlockLoad::TempStorage load_temp;
        typename BlockStore::TempStorage store_temp;
    };
    __shared__ CubTempStorage cub_temp_storage;

    BlockLoad(cub_temp_storage.load_temp).Load(A_block, regA);
    BlockLoad(cub_temp_storage.load_temp).Load(s0_block, regs0);
#else
    const int stride_s0 = src0_nb1 / sizeof(float);
    const int stride_A  = src3_nb1 / sizeof(float);
#pragma unroll
    for (size_t n = 0; n < N; ++n) {
        regA[n]   = A_block[threadIdx.x * stride_A + n];
        regs0[n]  = s0_block[threadIdx.x * stride_s0 + n];
    }
#endif

#pragma unroll
    for (size_t i = 0; i < L; i++) {
        if (threadIdx.x < N) {
            smemB[threadIdx.x] = B_block[i * stride_B + threadIdx.x];
            smemC[threadIdx.x] = C_block[i * stride_C + threadIdx.x];
        }
        __syncthreads();

        float dt_soft_plus = dt_block[i * stride_dt + threadIdx.x];
        if (dt_soft_plus <= 20.0f) {
            dt_soft_plus = log1pf(expf(dt_soft_plus));
        }
        float x_dt = x_block[i * stride_x + threadIdx.x] * dt_soft_plus;

        float sumf = 0.0f;
#pragma unroll
        for (size_t n = 0; n < N; n++) {
            float state = regs0[n] * expf(dt_soft_plus * regA[n]) + smemB[n] * x_dt;
            sumf += state * smemC[n];
            regs0[n] = state;
        }
        y_block[i * stride_y + threadIdx.x] = sumf;
    }

#ifdef USE_CUB
    BlockStore(cub_temp_storage.store_temp).Store(s_block, regs0);
#else
    const int stride_s = src0_nb1 / sizeof(float);
#pragma unroll
    for (size_t n = 0; n < N; ++n) {
        s_block[threadIdx.x * stride_s + n] = regs0[n];
    }
#endif
}
#ifdef __clang__
#pragma clang diagnostic pop
#endif

static void ssm_scan_f32_cuda(const float * src0, const float * src1, const float * src2, const float * src3,
                              const float * src4, const float * src5, const int src0_nb1, const int src0_nb2,
                              const int src1_nb1, const int src1_nb2, const int src1_nb3,
                              const int src2_nb1, const int src2_nb2, const int src3_nb1,
                              const int src4_nb1, const int src4_nb2, const int src5_nb1, const int src5_nb2,
                              float * dst, const int N, const int D, const int L, const int B, cudaStream_t stream) {
    const int threads = 128;
    GGML_ASSERT(D % threads == 0);
    const dim3 blocks(B, (D + threads - 1) / threads, 1);
    if (N == 16) {
        switch (L) {
        case 1:
            ssm_scan_f32<threads, 16, 1><<<blocks, threads, 0, stream>>>(
                src0, src1, src2, src3, src4, src5, dst,
                src0_nb1, src0_nb2, src1_nb1, src1_nb2, src1_nb3,
                src2_nb1, src2_nb2, src3_nb1, src4_nb1, src4_nb2, src5_nb1, src5_nb2,
                D, L, B);
            break;
        case 2:
            ssm_scan_f32<threads, 16, 2><<<blocks, threads, 0, stream>>>(
                src0, src1, src2, src3, src4, src5, dst,
                src0_nb1, src0_nb2, src1_nb1, src1_nb2, src1_nb3,
                src2_nb1, src2_nb2, src3_nb1, src4_nb1, src4_nb2, src5_nb1, src5_nb2,
                D, L, B);
            break;
        case 3:
            ssm_scan_f32<threads, 16, 3><<<blocks, threads, 0, stream>>>(
                src0, src1, src2, src3, src4, src5, dst,
                src0_nb1, src0_nb2, src1_nb1, src1_nb2, src1_nb3,
                src2_nb1, src2_nb2, src3_nb1, src4_nb1, src4_nb2, src5_nb1, src5_nb2,
                D, L, B);
            break;
        case 4:
            ssm_scan_f32<threads, 16, 4><<<blocks, threads, 0, stream>>>(
                src0, src1, src2, src3, src4, src5, dst,
                src0_nb1, src0_nb2, src1_nb1, src1_nb2, src1_nb3,
                src2_nb1, src2_nb2, src3_nb1, src4_nb1, src4_nb2, src5_nb1, src5_nb2,
                D, L, B);
            break;
        case 5:
            ssm_scan_f32<threads, 16, 5><<<blocks, threads, 0, stream>>>(
                src0, src1, src2, src3, src4, src5, dst,
                src0_nb1, src0_nb2, src1_nb1, src1_nb2, src1_nb3,
                src2_nb1, src2_nb2, src3_nb1, src4_nb1, src4_nb2, src5_nb1, src5_nb2,
                D, L, B);
            break;
        case 6:
            ssm_scan_f32<threads, 16, 6><<<blocks, threads, 0, stream>>>(
                src0, src1, src2, src3, src4, src5, dst,
                src0_nb1, src0_nb2, src1_nb1, src1_nb2, src1_nb3,
                src2_nb1, src2_nb2, src3_nb1, src4_nb1, src4_nb2, src5_nb1, src5_nb2,
                D, L, B);
            break;
        case 7:
            ssm_scan_f32<threads, 16, 7><<<blocks, threads, 0, stream>>>(
                src0, src1, src2, src3, src4, src5, dst,
                src0_nb1, src0_nb2, src1_nb1, src1_nb2, src1_nb3,
                src2_nb1, src2_nb2, src3_nb1, src4_nb1, src4_nb2, src5_nb1, src5_nb2,
                D, L, B);
            break;
        case 8:
            ssm_scan_f32<threads, 16, 8><<<blocks, threads, 0, stream>>>(
                src0, src1, src2, src3, src4, src5, dst,
                src0_nb1, src0_nb2, src1_nb1, src1_nb2, src1_nb3,
                src2_nb1, src2_nb2, src3_nb1, src4_nb1, src4_nb2, src5_nb1, src5_nb2,
                D, L, B);
            break;
        default:
            ssm_scan_f32<threads, 16, 0><<<blocks, threads, 0, stream>>>(
                src0, src1, src2, src3, src4, src5, dst,
                src0_nb1, src0_nb2, src1_nb1, src1_nb2, src1_nb3,
                src2_nb1, src2_nb2, src3_nb1, src4_nb1, src4_nb2, src5_nb1, src5_nb2,
                D, L, B);
            break;
        }
    } else {
        GGML_ABORT("doesn't support N!=16.");
    }
}

void ggml_cuda_op_ssm_scan(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const struct ggml_tensor * src0 = dst->src[0];
    const struct ggml_tensor * src1 = dst->src[1];
    const struct ggml_tensor * src2 = dst->src[2];
    const struct ggml_tensor * src3 = dst->src[3];
    const struct ggml_tensor * src4 = dst->src[4];
    const struct ggml_tensor * src5 = dst->src[5];

    const int64_t nc  = src0->ne[0];
    const int64_t nr  = src0->ne[1];
    const int64_t n_t = src1->ne[1];
    const int64_t n_s = src0->ne[2];

    GGML_ASSERT(ggml_nelements(src1) + ggml_nelements(src0) == ggml_nelements(dst));
    GGML_ASSERT(src0->nb[0] == sizeof(float));
    GGML_ASSERT(src1->nb[0] == sizeof(float));
    GGML_ASSERT(src2->nb[0] == sizeof(float));
    GGML_ASSERT(src3->nb[0] == sizeof(float));
    GGML_ASSERT(src4->nb[0] == sizeof(float));
    GGML_ASSERT(src5->nb[0] == sizeof(float));
    GGML_ASSERT(src0->nb[1] == src0->ne[0] * sizeof(float));
    GGML_ASSERT(src0->nb[2] == src0->ne[0] * src0->ne[1] * sizeof(float));
    GGML_ASSERT(src1->nb[3] == src1->ne[0] * src1->ne[1] * src1->ne[2] * sizeof(float));

    const float * src0_d = (const float *) src0->data;
    const float * src1_d = (const float *) src1->data;
    const float * src2_d = (const float *) src2->data;
    const float * src3_d = (const float *) src3->data;
    const float * src4_d = (const float *) src4->data;
    const float * src5_d = (const float *) src5->data;
    float *       dst_d  = (float *) dst->data;
    cudaStream_t  stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    ssm_scan_f32_cuda(src0_d, src1_d, src2_d, src3_d, src4_d, src5_d, src0->nb[1], src0->nb[2],
                      src1->nb[1], src1->nb[2], src1->nb[3],
                      src2->nb[1], src2->nb[2], src3->nb[1],
                      src4->nb[1], src4->nb[2], src5->nb[1], src5->nb[2],
                      dst_d, nc, nr, n_t, n_s, stream);
}
