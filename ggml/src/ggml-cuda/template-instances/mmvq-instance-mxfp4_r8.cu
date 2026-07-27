#include "../iqk_mmvq_templates.cuh"

#define VDR_MXFP4_R8_Q8_1_MMVQ 2

static __device__ __forceinline__ void vec_dot_mxfp4_r8_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs, float * result) {

    constexpr uint32_t uval[2] = { 0x00200000, 0x00400000 };
    constexpr int VDR = VDR_MXFP4_R8_Q8_1_MMVQ;

    const block_mxfp4_r8 * bq = (const block_mxfp4_r8 *)vbq + kbx;
    const int * q8 = (const int *)bq8_1->qs + iqs;
    const float d8 = __low2float(bq8_1->ds);

    for (int sr = 0; sr < 8; ++sr) {
        const uint8_t e = bq->e[sr];
        const int * q4_base = (const int *)(bq->qs + 4*sr);
        int2 sumi = {0, 0};
        for (int l = 0; l < VDR; ++l) {
            const int aux_q4 = q4_base[8*(iqs + l)];
            const int2 v = get_int_from_table_16(aux_q4, kvalues_mxfp4);
            sumi.x = ggml_cuda_dp4a(v.x, q8[l + 0], sumi.x);
            sumi.y = ggml_cuda_dp4a(v.y, q8[l + 4], sumi.y);
        }
        union { float f; uint32_t u; } helper;
        helper.u = e >= 2 ? uint32_t(e - 1) << 23u : uval[e];
        result[sr] = 0.5f * helper.f * d8 * (sumi.x + sumi.y);
    }
}

void mul_mat_vec_mxfp4_r8_q8_1_cuda(const mmvq_args & args, cudaStream_t stream) {
    iqk_mul_mat_vec_q_cuda<GGML_TYPE_MXFP4_R8, VDR_MXFP4_R8_Q8_1_MMVQ, vec_dot_mxfp4_r8_q8_1, 8>(args, stream);
}
