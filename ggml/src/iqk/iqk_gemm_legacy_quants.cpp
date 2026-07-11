#include "iqk_gemm_legacy_quants.h"

#include <type_traits>

#ifdef IQK_IMPLEMENT

#include "ggml-impl.h"
#include "iqk_utils.h"

#define GGML_COMMON_IMPL_C
#include "ggml-common.h"

//
// ============================== Legacy quants
//

#ifdef __x86_64__

namespace {

struct DotHelper {
    const __m256i m1 = _mm256_set1_epi16(1);
#ifdef HAVE_VNNI256
    inline __m256i dot(__m256i x, __m256i y) const {
        return ggml_mm256_dpbusd_epi32(_mm256_setzero_si256(), x, y);
    }
#else
    inline __m256i dot(__m256i x, __m256i y) const {
        return _mm256_madd_epi16(m1, _mm256_maddubs_epi16(x, y));
    }
#endif
};

struct SignedDot {
    DotHelper helper;
    inline __m256i compute(__m256i x, __m256i y) const {
        return helper.dot(_mm256_sign_epi8(x, x), _mm256_sign_epi8(y, x));
    }
};
struct UnsignedDot {
    DotHelper helper;
    inline __m256i compute(__m256i x, __m256i y) const {
        return helper.dot(x, y);
    }
};

template <typename Q8, typename Q8x4, typename Dot, bool can_pack = true> struct Sum4 {
    Dot dot;
    inline __m256i compute(const __m256i * qx, const Q8 * y) const {
        const Q8x4 * y4 = (const Q8x4 *)y;
        const __m256i p0 = dot.compute(qx[0], _mm256_loadu_si256((const __m256i *)y4->qs+0)); // 8x block 0
        const __m256i p1 = dot.compute(qx[1], _mm256_loadu_si256((const __m256i *)y4->qs+1)); // 8x block 1
        const __m256i p2 = dot.compute(qx[2], _mm256_loadu_si256((const __m256i *)y4->qs+2)); // 8x block 2
        const __m256i p3 = dot.compute(qx[3], _mm256_loadu_si256((const __m256i *)y4->qs+3)); // 8x block 3
        if constexpr (can_pack) {
            const __m256i p01 = _mm256_madd_epi16(dot.helper.m1, _mm256_packs_epi32(p0, p1));    // 0,0, 1,1, 0,0, 1,1
            const __m256i p23 = _mm256_madd_epi16(dot.helper.m1, _mm256_packs_epi32(p2, p3));    // 2,2, 3,3, 2,2, 3,3
            return _mm256_madd_epi16(dot.helper.m1, _mm256_packs_epi32(p01, p23)); // 0,1,2,3, 0,1,2,3
        } else {
            // Note to myself: this is much faster than using _mm256_hadd_epi32()
            auto p01 = _mm256_add_epi32(_mm256_unpacklo_epi32(p0, p1), _mm256_unpackhi_epi32(p0, p1)); // 0,1, 0,1, 0,1, 0,1
            auto p23 = _mm256_add_epi32(_mm256_unpacklo_epi32(p2, p3), _mm256_unpackhi_epi32(p2, p3)); // 2,3, 2,3, 2,3, 2,3
            return _mm256_add_epi32(_mm256_unpacklo_epi64(p01, p23), _mm256_unpackhi_epi64(p01, p23)); // 0,1,2,3, 0,1,2,3
        }
    }
    inline __m256i compute(__m256i x, __m256i y) const { return dot.compute(x, y); }
};

template <typename Q8, typename Q8x4> struct Sum4q4 {
    inline __m256i compute(const __m256i * qx, const Q8 * y) const {
        const Q8x4 * y4 = (const Q8x4 *)y;
        auto p0 = _mm256_maddubs_epi16(qx[0], _mm256_loadu_si256((const __m256i *)y4->qs+0)); // 16x block 0
        auto p1 = _mm256_maddubs_epi16(qx[1], _mm256_loadu_si256((const __m256i *)y4->qs+1)); // 16x block 1
        auto p2 = _mm256_maddubs_epi16(qx[2], _mm256_loadu_si256((const __m256i *)y4->qs+2)); // 16x block 2
        auto p3 = _mm256_maddubs_epi16(qx[3], _mm256_loadu_si256((const __m256i *)y4->qs+3)); // 16x block 3
        auto p01 = _mm256_add_epi16(_mm256_unpacklo_epi32(p0, p1), _mm256_unpackhi_epi32(p0, p1)); // 0,0, 1,1, 0,0, 1,1, 0,0, 1,1, 0,0, 1,1
        auto p23 = _mm256_add_epi16(_mm256_unpacklo_epi32(p2, p3), _mm256_unpackhi_epi32(p2, p3)); // 2,2, 3,3, 2,2, 3,3, 2,2, 3,3, 2,2, 3,3
        auto p0123 = _mm256_add_epi16(_mm256_unpacklo_epi64(p01, p23), _mm256_unpackhi_epi64(p01, p23)); // 0,0, 1,1, 2,2, 3,3, 0,0, 1,1, 2,2, 3,3
        return _mm256_madd_epi16(_mm256_set1_epi16(1), p0123);
    }
    inline __m256i compute(__m256i x, __m256i y) const { return _mm256_madd_epi16(_mm256_set1_epi16(1), _mm256_maddubs_epi16(x, y)); }
};

inline __m256 convert_scales(const uint16_t * scales) {
    auto aux_d = _mm_castsi128_ps(_mm_slli_epi32(_mm_cvtepu16_epi32(_mm_loadl_epi64((const __m128i *)scales)), 16));
    auto aux_m = _mm_cvtepi32_ps(_mm_cvtepi16_epi32(_mm_loadl_epi64((const __m128i *)(scales+4))));
    return _mm256_set_m128(_mm_mul_ps(aux_d, aux_m), aux_d);
}

inline __m128 convert_scales_s(const uint16_t * scales) {
    return _mm_castsi128_ps(_mm_slli_epi32(_mm_cvtepu16_epi32(_mm_loadl_epi64((const __m128i *)scales)), 16));
}

struct ScaleHelperQ8_0 {
    inline __m128 prepare4(const block_q8_0 * y) {
        const block_q8_0_x4 * y4 = (const block_q8_0_x4 *)y;
        return _mm_cvtph_ps(_mm_loadl_epi64((const __m128i *)y4->d));
    }
    inline __m128 prepare4(__m128 other_scales, const block_q8_0 * y) {
        return _mm_mul_ps(other_scales, prepare4(y));
    }
    template <typename Q> inline float prepare1(const Q * y) const { return GGML_FP16_TO_FP32(y->d); }
    template <typename Q> inline float prepare1(float d, const Q * y) const { return d*prepare1(y); }
};

struct ScaleHelperQ_0 {
    ggml_half scales8[4];
    template <typename Q>
    inline __m128 prepare4(const Q * y) {
        for (int j = 0; j < 4; ++j) scales8[j] = y[j].d;
        return _mm_cvtph_ps(_mm_loadl_epi64((const __m128i *)scales8));
    }
    template <typename Q>
    inline __m128 prepare4(__m128 other_scales, const Q * y) {
        return _mm_mul_ps(other_scales, prepare4<Q>(y));
    }
    template <typename Q> inline float prepare1(const Q * y) const { return GGML_FP16_TO_FP32(y->d); }
    template <typename Q> inline float prepare1(float d, const Q * y) const { return d*prepare1(y); }
};

struct ScaleHelperQ8_2S {
    template <typename Q>
    inline __m128 prepare4(const Q * y) {
        const block_q8_2_x4 * y4 = (const block_q8_2_x4 *)y;
        return convert_scales_s((const uint16_t *)y4->d);
    }
    template <typename Q>
    inline __m128 prepare4(__m128 other_scales, const Q * y) {
        return _mm_mul_ps(other_scales, prepare4<Q>(y));
    }
    template <typename Q> static inline float prepare1(const Q * y) { return GGML_BF16_TO_FP32(ggml_bf16_t{y->d}); }
    template <typename Q> static inline float prepare1(float d, const Q * y) { return d*prepare1(y); }
};

struct ScaleHelperQ_0_MXFP4 {
    float scales[4];
    template <typename Q>
    inline __m128 prepare4(const Q * y) {
        for (int j = 0; j < 4; ++j) scales[j] = GGML_E8M0_TO_FP32_HALF(y[j].e);
        return _mm_loadu_ps(scales);
    }
    template <typename Q>
    inline __m128 prepare4(__m128 other_scales, const Q * y) {
        return _mm_mul_ps(other_scales, prepare4<Q>(y));
    }
    template <typename Q> inline float prepare1(const Q * y) const { return GGML_E8M0_TO_FP32_HALF(y->e); }
    template <typename Q> inline float prepare1(float d, const Q * y) const { return d*prepare1(y); }
};

template <int min_value>
struct ScaleHelperQ_0_1 {
    ggml_half scales8[4];
    template <typename Q>
    inline __m256 prepare4(const Q * y) {
        for (int j = 0; j < 4; ++j) scales8[j] = y[j].d;
        auto s4 = _mm_cvtph_ps(_mm_loadl_epi64((const __m128i *)scales8));
        return _mm256_set_m128(_mm_mul_ps(s4, min), s4);
    }
    template <typename Q>
    inline __m256 prepare4(__m256 other_scales, const Q * y) {
        return _mm_mul256_ps(other_scales, prepare4<Q>(y));
    }
    template <typename Q> inline std::pair<float, float> prepare1(const Q * y) const {
        float d = GGML_FP16_TO_FP32(y->d);
        return std::make_pair(d, -d*float(min_value));
    }
    std::pair<float, float> inline prepare1(const std::pair<float, float>& dm, const block_q8_1 * y) const {
        return std::make_pair(dm.first*GGML_FP16_TO_FP32(y->d), dm.second*GGML_FP16_TO_FP32(y->s));
    }
    const __m128 min = _mm_set1_ps(float(-min_value));
};

template <int min_value>
struct ScaleHelperQ_0_1_MXFP4 {
    float scales[4];
    template <typename Q>
    inline __m256 prepare4(const Q * y) {
        for (int j = 0; j < 4; ++j) scales[j] = GGML_E8M0_TO_FP32_HALF(y[j].e);
        auto s4 = _mm_loadu_ps(scales);
        return _mm256_set_m128(_mm_mul_ps(s4, min), s4);
    }
    template <typename Q>
    inline __m256 prepare4(__m256 other_scales, const Q * y) {
        return _mm_mul256_ps(other_scales, prepare4<Q>(y));
    }
    template <typename Q> inline std::pair<float, float> prepare1(const Q * y) const {
        float d = GGML_E8M0_TO_FP32_HALF(y->e);
        return std::make_pair(d, -d*float(min_value));
    }
    std::pair<float, float> inline prepare1(const std::pair<float, float>& dm, const block_q8_1 * y) const {
        return std::make_pair(dm.first*GGML_FP16_TO_FP32(y->d), dm.second*GGML_FP16_TO_FP32(y->s));
    }
    const __m128 min = _mm_set1_ps(float(-min_value));
};

struct ScaleHelperQ8_1 {
    template <typename Q>
    inline __m256 prepare4(const Q * y) {
        const block_q8_1_x4 * y4 = (const block_q8_1_x4 *)y;
        return _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)y4->d));
    }
    template <typename Q>
    inline __m256 prepare4(__m256 other_scales, const Q * y) {
        return _mm256_mul_ps(other_scales, prepare4<Q>(y));
    }
    template <typename Q> inline std::pair<float, float> prepare1(const Q * y) const {
        return std::make_pair(GGML_FP16_TO_FP32(y->d), GGML_FP16_TO_FP32(y->m));
    }
    template <typename Q> inline std::pair<float, float> prepare1(const std::pair<float, float>& dm, const Q * y) const {
        return std::make_pair(dm.first*GGML_FP16_TO_FP32(y->d), dm.second*GGML_FP16_TO_FP32(y->m));
    }
    std::pair<float, float> inline prepare1(const std::pair<float, float>& dm, const block_q8_1 * y) const {
        return std::make_pair(dm.first*GGML_FP16_TO_FP32(y->d), dm.second*GGML_FP16_TO_FP32(y->s));
    }
};

struct ScaleHelperQ8_2 {
    template <typename Q>
    inline __m256 prepare4(const Q * y) {
        const block_q8_2_x4 * y4 = (const block_q8_2_x4 *)y;
        return convert_scales((const uint16_t *)y4->d);
    }
    template <typename Q>
    inline __m256 prepare4(__m256 other_scales, const Q * y) {
        return _mm256_mul_ps(other_scales, prepare4<Q>(y));
    }
    template <typename Q> static inline std::pair<float, float> prepare1(const Q * y) {
        float   d = GGML_BF16_TO_FP32(ggml_bf16_t{y->d});
        int16_t m = *(const int16_t *)&y->s;
        return std::make_pair(d, d*m);
    }
    static inline std::pair<float, float> prepare1(const std::pair<float, float>& dm, const block_q8_2 * y) {
        auto d = prepare1(y);
        return std::make_pair(dm.first*d.first, dm.second*d.second);
    }
};

struct ScaleHelperQ_1 {
    uint32_t scales8[4];
    const __m128i shuffle = _mm_set_epi16(0x0f0e, 0x0b0a, 0x0706, 0x0302, 0x0d0c, 0x0908, 0x0504, 0x0100);

    template <typename Q>
    inline __m256 prepare4(const Q * y) {
        for (int j = 0; j < 4; ++j) {
            // it is slightly faster to directly dereference (const uint32 *)&y[j].d, but some compilers
            // complain that this breaks strict-aliasing rules.
            memcpy(scales8 + j, &y[j].d, sizeof(uint32_t));
        }
        return _mm256_cvtph_ps(_mm_shuffle_epi8(_mm_loadu_si128((const __m128i *)scales8), shuffle));
    }

    template <typename Q>
    inline __m256 prepare4(__m256 other_scales, const Q * y) {
        return _mm256_mul_ps(other_scales, prepare4<Q>(y));
    }

    template <typename Q> inline std::pair<float, float> prepare1(const Q * y) const {
        return std::make_pair(GGML_FP16_TO_FP32(y->d), GGML_FP16_TO_FP32(y->m));
    }
    template <typename Q> inline std::pair<float, float> prepare1(const std::pair<float, float>& dm, const Q * y) const {
        return std::make_pair(dm.first*GGML_FP16_TO_FP32(y->d), dm.second*GGML_FP16_TO_FP32(y->m));
    }
    std::pair<float, float> inline prepare1(const std::pair<float, float>& dm, const block_q8_1 * y) const {
        return std::make_pair(dm.first*GGML_FP16_TO_FP32(y->d), dm.second*GGML_FP16_TO_FP32(y->s));
    }
};

struct MinusType0 {
    inline __m256 compute(__m128 d, int) const { return _mm256_set_m128(d, d); }
    inline float compute(float d, int) const { return d; }
    inline float result(__m256 acc, int) const { return hsum_float_8(acc); }
    inline __m256 vresult(__m256 acc, int) const { return acc; }
};

template <int nrc_y> struct MinusType1 {
    __m128 accm[nrc_y];
    MinusType1() { for (int iy = 0; iy < nrc_y; ++iy) accm[iy] = _mm_setzero_ps(); }
    inline __m256 compute(__m256 dm, int iy) {
        const __m128 d = _mm256_castps256_ps128(dm);
        const __m128 m = _mm256_extractf128_ps(dm, 1);
        accm[iy] = _mm_add_ps(accm[iy], m);
        return _mm256_set_m128(d, d);
    }
    inline float compute(const std::pair<float, float>& dm, int iy) {
        accm[iy] = _mm_add_ps(accm[iy], _mm_set1_ps(dm.second*0.25f));
        return dm.first;
    }
    inline float result(__m256 acc, int iy) const {
        const __m128 sum = _mm_add_ps(_mm256_castps256_ps128(acc), _mm256_extractf128_ps(acc, 1));
        return hsum_float_4(_mm_add_ps(sum, accm[iy]));
    }
    inline __m256 vresult(__m256 acc, int iy) const {
        return _mm256_add_ps(acc, _mm256_insertf128_ps(_mm256_setzero_ps(), accm[iy], 0));
    }
};

template <typename Minus, int nrc_y, bool is_multiple_of_4> struct AccumT {
    __m256 acc[nrc_y];
    Minus accm;
    AccumT() {  for (int iy = 0; iy < nrc_y; ++iy) acc[iy] = _mm256_setzero_ps(); }
    template <typename Unpacker, typename Scales, typename Sum, typename Q8>
    inline void compute(int nb, Unpacker& unp, Scales& scales, Sum& sum, const Q8 ** y, const DataInfo& info, int ix) {
        auto qx = unp.quants();
        __m256 dall[nrc_y];
        for (int i = 0; i < nb/4; ++i) {
            auto other_scales = unp.set_block_4(i);
            for (int iy = 0; iy < nrc_y; ++iy) {
                auto s12 = scales.prepare4(other_scales, y[iy] + 4*i);
                dall[iy] = accm.compute(s12, iy);
            }
            for (int iy = 0; iy < nrc_y; ++iy) {
                auto pall = sum.compute(qx, y[iy] + 4*i);
                acc[iy] = _mm256_fmadd_ps(dall[iy], _mm256_cvtepi32_ps(pall), acc[iy]);
            }
        }
        if (!is_multiple_of_4) {
            for (int i = 4*(nb/4); i < nb; ++i) {
                auto other_scales = unp.set_block(i);
                for (int iy = 0; iy < nrc_y; ++iy) {
                    auto s12 = scales.prepare1(other_scales, y[iy] + i);
                    auto d = accm.compute(s12, iy);
                    const __m256i p0 = sum.compute(qx[0], _mm256_loadu_si256((const __m256i *)y[iy][i].qs));
                    acc[iy] = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(p0), acc[iy]);
                }
            }
        }
        for (int iy = 0; iy < nrc_y; ++iy) {
            info.store(ix, iy, accm.result(acc[iy], iy));
        }
    }
    template <typename Unpacker, typename Scales, typename Sum, typename Q8>
    inline void compute(int nb, Unpacker& unp, Scales& scales, Sum& sum, const Q8 ** y, __m256 * result) {
        auto qx = unp.quants();
        __m256 dall[nrc_y];
        for (int i = 0; i < nb/4; ++i) {
            auto other_scales = unp.set_block_4(i);
            for (int iy = 0; iy < nrc_y; ++iy) {
                auto s12 = scales.prepare4(other_scales, y[iy] + 4*i);
                dall[iy] = accm.compute(s12, iy);
            }
            for (int iy = 0; iy < nrc_y; ++iy) {
                auto pall = sum.compute(qx, y[iy] + 4*i);
                acc[iy] = _mm256_fmadd_ps(dall[iy], _mm256_cvtepi32_ps(pall), acc[iy]);
            }
        }
        if (!is_multiple_of_4) {
            for (int i = 4*(nb/4); i < nb; ++i) {
                auto other_scales = unp.set_block(i);
                for (int iy = 0; iy < nrc_y; ++iy) {
                    auto s12 = scales.prepare1(other_scales, y[iy] + i);
                    auto d = accm.compute(s12, iy);
                    const __m256i p0 = sum.compute(qx[0], _mm256_loadu_si256((const __m256i *)y[iy][i].qs));
                    acc[iy] = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(p0), acc[iy]);
                }
            }
        }
        for (int iy = 0; iy < nrc_y; ++iy) {
            result[iy] = accm.vresult(acc[iy], iy);
        }
    }
};

template <int nrc_y, bool is_multiple_of_4>
using AccumType0 = AccumT<MinusType0, nrc_y, is_multiple_of_4>;

template <int nrc_y, bool is_multiple_of_4>
using AccumType1 = AccumT<MinusType1<nrc_y>, nrc_y, is_multiple_of_4>;

using Sum4TypeQ80 = Sum4<block_q8_0, block_q8_0_x4, SignedDot, false>;
using Sum4TypeQ82 = Sum4<block_q8_2, block_q8_2_x4, UnsignedDot, false>;
using Sum4TypeQ82S = Sum4<block_q8_2, block_q8_2_x4, SignedDot, false>;

template <typename Unpacker, typename AccumType, typename Scales, typename Q8, int nrc_y>
void mul_mat_qX_q8_Helper(int nb, const void * vx, size_t bx, const DataInfo& info, const Q8 ** y, int nrc_x) {
    Unpacker unp(vx, bx);
    typename Unpacker::Sum4T sum4;
    Scales scales;
    for (int ix = 0; ix < nrc_x; ++ix) {
        unp.set_row(ix);
        AccumType accum;
        accum.compute(nb, unp, scales, sum4, y, info, ix);
    }
}

template <typename Unpacker, typename AccumType, typename Scales, typename Q8, int nrc_y>
void mul_mat_qX_q8_Helper_x2(int nb, const void * vx, size_t bx, const DataInfo& info, const Q8 ** y, int nrc_x) {
    GGML_ASSERT(nrc_x%2 == 0);
    Unpacker unp(vx, bx);
    typename Unpacker::Sum4T sum4;
    Scales scales;
    for (int ix = 0; ix < nrc_x; ix += 2) {
        unp.set_row(ix);
        AccumType accum;
        accum.compute(nb, unp, scales, sum4, y, info, ix);
    }
}

template <typename Unpacker, int nrc_y, typename Block = block_q8_0>
void mul_mat_qX_0_q8_0_T(int n, const void * vx, size_t bx, const DataInfo& info, int nrc_x) {
    assert(n%Unpacker::block_size() == 0);
    Q8<nrc_y, Block> q8(info);
    int nb = n/Unpacker::block_size();
    if constexpr (std::is_same_v<Block, block_q8_2>) {
        if (nb%4 == 0) {
            mul_mat_qX_q8_Helper<Unpacker, AccumType0<nrc_y, true>, ScaleHelperQ8_2S, Block, nrc_y>(
                    nb, vx, bx, info, q8.y, nrc_x);
        } else {
            mul_mat_qX_q8_Helper<Unpacker, AccumType0<nrc_y, false>, ScaleHelperQ8_2S, Block, nrc_y>(
                    nb, vx, bx, info, q8.y, nrc_x);
        }
    }
    else {
        if (nb%4 == 0) {
            mul_mat_qX_q8_Helper<Unpacker, AccumType0<nrc_y, true>, ScaleHelperQ8_0, Block, nrc_y>(
                    nb, vx, bx, info, q8.y, nrc_x);
        } else {
            mul_mat_qX_q8_Helper<Unpacker, AccumType0<nrc_y, false>, ScaleHelperQ8_0, Block, nrc_y>(
                    nb, vx, bx, info, q8.y, nrc_x);
        }
    }
}

template <typename Unpacker, int nrc_y>
void mul_mat_qX_0_q8_2_T(int n, const void * vx, size_t bx, const DataInfo& info, int nrc_x) {
    assert(n%Unpacker::block_size() == 0);
    Q8<nrc_y, block_q8_2> q8(info);
    int nb = n/Unpacker::block_size();
    if (nb%4 == 0) {
        mul_mat_qX_q8_Helper<Unpacker, AccumType0<nrc_y, true>, ScaleHelperQ8_0, block_q8_0, nrc_y>(
                nb, vx, bx, info, q8.y, nrc_x
        );
    } else {
        mul_mat_qX_q8_Helper<Unpacker, AccumType0<nrc_y, false>, ScaleHelperQ8_0, block_q8_0, nrc_y>(
                nb, vx, bx, info, q8.y, nrc_x
        );
    }
}

template <typename Unpacker, int nrc_y, int nrc_x>
void mul_mat_qX_0_q8_0_Tx(int n, const void * vx, size_t bx, const DataInfo& info, int) {
    static_assert(8%nrc_y == 0);
    Q8<nrc_y, block_q8_2> q8(info);
    int nb = n/Unpacker::block_size();
    Unpacker unp(vx, bx);
    typename Unpacker::Sum4T sum4;
    ScaleHelperQ8_2S scales;
    __m256 result[8];
    auto store = [&info, &result] (int ix0) {
        if constexpr (nrc_y == 1) {
            info.store(ix0, 0, hsum_float_8x8(result));
        }
        else if constexpr (nrc_y == 2) {
            auto value = hsum_float_8x8(result);
            auto value1 = _mm256_extractf128_ps(value, 1);
            info.store(ix0, 0, _mm_shuffle_ps(_mm256_castps256_ps128(value), value1, 0x88));
            info.store(ix0, 1, _mm_shuffle_ps(_mm256_castps256_ps128(value), value1, 0xdd));
        }
        else {
            float val[8];
            _mm256_storeu_ps(val, hsum_float_8x8(result));
            for (int iy = 0; iy < nrc_y; ++iy) for (int ix = 0; ix < 8/nrc_y; ++ix) info.store(ix0+ix, iy, val[nrc_y*ix+iy]);
        }
    };
    if (nb%4 == 0) {
        for (int ix0 = 0; ix0 < nrc_x; ix0 += 8/nrc_y) {
            for (int ix = 0; ix < 8/nrc_y; ++ix) {
                unp.set_row(ix0 + ix);
                AccumType0<nrc_y, true> accum;
                accum.compute(nb, unp, scales, sum4, q8.y, result + nrc_y*ix);
            }
            store(ix0);
        }
    } else {
        for (int ix0 = 0; ix0 < nrc_x; ix0 += 8/nrc_y) {
            for (int ix = 0; ix < 8/nrc_y; ++ix) {
                unp.set_row(ix0 + ix);
                AccumType0<nrc_y, false> accum;
                accum.compute(nb, unp, scales, sum4, q8.y, result + nrc_y*ix);
            }
            store(ix0);
        }
    }
}

template <typename Unpacker, int nrc_y>
void mul_mat_qX_1_q8_1_T(int n, const void * vx, size_t bx, const DataInfo& info, int nrc_x) {
    assert(n%Unpacker::block_size() == 0);
    Q8<nrc_y, block_q8_1> q8(info);
    int nb = n/Unpacker::block_size();
    if (nb%4 == 0) {
        mul_mat_qX_q8_Helper<Unpacker, AccumType1<nrc_y, true>, ScaleHelperQ8_1, block_q8_1, nrc_y>(
                nb, vx, bx, info, q8.y, nrc_x
        );
    } else {
        mul_mat_qX_q8_Helper<Unpacker, AccumType1<nrc_y, false>, ScaleHelperQ8_1, block_q8_1, nrc_y>(
                nb, vx, bx, info, q8.y, nrc_x
        );
    }
}

template <typename Unpacker, int nrc_y>
void mul_mat_qX_1_q8_2_T(int n, const void * vx, size_t bx, const DataInfo& info, int nrc_x) {
    assert(n%Unpacker::block_size() == 0);
    Q8<nrc_y, block_q8_2> q8(info);
    int nb = n/Unpacker::block_size();
    if (nb%4 == 0) {
        mul_mat_qX_q8_Helper<Unpacker, AccumType1<nrc_y, true>, ScaleHelperQ8_2, block_q8_2, nrc_y>(
                nb, vx, bx, info, q8.y, nrc_x
        );
    } else {
        mul_mat_qX_q8_Helper<Unpacker, AccumType1<nrc_y, false>, ScaleHelperQ8_2, block_q8_2, nrc_y>(
                nb, vx, bx, info, q8.y, nrc_x
        );
    }
}

template <typename Unpacker, int nrc_y, int nrc_x>
void mul_mat_qX_0_q8_2_Tx(int n, const void * vx, size_t bx, const DataInfo& info, int) {
    static_assert(8%nrc_y == 0);
    Q8<nrc_y, block_q8_2> q8(info);
    int nb = n/Unpacker::block_size();
    Unpacker unp(vx, bx);
    typename Unpacker::Sum4T sum4;
    ScaleHelperQ8_2 scales;
    __m256 result[8];
    auto store = [&info, &result] (int ix0) {
        if constexpr (nrc_y == 1) {
            info.store(ix0, 0, hsum_float_8x8(result));
        }
        else if constexpr (nrc_y == 2) {
            auto value = hsum_float_8x8(result);
            auto value1 = _mm256_extractf128_ps(value, 1);
            info.store(ix0, 0, _mm_shuffle_ps(_mm256_castps256_ps128(value), value1, 0x88));
            info.store(ix0, 1, _mm_shuffle_ps(_mm256_castps256_ps128(value), value1, 0xdd));
        }
        else {
            float val[8];
            _mm256_storeu_ps(val, hsum_float_8x8(result));
            for (int iy = 0; iy < nrc_y; ++iy) for (int ix = 0; ix < 8/nrc_y; ++ix) info.store(ix0+ix, iy, val[nrc_y*ix+iy]);
        }
    };
    if (nb%4 == 0) {
        for (int ix0 = 0; ix0 < nrc_x; ix0 += 8/nrc_y) {
            for (int ix = 0; ix < 8/nrc_y; ++ix) {
                unp.set_row(ix0 + ix);
                AccumType1<nrc_y, true> accum;
                accum.compute(nb, unp, scales, sum4, q8.y, result + nrc_y*ix);
            }
            store(ix0);
        }
    } else {
        for (int ix0 = 0; ix0 < nrc_x; ix0 += 8/nrc_y) {
            for (int ix = 0; ix < 8/nrc_y; ++ix) {
                unp.set_row(ix0 + ix);
                AccumType1<nrc_y, false> accum;
                accum.compute(nb, unp, scales, sum4, q8.y, result + nrc_y*ix);
            }
            store(ix0);
        }
    }
}

struct Dequantizer4bit {
    const __m256i m4 = _mm256_set1_epi8(0xf);
    inline __m256i dequant(const uint8_t * qs) const {
        const __m128i aux128 = _mm_loadu_si128((const __m128i *)qs);
        return _mm256_and_si256(MM256_SRLI128_M128I(aux128, 4), m4);
    }
};

struct Q8_0_Dequantizer {
    inline __m256i dequant(const block_q8_0 * x) const {
        return _mm256_loadu_si256((const __m256i *)x->qs);
    }
};

struct Q8_0_1_Dequantizer {
    inline __m256i dequant(const block_q8_0 * x) const {
        return _mm256_add_epi8(_mm256_set1_epi8(127), _mm256_loadu_si256((const __m256i *)x->qs));
    }
};

struct Q4_0_Dequantizer {
    Dequantizer4bit b4;
    const __m256i m8 = _mm256_set1_epi8(-8);
    inline __m256i dequant(const block_q4_0 * x) const {
        return _mm256_add_epi8(b4.dequant(x->qs), m8);
    }
};

struct Q4_0_1_Dequantizer {
    Dequantizer4bit b4;
    inline __m256i dequant(const block_q4_0 * x) const {
        return b4.dequant(x->qs);
    }
};

struct IQ4_NL_DequantizerU {
    Dequantizer4bit b4;
    const __m256i values = load_iq4nl_values_256();
    inline __m256i dequant(const block_iq4_nl * x) const {
        return _mm256_shuffle_epi8(values, b4.dequant(x->qs));
    }
};

struct IQ4_NL_DequantizerS {
    Dequantizer4bit b4;
    const __m256i values = load_iq4k_values_256();
    inline __m256i dequant(const block_iq4_nl * x) const {
        return _mm256_shuffle_epi8(values, b4.dequant(x->qs));
    }
};

//=============================
static inline __m128i load_unsigned_mxfp4_values_128() {
    static const uint8_t kvalues_mxfp4_unsigned[16] = {12, 13, 14, 15, 16, 18, 20, 24, 12, 11, 10, 9, 8, 6, 4, 0};
    return _mm_loadu_si128((const __m128i *)kvalues_mxfp4_unsigned);
}

static inline __m256i load_unsigned_mxfp4_values_256() {
    auto val128 = load_unsigned_mxfp4_values_128();
    return MM256_SET1_M128I(val128);
}

#ifdef HAVE_FANCY_SIMD
static inline __m512i load_unsigned_mxfp4_values_512() {
    auto val256 = load_unsigned_mxfp4_values_256();
    return _mm512_inserti32x8(_mm512_castsi256_si512(val256), val256, 1);
}
#endif

static inline __m128i load_mxfp4_values_128() {
    return _mm_loadu_si128((const __m128i *)kvalues_mxfp4);
}

static inline __m256i load_mxfp4_values_256() {
    auto val128 = load_mxfp4_values_128();
    return MM256_SET1_M128I(val128);
}

struct MXFP4_Dequantizer {
    Dequantizer4bit b4;
    const __m256i values = load_unsigned_mxfp4_values_256();
    inline __m256i dequant(const block_mxfp4 * x) const {
        return _mm256_shuffle_epi8(values, b4.dequant(x->qs));
    }
};

struct MXFP40_Dequantizer {
    Dequantizer4bit b4;
    const __m256i values = load_mxfp4_values_256();
    inline __m256i dequant(const block_mxfp4 * x) const {
        return _mm256_shuffle_epi8(values, b4.dequant(x->qs));
    }
};

struct Q4_1_Dequantizer {
    Dequantizer4bit b4;
    inline __m256i dequant(const block_q4_1 * x) const {
        return b4.dequant(x->qs);
    }
};

struct HBitDequantizer {
    const __m256i shuffle = _mm256_set_epi64x(0x0303030303030303, 0x0202020202020202, 0x0101010101010101, 0x0000000000000000);
    const __m256i mask = _mm256_set1_epi64x(0x7fbfdfeff7fbfdfe);
    const __m256i minus1 = _mm256_set1_epi64x(-1);
    inline __m256i to_bytes(const uint8_t * bits) const {
        // Note: Data in all ggml quants is at least 2-byte aligned.
        // => we can cast to uint16_t and use or on two consecutive entries
        // which is faster than memcpy
        const uint16_t * aux16 = (const uint16_t *)bits;
        const uint32_t aux32 = aux16[0] | (aux16[1] << 16);
        //uint32_t aux32; memcpy(&aux32, bits, sizeof(uint32_t));
        __m256i bytes = _mm256_shuffle_epi8(_mm256_set1_epi32(aux32), shuffle);
        bytes = _mm256_or_si256(bytes, mask);
        return _mm256_cmpeq_epi8(bytes, minus1);
    }
};

struct Q5_0_Dequantizer {
    Dequantizer4bit b4;
    HBitDequantizer hbit;
    const __m256i mh = _mm256_set1_epi8((char)0xF0);
    inline __m256i dequant(const block_q5_0 * x) const {
        const __m256i vqh = _mm256_andnot_si256(hbit.to_bytes(x->qh), mh);
        return _mm256_or_si256(b4.dequant(x->qs), vqh);
    }
};

template <typename Q5>
struct Q5_1_Dequantizer {
    Dequantizer4bit b4;
    HBitDequantizer hbit;
    const __m256i mh = _mm256_set1_epi8(0x10);
    inline __m256i dequant(const Q5 * x) const {
        const __m256i vqh = _mm256_and_si256(hbit.to_bytes(x->qh), mh);
        return _mm256_or_si256(b4.dequant(x->qs), vqh);
    }
};
template <typename Q6>
struct Q6_1_Dequantizer {
    Dequantizer4bit b4;
    const __m256i mh = _mm256_set1_epi8(0x30);
    const __m256i shift1 = _mm256_set_epi64x(0, 2, 0, 4);
    const __m256i shift2 = _mm256_set_epi64x(2, 0, 0, 0);
    inline __m256i dequant(const Q6 * x) const {
        uint64_t aux64; std::memcpy(&aux64, x->qh, 8);
        auto h256 = _mm256_sllv_epi64(_mm256_set1_epi64x(aux64), shift1);
        return _mm256_or_si256(b4.dequant(x->qs), _mm256_and_si256(_mm256_srlv_epi64(h256, shift2), mh));
    }
};
struct Q6_0_1_Dequantizer {
    Dequantizer4bit b4;
    const __m256i mh = _mm256_set1_epi8(0x30);
    const __m256i shift1 = _mm256_set_epi64x(0, 2, 0, 4);
    const __m256i shift2 = _mm256_set_epi64x(2, 0, 0, 0);
    inline __m256i dequant(const block_q6_0 * x) const {
        uint64_t aux64; std::memcpy(&aux64, x->qh, 8);
        auto h256 = _mm256_sllv_epi64(_mm256_set1_epi64x(aux64), shift1);
        return _mm256_or_si256(b4.dequant(x->qs), _mm256_and_si256(_mm256_srlv_epi64(h256, shift2), mh));
    }
};
struct Q6_0_Dequantizer {
    Q6_0_1_Dequantizer deq;
    inline __m256i dequant(const block_q6_0 * x) const {
        return _mm256_add_epi8(deq.dequant(x), _mm256_set1_epi8(-32));
    }
};

template <typename Q, typename Scales, typename Dequantizer>
struct Q_Unpacker {
    Q_Unpacker(const void * vx, size_t bx) : cx_0((const char *)vx), x((const Q*)cx_0), bx(bx) {}

    const char * cx_0;
    const Q    * x;
    size_t       bx;

    Scales scales;
    Dequantizer deq;

    __m256i qx[4];

    inline const __m256i* quants() const { return qx; }

    inline void set_row(int ix) { x = (const Q*)(cx_0 + ix*bx); }

    inline auto set_block_4(int i) {
        for (int j = 0; j < 4; ++j) {
            qx[j] = deq.dequant(x + 4*i + j);
        }
        return scales.prepare4(x + 4*i);
    }
    inline auto set_block(int i) {
        qx[0] = deq.dequant(x + i);
        return scales.prepare1(x + i);
    }
};

struct Q8_0_Unpacker final : public Q_Unpacker<block_q8_0, ScaleHelperQ_0, Q8_0_Dequantizer> {
    Q8_0_Unpacker(const void * vx, size_t bx) : Q_Unpacker(vx, bx) {}
    using Sum4T = Sum4TypeQ82S;
    inline static int block_size() { return QK8_0; }
};
struct Q8_0_1_Unpacker final : public Q_Unpacker<block_q8_0, ScaleHelperQ_0_1<127>, Q8_0_1_Dequantizer> {
    Q8_0_1_Unpacker(const void * vx, size_t bx) : Q_Unpacker(vx, bx) {}
    using Sum4T = Sum4TypeQ82;
    inline static int block_size() { return QK8_0; }
};
struct Q8_0_2_Unpacker final : public Q_Unpacker<block_q8_0, ScaleHelperQ_0, Q8_0_Dequantizer> {
    Q8_0_2_Unpacker(const void * vx, size_t bx) : Q_Unpacker(vx, bx) {}
    using Sum4T = Sum4TypeQ82;
    inline static int block_size() { return QK8_0; }
};
struct Q4_0_Unpacker final : public Q_Unpacker<block_q4_0, ScaleHelperQ_0, Q4_0_Dequantizer> {
    Q4_0_Unpacker(const void * vx, size_t bx) : Q_Unpacker(vx, bx) {}
    using Sum4T = Sum4TypeQ80;
    inline static int block_size() { return QK4_0; }
};
struct Q4_0_1_Unpacker final : public Q_Unpacker<block_q4_0, ScaleHelperQ_0_1<8>, Q4_0_1_Dequantizer> {
    Q4_0_1_Unpacker(const void * vx, size_t bx) : Q_Unpacker(vx, bx) {}
    //using Sum4T = Sum4TypeQ82;
    using Sum4T = Sum4q4<block_q8_2, block_q8_2_x4>;
    inline static int block_size() { return QK4_0; }
};
struct MXFP4_Unpacker final : public Q_Unpacker<block_mxfp4, ScaleHelperQ_0_1_MXFP4<12>, MXFP4_Dequantizer> {
    MXFP4_Unpacker(const void * vx, size_t bx) : Q_Unpacker(vx, bx) {}
    using Sum4T = Sum4TypeQ82;
    inline static int block_size() { return QK4_NL; }
};
struct IQ4_NL_UnpackerU final : public Q_Unpacker<block_iq4_nl, ScaleHelperQ_0_1<128>, IQ4_NL_DequantizerU> {
    IQ4_NL_UnpackerU(const void * vx, size_t bx) : Q_Unpacker(vx, bx) {}
    using Sum4T = Sum4TypeQ82;
    inline static int block_size() { return QK4_NL; }
};
struct IQ4_NL_UnpackerS final : public Q_Unpacker<block_iq4_nl, ScaleHelperQ_0, IQ4_NL_DequantizerS> {
    IQ4_NL_UnpackerS(const void * vx, size_t bx) : Q_Unpacker(vx, bx) {}
    using Sum4T = Sum4TypeQ82S;
    inline static int block_size() { return QK4_NL; }
};
struct Q5_0_Unpacker final : public Q_Unpacker<block_q5_0, ScaleHelperQ_0, Q5_0_Dequantizer> {
    Q5_0_Unpacker(const void * vx, size_t bx) : Q_Unpacker(vx, bx) {}
    using Sum4T = Sum4TypeQ80;
    inline static int block_size() { return QK5_0; }
};
struct Q5_0_1_Unpacker final : public Q_Unpacker<block_q5_0, ScaleHelperQ_0_1<16>, Q5_1_Dequantizer<block_q5_0>> {
    Q5_0_1_Unpacker(const void * vx, size_t bx) : Q_Unpacker(vx, bx) {}
    using Sum4T = Sum4TypeQ82;
    inline static int block_size() { return QK5_0; }
};
struct Q4_1_Unpacker final : public Q_Unpacker<block_q4_1, ScaleHelperQ_1, Q4_1_Dequantizer> {
    Q4_1_Unpacker(const void * vx, size_t bx) : Q_Unpacker(vx, bx) {}
    using Sum4T = Sum4TypeQ82;
    inline static int block_size() { return QK4_1; }
};
struct Q5_1_Unpacker final : public Q_Unpacker<block_q5_1, ScaleHelperQ_1, Q5_1_Dequantizer<block_q5_1>> {
    Q5_1_Unpacker(const void * vx, size_t bx) : Q_Unpacker(vx, bx) {}
    using Sum4T = Sum4TypeQ82;
    inline static int block_size() { return QK5_1; }
};
struct Q6_0_1_Unpacker final : public Q_Unpacker<block_q6_0, ScaleHelperQ_0_1<32>, Q6_0_1_Dequantizer> {
    Q6_0_1_Unpacker(const void * vx, size_t bx) : Q_Unpacker(vx, bx) {}
    using Sum4T = Sum4TypeQ82;
    inline static int block_size() { return QK6_0; }
};
struct Q6_1_Unpacker final : public Q_Unpacker<block_q6_1, ScaleHelperQ_1, Q6_1_Dequantizer<block_q6_1>> {
    Q6_1_Unpacker(const void * vx, size_t bx) : Q_Unpacker(vx, bx) {}
    using Sum4T = Sum4TypeQ82;
    inline static int block_size() { return QK6_1; }
};

#ifdef HAVE_FANCY_SIMD
template <int nrc_y>
static void mul_mat_iq4_nl_r4_q8_2(int n, const void * vx, size_t bx, const DataInfo& info, int nrc_x) {
    GGML_ASSERT(nrc_x%8 == 0);
    Q8<nrc_y, block_q8_2_x4> q8(info);
    auto m4 = _mm512_set1_epi8(0xf);
    auto values = load_iq4nl_values_512();
    int nb = n / QK4_NL;
    __m512  acc[2*nrc_y] = {};
    __m512i qx[4];
    float d8[8*nrc_y];
    auto prepare = [&qx, &m4, &values] (const block_iq4_nl_r4& iq4l, const block_iq4_nl_r4& iq4h) {
        auto scales128 = _mm_cvtph_ps(_mm_loadl_epi64((const __m128i *)iq4l.d));
        auto scales1 = _mm256_set_m128(scales128, scales128);
        scales128 = _mm_cvtph_ps(_mm_loadl_epi64((const __m128i *)iq4h.d));
        auto scales2 = _mm256_set_m128(scales128, scales128);
        auto scales = _mm512_insertf32x8(_mm512_castps256_ps512(scales1), scales2, 1);
        auto bits1 = _mm512_inserti32x8(_mm512_castsi256_si512(_mm256_loadu_si256((const __m256i *)iq4l.qs+0)),
                                                               _mm256_loadu_si256((const __m256i *)iq4h.qs+0), 1);
        auto bits2 = _mm512_inserti32x8(_mm512_castsi256_si512(_mm256_loadu_si256((const __m256i *)iq4l.qs+1)),
                                                               _mm256_loadu_si256((const __m256i *)iq4h.qs+1), 1);
        qx[0] = _mm512_shuffle_epi8(values, _mm512_and_si512(bits1, m4));
        qx[1] = _mm512_shuffle_epi8(values, _mm512_and_si512(bits2, m4));
        qx[2] = _mm512_shuffle_epi8(values, _mm512_and_si512(_mm512_srli_epi16(bits1, 4), m4));
        qx[3] = _mm512_shuffle_epi8(values, _mm512_and_si512(_mm512_srli_epi16(bits2, 4), m4));
        return scales;
    };
    auto dot = [&qx] (__m256i y8) {
        auto y = _mm512_inserti32x8(_mm512_castsi256_si512(y8), y8, 1);
        auto sumi = _mm512_setzero_si512();
        sumi = _mm512_dpbusd_epi32(sumi, qx[0], _mm512_shuffle_epi32(y, _MM_PERM_ENUM(0x00)));
        sumi = _mm512_dpbusd_epi32(sumi, qx[1], _mm512_shuffle_epi32(y, _MM_PERM_ENUM(0x55)));
        sumi = _mm512_dpbusd_epi32(sumi, qx[2], _mm512_shuffle_epi32(y, _MM_PERM_ENUM(0xaa)));
        sumi = _mm512_dpbusd_epi32(sumi, qx[3], _mm512_shuffle_epi32(y, _MM_PERM_ENUM(0xff)));
        return sumi;
    };
    for (int ix = 0; ix < nrc_x; ix += 8) {
        const block_iq4_nl_r4 * iq4l = (const block_iq4_nl_r4 *)((const char *)vx + (ix+0)*bx);
        const block_iq4_nl_r4 * iq4h = (const block_iq4_nl_r4 *)((const char *)vx + (ix+4)*bx);
        for (int ib4 = 0; ib4 < nb/4; ++ib4) {
            for (int iy = 0; iy < nrc_y; ++iy) {
                _mm256_storeu_ps(d8+8*iy, convert_scales((const uint16_t *)q8.y[iy][ib4].d));
            }
            for (int k = 0; k < 4; ++k) {
                auto scales = prepare(iq4l[4*ib4+k], iq4h[4*ib4+k]);
                for (int iy = 0; iy < nrc_y; ++iy) {
                    auto sumi = dot(_mm256_loadu_si256((const __m256i*)q8.y[iy][ib4].qs+k));
                    auto dy = _mm512_set1_ps(d8[8*iy+k]);
                    acc[2*iy+0] = _mm512_fmadd_ps(_mm512_mul_ps(scales, dy), _mm512_cvtepi32_ps(sumi), acc[2*iy+0]);
                    acc[2*iy+1] = _mm512_fmadd_ps(scales, _mm512_set1_ps(d8[8*iy+k+4]), acc[2*iy+1]);
                }
            }
        }
        for (int ib = 4*(nb/4); ib < nb; ++ib) {
            auto scales = prepare(iq4l[ib], iq4h[ib]);
            for (int iy = 0; iy < nrc_y; ++iy) {
                auto qy = (const block_q8_1 *)q8.y[iy];
                auto sumi = dot(_mm256_loadu_si256((const __m256i*)qy[ib].qs));
                ggml_bf16_t d, s; d.bits = qy[ib].d; s.bits = qy[ib].s;
                auto dy = _mm512_set1_ps(GGML_BF16_TO_FP32(d));
                acc[2*iy+0] = _mm512_fmadd_ps(_mm512_mul_ps(scales, dy), _mm512_cvtepi32_ps(sumi), acc[2*iy+0]);
                acc[2*iy+1] = _mm512_fmadd_ps(scales, _mm512_set1_ps(GGML_BF16_TO_FP32(s)), acc[2*iy+1]);
            }
        }
        for (int iy = 0; iy < nrc_y; ++iy) {
            auto sum512 = _mm512_fmadd_ps(_mm512_set1_ps(-64.f), acc[2*iy+1], acc[2*iy+0]);
            acc[2*iy+0] = acc[2*iy+1] = _mm512_setzero_ps();
            auto sum1 = _mm_add_ps(_mm512_extractf32x4_ps(sum512, 0), _mm512_extractf32x4_ps(sum512, 1));
            auto sum2 = _mm_add_ps(_mm512_extractf32x4_ps(sum512, 2), _mm512_extractf32x4_ps(sum512, 3));
            info.store(ix+0, iy, sum1);
            info.store(ix+4, iy, sum2);
        }
    }
}
#else
template <int nrc_y>
static void mul_mat_iq4_nl_r4_q8_2(int n, const void * vx, size_t bx, const DataInfo& info, int nrc_x) {
    GGML_ASSERT(nrc_x%4 == 0);
    Q8<nrc_y, block_q8_2_x4> q8(info);
    auto m4 = _mm256_set1_epi8(0xf);
#ifndef HAVE_VNNI256
    auto m1 = _mm256_set1_epi16(1);
#endif
    auto values128 = _mm_loadu_si128((const __m128i *)iq4k_values);
    auto values = MM256_SET1_M128I(values128);
    int nb = n / QK4_NL;
    __m256 acc[nrc_y] = {};
    __m256i qs[4];
    float d8[4*nrc_y];
    auto prepare = [&qs, &values, &m4] (const block_iq4_nl_r4& iq4) {
        auto scales128 = _mm_cvtph_ps(_mm_loadl_epi64((const __m128i *)iq4.d));
        auto scales = _mm256_set_m128(scales128, scales128);
        auto bits1 = _mm256_loadu_si256((const __m256i *)iq4.qs+0);
        auto bits2 = _mm256_loadu_si256((const __m256i *)iq4.qs+1);
        qs[0] = _mm256_shuffle_epi8(values, _mm256_and_si256(bits1, m4));
        qs[1] = _mm256_shuffle_epi8(values, _mm256_and_si256(bits2, m4));
        qs[2] = _mm256_shuffle_epi8(values, _mm256_and_si256(_mm256_srli_epi16(bits1, 4), m4));
        qs[3] = _mm256_shuffle_epi8(values, _mm256_and_si256(_mm256_srli_epi16(bits2, 4), m4));
        return scales;
    };
#ifdef HAVE_VNNI256
    auto dot = [&qs] (__m256i y) {
        auto sumi = _mm256_setzero_si256();
        sumi = ggml_mm256_dpbusd_epi32(sumi, _mm256_sign_epi8(qs[0], qs[0]), _mm256_sign_epi8(_mm256_shuffle_epi32(y, 0x00), qs[0]));
        sumi = ggml_mm256_dpbusd_epi32(sumi, _mm256_sign_epi8(qs[1], qs[1]), _mm256_sign_epi8(_mm256_shuffle_epi32(y, 0x55), qs[1]));
        sumi = ggml_mm256_dpbusd_epi32(sumi, _mm256_sign_epi8(qs[2], qs[2]), _mm256_sign_epi8(_mm256_shuffle_epi32(y, 0xaa), qs[2]));
        sumi = ggml_mm256_dpbusd_epi32(sumi, _mm256_sign_epi8(qs[3], qs[3]), _mm256_sign_epi8(_mm256_shuffle_epi32(y, 0xff), qs[3]));
        return sumi;
    };
#else
    auto dot = [&qs, &m1] (__m256i y) {
        auto u1 = _mm256_sign_epi8(qs[0], qs[0]);
        auto u2 = _mm256_sign_epi8(qs[1], qs[1]);
        auto sumi1 = _mm256_add_epi32(
                _mm256_madd_epi16(m1, _mm256_maddubs_epi16(u1, _mm256_sign_epi8(_mm256_shuffle_epi32(y, 0x00), qs[0]))),
                _mm256_madd_epi16(m1, _mm256_maddubs_epi16(u2, _mm256_sign_epi8(_mm256_shuffle_epi32(y, 0x55), qs[1]))));
        u1 = _mm256_sign_epi8(qs[2], qs[2]);
        u2 = _mm256_sign_epi8(qs[3], qs[3]);
        auto sumi2 = _mm256_add_epi32(
                _mm256_madd_epi16(m1, _mm256_maddubs_epi16(u1, _mm256_sign_epi8(_mm256_shuffle_epi32(y, 0xaa), qs[2]))),
                _mm256_madd_epi16(m1, _mm256_maddubs_epi16(u2, _mm256_sign_epi8(_mm256_shuffle_epi32(y, 0xff), qs[3]))));
        return _mm256_add_epi32(sumi1, sumi2);
    };
#endif
    for (int ix = 0; ix < nrc_x; ix += 4) {
        const block_iq4_nl_r4 * iq4 = (const block_iq4_nl_r4 *)((const char *)vx + ix*bx);
        for (int ib4 = 0; ib4 < nb/4; ++ib4) {
            for (int iy = 0; iy < nrc_y; ++iy) {
                auto aux = _mm_slli_epi32(_mm_cvtepu16_epi32(_mm_loadl_epi64((const __m128i *)q8.y[iy][ib4].d)), 16);
                _mm_storeu_ps(d8+4*iy, _mm_castsi128_ps(aux));
            }
            for (int k = 0; k < 4; ++k) {
                auto scales = prepare(iq4[4*ib4+k]);
                for (int iy = 0; iy < nrc_y; ++iy) {
                    auto sumi = dot(_mm256_loadu_si256((const __m256i*)q8.y[iy][ib4].qs+k));
                    auto d4d8 = _mm256_mul_ps(scales, _mm256_set1_ps(d8[4*iy+k]));
                    acc[iy] = _mm256_fmadd_ps(d4d8, _mm256_cvtepi32_ps(sumi), acc[iy]);
                }
            }
        }
        for (int ib = 4*(nb/4); ib < nb; ++ib) {
            auto scales = prepare(iq4[ib]);
            for (int iy = 0; iy < nrc_y; ++iy) {
                auto qy = (const block_q8_1 *)q8.y[iy];
                auto sumi = dot(_mm256_loadu_si256((const __m256i*)qy[ib].qs));
                ggml_bf16_t d{qy[ib].d};
                auto d4d8 = _mm256_mul_ps(scales, _mm256_set1_ps(GGML_BF16_TO_FP32(d)));
                acc[iy] = _mm256_fmadd_ps(d4d8, _mm256_cvtepi32_ps(sumi), acc[iy]);
            }
        }
        for (int iy = 0; iy < nrc_y; ++iy) {
            auto sum = _mm_add_ps(_mm256_castps256_ps128(acc[iy]), _mm256_extractf128_ps(acc[iy], 1));
            info.store(ix, iy, sum);
            acc[iy] = _mm256_setzero_ps();
        }
    }
}
#endif

inline void prepare_q4_0_quants_avx2(const uint8_t * qs, __m256i * v, const __m256i& m4) {
    auto bits1 = _mm256_loadu_si256((const __m256i *)qs+0);
    auto bits2 = _mm256_loadu_si256((const __m256i *)qs+1);
    auto bits3 = _mm256_loadu_si256((const __m256i *)qs+2);
    auto bits4 = _mm256_loadu_si256((const __m256i *)qs+3);
    v[0] = _mm256_and_si256(bits1, m4);
    v[1] = _mm256_and_si256(bits2, m4);
    v[2] = _mm256_and_si256(bits3, m4);
    v[3] = _mm256_and_si256(bits4, m4);
    v[4] = _mm256_and_si256(_mm256_srli_epi16(bits1, 4), m4);
    v[5] = _mm256_and_si256(_mm256_srli_epi16(bits2, 4), m4);
    v[6] = _mm256_and_si256(_mm256_srli_epi16(bits3, 4), m4);
    v[7] = _mm256_and_si256(_mm256_srli_epi16(bits4, 4), m4);
}

inline __m256i accum_q4_0_quants(const __m256i * v, const int8_t * qs) {
    auto y4l = _mm_loadu_si128((const __m128i*)qs+0);
    auto y4h = _mm_loadu_si128((const __m128i*)qs+1);
    auto yl  = MM256_SET1_M128I(y4l);
    auto yh  = MM256_SET1_M128I(y4h);
#ifdef HAVE_VNNI256
    auto sumi = _mm256_setzero_si256();
    sumi = ggml_mm256_dpbusd_epi32(sumi, v[0], _mm256_shuffle_epi32(yl, 0x00));
    sumi = ggml_mm256_dpbusd_epi32(sumi, v[1], _mm256_shuffle_epi32(yl, 0x55));
    sumi = ggml_mm256_dpbusd_epi32(sumi, v[2], _mm256_shuffle_epi32(yl, 0xaa));
    sumi = ggml_mm256_dpbusd_epi32(sumi, v[3], _mm256_shuffle_epi32(yl, 0xff));
    sumi = ggml_mm256_dpbusd_epi32(sumi, v[4], _mm256_shuffle_epi32(yh, 0x00));
    sumi = ggml_mm256_dpbusd_epi32(sumi, v[5], _mm256_shuffle_epi32(yh, 0x55));
    sumi = ggml_mm256_dpbusd_epi32(sumi, v[6], _mm256_shuffle_epi32(yh, 0xaa));
    sumi = ggml_mm256_dpbusd_epi32(sumi, v[7], _mm256_shuffle_epi32(yh, 0xff));
#else
    auto sumi1 = _mm256_add_epi16(_mm256_maddubs_epi16(v[0], _mm256_shuffle_epi32(yl, 0x00)),
                                  _mm256_maddubs_epi16(v[1], _mm256_shuffle_epi32(yl, 0x55)));
    auto sumi2 = _mm256_add_epi16(_mm256_maddubs_epi16(v[2], _mm256_shuffle_epi32(yl, 0xaa)),
                                  _mm256_maddubs_epi16(v[3], _mm256_shuffle_epi32(yl, 0xff)));
    auto sumi3 = _mm256_add_epi16(_mm256_maddubs_epi16(v[4], _mm256_shuffle_epi32(yh, 0x00)),
                                  _mm256_maddubs_epi16(v[5], _mm256_shuffle_epi32(yh, 0x55)));
    auto sumi4 = _mm256_add_epi16(_mm256_maddubs_epi16(v[6], _mm256_shuffle_epi32(yh, 0xaa)),
                                  _mm256_maddubs_epi16(v[7], _mm256_shuffle_epi32(yh, 0xff)));
    auto sumi = _mm256_madd_epi16(_mm256_set1_epi16(1), _mm256_add_epi16(_mm256_add_epi16(sumi1, sumi2), _mm256_add_epi16(sumi3, sumi4)));
#endif
    return sumi;
}

template <int nrc_y>
static void mul_mat_q4_0_r8_q8_2_avx2(int n, const void * vx, size_t bx, const DataInfo& info, int nrc_x) {
    GGML_ASSERT(nrc_x%8 == 0);
    Q8<nrc_y, block_q8_2_x4> q8(info);
    auto m4 = _mm256_set1_epi8(0xf);
    int nb = n / QK4_NL;
    __m256i v[8];
    if constexpr (nrc_y == 1) {
        union { __m256 vec; float val[8]; } helper;
        for (int ix = 0; ix < nrc_x; ix += 8) {
            const block_iq4_nl_r8 * iq4 = (const block_iq4_nl_r8 *)((const char *)vx + ix*bx);
            auto acc1 = _mm256_setzero_ps();
            auto acc2 = _mm256_setzero_ps();
            for (int ib4 = 0; ib4 < nb/4; ++ib4) {
                helper.vec = convert_scales((const uint16_t *)q8.y[0][ib4].d);
                for (int k = 0; k < 4; ++k) {
                    auto scales = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)iq4[4*ib4+k].d));
                    prepare_q4_0_quants_avx2(iq4[4*ib4+k].qs, v, m4);
                    auto sumi = accum_q4_0_quants(v, q8.y[0][ib4].qs+32*k);
                    auto d4d8 = _mm256_mul_ps(scales, _mm256_set1_ps(helper.val[k]));
                    acc1 = _mm256_fmadd_ps(d4d8, _mm256_cvtepi32_ps(sumi), acc1);
                    acc2 = _mm256_fmadd_ps(scales, _mm256_set1_ps(helper.val[k+4]), acc2);
                }
            }
            for (int ib = 4*(nb/4); ib < nb; ++ib) {
                auto qy = (const block_q8_2 *)q8.y[0];
                auto scales = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)iq4[ib].d));
                prepare_q4_0_quants_avx2(iq4[ib].qs, v, m4);
                auto sumi = accum_q4_0_quants(v, qy[ib].qs);
                auto [d8, m8] = ScaleHelperQ8_2::prepare1(qy + ib);
                auto d4d8 = _mm256_mul_ps(scales, _mm256_set1_ps(d8));
                acc1 = _mm256_fmadd_ps(d4d8, _mm256_cvtepi32_ps(sumi), acc1);
                acc2 = _mm256_fmadd_ps(scales, _mm256_set1_ps(m8), acc2);
            }
            acc1 = _mm256_fmadd_ps(acc2, _mm256_set1_ps(-8.f), acc1);
            info.store(ix, 0, acc1);
        }
    }
    else {
    __m256 acc[nrc_y] = {};
    float d8[8*nrc_y];
    for (int ix = 0; ix < nrc_x; ix += 8) {
        const block_iq4_nl_r8 * iq4 = (const block_iq4_nl_r8 *)((const char *)vx + ix*bx);
        for (int ib4 = 0; ib4 < nb/4; ++ib4) {
            {
                __m256 d4[4];
                for (int k = 0; k < 4; ++k) {
                    d4[k] = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)iq4[4*ib4+k].d));
                }
                for (int iy = 0; iy < nrc_y; ++iy) {
                    auto scales = convert_scales((const uint16_t *)q8.y[iy][ib4].d);
                    _mm256_storeu_ps(d8 + 8*iy, scales);
                    auto m4 = _mm256_extractf128_ps(scales, 1);
                    auto m8 = _mm256_set_m128(m4, m4);
                    auto sumf = _mm256_mul_ps(d4[0], _mm256_shuffle_ps(m8, m8, 0x00));
                    sumf = _mm256_fmadd_ps(d4[1], _mm256_shuffle_ps(m8, m8, 0x55), sumf);
                    sumf = _mm256_fmadd_ps(d4[2], _mm256_shuffle_ps(m8, m8, 0xaa), sumf);
                    sumf = _mm256_fmadd_ps(d4[3], _mm256_shuffle_ps(m8, m8, 0xff), sumf);
                    acc[iy] = _mm256_fmadd_ps(sumf, _mm256_set1_ps(-8.f), acc[iy]);
                }
            }
            for (int k = 0; k < 4; ++k) {
                auto scales = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)iq4[4*ib4+k].d));
                prepare_q4_0_quants_avx2(iq4[4*ib4+k].qs, v, m4);
                for (int iy = 0; iy < nrc_y; ++iy) {
                    auto sumi = accum_q4_0_quants(v, q8.y[iy][ib4].qs+32*k);
                    auto d4d8 = _mm256_mul_ps(scales, _mm256_set1_ps(d8[8*iy+k]));
                    acc[iy] = _mm256_fmadd_ps(d4d8, _mm256_cvtepi32_ps(sumi), acc[iy]);
                }
            }
        }
        for (int ib = 4*(nb/4); ib < nb; ++ib) {
            auto scales = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)iq4[ib].d));
            auto scales_m = _mm256_mul_ps(scales, _mm256_set1_ps(-8.f));
            prepare_q4_0_quants_avx2(iq4[ib].qs, v, m4);
            for (int iy = 0; iy < nrc_y; ++iy) {
                auto qy = (const block_q8_2 *)q8.y[iy];
                auto sumi = accum_q4_0_quants(v, qy[ib].qs);
                auto [d8, m8] = ScaleHelperQ8_2::prepare1(qy + ib);
                auto d4d8 = _mm256_mul_ps(scales, _mm256_set1_ps(d8));
                acc[iy] = _mm256_fmadd_ps(d4d8, _mm256_cvtepi32_ps(sumi), acc[iy]);
                acc[iy] = _mm256_fmadd_ps(scales_m, _mm256_set1_ps(m8), acc[iy]);
            }
        }
        for (int iy = 0; iy < nrc_y; ++iy) {
            info.store(ix, iy, acc[iy]);
            acc[iy] = _mm256_setzero_ps();
        }
    }
    }
}

#ifdef HAVE_FANCY_SIMD
template <int nrc_y>
static void mul_mat_q4_0_r8_q8_2(int n, const void * vx, size_t bx, const DataInfo& info, int nrc_x) {
    if constexpr (nrc_y == 1) {
        mul_mat_q4_0_r8_q8_2_avx2<1>(n, vx, bx, info, nrc_x);
        return;
    }
    GGML_ASSERT(nrc_x%16 == 0);
    Q8<nrc_y, block_q8_2_x4> q8(info);
    auto m4 = _mm512_set1_epi8(0xf);
    int nb = n / QK4_NL;
    __m512  acc[2*nrc_y] = {};
    __m512i qx[8];
    auto prepare = [&qx, &m4] (const block_iq4_nl_r8& iq4l, const block_iq4_nl_r8& iq4h) {
        auto scales1 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)iq4l.d));
        auto scales2 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)iq4h.d));
        auto scales = _mm512_insertf32x8(_mm512_castps256_ps512(scales1), scales2, 1);
        for (int j = 0; j < 4; ++j) {
            auto bits = _mm512_inserti32x8(_mm512_castsi256_si512(_mm256_loadu_si256((const __m256i *)iq4l.qs+j)),
                    _mm256_loadu_si256((const __m256i *)iq4h.qs+j), 1);
            qx[j+0] = _mm512_and_si512(bits, m4);
            qx[j+4] = _mm512_and_si512(_mm512_srli_epi16(bits, 4), m4);
        }
        return scales;
    };
    auto dot = [&qx] (const int8_t * qy) {
        auto y4l = _mm_loadu_si128((const __m128i*)qy+0);
        auto y4h = _mm_loadu_si128((const __m128i*)qy+1);
        auto y8l = MM256_SET1_M128I(y4l);
        auto y8h = MM256_SET1_M128I(y4h);
        auto yl = _mm512_inserti32x8(_mm512_castsi256_si512(y8l), y8l, 1);
        auto yh = _mm512_inserti32x8(_mm512_castsi256_si512(y8h), y8h, 1);
        auto sumi = _mm512_setzero_si512();
        sumi = _mm512_dpbusd_epi32(sumi, qx[0], _mm512_shuffle_epi32(yl, _MM_PERM_ENUM(0x00)));
        sumi = _mm512_dpbusd_epi32(sumi, qx[1], _mm512_shuffle_epi32(yl, _MM_PERM_ENUM(0x55)));
        sumi = _mm512_dpbusd_epi32(sumi, qx[2], _mm512_shuffle_epi32(yl, _MM_PERM_ENUM(0xaa)));
        sumi = _mm512_dpbusd_epi32(sumi, qx[3], _mm512_shuffle_epi32(yl, _MM_PERM_ENUM(0xff)));
        sumi = _mm512_dpbusd_epi32(sumi, qx[4], _mm512_shuffle_epi32(yh, _MM_PERM_ENUM(0x00)));
        sumi = _mm512_dpbusd_epi32(sumi, qx[5], _mm512_shuffle_epi32(yh, _MM_PERM_ENUM(0x55)));
        sumi = _mm512_dpbusd_epi32(sumi, qx[6], _mm512_shuffle_epi32(yh, _MM_PERM_ENUM(0xaa)));
        sumi = _mm512_dpbusd_epi32(sumi, qx[7], _mm512_shuffle_epi32(yh, _MM_PERM_ENUM(0xff)));
        return sumi;
    };
    float d8[8*nrc_y];
    for (int ix = 0; ix < nrc_x; ix += 16) {
        const block_iq4_nl_r8 * iq4l = (const block_iq4_nl_r8 *)((const char *)vx + (ix+0)*bx);
        const block_iq4_nl_r8 * iq4h = (const block_iq4_nl_r8 *)((const char *)vx + (ix+8)*bx);
        for (int ib4 = 0; ib4 < nb/4; ++ib4) {
            for (int iy = 0; iy < nrc_y; ++iy) {
                _mm256_storeu_ps(d8+8*iy, convert_scales((const uint16_t *)q8.y[iy][ib4].d));
            }
            for (int k = 0; k < 4; ++k) {
                auto scales = prepare(iq4l[4*ib4+k], iq4h[4*ib4+k]);
                for (int iy = 0; iy < nrc_y; ++iy) {
                    auto sumi = dot(q8.y[iy][ib4].qs+32*k);
                    auto dy = _mm512_set1_ps(d8[8*iy+k]);
                    acc[2*iy+0] = _mm512_fmadd_ps(_mm512_mul_ps(scales, dy), _mm512_cvtepi32_ps(sumi), acc[2*iy+0]);
                    acc[2*iy+1] = _mm512_fmadd_ps(scales, _mm512_set1_ps(d8[8*iy+k+4]), acc[2*iy+1]);
                }
            }
        }
        for (int ib = 4*(nb/4); ib < nb; ++ib) {
            auto scales = prepare(iq4l[ib], iq4h[ib]);
            for (int iy = 0; iy < nrc_y; ++iy) {
                auto qy = (const block_q8_1 *)q8.y[iy];
                auto sumi = dot(qy[ib].qs);
                auto [d8, m8] = ScaleHelperQ8_2::prepare1(qy + ib);
                auto dy = _mm512_set1_ps(d8);
                acc[2*iy+0] = _mm512_fmadd_ps(_mm512_mul_ps(scales, dy), _mm512_cvtepi32_ps(sumi), acc[2*iy+0]);
                acc[2*iy+1] = _mm512_fmadd_ps(scales, _mm512_set1_ps(m8), acc[2*iy+1]);
            }
        }
        for (int iy = 0; iy < nrc_y; ++iy) {
            auto sum = _mm512_fmadd_ps(_mm512_set1_ps(-8.f), acc[2*iy+1], acc[2*iy+0]);
            acc[2*iy+0] = acc[2*iy+1] = _mm512_setzero_ps();
            info.store(ix, iy, sum);
        }
    }
}
#else
template <int nrc_y>
static void mul_mat_q4_0_r8_q8_2(int n, const void * vx, size_t bx, const DataInfo& info, int nrc_x) {
    mul_mat_q4_0_r8_q8_2_avx2<nrc_y>(n, vx, bx, info, nrc_x);
}
#endif

inline void prepare_mxfp4_quants_avx2(const uint8_t * qs, __m256i * v, const __m256i& m4, const __m256i & table) {
    auto bits1 = _mm256_loadu_si256((const __m256i *)qs+0);
    auto bits2 = _mm256_loadu_si256((const __m256i *)qs+1);
    auto bits3 = _mm256_loadu_si256((const __m256i *)qs+2);
    auto bits4 = _mm256_loadu_si256((const __m256i *)qs+3);
    v[0] = _mm256_shuffle_epi8(table, _mm256_and_si256(bits1, m4));
    v[1] = _mm256_shuffle_epi8(table, _mm256_and_si256(bits2, m4));
    v[2] = _mm256_shuffle_epi8(table, _mm256_and_si256(bits3, m4));
    v[3] = _mm256_shuffle_epi8(table, _mm256_and_si256(bits4, m4));
    v[4] = _mm256_shuffle_epi8(table, _mm256_and_si256(_mm256_srli_epi16(bits1, 4), m4));
    v[5] = _mm256_shuffle_epi8(table, _mm256_and_si256(_mm256_srli_epi16(bits2, 4), m4));
    v[6] = _mm256_shuffle_epi8(table, _mm256_and_si256(_mm256_srli_epi16(bits3, 4), m4));
    v[7] = _mm256_shuffle_epi8(table, _mm256_and_si256(_mm256_srli_epi16(bits4, 4), m4));
}

inline __m256i accum_mxfp4_quants(const __m256i * v, const int8_t * qs) {
    auto y4l = _mm_loadu_si128((const __m128i*)qs+0);
    auto y4h = _mm_loadu_si128((const __m128i*)qs+1);
    auto yl  = MM256_SET1_M128I(y4l);
    auto yh  = MM256_SET1_M128I(y4h);
#ifdef HAVE_VNNI256
    auto sumi = _mm256_setzero_si256();
    sumi = ggml_mm256_dpbusd_epi32(sumi, v[0], _mm256_shuffle_epi32(yl, 0x00));
    sumi = ggml_mm256_dpbusd_epi32(sumi, v[1], _mm256_shuffle_epi32(yl, 0x55));
    sumi = ggml_mm256_dpbusd_epi32(sumi, v[2], _mm256_shuffle_epi32(yl, 0xaa));
    sumi = ggml_mm256_dpbusd_epi32(sumi, v[3], _mm256_shuffle_epi32(yl, 0xff));
    sumi = ggml_mm256_dpbusd_epi32(sumi, v[4], _mm256_shuffle_epi32(yh, 0x00));
    sumi = ggml_mm256_dpbusd_epi32(sumi, v[5], _mm256_shuffle_epi32(yh, 0x55));
    sumi = ggml_mm256_dpbusd_epi32(sumi, v[6], _mm256_shuffle_epi32(yh, 0xaa));
    sumi = ggml_mm256_dpbusd_epi32(sumi, v[7], _mm256_shuffle_epi32(yh, 0xff));
#else
    auto sumi1 = _mm256_add_epi16(_mm256_maddubs_epi16(v[0], _mm256_shuffle_epi32(yl, 0x00)),
                                  _mm256_maddubs_epi16(v[1], _mm256_shuffle_epi32(yl, 0x55)));
    auto sumi2 = _mm256_add_epi16(_mm256_maddubs_epi16(v[2], _mm256_shuffle_epi32(yl, 0xaa)),
                                  _mm256_maddubs_epi16(v[3], _mm256_shuffle_epi32(yl, 0xff)));
    auto sumi3 = _mm256_add_epi16(_mm256_maddubs_epi16(v[4], _mm256_shuffle_epi32(yh, 0x00)),
                                  _mm256_maddubs_epi16(v[5], _mm256_shuffle_epi32(yh, 0x55)));
    auto sumi4 = _mm256_add_epi16(_mm256_maddubs_epi16(v[6], _mm256_shuffle_epi32(yh, 0xaa)),
                                  _mm256_maddubs_epi16(v[7], _mm256_shuffle_epi32(yh, 0xff)));
    auto m1 = _mm256_set1_epi16(1);
    auto sumi12 = _mm256_add_epi32(_mm256_madd_epi16(m1, sumi1), _mm256_madd_epi16(m1, sumi2));
    auto sumi34 = _mm256_add_epi32(_mm256_madd_epi16(m1, sumi3), _mm256_madd_epi16(m1, sumi4));
    auto sumi = _mm256_add_epi32(sumi12, sumi34);
#endif
    return sumi;
}

inline __m256 convert_mxfp4_scales(const uint8_t * e) {
    auto aux  = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i *)e));
    auto mask = _mm256_cmpgt_epi32(aux, _mm256_set1_epi32(1));
    auto d1   = _mm256_slli_epi32(_mm256_sub_epi32(aux, _mm256_set1_epi32(1)), 23);
    auto d2   = _mm256_slli_epi32(_mm256_add_epi32(aux, _mm256_set1_epi32(1)), 21);
    return _mm256_castsi256_ps(_mm256_or_si256(_mm256_and_si256(mask, d1), _mm256_andnot_si256(mask, d2)));
}

template <int nrc_y>
static void mul_mat_mxfp4_r8_q8_2_avx2(int n, const void * vx, size_t bx, const DataInfo& info, int nrc_x) {
    GGML_ASSERT(nrc_x%8 == 0);
    Q8<nrc_y, block_q8_2_x4> q8(info);
    auto m4 = _mm256_set1_epi8(0xf);
    int nb = n / QK_MXFP4;
    auto table128 = _mm_loadu_si128((const __m128i *)kvalues_mxfp4);
    auto table = MM256_SET1_M128I(table128);
    table = _mm256_add_epi8(table, _mm256_set1_epi8(12));
    __m256i v[8];
    if constexpr (nrc_y == 1) {
        union { __m256 vec; float val[8]; } helper;
        for (int ix = 0; ix < nrc_x; ix += 8) {
            auto * iq4 = (const block_mxfp4_r8 *)((const char *)vx + ix*bx);
            auto acc1 = _mm256_setzero_ps();
            auto acc2 = _mm256_setzero_ps();
            for (int ib4 = 0; ib4 < nb/4; ++ib4) {
                helper.vec = convert_scales((const uint16_t *)q8.y[0][ib4].d);
                for (int k = 0; k < 4; ++k) {
                    auto scales = convert_mxfp4_scales(iq4[4*ib4+k].e);
                    prepare_mxfp4_quants_avx2(iq4[4*ib4+k].qs, v, m4, table);
                    auto sumi = accum_mxfp4_quants(v, q8.y[0][ib4].qs+32*k);
                    auto d4d8 = _mm256_mul_ps(scales, _mm256_set1_ps(helper.val[k]));
                    acc1 = _mm256_fmadd_ps(d4d8, _mm256_cvtepi32_ps(sumi), acc1);
                    acc2 = _mm256_fmadd_ps(scales, _mm256_set1_ps(helper.val[k+4]), acc2);
                }
            }
            for (int ib = 4*(nb/4); ib < nb; ++ib) {
                auto qy = (const block_q8_2 *)q8.y[0];
                auto scales = convert_mxfp4_scales(iq4[ib].e);
                prepare_mxfp4_quants_avx2(iq4[ib].qs, v, m4, table);
                auto sumi = accum_mxfp4_quants(v, qy[ib].qs);
                auto [d8, m8] = ScaleHelperQ8_2::prepare1(qy + ib);
                auto d4d8 = _mm256_mul_ps(scales, _mm256_set1_ps(d8));
                acc1 = _mm256_fmadd_ps(d4d8, _mm256_cvtepi32_ps(sumi), acc1);
                acc2 = _mm256_fmadd_ps(scales, _mm256_set1_ps(m8), acc2);
            }
            acc1 = _mm256_fmadd_ps(acc2, _mm256_set1_ps(-12.f), acc1);
            info.store(ix, 0, acc1);
        }
    }
    else {
        __m256 acc[nrc_y] = {};
        float d8[8*nrc_y];
        for (int ix = 0; ix < nrc_x; ix += 8) {
            auto * iq4 = (const block_mxfp4_r8 *)((const char *)vx + ix*bx);
            for (int ib4 = 0; ib4 < nb/4; ++ib4) {
                __m256 d4[4];
                {
                    for (int k = 0; k < 4; ++k) {
                        d4[k] = convert_mxfp4_scales(iq4[4*ib4+k].e);
                    }
                    for (int iy = 0; iy < nrc_y; ++iy) {
                        auto scales = convert_scales((const uint16_t *)q8.y[iy][ib4].d);
                        _mm256_storeu_ps(d8 + 8*iy, scales);
                        auto m4 = _mm256_extractf128_ps(scales, 1);
                        auto m8 = _mm256_set_m128(m4, m4);
                        auto sumf = _mm256_mul_ps(d4[0], _mm256_shuffle_ps(m8, m8, 0x00));
                        sumf = _mm256_fmadd_ps(d4[1], _mm256_shuffle_ps(m8, m8, 0x55), sumf);
                        sumf = _mm256_fmadd_ps(d4[2], _mm256_shuffle_ps(m8, m8, 0xaa), sumf);
                        sumf = _mm256_fmadd_ps(d4[3], _mm256_shuffle_ps(m8, m8, 0xff), sumf);
                        acc[iy] = _mm256_fmadd_ps(sumf, _mm256_set1_ps(-12.f), acc[iy]);
                    }
                }
                for (int k = 0; k < 4; ++k) {
                    //auto scales = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)iq4[4*ib4+k].d));
                    prepare_mxfp4_quants_avx2(iq4[4*ib4+k].qs, v, m4, table);
                    for (int iy = 0; iy < nrc_y; ++iy) {
                        auto sumi = accum_mxfp4_quants(v, q8.y[iy][ib4].qs+32*k);
                        auto d4d8 = _mm256_mul_ps(d4[k], _mm256_set1_ps(d8[8*iy+k]));
                        acc[iy] = _mm256_fmadd_ps(d4d8, _mm256_cvtepi32_ps(sumi), acc[iy]);
                    }
                }
            }
            for (int ib = 4*(nb/4); ib < nb; ++ib) {
                auto scales = convert_mxfp4_scales(iq4[ib].e);
                auto scales_m = _mm256_mul_ps(scales, _mm256_set1_ps(-12.f));
                prepare_mxfp4_quants_avx2(iq4[ib].qs, v, m4, table);
                for (int iy = 0; iy < nrc_y; ++iy) {
                    auto qy = (const block_q8_2 *)q8.y[iy];
                    auto sumi = accum_mxfp4_quants(v, qy[ib].qs);
                    auto [d8, m8] = ScaleHelperQ8_2::prepare1(qy + ib);
                    auto d4d8 = _mm256_mul_ps(scales, _mm256_set1_ps(d8));
                    acc[iy] = _mm256_fmadd_ps(d4d8, _mm256_cvtepi32_ps(sumi), acc[iy]);
                    acc[iy] = _mm256_fmadd_ps(scales_m, _mm256_set1_ps(m8), acc[iy]);
                }
            }
            for (int iy = 0; iy < nrc_y; ++iy) {
                info.store(ix, iy, acc[iy]);
                acc[iy] = _mm256_setzero_ps();
            }
        }
    }
}

#ifdef HAVE_FANCY_SIMD
template <int nrc_y>
static void mul_mat_mxfp4_r8_q8_2(int n, const void * vx, size_t bx, const DataInfo& info, int nrc_x) {
    if constexpr (nrc_y == 1) {
        mul_mat_mxfp4_r8_q8_2_avx2<1>(n, vx, bx, info, nrc_x);
        return;
    }
    GGML_ASSERT(nrc_x%16 == 0);
    Q8<nrc_y, block_q8_2_x4> q8(info);
    auto m4 = _mm512_set1_epi8(0xf);
    int nb = n / QK4_NL;
    //auto table = _mm512_broadcast_i32x4(_mm_loadu_si128((const __m128i *)kvalues_mxfp4));
    auto table128 = _mm_loadu_si128((const __m128i *)kvalues_mxfp4);
    auto table256 = MM256_SET1_M128I(table128);
    auto table = _mm512_inserti32x8(_mm512_castsi256_si512(table256), table256, 1);
    table = _mm512_add_epi8(table, _mm512_set1_epi8(12));
    __m512  acc[2*nrc_y] = {};
    __m512i qx[8];
    auto prepare = [&qx, &m4, &table] (const block_mxfp4_r8& iq4l, const block_mxfp4_r8& iq4h) {
        auto scales1 = convert_mxfp4_scales(iq4l.e);
        auto scales2 = convert_mxfp4_scales(iq4h.e);
        auto scales = _mm512_insertf32x8(_mm512_castps256_ps512(scales1), scales2, 1);
        for (int j = 0; j < 4; ++j) {
            auto bits = _mm512_inserti32x8(_mm512_castsi256_si512(_mm256_loadu_si256((const __m256i *)iq4l.qs+j)),
                    _mm256_loadu_si256((const __m256i *)iq4h.qs+j), 1);
            qx[j+0] = _mm512_and_si512(bits, m4);
            qx[j+4] = _mm512_and_si512(_mm512_srli_epi16(bits, 4), m4);
        }
        for (int j = 0; j < 8; ++j) qx[j] = _mm512_shuffle_epi8(table, qx[j]);
        return scales;
    };
    auto dot = [&qx] (const int8_t * qy) {
        auto y4l = _mm_loadu_si128((const __m128i*)qy+0);
        auto y4h = _mm_loadu_si128((const __m128i*)qy+1);
        //auto yl  = _mm512_broadcast_i32x4(y4l);
        //auto yh  = _mm512_broadcast_i32x4(y4h);
        auto y8l = MM256_SET1_M128I(y4l);
        auto y8h = MM256_SET1_M128I(y4h);
        auto yl = _mm512_inserti32x8(_mm512_castsi256_si512(y8l), y8l, 1);
        auto yh = _mm512_inserti32x8(_mm512_castsi256_si512(y8h), y8h, 1);
        auto sumi = _mm512_setzero_si512();
        sumi = _mm512_dpbusd_epi32(sumi, qx[0], _mm512_shuffle_epi32(yl, _MM_PERM_ENUM(0x00)));
        sumi = _mm512_dpbusd_epi32(sumi, qx[1], _mm512_shuffle_epi32(yl, _MM_PERM_ENUM(0x55)));
        sumi = _mm512_dpbusd_epi32(sumi, qx[2], _mm512_shuffle_epi32(yl, _MM_PERM_ENUM(0xaa)));
        sumi = _mm512_dpbusd_epi32(sumi, qx[3], _mm512_shuffle_epi32(yl, _MM_PERM_ENUM(0xff)));
        sumi = _mm512_dpbusd_epi32(sumi, qx[4], _mm512_shuffle_epi32(yh, _MM_PERM_ENUM(0x00)));
        sumi = _mm512_dpbusd_epi32(sumi, qx[5], _mm512_shuffle_epi32(yh, _MM_PERM_ENUM(0x55)));
        sumi = _mm512_dpbusd_epi32(sumi, qx[6], _mm512_shuffle_epi32(yh, _MM_PERM_ENUM(0xaa)));
        sumi = _mm512_dpbusd_epi32(sumi, qx[7], _mm512_shuffle_epi32(yh, _MM_PERM_ENUM(0xff)));
        return sumi;
    };
    float d8[8*nrc_y];
    for (int ix = 0; ix < nrc_x; ix += 16) {
        auto iq4l = (const block_mxfp4_r8 *)((const char *)vx + (ix+0)*bx);
        auto iq4h = (const block_mxfp4_r8 *)((const char *)vx + (ix+8)*bx);
        for (int ib4 = 0; ib4 < nb/4; ++ib4) {
            for (int iy = 0; iy < nrc_y; ++iy) {
                _mm256_storeu_ps(d8+8*iy, convert_scales((const uint16_t *)q8.y[iy][ib4].d));
            }
            for (int k = 0; k < 4; ++k) {
                auto scales = prepare(iq4l[4*ib4+k], iq4h[4*ib4+k]);
                for (int iy = 0; iy < nrc_y; ++iy) {
                    auto sumi = dot(q8.y[iy][ib4].qs+32*k);
                    auto dy = _mm512_set1_ps(d8[8*iy+k]);
                    acc[2*iy+0] = _mm512_fmadd_ps(_mm512_mul_ps(scales, dy), _mm512_cvtepi32_ps(sumi), acc[2*iy+0]);
                    acc[2*iy+1] = _mm512_fmadd_ps(scales, _mm512_set1_ps(d8[8*iy+k+4]), acc[2*iy+1]);
                }
            }
        }
        for (int ib = 4*(nb/4); ib < nb; ++ib) {
            auto scales = prepare(iq4l[ib], iq4h[ib]);
            for (int iy = 0; iy < nrc_y; ++iy) {
                auto qy = (const block_q8_1 *)q8.y[iy];
                auto sumi = dot(qy[ib].qs);
                auto [d8, m8] = ScaleHelperQ8_2::prepare1(qy + ib);
                auto dy = _mm512_set1_ps(d8);
                acc[2*iy+0] = _mm512_fmadd_ps(_mm512_mul_ps(scales, dy), _mm512_cvtepi32_ps(sumi), acc[2*iy+0]);
                acc[2*iy+1] = _mm512_fmadd_ps(scales, _mm512_set1_ps(m8), acc[2*iy+1]);
            }
        }
        for (int iy = 0; iy < nrc_y; ++iy) {
            auto sum = _mm512_fmadd_ps(_mm512_set1_ps(-12.f), acc[2*iy+1], acc[2*iy+0]);
            acc[2*iy+0] = acc[2*iy+1] = _mm512_setzero_ps();
            info.store(ix, iy, sum);
        }
    }
}
#else
template <int nrc_y>
static void mul_mat_mxfp4_r8_q8_2(int n, const void * vx, size_t bx, const DataInfo& info, int nrc_x) {
    mul_mat_mxfp4_r8_q8_2_avx2<nrc_y>(n, vx, bx, info, nrc_x);
}
#endif

template <int nrc_y>
static void mul_mat_q5_0_r4_q8_2_avx2(int n, const void * vx, size_t bx, const DataInfo& info, int nrc_x) {
    GGML_ASSERT(nrc_x%4 == 0);
    Q8<nrc_y, block_q8_2_x4> q8(info);
    auto m4 = _mm256_set1_epi8(0xf);
    auto m5 = _mm256_set1_epi8(0x10);
#ifndef HAVE_VNNI256
    auto m1 = _mm256_set1_epi16(1);
#endif
    auto mscale = _mm256_set_m128(_mm_set1_ps(-8.f), _mm_set1_ps(1.f));
    int nb = n / QK5_0;
    __m256 acc[nrc_y] = {};
    __m256i qx[4];
    float d8[8*nrc_y];
    auto prepare = [&qx, &m4, &m5] (const block_q5_0_r4& iq5) {
        auto scales128 = _mm_cvtph_ps(_mm_loadl_epi64((const __m128i *)iq5.d));
        auto scales = _mm256_set_m128(scales128, scales128);
        auto bits1 = _mm256_loadu_si256((const __m256i *)iq5.qs+0);
        auto bits2 = _mm256_loadu_si256((const __m256i *)iq5.qs+1);
        auto hbits = _mm_loadu_si128((const __m128i *)iq5.qh);
        auto hb = MM256_SRLI128_M128I(hbits, 1);
        qx[0] = _mm256_or_si256(_mm256_and_si256(bits1, m4), _mm256_and_si256(_mm256_slli_epi16(hb, 4), m5));
        qx[1] = _mm256_or_si256(_mm256_and_si256(bits2, m4), _mm256_and_si256(_mm256_slli_epi16(hb, 2), m5));
        qx[2] = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(bits1, 4), m4), _mm256_and_si256(hb, m5));
        qx[3] = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(bits2, 4), m4), _mm256_and_si256(_mm256_srli_epi16(hb, 2), m5));;
        return scales;
    };
#ifdef HAVE_VNNI256
    auto dot = [&qx] (__m256i y) {
        auto sumi = _mm256_setzero_si256();
        sumi = ggml_mm256_dpbusd_epi32(sumi, qx[0], _mm256_shuffle_epi32(y, 0x00));
        sumi = ggml_mm256_dpbusd_epi32(sumi, qx[1], _mm256_shuffle_epi32(y, 0x55));
        sumi = ggml_mm256_dpbusd_epi32(sumi, qx[2], _mm256_shuffle_epi32(y, 0xaa));
        sumi = ggml_mm256_dpbusd_epi32(sumi, qx[3], _mm256_shuffle_epi32(y, 0xff));
        return sumi;
    };
#else
    auto dot = [&qx, &m1] (__m256i y) {
        auto sumi1 = _mm256_add_epi16(_mm256_maddubs_epi16(qx[0], _mm256_shuffle_epi32(y, 0x00)),
                                      _mm256_maddubs_epi16(qx[1], _mm256_shuffle_epi32(y, 0x55)));
        auto sumi2 = _mm256_add_epi16(_mm256_maddubs_epi16(qx[2], _mm256_shuffle_epi32(y, 0xaa)),
                                      _mm256_maddubs_epi16(qx[3], _mm256_shuffle_epi32(y, 0xff)));
        auto sumi = _mm256_madd_epi16(m1, _mm256_add_epi16(sumi1, sumi2));
        return sumi;
    };
#endif
    for (int ix = 0; ix < nrc_x; ix += 4) {
        const block_q5_0_r4 * iq5 = (const block_q5_0_r4 *)((const char *)vx + ix*bx);
        for (int ib4 = 0; ib4 < nb/4; ++ib4) {
            for (int iy = 0; iy < nrc_y; ++iy) {
                auto scales = convert_scales((const uint16_t *)q8.y[iy][ib4].d);
                _mm256_storeu_ps(d8 + 8*iy, _mm256_mul_ps(mscale, scales));
            }
            for (int k = 0; k < 4; ++k) {
                auto scales = prepare(iq5[4*ib4+k]);
                for (int iy = 0; iy < nrc_y; ++iy) {
                    auto sumi = dot(_mm256_loadu_si256((const __m256i*)q8.y[iy][ib4].qs+k));
                    auto d4d8 = _mm256_mul_ps(scales, _mm256_set1_ps(d8[8*iy+k]));
                    acc[iy] = _mm256_fmadd_ps(d4d8, _mm256_cvtepi32_ps(sumi), acc[iy]);
                    acc[iy] = _mm256_fmadd_ps(scales, _mm256_set1_ps(d8[8*iy+k+4]), acc[iy]);
                }
            }
        }
        for (int ib = 4*(nb/4); ib < nb; ++ib) {
            auto scales = prepare(iq5[ib]);
            for (int iy = 0; iy < nrc_y; ++iy) {
                auto qy = (const block_q8_2 *)q8.y[iy];
                auto sumi = dot(_mm256_loadu_si256((const __m256i*)qy[ib].qs));
                auto [d8, m8] = ScaleHelperQ8_2::prepare1(qy + ib);
                auto d4d8 = _mm256_mul_ps(scales, _mm256_set1_ps(d8));
                acc[iy] = _mm256_fmadd_ps(d4d8, _mm256_cvtepi32_ps(sumi), acc[iy]);
                acc[iy] = _mm256_fmadd_ps(scales, _mm256_set1_ps(-8.f*m8), acc[iy]);
            }
        }
        for (int iy = 0; iy < nrc_y; ++iy) {
            auto sum = _mm_add_ps(_mm256_castps256_ps128(acc[iy]), _mm256_extractf128_ps(acc[iy], 1));
            info.store(ix, iy, sum);
            acc[iy] = _mm256_setzero_ps();
        }
    }
}

#ifdef HAVE_FANCY_SIMD
template <int nrc_y>
static void mul_mat_q5_0_r4_q8_2(int n, const void * vx, size_t bx, const DataInfo& info, int nrc_x) {
    if constexpr (nrc_y == 1) {
        mul_mat_q5_0_r4_q8_2_avx2<1>(n, vx, bx, info, nrc_x);
    } else {
    GGML_ASSERT(nrc_x%8 == 0);
    Q8<nrc_y, block_q8_2_x4> q8(info);
    auto m4 = _mm512_set1_epi8(0xf);
    auto m5 = _mm512_set1_epi8(0x10);
    int nb = n / QK5_0;
    __m512  acc[2*nrc_y] = {};
    __m512i qx[4];
    float d8[8*nrc_y];
    auto prepare = [&qx, &m4, &m5] (const block_q5_0_r4& iq5l, const block_q5_0_r4& iq5h) {
        auto scales128 = _mm_cvtph_ps(_mm_loadl_epi64((const __m128i *)iq5l.d));
        auto scales1 = _mm256_set_m128(scales128, scales128);
        scales128 = _mm_cvtph_ps(_mm_loadl_epi64((const __m128i *)iq5h.d));
        auto scales2 = _mm256_set_m128(scales128, scales128);
        auto scales = _mm512_insertf32x8(_mm512_castps256_ps512(scales1), scales2, 1);
        auto bits1 = _mm512_inserti32x8(_mm512_castsi256_si512(_mm256_loadu_si256((const __m256i *)iq5l.qs+0)),
                _mm256_loadu_si256((const __m256i *)iq5h.qs+0), 1);
        auto bits2 = _mm512_inserti32x8(_mm512_castsi256_si512(_mm256_loadu_si256((const __m256i *)iq5l.qs+1)),
                _mm256_loadu_si256((const __m256i *)iq5h.qs+1), 1);
        auto hbits1 = _mm_loadu_si128((const __m128i *)iq5l.qh);
        auto hbits2 = _mm_loadu_si128((const __m128i *)iq5h.qh);
        auto hb1 = MM256_SRLI128_M128I(hbits1, 1);
        auto hb2 = MM256_SRLI128_M128I(hbits2, 1);
        auto hb = _mm512_inserti32x8(_mm512_castsi256_si512(hb1), hb2, 1);
        qx[0] = _mm512_or_si512(_mm512_and_si512(bits1, m4), _mm512_and_si512(_mm512_slli_epi16(hb, 4), m5));
        qx[1] = _mm512_or_si512(_mm512_and_si512(bits2, m4), _mm512_and_si512(_mm512_slli_epi16(hb, 2), m5));
        qx[2] = _mm512_or_si512(_mm512_and_si512(_mm512_srli_epi16(bits1, 4), m4), _mm512_and_si512(hb, m5));
        qx[3] = _mm512_or_si512(_mm512_and_si512(_mm512_srli_epi16(bits2, 4), m4), _mm512_and_si512(_mm512_srli_epi16(hb, 2), m5));
        return scales;
    };
    auto dot = [&qx] (__m256i y8) {
        auto y = _mm512_inserti32x8(_mm512_castsi256_si512(y8), y8, 1);
        auto sumi = _mm512_setzero_si512();
        sumi = _mm512_dpbusd_epi32(sumi, qx[0], _mm512_shuffle_epi32(y, _MM_PERM_ENUM(0x00)));
        sumi = _mm512_dpbusd_epi32(sumi, qx[1], _mm512_shuffle_epi32(y, _MM_PERM_ENUM(0x55)));
        sumi = _mm512_dpbusd_epi32(sumi, qx[2], _mm512_shuffle_epi32(y, _MM_PERM_ENUM(0xaa)));
        sumi = _mm512_dpbusd_epi32(sumi, qx[3], _mm512_shuffle_epi32(y, _MM_PERM_ENUM(0xff)));
        return sumi;
    };
    for (int ix = 0; ix < nrc_x; ix += 8) {
        const block_q5_0_r4 * iq5l = (const block_q5_0_r4 *)((const char *)vx + (ix+0)*bx);
        const block_q5_0_r4 * iq5h = (const block_q5_0_r4 *)((const char *)vx + (ix+4)*bx);
        for (int ib4 = 0; ib4 < nb/4; ++ib4) {
            for (int iy = 0; iy < nrc_y; ++iy) {
                _mm256_storeu_ps(d8+8*iy, convert_scales((const uint16_t *)q8.y[iy][ib4].d));
            }
            for (int k = 0; k < 4; ++k) {
                auto scales = prepare(iq5l[4*ib4+k], iq5h[4*ib4+k]);
                for (int iy = 0; iy < nrc_y; ++iy) {
                    auto sumi = dot(_mm256_loadu_si256((const __m256i*)q8.y[iy][ib4].qs+k));
                    auto dy = _mm512_set1_ps(d8[8*iy+k]);
                    acc[2*iy+0] = _mm512_fmadd_ps(_mm512_mul_ps(scales, dy), _mm512_cvtepi32_ps(sumi), acc[2*iy+0]);
                    acc[2*iy+1] = _mm512_fmadd_ps(scales, _mm512_set1_ps(d8[8*iy+k+4]), acc[2*iy+1]);
                }
            }
        }
        for (int ib = 4*(nb/4); ib < nb; ++ib) {
            auto scales = prepare(iq5l[ib], iq5h[ib]);
            for (int iy = 0; iy < nrc_y; ++iy) {
                auto qy = (const block_q8_2 *)q8.y[iy];
                auto sumi = dot(_mm256_loadu_si256((const __m256i*)qy[ib].qs));
                auto [d8, m8] = ScaleHelperQ8_2::prepare1(qy + ib);
                auto dy = _mm512_set1_ps(d8);
                acc[2*iy+0] = _mm512_fmadd_ps(_mm512_mul_ps(scales, dy), _mm512_cvtepi32_ps(sumi), acc[2*iy+0]);
                acc[2*iy+1] = _mm512_fmadd_ps(scales, _mm512_set1_ps(m8), acc[2*iy+1]);
            }
        }
        for (int iy = 0; iy < nrc_y; ++iy) {
            auto sum512 = _mm512_fmadd_ps(_mm512_set1_ps(-8.f), acc[2*iy+1], acc[2*iy+0]);
            acc[2*iy+0] = acc[2*iy+1] = _mm512_setzero_ps();
            auto sum1 = _mm_add_ps(_mm512_extractf32x4_ps(sum512, 0), _mm512_extractf32x4_ps(sum512, 1));
            auto sum2 = _mm_add_ps(_mm512_extractf32x4_ps(sum512, 2), _mm512_extractf32x4_ps(sum512, 3));
            info.store(ix+0, iy, sum1);
            info.store(ix+4, iy, sum2);
        }
    }
    }
}
#else
template <int nrc_y>
static void mul_mat_q5_0_r4_q8_2(int n, const void * vx, size_t bx, const DataInfo& info, int nrc_x) {
    mul_mat_q5_0_r4_q8_2_avx2<nrc_y>(n, vx, bx, info, nrc_x);
}
#endif

template <int nrc_y>
static void mul_mat_q6_0_r4_q8_2_avx2(int n, const void * vx, size_t bx, const DataInfo& info, int nrc_x) {
    GGML_ASSERT(nrc_x%4 == 0);
    Q8<nrc_y, block_q8_2_x4> q8(info);
    auto m4 = _mm256_set1_epi8(0xf);
    auto m6 = _mm256_set1_epi8(0x30);
    auto mscale = _mm256_set_m128(_mm_set1_ps(-16.f), _mm_set1_ps(1.f));
#ifndef HAVE_VNNI256
    auto m1 = _mm256_set1_epi16(1);
#endif
    int nb = n / QK6_0;
    __m256 acc[nrc_y] = {};
    float d8[8*nrc_y];
    __m256i qx[4];
    auto prepare = [&qx, &m4, &m6] (const block_q6_0_r4& iq6) {
        auto scales128 = _mm_cvtph_ps(_mm_loadl_epi64((const __m128i *)iq6.d));
        auto scales = _mm256_set_m128(scales128, scales128);
        auto bits1 = _mm256_loadu_si256((const __m256i *)iq6.qs+0);
        auto bits2 = _mm256_loadu_si256((const __m256i *)iq6.qs+1);
        auto hbits = _mm256_loadu_si256((const __m256i *)iq6.qh);
        qx[0] = _mm256_or_si256(_mm256_and_si256(bits1, m4), _mm256_and_si256(_mm256_slli_epi16(hbits, 4), m6));
        qx[1] = _mm256_or_si256(_mm256_and_si256(bits2, m4), _mm256_and_si256(_mm256_slli_epi16(hbits, 2), m6));
        qx[2] = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(bits1, 4), m4), _mm256_and_si256(hbits, m6));
        qx[3] = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(bits2, 4), m4), _mm256_and_si256(_mm256_srli_epi16(hbits, 2), m6));
        return scales;
    };
#ifdef HAVE_VNNI256
    auto dot = [&qx] (__m256i y) {
        auto sumi = ggml_mm256_dpbusd_epi32(_mm256_setzero_si256(), qx[0], _mm256_shuffle_epi32(y, 0x00));
        sumi = ggml_mm256_dpbusd_epi32(sumi, qx[1], _mm256_shuffle_epi32(y, 0x55));
        sumi = ggml_mm256_dpbusd_epi32(sumi, qx[2], _mm256_shuffle_epi32(y, 0xaa));
        sumi = ggml_mm256_dpbusd_epi32(sumi, qx[3], _mm256_shuffle_epi32(y, 0xff));
        return sumi;
    };
#else
    auto dot = [&qx, &m1] (__m256i y) {
        auto sumi1 = _mm256_add_epi16(_mm256_maddubs_epi16(qx[0], _mm256_shuffle_epi32(y, 0x00)),
                                      _mm256_maddubs_epi16(qx[1], _mm256_shuffle_epi32(y, 0x55)));
        auto sumi2 = _mm256_add_epi16(_mm256_maddubs_epi16(qx[2], _mm256_shuffle_epi32(y, 0xaa)),
                                      _mm256_maddubs_epi16(qx[3], _mm256_shuffle_epi32(y, 0xff)));
        auto sumi = _mm256_add_epi32(_mm256_madd_epi16(m1, sumi1), _mm256_madd_epi16(m1, sumi2));
        return sumi;
    };
#endif
    for (int ix = 0; ix < nrc_x; ix += 4) {
        const block_q6_0_r4 * iq6 = (const block_q6_0_r4 *)((const char *)vx + ix*bx);
        for (int ib4 = 0; ib4 < nb/4; ++ib4) {
            for (int iy = 0; iy < nrc_y; ++iy) {
                auto scales = convert_scales((const uint16_t *)q8.y[iy][ib4].d);
                _mm256_storeu_ps(d8 + 8*iy,  _mm256_mul_ps(scales, mscale));
            }
            for (int k = 0; k < 4; ++k) {
                auto scales = prepare(iq6[4*ib4+k]);
                for (int iy = 0; iy < nrc_y; ++iy) {
                    auto sumi = dot(_mm256_loadu_si256((const __m256i*)q8.y[iy][ib4].qs+k));
                    auto d4d8 = _mm256_mul_ps(scales, _mm256_set1_ps(d8[8*iy+k]));
                    acc[iy] = _mm256_fmadd_ps(d4d8, _mm256_cvtepi32_ps(sumi), acc[iy]);
                    acc[iy] = _mm256_fmadd_ps(scales, _mm256_set1_ps(d8[8*iy+k+4]), acc[iy]);
                }
            }
        }
        for (int ib = 4*(nb/4); ib < nb; ++ib) {
            auto scales = prepare(iq6[ib]);
            for (int iy = 0; iy < nrc_y; ++iy) {
                auto qy = (const block_q8_2 *)q8.y[iy];
                auto sumi = dot(_mm256_loadu_si256((const __m256i*)qy[ib].qs));
                auto [d8, m8] = ScaleHelperQ8_2::prepare1(qy + ib);
                auto d4d8 = _mm256_mul_ps(scales, _mm256_set1_ps(d8));
                acc[iy] = _mm256_fmadd_ps(d4d8, _mm256_cvtepi32_ps(sumi), acc[iy]);
                acc[iy] = _mm256_fmadd_ps(scales, _mm256_set1_ps(-16.f*m8), acc[iy]);
            }
        }

        for (int iy = 0; iy < nrc_y; ++iy) {
            auto sum = _mm_add_ps(_mm256_castps256_ps128(acc[iy]), _mm256_extractf128_ps(acc[iy], 1));
            info.store(ix, iy, sum);
            acc[iy] = _mm256_setzero_ps();
        }
    }
}

#ifdef HAVE_FANCY_SIMD
template <int nrc_y>
static void mul_mat_q6_0_r4_q8_2(int n, const void * vx, size_t bx, const DataInfo& info, int nrc_x) {
    if constexpr (nrc_y == 1) {
        mul_mat_q6_0_r4_q8_2_avx2<1>(n, vx, bx, info, nrc_x);
    } else {
    GGML_ASSERT(nrc_x%8 == 0);
    Q8<nrc_y, block_q8_2_x4> q8(info);
    auto m4 = _mm512_set1_epi8(0xf);
    auto m6 = _mm512_set1_epi8(0x30);
    int nb = n / QK6_0;
    __m512  acc[2*nrc_y] = {};
    __m512i qx[4];
    float d8[8*nrc_y];
    auto prepare = [&qx, &m4, &m6] (const block_q6_0_r4& iq6l, const block_q6_0_r4& iq6h) {
        auto scales128 = _mm_cvtph_ps(_mm_loadl_epi64((const __m128i *)iq6l.d));
        auto scales1 = _mm256_set_m128(scales128, scales128);
        scales128 = _mm_cvtph_ps(_mm_loadl_epi64((const __m128i *)iq6h.d));
        auto scales2 = _mm256_set_m128(scales128, scales128);
        auto scales = _mm512_insertf32x8(_mm512_castps256_ps512(scales1), scales2, 1);
        auto bits1 = _mm512_inserti32x8(_mm512_castsi256_si512(_mm256_loadu_si256((const __m256i *)iq6l.qs+0)),
                                                               _mm256_loadu_si256((const __m256i *)iq6h.qs+0), 1);
        auto bits2 = _mm512_inserti32x8(_mm512_castsi256_si512(_mm256_loadu_si256((const __m256i *)iq6l.qs+1)),
                                                               _mm256_loadu_si256((const __m256i *)iq6h.qs+1), 1);
        auto hbits1 = _mm256_loadu_si256((const __m256i *)iq6l.qh);
        auto hbits2 = _mm256_loadu_si256((const __m256i *)iq6h.qh);
        auto hb = _mm512_inserti32x8(_mm512_castsi256_si512(hbits1), hbits2, 1);
        qx[0] = _mm512_and_si512(bits1, m4) | _mm512_and_si512(_mm512_slli_epi16(hb, 4), m6);
        qx[1] = _mm512_and_si512(bits2, m4) | _mm512_and_si512(_mm512_slli_epi16(hb, 2), m6);;
        qx[2] = _mm512_and_si512(_mm512_srli_epi16(bits1, 4), m4) | _mm512_and_si512(hb, m6);
        qx[3] = _mm512_and_si512(_mm512_srli_epi16(bits2, 4), m4) | _mm512_and_si512(_mm512_srli_epi16(hb, 2), m6);
        return scales;
    };
    auto dot = [&qx] (__m256i y8) {
        auto y = _mm512_inserti32x8(_mm512_castsi256_si512(y8), y8, 1);
        auto sumi = _mm512_setzero_si512();
        sumi = _mm512_dpbusd_epi32(sumi, qx[0], _mm512_shuffle_epi32(y, _MM_PERM_ENUM(0x00)));
        sumi = _mm512_dpbusd_epi32(sumi, qx[1], _mm512_shuffle_epi32(y, _MM_PERM_ENUM(0x55)));
        sumi = _mm512_dpbusd_epi32(sumi, qx[2], _mm512_shuffle_epi32(y, _MM_PERM_ENUM(0xaa)));
        sumi = _mm512_dpbusd_epi32(sumi, qx[3], _mm512_shuffle_epi32(y, _MM_PERM_ENUM(0xff)));
        return sumi;
    };
    for (int ix = 0; ix < nrc_x; ix += 8) {
        const block_q6_0_r4 * iq6l = (const block_q6_0_r4 *)((const char *)vx + (ix+0)*bx);
        const block_q6_0_r4 * iq6h = (const block_q6_0_r4 *)((const char *)vx + (ix+4)*bx);
        for (int ib4 = 0; ib4 < nb/4; ++ib4) {
            for (int iy = 0; iy < nrc_y; ++iy) {
                auto scales = convert_scales((const uint16_t *)q8.y[iy][ib4].d);
                _mm256_storeu_ps(d8 + 8*iy, scales);
            }
            for (int k = 0; k < 4; ++k) {
                auto scales = prepare(iq6l[4*ib4+k], iq6h[4*ib4+k]);
                for (int iy = 0; iy < nrc_y; ++iy) {
                    auto sumi = dot(_mm256_loadu_si256((const __m256i*)q8.y[iy][ib4].qs+k));
                    auto dy = _mm512_set1_ps(d8[8*iy+k]);
                    acc[2*iy+0] = _mm512_fmadd_ps(_mm512_mul_ps(scales, dy), _mm512_cvtepi32_ps(sumi), acc[2*iy+0]);
                    acc[2*iy+1] = _mm512_fmadd_ps(scales, _mm512_set1_ps(d8[8*iy+k+4]), acc[2*iy+1]);
                }
            }
        }
        for (int ib = 4*(nb/4); ib < nb; ++ib) {
            auto scales = prepare(iq6l[ib], iq6h[ib]);
            for (int iy = 0; iy < nrc_y; ++iy) {
                auto qy = (const block_q8_2 *)q8.y[iy];
                auto sumi = dot(_mm256_loadu_si256((const __m256i*)qy[ib].qs));
                auto [d8, m8] = ScaleHelperQ8_2::prepare1(qy + ib);
                auto dy = _mm512_set1_ps(d8);
                acc[2*iy+0] = _mm512_fmadd_ps(_mm512_mul_ps(scales, dy), _mm512_cvtepi32_ps(sumi), acc[2*iy+0]);
                acc[2*iy+1] = _mm512_fmadd_ps(scales, _mm512_set1_ps(m8), acc[2*iy+1]);
            }
        }
        for (int iy = 0; iy < nrc_y; ++iy) {
            auto sum512 = _mm512_fmadd_ps(_mm512_set1_ps(-16.f), acc[2*iy+1], acc[2*iy+0]);
            acc[2*iy+0] = acc[2*iy+1] = _mm512_setzero_ps();
            auto sum1 = _mm_add_ps(_mm512_extractf32x4_ps(sum512, 0), _mm512_extractf32x4_ps(sum512, 1));
            auto sum2 = _mm_add_ps(_mm512_extractf32x4_ps(sum512, 2), _mm512_extractf32x4_ps(sum512, 3));
            info.store(ix+0, iy, sum1);
            info.store(ix+4, iy, sum2);
        }
    }
    }
}
#else
template <int nrc_y>
static void mul_mat_q6_0_r4_q8_2(int n, const void * vx, size_t bx, const DataInfo& info, int nrc_x) {
    mul_mat_q6_0_r4_q8_2_avx2<nrc_y>(n, vx, bx, info, nrc_x);
}
#endif

#ifdef HAVE_FANCY_SIMD
inline __m512i qx_r8_q8_dot_product(const __m512i * qx, const int8_t * y) {
    auto y4l = _mm_loadu_si128((const __m128i*)y+0);
    auto y4h = _mm_loadu_si128((const __m128i*)y+1);
    auto y8l = MM256_SET1_M128I(y4l);
    auto y8h = MM256_SET1_M128I(y4h);
    auto yl  = _mm512_inserti32x8(_mm512_castsi256_si512(y8l), y8l, 1);
    auto yh  = _mm512_inserti32x8(_mm512_castsi256_si512(y8h), y8h, 1);
    auto sumi = _mm512_setzero_si512();
    sumi = _mm512_dpbusd_epi32(sumi, qx[0], _mm512_shuffle_epi32(yl, _MM_PERM_ENUM(0x00)));
    sumi = _mm512_dpbusd_epi32(sumi, qx[1], _mm512_shuffle_epi32(yl, _MM_PERM_ENUM(0x55)));
    sumi = _mm512_dpbusd_epi32(sumi, qx[2], _mm512_shuffle_epi32(yl, _MM_PERM_ENUM(0xaa)));
    sumi = _mm512_dpbusd_epi32(sumi, qx[3], _mm512_shuffle_epi32(yl, _MM_PERM_ENUM(0xff)));
    sumi = _mm512_dpbusd_epi32(sumi, qx[4], _mm512_shuffle_epi32(yh, _MM_PERM_ENUM(0x00)));
    sumi = _mm512_dpbusd_epi32(sumi, qx[5], _mm512_shuffle_epi32(yh, _MM_PERM_ENUM(0x55)));
    sumi = _mm512_dpbusd_epi32(sumi, qx[6], _mm512_shuffle_epi32(yh, _MM_PERM_ENUM(0xaa)));
    sumi = _mm512_dpbusd_epi32(sumi, qx[7], _mm512_shuffle_epi32(yh, _MM_PERM_ENUM(0xff)));
    return sumi;
}
inline __m256i qx_r8_q8_dot_product(const __m256i * qx, const int8_t * y) {
    auto y4l = _mm_loadu_si128((const __m128i*)y+0);
    auto y4h = _mm_loadu_si128((const __m128i*)y+1);
    auto yl  = MM256_SET1_M128I(y4l);
    auto yh  = MM256_SET1_M128I(y4h);
    auto sumi = _mm256_setzero_si256();
    sumi = ggml_mm256_dpbusd_epi32(sumi, qx[0], _mm256_shuffle_epi32(yl, 0x00));
    sumi = ggml_mm256_dpbusd_epi32(sumi, qx[1], _mm256_shuffle_epi32(yl, 0x55));
    sumi = ggml_mm256_dpbusd_epi32(sumi, qx[2], _mm256_shuffle_epi32(yl, 0xaa));
    sumi = ggml_mm256_dpbusd_epi32(sumi, qx[3], _mm256_shuffle_epi32(yl, 0xff));
    sumi = ggml_mm256_dpbusd_epi32(sumi, qx[4], _mm256_shuffle_epi32(yh, 0x00));
    sumi = ggml_mm256_dpbusd_epi32(sumi, qx[5], _mm256_shuffle_epi32(yh, 0x55));
    sumi = ggml_mm256_dpbusd_epi32(sumi, qx[6], _mm256_shuffle_epi32(yh, 0xaa));
    sumi = ggml_mm256_dpbusd_epi32(sumi, qx[7], _mm256_shuffle_epi32(yh, 0xff));
    return sumi;
}
inline __m256i q8_0_r8_dot_product(const uint8_t * x, const int8_t * y, __m256i * qx) {
    for (int i = 0; i < 8; ++i) {
        qx[i] = _mm256_add_epi8(_mm256_loadu_si256((const __m256i *)x+i), _mm256_set1_epi8(127));
    }
    return qx_r8_q8_dot_product(qx, y);
}
template <int nrc_y>
static void mul_mat_q8_0_r8_q8_2(int n, const void * vx, size_t bx, const DataInfo& info, int nrc_x) {
    GGML_ASSERT(nrc_x%16 == 0);
    Q8<nrc_y, block_q8_2_x4> q8(info);
    int nb = n / QK8_0;
    if constexpr (nrc_y == 1) {
        __m256 acc[2] = {};
        __m256i qx[8];
        float d8[8];
        for (int ix = 0; ix < nrc_x; ix += 8) {
            const block_q8_0_r8 * iq8 = (const block_q8_0_r8 *)((const char *)vx + ix*bx);
            for (int ib4 = 0; ib4 < nb/4; ++ib4) {
                _mm256_storeu_ps(d8, convert_scales((const uint16_t *)q8.y[0][ib4].d));
                for (int k = 0; k < 4; ++k) {
                    auto scales = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)iq8[4*ib4+k].d));
                    auto sumi = q8_0_r8_dot_product((const uint8_t *)iq8[4*ib4+k].qs, q8.y[0][ib4].qs+32*k, qx);
                    auto d4d8 = _mm256_mul_ps(scales, _mm256_set1_ps(d8[k]));
                    acc[0] = _mm256_fmadd_ps(d4d8, _mm256_cvtepi32_ps(sumi), acc[0]);
                    acc[1] = _mm256_fmadd_ps(scales, _mm256_set1_ps(d8[k+4]), acc[1]);
                }
            }
            if (4*(nb/4) < nb) {
                auto qy = (const block_q8_2 *)q8.y[0];
                for (int ib = 4*(nb/4); ib < nb; ++ib) {
                    auto scales = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)iq8[ib].d));
                    auto sumi = q8_0_r8_dot_product((const uint8_t *)iq8[ib].qs, qy[ib].qs, qx);
                    auto [d8, m8] = ScaleHelperQ8_2::prepare1(qy + ib);
                    auto d4d8 = _mm256_mul_ps(scales, _mm256_set1_ps(d8));
                    acc[0] = _mm256_fmadd_ps(d4d8, _mm256_cvtepi32_ps(sumi), acc[0]);
                    acc[1] = _mm256_fmadd_ps(scales, _mm256_set1_ps(m8), acc[1]);
                }
            }
            info.store(ix, 0, _mm256_fmadd_ps(_mm256_set1_ps(-127.f), acc[1], acc[0]));
            acc[0] = acc[1] = _mm256_setzero_ps();
        }
    } else {
        __m512  acc[2*nrc_y] = {};
        __m512i qx[8];
        float d8[8*nrc_y];
        for (int ix = 0; ix < nrc_x; ix += 16) {
            const block_q8_0_r8 * q8l = (const block_q8_0_r8 *)((const char *)vx + (ix+0)*bx);
            const block_q8_0_r8 * q8h = (const block_q8_0_r8 *)((const char *)vx + (ix+8)*bx);
            for (int ib4 = 0; ib4 < nb/4; ++ib4) {
                for (int iy = 0; iy < nrc_y; ++iy) {
                    _mm256_storeu_ps(d8+8*iy, convert_scales((const uint16_t *)q8.y[iy][ib4].d));
                }
                for (int k = 0; k < 4; ++k) {
                    auto scales1  = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)q8l[4*ib4+k].d));
                    auto scales2  = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)q8h[4*ib4+k].d));
                    auto scales   = _mm512_insertf32x8(_mm512_castps256_ps512(scales1), scales2, 1);
                    for (int j = 0; j < 8; ++j) {
                        qx[j] = _mm512_inserti32x8(_mm512_castsi256_si512(_mm256_loadu_si256((const __m256i *)q8l[4*ib4+k].qs+j)),
                                                                          _mm256_loadu_si256((const __m256i *)q8h[4*ib4+k].qs+j), 1);
                        qx[j] = _mm512_add_epi8(qx[j], _mm512_set1_epi8(127));
                    }
                    for (int iy = 0; iy < nrc_y; ++iy) {
                        auto sumi = qx_r8_q8_dot_product(qx, q8.y[iy][ib4].qs+32*k);
                        auto dy = _mm512_set1_ps(d8[8*iy+k]);
                        acc[2*iy+0] = _mm512_fmadd_ps(_mm512_mul_ps(scales, dy), _mm512_cvtepi32_ps(sumi), acc[2*iy+0]);
                        acc[2*iy+1] = _mm512_fmadd_ps(scales, _mm512_set1_ps(d8[8*iy+k+4]), acc[2*iy+1]);
                    }
                }
            }
            for (int ib = 4*(nb/4); ib < nb; ++ib) {
                auto scales1  = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)q8l[ib].d));
                auto scales2  = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)q8h[ib].d));
                auto scales   = _mm512_insertf32x8(_mm512_castps256_ps512(scales1), scales2, 1);
                for (int j = 0; j < 8; ++j) {
                    qx[j] = _mm512_inserti32x8(_mm512_castsi256_si512(_mm256_loadu_si256((const __m256i *)q8l[ib].qs+j)),
                                                                      _mm256_loadu_si256((const __m256i *)q8h[ib].qs+j), 1);
                    qx[j] = _mm512_add_epi8(qx[j], _mm512_set1_epi8(127));
                }
                for (int iy = 0; iy < nrc_y; ++iy) {
                    auto qy = (const block_q8_2 *)q8.y[iy];
                    auto sumi = qx_r8_q8_dot_product(qx, qy[ib].qs);
                    auto [d8, m8] = ScaleHelperQ8_2::prepare1(qy + ib);
                    auto dy = _mm512_set1_ps(d8);
                    acc[2*iy+0] = _mm512_fmadd_ps(_mm512_mul_ps(scales, dy), _mm512_cvtepi32_ps(sumi), acc[2*iy+0]);
                    acc[2*iy+1] = _mm512_fmadd_ps(scales, _mm512_set1_ps(m8), acc[2*iy+1]);
                }
            }
            for (int iy = 0; iy < nrc_y; ++iy) {
                auto sum512 = _mm512_fmadd_ps(_mm512_set1_ps(-127.f), acc[2*iy+1], acc[2*iy+0]);
                info.store(ix, iy, sum512);
                acc[2*iy+0] = acc[2*iy+1] = _mm512_setzero_ps();
            }
        }
    }
}
#else
template <int nrc_y>
static void mul_mat_q8_0_r8_q8_2(int n, const void * vx, size_t bx, const DataInfo& info, int nrc_x) {
    GGML_ASSERT(nrc_x%8 == 0);
    Q8<nrc_y, block_q8_2_x4> q8(info);
    auto m1 = _mm256_set1_epi16(1);
    int nb = n / QK8_0;
    __m256 acc[nrc_y] = {};
    float d8[4*nrc_y];
    __m256i qx[4], sx[4];
    auto dot = [&qx, &sx, &m1] (const int8_t * qy) {
        auto y128 = _mm_loadu_si128((const __m128i*)qy);
        auto y = MM256_SET1_M128I(y128);
#ifdef HAVE_VNNI256
        auto sumi = _mm256_setzero_si256();
        sumi = ggml_mm256_dpbusd_epi32(sumi, sx[0], _mm256_sign_epi8(_mm256_shuffle_epi32(y, 0x00), qx[0]));
        sumi = ggml_mm256_dpbusd_epi32(sumi, sx[1], _mm256_sign_epi8(_mm256_shuffle_epi32(y, 0x55), qx[1]));
        sumi = ggml_mm256_dpbusd_epi32(sumi, sx[2], _mm256_sign_epi8(_mm256_shuffle_epi32(y, 0xaa), qx[2]));
        sumi = ggml_mm256_dpbusd_epi32(sumi, sx[3], _mm256_sign_epi8(_mm256_shuffle_epi32(y, 0xff), qx[3]));
        return sumi;
#else
        auto sumi1 = _mm256_add_epi32(
                _mm256_madd_epi16(m1, _mm256_maddubs_epi16(sx[0], _mm256_sign_epi8(_mm256_shuffle_epi32(y, 0x00), qx[0]))),
                _mm256_madd_epi16(m1, _mm256_maddubs_epi16(sx[1], _mm256_sign_epi8(_mm256_shuffle_epi32(y, 0x55), qx[1])))
        );
        auto sumi2 = _mm256_add_epi32(
                _mm256_madd_epi16(m1, _mm256_maddubs_epi16(sx[2], _mm256_sign_epi8(_mm256_shuffle_epi32(y, 0xaa), qx[2]))),
                _mm256_madd_epi16(m1, _mm256_maddubs_epi16(sx[3], _mm256_sign_epi8(_mm256_shuffle_epi32(y, 0xff), qx[3])))
        );
        return _mm256_add_epi32(sumi1, sumi2);
#endif
    };
    for (int ix = 0; ix < nrc_x; ix += 8) {
        const block_q8_0_r8 * iq8 = (const block_q8_0_r8 *)((const char *)vx + ix*bx);
        for (int ib4 = 0; ib4 < nb/4; ++ib4) {
            for (int iy = 0; iy < nrc_y; ++iy) {
                auto scales = _mm_castsi128_ps(_mm_slli_epi32(_mm_cvtepu16_epi32(_mm_loadl_epi64((const __m128i *)q8.y[iy][ib4].d)), 16));
                _mm_storeu_ps(d8 + 4*iy, scales);
            }
            for (int k = 0; k < 4; ++k) {
                auto scales = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)iq8[4*ib4+k].d));
                __m256i sumi_first[nrc_y];
                for (int j = 0; j < 4; ++j) {
                    qx[j] = _mm256_loadu_si256((const __m256i *)iq8[4*ib4+k].qs+j);
                    sx[j] = _mm256_sign_epi8(qx[j], qx[j]);
                }
                for (int iy = 0; iy < nrc_y; ++iy) {
                    sumi_first[iy] = dot(q8.y[iy][ib4].qs+32*k);
                }
                for (int j = 0; j < 4; ++j) {
                    qx[j] = _mm256_loadu_si256((const __m256i *)iq8[4*ib4+k].qs+4+j);
                    sx[j] = _mm256_sign_epi8(qx[j], qx[j]);
                }
                for (int iy = 0; iy < nrc_y; ++iy) {
                    auto sumi = _mm256_add_epi32(sumi_first[iy], dot(q8.y[iy][ib4].qs+32*k+16));
                    auto d4d8 = _mm256_mul_ps(scales, _mm256_set1_ps(d8[4*iy+k]));
                    acc[iy] = _mm256_fmadd_ps(d4d8, _mm256_cvtepi32_ps(sumi), acc[iy]);
                }
            }
        }
        for (int ib = 4*(nb/4); ib < nb; ++ib) {
            auto scales = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)iq8[ib].d));
            __m256i sumi_first[nrc_y];
            for (int j = 0; j < 4; ++j) {
                qx[j] = _mm256_loadu_si256((const __m256i *)iq8[ib].qs+j);
                sx[j] = _mm256_sign_epi8(qx[j], qx[j]);
            }
            for (int iy = 0; iy < nrc_y; ++iy) {
                auto qy = (const block_q8_2 *)q8.y[iy];
                sumi_first[iy] = dot(qy[ib].qs);
            }
            for (int j = 0; j < 4; ++j) {
                qx[j] = _mm256_loadu_si256((const __m256i *)iq8[ib].qs+4+j);
                sx[j] = _mm256_sign_epi8(qx[j], qx[j]);
            }
            for (int iy = 0; iy < nrc_y; ++iy) {
                auto qy = (const block_q8_2 *)q8.y[iy];
                auto sumi = _mm256_add_epi32(sumi_first[iy], dot(qy[ib].qs+16));
                auto d4d8 = _mm256_mul_ps(scales, _mm256_set1_ps(GGML_BF16_TO_FP32(ggml_bf16_t{qy[ib].d})));
                acc[iy] = _mm256_fmadd_ps(d4d8, _mm256_cvtepi32_ps(sumi), acc[iy]);
            }
        }
        for (int iy = 0; iy < nrc_y; ++iy) {
            info.store(ix, iy, acc[iy]);
            acc[iy] = _mm256_setzero_ps();
        }
    }
}
#endif

typedef struct {
    ggml_half d[16];
    uint8_t   qs[256];
} block_q8_1_r8;

#ifdef HAVE_FANCY_SIMD
template <int nrc_y>
static void mul_mat_q8_1_r8_q8_2(int n, const void * vx, size_t bx, const DataInfo& info, int nrc_x) {
    GGML_ASSERT(nrc_x%16 == 0);
    Q8<nrc_y, block_q8_2_x4> q8(info);
    int nb = n / QK8_0;
    if constexpr (nrc_y == 1) {
        __m256 acc[1] = {};
        float d8[4];
        __m256i qx[4];
        auto dot = [&qx] (const int8_t * qy) {
            auto y128 = _mm_loadu_si128((const __m128i*)qy);
            auto y = MM256_SET1_M128I(y128);
            auto sumi = _mm256_setzero_si256();
            sumi = ggml_mm256_dpbusd_epi32(sumi, qx[0], _mm256_shuffle_epi32(y, 0x00));
            sumi = ggml_mm256_dpbusd_epi32(sumi, qx[1], _mm256_shuffle_epi32(y, 0x55));
            sumi = ggml_mm256_dpbusd_epi32(sumi, qx[2], _mm256_shuffle_epi32(y, 0xaa));
            sumi = ggml_mm256_dpbusd_epi32(sumi, qx[3], _mm256_shuffle_epi32(y, 0xff));
            return sumi;
        };
        for (int ix = 0; ix < nrc_x; ix += 8) {
            const block_q8_1_r8 * iq8 = (const block_q8_1_r8 *)((const char *)vx + ix*bx);
            for (int i4 = 0; i4 < nb/4; ++i4) {
                {
                    __m256 mx[4];
                    for (int ib32 = 0; ib32 < 4; ++ib32) mx[ib32] = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)iq8[4*i4+ib32].d+1));
                    auto scales = _mm_castsi128_ps(_mm_slli_epi32(_mm_cvtepu16_epi32(_mm_loadl_epi64((const __m128i *)q8.y[0][i4].d)), 16));
                    _mm_storeu_ps(d8, scales);
                    auto bsums4 = _mm_cvtepi32_ps(_mm_cvtepi16_epi32(_mm_loadl_epi64((const __m128i *)(q8.y[0][i4].d+4))));
                    bsums4 = _mm_mul_ps(bsums4, scales);
                    auto bsums  = _mm256_set_m128(bsums4, bsums4);
                    acc[0] = _mm256_fmadd_ps(mx[0], _mm256_shuffle_ps(bsums, bsums, 0x00), acc[0]);
                    acc[0] = _mm256_fmadd_ps(mx[1], _mm256_shuffle_ps(bsums, bsums, 0x55), acc[0]);
                    acc[0] = _mm256_fmadd_ps(mx[2], _mm256_shuffle_ps(bsums, bsums, 0xaa), acc[0]);
                    acc[0] = _mm256_fmadd_ps(mx[3], _mm256_shuffle_ps(bsums, bsums, 0xff), acc[0]);
                }
                for (int ib32 = 0; ib32 < 4; ++ib32) {
                    auto scales = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)iq8[4*i4+ib32].d));
                    for (int j = 0; j < 4; ++j) {
                        qx[j] = _mm256_loadu_si256((const __m256i *)iq8[4*i4+ib32].qs+j);
                    }
                    auto sumi = dot(q8.y[0][i4].qs+32*ib32);
                    auto d4d8 = _mm256_mul_ps(scales, _mm256_set1_ps(d8[ib32]));
                    acc[0] = _mm256_fmadd_ps(d4d8, _mm256_cvtepi32_ps(sumi), acc[0]);
                    for (int j = 0; j < 4; ++j) {
                        qx[j] = _mm256_loadu_si256((const __m256i *)iq8[4*i4+ib32].qs+4+j);
                    }
                    sumi = dot(q8.y[0][i4].qs+32*ib32+16);
                    d4d8 = _mm256_mul_ps(scales, _mm256_set1_ps(d8[ib32]));
                    acc[0] = _mm256_fmadd_ps(d4d8, _mm256_cvtepi32_ps(sumi), acc[0]);
                }
            }
            info.store(ix, 0, acc[0]);
            acc[0] = _mm256_setzero_ps();
        }
    } else {
        __m512  acc[nrc_y] = {};
        __m512i qx[8];
        float d8[4*nrc_y];
        for (int ix = 0; ix < nrc_x; ix += 16) {
            const block_q8_1_r8 * q8l = (const block_q8_1_r8 *)((const char *)vx + (ix+0)*bx);
            const block_q8_1_r8 * q8h = (const block_q8_1_r8 *)((const char *)vx + (ix+8)*bx);
            for (int i4 = 0; i4 < nb/4; ++i4) {
                {
                    __m512 mx[4];
                    for (int ib32 = 0; ib32 < 4; ++ib32) {
                        auto mx_l = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)q8l[4*i4+ib32].d+1));
                        auto mx_h = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)q8h[4*i4+ib32].d+1));
                        mx[ib32] = _mm512_insertf32x8(_mm512_castps256_ps512(mx_l), mx_h, 1);
                    }
                    for (int iy = 0; iy < nrc_y; ++iy) {
                        auto scales128 = _mm_castsi128_ps(_mm_slli_epi32(_mm_cvtepu16_epi32(_mm_loadl_epi64((const __m128i *)q8.y[iy][i4].d)), 16));
                        _mm_storeu_ps(d8 + 4*iy, scales128);
                        auto bsums4 = _mm_cvtepi32_ps(_mm_cvtepi16_epi32(_mm_loadl_epi64((const __m128i *)(q8.y[iy][i4].d+4))));
                        bsums4 = _mm_mul_ps(bsums4, scales128);
                        auto bsums256 = _mm256_set_m128(bsums4, bsums4);
                        auto bsums = _mm512_insertf32x8(_mm512_castps256_ps512(bsums256), bsums256, 1);
                        acc[iy] = _mm512_fmadd_ps(mx[0], _mm512_shuffle_ps(bsums, bsums, 0x00), acc[iy]);
                        acc[iy] = _mm512_fmadd_ps(mx[1], _mm512_shuffle_ps(bsums, bsums, 0x55), acc[iy]);
                        acc[iy] = _mm512_fmadd_ps(mx[2], _mm512_shuffle_ps(bsums, bsums, 0xaa), acc[iy]);
                        acc[iy] = _mm512_fmadd_ps(mx[3], _mm512_shuffle_ps(bsums, bsums, 0xff), acc[iy]);
                    }
                }
                for (int ib32 = 0; ib32 < 4; ++ib32) {
                    auto scales_l = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)q8l[4*i4+ib32].d));
                    auto scales_h = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)q8h[4*i4+ib32].d));
                    auto scales   = _mm512_insertf32x8(_mm512_castps256_ps512(scales_l), scales_h, 1);
                    for (int j = 0; j < 8; ++j) {
                        qx[j] = _mm512_inserti32x8(_mm512_castsi256_si512(_mm256_loadu_si256((const __m256i *)q8l[4*i4+ib32].qs+j)),
                                                                          _mm256_loadu_si256((const __m256i *)q8h[4*i4+ib32].qs+j), 1);
                    }
                    for (int iy = 0; iy < nrc_y; ++iy) {
                        auto sumi = qx_r8_q8_dot_product(qx, q8.y[iy][i4].qs+32*ib32);
                        auto dy = _mm512_set1_ps(d8[4*iy+ib32]);
                        acc[iy] = _mm512_fmadd_ps(_mm512_mul_ps(scales, dy), _mm512_cvtepi32_ps(sumi), acc[iy]);
                    }
                }
            }
            for (int iy = 0; iy < nrc_y; ++iy) {
                info.store(ix, iy, acc[iy]);
                acc[iy] = _mm512_setzero_ps();
            }
        }
    }
}
#else
template <int nrc_y>
static void mul_mat_q8_1_r8_q8_2(int n, const void * vx, size_t bx, const DataInfo& info, int nrc_x) {
    GGML_ASSERT(nrc_x%8 == 0);
    Q8<nrc_y, block_q8_2_x4> q8(info);
    int nb = n / QK8_0;
    __m256 acc[nrc_y] = {};
    float d8[4*nrc_y];
    __m256i qx[4];
    auto dot = [&qx] (const int8_t * qy) {
        auto y128 = _mm_loadu_si128((const __m128i*)qy);
        auto y = MM256_SET1_M128I(y128);
#ifdef HAVE_VNNI256
        auto sumi = _mm256_setzero_si256();
        sumi = ggml_mm256_dpbusd_epi32(sumi, qx[0], _mm256_shuffle_epi32(y, 0x00));
        sumi = ggml_mm256_dpbusd_epi32(sumi, qx[1], _mm256_shuffle_epi32(y, 0x55));
        sumi = ggml_mm256_dpbusd_epi32(sumi, qx[2], _mm256_shuffle_epi32(y, 0xaa));
        sumi = ggml_mm256_dpbusd_epi32(sumi, qx[3], _mm256_shuffle_epi32(y, 0xff));
        return sumi;
#else
        auto sumi1 = _mm256_add_epi16(_mm256_maddubs_epi16(qx[0], _mm256_shuffle_epi32(y, 0x00)),
                                      _mm256_maddubs_epi16(qx[1], _mm256_shuffle_epi32(y, 0x55)));
        auto sumi2 = _mm256_add_epi16(_mm256_maddubs_epi16(qx[2], _mm256_shuffle_epi32(y, 0xaa)),
                                      _mm256_maddubs_epi16(qx[3], _mm256_shuffle_epi32(y, 0xff)));
        return _mm256_add_epi32(_mm256_madd_epi16(_mm256_set1_epi16(1), sumi1), _mm256_madd_epi16(_mm256_set1_epi16(1), sumi2));
#endif
    };
    for (int ix = 0; ix < nrc_x; ix += 8) {
        const block_q8_1_r8 * iq8 = (const block_q8_1_r8 *)((const char *)vx + ix*bx);
        for (int i4 = 0; i4 < nb/4; ++i4) {
            {
                __m256 mx[4];
                for (int ib32 = 0; ib32 < 4; ++ib32) mx[ib32] = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)iq8[4*i4+ib32].d+1));
                for (int iy = 0; iy < nrc_y; ++iy) {
                    auto scales = _mm_castsi128_ps(_mm_slli_epi32(_mm_cvtepu16_epi32(_mm_loadl_epi64((const __m128i *)q8.y[iy][i4].d)), 16));
                    _mm_storeu_ps(d8 + 4*iy + 0, scales);
                    auto bsums4 = _mm_cvtepi32_ps(_mm_cvtepi16_epi32(_mm_loadl_epi64((const __m128i *)(q8.y[iy][i4].d+4))));
                    bsums4 = _mm_mul_ps(bsums4, scales);
                    auto bsums  = _mm256_set_m128(bsums4, bsums4);
                    acc[iy] = _mm256_fmadd_ps(mx[0], _mm256_shuffle_ps(bsums, bsums, 0x00), acc[iy]);
                    acc[iy] = _mm256_fmadd_ps(mx[1], _mm256_shuffle_ps(bsums, bsums, 0x55), acc[iy]);
                    acc[iy] = _mm256_fmadd_ps(mx[2], _mm256_shuffle_ps(bsums, bsums, 0xaa), acc[iy]);
                    acc[iy] = _mm256_fmadd_ps(mx[3], _mm256_shuffle_ps(bsums, bsums, 0xff), acc[iy]);
                }
            }
            for (int ib32 = 0; ib32 < 4; ++ib32) {
                auto scales = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)iq8[4*i4+ib32].d));
                for (int j = 0; j < 4; ++j) {
                    qx[j] = _mm256_loadu_si256((const __m256i *)iq8[4*i4+ib32].qs+j);
                }
                for (int iy = 0; iy < nrc_y; ++iy) {
                    auto sumi = dot(q8.y[iy][i4].qs+32*ib32);
                    auto d4d8 = _mm256_mul_ps(scales, _mm256_set1_ps(d8[4*iy+ib32]));
                    acc[iy] = _mm256_fmadd_ps(d4d8, _mm256_cvtepi32_ps(sumi), acc[iy]);
                }
                for (int j = 0; j < 4; ++j) {
                    qx[j] = _mm256_loadu_si256((const __m256i *)iq8[4*i4+ib32].qs+4+j);
                }
                for (int iy = 0; iy < nrc_y; ++iy) {
                    auto sumi = dot(q8.y[iy][i4].qs+32*ib32+16);
                    auto d4d8 = _mm256_mul_ps(scales, _mm256_set1_ps(d8[4*iy+ib32]));
                    acc[iy] = _mm256_fmadd_ps(d4d8, _mm256_cvtepi32_ps(sumi), acc[iy]);
                }
            }
        }
        for (int iy = 0; iy < nrc_y; ++iy) {
            info.store(ix, iy, acc[iy]);
            acc[iy] = _mm256_setzero_ps();
        }
    }
}
#endif

void iqk_convert_q80_q80_r8(int n, const void * vx, size_t bx, void * vy, int nrc_x) {
    static_assert(QK4_0 == QK8_0);
    GGML_ASSERT(n%QK4_0 == 0);
    GGML_ASSERT(nrc_x%8 == 0);

    const int nb = n/QK4_0;

    block_q8_0_r8 * y = (block_q8_0_r8 *)vy;

    const block_q8_0 * x8[8];

    uint32_t block[8];

    for (int ix = 0; ix < nrc_x; ix += 8) {

        for (int k = 0; k < 8; ++k) x8[k] = (const block_q8_0 *)((const char *)vx + (ix + k)*bx);

        for (int i = 0; i < nb; ++i) {
            for (int k = 0; k < 8; ++k) {
                y[i].d[k] = x8[k][i].d;
                _mm256_storeu_si256((__m256i *)block, _mm256_loadu_si256((const __m256i *)x8[k][i].qs));
                auto qs = (uint32_t *)y[i].qs;
                for (int l = 0; l < 4; ++l) {
                    qs[8*l + k +  0] = block[l + 0];
                    qs[8*l + k + 32] = block[l + 4];
                }
            }
        }
        y += nb;
    }
}

template <typename Block, typename Dequantizer>
void iqk_convert_qX_q80_r8(int n, const void * vx, size_t bx, void * vy, int nrc_x) {
    GGML_ASSERT(n%QK4_0 == 0);
    GGML_ASSERT(nrc_x%8 == 0);

    const int nb = n/QK8_0;

    block_q8_0_r8 * y = (block_q8_0_r8 *)vy;

    const Block * x8[8];

    uint32_t block[8];

    Dequantizer deq;

    for (int ix = 0; ix < nrc_x; ix += 8) {

        for (int k = 0; k < 8; ++k) x8[k] = (const Block *)((const char *)vx + (ix + k)*bx);

        for (int i = 0; i < nb; ++i) {
            for (int k = 0; k < 8; ++k) {
                if constexpr (std::is_same_v<Dequantizer, MXFP40_Dequantizer>) {
                    y[i].d[k] = GGML_FP32_TO_FP16(GGML_E8M0_TO_FP32_HALF(x8[k][i].e));
                } else {
                    y[i].d[k] = x8[k][i].d;
                }
                _mm256_storeu_si256((__m256i *)block, deq.dequant(x8[k] + i));
                auto qs = (uint32_t *)y[i].qs;
                for (int l = 0; l < 4; ++l) {
                    qs[8*l + k +  0] = block[l + 0];
                    qs[8*l + k + 32] = block[l + 4];
                }
            }
        }
        y += nb;
    }
}

template <typename Block, typename Dequantizer>
void iqk_convert_qX_1_q8_1_r8(int n, const void * vx, size_t bx, void * vy, int nrc_x) {
    GGML_ASSERT(n%QK8_0 == 0);
    GGML_ASSERT(nrc_x%8 == 0);

    int nb = n/QK8_0;

    const Block * x8[8];

    block_q8_1_r8 * y = (block_q8_1_r8 *)vy;

    uint32_t block[8];

    Dequantizer deq;

    for (int ix = 0; ix < nrc_x; ix += 8) {
        for (int k = 0; k < 8; ++k) x8[k] = (const Block *)((const char *)vx + (ix + k)*bx);
        for (int i = 0; i < nb; ++i) {
            for (int k = 0; k < 8; ++k) {
                y[i].d[k+0] = x8[k][i].d;
                y[i].d[k+8] = x8[k][i].m;
                _mm256_storeu_si256((__m256i *)block, deq.dequant(x8[k]+i));
                auto qs = (uint32_t *)y[i].qs;
                for (int l = 0; l < 4; ++l) {
                    qs[8*l + k +  0] = block[l + 0];
                    qs[8*l + k + 32] = block[l + 4];
                }
            }
        }
        y += nb;
    }
}

template <typename Dequantizer> void set_functions(std::array<mul_mat_t, IQK_MAX_NY>& funcs) {
    if constexpr (std::is_same_v<Dequantizer, Q4_0_Unpacker> || std::is_same_v<Dequantizer, Q5_0_Unpacker>) {
        IQK_SET_MUL_MAT_FUNCTIONS_T(mul_mat_qX_0_q8_0_T, Dequantizer, funcs)
    }
    else if constexpr (std::is_same_v<Dequantizer, Q8_0_Unpacker>) {
        IQK_SET_MUL_MAT_FUNCTIONS_T2(mul_mat_qX_0_q8_0_T, Dequantizer, block_q8_2, funcs)
    }
    else if constexpr (std::is_same_v<Dequantizer, Q4_1_Unpacker> || std::is_same_v<Dequantizer, Q5_1_Unpacker> || std::is_same_v<Dequantizer, Q6_1_Unpacker>) {
        IQK_SET_MUL_MAT_FUNCTIONS_T(mul_mat_qX_1_q8_2_T, Dequantizer, funcs)
    }
    else if constexpr (std::is_same_v<Dequantizer, IQ4_NL_UnpackerU>) {
        IQK_SET_MUL_MAT_FUNCTIONS_T(mul_mat_qX_1_q8_2_T, Dequantizer, funcs)
    }
    else if constexpr (std::is_same_v<Dequantizer, IQ4_NL_UnpackerS>) {
        IQK_SET_MUL_MAT_FUNCTIONS_T2(mul_mat_qX_0_q8_0_T, Dequantizer, block_q8_2, funcs)
    }
    else if constexpr (std::is_same_v<Dequantizer, Q8_0_1_Unpacker> || std::is_same_v<Dequantizer, Q4_0_1_Unpacker> ||
                       std::is_same_v<Dequantizer, Q5_0_1_Unpacker> || std::is_same_v<Dequantizer, Q6_0_1_Unpacker> ||
                       std::is_same_v<Dequantizer, MXFP4_Unpacker>) {
        IQK_SET_MUL_MAT_FUNCTIONS_T(mul_mat_qX_1_q8_2_T, Dequantizer, funcs)
    }
}

} // namespace

bool iqk_convert_legacy_quants_q8_r8(int type, int n, const void * vx, size_t bx, void * vy, int nrc_x) {
    switch (type) {
        case GGML_TYPE_Q4_0  : iqk_convert_qX_q80_r8<block_q4_0, Q4_0_Dequantizer>(n, vx, bx, vy, nrc_x); break;
        case GGML_TYPE_Q4_1  : iqk_convert_qX_1_q8_1_r8<block_q4_1, Q4_1_Dequantizer>(n, vx, bx, vy, nrc_x); break;
        case GGML_TYPE_Q5_0  : iqk_convert_qX_q80_r8<block_q5_0, Q5_0_Dequantizer>(n, vx, bx, vy, nrc_x); break;
        case GGML_TYPE_Q5_1  : iqk_convert_qX_1_q8_1_r8<block_q5_1, Q5_1_Dequantizer<block_q5_1>>(n, vx, bx, vy, nrc_x); break;
        case GGML_TYPE_Q6_1  : iqk_convert_qX_1_q8_1_r8<block_q6_1, Q6_1_Dequantizer<block_q6_1>>(n, vx, bx, vy, nrc_x); break;
        case GGML_TYPE_Q6_0  : iqk_convert_qX_q80_r8<block_q6_0, Q6_0_Dequantizer>(n, vx, bx, vy, nrc_x); break;
        case GGML_TYPE_IQ4_NL: iqk_convert_qX_q80_r8<block_iq4_nl, IQ4_NL_DequantizerS>(n, vx, bx, vy, nrc_x); break;
        case GGML_TYPE_Q8_0  : iqk_convert_q80_q80_r8(n, vx, bx, vy, nrc_x); break;
        case GGML_TYPE_MXFP4 : iqk_convert_qX_q80_r8<block_mxfp4, MXFP40_Dequantizer>(n, vx, bx, vy, nrc_x); break;
        default: return false;
    }
    return true;
}

bool iqk_set_kernels_legacy_quants(int ne00, int typeA, int typeB, std::array<mul_mat_t, IQK_MAX_NY>& kernels, mul_mat_t& func16) {

    if (ne00%QK8_0 != 0) return false;

    auto expected_typeB = GGML_TYPE_Q8_2_X4;

    func16 = nullptr;

    switch (typeA) {
        case GGML_TYPE_Q4_0:
            set_functions<Q4_0_1_Unpacker>(kernels);
            break;
        case GGML_TYPE_Q4_1:
            set_functions<Q4_1_Unpacker>(kernels);
            break;
        case GGML_TYPE_Q5_0:
            set_functions<Q5_0_1_Unpacker>(kernels);
            break;
        case GGML_TYPE_Q5_1:
            set_functions<Q5_1_Unpacker>(kernels);
            break;
        case GGML_TYPE_Q6_1:
            set_functions<Q6_1_Unpacker>(kernels);
            break;
        case GGML_TYPE_Q6_0:
            set_functions<Q6_0_1_Unpacker>(kernels);
            break;
        case GGML_TYPE_Q8_0:
#ifdef HAVE_VNNI256
            set_functions<Q8_0_1_Unpacker>(kernels);
#else
            set_functions<Q8_0_Unpacker>(kernels);
#endif
            break;
        case GGML_TYPE_IQ4_NL:
#ifdef HAVE_VNNI256
            set_functions<IQ4_NL_UnpackerU>(kernels);
#else
            set_functions<IQ4_NL_UnpackerS>(kernels);
#endif
            break;
        case GGML_TYPE_MXFP4:
            set_functions<MXFP4_Unpacker>(kernels);
            break;
        case GGML_TYPE_Q4_0_R8:
            IQK_SET_MUL_MAT_FUNCTIONS(mul_mat_q4_0_r8_q8_2, kernels)
#ifdef HAVE_VNNI256
            func16 = mul_mat_q4_0_r8_q8_2<16>;
#endif
            break;
        case GGML_TYPE_MXFP4_R8:
            IQK_SET_MUL_MAT_FUNCTIONS(mul_mat_mxfp4_r8_q8_2, kernels)
//#ifdef HAVE_FANCY_SIMD
//            func16 = mul_mat_mxfp4_r8_q8_2<16>;
//#endif
            break;
        case GGML_TYPE_Q5_0_R4:
            IQK_SET_MUL_MAT_FUNCTIONS(mul_mat_q5_0_r4_q8_2, kernels)
            break;
        case GGML_TYPE_Q6_0_R4:
            IQK_SET_MUL_MAT_FUNCTIONS(mul_mat_q6_0_r4_q8_2, kernels)
            break;
        case GGML_TYPE_Q8_0_R8:
            IQK_SET_MUL_MAT_FUNCTIONS(mul_mat_q8_0_r8_q8_2, kernels)
            break;
        case GGML_TYPE_IQ4_NL_R4:
            IQK_SET_MUL_MAT_FUNCTIONS(mul_mat_iq4_nl_r4_q8_2, kernels)
            break;
        case GGML_TYPE_Q8_1: // Note: we are misusing the Q8_1 type for Q8_1_R8
            IQK_SET_MUL_MAT_FUNCTIONS(mul_mat_q8_1_r8_q8_2, kernels)
            break;
        default:
            return false;
    }

    return ggml_type(typeB) == expected_typeB;
}

#else
// ---------------------------- __aarch64__ ----------------------------------------------

Removed aarch64 for clarity in my fork.

// ---------------------------------------------------------------------------------------

bool iqk_convert_legacy_quants_q8_r8(int type, int n, const void * vx, size_t bx, void * vy, int nrc_x) {
    switch (type) {
        case GGML_TYPE_Q4_0  : iqk_convert_qX_q80_r8<block_q4_0, DeqQ40>(n, vx, bx, vy, nrc_x); break;
        case GGML_TYPE_Q4_1  : iqk_convert_qX_1_q8_1_r8<block_q4_1, DeqQ41>(n, vx, bx, vy, nrc_x); break;
        case GGML_TYPE_Q5_0  : iqk_convert_qX_q80_r8<block_q5_0, DeqQ50>(n, vx, bx, vy, nrc_x); break;
        case GGML_TYPE_Q5_1  : iqk_convert_qX_1_q8_1_r8<block_q5_1, DeqQ51>(n, vx, bx, vy, nrc_x); break;
        case GGML_TYPE_Q6_1  : iqk_convert_qX_1_q8_1_r8<block_q6_1, DeqQ61>(n, vx, bx, vy, nrc_x); break;
        case GGML_TYPE_Q6_0  : iqk_convert_qX_q80_r8<block_q6_0, DeqQ60>(n, vx, bx, vy, nrc_x); break;
        case GGML_TYPE_IQ4_NL: iqk_convert_qX_q80_r8<block_iq4_nl, DeqIQ4NL>(n, vx, bx, vy, nrc_x); break;
        case GGML_TYPE_MXFP4 : iqk_convert_qX_q80_r8<block_mxfp4, DeqMXFP4>(n, vx, bx, vy, nrc_x); break;
        case GGML_TYPE_Q8_0  : iqk_convert_qX_q80_r8<block_q8_0, DeqQ80>(n, vx, bx, vy, nrc_x); break;
        default: return false;
    }
    return true;
}

bool iqk_set_kernels_legacy_quants(int ne00, int typeA, int typeB, std::array<mul_mat_t, IQK_MAX_NY>& kernels, mul_mat_t& func16) {

    if (ne00%QK8_0 != 0) return false;

    auto etypeA = ggml_type(typeA);
    auto expected_typeB = etypeA == GGML_TYPE_Q4_1 || etypeA == GGML_TYPE_Q5_1 || etypeA == GGML_TYPE_Q6_1 || etypeA == GGML_TYPE_Q8_1 ? GGML_TYPE_Q8_1_X4 : GGML_TYPE_Q8_0_X4;
    if (ggml_type(typeB) != expected_typeB) return false;

    func16 = nullptr;

    switch (typeA) {
        case GGML_TYPE_Q4_0:
            IQK_SET_MUL_MAT_FUNCTIONS_T(mul_mat_qX_0_q8_0, DequantizerQ40, kernels);
            break;
        case GGML_TYPE_Q4_1:
            IQK_SET_MUL_MAT_FUNCTIONS_T(mul_mat_qX_1_q8_1, DequantizerQ41, kernels);
            break;
        case GGML_TYPE_Q5_0:
            IQK_SET_MUL_MAT_FUNCTIONS_T(mul_mat_qX_0_q8_0, DequantizerQ50, kernels);
            break;
        case GGML_TYPE_Q5_1:
            IQK_SET_MUL_MAT_FUNCTIONS_T(mul_mat_qX_1_q8_1, DequantizerQ51, kernels);
            break;
        case GGML_TYPE_Q6_1:
            IQK_SET_MUL_MAT_FUNCTIONS_T(mul_mat_qX_1_q8_1, DequantizerQ61, kernels);
            break;
        case GGML_TYPE_Q6_0:
            IQK_SET_MUL_MAT_FUNCTIONS_T(mul_mat_qX_0_q8_0, DequantizerQ60, kernels);
            break;
        case GGML_TYPE_Q8_0:
            IQK_SET_MUL_MAT_FUNCTIONS_T(mul_mat_qX_0_q8_0, DequantizerQ80, kernels);
            break;
        case GGML_TYPE_IQ4_NL:
            IQK_SET_MUL_MAT_FUNCTIONS_T(mul_mat_qX_0_q8_0, DequantizerIQ4NL, kernels);
            break;
        case GGML_TYPE_MXFP4:
            IQK_SET_MUL_MAT_FUNCTIONS_T(mul_mat_qX_0_q8_0, DequantizerMXFP4, kernels);
            break;
        case GGML_TYPE_Q4_0_R8:
            IQK_SET_MUL_MAT_FUNCTIONS_T(mul_mat_qx_r8_q8_0, Q4_0_R8_Dequantizer, kernels);
            break;
        case GGML_TYPE_Q5_0_R4:
            IQK_SET_MUL_MAT_FUNCTIONS_T(mul_mat_qx_r4_q8_0, Q5_0_R4_Dequantizer, kernels);
            break;
        case GGML_TYPE_Q6_0_R4:
            IQK_SET_MUL_MAT_FUNCTIONS_T(mul_mat_qx_r4_q8_0, Q6_0_R4_Dequantizer, kernels);
            break;
        case GGML_TYPE_Q8_0_R8:
            IQK_SET_MUL_MAT_FUNCTIONS(mul_mat_q8_0_r8_q8_0, kernels);
            break;
        case GGML_TYPE_Q8_1:
            IQK_SET_MUL_MAT_FUNCTIONS(mul_mat_q8_1_r8_q8_1, kernels);
            break;
        case GGML_TYPE_IQ4_NL_R4:
            IQK_SET_MUL_MAT_FUNCTIONS_T(mul_mat_qx_r4_q8_0, IQ4_NL_R4_Dequantizer, kernels);
            break;
        default:
            return false;
    }

    return true;
}

#endif

namespace {
template <int k_step>
inline std::pair<mul_mat_t, int> mul_mat_kernel(int int_typeA, int nq) {
    auto typeA = ggml_type(int_typeA);
    constexpr int kMaxQ = 8;
#define MAKE_FUNCS(mul_mat, n) \
    if (n >= kMaxQ) return std::make_pair(mul_mat, kMaxQ>, kMaxQ);\
    else {\
        switch (n) {\
            case 1: return std::make_pair(mul_mat, 1>, 1);\
            case 2: return std::make_pair(mul_mat, 2>, 2);\
            case 3: return std::make_pair(mul_mat, 3>, 3);\
            case 4: return std::make_pair(mul_mat, 4>, 4);\
            case 5: return std::make_pair(mul_mat, 5>, 5);\
            case 6: return std::make_pair(mul_mat, 6>, 6);\
            case 7: return std::make_pair(mul_mat, 7>, 7);\
        }\
    }
#define MAKE_FUNCS2(mul_mat, block, n) \
    if (n >= kMaxQ) return std::make_pair(mul_mat, kMaxQ, block>, kMaxQ);\
    else {\
        switch (n) {\
            case 1: return std::make_pair(mul_mat, 1, block>, 1);\
            case 2: return std::make_pair(mul_mat, 2, block>, 2);\
            case 3: return std::make_pair(mul_mat, 3, block>, 3);\
            case 4: return std::make_pair(mul_mat, 4, block>, 4);\
            case 5: return std::make_pair(mul_mat, 5, block>, 5);\
            case 6: return std::make_pair(mul_mat, 6, block>, 6);\
            case 7: return std::make_pair(mul_mat, 7, block>, 7);\
        }\
    }
#define MAKE_FUNCS_ONLY_NRC(mul_mat, n) \
    if (n >= kMaxQ) return std::make_pair(mul_mat<kMaxQ>, kMaxQ);\
    else {\
        switch (n) {\
            case 1: return std::make_pair(mul_mat<1>, 1);\
            case 2: return std::make_pair(mul_mat<2>, 2);\
            case 3: return std::make_pair(mul_mat<3>, 3);\
            case 4: return std::make_pair(mul_mat<4>, 4);\
            case 5: return std::make_pair(mul_mat<5>, 5);\
            case 6: return std::make_pair(mul_mat<6>, 6);\
            case 7: return std::make_pair(mul_mat<7>, 7);\
        }\
    }
    if (typeA == GGML_TYPE_Q8_0) {
#ifdef __aarch64__
        MAKE_FUNCS(mul_mat_qX_0_q8_0<DequantizerQ80, nq);
#else
#ifdef HAVE_VNNI256
        if (nq == 1) return std::make_pair(mul_mat_qX_0_q8_2_Tx<Q8_0_1_Unpacker, 1, k_step>, 1);
        if (nq == 2) return std::make_pair(mul_mat_qX_0_q8_2_Tx<Q8_0_1_Unpacker, 2, k_step>, 2);
        if (nq == 4) return std::make_pair(mul_mat_qX_0_q8_2_Tx<Q8_0_1_Unpacker, 4, k_step>, 4);
        MAKE_FUNCS(mul_mat_qX_1_q8_2_T<Q8_0_1_Unpacker, nq);
#else
        //if (nq == 1) return std::make_pair(mul_mat_qX_0_q8_0_Tx<Q8_0_Unpacker, 1, k_step>, 1);
        //if (nq == 2) return std::make_pair(mul_mat_qX_0_q8_0_Tx<Q8_0_Unpacker, 2, k_step>, 2);
        //if (nq == 4) return std::make_pair(mul_mat_qX_0_q8_0_Tx<Q8_0_Unpacker, 4, k_step>, 4);
        if (nq == 1) return std::make_pair(mul_mat_qX_0_q8_0_T<Q8_0_Unpacker, 1, block_q8_2>, 1);
        if (nq == 2) return std::make_pair(mul_mat_qX_0_q8_0_T<Q8_0_Unpacker, 2, block_q8_2>, 2);
        if (nq == 4) return std::make_pair(mul_mat_qX_0_q8_0_T<Q8_0_Unpacker, 4, block_q8_2>, 4);
        if (nq == 3) return std::make_pair(mul_mat_qX_0_q8_0_T<Q8_0_Unpacker, 3, block_q8_2>, 3);
        if (nq == 5) return std::make_pair(mul_mat_qX_0_q8_0_T<Q8_0_Unpacker, 5, block_q8_2>, 5);
        if (nq == 6) return std::make_pair(mul_mat_qX_0_q8_0_T<Q8_0_Unpacker, 6, block_q8_2>, 6);
        if (nq == 7) return std::make_pair(mul_mat_qX_0_q8_0_T<Q8_0_Unpacker, 7, block_q8_2>, 7);
        return std::make_pair(mul_mat_qX_0_q8_0_T<Q8_0_Unpacker, kMaxQ, block_q8_2>, kMaxQ);
#endif
#endif
    }
    else if (typeA == GGML_TYPE_Q8_0_R8) {
#ifdef __aarch64__
        MAKE_FUNCS_ONLY_NRC(mul_mat_q8_0_r8_q8_0, nq);
#else
        MAKE_FUNCS_ONLY_NRC(mul_mat_q8_0_r8_q8_2, nq);
#endif
    }
    else if (typeA == GGML_TYPE_Q6_1) {
#ifdef __aarch64__
        MAKE_FUNCS(mul_mat_qX_1_q8_1<DequantizerQ61, nq);
#else
        MAKE_FUNCS(mul_mat_qX_1_q8_2_T<Q6_1_Unpacker, nq);
#endif
    }
    else if (typeA == GGML_TYPE_Q6_0) {
#ifdef __aarch64__
        MAKE_FUNCS(mul_mat_qX_0_q8_0<DequantizerQ60, nq);
#else
        if (nq == 1) return std::make_pair(mul_mat_qX_0_q8_2_Tx<Q6_0_1_Unpacker, 1, k_step>, 1);
        if (nq == 2) return std::make_pair(mul_mat_qX_0_q8_2_Tx<Q6_0_1_Unpacker, 2, k_step>, 2);
        if (nq == 4) return std::make_pair(mul_mat_qX_0_q8_2_Tx<Q6_0_1_Unpacker, 4, k_step>, 4);
        MAKE_FUNCS(mul_mat_qX_1_q8_2_T<Q6_0_1_Unpacker, nq);
#endif
    }
    else if (typeA == GGML_TYPE_Q4_0) {
#ifdef __aarch64__
        MAKE_FUNCS(mul_mat_qX_0_q8_0<DequantizerQ40, nq);
#else
        if (nq == 1) return std::make_pair(mul_mat_qX_0_q8_2_Tx<Q4_0_1_Unpacker, 1, k_step>, 1);
        if (nq == 2) return std::make_pair(mul_mat_qX_0_q8_2_Tx<Q4_0_1_Unpacker, 2, k_step>, 2);
        if (nq == 4) return std::make_pair(mul_mat_qX_0_q8_2_Tx<Q4_0_1_Unpacker, 4, k_step>, 4);
        MAKE_FUNCS(mul_mat_qX_1_q8_2_T<Q4_0_1_Unpacker, nq);
#endif
    }
#if GGML_IQK_FA_ALL_QUANTS
    else if (typeA == GGML_TYPE_Q4_1) {
#ifdef __aarch64__
        MAKE_FUNCS(mul_mat_qX_1_q8_1<DequantizerQ41, nq);
#else
        MAKE_FUNCS(mul_mat_qX_1_q8_2_T<Q4_1_Unpacker, nq);
#endif
    }
    else if (typeA == GGML_TYPE_IQ4_NL) {
#ifdef __aarch64__
       MAKE_FUNCS(mul_mat_qX_0_q8_0<DequantizerIQ4NL, nq);
#else
#ifdef HAVE_VNNI256
       MAKE_FUNCS(mul_mat_qX_1_q8_2_T<IQ4_NL_UnpackerU, nq);
#else
       MAKE_FUNCS2(mul_mat_qX_0_q8_0_T<IQ4_NL_UnpackerS, block_q8_2, nq);
#endif
#endif
    }
#endif
    else {
        GGML_ASSERT(false);
    }
    return std::make_pair<mul_mat_t, int>(nullptr, 0);
}

inline std::pair<mul_mat_t, int> mul_mat_kernel(int int_typeA, int nq, int k_step) {
    switch (k_step) {
        case  32: return mul_mat_kernel< 32>(int_typeA, nq);
        case  64: return mul_mat_kernel< 64>(int_typeA, nq);
        case 128: return mul_mat_kernel<128>(int_typeA, nq);
        default: GGML_ABORT("Fatal error");
    }
}
}

void iqk_gemm_legacy_fa(int D, int nq, int type_k, const char * k, size_t stride_k, DataInfo& info, int k_step) {
    auto [mul_mat, nrc_q] = mul_mat_kernel(type_k, nq, k_step);
    for (int iq = 0; iq < nq/nrc_q; ++iq) {
        mul_mat(D, k, stride_k, info, k_step);
        info.cur_y += nrc_q;
    }
    int iq = nrc_q*(nq/nrc_q);
    if (iq < nq) {
        auto [mul_mat1, nrc_q1] = mul_mat_kernel(type_k, nq - iq, k_step);
        GGML_ASSERT(nrc_q1 == nq - iq);
        mul_mat1(D, k, stride_k, info, k_step);
    }
}

#endif
