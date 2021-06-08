#include <immintrin.h>
#include "cpu/aten/aten.hpp"
#include "cpu/dil/dil.hpp"
#include "cpu/dbl/Common.h"
#include "cpu/int8/Config.h"
#include "cpu/ExtendOPs.h"
#include "utils.h"
namespace torch_ipex {
    namespace mlperf {
        namespace dlrm {
            static inline void update_cache(const int ci, const int i,
                                            const int8_t *p, __m512i (&cache_reg)[27][4],
                                            int (&cache_idx)[27]) {
                if (cache_idx[ci] != i) {
                    cache_idx[ci] = i;
                    cache_reg[ci][0] = _mm512_cvtepi8_epi16(_mm256_load_si256((const __m256i *)p));
                    cache_reg[ci][1] = _mm512_cvtepi8_epi16(_mm256_load_si256((const __m256i *)(p + 32)));
                    cache_reg[ci][2] = _mm512_cvtepi8_epi16(_mm256_load_si256((const __m256i *)(p + 64)));
                    cache_reg[ci][3] = _mm512_cvtepi8_epi16(_mm256_load_si256((const __m256i *)(p + 96)));
                }
            }

#define dp8accum(N)                                                     \
            int64_t j_##N = ind1[(m + N)];                              \
            int64_t k_##N = ind2[(m + N)];                              \
            __m512i out_##N = _mm512_setzero_si512();                   \
            out_##N = _mm512_dpwssds_epi32(out_##N, cache_reg[j_##N][0], \
                                           cache_reg[k_##N][0]);        \
            out_##N = _mm512_dpwssds_epi32(out_##N, cache_reg[j_##N][1], \
                                           cache_reg[k_##N][1]);        \
            out_##N = _mm512_dpwssds_epi32(out_##N, cache_reg[j_##N][2], \
                                           cache_reg[k_##N][2]);        \
            out_##N = _mm512_dpwssds_epi32(out_##N, cache_reg[j_##N][3], \
                                           cache_reg[k_##N][3]);

            dil::tensor fuse_emb_int_128_27(at::Tensor &lS_o,
                                            at::Tensor &lS_i,
                                            std::vector<at::Tensor> &emb,
                                            at::Tensor &dx,
                                            const size_t Batch,
                                            const int64_t ops_id) {
                /*
                 * Batch for batch
                 * lS_o is offset, lS_i is index, emb is embedding weight
                 * dx is output of MLP bottom
                 */
                const size_t s = 26;
                const size_t Dim = 128;
                // get ptr
                const int64_t *offset[26] __attribute__((aligned(64)));
                const int64_t *index[26] __attribute__((aligned(64)));
                const int8_t *weight[26] __attribute__((aligned(64)));
                for (int i = 0; i < emb.size(); ++i) {
                    weight[i] = static_cast<int8_t *>(
                        cpu::dbl::comm::try_gen_dil_tensor(emb[i]).get_data_handle());
                    index[i] = lS_i.data_ptr<int64_t>() + i * Batch;
                    offset[i] = lS_o.data_ptr<int64_t>() + i * Batch;
                }
                int8_t *densex = static_cast<int8_t *>(
                    cpu::dbl::comm::try_gen_dil_tensor(dx).get_data_handle());
                // get scale
                std::vector<std::vector<float>> scales_json = cpu::dbl::comm::get_int8_scales({dx}, /*uint8_used for output*/ false, ops_id);
                const float x_scale = cpu::dbl::comm::try_gen_dil_tensor(dx).get_scale()[0]; // dx.get_scale();
                std::vector<float> weight_scale;
                const float r_scale = scales_json[1][0];
                for (auto &ebd : emb) {
                    weight_scale.push_back(
                        cpu::dbl::comm::try_gen_dil_tensor(ebd).get_scale()[0]);
                }
                // setup size and create output
                size_t J = s + 1;
                const size_t ROW = Dim + (s + 1) * s / 2; // 128 + 27 * 26/2
                size_t total_len = Batch * ROW; // 510000*479

                dil::dims dst_dims{Batch, ROW};
                dil::tensor::desc dst_desc(dst_dims, dil::data_type::s8);
                dil::tensor output{dst_desc};
                output.set_scale(scales_json[1]);

                int8_t * res = static_cast<int8_t *>(output.get_data_handle());

                int ind1[351] __attribute__((aligned(64))) = {
                    1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6,
                    7, 7, 7, 7, 7, 7, 7, 8, 8, 8, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9,
                    9, 9, 9, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 11, 11, 11, 11,
                    11, 11, 11, 11, 11, 11, 11, 12, 12, 12, 12, 12, 12, 12, 12, 12,
                    12, 12, 12, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13,
                    14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 15, 15,
                    15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 16, 16, 16,
                    16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 17, 17, 17,
                    17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 18, 18,
                    18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18,
                    19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19,
                    19, 19, 19, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20,
                    20, 20, 20, 20, 20, 20, 20, 21, 21, 21, 21, 21, 21, 21, 21, 21,
                    21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 22, 22, 22, 22,
                    22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22,
                    22, 22, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23,
                    23, 23, 23, 23, 23, 23, 23, 23, 23, 24, 24, 24, 24, 24, 24, 24,
                    24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24,
                    24, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25,
                    25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 26, 26, 26, 26, 26, 26,
                    26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26,
                    26, 26, 26, 26};
                int ind2[351] __attribute__((aligned(64))) = {
                    0, 0, 1, 0, 1, 2, 0, 1, 2, 3, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 5,
                    0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5,
                    6, 7, 8, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7,
                    8, 9, 10, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 0, 1, 2, 3, 4,
                    5, 6, 7, 8, 9, 10, 11, 12, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
                    12, 13, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 0, 1,
                    2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0, 1, 2, 3, 4,
                    5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 0, 1, 2, 3, 4, 5, 6,
                    7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 0, 1, 2, 3, 4, 5, 6, 7,
                    8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 0, 1, 2, 3, 4, 5, 6,
                    7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 0, 1, 2, 3, 4,
                    5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 0, 1,
                    2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
                    20, 21, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
                    17, 18, 19, 20, 21, 22, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
                    13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 0, 1, 2, 3, 4, 5, 6,
                    7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
                    24, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17,
                    18, 19, 20, 21, 22, 23, 24, 25};

                const float c_scale = r_scale / x_scale;
                float scales[384] __attribute__((aligned(64)));

                for (int i = 0; i < 351; ++i) {
                    if (ind2[i] == 0)
                        scales[i] = 1.0f / weight_scale[ind1[i] - 1];
                    else
                        scales[i] = x_scale / (weight_scale[ind1[i] - 1] * weight_scale[ind2[i] - 1]);
                }
                for (int i = 351; i< 384; ++i) {
                    scales[i] = 0.0f;
                }

                __m512i cache_reg[27][4];
                int cache_idx[27] __attribute__((aligned(64))) = {
                    -1, -1, -1, -1, -1, -1, -1, -1,
                    -1, -1, -1, -1, -1, -1, -1, -1,
                    -1, -1, -1, -1, -1, -1, -1, -1,
                    -1, -1, -1};

#pragma omp parallel for schedule(guided, 1024) default(none) firstprivate(ind1, ind2, scales, c_scale, Batch, cache_reg, cache_idx) shared(densex, index, offset, weight, res)
                for (int i = 0; i < Batch; ++i) {
                    // dot product of each pair
                    __m512 onehalf = _mm512_broadcastss_ps(_mm_set_ss(0.5));
                    if (abs(c_scale - 1.0f) < 0.003) {
                        __m512i l = _mm512_load_si512(densex + i * Dim);
                        __m512i r = _mm512_load_si512(densex + i * Dim + 64);
                        _mm512_storeu_si512(res + i * ROW, l);
                        _mm512_storeu_si512(res + i * ROW + 64, r);
                    } else {
                        __m512 scale = _mm512_set1_ps(c_scale);
                        __m512i x_0_8i = _mm512_load_si512(densex + i * Dim);
                        __m512i x_1_8i = _mm512_load_si512(densex + i * Dim);
                        __mmask16 mask = 0xffff;
                        __m512 x_0_f32 = _mm512_cvt_roundepi32_ps(
                            _mm512_cvtepi8_epi32(_mm_load_si128((__m128i *)(densex + i * Dim))),
                            (_MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC));
                        __m512 x_1_f32 = _mm512_cvt_roundepi32_ps(
                            _mm512_cvtepi8_epi32(_mm_load_si128((__m128i *)(densex + i * Dim + 16))),
                            (_MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC));
                        __m512 x_2_f32 = _mm512_cvt_roundepi32_ps(
                            _mm512_cvtepi8_epi32(_mm_load_si128((__m128i *)(densex + i * Dim + 32))),
                            (_MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC));
                        __m512 x_3_f32 = _mm512_cvt_roundepi32_ps(
                            _mm512_cvtepi8_epi32(_mm_load_si128((__m128i *)(densex + i * Dim + 48))),
                            (_MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC));
                        __m512 x_4_f32 = _mm512_cvt_roundepi32_ps(
                            _mm512_cvtepi8_epi32(_mm_load_si128((__m128i *)(densex + i * Dim + 64))),
                            (_MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC));
                        __m512 x_5_f32 = _mm512_cvt_roundepi32_ps(
                            _mm512_cvtepi8_epi32(_mm_load_si128((__m128i *)(densex + i * Dim + 80))),
                            (_MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC));
                        __m512 x_6_f32 = _mm512_cvt_roundepi32_ps(
                            _mm512_cvtepi8_epi32(_mm_load_si128((__m128i *)(densex + i * Dim + 96))),
                            (_MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC));
                        __m512 x_7_f32 = _mm512_cvt_roundepi32_ps(
                            _mm512_cvtepi8_epi32(_mm_load_si128((__m128i *)(densex + i * Dim + 112))),
                            (_MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC));
                        __m512i x_0_i32 = _mm512_cvt_roundps_epi32(
                            _mm512_mul_round_ps(x_0_f32, scale,
                                                (_MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC)),
                            (_MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC));
                        __m512i x_1_i32 = _mm512_cvt_roundps_epi32(
                            _mm512_mul_round_ps(x_1_f32, scale,
                                                (_MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC)),
                            (_MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC));
                        __m512i x_2_i32 = _mm512_cvt_roundps_epi32(
                            _mm512_mul_round_ps(x_2_f32, scale,
                                                (_MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC)),
                            (_MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC));
                        __m512i x_3_i32 = _mm512_cvt_roundps_epi32(
                            _mm512_mul_round_ps(x_3_f32, scale,
                                                (_MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC)),
                            (_MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC));
                        __m512i x_4_i32 = _mm512_cvt_roundps_epi32(
                            _mm512_mul_round_ps(x_4_f32, scale,
                                                (_MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC)),
                            (_MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC));
                        __m512i x_5_i32 = _mm512_cvt_roundps_epi32(
                            _mm512_mul_round_ps(x_5_f32, scale,
                                                (_MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC)),
                            (_MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC));
                        __m512i x_6_i32 = _mm512_cvt_roundps_epi32(
                            _mm512_mul_round_ps(x_6_f32, scale,
                                                (_MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC)),
                            (_MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC));
                        __m512i x_7_i32 = _mm512_cvt_roundps_epi32(
                            _mm512_mul_round_ps(x_7_f32, scale,
                                                (_MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC)),
                            (_MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC));
                        _mm512_mask_cvtsepi32_storeu_epi8(res + i * ROW, mask, x_0_i32);
                        _mm512_mask_cvtsepi32_storeu_epi8(res + i * ROW + 16, mask, x_1_i32);
                        _mm512_mask_cvtsepi32_storeu_epi8(res + i * ROW + 32, mask, x_2_i32);
                        _mm512_mask_cvtsepi32_storeu_epi8(res + i * ROW + 48, mask, x_3_i32);
                        _mm512_mask_cvtsepi32_storeu_epi8(res + i * ROW + 64, mask, x_4_i32);
                        _mm512_mask_cvtsepi32_storeu_epi8(res + i * ROW + 80, mask, x_5_i32);
                        _mm512_mask_cvtsepi32_storeu_epi8(res + i * ROW + 96, mask, x_6_i32);
                        _mm512_mask_cvtsepi32_storeu_epi8(res + i * ROW + 112, mask, x_7_i32);
                    }
                    for (int j = 0; j < 27; ++j) {
                        if (j == 0)
                            update_cache(0, i, &densex[i * Dim], cache_reg, cache_idx);
                        else {
                            int ii = index[j - 1][offset[j - 1][i]];
                            update_cache(j, ii, &(weight[j - 1][ii * Dim]),
                                         cache_reg, cache_idx);
                        }
                    }
                    int m = 0;
                    for (m = 0; m < 351 - 15; m += 16) {
                        dp8accum(0);
                        dp8accum(1);
                        dp8accum(2);
                        dp8accum(3);
                        dp8accum(4);
                        dp8accum(5);
                        dp8accum(6);
                        dp8accum(7);
                        dp8accum(8);
                        dp8accum(9);
                        dp8accum(10);
                        dp8accum(11);
                        dp8accum(12);
                        dp8accum(13);
                        dp8accum(14);
                        dp8accum(15);

                        __m512i itv0 = _mm512_unpacklo_epi32(out_0, out_1);
                        __m512i itv1 = _mm512_unpackhi_epi32(out_0, out_1);
                        __m512i itv2 = _mm512_unpacklo_epi32(out_2, out_3);
                        __m512i itv3 = _mm512_unpackhi_epi32(out_2, out_3);
                        __m512i itv4 = _mm512_unpacklo_epi32(out_4, out_5);
                        __m512i itv5 = _mm512_unpackhi_epi32(out_4, out_5);
                        __m512i itv6 = _mm512_unpacklo_epi32(out_6, out_7);
                        __m512i itv7 = _mm512_unpackhi_epi32(out_6, out_7);

                        itv0 = _mm512_add_epi32(itv0, itv1);
                        itv2 = _mm512_add_epi32(itv2, itv3);
                        itv4 = _mm512_add_epi32(itv4, itv5);
                        itv6 = _mm512_add_epi32(itv6, itv7);

                        itv1 = _mm512_unpacklo_epi64(itv0, itv2);
                        itv3 = _mm512_unpackhi_epi64(itv0, itv2);
                        itv5 = _mm512_unpacklo_epi64(itv4, itv6);
                        itv7 = _mm512_unpackhi_epi64(itv4, itv6);

                        itv1 = _mm512_add_epi32(itv1, itv3);
                        itv5 = _mm512_add_epi32(itv5, itv7);

                        itv0 = _mm512_shuffle_i32x4(itv1, itv5, 136);
                        itv2 = _mm512_shuffle_i32x4(itv1, itv5, 221);
                        itv0 = _mm512_add_epi32(itv0, itv2);

                        __m512i itv8 = _mm512_unpacklo_epi32(out_8, out_9);
                        __m512i itv9 = _mm512_unpackhi_epi32(out_8, out_9);
                        __m512i itva = _mm512_unpacklo_epi32(out_10, out_11);
                        __m512i itvb = _mm512_unpackhi_epi32(out_10, out_11);
                        __m512i itvc = _mm512_unpacklo_epi32(out_12, out_13);
                        __m512i itvd = _mm512_unpackhi_epi32(out_12, out_13);
                        __m512i itve = _mm512_unpacklo_epi32(out_14, out_15);
                        __m512i itvf = _mm512_unpackhi_epi32(out_14, out_15);

                        itv8 = _mm512_add_epi32(itv8, itv9);
                        itva = _mm512_add_epi32(itva, itvb);
                        itvc = _mm512_add_epi32(itvc, itvd);
                        itve = _mm512_add_epi32(itve, itvf);

                        itv9 = _mm512_unpacklo_epi64(itv8, itva);
                        itvb = _mm512_unpackhi_epi64(itv8, itva);
                        itvd = _mm512_unpacklo_epi64(itvc, itve);
                        itvf = _mm512_unpackhi_epi64(itvc, itve);

                        itv9 = _mm512_add_epi32(itv9, itvb);
                        itvd = _mm512_add_epi32(itvd, itvf);

                        itv8 = _mm512_shuffle_i32x4(itv9, itvd, 136);
                        itva = _mm512_shuffle_i32x4(itv9, itvd, 221);
                        itv8 = _mm512_add_epi32(itv8, itva);

                        itv1 = _mm512_shuffle_i32x4(itv0, itv8, 136);
                        itv2 = _mm512_shuffle_i32x4(itv0, itv8, 221);
                        __m512i resi32 = _mm512_add_epi32(itv1, itv2);
                        __m512 scale = _mm512_load_ps(&scales[m]);
                        __m512 resf32 = _mm512_cvtepi32_ps(resi32);

                        resf32 = _mm512_mul_ps(resf32, scale);
                        resi32 = _mm512_cvt_roundps_epi32(resf32, (_MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC));
                        __m128i resi8 = _mm512_cvtepi32_epi8(resi32);
                        _mm_storeu_si128((__m128i * )&res[i * ROW + Dim + m], resi8);
                    }
                    dp8accum(0);
                    dp8accum(1);
                    dp8accum(2);
                    dp8accum(3);
                    dp8accum(4);
                    dp8accum(5);
                    dp8accum(6);
                    dp8accum(7);
                    dp8accum(8);
                    dp8accum(9);
                    dp8accum(10);
                    dp8accum(11);
                    dp8accum(12);
                    dp8accum(13);
                    dp8accum(14);

                    __m512i out_15 = _mm512_setzero_si512();

                    __m512i itv0 = _mm512_unpacklo_epi32(out_0, out_1);
                    __m512i itv1 = _mm512_unpackhi_epi32(out_0, out_1);
                    __m512i itv2 = _mm512_unpacklo_epi32(out_2, out_3);
                    __m512i itv3 = _mm512_unpackhi_epi32(out_2, out_3);
                    __m512i itv4 = _mm512_unpacklo_epi32(out_4, out_5);
                    __m512i itv5 = _mm512_unpackhi_epi32(out_4, out_5);
                    __m512i itv6 = _mm512_unpacklo_epi32(out_6, out_7);
                    __m512i itv7 = _mm512_unpackhi_epi32(out_6, out_7);

                    itv0 = _mm512_add_epi32(itv0, itv1);
                    itv2 = _mm512_add_epi32(itv2, itv3);
                    itv4 = _mm512_add_epi32(itv4, itv5);
                    itv6 = _mm512_add_epi32(itv6, itv7);

                    itv1 = _mm512_unpacklo_epi64(itv0, itv2);
                    itv3 = _mm512_unpackhi_epi64(itv0, itv2);
                    itv5 = _mm512_unpacklo_epi64(itv4, itv6);
                    itv7 = _mm512_unpackhi_epi64(itv4, itv6);

                    itv1 = _mm512_add_epi32(itv1, itv3);
                    itv5 = _mm512_add_epi32(itv5, itv7);

                    itv0 = _mm512_shuffle_i32x4(itv1, itv5, 136);
                    itv2 = _mm512_shuffle_i32x4(itv1, itv5, 221);
                    itv0 = _mm512_add_epi32(itv0, itv2);

                    __m512i itv8 = _mm512_unpacklo_epi32(out_8, out_9);
                    __m512i itv9 = _mm512_unpackhi_epi32(out_8, out_9);
                    __m512i itva = _mm512_unpacklo_epi32(out_10, out_11);
                    __m512i itvb = _mm512_unpackhi_epi32(out_10, out_11);
                    __m512i itvc = _mm512_unpacklo_epi32(out_12, out_13);
                    __m512i itvd = _mm512_unpackhi_epi32(out_12, out_13);
                    __m512i itve = _mm512_unpacklo_epi32(out_14, out_15);
                    __m512i itvf = _mm512_unpackhi_epi32(out_14, out_15);

                    itv8 = _mm512_add_epi32(itv8, itv9);
                    itva = _mm512_add_epi32(itva, itvb);
                    itvc = _mm512_add_epi32(itvc, itvd);
                    itve = _mm512_add_epi32(itve, itvf);

                    itv9 = _mm512_unpacklo_epi64(itv8, itva);
                    itvb = _mm512_unpackhi_epi64(itv8, itva);
                    itvd = _mm512_unpacklo_epi64(itvc, itve);
                    itvf = _mm512_unpackhi_epi64(itvc, itve);

                    itv9 = _mm512_add_epi32(itv9, itvb);
                    itvd = _mm512_add_epi32(itvd, itvf);

                    itv8 = _mm512_shuffle_i32x4(itv9, itvd, 136);
                    itva = _mm512_shuffle_i32x4(itv9, itvd, 221);
                    itv8 = _mm512_add_epi32(itv8, itva);

                    itv1 = _mm512_shuffle_i32x4(itv0, itv8, 136);
                    itv2 = _mm512_shuffle_i32x4(itv0, itv8, 221);
                    __m512i resi32 = _mm512_add_epi32(itv1, itv2);

                    __m512 scale = _mm512_load_ps(&scales[m]);
                    __m512 resf32 = _mm512_cvtepi32_ps(resi32);
                    resf32 = _mm512_mul_ps(resf32, scale);
                    resi32 = _mm512_cvt_roundps_epi32(resf32, (_MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC));
                    __m128i resi8 = _mm512_cvtepi32_epi8(resi32);
                    _mm_mask_storeu_epi8(&res[i * ROW + Dim + m], 32767, resi8);
                }
                return output;
            }

            // template<typename T>
            // at::Tensor _fuseembint_forward(
            //     at::Tensor &lS_o,
            //     at::Tensor &lS_i,
            //     std::vector<at::Tensor> &emb,
            //     at::Tensor &densex,
            //     int64_t ops_id = -1) {
            //     std::vector<at::Tensor> input;
            //     input.push_back(densex);
            //     for (int i = 0; i < emb.size(); ++i) {
            //         input.push_back(
            //             cpu::aten::embedding_bag::embedding_bag_impl(emb[i], lS_i[i], lS_o[i], false, 0, false, emb[i], true));
            //     }
            //     at::Tensor out = _interaction_forward<T>(input);
            //     return out;
            // }

            at::Tensor _fuseembint_forward_int8(
                at::Tensor &lS_o,
                at::Tensor &lS_i,
                std::vector<at::Tensor> &emb,
                at::Tensor &densex,
                int64_t ops_id) {
                if (emb.size() == 26) {
                    if (emb[0].size(1) == 128) {
                        auto dil_output = fuse_emb_int_128_27(lS_o, lS_i, emb, densex, lS_i.size(1), ops_id);
                        auto output = cpu::dbl::comm::gen_aten_tensor_by(std::move(dil_output));
                        return output;
                    }
                }
                assert(false & "Only support 26 embeddings and emb size=128");
                at::Tensor out;
                return out;
                // cpu::aten::embedding_bag::embedding_bag_impl(emb[i], lS_i[i], lS_o[i], false, 0, false, emb[i], true);
                // return _interaction_forward<int8_t>(input);
            }

            at::Tensor fuseembint_forward(
                at::Tensor &lS_o,
                at::Tensor &lS_i,
                std::vector<at::Tensor> &emb,
                at::Tensor &densex) {
                if (check_auto_mix_int8_fp32() && !check_int8_calibration()) {
                    int64_t num_ops_id = Int8OptConfig::fetch_and_add_ops_id();
                    bool quantized = cpu::dbl::comm::get_int8_quantized_status(num_ops_id);
                    if (quantized) {
                        for (auto &w : emb) {
                            cpu::dbl::comm::reorder_to_int8_for_mix_prec(w, {}, false, {}, true);
                        }
                        return _fuseembint_forward_int8(lS_o, lS_i, emb, densex, num_ops_id);
                    }
                }
                // cpu::dbl::comm::reorder_to_dtype(densex, at::kFloat);
                // at::Tensor out = _fuseembint_forward<float>(lS_o, lS_i, emb, densex);
                // if (check_int8_calibration() && check_auto_mix_int8_fp32()) {
                //     insert_or_updata_observer({densex}, {out}, "fuseinteractionembedding", Int8OptConfig::fetch_and_add_ops_id());
                // }
                assert(false & "Only support int8");
                at::Tensor out;
                return out;
            }
        }
    }
}
