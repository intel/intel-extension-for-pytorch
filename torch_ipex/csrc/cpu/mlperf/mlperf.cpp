#include <immintrin.h>
#include "cpu/aten/aten.hpp"
#include "cpu/dil/dil.hpp"
#include "cpu/dbl/Common.h"
#include "cpu/int8/Config.h"
#include "cpu/ExtendOPs.h"
#include "utils.h"
#include "cpu/interaction.h"

namespace torch_ipex {
    namespace mlperf {
        namespace dlrm {
            static inline void update_cache(const int ci, const int i,
                                            const int8_t *p, __m512i (&cache_reg)[27*4]) {
              cache_reg[ci * 4 + 0] = _mm512_cvtepi8_epi16(_mm256_load_si256((const __m256i *)p));
              cache_reg[ci * 4 + 1] = _mm512_cvtepi8_epi16(_mm256_load_si256((const __m256i *)(p + 32)));
              cache_reg[ci * 4 + 2] = _mm512_cvtepi8_epi16(_mm256_load_si256((const __m256i *)(p + 64)));
              cache_reg[ci * 4 + 3] = _mm512_cvtepi8_epi16(_mm256_load_si256((const __m256i *)(p + 96)));
            }

            static inline void update_cache(const int ci, const int i,
                                            const int8_t *p, __m512i (&cache_reg)[27*4],
                                            int (&cache_idx)[27]) {
                if (cache_idx[ci] != i) {
                    cache_idx[ci] = i;
                    cache_reg[ci * 4 + 0] = _mm512_cvtepi8_epi16(_mm256_load_si256((const __m256i *)p));
                    cache_reg[ci * 4 + 1] = _mm512_cvtepi8_epi16(_mm256_load_si256((const __m256i *)(p + 32)));
                    cache_reg[ci * 4 + 2] = _mm512_cvtepi8_epi16(_mm256_load_si256((const __m256i *)(p + 64)));
                    cache_reg[ci * 4 + 3] = _mm512_cvtepi8_epi16(_mm256_load_si256((const __m256i *)(p + 96)));
                }
	    }

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
                std::vector<float> in_scales(27);
                for (int i = 0; i < emb.size(); ++i) {
		    auto emb_dil_tensor = cpu::dbl::comm::try_gen_dil_tensor(emb[i]);
                    weight[i] = static_cast<int8_t *>(emb_dil_tensor.get_data_handle());
                    in_scales[i + 1] = emb_dil_tensor.get_scale()[0];
                    index[i] = lS_i.data_ptr<int64_t>() + i * Batch;
                    offset[i] = lS_o.data_ptr<int64_t>() + i * Batch;
                }

		auto dx_dil_tensor = cpu::dbl::comm::try_gen_dil_tensor(dx);
                int8_t *densex = static_cast<int8_t *>(dx_dil_tensor.get_data_handle());
                // get scale
                std::vector<std::vector<float>> scales_json = cpu::dbl::comm::get_int8_scales({dx}, /*uint8_used for output*/ false, ops_id);
                in_scales[0] = dx_dil_tensor.get_scale()[0]; // dx.get_scale();
                const float r_scale = scales_json[1][0];
                const float c_scale = r_scale / in_scales[0];

                // setup size and create output
                size_t J = s + 1;
                const size_t ROW = Dim + (s + 1) * s / 2; // 128 + 27 * 26/2

                dil::dims dst_dims{Batch, ROW};
                dil::tensor::desc dst_desc(dst_dims, dil::data_type::s8);
                dil::tensor output{dst_desc};
                output.set_scale(scales_json[1]);

                int8_t * res = static_cast<int8_t *>(output.get_data_handle());

                float scales[352] __attribute__((aligned(64)));
                size_t off = 0;
                for (int i = 1; i < 27; i++) {
                  for (int j = 0; j < i; j++) {
                    auto input_scale = in_scales[i] * in_scales[j];
                    scales[off] = r_scale / input_scale;
                    off++;
                  }
                }
                scales[351] = 0.0f;

                at::parallel_for(0, Batch, 0, [&](int64_t start, int64_t end) {
                  __m512i cat_buf[352] __attribute__((aligned(64)));
                  __m512i convert_to_s16_buf[27*4] __attribute__((aligned(64)));
                  int cache_idx[27] __attribute__((aligned(64))) = {
                      -1, -1, -1, -1, -1, -1, -1, -1,
                      -1, -1, -1, -1, -1, -1, -1, -1,
                      -1, -1, -1, -1, -1, -1, -1, -1,
                      -1, -1, -1};
                  for (int i = start; i < end; ++i) {
                    int8_t* input0_ptr = densex + i * Dim;
                    int8_t* output0_ptr = res + i * ROW;
                    scale_and_move_ker_128(output0_ptr, input0_ptr, c_scale);

                    // dot product of each pair
                    update_cache(0, i, input0_ptr, convert_to_s16_buf);
                    for (int j = 1; j < 27; ++j) {
                      int ii = index[j - 1][offset[j - 1][i]];
                      update_cache(j, ii, &(weight[j - 1][ii * Dim]), convert_to_s16_buf, cache_idx);
                    }

                    auto * a = (const __m512i *)&convert_to_s16_buf[0];
                    auto * b = (const __m512i *)&convert_to_s16_buf[4];
                    mul_and_sum_s16x128_to_s32x16(cat_buf[0], b, a);
                    size_t total_off = 1;
                    for (int i = 2; i < 27; i++) {
                      auto * c = (const __m512i *)&convert_to_s16_buf[i * 4];
                      int j = 0;
	              for (; j < i - 1; j += 2) {
		        a = (const __m512i *)&convert_to_s16_buf[j * 4];
		        b = (const __m512i *)&convert_to_s16_buf[j * 4 + 4];
	                mul_and_sum_s16x128x2_to_s32x16x2(cat_buf[total_off], cat_buf[total_off + 1], c, a, c, b);
                        total_off += 2;
                      }
	              for (; j < i; j++) {
	                a = (const __m512i *)&convert_to_s16_buf[j * 4];
                        mul_and_sum_s16x128_to_s32x16(cat_buf[total_off], c, a);
                        total_off++;
                      }
                    }

                    int8_t* output_ptr = output0_ptr + Dim;
                    //Do reduce add with scale
                    size_t off = 0;
                    for (; off < total_off - 15 ; off += 16) {
                      __m512 scale_16 = _mm512_load_ps((const void *)(scales + off));
                      reduce_add_s32x16x16_with_scales(output_ptr + off, cat_buf + off, scale_16);
                    }

                    __m512i itv0 = _mm512_unpacklo_epi32(cat_buf[off + 0], cat_buf[off + 1]);
                    __m512i itv1 = _mm512_unpackhi_epi32(cat_buf[off + 0], cat_buf[off + 1]);
                    __m512i itv2 = _mm512_unpacklo_epi32(cat_buf[off + 2], cat_buf[off + 3]);
                    __m512i itv3 = _mm512_unpackhi_epi32(cat_buf[off + 2], cat_buf[off + 3]);
                    __m512i itv4 = _mm512_unpacklo_epi32(cat_buf[off + 4], cat_buf[off + 5]);
                    __m512i itv5 = _mm512_unpackhi_epi32(cat_buf[off + 4], cat_buf[off + 5]);
                    __m512i itv6 = _mm512_unpacklo_epi32(cat_buf[off + 6], cat_buf[off + 7]);
                    __m512i itv7 = _mm512_unpackhi_epi32(cat_buf[off + 6], cat_buf[off + 7]);

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

                    __m512i out_15 = _mm512_setzero_si512();

                    __m512i itv8 = _mm512_unpacklo_epi32(cat_buf[off + 8], cat_buf[off + 9]);
                    __m512i itv9 = _mm512_unpackhi_epi32(cat_buf[off + 8], cat_buf[off + 9]);
                    __m512i itva = _mm512_unpacklo_epi32(cat_buf[off + 10], cat_buf[off + 11]);
                    __m512i itvb = _mm512_unpackhi_epi32(cat_buf[off + 10], cat_buf[off + 11]);
                    __m512i itvc = _mm512_unpacklo_epi32(cat_buf[off + 12], cat_buf[off + 13]);
                    __m512i itvd = _mm512_unpackhi_epi32(cat_buf[off + 12], cat_buf[off + 13]);
                    __m512i itve = _mm512_unpacklo_epi32(cat_buf[off + 14], out_15);
                    __m512i itvf = _mm512_unpackhi_epi32(cat_buf[off + 14], out_15);

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

                    __m512 scale = _mm512_load_ps(&scales[off]);
                    __m512 resf32 = _mm512_cvtepi32_ps(resi32);
                    resf32 = _mm512_mul_ps(resf32, scale);
                    resi32 = _mm512_cvt_roundps_epi32(resf32, (_MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC));
                    __m128i resi8 = _mm512_cvtepi32_epi8(resi32);
                    _mm_mask_storeu_epi8(output_ptr + off, 32767, resi8);
                  }
		});
                return output;
            }

            template<typename T>
            at::Tensor _fuseembint_forward(
                at::Tensor &lS_o,
                at::Tensor &lS_i,
                std::vector<at::Tensor> &emb,
                at::Tensor &densex,
                int64_t ops_id = -1) {
                std::vector<at::Tensor> input;
                assert(false & "Only support float and int8");
                at::Tensor out;
                return out;
            }

            template<>
            at::Tensor _fuseembint_forward<float>(
                at::Tensor &lS_o,
                at::Tensor &lS_i,
                std::vector<at::Tensor> &emb,
                at::Tensor &densex,
                int64_t ops_id) {
                std::vector<at::Tensor> input;
                input.push_back(densex);
                for (int i = 0; i < emb.size(); ++i) {
                    auto embo = cpu::aten::embedding_bag::_embedding_bag_index_add_select_fast<float>(lS_i[i], emb[i], lS_o[i], false);
                    input.push_back(embo);
                }
                at::Tensor out = _interaction_forward<float>(input);
                return out;
            }

            template<>
            at::Tensor _fuseembint_forward<int8_t>(
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
                        return _fuseembint_forward<int8_t>(lS_o, lS_i, emb, densex, num_ops_id);
                    }
                }
                cpu::dbl::comm::reorder_to_dtype(densex, at::kFloat);
                at::Tensor out = _fuseembint_forward<float>(lS_o, lS_i, emb, densex);
                if (check_int8_calibration() && check_auto_mix_int8_fp32()) {
                    insert_or_updata_observer({densex}, {out}, "fuseinteractionembedding", Int8OptConfig::fetch_and_add_ops_id());
                }
                return out;
            }
        }
    }
}
