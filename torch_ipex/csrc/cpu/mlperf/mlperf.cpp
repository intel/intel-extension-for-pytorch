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
            static inline __attribute__((always_inline))
            void convert_to_s16(const int ci, const int8_t *p, __m512i (&cache_reg)[27*4]) {
              cache_reg[ci * 4 + 0] = _mm512_cvtepi8_epi16(_mm256_load_si256((const __m256i *)p));
              cache_reg[ci * 4 + 1] = _mm512_cvtepi8_epi16(_mm256_load_si256((const __m256i *)(p + 32)));
              cache_reg[ci * 4 + 2] = _mm512_cvtepi8_epi16(_mm256_load_si256((const __m256i *)(p + 64)));
              cache_reg[ci * 4 + 3] = _mm512_cvtepi8_epi16(_mm256_load_si256((const __m256i *)(p + 96)));
            }

            static inline __attribute__((always_inline))
            void update_cache(const int ci, const int i, const int8_t *p,
                              __m512i (&cache_reg)[27*4], int (&cache_idx)[27]) {
                if (cache_idx[ci] != i) {
                    cache_idx[ci] = i;
		    convert_to_s16(ci, p, cache_reg);
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
                const int8_t *densex = static_cast<const int8_t *>(dx_dil_tensor.get_data_handle());
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

                int8_t * res = static_cast<int8_t *>(output.get_data_handle());
                at::parallel_for(0, Batch, 0, [&](int64_t start, int64_t end) {
                  __m512i cat_buf[352] __attribute__((aligned(64)));
                  cat_buf[351] = _mm512_setzero_si512();
                  __m512i convert_to_s16_buf[27*4] __attribute__((aligned(64)));
                  int cache_idx[27] __attribute__((aligned(64))) = {
                      -1, -1, -1, -1, -1, -1, -1, -1,
                      -1, -1, -1, -1, -1, -1, -1, -1,
                      -1, -1, -1, -1, -1, -1, -1, -1,
                      -1, -1, -1};
		  __m512 scale_m512[4];
                  for (int i = start; i < end; ++i) {
                    const int8_t* input0_ptr = densex + i * Dim;
                    int8_t* output0_ptr = res + i * ROW;
                    scale_and_move_ker_128(output0_ptr, input0_ptr, c_scale);
                    convert_to_s16(0, input0_ptr, convert_to_s16_buf);

                    int ii = index[0][offset[0][i]];
                    const int8_t *p = &(weight[0][ii * Dim]);
                    for (int j = 2; j < 27; ++j) {
                      update_cache(j - 1, ii, p, convert_to_s16_buf, cache_idx);
                      ii = index[j - 1][offset[j - 1][i]];
                      p = &(weight[j - 1][ii * Dim]);
                    }
                    update_cache(26, ii, p, convert_to_s16_buf, cache_idx);

                    // dot product of each pair
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

                    int8_t* outp = output0_ptr + Dim;
                    //Do reduce add with scale
                    size_t off = 0;
                    for (; off < total_off - 63 ; off += 64) {
		      scale_m512[0] = _mm512_load_ps((const void *)(scales + off));
		      scale_m512[1] = _mm512_load_ps((const void *)(scales + off + 16));
		      scale_m512[2] = _mm512_load_ps((const void *)(scales + off + 32));
		      scale_m512[3] = _mm512_load_ps((const void *)(scales + off + 48));
                      reduce_add_s32x16x16x4_with_scales(outp + off, cat_buf + off, scale_m512);
                    }

		    scale_m512[0] = _mm512_load_ps((const void *)(scales + off));
                    reduce_add_s32x16x16_with_scales(outp + off, cat_buf + off, scale_m512[0]);
                    off += 16;
		    scale_m512[0] = _mm512_load_ps((const void *)(scales + off));
                    reduce_add_s32x16x16_with_scales_and_mask_store(outp + off, 0x7fff, cat_buf + off, scale_m512[0]);
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
