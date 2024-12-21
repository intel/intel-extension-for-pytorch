// weight-only quantization gemm kernel (int8, int4 etc.)
#ifdef USE_LIBXSMM
#include <ATen/ATen.h>
#include <ATen/Tensor.h>
#include <ATen/cpu/vec/functional.h>
#include <ATen/cpu/vec/vec.h>
#include <aten/Linear.h>
#include "aten/utils/woq.h"
#include "csrc/cpu/vec/vec.h"

#ifdef __GNUC__
#include <features.h>
#if __GNUC_PREREQ(12, 3)
#define COMPILER_PREREQ_MET
#endif
#endif

namespace torch_ipex {
namespace cpu {
namespace {

// Aligned with WoqTppKrnl.cpp
#if defined(CPU_CAPABILITY_AVX512_FP16) && defined(COMPILER_PREREQ_MET)

/**
 * @brief pack the weight in quantized format.
 * @param qw quantized weight with shape [N, K]
 * @param block_n block size along N, N % block_n == 0, block_n % 16 == 0
 * @param block_k block size along K, K % block_k == 0. block_k % 2 == 0 for
 * bf16 compute_dtype. false if activation is expected to be float32.
 */
at::Tensor qlinear_woq_pack(
    const at::Tensor& qw,
    int qw_type,
    size_t block_n,
    size_t block_k,
    int64_t lowp_mode) {
  TLA_ASSERT(qw.is_contiguous(), "qw must be contiguous");
  bool is_4bit_flag = is_4bit(qw_type);
  auto sizes = qw.sizes();
  auto N = sizes[0];
  auto K = is_4bit_flag ? sizes[1] * 2 : sizes[1];
  TLA_ASSERT(N % block_n == 0, "N must be multiple of block_n");
  TLA_ASSERT(K % block_k == 0, "K must be multiple of block_k");
  if (is_4bit_flag) {
    TLA_ASSERT(
        block_n % 16 == 0, "block_n must be multiple of 16 for 4bit weight");
  }
  if (lowp_mode == LOWP_MODE_INT8) {
    TLA_ASSERT(
        block_k % 4 == 0,
        "block_k must be multiple of 4 for int8 for LOWP_MODE_INT8");
  }
  const int N_GROUP_SIZE =
      lowp_mode != LOWP_MODE_INT8 ? get_n_group_size(block_n) : 16;
  const int Nc = N / block_n;
  const int Kc = K / block_k;
  if (is_4bit_flag) {
    // TODO(jgong5): support lowp_mode == LOWP_MODE_INT8
    auto result = at::empty({Nc, Kc, block_k, block_n / 2}, qw.options());
    // Pack weight in [N,K] to [N/block_n, K/block_k, block_k, block_n]
    // And then, pre-shuffle per 32 or 64 4-bit values to save shuffle at
    // runtime Take 32 4-bit values as an example below: x0 x1 x2 x3 x4 x5 x6 x7
    // x8 x9 x10 x11 x12 x13 x14 x15 y0 y1 y2 y3 y4 y5 y6 y7 y8 y9 y10 y11 y12
    // y13 y14 y15 becomes x0 y0 x1 y1 x2 y2 x3 y3 x4 y4 x5 y5 x6 y6 x7 y7 x8 y8
    // x9 y9 x10 y10 x11 y11 x12 y12 x13 y13 x14 y14 x15 y15 Here, xi and yj are
    // 4-bit values.
    uint8_t* src_data = (uint8_t*)qw.data_ptr();
    uint8_t* dst_data = (uint8_t*)result.data_ptr();
    auto psrc = GetVLAPtr<uint8_t>(src_data, {block_n, Kc, block_k / 2});
    auto pdst = GetVLAPtr<uint8_t>(dst_data, {Kc, block_k, block_n / 2});
    auto pdst_4vnni =
        GetVLAPtr<uint8_t>(dst_data, {Kc, block_k / 4, block_n / 2, 4});
    auto pack_loop =
        ThreadedLoop<3>({{Nc}, {Kc}, {0, block_n, N_GROUP_SIZE, false}}, "ABc");
    pack_loop([&](int* idx) {
      int nc = idx[0];
      int kc = idx[1];
      int nb = idx[2];
      for (int i = 0; i < N_GROUP_SIZE / 2; i++) {
        for (int kb = 0; kb < block_k; kb += 2) {
          auto src0 = psrc[nc][nb + i][kc][kb / 2];
          auto src1 = psrc[nc][nb + i + N_GROUP_SIZE / 2][kc][kb / 2];
          auto dst0 = (src0 & 0xf) | ((src1 & 0xf) << 4);
          auto dst1 = (src0 >> 4) | ((src1 >> 4) << 4);
          if (lowp_mode != LOWP_MODE_INT8) {
            pdst[nc][kc][kb][nb / 2 + i] = dst0;
            pdst[nc][kc][kb + 1][nb / 2 + i] = dst1;
          } else {
            pdst_4vnni[nc][kc][kb / 4][nb / 2 + i][kb % 4] = dst0;
            pdst_4vnni[nc][kc][(kb + 1) / 4][nb / 2 + i][(kb + 1) % 4] = dst1;
          }
        }
      }
    });
    return result;
  } else {
    if (lowp_mode == LOWP_MODE_INT8) {
      // pack weight o [Nc, Kc, Kb/4, Nb, 4] but view it as [Nc, Kc, Kb, Nb]
      auto packed_weight = qw.reshape({Nc, block_n, Kc, block_k / 4, 4})
                               .permute({0, 2, 3, 1, 4})
                               .contiguous();
      return packed_weight.view({Nc, Kc, block_k, block_n});
    }
    auto result = at::empty({Nc, Kc, block_k, block_n}, qw.options());
    // Pack weight in [N,K] to [N/block_n, K/block_k, block_k, block_n]
    int8_t* src_data = (int8_t*)qw.data_ptr();
    int8_t* dst_data = (int8_t*)result.data_ptr();
    auto psrc = GetVLAPtr<int8_t>(src_data, {block_n, Kc, block_k});
    auto pdst = GetVLAPtr<int8_t>(dst_data, {Kc, block_k, block_n});
    auto pack_loop =
        ThreadedLoop<3>({{Nc}, {Kc}, {0, block_n, N_GROUP_SIZE, false}}, "ABc");
    pack_loop([&](int* idx) {
      int nc = idx[0];
      int kc = idx[1];
      int nb = idx[2];
      for (int i = 0; i < N_GROUP_SIZE; i++) {
        for (int kb = 0; kb < block_k; kb++) {
          pdst[nc][kc][kb][nb + i] = psrc[nc][nb + i][kc][kb];
        }
      }
    });
    return result;
  }
}

at::Tensor qlinear_woq_unpack(
    const at::Tensor& qw_packed,
    int qw_type,
    int64_t lowp_mode) {
  bool is_4bit_flag = is_4bit(qw_type);
  if (qw_packed.dim() == 4) {
    auto w_sizes = qw_packed.sizes();
    auto Nc = w_sizes[0];
    auto Nb = is_4bit_flag ? w_sizes[3] * 2 : w_sizes[3];
    auto Kc = w_sizes[1];
    auto Kb = w_sizes[2];
    auto N = Nc * Nb;
    auto K = Kc * Kb;
    const int N_GROUP_SIZE =
        lowp_mode != LOWP_MODE_INT8 ? get_n_group_size(Nb) : 16;
    if (is_4bit_flag) {
      // TODO: support lowp_mode == 3
      auto result = at::empty({N, K / 2}, qw_packed.options());
      uint8_t* src_data = (uint8_t*)qw_packed.data_ptr();
      uint8_t* dst_data = (uint8_t*)result.data_ptr();
      auto psrc = GetVLAPtr<uint8_t>(src_data, {Kc, Kb, Nb / 2});
      auto psrc_4vnni = GetVLAPtr<uint8_t>(src_data, {Kc, Kb / 4, Nb / 2, 4});
      auto pdst = GetVLAPtr<uint8_t>(dst_data, {Nb, Kc, Kb / 2});
      auto unpack_loop =
          ThreadedLoop<3>({{Nc}, {Kc}, {0, Nb, N_GROUP_SIZE, false}}, "ABc");
      unpack_loop([&](int* idx) {
        int nc = idx[0];
        int kc = idx[1];
        int nb = idx[2];
        for (int kb = 0; kb < Kb; kb += 2) {
          for (int i = 0; i < N_GROUP_SIZE / 2; i++) {
            uint8_t src0, src1;
            if (lowp_mode != LOWP_MODE_INT8) {
              src0 = psrc[nc][kc][kb][nb / 2 + i];
              src1 = psrc[nc][kc][kb + 1][nb / 2 + i];
            } else {
              src0 = psrc_4vnni[nc][kc][kb / 4][nb / 2 + i][kb % 4];
              src1 = psrc_4vnni[nc][kc][(kb + 1) / 4][nb / 2 + i][(kb + 1) % 4];
            }
            pdst[nc][nb + i][kc][kb / 2] = (src0 & 0xf) | ((src1 & 0xf) << 4);
            pdst[nc][nb + i + N_GROUP_SIZE / 2][kc][kb / 2] =
                (src0 >> 4) | ((src1 >> 4) << 4);
          }
        }
      });
      return result;
    } else {
      TLA_ASSERT(
          lowp_mode != LOWP_MODE_INT8,
          "lowp mode int8 is not supported yet with int8 weight");
      auto result = at::empty({N, K}, qw_packed.options());
      int8_t* src_data = (int8_t*)qw_packed.data_ptr();
      int8_t* dst_data = (int8_t*)result.data_ptr();
      auto psrc = GetVLAPtr<int8_t>(src_data, {Kc, Kb, Nb});
      auto pdst = GetVLAPtr<int8_t>(dst_data, {Nb, Kc, Kb});
      auto unpack_loop =
          ThreadedLoop<3>({{Nc}, {Kc}, {0, Nb, N_GROUP_SIZE, false}}, "ABc");
      unpack_loop([&](int* idx) {
        int nc = idx[0];
        int kc = idx[1];
        int nb = idx[2];
        for (int kb = 0; kb < Kb; kb++) {
          for (int i = 0; i < N_GROUP_SIZE; i++) {
            pdst[nc][nb + i][kc][kb] = psrc[nc][kc][kb][nb + i];
          }
        }
      });
      return result;
    }
  } else {
    TLA_ASSERT(qw_packed.dim() == 2, "qw_packed must be 2D or 4D");
    return qw_packed;
  }
}

/*
Take an int4 weight in blocked format. Dequantize it and subtract compensation
per block. The returned tensor is still in block format. Compensation is also
returned through the argument.
*/
at::Tensor dequantize_int4_weight_to_int8_packed(
    const at::Tensor& qw_packed,
    const at::Tensor& scales,
    const at::Tensor& zps,
    int quant_block_k,
    int64_t quant_w_mode,
    at::Tensor& compensation) {
  auto w_sizes = qw_packed.sizes();
  auto Nc = w_sizes[0];
  auto Kc = w_sizes[1];
  auto Kb = w_sizes[2];
  auto Nb = w_sizes[3] * 2;
  auto N = Nc * Nb;
  auto K = Kc * Kb;
  TLA_ASSERT(
      compensation.sizes().vec() == std::vector<int64_t>({Nc, Kc, Nb}),
      "WOQ linear: compensation size mismatch");
  constexpr const long MICRO_BLOCK_M = 8;
  constexpr const int N_GROUP_SIZE = 16;
  auto quant_block_multiple = quant_block_k == 0 ? 1 : quant_block_k / Kb;
  auto quant_k_blocks =
      quant_block_k == 0 ? 1 : (K + quant_block_k - 1) / quant_block_k;
  int scales_kc = quant_w_mode == QUANT_W_PER_CHANNEL ||
          quant_w_mode == QUANT_W_PER_CHANNEL_SYM
      ? 1
      : quant_k_blocks;
  auto pw =
      GetVLAPtr<uint8_t>((uint8_t*)qw_packed.data_ptr(), {Kc, Kb * Nb / 2});
  auto pscales = GetVLAPtr<float>(scales, {scales_kc, Nb});
  torch::Tensor dqw = torch::empty({Nc, Kc, Kb, Nb}, c10::kChar);
  auto dqw_ptr = GetVLAPtr<int8_t>(dqw, {Kc, Kb * Nb});
  auto comp_ptr = GetVLAPtr<int32_t>(compensation, {Kc, Nb});

  product_dispatcher<
      std::tuple<long, long>,
      std::tuple<
          enumerate_dispatcher<long, WOQ_N_BLOCK_SIZE>,
          range_dispatcher<long, 0, 3>>>::
      call(
          std::make_tuple(Nb, quant_w_mode),
          [&](auto tuple) {
            auto block_n = std::get<0>(tuple);
            auto quant_w_mode = std::get<1>(tuple);
            constexpr bool asym_quant_w = is_asymmetric_quant_w(quant_w_mode);
            auto pzps = asym_quant_w ? GetVLAPtr<int8_t>(zps, {scales_kc, Nb})
                                     : GetVLAPtr<int8_t>(nullptr, {1, 1});
            auto loop_scheme = "bA";
            auto dequant_loop =
                ThreadedLoop<2>({{Nc}, {0, Kc, Kc}}, loop_scheme);
            dequant_loop(
                [&](int* idx) {
                  int nc = idx[0];
                  int kc_start = idx[1];
                  int kc_end = kc_start + Kc;
                  for (int kc = kc_start; kc < kc_end; kc++) {
                    int32_t quant_offset = kc / quant_block_multiple;
                    float* scale_w = nullptr;
                    int8_t* zp_w = nullptr;
                    if constexpr (
                        quant_w_mode == QUANT_W_PER_CHANNEL ||
                        quant_w_mode == QUANT_W_PER_CHANNEL_SYM) {
                      scale_w = pscales[nc][0];
                      if constexpr (asym_quant_w) {
                        zp_w = pzps[nc][0];
                      }
                    } else {
                      scale_w = pscales[nc][quant_offset];
                      if constexpr (asym_quant_w) {
                        zp_w = pzps[nc][quant_offset];
                      }
                    }
                    constexpr bool sym_quant_w =
                        !is_asymmetric_quant_w(quant_w_mode);
                    // quant_a_mode is used to determine whether compensation is
                    // needed or not Here we assume compensation is always
                    // needed.
                    constexpr long quant_a_mode = 0;
                    Dequantize<
                        int8_t,
                        block_n,
                        N_GROUP_SIZE,
                        /*qw_type*/ WOQ_DTYPE_INT4,
                        sym_quant_w,
                        /*use_g_idx*/ false>::
                        template call<quant_a_mode>(
                            pw[nc][kc],
                            Kb,
                            block_n,
                            zp_w,
                            dqw_ptr[nc][kc],
                            comp_ptr[nc][kc]);
                  }
                },
                [&]() {},
                [&]() {});
          },
          [](auto tuple) { failing_fallback(); });
  return dqw;
}

at::Tensor dequantize_nf4(
    const at::Tensor& t,
    const at::Tensor& scales,
    int64_t group_size,
    c10::ScalarType out_dtype) {
  TORCH_CHECK(
      t.dim() == 2,
      "dequantize_nf4 only supports 2D input, but got ",
      t.dim(),
      "D");
  TORCH_CHECK(
      t.scalar_type() == c10::kByte || t.scalar_type() == c10::kChar,
      "dequantize_nf4 only supports uint8 or int8 input, but got ",
      t.scalar_type());
  TORCH_CHECK(
      scales.dim() <= 2,
      "dequantize_nf4: scales must be 1D (per-channel) or 2D (per-group), but got ",
      scales.dim(),
      "D");
  TORCH_CHECK(
      out_dtype == c10::kFloat || out_dtype == c10::kBFloat16 ||
          out_dtype == c10::kHalf,
      "dequantize_nf4 only supports float, bfloat16 or float16 output, but got ",
      out_dtype);
  auto N = t.size(0);
  auto K = t.size(1) * 2;
  using Tcomp = float;
  constexpr auto VEC_LEN = sizeof(__m512i) / sizeof(Tcomp);
  if (K % VEC_LEN == 0 && (group_size >= VEC_LEN || group_size < 0)) {
    auto n_groups = K / group_size;
    torch::Tensor out = torch::empty({N, K}, out_dtype);
    product_dispatcher<
        std::tuple<c10::ScalarType, c10::ScalarType>,
        std::tuple<
            enumerate_dispatcher<
                c10::ScalarType,
                c10::kFloat,
                c10::kBFloat16,
                c10::kHalf>,
            enumerate_dispatcher<
                c10::ScalarType,
                c10::kFloat,
                c10::kBFloat16,
                c10::kHalf>>>::
        call(
            std::make_tuple(out_dtype, scales.scalar_type()),
            [&](auto tuple) {
              // Note: we don't use the `Dequantize` utilities because they are
              // designed for packed layout but we are handling plain layout
              // here.
              auto out_dtype_ = std::get<0>(tuple);
              auto scales_dtype_ = std::get<1>(tuple);
              using T =
                  typename c10::impl::ScalarTypeToCPPType<out_dtype_>::type;
              using Tscale =
                  typename c10::impl::ScalarTypeToCPPType<scales_dtype_>::type;
              using VT = typename VecType<Tcomp>::type;
              using V = VecOps<VT>;
              VT lut = V::set_nf4_lut();
              constexpr auto k_step = VEC_LEN / 2;
              auto pt = GetVLAPtr<uint8_t>(
                  (uint8_t*)t.data_ptr(), {t.size(1) / k_step, k_step});
              auto pscales = group_size < 0
                  ? GetVLAPtr<Tscale>(scales, {1})
                  : GetVLAPtr<Tscale>(scales, {n_groups});
              auto out_ptr = GetVLAPtr<T>(out, {K / VEC_LEN, VEC_LEN});

              auto dequant_loop = ThreadedLoop<2>({{N}}, /* loop_scheme */ "A");
              dequant_loop(
                  [&](int* idx) {
                    int n = idx[0];
                    for (int k = 0; k < t.size(1); k += k_step) {
                      // Load 64 bits of nf4 data and a single scale data
                      auto p = pt[n][k / k_step];
                      auto scale_idx = group_size < 0 ? 0 : k * 2 / group_size;
                      auto vscales = V::set1((float)pscales[n][scale_idx]);
                      uint64_t packed = reinterpret_cast<uint64_t*>(p)[0];
                      // unpack nf4 data to 32-bit integers
                      uint64_t high = 0;
                      uint64_t low = 0;
                      for (int i = 0; i < 8; ++i) {
                        low |= ((packed >> (i * 4)) & 0xf) << (i * 8);
                        high |= ((packed >> (i * 4 + 32)) & 0xf) << (i * 8);
                      }
                      __m128i packed_128 = _mm_set_epi64x(high, low);
                      __m512i vint32 = _mm512_cvtepu8_epi32(packed_128);
                      // Table look-up
                      __m512 vout = _mm512_permutexvar_ps(vint32, lut);
                      // Apply scale
                      vout = V::mul(vout, vscales);
                      // Store results
                      auto pout = out_ptr[n][k / k_step];
                      if constexpr (std::is_same<T, float>()) {
                        _mm512_storeu_ps(pout, vout);
                      } else if constexpr (std::is_same<T, at::BFloat16>()) {
                        _mm256_storeu_si256(
                            (__m256i*)pout, cvt_fp32_to_bf16(vout));
                      } else if constexpr (std::is_same<T, at::Half>()) {
                        _mm256_storeu_si256(
                            (__m256i*)pout, cvt_fp32_to_fp16(vout));
                      } else {
                        TORCH_CHECK(false, "Unexpected dtype");
                      }
                    }
                  },
                  [&]() {},
                  [&]() {});
            },
            [&](auto out_dtype_) { failing_fallback(); });
    return out;
  }
  auto out = dequantize_woq_weight(
      t, {N, K}, scales.unsqueeze(-1), at::Tensor(), WOQ_DTYPE_NF4, group_size);
  return out.to(out_dtype);
}

#else

at::Tensor qlinear_woq_pack(
    const at::Tensor& qw,
    int qw_type,
    size_t block_n,
    size_t block_k,
    int64_t lowp_mode) {
  return qw;
}

at::Tensor qlinear_woq_unpack(
    const at::Tensor& qw_packed,
    int qw_type,
    int64_t lowp_mode) {
  return qw_packed;
}

at::Tensor dequantize_int4_weight_to_int8_packed(
    const at::Tensor& qw_packed,
    const at::Tensor& scales,
    const at::Tensor& zps,
    int quant_block_k,
    int64_t quant_w_mode,
    at::Tensor& compensation) {
  return qw_packed;
}

at::Tensor dequantize_nf4(
    const at::Tensor& t,
    const at::Tensor& scales,
    int64_t group_size,
    c10::ScalarType out_dtype) {
  auto N = t.size(0);
  auto K = t.size(1) * 2;
  auto out = dequantize_woq_weight(
      t, {N, K}, scales.unsqueeze(-1), at::Tensor(), WOQ_DTYPE_NF4, group_size);
  return out.to(out_dtype);
}

#endif

} // namespace

IPEX_REGISTER_DISPATCH(woq_tpp_gemm_packB_stub, &qlinear_woq_pack);
IPEX_REGISTER_DISPATCH(woq_tpp_gemm_unpackB_stub, &qlinear_woq_unpack);
IPEX_REGISTER_DISPATCH(
    woq_dequant_int4_to_int8_packed_stub,
    &dequantize_int4_weight_to_int8_packed);
IPEX_REGISTER_DISPATCH(dequant_nf4_stub, &dequantize_nf4);

#endif

} // namespace cpu
} // namespace torch_ipex
