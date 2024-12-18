// weight-only quantization gemm kernel (int8, int4 etc.)
#ifdef USE_LIBXSMM
#include <ATen/ATen.h>
#include <ATen/Tensor.h>
#include <ATen/cpu/vec/functional.h>
#include <ATen/cpu/vec/vec.h>
#include "aten/utils/woq.h"

#ifdef __GNUC__
#include <features.h>
#if __GNUC_PREREQ(12, 3)
#define COMPILER_PREREQ_MET
#endif
#endif

namespace torch_ipex {
namespace cpu {
namespace {

using TensorList = std::vector<at::Tensor>;

// We only build optimized kernels if AVX512_FP16 is supported and gcc>=12.3
#if defined(CPU_CAPABILITY_AVX512_FP16) && defined(COMPILER_PREREQ_MET)

// We separate GEMM kernel in different files to avoid long compile time
/**
 * @brief quantized linear with quantized weight but activation in floating
 * point. Compute in int8. Activation is per-tensor quantized.
 *
 * @param x input activation in floating point format, 2D plain format [M,K]
 * @param qw weight in affine quantized format, could be 4-bit or 8-bit
 * quantized in 4D blocked format [Nc,Kc,Kb,Nb] or 2D plain format [N,K].
 * @param scales_list a list of fp32/fp16/bf16 scales tensors
 * @param zp_list a list of fp32/fp16/bf16/int8 zero points tensors
 * @param bias_list a list of fp32/fp16/bf16 bias tensors
 * @param qw_type weight dtype, such as int8, int4, etc.
 * @param fusion_type fusion type, such as gelu, add, etc.
 * @param others_list a list of other inputs for post ops, such as binary add,
 * etc.
 * @param quant_a_mode quantization mode for activation
 * @param quant_w_mode quantization mode for weight
 * @param quant_block_k block size for quantization
 * @param compensation a tensor for quantization compensation, used when
 * activation is asymmetric quantized
 * @return at::Tensor output in same dtype as `x`, 2D plain format [M,N]
 */
at::Tensor woq_gemm_int8(
    const at::Tensor& x,
    const at::Tensor& qw,
    const TensorList& scales_list,
    const TensorList& zp_list,
    const TensorList& bias_list,
    const int qw_type,
    int64_t fusion_type,
    const TensorList& others_list,
    int64_t quant_a_mode = -1,
    int64_t quant_w_mode = 0,
    int64_t quant_block_k = 0,
    const c10::optional<at::Tensor>& compensation = c10::nullopt) {
  const int64_t k_splits = 0;
  quant_block_k = std::max(0L, quant_block_k);
  // int8_idx is only valid with zp_list when lowp_mode == LOWP_MODE_INT8
  constexpr size_t fp32_idx = 0, fp16_idx = 1, bf16_idx = 2, int8_idx = 3;
  auto biases = bias_list.empty()
      ? TensorList({at::Tensor(), at::Tensor(), at::Tensor()})
      : bias_list;
  const bool is_4bit_flag = is_4bit(qw_type);
  const bool asym_quant_w = is_asymmetric_quant_w(quant_w_mode);
  if (qw_type == WOQ_DTYPE_NF4 || qw_type == WOQ_DTYPE_INT8) {
    TORCH_CHECK(
        !asym_quant_w,
        "WOQ: symmetric quantization is required for NF4 or INT8 with lowp-mode INT8");
  }
  if (qw.dim() == 4) {
    auto w_sizes = qw.sizes();
    auto K = x.size(-1);
    auto M = x.numel() / K;
    auto N = w_sizes[0] * w_sizes[3];
    if (is_4bit_flag && !compensation.has_value()) {
      N *= 2;
    }
    auto out_sizes = x.sizes().vec();
    out_sizes.back() = N;
    auto y = at::empty(out_sizes, x.options());
    product_dispatcher<
        std::tuple<at::ScalarType, long>,
        std::tuple<
            enumerate_dispatcher<
                at::ScalarType,
                at::kFloat,
                at::kBFloat16,
                at::kHalf>,
            range_dispatcher<long, 0, 3>>>::
        call(
            std::make_tuple(x.scalar_type(), quant_w_mode),
            [&](auto tuple) {
              auto act_dtype = std::get<0>(tuple);
              auto quant_w_mode_ = std::get<1>(tuple);
              using act_type =
                  typename c10::impl::ScalarTypeToCPPType<act_dtype>::type;
              float scale_a;
              int32_t zp_a;
              bool is_sym_quant = !is_asymmetric_quant_a(quant_a_mode);
              compute_int8_qparams_per_tensor<act_type>(
                  x, &scale_a, &zp_a, is_sym_quant);
              auto x_quantized =
                  quantize_per_tensor<act_type>(x, scale_a, zp_a, is_sym_quant);
              auto scale_a_ptr = &scale_a;
              int32_t* zp_a_ptr = is_sym_quant ? nullptr : &zp_a;
              if (quant_a_mode == QUANT_A_PER_TENSOR) {
                CALL_WOQ_KERNEL_IMPL_INT8(uint8_t, QUANT_A_PER_TENSOR);
              } else if (quant_a_mode == QUANT_A_PER_TENSOR_SYM) {
                CALL_WOQ_KERNEL_IMPL_INT8(int8_t, QUANT_A_PER_TENSOR_SYM);
              } else {
                TORCH_CHECK(
                    false,
                    "Unexpected quant_a_mode for lowp-mode INT8 per-tensor quant: ",
                    quant_a_mode);
              }
            },
            [](auto tuple) { failing_fallback(); });
    return y;
  } else {
    return woq_gemm_ref_impl(
        x,
        qw,
        scales_list,
        zp_list,
        bias_list,
        qw_type,
        x.scalar_type(),
        fusion_type,
        others_list,
        quant_w_mode,
        quant_block_k);
  }
}

#else // defined(CPU_CAPABILITY_AVX512_FP16) && defined(COMPILER_PREREQ_MET)

at::Tensor woq_gemm_int8(
    const at::Tensor& x,
    const at::Tensor& qw,
    const TensorList& scales_list,
    const TensorList& zp_list,
    const TensorList& bias_list,
    const int qw_type,
    int64_t fusion_type,
    const TensorList& others_list,
    int64_t quant_a_mode = -1,
    int64_t quant_w_mode = 0,
    int64_t quant_block_k = 0,
    const c10::optional<at::Tensor>& compensation = c10::nullopt) {
  return woq_gemm_ref_impl(
      x,
      qw,
      scales_list,
      zp_list,
      bias_list,
      qw_type,
      x.scalar_type(),
      fusion_type,
      others_list,
      quant_w_mode,
      quant_block_k);
}

#endif // defined(CPU_CAPABILITY_AVX512_FP16) && defined(COMPILER_PREREQ_MET)

} // namespace

IPEX_REGISTER_DISPATCH(woq_int8_gemm_pre_tensor_kernel_stub, &woq_gemm_int8);

} // namespace cpu
} // namespace torch_ipex

#endif
