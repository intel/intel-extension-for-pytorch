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

IPEX_DEFINE_DISPATCH(woq_fp32_gemm_kernel_stub);
IPEX_DEFINE_DISPATCH(woq_fp16_gemm_kernel_stub);
IPEX_DEFINE_DISPATCH(woq_bf16_gemm_kernel_stub);
IPEX_DEFINE_DISPATCH(woq_int8_gemm_pre_tensor_kernel_stub);
IPEX_DEFINE_DISPATCH(woq_int8_gemm_pre_k_block_kernel_stub);
IPEX_DEFINE_DISPATCH(woq_int8_gemm_pre_m_block_kernel_stub);
IPEX_DEFINE_DISPATCH(woq_int8_gemm_pre_m_k_block_kernel_stub);

/**
 * @brief Weight only quantization GEMM kernel. Here we dispatch to different
 * kernels according to the lowp_mode, the input dtype and input shape.
 *
 * @param x input activation in floating point format, 2D plain format [M,K]
 * @param qw quantized weight in 4D blocked format [Nc,Kc,Kb,Nb] or 2D plain
 * format [N,K].
 * @param scales_list a list of fp32/fp16/bf16 scales tensors
 * @param zp_list a list of fp32/fp16/bf16/int8 zero points tensors
 * @param bias_list a list of fp32/fp16/bf16 bias tensors
 * @param qw_type weight dtype, such as int8, int4, etc.
 * @param lowp_mode decide the compute dtype to use.
 *        LOWP_MODE_NONE: use activation dtype to compute
 *        LOWP_MODE_FP16: use FP16 to compute
 *        LOWP_MODE_BF16: use BF16 or FP16 to compute
 *        LOWP_MODE_INT8: use int8 to compute
 * @param fusion_type fusion type, such as gelu, add, etc.
 * @param others_list a list of other inputs for post ops, such as binary add,
 * etc.
 * @param quant_a_mode quantization mode for activation
 * @param quant_w_mode quantization mode for weight
 * @param quant_block_k block size for quantization
 * @param compensation a tensor for quantization compensation, for
 * LOWP_MODE_INT8 only. Used when activation is asymmetric quantized.
 * @return at::Tensor output in same dtype as `x`, 2D plain format [M,N]
 */
at::Tensor qlinear_woq_affine(
    const at::Tensor& x,
    const at::Tensor& qw,
    const TensorList& scales_list,
    const TensorList& zp_list,
    const TensorList& bias_list,
    const int qw_type,
    int64_t lowp_mode,
    int64_t fusion_type,
    const TensorList& others_list,
    int64_t quant_a_mode = -1,
    int64_t quant_w_mode = 0,
    int64_t quant_block_k = 0,
    const c10::optional<at::Tensor>& compensation = c10::nullopt,
    const c10::optional<at::Tensor>& g_idx = c10::nullopt) {
  auto K = x.size(-1);
  auto M = x.numel() / K;
  auto act_dtype = x.scalar_type();
  // Dispatch to different kernels by compute dtype
  if (lowp_mode == LOWP_MODE_FP16 ||
      (lowp_mode == LOWP_MODE_NONE && act_dtype == at::kHalf) ||
      (lowp_mode == LOWP_MODE_BF16 && M < SMALL_BATCH_THRESHOLD)) {
    return woq_fp16_gemm_kernel_stub(
        kCPU,
        x,
        qw,
        scales_list,
        zp_list,
        bias_list,
        qw_type,
        fusion_type,
        others_list,
        quant_w_mode,
        quant_block_k,
        g_idx);
  } else if (
      (lowp_mode == LOWP_MODE_NONE && act_dtype == at::kBFloat16) ||
      lowp_mode == LOWP_MODE_BF16 && M >= SMALL_BATCH_THRESHOLD) {
    return woq_bf16_gemm_kernel_stub(
        kCPU,
        x,
        qw,
        scales_list,
        zp_list,
        bias_list,
        qw_type,
        fusion_type,
        others_list,
        quant_w_mode,
        quant_block_k,
        g_idx);
  } else if (lowp_mode == LOWP_MODE_INT8) {
#define CALL_INT8_KERNEL(kernel) \
  kernel(                        \
      kCPU,                      \
      x,                         \
      qw,                        \
      scales_list,               \
      zp_list,                   \
      bias_list,                 \
      qw_type,                   \
      fusion_type,               \
      others_list,               \
      quant_a_mode,              \
      quant_w_mode,              \
      quant_block_k,             \
      compensation);

    if (quant_a_mode == QUANT_A_PER_TENSOR ||
        quant_a_mode == QUANT_A_PER_TENSOR_SYM) {
      return CALL_INT8_KERNEL(woq_int8_gemm_pre_tensor_kernel_stub);
    } else if (
        quant_a_mode == QUANT_A_PER_K_BLOCK ||
        quant_a_mode == QUANT_A_PER_K_BLOCK_SYM) {
      return CALL_INT8_KERNEL(woq_int8_gemm_pre_k_block_kernel_stub);
    } else if (
        quant_a_mode == QUANT_A_PER_M || quant_a_mode == QUANT_A_PER_M_SYM) {
      return CALL_INT8_KERNEL(woq_int8_gemm_pre_m_block_kernel_stub);
    } else if (
        quant_a_mode == QUANT_A_PER_M_K_BLOCK ||
        quant_a_mode == QUANT_A_PER_M_K_BLOCK_SYM) {
      return CALL_INT8_KERNEL(woq_int8_gemm_pre_m_k_block_kernel_stub);
    } else {
      TORCH_CHECK(false, "Unsupported quant_a_mode: ", quant_a_mode);
    }
  }
  TORCH_CHECK(
      lowp_mode == LOWP_MODE_NONE && act_dtype == at::kFloat,
      "Unsupported lowp_mode: ",
      lowp_mode,
      " with activation dtype: ",
      act_dtype);
  return woq_fp32_gemm_kernel_stub(
      kCPU,
      x,
      qw,
      scales_list,
      zp_list,
      bias_list,
      qw_type,
      fusion_type,
      others_list,
      quant_w_mode,
      quant_block_k,
      g_idx);
}

} // namespace

IPEX_REGISTER_DISPATCH(woq_tpp_gemm_kernel_stub, &qlinear_woq_affine);

} // namespace cpu
} // namespace torch_ipex

#endif
