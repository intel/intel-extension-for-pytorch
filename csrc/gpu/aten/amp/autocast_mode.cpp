#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <ATen/autocast_mode.h>
#include <torch/library.h>

#include <c10/core/impl/LocalDispatchKeySet.h>
#include <c10/util/intrusive_ptr.h>

#include <exception>
#include <iostream>

namespace at {
namespace autocast {

#define KERNEL_XPU(FUNC, REGISTER_NAME, SIGNATURE, POLICY) \
  m.impl(                                                  \
      TORCH_SELECTIVE_NAME("aten::" REGISTER_NAME),        \
      &WrapFunction<                                       \
          CastPolicy::POLICY,                              \
          DeviceType::XPU,                                 \
          SIGNATURE,                                       \
          SIGNATURE,                                       \
          &FUNC>::type::call);

// Less-common but still useful case: redispatching to a function with a new
// signature (e.g. appending a dtype)
#define KERNEL_XPU_DIFFERENT_REDISPATCH_SIGNATURE(  \
    REDISPATCH_FUNC,                                \
    REGISTER_NAME,                                  \
    REGISTER_SIGNATURE,                             \
    REDISPATCH_SIGNATURE,                           \
    POLICY)                                         \
  m.impl(                                           \
      TORCH_SELECTIVE_NAME("aten::" REGISTER_NAME), \
      &WrapFunction<                                \
          CastPolicy::POLICY,                       \
          DeviceType::XPU,                          \
          REGISTER_SIGNATURE,                       \
          REDISPATCH_SIGNATURE,                     \
          &REDISPATCH_FUNC>::type::call);

/*****************************************
Explicit registration for out-of-place ops
*****************************************/
TORCH_LIBRARY_IMPL(_, AutocastXPU, m) {
  m.fallback(torch::CppFunction::makeFallthrough());
}

TORCH_LIBRARY_IMPL(aten, AutocastXPU, m) {
  // lower_precision_fp cast policy
  KERNEL_XPU(
      ADD_NS(_convolution),
      "_convolution.deprecated",
      Tensor(
          const Tensor&,
          const Tensor&,
          const c10::optional<Tensor>&,
          IntArrayRef,
          IntArrayRef,
          IntArrayRef,
          bool,
          IntArrayRef,
          int64_t,
          bool,
          bool,
          bool),
      lower_precision_fp)
  KERNEL_XPU(
      ADD_NS(_convolution),
      "_convolution",
      Tensor(
          const Tensor&,
          const Tensor&,
          const c10::optional<Tensor>&,
          IntArrayRef,
          IntArrayRef,
          IntArrayRef,
          bool,
          IntArrayRef,
          int64_t,
          bool,
          bool,
          bool,
          bool),
      lower_precision_fp)
  KERNEL_XPU(
      ADD_NS(conv1d),
      "conv1d",
      Tensor(
          const Tensor&,
          const Tensor&,
          const c10::optional<Tensor>&,
          IntArrayRef,
          IntArrayRef,
          IntArrayRef,
          int64_t),
      lower_precision_fp)
  KERNEL_XPU(
      ADD_NS(conv2d),
      "conv2d",
      Tensor(
          const Tensor&,
          const Tensor&,
          const c10::optional<Tensor>&,
          IntArrayRef,
          IntArrayRef,
          IntArrayRef,
          int64_t),
      lower_precision_fp)
  KERNEL_XPU(
      ADD_NS(conv3d),
      "conv3d",
      Tensor(
          const Tensor&,
          const Tensor&,
          const c10::optional<Tensor>&,
          IntArrayRef,
          IntArrayRef,
          IntArrayRef,
          int64_t),
      lower_precision_fp)
  KERNEL_XPU(
      ADD_NS(conv_tbc),
      "conv_tbc",
      Tensor(const Tensor&, const Tensor&, const Tensor&, int64_t),
      lower_precision_fp)
  KERNEL_XPU(
      ADD_NS(conv_transpose1d),
      "conv_transpose1d",
      Tensor(
          const Tensor&,
          const Tensor&,
          const c10::optional<Tensor>&,
          IntArrayRef,
          IntArrayRef,
          IntArrayRef,
          int64_t,
          IntArrayRef),
      lower_precision_fp)
  KERNEL_XPU(
      ADD_NS(conv_transpose2d),
      "conv_transpose2d.input",
      Tensor(
          const Tensor&,
          const Tensor&,
          const c10::optional<Tensor>&,
          IntArrayRef,
          IntArrayRef,
          IntArrayRef,
          int64_t,
          IntArrayRef),
      lower_precision_fp)
  KERNEL_XPU(
      ADD_NS(conv_transpose3d),
      "conv_transpose3d.input",
      Tensor(
          const Tensor&,
          const Tensor&,
          const c10::optional<Tensor>&,
          IntArrayRef,
          IntArrayRef,
          IntArrayRef,
          int64_t,
          IntArrayRef),
      lower_precision_fp)
  KERNEL_XPU(
      ADD_NS(convolution),
      "convolution",
      Tensor(
          const Tensor&,
          const Tensor&,
          const c10::optional<Tensor>&,
          IntArrayRef,
          IntArrayRef,
          IntArrayRef,
          bool,
          IntArrayRef,
          int64_t),
      lower_precision_fp)
  KERNEL_XPU(
      ADD_NS(prelu),
      "prelu",
      Tensor(const Tensor&, const Tensor&),
      lower_precision_fp)
  KERNEL_XPU(
      ADD_NS(addmm),
      "addmm",
      Tensor(
          const Tensor&,
          const Tensor&,
          const Tensor&,
          const Scalar&,
          const Scalar&),
      lower_precision_fp)
  KERNEL_XPU(
      ADD_NS(addmv),
      "addmv",
      Tensor(
          const Tensor&,
          const Tensor&,
          const Tensor&,
          const Scalar&,
          const Scalar&),
      lower_precision_fp)
  KERNEL_XPU(
      ADD_NS(addr),
      "addr",
      Tensor(
          const Tensor&,
          const Tensor&,
          const Tensor&,
          const Scalar&,
          const Scalar&),
      lower_precision_fp)
  KERNEL_XPU(
      ADD_NS(matmul),
      "matmul",
      Tensor(const Tensor&, const Tensor&),
      lower_precision_fp)
  KERNEL_XPU(
      ADD_NS(mm),
      "mm",
      Tensor(const Tensor&, const Tensor&),
      lower_precision_fp)
  KERNEL_XPU(
      ADD_NS(mv),
      "mv",
      Tensor(const Tensor&, const Tensor&),
      lower_precision_fp)
  KERNEL_XPU(
      ADD_NS(linear),
      "linear",
      Tensor(const Tensor&, const Tensor&, const c10::optional<Tensor>&),
      lower_precision_fp)
  KERNEL_XPU(
      ADD_NS(addbmm),
      "addbmm",
      Tensor(
          const Tensor&,
          const Tensor&,
          const Tensor&,
          const Scalar&,
          const Scalar&),
      lower_precision_fp)
  KERNEL_XPU(
      ADD_NS(baddbmm),
      "baddbmm",
      Tensor(
          const Tensor&,
          const Tensor&,
          const Tensor&,
          const Scalar&,
          const Scalar&),
      lower_precision_fp)
  KERNEL_XPU(
      ADD_NS(bmm),
      "bmm",
      Tensor(const Tensor&, const Tensor&),
      lower_precision_fp)
  KERNEL_XPU(
      ADD_NS(chain_matmul),
      "chain_matmul",
      Tensor(TensorList),
      lower_precision_fp)
  KERNEL_XPU(
      ADD_NS(linalg_multi_dot),
      "linalg_multi_dot",
      Tensor(TensorList),
      lower_precision_fp)
  KERNEL_XPU(
      ADD_NS(scaled_dot_product_attention),
      "scaled_dot_product_attention",
      Tensor(
          const Tensor&,
          const Tensor&,
          const Tensor&,
          const c10::optional<Tensor>&,
          double,
          bool,
          c10::optional<double>),
      lower_precision_fp)
  // The macro doesn't like these (I think it chokes on commas inside <>) so
  // write them manually
  m.impl(
      "_thnn_fused_gru_cell",
      TORCH_FN((&WrapFunction<
                CastPolicy::lower_precision_fp,
                DeviceType::XPU,
                std::tuple<Tensor, Tensor>(
                    const Tensor&,
                    const Tensor&,
                    const Tensor&,
                    const c10::optional<Tensor>&,
                    const c10::optional<Tensor>&),
                std::tuple<Tensor, Tensor>(
                    const Tensor&,
                    const Tensor&,
                    const Tensor&,
                    const c10::optional<Tensor>&,
                    const c10::optional<Tensor>&),
                &ADD_NS(_thnn_fused_gru_cell)>::type::call)));
  m.impl(
      "gru_cell",
      TORCH_FN((&WrapFunction<
                CastPolicy::lower_precision_fp,
                DeviceType::XPU,
                Tensor(
                    const Tensor&,
                    const Tensor&,
                    const Tensor&,
                    const Tensor&,
                    const c10::optional<Tensor>&,
                    const c10::optional<Tensor>&),
                Tensor(
                    const Tensor&,
                    const Tensor&,
                    const Tensor&,
                    const Tensor&,
                    const c10::optional<Tensor>&,
                    const c10::optional<Tensor>&),
                &ADD_NS(gru_cell)>::type::call)));
  // fp32
  KERNEL_XPU(
      ADD_NS(binary_cross_entropy),
      "binary_cross_entropy",
      Tensor(
          const Tensor&, const Tensor&, const c10::optional<Tensor>&, int64_t),
      fp32)
  KERNEL_XPU(
      ADD_NS(binary_cross_entropy_with_logits),
      "binary_cross_entropy_with_logits",
      Tensor(
          const Tensor&,
          const Tensor&,
          const c10::optional<Tensor>&,
          const c10::optional<Tensor>&,
          int64_t),
      fp32)
  KERNEL_XPU(
      ADD_NS(cross_entropy_loss),
      "cross_entropy_loss",
      Tensor(
          const Tensor&,
          const Tensor&,
          const c10::optional<Tensor>&,
          int64_t,
          int64_t,
          double),
      fp32)
  KERNEL_XPU(
      ADD_NS(log_softmax),
      "log_softmax.int",
      Tensor(const Tensor&, int64_t, c10::optional<ScalarType>),
      fp32)
  KERNEL_XPU(
      ADD_NS(log_softmax),
      "log_softmax.Dimname",
      Tensor(const Tensor&, Dimname, c10::optional<ScalarType>),
      fp32)
  KERNEL_XPU(
      ADD_NS(nll_loss),
      "nll_loss",
      Tensor(
          const Tensor&,
          const Tensor&,
          const c10::optional<Tensor>&,
          int64_t,
          int64_t),
      fp32)
  KERNEL_XPU(
      ADD_NS(nll_loss2d),
      "nll_loss2d",
      Tensor(
          const Tensor&,
          const Tensor&,
          const c10::optional<Tensor>&,
          int64_t,
          int64_t),
      fp32)
  KERNEL_XPU(
      ADD_NS(nll_loss_nd),
      "nll_loss_nd",
      Tensor(
          const Tensor&,
          const Tensor&,
          const c10::optional<Tensor>&,
          int64_t,
          int64_t),
      fp32)
  KERNEL_XPU(ADD_NS(reciprocal), "reciprocal", Tensor(const Tensor&), fp32)
  KERNEL_XPU(
      ADD_NS(pow),
      "pow.Tensor_Scalar",
      Tensor(const Tensor&, const Scalar&),
      fp32)
  KERNEL_XPU(
      ADD_NS(pow),
      "pow.Tensor_Tensor",
      Tensor(const Tensor&, const Tensor&),
      fp32)
  KERNEL_XPU(
      ADD_NS(pow), "pow.Scalar", Tensor(const Scalar&, const Tensor&), fp32)
  KERNEL_XPU(
      ADD_NS(frobenius_norm),
      "frobenius_norm.dim",
      Tensor(const Tensor&, IntArrayRef, bool),
      fp32)
  KERNEL_XPU(
      ADD_NS(nuclear_norm), "nuclear_norm", Tensor(const Tensor&, bool), fp32)
  KERNEL_XPU(
      ADD_NS(nuclear_norm),
      "nuclear_norm.dim",
      Tensor(const Tensor&, IntArrayRef, bool),
      fp32)
  KERNEL_XPU(
      ADD_NS(cosine_similarity),
      "cosine_similarity",
      Tensor(const Tensor&, const Tensor&, int64_t, double),
      fp32)
  KERNEL_XPU(
      ADD_NS(poisson_nll_loss),
      "poisson_nll_loss",
      Tensor(const Tensor&, const Tensor&, bool, bool, double, int64_t),
      fp32)
  KERNEL_XPU(
      ADD_NS(cosine_embedding_loss),
      "cosine_embedding_loss",
      Tensor(const Tensor&, const Tensor&, const Tensor&, double, int64_t),
      fp32)
  KERNEL_XPU(
      ADD_NS(hinge_embedding_loss),
      "hinge_embedding_loss",
      Tensor(const Tensor&, const Tensor&, double, int64_t),
      fp32)
  KERNEL_XPU(
      ADD_NS(kl_div),
      "kl_div",
      Tensor(const Tensor&, const Tensor&, int64_t, bool),
      fp32)
  KERNEL_XPU(
      ADD_NS(l1_loss),
      "l1_loss",
      Tensor(const Tensor&, const Tensor&, int64_t),
      fp32)
  KERNEL_XPU(
      ADD_NS(smooth_l1_loss),
      "smooth_l1_loss",
      Tensor(const Tensor&, const Tensor&, int64_t, double),
      fp32)
  KERNEL_XPU(
      ADD_NS(huber_loss),
      "huber_loss",
      Tensor(const Tensor&, const Tensor&, int64_t, double),
      fp32)
  KERNEL_XPU(
      ADD_NS(mse_loss),
      "mse_loss",
      Tensor(const Tensor&, const Tensor&, int64_t),
      fp32)
  KERNEL_XPU(
      ADD_NS(margin_ranking_loss),
      "margin_ranking_loss",
      Tensor(const Tensor&, const Tensor&, const Tensor&, double, int64_t),
      fp32)
  KERNEL_XPU(
      ADD_NS(multilabel_margin_loss),
      "multilabel_margin_loss",
      Tensor(const Tensor&, const Tensor&, int64_t),
      fp32)
  KERNEL_XPU(
      ADD_NS(soft_margin_loss),
      "soft_margin_loss",
      Tensor(const Tensor&, const Tensor&, int64_t),
      fp32)
  KERNEL_XPU(
      ADD_NS(triplet_margin_loss),
      "triplet_margin_loss",
      Tensor(
          const Tensor&,
          const Tensor&,
          const Tensor&,
          double,
          double,
          double,
          bool,
          int64_t),
      fp32)
  KERNEL_XPU(
      ADD_NS(multi_margin_loss),
      "multi_margin_loss",
      Tensor(
          const Tensor&,
          const Tensor&,
          const Scalar&,
          const Scalar&,
          const c10::optional<Tensor>&,
          int64_t),
      fp32)
  KERNEL_XPU(
      ADD_NS(dist),
      "dist",
      Tensor(const Tensor&, const Tensor&, const Scalar&),
      fp32)
  KERNEL_XPU(ADD_NS(pdist), "pdist", Tensor(const Tensor&, double), fp32)
  KERNEL_XPU(
      ADD_NS(cdist),
      "cdist",
      Tensor(const Tensor&, const Tensor&, double, c10::optional<int64_t>),
      fp32)
  KERNEL_XPU(
      ADD_NS(renorm),
      "renorm",
      Tensor(const Tensor&, const Scalar&, int64_t, const Scalar&),
      fp32)
  KERNEL_XPU(
      ADD_NS(fft_ifft),
      "fft_ifft",
      Tensor(
          const Tensor&,
          c10::optional<int64_t>,
          int64_t,
          c10::optional<c10::string_view>),
      fp32)
  KERNEL_XPU(
      ADD_NS(fft_fft2),
      "fft_fft2",
      Tensor(
          const Tensor&,
          at::OptionalIntArrayRef,
          at::IntArrayRef,
          c10::optional<c10::string_view>),
      fp32)
  KERNEL_XPU(
      ADD_NS(fft_ifft2),
      "fft_ifft2",
      Tensor(
          const Tensor&,
          at::OptionalIntArrayRef,
          at::IntArrayRef,
          c10::optional<c10::string_view>),
      fp32)
  KERNEL_XPU(
      ADD_NS(fft_fftn),
      "fft_fftn",
      Tensor(
          const Tensor&,
          at::OptionalIntArrayRef,
          at::OptionalIntArrayRef,
          c10::optional<c10::string_view>),
      fp32)
  KERNEL_XPU(
      ADD_NS(fft_ifftn),
      "fft_ifftn",
      Tensor(
          const Tensor&,
          at::OptionalIntArrayRef,
          at::OptionalIntArrayRef,
          c10::optional<c10::string_view>),
      fp32)
  KERNEL_XPU(
      ADD_NS(fft_rfft),
      "fft_rfft",
      Tensor(
          const Tensor&,
          c10::optional<int64_t>,
          int64_t,
          c10::optional<c10::string_view>),
      fp32)
  KERNEL_XPU(
      ADD_NS(fft_irfft),
      "fft_irfft",
      Tensor(
          const Tensor&,
          c10::optional<int64_t>,
          int64_t,
          c10::optional<c10::string_view>),
      fp32)
  KERNEL_XPU(
      ADD_NS(fft_rfft2),
      "fft_rfft2",
      Tensor(
          const Tensor&,
          at::OptionalIntArrayRef,
          at::IntArrayRef,
          c10::optional<c10::string_view>),
      fp32)
  KERNEL_XPU(
      ADD_NS(fft_irfft2),
      "fft_irfft2",
      Tensor(
          const Tensor&,
          at::OptionalIntArrayRef,
          at::IntArrayRef,
          c10::optional<c10::string_view>),
      fp32)
  KERNEL_XPU(
      ADD_NS(fft_rfftn),
      "fft_rfftn",
      Tensor(
          const Tensor&,
          at::OptionalIntArrayRef,
          at::OptionalIntArrayRef,
          c10::optional<c10::string_view>),
      fp32)
  KERNEL_XPU(
      ADD_NS(fft_irfftn),
      "fft_irfftn",
      Tensor(
          const Tensor&,
          at::OptionalIntArrayRef,
          at::OptionalIntArrayRef,
          c10::optional<c10::string_view>),
      fp32)
  KERNEL_XPU(
      ADD_NS(fft_hfft),
      "fft_hfft",
      Tensor(
          const Tensor&,
          c10::optional<int64_t>,
          int64_t,
          c10::optional<c10::string_view>),
      fp32)
  KERNEL_XPU(
      ADD_NS(fft_ihfft),
      "fft_ihfft",
      Tensor(
          const Tensor&,
          c10::optional<int64_t>,
          int64_t,
          c10::optional<c10::string_view>),
      fp32)
  // promote
  KERNEL_XPU(
      ADD_NS(cat), "cat", Tensor(const at::ITensorListRef&, int64_t), promote)
  KERNEL_XPU(ADD_NS(stack), "stack", Tensor(TensorList, int64_t), promote)
  KERNEL_XPU(
      ADD_NS(addcdiv),
      "addcdiv",
      Tensor(const Tensor&, const Tensor&, const Tensor&, const Scalar&),
      promote)
  KERNEL_XPU(
      ADD_NS(addcmul),
      "addcmul",
      Tensor(const Tensor&, const Tensor&, const Tensor&, const Scalar&),
      promote)
  KERNEL_XPU(
      ADD_NS(atan2), "atan2", Tensor(const Tensor&, const Tensor&), promote)
  KERNEL_XPU(
      ADD_NS(bilinear),
      "bilinear",
      Tensor(
          const Tensor&,
          const Tensor&,
          const Tensor&,
          const c10::optional<Tensor>&),
      promote)
  KERNEL_XPU(
      ADD_NS(cross),
      "cross",
      Tensor(const Tensor&, const Tensor&, c10::optional<int64_t>),
      promote)
  KERNEL_XPU(ADD_NS(dot), "dot", Tensor(const Tensor&, const Tensor&), promote)
  KERNEL_XPU(
      ADD_NS(grid_sampler),
      "grid_sampler",
      Tensor(const Tensor&, const Tensor&, int64_t, int64_t, bool),
      promote)
  KERNEL_XPU(
      ADD_NS(index_put),
      "index_put",
      Tensor(
          const Tensor&,
          const torch::List<c10::optional<Tensor>>&,
          const Tensor&,
          bool),
      promote)
  KERNEL_XPU(
      ADD_NS(tensordot),
      "tensordot",
      Tensor(const Tensor&, const Tensor&, IntArrayRef, IntArrayRef),
      promote)
  KERNEL_XPU(
      ADD_NS(scatter_add),
      "scatter_add",
      Tensor(const Tensor&, int64_t, const Tensor&, const Tensor&),
      promote)
}
} // namespace autocast
} // namespace at
