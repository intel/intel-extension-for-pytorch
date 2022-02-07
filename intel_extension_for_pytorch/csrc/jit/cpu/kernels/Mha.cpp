#include "Mha.h"
#include "AddSoftmax.h"
#include "Softmax.h"

#include <ATen/Context.h>
#include <ATen/InferSize.h>
#include <c10/util/Exception.h>
#include <c10/util/Logging.h>
#include <torch/csrc/autograd/function.h>

#include <limits>

#include "csrc/cpu/ideep/ideep.hpp"

namespace torch_ipex {
namespace cpu {

/**
 * We tried to fuse Div+Matmul+Add+Softmax as a signel operator. But
 * the oneDNN matmul performance with binary postop is poor, then we splited
 * the fusion into two parts - Div+Matmul and Add+Softmax. When the oneDNN
 * fixes the performance issue, we can directly leverage oneDNN's
 * implementation.
 **/
at::Tensor dil_mha_scores_calc(
    const at::Tensor& q,
    const at::Tensor& k,
    const at::Tensor& rel_kv,
    const at::Scalar& alpha,
    const at::Scalar& dim_per_head,
    const int64_t& softmax_dim,
    const at::IValue& dtype) {
#if defined(IPEX_PROFILE_OP)
  RECORD_FUNCTION("dil_mha_scores_calc", std::vector<c10::IValue>({}));
#endif
  auto _dim_per_head = dim_per_head.to<float>();
  auto _alpha = alpha.to<float>();
  auto qk = at::Tensor();

  auto q_dim = q.dim();
  auto k_dim = k.dim();
  qk = at::matmul(q, k);

  // Only support last dimension
  bool is_last_dim = (softmax_dim == -1);
  // Only support the non-last-dimension broadcast
  bool not_last_dim_broadcast = (rel_kv.size(rel_kv.ndimension() - 1) != 1);
  // Only support >=2D
  bool not_one_dim = q_dim >= 2;
  // Only support 64byte aligned
  bool aligned_64_bytes = rel_kv.size(rel_kv.ndimension() - 1) % 16 == 0;
  // Only support contiguous tensor
  bool is_contiguous = rel_kv.is_contiguous() && qk.is_contiguous();
  if (is_last_dim && not_last_dim_broadcast && not_one_dim &&
      aligned_64_bytes && is_contiguous && dtype.isNone() && _alpha == 1.0f) {
    return jit::cpu::kernels::DivAddSoftmax(qk, rel_kv, _dim_per_head);
  } else {
    qk = at::div(qk, dim_per_head);
    qk = at::add(qk, rel_kv, _alpha);
    return dil_softmax(qk, softmax_dim, dtype);
  }
}

} // namespace cpu
} // namespace torch_ipex
