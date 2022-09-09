#include "Mha.h"
#include "Matmul.h"
#include "Softmax.h"
#include "aten/AddSoftmax.h"
#include "aten/DivSoftmax.h"

#include <ATen/Context.h>
#include <ATen/InferSize.h>
#include <c10/util/Exception.h>
#include <c10/util/Logging.h>
#include <torch/csrc/autograd/function.h>

#include <limits>

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
  RECORD_FUNCTION("dil_mha_scores_calc", c10::ArrayRef<c10::IValue>({}));

  auto _dim_per_head = dim_per_head.to<float>();
  auto _alpha = alpha.to<float>();
  auto qk = at::Tensor();

  auto q_dim = q.dim();
  auto k_dim = k.dim();
  qk = bmm_impl(q, k, qk, ideep::attr_t(), {}, 1.f);

  // Only support last dimension
  bool is_last_dim = (softmax_dim == -1);
  // Only support the non-last-dimension broadcast
  bool not_last_dim_broadcast = (rel_kv.size(rel_kv.ndimension() - 1) != 1);
  // Only support >=2D
  bool not_one_dim = q_dim >= 2;
  // Only support contiguous tensor
  bool is_contiguous = rel_kv.is_contiguous() && qk.is_contiguous();
  if (is_last_dim && not_last_dim_broadcast && not_one_dim && is_contiguous &&
      dtype.isNone() && _alpha == 1.0f) {
    return DivAddSoftmax(qk, rel_kv, _dim_per_head);
  } else {
    qk = at::div(qk, dim_per_head);
    qk = at::add(qk, rel_kv, _alpha);
    return dil_softmax(qk, softmax_dim, dtype);
  }
}
/**
 * For BF16/FP32 path, We split the distil mha fusion into two parts - Matmul
 * and Div+Maskedfill+Softmax.
 * We do input checkings at graph rewrite time,
 * so we assume here:
 * Only support last dimension for softmax
 * Only support contiguous tensor for qk and mask
 * Only support qk.dim >=2D
 * Only support 64byte aligned
 * Only support when expand from the mid dims shape (bs :: seq_length)
 * Also checking the dtype as None
 **/
at::Tensor dil_distil_mha_scores_calc(
    const at::Tensor& q,
    const at::Tensor& k,
    const at::Tensor& mask_qk,
    const at::IntArrayRef& mask_qk_reshp,
    const at::Scalar& fill,
    const at::Scalar& dim_per_head) {
  RECORD_FUNCTION("dil_distil_mha_scores_calc", c10::ArrayRef<c10::IValue>({}));
  auto _dim_per_head = dim_per_head.to<float>();
  auto _fill = fill.to<float>();
  auto qk = at::Tensor();
  qk = bmm_impl(q, k, qk, ideep::attr_t(), {}, 1.f);
  //  convert the mask to float for creating vec mask for kernel computation
  auto _mask_qk = mask_qk.toType(at::kFloat);
  return DivMaskedfillSoftmax(
      qk, _mask_qk, mask_qk_reshp, _fill, _dim_per_head);
}

/**
 * For INT8 path, since matmul and div would be handled by LLGA fusion group,
 * We have to handle the rest fusion - Maskedfill+Softmax.
 * We also do the same input checkings at graph rewrite time like mentioned in
 * above overload function notes.
 **/
at::Tensor dil_maskedfill_softmax(
    at::Tensor& qk,
    const at::Tensor& mask_qk,
    const at::IntArrayRef& mask_qk_reshp,
    const at::Scalar& fill) {
  RECORD_FUNCTION("dil_maskedfill_softmax", c10::ArrayRef<c10::IValue>({}));
  float _dim_per_head = 1;
  auto _fill = fill.to<float>();
  // convert the mask to float for creating vec mask for kernel computation
  auto _mask_qk = mask_qk.toType(at::kFloat);
  return DivMaskedfillSoftmax(
      qk, _mask_qk, mask_qk_reshp, _fill, _dim_per_head);
}

at::Tensor dil_transfree_mha(
    const at::Tensor& qkv,
    const at::Tensor& rel_kv,
    const at::Scalar& alpha,
    const at::Scalar& dim_per_head,
    const int64_t& softmax_dim,
    const at::IValue& dtype,
    const int64_t& head_num,
    const int64_t& head_size) {
  RECORD_FUNCTION("dil_transfree_mha", c10::ArrayRef<c10::IValue>({}));

  auto _dim_per_head = dim_per_head.to<float>();
  auto _alpha = alpha.to<float>();
  int64_t batchSize = qkv.dim() > 2 ? qkv.size(0) : 1;
  int64_t sequenceSize = qkv.dim() > 2 ? qkv.size(1) : qkv.size(0);
  int64_t hiddenSize = head_num * head_size;
  at::Tensor qk =
      at::empty({batchSize, head_num, sequenceSize, sequenceSize}, qkv.dtype());

  // Currently, oneDNN Matmul primitive has some limitations to enable AMX
  // instructions. One critical condition is that the input tensor A should meet
  // the following requirement: A.size(0) * A.stride(0) == A.numel(), which
  // means the input tensor should not be a part of the other bigger tensor.
  // Since the Query, Key, and Value matrices are horizontally connected due to
  // the ConcatLinear optimization, they do not meet the above condition.
  // Thus they should be split before sending into the Matmul OP to enalbe AMX.
  auto qkv_mat = (qkv.dtype() == at::kFloat) ? dil_qkv_split<float>(qkv)
                                             : dil_qkv_split<at::BFloat16>(qkv);
  auto query = std::get<0>(qkv_mat);
  auto key = std::get<1>(qkv_mat);
  auto value = std::get<2>(qkv_mat);
  query.resize_({batchSize, sequenceSize, head_num, head_size})
      .transpose_(1, 2);
  key.resize_({batchSize, sequenceSize, head_num, head_size})
      .transpose_(1, 2)
      .transpose_(2, 3);
  value.resize_({batchSize, sequenceSize, head_num, head_size})
      .transpose_(1, 2);

  bmm_impl(query, key, qk, ideep::attr_t(), {}, 1.f);
  if (dtype.isNone() && _alpha == 1.0f) {
    qk = DivAddSoftmax(qk, rel_kv, _dim_per_head);
  } else {
    qk = at::div(qk, dim_per_head);
    qk = at::add(qk, rel_kv, _alpha);
    qk = dil_softmax(qk, softmax_dim, dtype);
  }

  auto output = dil_mha_matmul_trans(qk, value);

  return output;
}

at::Tensor dil_transfree_distil_mha(
    const at::Tensor& qkv,
    const at::Tensor& mask_qk,
    const at::IntArrayRef& mask_qk_reshp,
    const at::Scalar& fill,
    const at::Scalar& dim_per_head,
    const int64_t& head_num,
    const int64_t& head_size) {
  RECORD_FUNCTION("dil_distil_transfree_mha", c10::ArrayRef<c10::IValue>({}));

  auto _fill = fill.to<float>();
  auto _dim_per_head = dim_per_head.to<float>();
  int64_t batchSize = qkv.dim() > 2 ? qkv.size(0) : 1;
  int64_t sequenceSize = qkv.dim() > 2 ? qkv.size(1) : qkv.size(0);
  int64_t hiddenSize = head_num * head_size;
  at::Tensor qk =
      at::empty({batchSize, head_num, sequenceSize, sequenceSize}, qkv.dtype());

  auto qkv_mat = (qkv.dtype() == at::kFloat) ? dil_qkv_split<float>(qkv)
                                             : dil_qkv_split<at::BFloat16>(qkv);
  auto query = std::get<0>(qkv_mat);
  auto key = std::get<1>(qkv_mat);
  auto value = std::get<2>(qkv_mat);
  query.resize_({batchSize, sequenceSize, head_num, head_size})
      .transpose_(1, 2);
  key.resize_({batchSize, sequenceSize, head_num, head_size})
      .transpose_(1, 2)
      .transpose_(2, 3);
  value.resize_({batchSize, sequenceSize, head_num, head_size})
      .transpose_(1, 2);

  bmm_impl(query, key, qk, ideep::attr_t(), {}, 1.f);
  auto _mask_qk = mask_qk.toType(at::kFloat);
  qk = DivMaskedfillSoftmax(qk, _mask_qk, mask_qk_reshp, _fill, _dim_per_head);

  auto output = dil_mha_matmul_trans(qk, value);

  return output;
}

at::Tensor dil_transfree_vit_mha(
    const at::Tensor& qkv,
    const at::Tensor& dim_per_head,
    const int64_t& softmax_dim,
    const at::IValue& dtype,
    const int64_t& head_num,
    const int64_t& head_size) {
  auto scale = dim_per_head.data_ptr<float>()[0];
  return dil_transfree_vit_mha(
      qkv, scale, softmax_dim, dtype, head_num, head_size);
}

at::Tensor dil_transfree_vit_mha(
    const at::Tensor& qkv,
    const double& dim_per_head,
    const int64_t& softmax_dim,
    const at::IValue& dtype,
    const int64_t& head_num,
    const int64_t& head_size) {
  RECORD_FUNCTION("dil_transfree_vit_mha", c10::ArrayRef<c10::IValue>({}));

  int64_t batchSize = qkv.dim() > 2 ? qkv.size(0) : 1;
  int64_t sequenceSize = qkv.dim() > 2 ? qkv.size(1) : qkv.size(0);
  int64_t hiddenSize = head_num * head_size;
  at::Tensor qk =
      at::empty({batchSize, head_num, sequenceSize, sequenceSize}, qkv.dtype());

  auto qkv_mat = (qkv.dtype() == at::kFloat) ? dil_qkv_split<float>(qkv)
                                             : dil_qkv_split<at::BFloat16>(qkv);
  auto query = std::get<0>(qkv_mat);
  auto key = std::get<1>(qkv_mat);
  auto value = std::get<2>(qkv_mat);
  query.resize_({batchSize, sequenceSize, head_num, head_size})
      .transpose_(1, 2);
  key.resize_({batchSize, sequenceSize, head_num, head_size})
      .transpose_(1, 2)
      .transpose_(2, 3);
  value.resize_({batchSize, sequenceSize, head_num, head_size})
      .transpose_(1, 2);

  bmm_impl(query, key, qk, ideep::attr_t(), {}, 1.f / dim_per_head);
  qk = dil_softmax_(qk, softmax_dim, dtype);

  auto output = dil_mha_matmul_trans(qk, value);

  return output;
}

/**
 * This BMM OP is designed for the second Batched-Matmul of MHA as
 * the output is fused with transpose OP.
 * All the tensors should be 4-dim and the transpose indices of the
 * output tensor be (1, 2).
 * If the tensors do not meet the above conditions, the performance
 * of bmm_impl may drop when it uses the DNNL Matmul primitive
 * as its backend.
 */
at::Tensor dil_mha_matmul_trans(
    const at::Tensor& tensor1,
    const at::Tensor& tensor2) {
  RECORD_FUNCTION("dil_mha_bmm", c10::ArrayRef<c10::IValue>({}));

  std::vector<int64_t> output_size = {
      tensor1.size(0), tensor1.size(2), tensor1.size(1), tensor2.size(-1)};

  auto out = at::empty(output_size, tensor1.options()).transpose(1, 2);
  out = bmm_impl(tensor1, tensor2, out, ideep::attr_t(), {}, 1.f)
            .transpose_(1, 2);

  return out;
}

template <typename T>
std::tuple<at::Tensor, at::Tensor, at::Tensor> dil_qkv_split(
    const at::Tensor& qkv) {
  int64_t batchSize = qkv.dim() > 2 ? qkv.size(0) : 1;
  int64_t sequenceSize = qkv.dim() > 2 ? qkv.size(1) : qkv.size(0);
  int64_t hiddenSize = (qkv.dim() > 2 ? qkv.size(2) : qkv.size(1)) / 3;

  at::Tensor query =
      at::empty({batchSize, sequenceSize, hiddenSize}, qkv.dtype());
  at::Tensor key =
      at::empty({batchSize, sequenceSize, hiddenSize}, qkv.dtype());
  at::Tensor value =
      at::empty({batchSize, sequenceSize, hiddenSize}, qkv.dtype());

  T* src = qkv.data_ptr<T>();
#ifdef _OPENMP
#if (_OPENMP >= 201307)
#pragma omp parallel for simd schedule( \
    static) if (omp_get_max_threads() > 1 && !omp_in_parallel())
#else
#pragma omp parallel for schedule( \
    static) if (omp_get_max_threads() > 1 && !omp_in_parallel())
#endif
#endif
  for (int i = 0; i < batchSize * sequenceSize; ++i) {
    memcpy(
        query.data_ptr<T>() + i * hiddenSize,
        src + i * hiddenSize * 3,
        sizeof(T) * hiddenSize);
    memcpy(
        key.data_ptr<T>() + i * hiddenSize,
        src + i * hiddenSize * 3 + hiddenSize,
        sizeof(T) * hiddenSize);
    memcpy(
        value.data_ptr<T>() + i * hiddenSize,
        src + i * hiddenSize * 3 + 2 * hiddenSize,
        sizeof(T) * hiddenSize);
  }

  return std::make_tuple(query, key, value);
}
} // namespace cpu
} // namespace torch_ipex
