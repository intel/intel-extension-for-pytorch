/*******************************************************************************
 * Copyright (C) 2025 Intel Corporation
 * This software and the related documents are Intel copyrighted materials,
 * and your use of them is governed by the express license under which they
 * were provided to you ("License"). Unless the License provides otherwise,
 * you may not use, modify, copy, publish, distribute, disclose or transmit
 * this software or the related documents without Intel's prior written
 * permission. This software and the related documents are provided as is,
 * with no express or implied warranties, other than those that are expressly
 * stated in the License.
 *******************************************************************************
 */
#include <ATen/WrapDimUtilsMulti.h>
#include <ATen/native/Resize.h>
#include "../BlasImpl.h"
#include "Cast.h"
#include "utils/CustomOperatorRegistration.h"

#include <ATen/ATen.h>
#include <ATen/Config.h>
#include <ATen/NativeFunctions.h>

#include <oneDNN/FP8_Matmul.h>
#include <quantized/QUtils.h>
#include <runtime/Utils.h>
#include "../comm/ParamUtils.h"

namespace at {
namespace AtenIpexTypeXPU {

Tensor fp8_gemm(
    const Tensor& m1_ /*act*/,
    int64_t in_format,
    int64_t input_meta_index,
    const Tensor& m2_ /*wei*/,
    int64_t wei_format,
    int64_t weight_meta_index,
    const Tensor& m3 /*bias*/,
    const Tensor& scale,
    const Tensor& scale_inv,
    const Tensor& amax_history) {
  auto result = at::linear(m1_, m2_, m3.to(m1_.scalar_type()));
  return result;
}

Tensor fp8_gemm_v2(
    const Tensor& A,
    bool trans_A,
    const Tensor& B,
    bool trans_B,
    const c10::optional<Tensor>& D,
    ScalarType out_dtype,
    const c10::optional<Tensor>& A_scale_inv_,
    const c10::optional<Tensor>& B_scale_inv_,
    const c10::optional<Tensor>& bias_,
    bool accumulate) {
  std::vector<int64_t> result_shape;
  if (A.dim() == 2) {
    if (trans_A) {
      A.transpose_(0, 1);
    }
    if (trans_B) {
      B.transpose_(0, 1);
    }
    // src{m, k}, wei{k, n}, bias{n}, dst{m, n}
    result_shape = {A.size(0), B.size(1)};
  } else if (A.dim() == 3) {
    if (trans_A) {
      A.transpose_(1, 2);
    }
    if (B.dim() == 2) {
      if (trans_B) {
        B.transpose_(0, 1);
      }
      // src{b, m, k}, wei{k, n}, bias{n}, dst{b, m, n}
      result_shape = {A.size(0) * A.size(1), B.size(1)};
    } else {
      if (trans_B) {
        B.transpose_(1, 2);
      }
      // src{b, m, k}, wei{b, k, n}, bias{n}, dst{b, m, n}
      result_shape = {A.size(0), A.size(1), B.size(2)};
    }
  } else {
    TORCH_CHECK(false, "linear only support for 2D and 3D tensors!\n");
  }

  at::Tensor result = at::empty(result_shape, A.options().dtype(out_dtype));

  c10::MaybeOwned<Tensor> bias_maybe_owned =
      at::borrow_from_optional_tensor(bias_);
  const Tensor& bias = *bias_maybe_owned;

  at::Tensor A_scale_inv = A_scale_inv_.has_value()
      ? A_scale_inv_.value()
      : at::ones({1}, A.options().dtype(out_dtype));
  at::Tensor B_scale_inv = B_scale_inv_.has_value()
      ? B_scale_inv_.value()
      : at::ones({1}, B.options().dtype(out_dtype));

  if (D.has_value()) {
    TORCH_CHECK(
        false,
        "D will be supported with post-op which is working in progress\n");
    // TODO: Support post-ops for fp8_gemm.
  }
  torch_ipex::xpu::oneDNN::fp8_matmul(
      result, A, B, bias, A_scale_inv, B_scale_inv);
  return result;
}

Tensor fp8_gemm_backward(
    const Tensor& m1 /*grad_out*/,
    int64_t m1_format,
    int64_t m1_meta_index,
    const Tensor& m2 /*act, wei*/,
    int64_t grad_format,
    int64_t grad_meta_index,
    const Tensor& scale,
    const Tensor& scale_inv,
    const Tensor& amax_history) {
  std::vector<int64_t> result_shape;
  if (m1.dim() == 2) {
    result_shape = {m1.size(0), m2.size(1)};
  } else if (m1.dim() == 3) {
    if (m2.dim() == 2) {
      result_shape = {m1.size(0) * m1.size(1), m2.size(1)};
    } else {
      result_shape = {m1.size(0), m1.size(1), m2.size(2)};
    }
  } else {
    TORCH_CHECK(false, "linear only support for 2D and 3D tensors!\n");
  }
  Tensor result = at::empty(result_shape, m1.options());
  if (m1.dim() == 3 && m2.dim() == 2) {
    torch_ipex::xpu::oneDNN::matmul(
        result,
        m1.reshape({m1.sizes()[0] * m1.sizes()[1], m1.sizes()[2]}),
        m2,
        at::Tensor(),
        true,
        Attr());
    return result.view_symint(
        {m1.sizes()[0], m1.sizes()[1], result.sym_size(1)});
  }
  torch_ipex::xpu::oneDNN::matmul(result, m1, m2, at::Tensor(), true, Attr());
  return result;
}

} // namespace AtenIpexTypeXPU
} // namespace at

namespace {
IPEX_LIBRARY_FRAGMENT() {
  IPEX_OP_REGISTER_DISPATCH(
      "fp8_gemm.xpu", at::AtenIpexTypeXPU::fp8_gemm, c10::DispatchKey::XPU);
  IPEX_OP_REGISTER_DISPATCH(
      "fp8_gemm.Tensor",
      at::AtenIpexTypeXPU::fp8_gemm_v2,
      c10::DispatchKey::XPU);
  IPEX_OP_REGISTER_DISPATCH(
      "fp8_gemm_backward.xpu",
      at::AtenIpexTypeXPU::fp8_gemm_backward,
      c10::DispatchKey::XPU);
}

// ================================================

namespace {
bool is_fp8_dtype(const at::ScalarType dtype) {
  return dtype == at::ScalarType::Float8_e5m2 or
      dtype == at::ScalarType::Float8_e4m3fn;
}

bool is_float_dtype(const at::ScalarType dtype) {
  return dtype == at::ScalarType::Float or dtype == at::ScalarType::BFloat16 or
      dtype == at::ScalarType::Half;
}

void check_scale(
    const c10::optional<at::Tensor>& scale,
    std::string_view name) {
  if (scale) {
    const auto scale_dtype = scale->scalar_type();
    TORCH_CHECK(
        is_float_dtype(scale_dtype),
        name,
        " tensor must be of torch.float, torch.bfloat16 or torch.float16 dtype dtype, got ",
        scale_dtype);
  }
}

// From documentation of torch.matmul
// If both tensors are 1-dimensional, the dot product (scalar) is returned.
// If both arguments are 2-dimensional, the matrix-matrix product is returned.
// If the first argument is 1-dimensional and the second argument is
// 2-dimensional, a 1 is prepended to its dimension for the purpose of the
// matrix multiply. After the matrix multiply, the prepended dimension is
// removed.
// If both arguments are at least 1-dimensional and at least one
// argument is N-dimensional (where N > 2), then a batched matrix multiply is
// returned. The non-matrix (i.e. batch) dimensions are broadcasted (and thus
// must be broadcastable).
std::vector<int64_t> get_batch_matmul_out_shape(
    at::IntArrayRef shape_a,
    at::IntArrayRef shape_b,
    bool transpose_a,
    bool transpose_b) {
  std::vector<int64_t> output_shape;
  const auto rank_a = shape_a.size();
  const auto rank_b = shape_b.size();

  size_t common_dim_a = 0;
  size_t common_dim_b = 0;

  if (rank_b > 1) {
    auto dim_b = transpose_b ? rank_b - 2 : rank_b - 1;
    common_dim_b = transpose_b ? rank_b - 1 : rank_b - 2;
    output_shape.push_back(shape_b[dim_b]);
  }
  if (rank_a > 1) {
    auto dim_a = transpose_a ? rank_a - 1 : rank_a - 2;
    common_dim_a = transpose_a ? rank_a - 2 : rank_a - 1;
    output_shape.push_back(shape_a[dim_a]);
  }

  auto common_size_a = shape_a[common_dim_a];
  auto common_size_b = shape_b[common_dim_b];

  TORCH_CHECK(
      common_size_a == common_size_b,
      "Common dimension sizes of matmul inputs should be the same. Got ",
      common_size_a,
      " and ",
      common_size_b);

  auto max_rank = std::max(rank_a, rank_b);
  for (size_t i = 3; i <= max_rank; i++) {
    int64_t dim_a = i > rank_a ? 1 : shape_a[rank_a - i];
    int64_t dim_b = i > rank_b ? 1 : shape_b[rank_b - i];
    TORCH_CHECK(
        dim_a == dim_b or dim_a == 1 or dim_b == 1,
        "Batch dimension ",
        max_rank - i,
        " of matmul inputs should be the same or at least one of them should be equal to 1. Got ",
        dim_a,
        " and ",
        dim_b);
    output_shape.push_back(dim_a == dim_b ? dim_a : dim_a * dim_b);
  }
  std::reverse(output_shape.begin(), output_shape.end());

  return output_shape;
}
} // namespace

at::Tensor fp8_gemm_v2(
    const at::Tensor& A,
    bool trans_A,
    const at::Tensor& B,
    bool trans_B,
    const c10::optional<at::Tensor>& D,
    at::ScalarType out_dtype,
    const c10::optional<at::Tensor>& A_scale_inv,
    const c10::optional<at::Tensor>& B_scale_inv,
    const c10::optional<at::Tensor>& bias,
    bool accumulate) {
  // dummy eager implementation
  return at::empty(A.sizes(), A.options().dtype(out_dtype));
}

// v2 ops are versions aligned with Gaudi's ops API.
// When eager implementation is adjusted, v2 suffix will be removed.
TORCH_LIBRARY_FRAGMENT(torch_ipex, m) {
  m.def(
      "fp8_gemm_v2(Tensor A, bool trans_A, Tensor B, bool trans_B, Tensor? D, ScalarType out_dtype, Tensor? A_scale_inv=None, Tensor? B_scale_inv=None, Tensor? bias=None, bool accumulate=False) -> Tensor");
}

TORCH_LIBRARY_IMPL(torch_ipex, XPU, m) {
  m.impl("fp8_gemm_v2", fp8_gemm_v2);
}

at::Tensor fp8_gemm_v2_meta(
    const at::Tensor& A,
    bool trans_A,
    const at::Tensor& B,
    bool trans_B,
    const c10::optional<at::Tensor>& D,
    at::ScalarType out_dtype,
    const c10::optional<at::Tensor>& A_scale_inv,
    const c10::optional<at::Tensor>& B_scale_inv,
    const c10::optional<at::Tensor>& bias,
    bool accumulate) {
  TORCH_CHECK(
      is_fp8_dtype(A.scalar_type()) and is_fp8_dtype(B.scalar_type()),
      "A and B must be of torch.float8_e5m2 or torch.float8_e4m3fn dtype, got ",
      A.scalar_type(),
      " and ",
      B.scalar_type());

  TORCH_CHECK(
      is_float_dtype(out_dtype),
      "out_dtype must be torch.float, torch.bfloat16 or torch.float16 dtype, got ",
      out_dtype);

  check_scale(A_scale_inv, "A_scale_inv");
  check_scale(B_scale_inv, "B_scale_inv");

  return at::empty(
      get_batch_matmul_out_shape(A.sizes(), B.sizes(), trans_A, trans_B),
      A.options().dtype(out_dtype));
}

TORCH_LIBRARY_IMPL(torch_ipex, Meta, m) {
  m.impl("fp8_gemm_v2", &fp8_gemm_v2_meta);
}
} // namespace
