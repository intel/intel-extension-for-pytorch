#include <ATen/Context.h>
#include <ATen/NativeFunctions.h>
#include <ATen/native/LinearAlgebraUtils.h>

#include <oneDNN/oneDNN.h>
#include <runtime/Utils.h>
#include <utils/oneMKLUtils.h>
#include "comm/ATDispatch.h"
#include "comm/ApplyUtils.h"
#include "comm/Numerics.h"

using namespace xpu::dpcpp;

namespace at {
namespace AtenIpexTypeXPU {
namespace impl {

template <typename scalar_t>
void copy_triangle_symmetric_template(Tensor& self, bool upper) {
  auto& dpcpp_queue = dpcppGetCurrentQueue();
  auto row_stride = self.stride(0);
  auto column_stride = self.stride(1);
  auto n = self.size(0);

  size_t work_item_num = n * (n - 1) / 2; // only start the triangle element

  auto cgf = DPCPP_Q_CGF(__cgh) {
    auto data_ptr = (scalar_t*)self.data_ptr();
    auto kfn = DPCPP_Q_KFN(DPCPP::item<1> item_id) {
      auto linear_id = item_id.get_linear_id();
      float triangle_row_ =
          (Numerics<float>::sqrt(1 + 8.0 * linear_id) - 1) / 2;
      int64_t triangle_row = triangle_row_;
      int64_t triangle_col =
          linear_id - (triangle_row * (triangle_row + 1)) / 2;
      int64_t r;
      int64_t c;

      if (upper) {
        r = triangle_col;
        c = triangle_row + 1;
      } else {
        r = triangle_row + 1;
        c = triangle_col;
      }

      auto src_off = r * row_stride + c * column_stride;
      auto dst_off = c * row_stride + r * column_stride;
      data_ptr[dst_off] = data_ptr[src_off];
    };

    __cgh.parallel_for(DPCPP::range</*dim=*/1>(work_item_num), kfn);
  };

  DPCPP_Q_SUBMIT(dpcpp_queue, cgf);
}

Tensor chain_matmul_three_matrices(TensorList matrices) {
  int64_t a = matrices[0].size(0); // first dimension
  int64_t b = matrices[1].size(
      0); // the common dimension between the first two matrices
  int64_t c =
      matrices[2].size(0); // the common dimension between the last two matrices
  int64_t d = matrices[2].size(1); // the last dimension

  int64_t cost_1 = (a * c) * (b + d);
  int64_t cost_2 = (b * d) * (a + c);
  if (cost_1 > cost_2) {
    return at::mm(matrices[0], at::mm(matrices[1], matrices[2]));
  } else {
    return at::mm(at::mm(matrices[0], matrices[1]), matrices[2]);
  }
}

Tensor chain_matmul_recursion(
    TensorList matrices,
    std::vector<std::vector<int64_t>>& order,
    int64_t i,
    int64_t j) {
  if (i == j)
    return matrices[i];
  else
    return at::mm(
        chain_matmul_recursion(matrices, order, i, order[i][j]),
        chain_matmul_recursion(matrices, order, order[i][j] + 1, j));
}

} // namespace impl

Tensor chain_matmul(TensorList matrices) {
  native::checkAllSameDim(matrices, 2);

  TORCH_CHECK(
      matrices.size() > 0, "chain_matmul: Expected one or more matrices");
  if (matrices.size() == 1) {
    return matrices[0];
  } else if (matrices.size() == 2) {
    return at::mm(matrices[0], matrices[1]);
  } else if (matrices.size() == 3) {
    return impl::chain_matmul_three_matrices(matrices);
  } else {
    auto n = matrices.size();

    std::vector<int64_t> p;
    p.push_back(matrices[0].size(0));
    for (size_t i = 0; i < n; i++) {
      p.push_back(matrices[i].size(1));
    }

    std::vector<std::vector<int64_t>> m(n, std::vector<int64_t>(n, 0));

    std::vector<std::vector<int64_t>> s(n, std::vector<int64_t>(n));

    int64_t j, q;

    for (int64_t l = 1; l < n; l++) {
      for (int64_t i = 0; i < n - l; i++) {
        j = i + l;
        m[i][j] = std::numeric_limits<int64_t>::max();
        for (int64_t k = i; k < j; k++) {
          q = m[i][k] + m[k + 1][j] + p[i] * p[k + 1] * p[j + 1];
          if (q < m[i][j]) {
            m[i][j] = q;
            s[i][j] = k;
          }
        }
      }
    }

    return impl::chain_matmul_recursion(matrices, s, 0, n - 1);
  }
}

Tensor& cholesky_inverse_out(Tensor& out, const Tensor& self, bool upper) {
#ifdef USE_ONEMKL
  TORCH_CHECK(
      self.dim() == 2, "input must be 2-d matrix. input shape=", self.sizes());
  TORCH_CHECK(
      self.size(0) == self.size(1),
      "input should be square. input shape=",
      self.sizes());

  int64_t n = self.size(0);
  int64_t lda = n;
  if (n == 0)
    return out;

  out = native::cloneBatchedColumnMajor(self);

  IPEX_DISPATCH_FLOATING_TYPES(out.scalar_type(), "potri_dpcpp_out", [&] {
    dnnl::primitive_attr attr;
    assert(attr.get_scratchpad_mode() == dnnl::scratchpad_mode::library);
    attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);
    auto& dpcpp_queue = dpcppGetCurrentQueue();
    auto upper_lower =
        upper ? (oneapi::mkl::uplo::upper) : (oneapi::mkl::uplo::lower);
    std::int64_t scratchpadsize =
        oneapi::mkl::lapack::potri_scratchpad_size<scalar_t>(
            dpcpp_queue, upper_lower, n, lda);
    Tensor scratchpad_at = at::empty({scratchpadsize}, out.options());
    DPCPP_ONEMKL_SUBMIT(
        dpcpp_queue,
        oneapi::mkl::lapack::potri,
        dpcpp_queue,
        upper_lower,
        n,
        (scalar_t*)out.data_ptr(),
        lda,
        (scalar_t*)scratchpad_at.data_ptr(),
        scratchpadsize);
    impl::copy_triangle_symmetric_template<scalar_t>(out, upper);
  });

  return out;
#else
  AT_ERROR("potri dpcpp: oneMKL library not found in compilation");
#endif
}

Tensor cholesky_inverse(const Tensor& self, bool upper) {
  TORCH_CHECK(
      self.dim() == 2, "input must be 2-d matrix. input shape=", self.sizes());
  TORCH_CHECK(
      self.size(0) == self.size(1),
      "input should be square. input shape=",
      self.sizes());
  Tensor out;
  return AtenIpexTypeXPU::cholesky_inverse_out(out, self, upper);
}

Tensor& ger_out(Tensor& out, const Tensor& self, const Tensor& vec2) {
#ifdef USE_ONEMKL
  TORCH_CHECK(
      self.dim() == 1, "input must be 1-d vector. input shape=", self.sizes());
  TORCH_CHECK(
      vec2.dim() == 1, "vec2 must be 1-d vector. vec2 shape=", vec2.sizes());

  int64_t n = self.size(0); // rows of matrix
  int64_t m = vec2.size(0); // columns of matrix
  if (m == 0 || n == 0)
    return out;
  int64_t input_stride = self.stride(0);
  int64_t vec2_stride = vec2.stride(0);

  out.resize_({n, m}).zero_();
  TORCH_CHECK(out.is_contiguous(), "the out is not contiguous");

  IPEX_DISPATCH_FLOATING_TYPES(out.scalar_type(), "ger_out", [&] {
    auto& dpcpp_queue = dpcppGetCurrentQueue();
    auto x = (scalar_t*)self.data_ptr();
    auto y = (scalar_t*)vec2.data_ptr();
    auto a = (scalar_t*)out.data_ptr();
    // The BLAS API is column major. To save the transpose and element move, we
    // switch the two input. The ger documents
    // https://spec.oneapi.com/versions/0.6.0/oneMKL/GUID-BD2E87B3-5FA7-4E0C-88E2-1982AB0773A2.html
    DPCPP_ONEMKL_SUBMIT(
        dpcpp_queue,
        oneapi::mkl::blas::ger,
        dpcpp_queue,
        m,
        n,
        (float)1.0,
        y,
        vec2_stride,
        x,
        input_stride,
        a,
        m);
  });

  return out;
#else
  AT_ERROR("ger: oneMKL library not found in compilation");
#endif
}

Tensor ger(const Tensor& self, const Tensor& vec2) {
  TORCH_CHECK(
      self.dim() == 1, "input must be 1-d vector. input shape=", self.sizes());
  TORCH_CHECK(
      vec2.dim() == 1, "vec2 must be 1-d vector. vec2 shape=", vec2.sizes());
  auto n = self.size(0); // rows of matrix
  auto m = vec2.size(0); // columns of matrix
  Tensor out = at::zeros({n, m}, self.options());
  return AtenIpexTypeXPU::ger_out(out, self, vec2);
}

inline void dot_check(const Tensor& self, const Tensor& other) {
  TORCH_CHECK(
      self.dim() == 1 && other.dim() == 1,
      "1D tensors expected, but got ",
      self.dim(),
      "D and ",
      other.dim(),
      "D tensors");

  TORCH_CHECK(
      self.scalar_type() == other.scalar_type(),
      "dot : expected both vectors to have same dtype, but found ",
      self.scalar_type(),
      " and ",
      other.scalar_type());

  TORCH_CHECK(
      self.numel() == other.numel(),
      "inconsistent tensor size, expected tensor [",
      self.numel(),
      "] and src [",
      other.numel(),
      "] to have the same number of elements, but got ",
      self.numel(),
      " and ",
      other.numel(),
      " elements respectively");
}

Tensor dot(const Tensor& self, const Tensor& other) {
#ifdef USE_ONEMKL
  dot_check(self, other);
  Tensor result = at::empty({}, self.options());
  // torch.dot supports all types and complex datatype, but oneapi::mkl::blas
  // only supports float/double
  IPEX_DISPATCH_FLOATING_TYPES(self.scalar_type(), "dot", [&] {
    auto& dpcpp_queue = dpcppGetCurrentQueue();
    DPCPP_ONEMKL_SUBMIT(
        dpcpp_queue,
        oneapi::mkl::blas::dot,
        dpcpp_queue,
        self.numel(),
        (scalar_t*)self.data_ptr(),
        self.stride(0),
        (scalar_t*)other.data_ptr(),
        other.stride(0),
        (scalar_t*)result.data_ptr());
  });
  return result;
#else
  AT_ERROR("dot: oneMKL library not found in compilation");
#endif
}

Tensor addr(
    const Tensor& self,
    const Tensor& vec1,
    const Tensor& vec2,
    Scalar beta,
    Scalar alpha) {
  Tensor result = at::AtenIpexTypeXPU::ger(vec1, vec2) * alpha;
  if (beta.to<double>() == 0.0) {
    return result;
  }
  return result + (self * beta);
}

Tensor& addr_out(
    Tensor& result,
    const Tensor& self,
    const Tensor& vec1,
    const Tensor& vec2,
    Scalar beta,
    Scalar alpha) {
  auto addr_result = at::AtenIpexTypeXPU::addr(self, vec1, vec2, beta, alpha);
  // Validates safe casting
  const auto result_dtype = addr_result.scalar_type();
  TORCH_CHECK(
      canCast(result_dtype, result.scalar_type()),
      "result type ",
      result_dtype,
      " can't be cast to the desired output type ",
      result.scalar_type());
  at::AtenIpexTypeXPU::resize_as_(result, addr_result, c10::nullopt);
  result.copy_(addr_result);
  return result;
}

Tensor& addr_(
    Tensor& self,
    const Tensor& vec1,
    const Tensor& vec2,
    Scalar beta,
    Scalar alpha) {
  return at::AtenIpexTypeXPU::addr_out(self, self, vec1, vec2, beta, alpha);
}

} // namespace AtenIpexTypeXPU
} // namespace at
