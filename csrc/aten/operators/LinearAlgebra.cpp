#include <ATen/Context.h>
#include <ATen/NativeFunctions.h>
#include <ATen/native/LinearAlgebraUtils.h>

#include <oneDNN/oneDNN.h>
#include <runtime/Utils.h>
#include <utils/oneMKLUtils.h>
#include "Resize.h"
#include "comm/ATDispatch.h"
#include "comm/AccumulateType.h"
#include "comm/ApplyUtils.h"
#include "comm/Numerics.h"
#include "comm/RegistrationDeclarations.h"

using namespace xpu::dpcpp;

namespace at {
namespace AtenIpexTypeXPU {

const Tensor& resize_as_(
    const Tensor& self,
    const Tensor& the_template,
    c10::optional<MemoryFormat> memory_format);

namespace impl {

// mkl dot: Computes the dot product of two real vectors.
#ifdef USE_ONEMKL
template <typename scalar_t>
void mkl_dot(
    DPCPP::queue& queue,
    int64_t n,
    scalar_t* x,
    int64_t incx,
    scalar_t* y,
    int64_t incy,
    scalar_t* result) {
  DPCPP_ONEMKL_SUBMIT(
      queue, oneapi::mkl::blas::dot, queue, n, x, incx, y, incy, result);
}

// mkl dotu: Computes the dot product of two complex vectors.
template <>
void mkl_dot<c10::complex<double>>(
    DPCPP::queue& queue,
    int64_t n,
    c10::complex<double>* x,
    int64_t incx,
    c10::complex<double>* y,
    int64_t incy,
    c10::complex<double>* result) {
  DPCPP_ONEMKL_SUBMIT(
      queue,
      oneapi::mkl::blas::dotu,
      queue,
      n,
      reinterpret_cast<std::complex<double>*>(x),
      incx,
      reinterpret_cast<std::complex<double>*>(y),
      incy,
      reinterpret_cast<std::complex<double>*>(result));
}

template <>
void mkl_dot<c10::complex<float>>(
    DPCPP::queue& queue,
    int64_t n,
    c10::complex<float>* x,
    int64_t incx,
    c10::complex<float>* y,
    int64_t incy,
    c10::complex<float>* result) {
  DPCPP_ONEMKL_SUBMIT(
      queue,
      oneapi::mkl::blas::dotu,
      queue,
      n,
      reinterpret_cast<std::complex<float>*>(x),
      incx,
      reinterpret_cast<std::complex<float>*>(y),
      incy,
      reinterpret_cast<std::complex<float>*>(result));
}

template <typename scalar_t>
void mkl_vdot(
    DPCPP::queue& queue,
    int64_t n,
    scalar_t* x,
    int64_t incx,
    scalar_t* y,
    int64_t incy,
    scalar_t* result) {
  AT_ERROR("mkl::dotc: not implemented for ", typeid(scalar_t).name());
}

// mkl dotc: Computes the dot product of two complex vectors, conjugating the
// first vector.
template <>
void mkl_vdot<c10::complex<double>>(
    DPCPP::queue& queue,
    int64_t n,
    c10::complex<double>* x,
    int64_t incx,
    c10::complex<double>* y,
    int64_t incy,
    c10::complex<double>* result) {
  DPCPP_ONEMKL_SUBMIT(
      queue,
      oneapi::mkl::blas::dotc,
      queue,
      n,
      reinterpret_cast<std::complex<double>*>(x),
      incx,
      reinterpret_cast<std::complex<double>*>(y),
      incy,
      reinterpret_cast<std::complex<double>*>(result));
}

template <>
void mkl_vdot<c10::complex<float>>(
    DPCPP::queue& queue,
    int64_t n,
    c10::complex<float>* x,
    int64_t incx,
    c10::complex<float>* y,
    int64_t incy,
    c10::complex<float>* result) {
  DPCPP_ONEMKL_SUBMIT(
      queue,
      oneapi::mkl::blas::dotc,
      queue,
      n,
      reinterpret_cast<std::complex<float>*>(x),
      incx,
      reinterpret_cast<std::complex<float>*>(y),
      incy,
      reinterpret_cast<std::complex<float>*>(result));
}
#endif

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

template <typename scalar_t>
void addr_kernel_out(
    Tensor& out,
    const Tensor& self,
    const Tensor& vec1,
    const Tensor& vec2,
    const int64_t vec1_numel,
    const int64_t vec2_numel,
    const Scalar& beta,
    const Scalar& alpha,
    const bool& self_ignore,
    const int64_t& broadcast_dim) {
  auto& queue = dpcppGetCurrentQueue();
  using accscalar_t = acc_type<scalar_t>;
  int64_t total_items = vec2_numel;

  auto alpha_scalar = alpha.to<scalar_t>();
  auto beta_scalar = beta.to<scalar_t>();

  auto cgf = DPCPP_Q_CGF(cgh) {
    auto out_ptr = out.data_ptr<scalar_t>();
    auto input_ptr = self.data_ptr<scalar_t>();
    auto vec1_ptr = vec1.data_ptr<scalar_t>();
    auto vec2_ptr = vec2.data_ptr<scalar_t>();
    auto kfn = DPCPP_Q_KFN(DPCPP::item<1> item) {
      auto item_id = item.get_id(0);
      auto vec2_elem = static_cast<accscalar_t>(alpha_scalar) *
          static_cast<accscalar_t>(vec2_ptr[item_id]);
      for (auto id = 0; id < vec1_numel; ++id) {
        // out = beta * self + alpha * (vec1 ⊗ vec2)
        auto vec1_elem = static_cast<accscalar_t>(vec1_ptr[id]);
        auto out_index = id * vec2_numel + item_id;
        if (self_ignore) {
          out_ptr[out_index] = static_cast<scalar_t>(vec1_elem * vec2_elem);
        } else {
          accscalar_t self_elem;
          if (broadcast_dim == 0) {
            self_elem = static_cast<accscalar_t>(beta_scalar) *
                static_cast<accscalar_t>(input_ptr[id]);
          } else if (broadcast_dim == 1) {
            self_elem = static_cast<accscalar_t>(beta_scalar) *
                static_cast<accscalar_t>(input_ptr[item_id]);
          } else if (broadcast_dim == -1) {
            self_elem = static_cast<accscalar_t>(beta_scalar) *
                static_cast<accscalar_t>(input_ptr[0]);
          } else {
            // no broadcast
            self_elem = static_cast<accscalar_t>(beta_scalar) *
                static_cast<accscalar_t>(input_ptr[out_index]);
          }
          out_ptr[out_index] =
              static_cast<scalar_t>(vec1_elem * vec2_elem + self_elem);
        }
      }
    };
    cgh.parallel_for(DPCPP::range<1>(total_items), kfn);
  };
  DPCPP_Q_SUBMIT(queue, cgf);
}

} // namespace impl

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

// PyTorch deprecates this op, which calls mul_out now
// we commented its onemkl impl here

// Tensor& ger_out(Tensor& out, const Tensor& self, const Tensor& vec2) {
// // TODO: Will implement ger kernel on all floating and int datatype, exclude
// // float32 and float64
// #ifdef USE_ONEMKL
//   TORCH_CHECK(
//       self.dim() == 1, "input must be 1-d vector. input shape=",
//       self.sizes());
//   TORCH_CHECK(
//       vec2.dim() == 1, "vec2 must be 1-d vector. vec2 shape=", vec2.sizes());

//   int64_t n = self.size(0); // rows of matrix
//   int64_t m = vec2.size(0); // columns of matrix
//   if (m == 0 || n == 0)
//     return out;
//   int64_t input_stride = self.stride(0);
//   int64_t vec2_stride = vec2.stride(0);

//   out.resize_({n, m}).zero_();
//   TORCH_CHECK(out.is_contiguous(), "the out is not contiguous");

//   IPEX_DISPATCH_FLOATING_TYPES(out.scalar_type(), "ger_out", [&] {
//     auto& dpcpp_queue = dpcppGetCurrentQueue();
//     auto x = (scalar_t*)self.data_ptr();
//     auto y = (scalar_t*)vec2.data_ptr();
//     auto a = (scalar_t*)out.data_ptr();
//     // The BLAS API is column major. To save the transpose and element move,
//     we
//     // switch the two input. The ger documents
//     //
//     https://spec.oneapi.com/versions/0.6.0/oneMKL/GUID-BD2E87B3-5FA7-4E0C-88E2-1982AB0773A2.html
//     DPCPP_ONEMKL_SUBMIT(
//         dpcpp_queue,
//         oneapi::mkl::blas::ger,
//         dpcpp_queue,
//         m,
//         n,
//         (float)1.0,
//         y,
//         vec2_stride,
//         x,
//         input_stride,
//         a,
//         m);
//   });

//   return out;
// #else
//   AT_ERROR("ger: oneMKL library not found in compilation");
// #endif
// }

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
  // if self and other are both complex,
  // dot computes self*other, without conjugating.
  if (self.is_complex()) {
    if (self.is_conj()) {
      if (other.is_conj()) {
        return (at::AtenIpexTypeXPU::dot(self.conj(), other.conj())).conj();
      } else {
        return at::AtenIpexTypeXPU::vdot(self.conj(), other);
      }
    } else if (other.is_conj()) {
      return at::AtenIpexTypeXPU::vdot(other.conj(), self);
    }
  }
#ifdef USE_ONEMKL
  dot_check(self, other);
  Tensor result = at::empty({}, self.options());

  IPEX_DISPATCH_FLOATING_AND_COMPLEX_TYPES(self.scalar_type(), "dot", [&] {
    auto& dpcpp_queue = dpcppGetCurrentQueue();
    impl::mkl_dot<scalar_t>(
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

Tensor vdot(const Tensor& self, const Tensor& other) {
  // Dispatch to `dot` for real dtypes.
  if (!self.is_complex()) {
    return at::dot(self, other);
  }

  // vdot computes dot product uses the complex conjugate of self
  if (self.is_conj()) {
    if (other.is_conj()) {
      return at::AtenIpexTypeXPU::vdot(other.conj(), self.conj());
    } else {
      return at::AtenIpexTypeXPU::dot(self.conj(), other);
    }
  } else if (other.is_conj()) {
    return (at::AtenIpexTypeXPU::dot(self, other.conj())).conj();
  }

#ifdef USE_ONEMKL
  dot_check(self, other);
  Tensor result = at::empty({}, self.options());

  IPEX_DISPATCH_COMPLEX_TYPES(self.scalar_type(), "vdot", [&] {
    auto& dpcpp_queue = dpcppGetCurrentQueue();
    impl::mkl_vdot<scalar_t>(
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
  AT_ERROR("vdot: oneMKL library not found in compilation");
#endif
}

static void check_1d(const Tensor& t, const char* arg, const char* fn) {
  TORCH_CHECK(
      t.dim() == 1,
      fn,
      ": Expected 1-D argument ",
      arg,
      ", but got ",
      t.dim(),
      "-D");
}

static void check_addr_scalar(
    const ScalarType dtype,
    const Scalar& scalar,
    const std::string& scalar_name) {
  TORCH_CHECK(
      !scalar.isBoolean() || dtype == at::ScalarType::Bool,
      "Boolean ",
      scalar_name,
      " only supported for Boolean results.");
  TORCH_CHECK(
      at::isFloatingType(dtype) || at::isComplexType(dtype) ||
          scalar.isIntegral(true),
      "For integral input tensors, "
      "argument ",
      scalar_name,
      " must not be a floating point number.");
}

Tensor addr(
    const Tensor& self,
    const Tensor& vec1,
    const Tensor& vec2,
    const Scalar& beta,
    const Scalar& alpha) {
  check_addr_scalar(self.scalar_type(), beta, "beta");
  check_addr_scalar(self.scalar_type(), alpha, "alpha");

  check_1d(vec1, "vec1", "addr");
  check_1d(vec2, "vec2", "addr");

  int64_t vec1_numel = vec1.numel();
  int64_t vec2_numel = vec2.numel();

  auto self_contiguous = self.contiguous();
  auto vec1_contiguous = vec1.contiguous();
  auto vec2_contiguous = vec2.contiguous();

  // when beta is not zero, self is needed to add on vec1 * vec2, additionally
  // it supports broadcast on dim0 or dim1
  bool self_ignore = bool(0.0 == beta.toComplexDouble());

  // which dim needs to broadcast, -2 means no need broadcast, -1 means
  // broadcast on all dims
  int64_t broadcast_dim = -2;
  if (!self_ignore) {
    if ((self_contiguous.dim() == 2) &&
        (self_contiguous.sizes()[0] == vec1_numel) &&
        (self_contiguous.sizes()[1] == 1)) {
      broadcast_dim = 0;
    } else if (
        (self_contiguous.dim() == 2) && (self_contiguous.sizes()[0] == 1) &&
        (self_contiguous.sizes()[1] == vec2_numel)) {
      broadcast_dim = 1;
    } else if (
        (self_contiguous.dim() == 1) &&
        (self_contiguous.sizes()[0] == vec1_numel)) {
      broadcast_dim = 0;
    } else if (self_contiguous.numel() == 1) {
      broadcast_dim = -1;
    } else {
      TORCH_CHECK(
          (self_contiguous.dim() == 2) &&
              (self_contiguous.sizes()[0] == vec1_numel) &&
              (self_contiguous.sizes()[1] == vec2_numel),
          "The expanded self size cannot match the out size");
    }
  }

  auto out = at::empty({vec1_numel, vec2_numel}, self_contiguous.options());
  IPEX_DISPATCH_ALL_TYPES_AND_COMPLEX_AND2(
      at::ScalarType::BFloat16,
      at::ScalarType::Half,
      vec1_contiguous.scalar_type(),
      "addr_kernel_out",
      [&]() {
        impl::addr_kernel_out<scalar_t>(
            out,
            self_contiguous,
            vec1_contiguous,
            vec2_contiguous,
            vec1_numel,
            vec2_numel,
            beta,
            alpha,
            self_ignore,
            broadcast_dim);
      });
  return out;
}

Tensor& addr_out(
    const Tensor& self,
    const Tensor& vec1,
    const Tensor& vec2,
    const Scalar& beta,
    const Scalar& alpha,
    Tensor& out) {
  auto addr_result = at::AtenIpexTypeXPU::addr(self, vec1, vec2, beta, alpha);
  // Validates safe casting
  const auto result_dtype = addr_result.scalar_type();
  TORCH_CHECK(
      canCast(result_dtype, out.scalar_type()),
      "out type ",
      result_dtype,
      " can't be cast to the desired output type ",
      out.scalar_type());
  at::AtenIpexTypeXPU::resize_as_(out, addr_result, c10::nullopt);
  out.copy_(addr_result);
  return out;
}

static inline void squareCheckInputs(const Tensor& self) {
  TORCH_CHECK(
      self.dim() >= 2, "Tensor of matrices must have at least 2 dimensions. ");
  TORCH_CHECK(
      self.size(-1) == self.size(-2),
      "A must be batches of square matrices, "
      "but they are ",
      self.size(-1),
      " by ",
      self.size(-2),
      " matrices");
}

using namespace native;

// Helper function for det methods.
// For pivoted LU factorization A = P * L * U. Since we always have det(L) = 1,
// det(P) = \pm 1, this method returns a 3-tuple:
//   (det(P), diag(U), info),
// where info helps us identify singular matrices.
static inline std::
    tuple<c10::ExclusivelyOwned<Tensor>, c10::ExclusivelyOwned<Tensor>>
    _lu_det_P_diag_U(const Tensor& self) {
  Tensor pivs, lu, infos;
  std::tie(lu, pivs, infos) =
      at::_lu_with_info(self, /*pivot=*/true, /*check_errors=*/false);
  TORCH_CHECK(
      infos.ge(0).all().item<uint8_t>(), "Invalid argument passed to lu");
  auto n = self.size(-1);
  auto num_exchanges = (at::arange(1, n + 1, pivs.options()) != pivs)
                           .sum(-1, /*keepdim=*/false, /*dtype=*/at::kLong)
                           .fmod_(2);
  auto u_diagonal = lu.diagonal(/*offset=*/0, /*dim1=*/-2, /*dim2=*/-1);
  num_exchanges.mul_(-2).add_(1);
  return std::make_tuple(
      c10::ExclusivelyOwned<Tensor>(std::move(num_exchanges)),
      c10::ExclusivelyOwned<Tensor>(std::move(u_diagonal)));
}

std::tuple<Tensor, Tensor> linalg_slogdet(const Tensor& self) {
  squareCheckInputs(self);
  ScalarType t = self.scalar_type();
  TORCH_CHECK(
      t == ScalarType::Double || t == ScalarType::Float ||
          t == ScalarType::ComplexFloat || t == ScalarType::ComplexDouble,
      "linalg_slogdet: expected a tensor of float, double, cfloat or cdouble types but got ",
      t);

  c10::ExclusivelyOwned<Tensor> det_P, diag_U;
  std::tie(det_P, diag_U) = _lu_det_P_diag_U(self);
  auto det_sign = diag_U->sgn().prod(-1).mul_(*det_P);
  // abslogdet_val is -inf if U is singular, in which case
  // diag_U.abs_().log_().sum(-1) will return -inf. U is singular when U(i, i) =
  // 0 for some i in [1, self.size(-1)]. Since abslogdet_val cannot take nan, no
  // special case handling is required. in-place abs is not supported for
  // complex tensors
  auto abslogdet_val = isComplexType(t) ? diag_U->abs().log_().sum(-1)
                                        : diag_U->abs_().log_().sum(-1);
  return std::make_tuple(det_sign, abslogdet_val);
}

// TODO: implement _out variant avoiding copy and using already allocated
// storage directly
std::tuple<Tensor&, Tensor&> linalg_slogdet_out(
    const Tensor& input,
    Tensor& sign,
    Tensor& logabsdet) {
  checkSameDevice("linalg_slogdet", sign, input, "sign");
  checkSameDevice("linalg_slogdet", logabsdet, input, "logabsdet");
  checkLinalgCompatibleDtype("linalg_slogdet", sign, input, "sign");
  ScalarType real_dtype = toValueType(input.scalar_type());
  // logabsdet is always real-valued here
  checkLinalgCompatibleDtype(
      "linalg_slogdet", logabsdet.scalar_type(), real_dtype, "logabsdet");

  Tensor sign_tmp, logabsdet_tmp;
  std::tie(sign_tmp, logabsdet_tmp) = at::linalg_slogdet(input);

  resize_output(sign, sign_tmp.sizes());
  sign.copy_(sign_tmp);
  resize_output(logabsdet, logabsdet_tmp.sizes());
  logabsdet.copy_(logabsdet_tmp);

  return std::tuple<Tensor&, Tensor&>(sign, logabsdet);
}

std::tuple<Tensor, Tensor> slogdet(const Tensor& self) {
  return at::linalg_slogdet(self);
}

std::tuple<Tensor, Tensor, Tensor> _det_lu_based_helper(const Tensor& self) {
  // fallback to at::native::_det_lu_based_helper
  return at::native::_det_lu_based_helper(self);
}

} // namespace AtenIpexTypeXPU
} // namespace at
