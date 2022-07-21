#include <ATen/Context.h>
#include <ATen/NativeFunctions.h>
#include <ATen/native/LinearAlgebraUtils.h>

#include <core/detail/OffsetCalculator.h>
#include <oneDNN/oneDNN.h>
#include <runtime/Utils.h>
#include <utils/oneMKLUtils.h>
#include <oneapi/dpl/utility>
#include "Loops.h"
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
    sycl::queue& queue,
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
    sycl::queue& queue,
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
    sycl::queue& queue,
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
    sycl::queue& queue,
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
    sycl::queue& queue,
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
    sycl::queue& queue,
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
    auto kfn = DPCPP_Q_KFN(sycl::item<1> item_id) {
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

    __cgh.parallel_for(sycl::range</*dim=*/1>(work_item_num), kfn);
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
    auto kfn = DPCPP_Q_KFN(sycl::item<1> item) {
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
    cgh.parallel_for(sycl::range<1>(total_items), kfn);
  };
  DPCPP_Q_SUBMIT(queue, cgf);
}

} // namespace impl

Tensor& cholesky_inverse_out(const Tensor& self, bool upper, Tensor& out) {
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
  return AtenIpexTypeXPU::cholesky_inverse_out(self, upper, out);
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
  ScalarType real_dtype = toRealValueType(input.scalar_type());
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

// std::tuple<Tensor, Tensor, Tensor> _det_lu_based_helper(const Tensor& self) {
//   // fallback to at::native::_det_lu_based_helper
//   return at::native::_det_lu_based_helper(self);
// }

constexpr int n_elems_per_work_item = UNROLLED_ELEM_PER_WORK_ITEM;

template <int n_elems_per_work_item, typename func_t>
void _elementwise_kernel(int total_n_elems, func_t f) {
  int total_work_items =
      (total_n_elems + n_elems_per_work_item - 1) / n_elems_per_work_item;
  auto& dpcpp_queue = dpcppGetCurrentQueue();
  auto cgf = DPCPP_Q_CGF(cgh) {
    cgh.parallel_for(
        sycl::range<1>(total_work_items), [=](sycl::item<1> itemId) {
          int idx = itemId.get_linear_id();
#pragma unroll
          for (int i = 0; i < n_elems_per_work_item; ++i) {
            if (idx < total_n_elems) {
              f(idx);
              idx += total_work_items;
            }
          }
        });
  };

  DPCPP_Q_SUBMIT(dpcpp_queue, cgf);
}

template <int n_elems_per_work_item, typename func_t>
static void _launch_kernel(int total_n_elems, func_t f) {
  TORCH_INTERNAL_ASSERT(
      total_n_elems >= 0 &&
      total_n_elems <= std::numeric_limits<int32_t>::max());
  _elementwise_kernel<n_elems_per_work_item, func_t>(total_n_elems, f);
}

void _unpack_pivots_internal_kernel_dpcpp(
    TensorIterator& iter,
    int64_t dim_size) {
  if (iter.numel() == 0) {
    return;
  }

  if (!iter.can_use_32bit_indexing()) {
    for (auto& sub_iter : iter.with_32bit_indexing()) {
      _unpack_pivots_internal_kernel_dpcpp(sub_iter, dim_size);
    }
    return;
  }

  auto offset_calculator = make_offset_calculator<2>(iter);

  char* unpacked_pivots_ptr = reinterpret_cast<char*>(iter.data_ptr(0));
  const char* const __restrict__ pivots_ptr =
      reinterpret_cast<const char*>(iter.data_ptr(1));

  auto loop = [=] DPCPP_DEVICE(int i) {
    auto offsets = offset_calculator.get(i);

    auto* unpacked_pivots_data =
        reinterpret_cast<int32_t*>(unpacked_pivots_ptr + offsets[0]);
    const auto* const __restrict__ pivots_data =
        reinterpret_cast<const int32_t*>(pivots_ptr + offsets[1]);

    // QUESTION: can we mix 64bit offsets with 32bit Iterator indexing?
    for (int64_t i = 0; i < dim_size; ++i) {
      oneapi::dpl::swap(
          unpacked_pivots_data[i], unpacked_pivots_data[pivots_data[i]]);
    }
  };

  _launch_kernel<n_elems_per_work_item>(iter.numel(), loop);
}

void unpack_pivots_kernel_dpcpp(TensorIterator& iter, int64_t dim_size) {
  _unpack_pivots_internal_kernel_dpcpp(iter, dim_size);
}

std::tuple<Tensor, Tensor, Tensor> lu_unpack(
    const Tensor& LU_data,
    const Tensor& LU_pivots,
    bool unpack_data,
    bool unpack_pivots) {
  TORCH_CHECK(
      LU_pivots.is_contiguous() && (LU_pivots.scalar_type() == at::kInt),
      "lu_unpack: LU_pivots is expected to be a contiguous tensor of torch.int32 dtype."
      "Note: this function is intended to be used with the output produced by torch{.linalg}.lu");

  // trivial case
  if (!unpack_data && !unpack_pivots) {
    return std::make_tuple(Tensor(), Tensor(), Tensor());
  }

  Tensor L, U;
  // In the generalized LU factorization, the following shape relations hold:
  // A.shape[-2:] == (m, n),
  // P.shape[-2:] == (m, m),
  // U.shape[-2:] == (m, k),
  // L.shape[-2:] == (k, n),
  // where k = min(m, n)
  int64_t m = LU_data.size(-2);
  int64_t n = LU_data.size(-1);
  int64_t k = std::min(m, n);

  if (unpack_data) {
    U = LU_data.triu();
    if (m != k) {
      U = U.narrow(-2, 0, k);
    }

    L = LU_data.tril();
    if (k != n) {
      L = L.narrow(-1, 0, k);
    }
    L.diagonal(/*offset=*/0, /*dim1=*/-2, /*dim2=*/-1).fill_(1);
  }

  if (!unpack_pivots) {
    return std::make_tuple(Tensor(), L, U);
  }

  auto unpacked_pivots_sizes = LU_pivots.sizes().vec();
  unpacked_pivots_sizes[LU_pivots.dim() - 1] = m;
  auto unpacked_pivots = at::empty(
      unpacked_pivots_sizes,
      LU_pivots.options().memory_format(at::MemoryFormat::Contiguous));

  // Fill `unpacked_pivots` with identity permutation
  auto id_perm = at::arange(m, LU_pivots.options());
  unpacked_pivots.copy_(id_perm);

  // WARNING: we assume that unchanged LAPACK pivots are provided.
  // Since LAPACK relies on the FORTRAN's 1-based indexing,
  // we subtract 1 to convert the pivots to the C-style 0-based indexing.
  // This behaviour could change in the future.
  auto LU_pivots_zero_idx = LU_pivots - 1;

  auto iter = TensorIteratorConfig()
                  .set_check_mem_overlap(false)
                  .check_all_same_dtype(false)
                  .resize_outputs(false)
                  .declare_static_shape(
                      LU_pivots.sizes(), /*squash_dim=*/LU_pivots.dim() - 1)
                  .add_output(unpacked_pivots)
                  .add_input(LU_pivots_zero_idx)
                  .build();
  // }

  unpack_pivots_kernel_dpcpp(iter, LU_pivots.size(-1));

  // The permutation matrix is converted to LU_data.dtype
  // because `matmul` does not work with integer matrices.
  unpacked_pivots_sizes.push_back(m);
  auto permutation_matrix = at::zeros(
      unpacked_pivots_sizes,
      LU_data.options().memory_format(at::MemoryFormat::Contiguous));

  // now that we know the final permutation,
  // scatter 1s at proper locations.
  permutation_matrix.scatter_(
      -2,
      unpacked_pivots.unsqueeze(-2).to(at::kLong),
      at::ones({1}, permutation_matrix.options())
          .expand(permutation_matrix.sizes()));

  return std::make_tuple(permutation_matrix, L, U);
}

using TupleTensorRefs3 = std::tuple<Tensor&, Tensor&, Tensor&>;

TupleTensorRefs3 lu_unpack_out(
    const Tensor& LU_data,
    const Tensor& LU_pivots,
    bool unpack_data,
    bool unpack_pivots,
    Tensor& P,
    Tensor& L,
    Tensor& U) {
  Tensor P_tmp, L_tmp, U_tmp;
  std::tie(P_tmp, L_tmp, U_tmp) =
      at::lu_unpack(LU_data, LU_pivots, unpack_data, unpack_pivots);

  if (unpack_pivots) {
    checkSameDevice("lu_unpack", P, LU_data, "P");
    // Note that lu_unpack returns P such that P.dtype == LU_data.dtype,
    // because otherwise we cannot use P in matric products (no int -> float
    // promotion)
    checkLinalgCompatibleDtype("lu_unpack", P, LU_data, "L");

    at::native::resize_output(P, P_tmp.sizes());
    P.copy_(P_tmp);
  }

  if (unpack_data) {
    checkSameDevice("lu_unpack", L, LU_data, "L");
    checkSameDevice("lu_unpack", U, LU_data, "U");
    checkLinalgCompatibleDtype("lu_unpack", L, LU_data, "L");
    checkLinalgCompatibleDtype("lu_unpack", U, LU_data, "U");

    at::native::resize_output(L, L_tmp.sizes());
    at::native::resize_output(U, U_tmp.sizes());
    L.copy_(L_tmp);
    U.copy_(U_tmp);
  }

  return TupleTensorRefs3(P, L, U);
}

// we consider 6 Taylor expansions of degree
// 1, 2, 4, 8, 12, 18
constexpr int total_n_degs = 6;

Tensor operator_1_norm(const Tensor& tensor) {
  return std::get<0>(tensor.abs().sum(-2).max(-1));
}

// Makes `buffer` to store `num_matrices` number of matrices needed for
// compute the matrix exponentials of different orders, i.e.
// first `num_matrices` matrices from the list l := {I, A, A^2, A^3, A^6}
// in a contiguous block of memory such that
// buffer[0, ...] = l[0], // I
// buffer[1, ...] = l[1], // A
// ...
// buffer[num_matrices - 1, ...] = l[num_matries - 1]
void _fill_matrix_powers(Tensor& buffer, const Tensor& a, int num_matrices) {
  auto a_sizes_minus_last = a.sizes().vec();
  a_sizes_minus_last.pop_back();
  // fill I
  buffer.select(0, 0).copy_(at::diag_embed(
      at::ones({1}, buffer.options()).expand(a_sizes_minus_last)));

  // fill a
  buffer.select(0, 1).copy_(a);

  // fill a^2
  if (2 <= num_matrices - 1) {
    auto out_for_2 = buffer.select(0, 2); // out for a^2
    at::native::matmul_out(buffer.select(0, 1), buffer.select(0, 1), out_for_2);
  }

  // fill a^3
  if (3 <= num_matrices - 1) {
    auto out_for_3 = buffer.select(0, 3); // out for a^3
    at::native::matmul_out(buffer.select(0, 1), buffer.select(0, 2), out_for_3);
  }

  // fill a^6
  if (4 <= num_matrices - 1) {
    auto out_for_4 = buffer.select(0, 4); // out for a^6
    at::native::matmul_out(buffer.select(0, 3), buffer.select(0, 3), out_for_4);
  }
}

inline Tensor _move_memory_if_xpu_input(const Tensor& mem, const Tensor& in) {
  return (in.device().type() == at::kXPU) ? mem.to(at::device_of(in).value())
                                          : mem;
}

// convert a 1D blob to a 2D Tensor of size [1, blob.size()]
// such that blob.device() == in.device())
// designed to be used with _compute_linear_combination
template <typename scalar_t>
inline Tensor _blob_to_Tensor(
    std::initializer_list<scalar_t> blob,
    const Tensor& in) {
  // we convert to void* expecitly because begin() returns
  // a pointer to a constant.
  // Blob is assumed to be a 1D array, that is why
  // we also insert a fake dimension so that the result could directly
  // be used in _compute_linear_combination
  auto tensor =
      at::from_blob(
          (void*)blob.begin(), blob.size(), c10::toValueType(in.scalar_type()))
          .unsqueeze(0);
  return _move_memory_if_xpu_input(tensor, in);
}

template <typename scalar_t, int ROW, int COL>
using array2d = std::array<std::array<scalar_t, COL>, ROW>;

// Allocates a buffers of uninitialized or zero values
// of shape [n_copies, a.size()]
Tensor _allocate_buffer(const Tensor& a, int n_copies, bool is_zero = false) {
  auto res = at::empty(
      {n_copies, a.size(0), a.size(1), a.size(2)},
      a.options().memory_format(at::MemoryFormat::Contiguous));

  if (is_zero) {
    res.zero_();
  }

  return res;
}

// I + A
Tensor compute_T1(const Tensor& A) {
  // 2 for {I, A}
  auto As = _allocate_buffer(A, 2);
  _fill_matrix_powers(As, A, 2);
  return As.sum(0);
}

// I + A + A^2 / 2
Tensor compute_T2(const Tensor& A) {
  auto As = _allocate_buffer(A, 3);
  // 3 for {I, A, A^2}
  _fill_matrix_powers(As, A, 3);
  As.select(0, 2).div_(2.0);
  return As.sum(0);
}

// I + A + A^2 * (I / 2 + A / 6 + A^2 / 24)
template <typename scalar_t>
Tensor compute_T4(const Tensor& A) {
  auto As = _allocate_buffer(A, 4);
  // 3 for {I, A, A^2}
  _fill_matrix_powers(As, A, 3);

  // output for A^2 * (I / 2 + A / 6 + A^2 / 24)
  auto out_for_a2 = As.select(0, 3);
  at::native::matmul_out(
      // contains A^2
      As.select(0, 2),
      // computes (I / 2 + A / 6 + A^2 / 24)
      AtenIpexTypeXPU::_compute_linear_combination(
          As.narrow(0, 0, 3),
          _blob_to_Tensor<scalar_t>({1 / 2.0, 1 / 6.0, 1 / 24.0}, A)),
      out_for_a2);

  // I + A + A^2 * (I / 2 + A / 6 + A^2 / 24)
  return AtenIpexTypeXPU::_compute_linear_combination(
      As, _blob_to_Tensor<scalar_t>({1.0, 1.0, 0.0, 1.0}, A));
}

template <typename scalar_t>
Tensor compute_T8(const Tensor& A) {
  constexpr scalar_t sqrt_177 = 0.1330413469565007072504e+2;
  constexpr scalar_t x3 = 2. / 3.;
  constexpr scalar_t x1 = x3 * ((1. + sqrt_177) / 88.);
  constexpr scalar_t x2 = x3 * ((1. + sqrt_177) / 352.);
  constexpr scalar_t x4 = (-271. + 29. * sqrt_177) / (315. * x3);
  constexpr scalar_t x5 = (-11. + 11. * sqrt_177) / (1260. * x3);
  constexpr scalar_t x6 = (-99. + 11. * sqrt_177) / (5040. * x3);
  constexpr scalar_t x7 = (89. - sqrt_177) / (5040. * x3);
  constexpr scalar_t y2 = (857. - 58. * sqrt_177) / 630.;

  auto As = _allocate_buffer(A, 5);
  // 3 for {I, A, A^2}
  _fill_matrix_powers(As, A, 3);

  // output for A4
  auto out_for_a4 = As.select(0, 3);
  // A4 =  A2 * (x1 * A + x2 * A2)
  at::native::matmul_out(
      // As.select(0, 2) = A^2
      As.select(0, 2),
      AtenIpexTypeXPU::_compute_linear_combination(
          // extract {A, A^2} from As
          As.narrow(0, 1, 2),
          _blob_to_Tensor<scalar_t>({x1, x2}, A)),
      out_for_a4);

  // output for A8
  auto out_for_a8 = As.select(0, 4);
  // A8 = (x3 * A2 + A4) * (x4 * I + x5 * A + x6 * A2 + x7 * A4)
  at::native::matmul_out(
      // x3 * A2 + A4
      AtenIpexTypeXPU::_compute_linear_combination(
          As.narrow(0, 2, 2), _blob_to_Tensor<scalar_t>({x3, 1.0}, A)),
      AtenIpexTypeXPU::_compute_linear_combination(
          As.narrow(0, 0, 4), _blob_to_Tensor<scalar_t>({x4, x5, x6, x7}, A)),
      out_for_a8);

  // return I + A + y2 * A2 + A8;
  return AtenIpexTypeXPU::_compute_linear_combination(
      As, _blob_to_Tensor<scalar_t>({1.0, 1.0, y2, 0.0, 1.0}, A));
}

template <typename scalar_t>
Tensor compute_T12(const Tensor& A) {
  constexpr int num_prods = 4;
  array2d<scalar_t, num_prods, num_prods> b = {
      {{9.0198e-16,
        0.46932117595418237389,
        -0.20099424927047284052,
        -0.04623946134063071740},
       {5.31597895759871264183,
        1.19926790417132231573,
        0.01179296240992997031,
        0.01108844528519167989},
       {0.18188869982170434744,
        0.05502798439925399070,
        0.09351590770535414968,
        0.00610700528898058230},
       {-2.0861320e-13,
        -0.13181061013830184015,
        -0.02027855540589259079,
        -0.00675951846863086359}}};

  // gather coefficients `b` from above into a tensor,
  // and move them to device `device_of(A)`
  auto bs = at::from_blob(
      reinterpret_cast<void*>(&b),
      {num_prods, num_prods},
      {num_prods, 1},
      c10::toValueType(A.scalar_type()));
  bs = _move_memory_if_xpu_input(bs, A);

  auto As = _allocate_buffer(A, num_prods);
  _fill_matrix_powers(As, A, num_prods);

  auto Bs = AtenIpexTypeXPU::_compute_linear_combination(As, bs);

  // tmp buffer for this matrix product
  auto out_for_a6 = As.select(0, 0);
  // compute A6
  Bs.select(0, 2).add_(
      at::native::matmul_out(Bs.select(0, 3), Bs.select(0, 3), out_for_a6));

  // tmp buffer for this matrix product
  auto out = As.select(0, 0);
  return Bs.select(0, 0).add_(at::native::matmul_out(
      Bs.select(0, 1).add_(Bs.select(0, 2)), Bs.select(0, 2), out));
}

template <typename scalar_t>
Tensor compute_T18(const Tensor& A) {
  constexpr int num_prods = 5;
  array2d<scalar_t, num_prods, num_prods> b = {
      {{0.,
        -1.00365581030144618291e-01,
        -8.02924648241156932449e-03,
        -8.92138498045729985177e-04,
        0.},
       {0.,
        3.97849749499645077844e-01,
        1.36783778460411720168e+00,
        4.98289622525382669416e-01,
        -6.37898194594723280150e-04},
       {-1.09676396052962061844e+01,
        1.68015813878906206114e+00,
        5.71779846478865511061e-02,
        -6.98210122488052056106e-03,
        3.34975017086070470649e-05},
       {-9.04316832390810593223e-02,
        -6.76404519071381882256e-02,
        6.75961301770459654925e-02,
        2.95552570429315521194e-02,
        -1.39180257516060693404e-05},
       {0.,
        0.,
        -9.23364619367118555360e-02,
        -1.69364939002081722752e-02,
        -1.40086798182036094347e-05}}};

  // gather coefficients `b` from above into a tensor,
  // and move them to device `device_of(A)`
  auto bs = at::from_blob(
      reinterpret_cast<void*>(&b),
      {num_prods, num_prods},
      {num_prods, 1},
      c10::toValueType(A.scalar_type()));
  bs = _move_memory_if_xpu_input(bs, A);

  auto As = _allocate_buffer(A, num_prods);
  _fill_matrix_powers(As, A, num_prods);

  auto Bs = AtenIpexTypeXPU::_compute_linear_combination(As, bs);

  // tmp buffer for this matrix product
  auto out_for_a9 = As.select(0, 0);
  // compute A9
  Bs.select(0, 3).add_(
      at::native::matmul_out(Bs.select(0, 0), Bs.select(0, 4), out_for_a9));

  // tmp buffer for this matrix product
  auto out = As.select(0, 0);
  return Bs.select(0, 1).add_(at::native::matmul_out(
      Bs.select(0, 2).add_(Bs.select(0, 3)), Bs.select(0, 3), out));
}

template <typename scalar_t>
void compute_T18_scale_square(
    Tensor& mexp_out,
    const Tensor& a,
    const Tensor& norm,
    scalar_t theta) {
  // Scale
  const auto s = at::max(at::zeros_like(norm), at::ceil(at::log2(norm / theta)))
                     .unsqueeze(-1)
                     .unsqueeze(-1)
                     .to(at::kLong);
  const auto pow2s = at::pow(2, s);
  const auto a_scaled = a / pow2s;

  // Square
  auto mexp_scaled = compute_T18<scalar_t>(a_scaled);
  auto s_cpu = (s.device().type() == at::kCPU) ? s : s.to(at::kCPU);
  for (int64_t i = 0; i < mexp_scaled.size(0); ++i) {
    auto s_val = s_cpu.select(0, i).template item<int64_t>();
    auto mexp = mexp_scaled.select(0, i);
    for (int64_t p = 0; p < s_val; ++p) {
      mexp = at::matmul(mexp, mexp);
    }
    mexp_out.select(0, i).copy_(mexp);
  }
}

template <typename scalar_t>
Tensor mexp_impl(
    const Tensor& a,
    std::array<scalar_t, total_n_degs> thetas,
    bool compute_highest_degree_approx = false) {
  auto res = at::empty_like(a);
  const auto norm = operator_1_norm(a);
  // `norm_cpu` is used to decide which Tensors require which approximation
  // based on their norm. This decision takes place on CPU.
  // It requires moving data back and forth between devices when `a` is on XPU,
  // but at the cost of only one sigle CPU-XPU synchronization (instead of 6),
  // and better performance overall (benchmarked).
  const auto norm_cpu =
      (a.device().type() == at::kXPU) ? norm.to(at::kCPU) : norm;

  if (!compute_highest_degree_approx) {
    constexpr std::array<Tensor (*)(const Tensor&), total_n_degs - 1>
        compute_Ts = {
            compute_T1,
            compute_T2,
            compute_T4<scalar_t>,
            compute_T8<scalar_t>,
            compute_T12<scalar_t>};

    for (int i = 0; i < total_n_degs - 1; ++i) {
      auto norm_lower_bound =
          (i == 0) ? static_cast<scalar_t>(-1) : thetas[i - 1];
      auto norm_upper_bound = thetas[i];
      // nonzero returns a 2D tensor, hence squeeze(-1) to make it 1D
      auto idx_curr_norm_interval =
          ((norm_lower_bound < norm_cpu) * (norm_cpu <= norm_upper_bound))
              .nonzero()
              .squeeze(-1);

      if (idx_curr_norm_interval.numel()) {
        auto idx_to_device =
            _move_memory_if_xpu_input(idx_curr_norm_interval, a);
        auto sub_a = at::index_select(a, 0, idx_to_device);
        res.index_put_({idx_to_device}, compute_Ts[i](sub_a));
      }
    }

    // nonzero returns a 2D tensor, hence squeeze(-1) to make it 1D
    auto idx_large_norm =
        (norm_cpu >= thetas[total_n_degs - 2]).nonzero().squeeze(-1);

    if (idx_large_norm.numel()) {
      auto idx_to_device = _move_memory_if_xpu_input(idx_large_norm, a);
      auto a_large_norm = at::index_select(a, 0, idx_to_device);
      auto large_norm_subset = at::index_select(norm, 0, idx_to_device);
      auto mexp_out = at::empty_like(a_large_norm);

      compute_T18_scale_square(
          mexp_out, a_large_norm, large_norm_subset, thetas[total_n_degs - 1]);
      res.index_put_({idx_large_norm}, mexp_out);
    }

    return res;
  }

  compute_T18_scale_square(res, a, norm, thetas[total_n_degs - 1]);

  return res;
}

Tensor mexp(const Tensor& a, bool compute_highest_degree_approx = false) {
  // squash batch dimensions to one dimension for simplicity
  const auto a_3d = a.view({-1, a.size(-2), a.size(-1)});

  if (a.scalar_type() == at::ScalarType::Float ||
      a.scalar_type() == at::ScalarType::ComplexFloat) {
    constexpr std::array<float, total_n_degs> thetas_float = {
        1.192092800768788e-07, // deg 1
        5.978858893805233e-04, // deg 2
        5.116619363445086e-02, // deg 4
        5.800524627688768e-01, // deg 8
        1.461661507209034e+00, // deg 12
        3.010066362817634e+00 // deg 18
    };

    return mexp_impl<float>(a_3d, thetas_float, compute_highest_degree_approx)
        .view(a.sizes());
  } else { // if Double or ComplexDouble
    constexpr std::array<double, total_n_degs> thetas_double = {
        2.220446049250313e-16, // deg 1
        2.580956802971767e-08, // deg 2
        3.397168839976962e-04, // deg 4
        4.991228871115323e-02, // deg 8
        2.996158913811580e-01, // deg 12
        1.090863719290036e+00 // deg 18
    };

    return mexp_impl<double>(a_3d, thetas_double, compute_highest_degree_approx)
        .view(a.sizes());
  }
}

// Computes the matrix exponential for a given batch of squared matrices.
// The implementaion is based on:
//
// Bader, P.; Blanes, S.; Casas, F.
// Computing the Matrix Exponential with an Optimized Taylor Polynomial
// Approximation. Mathematics 2019, 7, 1174.
//
at::Tensor matrix_exp(const at::Tensor& self) {
  TORCH_CHECK(
      self.dim() >= 2 &&
          (at::isFloatingType(self.scalar_type()) ||
           at::isComplexType(self.scalar_type())),
      "matrix_exp(",
      self.scalar_type(),
      "{",
      self.sizes(),
      "}): expected self tensor "
      "of floating or complex types with dim at least 2");
  TORCH_CHECK(
      self.size(-1) == self.size(-2),
      "matrix_exp(",
      self.scalar_type(),
      "{",
      self.sizes(),
      "}): expected self tensor "
      "of squared matrices");

  NoTF32Guard disable_tf32;

  if (self.size(-1) == 1) {
    return self.exp();
  }

  return mexp(self);
}

} // namespace AtenIpexTypeXPU
} // namespace at
