#include <ATen/Context.h>
#include <ATen/NativeFunctions.h>
#include <ATen/native/LinearAlgebraUtils.h>
#include <ATen/native/Resize.h>
#include <core/detail/OffsetCalculator.h>
#include <oneDNN/oneDNN.h>
#include <runtime/Utils.h>
#include <tensor/TensorMeta.h>
#include <iostream>
#include "Loops.h"
#include "Resize.h"
#include "comm/ATDispatch.h"
#include "comm/AccumulateType.h"
#include "comm/ApplyUtils.h"
#include "comm/Numerics.h"
#include "comm/RegistrationDeclarations.h"

using namespace torch_ipex::xpu::dpcpp;

namespace at {
namespace AtenIpexTypeXPU {

const Tensor& resize_as_(
    const Tensor& self,
    const Tensor& the_template,
    c10::optional<MemoryFormat> memory_format);

namespace impl {

template <typename scalar_t>
struct CopyTriangleSymmetricTemplateKernelFunctor {
  void operator()(sycl::item<1> item_id) const {
    auto linear_id = item_id.get_linear_id();
    float triangle_row_ = (Numerics<float>::sqrt(1 + 8.0f * linear_id) - 1) / 2;
    int64_t triangle_row = triangle_row_;
    int64_t triangle_col = linear_id - (triangle_row * (triangle_row + 1)) / 2;
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
  }
  CopyTriangleSymmetricTemplateKernelFunctor(
      bool upper_,
      int64_t row_stride_,
      int64_t column_stride_,
      scalar_t* data_ptr_)
      : upper(upper_),
        row_stride(row_stride_),
        column_stride(column_stride_),
        data_ptr(data_ptr_) {}

 private:
  bool upper;
  int64_t row_stride;
  int64_t column_stride;
  scalar_t* data_ptr;
};

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
struct addr_kernel_functor {
  scalar_t operator()(scalar_t self_val, scalar_t vec1_val, scalar_t vec2_val)
      const {
    return alpha_val && vec1_val && vec2_val;
  }
  addr_kernel_functor(scalar_t alpha_val) : alpha_val(alpha_val) {}

 private:
  scalar_t alpha_val;
};

template <typename scalar_t>
struct addr_kernel_functor_2 {
  scalar_t operator()(scalar_t self_val, scalar_t vec1_val, scalar_t vec2_val)
      const {
    return (beta_val && self_val) || (alpha_val && vec1_val && vec2_val);
  }

  addr_kernel_functor_2(scalar_t alpha_val, scalar_t beta_val)
      : alpha_val(alpha_val), beta_val(beta_val) {}

 private:
  scalar_t alpha_val;
  scalar_t beta_val;
};

template <typename scalar_t>
struct addr_kernel_functor_3 {
  scalar_t operator()(scalar_t self_val, scalar_t vec1_val, scalar_t vec2_val)
      const {
    return alpha_val * vec1_val * vec2_val;
  }
  addr_kernel_functor_3(scalar_t alpha_val) : alpha_val(alpha_val) {}

 private:
  scalar_t alpha_val;
};

template <typename scalar_t>
struct addr_kernel_functor_4 {
  scalar_t operator()(scalar_t self_val, scalar_t vec1_val, scalar_t vec2_val)
      const {
    return beta_val * self_val + alpha_val * vec1_val * vec2_val;
  }

  addr_kernel_functor_4(scalar_t alpha_val, scalar_t beta_val)
      : alpha_val(alpha_val), beta_val(beta_val) {}

 private:
  scalar_t alpha_val;
  scalar_t beta_val;
};

void addr_kernel(
    TensorIterator& iter,
    const Scalar& beta,
    const Scalar& alpha) {
  if (iter.dtype() == at::ScalarType::Bool) {
    using scalar_t = bool;
    auto beta_val = beta.to<scalar_t>();
    auto alpha_val = alpha.to<scalar_t>();

    // when beta is false, values in self should be ignored,
    // nans and infs in self should not propagate.
    if (beta_val == false) {
      addr_kernel_functor<scalar_t> f(alpha_val);
      dpcpp_kernel_for_tensor_iter(iter, f);
    } else {
      addr_kernel_functor_2<scalar_t> f(alpha_val, beta_val);
      dpcpp_kernel_for_tensor_iter(iter, f);
    }
    return;
  }

  IPEX_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(
      kBFloat16, kHalf, kBool, iter.dtype(), "addr", [&] {
        auto beta_val = beta.to<scalar_t>();
        auto alpha_val = alpha.to<scalar_t>();

        scalar_t zero_val(0);
        // when beta==0, values in self should be ignored,
        // nans and infs in self should not propagate.
        if (beta_val == zero_val) {
          addr_kernel_functor_3<scalar_t> f(alpha_val);
          dpcpp_kernel_for_tensor_iter(iter, f);
        } else {
          addr_kernel_functor_4<scalar_t> f(alpha_val, beta_val);
          dpcpp_kernel_for_tensor_iter(iter, f);
        }
      });
}

} // namespace impl

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
  check_1d(vec1, "vec1", "addr");
  check_1d(vec2, "vec2", "addr");

  Tensor result;

  const auto vec1_size0 = vec1.sizes()[0];
  const auto vec2_size0 = vec2.sizes()[0];
  auto self_ = &result == &self
      ? c10::MaybeOwned<Tensor>::borrowed(self)
      : expand_size(self, {vec1_size0, vec2_size0}, "addr");
  TORCH_CHECK(
      self_->dim() == 2,
      "2D tensor expected, got ",
      self_->dim(),
      "D tensor for input");
  TORCH_CHECK(
      self_->sizes()[0] == vec1_size0 && self_->sizes()[1] == vec2_size0,
      "size mismatch, input: ",
      self_->sizes(),
      ", v1: ",
      vec1.sizes(),
      ", v2: ",
      vec2.sizes());

  auto iter = TensorIteratorConfig()
                  .set_check_mem_overlap(true)
                  .add_output(result)
                  .add_owned_input(*self_)
                  .add_owned_input(vec1.reshape({vec1_size0, 1}))
                  .add_input(vec2)
                  .allow_cpu_scalars(true)
                  .promote_inputs_to_common_dtype(true)
                  .cast_common_dtype_to_outputs(true)
                  .enforce_safe_casting_to_output(true)
                  .build();

  check_addr_scalar(iter.dtype(), beta, "beta");
  check_addr_scalar(iter.dtype(), alpha, "alpha");

  impl::addr_kernel(iter, beta, alpha);
  return iter.output();
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

// std::tuple<Tensor, Tensor, Tensor> _det_lu_based_helper(const Tensor& self) {
//   // fallback to at::native::_det_lu_based_helper
//   return at::native::_det_lu_based_helper(self);
// }

// we consider 6 Taylor expansions of degree
// 1, 2, 4, 8, 12, 18
constexpr int total_n_degs = 6;

Tensor operator_1_norm(const Tensor& tensor) {
  return std::get<0>(tensor.abs().sum(-2).max(-1));
}

static inline void squareCheckInputs(
    const Tensor& self,
    const char* const f_name,
    const char* const arg_name = "A") {
  checkIsMatrix(self, f_name, arg_name);
  TORCH_CHECK(
      self.size(-1) == self.size(-2),
      f_name,
      ": ",
      arg_name,
      " must be batches of square matrices, "
      "but they are ",
      self.size(-2),
      " by ",
      self.size(-1),
      " matrices");
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
  auto tensor = at::from_blob(
                    (void*)blob.begin(),
                    blob.size(),
                    c10::toRealValueType(in.scalar_type()))
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

template <typename scalar_t>
inline Tensor _linear_combination(
    const Tensor& t,
    std::initializer_list<scalar_t> blob) {
  // _blob_to_Tensor converts blob to a 2D tensor for
  // _compute_linear_combination. If this tensor is of shape (1, *), the result
  // of _compute_linear_combination is going to be of shape (1, *t.shape) so we
  // squeeze(0) so that for any t with t.dim() >= 1: t.dim() ==
  // _compute_linear_combination(t, ...).dim().
  return AtenIpexTypeXPU::_compute_linear_combination(
             t, _blob_to_Tensor<scalar_t>(blob, t))
      .squeeze(0);
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
      _linear_combination<scalar_t>(
          As.narrow(0, 0, 3), {1 / 2.0, 1 / 6.0, 1 / 24.0}),
      out_for_a2);

  // I + A + A^2 * (I / 2 + A / 6 + A^2 / 24)
  return _linear_combination<scalar_t>(As, {1.0, 1.0, 0.0, 1.0});
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
      _linear_combination<scalar_t>(
          // extract {A, A^2} from As
          As.narrow(0, 1, 2),
          {x1, x2}),
      out_for_a4);

  // output for A8
  auto out_for_a8 = As.select(0, 4);
  // A8 = (x3 * A2 + A4) * (x4 * I + x5 * A + x6 * A2 + x7 * A4)
  at::native::matmul_out(
      // x3 * A2 + A4
      _linear_combination<scalar_t>(As.narrow(0, 2, 2), {x3, 1.0}),
      _linear_combination<scalar_t>(As.narrow(0, 0, 4), {x4, x5, x6, x7}),
      out_for_a8);

  // return I + A + y2 * A2 + A8;
  return _linear_combination<scalar_t>(As, {1.0, 1.0, y2, 0.0, 1.0});
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
      c10::toRealValueType(A.scalar_type()));
  bs = _move_memory_if_xpu_input(bs, A);

  auto As = _allocate_buffer(A, num_prods);
  _fill_matrix_powers(As, A, num_prods);

  auto Bs = AtenIpexTypeXPU::_compute_linear_combination(As, bs);

  // output for A6
  auto out_for_a6 = As.select(0, 0);
  // compute A6
  Bs.select(0, 2).add_(
      at::native::matmul_out(Bs.select(0, 3), Bs.select(0, 3), out_for_a6));

  // tmp buffer for this matrix product
  return Bs.select(0, 0).add_(at::native::matmul_out(
      Bs.select(0, 1).add_(Bs.select(0, 2)), Bs.select(0, 2), out_for_a6));
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
      c10::toRealValueType(A.scalar_type()));
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
  return Bs.select(0, 1).add_(at::native::matmul_out(
      Bs.select(0, 2).add_(Bs.select(0, 3)), Bs.select(0, 3), out_for_a9));
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
Tensor linalg_matrix_exp(const Tensor& a) {
  squareCheckInputs(a, "linalg.matrix_exp");
  checkFloatingOrComplex(a, "matrix_exp");

  NoTF32Guard disable_tf32;

  // Trivial cases
  const auto n = a.size(-1);
  if (n == 0) {
    return a.clone();
  } else if (n == 1) {
    return a.exp();
  } else {
    return AtenIpexTypeXPU::mexp(a);
  }
}

// Alias
Tensor matrix_exp(const Tensor& a) {
  return AtenIpexTypeXPU::linalg_matrix_exp(a);
}

} // namespace AtenIpexTypeXPU
} // namespace at
