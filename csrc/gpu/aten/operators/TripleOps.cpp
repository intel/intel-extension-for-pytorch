#include <ATen/ATen.h>
#include <ATen/Context.h>
#include <ATen/OpMathType.h>
#include <ATen/autocast_mode.h>
#include <ATen/native/Activation.h>
#include <ATen/native/BinaryOps.h>
#include <ATen/native/SparseTensorUtils.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/record_function.h>
#include <core/detail/ListUtils.h>
#include <gpu/aten/tensor/Tensor.h>
#include <math.h>
#include <oneDNN/oneDNN.h>
#include <runtime/Utils.h>
#include <sys/types.h>
#include <utils/DPCPP.h>
#include <cstdint>
#include "comm/Numerics.h"

#include "Loops.h"
#include "PSTLFunctions.h"
#include "SparseTensorUtils.h"
#include "comm/AccumulateType.h"
#include "comm/Pointwise.h"
#include "comm/RegistrationDeclarations.h"

#include "utils/ComputeEngine.h"
#include "utils/CustomOperatorRegistration.h"

using namespace torch_ipex::xpu::dpcpp;
using namespace at::sparse;

namespace at {
namespace AtenIpexTypeXPU {
using autocast::cached_cast;
using autocast::get_lower_precision_fp_from_device_type;
using autocast::promote_type;

std::tuple<Tensor, Tensor> sort(
    const Tensor& self,
    int64_t dim,
    bool descending);

namespace impl {

template <typename scalar_t, typename accscalar_t>
struct mul_add_kernel_dpcpp_functor {
  scalar_t operator()(scalar_t a, scalar_t b, scalar_t c) const {
    return a * b + alpha * c;
  }

  mul_add_kernel_dpcpp_functor(accscalar_t alpha) : alpha(alpha) {}

 private:
  accscalar_t alpha;
};

template <typename scalar_t, typename accscalar_t>
struct silu_mul_dpcpp_functor {
  scalar_t operator()(scalar_t a, scalar_t b) const {
    return (accscalar_t(a)) / (1.0f + expf(accscalar_t(-a))) * accscalar_t(b);
  }
};

template <typename scalar_t, typename accscalar_t>
struct gelu_erf_mul_dpcpp_functor {
  scalar_t operator()(scalar_t a, scalar_t b) const {
    return (accscalar_t(a) * accscalar_t(0.5) *
            (accscalar_t(1) + ::erf(accscalar_t(a) * accscalar_t(M_SQRT1_2)))) *
        accscalar_t(b);
  }
};

template <typename scalar_t, typename accscalar_t>
struct gelu_tanh_mul_dpcpp_functor {
  scalar_t operator()(scalar_t a, scalar_t b) const {
    constexpr accscalar_t kBeta = M_SQRT2 * M_2_SQRTPI * accscalar_t(0.5);
    constexpr accscalar_t kKappa = 0.044715;
    auto x_cube = accscalar_t(a) * accscalar_t(a) * accscalar_t(a);
    auto inner = kBeta * (accscalar_t(a) + kKappa * x_cube);
    return (accscalar_t(0.5) * accscalar_t(a) *
            (accscalar_t(1) + Numerics<accscalar_t>::tanh(inner))) *
        accscalar_t(b);
  }
};

template <typename scalar_t, typename func_t, int N>
struct op_and_mul_functor {
  void operator()(sycl::nd_item<1> item) const {
    using accscalar_t = at::opmath_type<scalar_t>;
    int64_t offset = item.get_local_linear_id();
    int64_t step = item.get_local_range(0);
    int64_t token_id = item.get_group(0);
    func_t fn;
    int64_t bound = dim / N;
    for (int64_t i = offset; i < bound; i += step) {
      auto unary_val =
          reinterpret_cast<Memory::aligned_vector_loop<scalar_t, N>*>(
              input_ptr)[token_id * bound * 2 + i];
      auto mul_val =
          reinterpret_cast<Memory::aligned_vector_loop<scalar_t, N>*>(
              input_ptr)[token_id * bound * 2 + i + bound];
#pragma unroll
      for (int i = 0; i < N; ++i) {
        unary_val[i] = fn(unary_val[i], mul_val[i]);
      }
      reinterpret_cast<Memory::aligned_vector_loop<scalar_t, N>*>(
          output_ptr)[token_id * bound + i] = unary_val;
    }
  }

  scalar_t* input_ptr;
  scalar_t* output_ptr;
  int64_t num_;
  int64_t dim;
};

static void mul_add_kernel_dpcpp(TensorIterator& iter, Scalar alpha_scalar) {
  IPEX_DISPATCH_ALL_TYPES_AND_COMPLEX_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      iter.dtype(),
      "mul_add",
      [&]() {
        using accscalar_t = acc_type<scalar_t>;
        auto alpha = alpha_scalar.to<accscalar_t>();
        mul_add_kernel_dpcpp_functor<scalar_t, accscalar_t> f(alpha);
        dpcpp_kernel_for_tensor_iter(iter, f);
      });
}

std::vector<int64_t> dim_expand(
    const Tensor& a_tensor,
    const Tensor& b_tensor,
    const Tensor& c_tensor,
    Tensor& view_a,
    Tensor& view_b,
    Tensor& view_c) {
  IntArrayRef a = a_tensor.sizes();
  IntArrayRef b = b_tensor.sizes();
  IntArrayRef c = c_tensor.sizes();
  size_t dimA = a.size();
  size_t dimB = b.size();
  size_t dimC = c.size();
  size_t max_dim = dimA > dimB ? dimA : dimB;
  max_dim = max_dim > dimC ? max_dim : dimC;
  std::vector<int64_t> view_size_a(max_dim);
  std::vector<int64_t> view_size_b(max_dim);
  std::vector<int64_t> view_size_c(max_dim);
  std::vector<int64_t> target_size(max_dim);

  for (int i = max_dim - 1; i >= 0; --i) {
    int offset = max_dim - 1 - i;
    int a_idx = dimA - 1 - offset;
    int b_idx = dimB - 1 - offset;
    int c_idx = dimC - 1 - offset;
    int sizeA = (a_idx >= 0) ? a[a_idx] : 1;
    int sizeB = (b_idx >= 0) ? b[b_idx] : 1;
    int sizeC = (c_idx >= 0) ? c[c_idx] : 1;

    view_size_a[i] = sizeA;
    view_size_b[i] = sizeB;
    view_size_c[i] = sizeC;
    target_size[i] = sizeA == 1 ? (sizeB == 1 ? sizeC : sizeB) : sizeA;
  }
  view_a = dimA != max_dim ? at::native::view(a_tensor, view_size_a) : a_tensor;
  view_b = dimB != max_dim ? at::native::view(b_tensor, view_size_b) : b_tensor;
  view_c = dimC != max_dim ? at::native::view(c_tensor, view_size_c) : c_tensor;

  int64_t target_numel = std::accumulate(
      target_size.begin(), target_size.end(), 1, [](int64_t a, int64_t b) {
        return a * b;
      });
  view_a = view_a.numel() == target_numel ? view_a : view_a.expand(target_size);
  view_b = view_b.numel() == target_numel ? view_b : view_b.expand(target_size);
  view_c = view_c.numel() == target_numel ? view_c : view_c.expand(target_size);
  return target_size;
}

} // namespace impl

bool check_opaque(std::vector<Tensor> tensor_list) {
  for (auto& tensor : tensor_list) {
    if (torch_ipex::xpu::oneDNN::is_onednn_layout(tensor))
      return true;
  }
  return false;
}

template <typename scalar_t, typename accscalar_t>
struct mul_scalar_add_scalar_functor {
  scalar_t operator()(scalar_t a) const {
    return a * other_scalar + add_scalar;
  }

  mul_scalar_add_scalar_functor(
      accscalar_t add_scalar,
      accscalar_t other_scalar)
      : add_scalar(add_scalar), other_scalar(other_scalar) {}

 private:
  accscalar_t add_scalar;
  accscalar_t other_scalar;
};

Tensor silu_mul(const Tensor& self, const Tensor& other, Tensor& out) {
  auto real_eng = choose_compute_eng(torch_ipex::xpu::COMPUTE_ENG::BASIC, self);
  if (torch_ipex::xpu::COMPUTE_ENG::ONEDNN == real_eng) {
    AtenIpexTypeXPU::silu_out(self, out);
    out = AtenIpexTypeXPU::mul(out, other);
  } else {
    auto iter = TensorIteratorConfig()
                    .set_check_mem_overlap(true)
                    .promote_inputs_to_common_dtype(true)
                    .cast_common_dtype_to_outputs(true)
                    .enforce_safe_casting_to_output(true)
                    .promote_integer_inputs_to_float(true)
                    .add_output(out)
                    .add_input(self)
                    .add_input(other)
                    .build();
    IPEX_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        iter.dtype(),
        "silu_mul",
        [&]() {
          using accscalar_t = acc_type<scalar_t>;
          typename impl::silu_mul_dpcpp_functor<scalar_t, accscalar_t> f;
          dpcpp_kernel_for_tensor_iter(iter, f);
        });
  }
  return out;
}

Tensor gelu_mul(
    const Tensor& self,
    const Tensor& other,
    Tensor& out,
    c10::string_view approximate) {
  auto real_eng = choose_compute_eng(torch_ipex::xpu::COMPUTE_ENG::BASIC, self);
  if (torch_ipex::xpu::COMPUTE_ENG::ONEDNN == real_eng) {
    AtenIpexTypeXPU::gelu_out(self, approximate, out);
    out = AtenIpexTypeXPU::mul(out, other);
  } else {
    auto _approximate = at::native::get_gelutype_enum(approximate);
    auto iter = TensorIteratorConfig()
                    .set_check_mem_overlap(true)
                    .promote_inputs_to_common_dtype(true)
                    .cast_common_dtype_to_outputs(true)
                    .enforce_safe_casting_to_output(true)
                    .promote_integer_inputs_to_float(true)
                    .add_output(out)
                    .add_input(self)
                    .add_input(other)
                    .build();
    if (_approximate == at::native::GeluType::Tanh) {
      IPEX_DISPATCH_FLOATING_TYPES_AND2(
          at::ScalarType::Half,
          at::ScalarType::BFloat16,
          iter.dtype(),
          "gelu_mul_tanh",
          [&]() {
            using accscalar_t = acc_type<scalar_t>;
            typename impl::gelu_tanh_mul_dpcpp_functor<scalar_t, accscalar_t> f;
            dpcpp_kernel_for_tensor_iter(iter, f);
          });
    } else {
      IPEX_DISPATCH_ALL_TYPES_AND2(
          at::ScalarType::Half,
          at::ScalarType::BFloat16,
          iter.dtype(),
          "gelu_mul_erf",
          [&]() {
            using accscalar_t = acc_type<scalar_t>;
            typename impl::gelu_erf_mul_dpcpp_functor<scalar_t, accscalar_t> f;
            dpcpp_kernel_for_tensor_iter(iter, f);
          });
    }
  }
  return out;
}

#define VEC_LAUNCH(KERNEL, N)                                              \
  case N: {                                                                \
    auto cgf = DPCPP_Q_CGF(cgh) {                                          \
      using accscalar_t = at::opmath_type<scalar_t>;                       \
      impl::op_and_mul_functor<scalar_t, KERNEL<scalar_t, accscalar_t>, N> \
          kfn = {                                                          \
              .input_ptr = self.data_ptr<scalar_t>(),                      \
              .output_ptr = out.data_ptr<scalar_t>(),                      \
              .num_ = numel,                                               \
              .dim = dim};                                                 \
      cgh.parallel_for<decltype(kfn)>(                                     \
          sycl::nd_range<1>(num_group * wg_size, wg_size), kfn);           \
    };                                                                     \
    DPCPP_Q_SUBMIT(queue, cgf);                                            \
    break;                                                                 \
  }

#define OP_AND_MUL_FUSION(KERNEL, KERNEL_NAME)                         \
  IPEX_DISPATCH_FLOATING_TYPES_AND2(                                   \
      at::ScalarType::Half,                                            \
      at::ScalarType::BFloat16,                                        \
      self.scalar_type(),                                              \
      KERNEL_NAME,                                                     \
      [=]() {                                                          \
        auto queue = dpcppGetCurrentQueue();                           \
        auto dev_id = dpcppGetDeviceIdOfCurrentQueue();                \
        int64_t max_wg_size = dpcppMaxWorkGroupSize(dev_id);           \
        int64_t max_group_num =                                        \
            dpcppMaxWorkItemsPerTile(dev_id) / max_wg_size;            \
        int64_t numel = out.numel();                                   \
        int64_t dim = out.size(-1);                                    \
        int64_t tokens = numel / dim;                                  \
        int64_t wg_size = std::min(dim, max_wg_size);                  \
        int64_t num_group = tokens;                                    \
        int vec_size = sizeof(float) * 4 / sizeof(scalar_t);           \
        while ((vec_size >> 1) * wg_size >= dim) {                     \
          vec_size = vec_size >> 1;                                    \
        }                                                              \
        if (dim % vec_size != 0)                                       \
          vec_size = 1;                                                \
        switch (vec_size) {                                            \
          VEC_LAUNCH(KERNEL, 1);                                       \
          VEC_LAUNCH(KERNEL, 2);                                       \
          VEC_LAUNCH(KERNEL, 4);                                       \
          VEC_LAUNCH(KERNEL, 8);                                       \
          VEC_LAUNCH(KERNEL, 16);                                      \
          default:                                                     \
            TORCH_CHECK(false, "Unsupported vector size: ", vec_size); \
        }                                                              \
      });

Tensor gelu_and_mul(Tensor& self, Tensor& out, c10::string_view approximate) {
  auto _approximate = at::native::get_gelutype_enum(approximate);
  TORCH_CHECK(
      self.size(-1) / 2 == out.size(-1),
      "input tensor's last dim shoule be the 2 time larger than the output tensor's last dim");
  self = self.contiguous();
  out = out.contiguous();
  if (_approximate == at::native::GeluType::Tanh) {
    OP_AND_MUL_FUSION(impl::gelu_tanh_mul_dpcpp_functor, "gelu_and_mul_tanh");
  } else {
    OP_AND_MUL_FUSION(impl::gelu_erf_mul_dpcpp_functor, "gelu_and_mul_erf");
  }
  return out;
}

Tensor silu_and_mul(Tensor& self, Tensor& out) {
  // auto _approximate = at::native::get_gelutype_enum(approximate);
  TORCH_CHECK(
      self.size(-1) / 2 == out.size(-1),
      "input tensor's last dim shoule be the 2 time larger than the output tensor's last dim");
  self = self.contiguous();
  out = out.contiguous();
  OP_AND_MUL_FUSION(impl::silu_mul_dpcpp_functor, "silu_and_mul");
  return out;
}

Tensor mul_scalar_add_scalar(
    const Tensor& self,
    Scalar other,
    Scalar accumu,
    Scalar alpha) {
  Tensor result;
  auto real_eng = choose_compute_eng(torch_ipex::xpu::COMPUTE_ENG::BASIC, self);
  if (torch_ipex::xpu::COMPUTE_ENG::ONEDNN == real_eng) {
    result = AtenIpexTypeXPU::mul(self, other);
    result = AtenIpexTypeXPU::add(result, accumu, alpha);
  } else {
    result = at::empty_like(self);
    auto iter = TensorIteratorConfig()
                    .set_check_mem_overlap(true)
                    .add_output(result)
                    .add_input(self)
                    .build();
    IPEX_DISPATCH_ALL_TYPES_AND_COMPLEX_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        iter.dtype(),
        "mul_scalar_add_scalar",
        [&]() {
          using accscalar_t = acc_type<scalar_t>;
          auto add_scalar = alpha.to<accscalar_t>() * accumu.to<accscalar_t>();
          auto other_scalar = other.to<accscalar_t>();
          mul_scalar_add_scalar_functor<scalar_t, accscalar_t> f(
              add_scalar, other_scalar);
          dpcpp_kernel_for_tensor_iter(iter, f);
        });
  }
  return result;
}

Tensor mul_scalar_add_scalar_autocast(
    const Tensor& self,
    Scalar other,
    Scalar accumu,
    Scalar alpha) {
  c10::impl::ExcludeDispatchKeyGuard no_autocast(c10::DispatchKey::AutocastXPU);
  return mul_scalar_add_scalar(self, other, accumu, alpha);
}

template <typename scalar_t, typename accscalar_t>
struct mul_add_scalar_functor {
  scalar_t operator()(scalar_t a, scalar_t b) const {
    return a * b + add_scalar;
  }
  mul_add_scalar_functor(accscalar_t add_scalar) : add_scalar(add_scalar) {}

 private:
  accscalar_t add_scalar;
};

Tensor mul_add_scalar(
    const Tensor& self,
    const Tensor& other,
    Scalar accumu,
    Scalar alpha) {
  Tensor result;
  auto real_eng =
      choose_compute_eng(torch_ipex::xpu::COMPUTE_ENG::BASIC, self, other);
  if (torch_ipex::xpu::COMPUTE_ENG::ONEDNN == real_eng) {
    result = AtenIpexTypeXPU::mul(self, other);
    return AtenIpexTypeXPU::add(result, accumu, alpha);
  }

  auto iter = TensorIteratorConfig()
                  .set_check_mem_overlap(true)
                  .promote_inputs_to_common_dtype(true)
                  .cast_common_dtype_to_outputs(true)
                  .enforce_safe_casting_to_output(true)
                  .promote_integer_inputs_to_float(true)
                  .add_output(result)
                  .add_input(self)
                  .add_input(other)
                  .build();
  IPEX_DISPATCH_ALL_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      iter.dtype(),
      "mul_add_scalar",
      [&]() {
        using accscalar_t = acc_type<scalar_t>;
        auto add_scalar = alpha.to<accscalar_t>() * accumu.to<accscalar_t>();
        mul_add_scalar_functor<scalar_t, accscalar_t> f(add_scalar);
        dpcpp_kernel_for_tensor_iter(iter, f);
      });
  result = iter.output();
  return result;
}

Tensor mul_add_scalar_autocast(
    const Tensor& self,
    const Tensor& other,
    Scalar accumu,
    Scalar alpha) {
  c10::impl::ExcludeDispatchKeyGuard no_autocast(c10::DispatchKey::AutocastXPU);
  auto to_type = promote_type(
      get_lower_precision_fp_from_device_type(c10::DeviceType::XPU),
      c10::DeviceType::XPU,
      self,
      other);
  return mul_add_scalar(
      cached_cast(to_type, self, c10::DeviceType::XPU),
      cached_cast(to_type, other, c10::DeviceType::XPU),
      accumu,
      alpha);
}

template <typename scalar_t, typename accscalar_t>
struct mul_scalar_add_functor {
  scalar_t operator()(scalar_t a, scalar_t b) const {
    return a * other_scalar + b * alpha_scalar;
  }
  mul_scalar_add_functor(accscalar_t alpha_scalar, accscalar_t other_scalar)
      : alpha_scalar(alpha_scalar), other_scalar(other_scalar) {}

 private:
  accscalar_t alpha_scalar;
  accscalar_t other_scalar;
};

Tensor mul_scalar_add(
    const Tensor& self,
    Scalar other,
    const Tensor& accumu,
    Scalar alpha) {
  Tensor result;
  torch_ipex::xpu::COMPUTE_ENG real_eng =
      choose_compute_eng(torch_ipex::xpu::COMPUTE_ENG::BASIC, self, accumu);
  if (torch_ipex::xpu::COMPUTE_ENG::ONEDNN == real_eng) {
    result = AtenIpexTypeXPU::mul(self, other);
    result = AtenIpexTypeXPU::add(result, accumu, alpha);
  } else {
    auto iter = TensorIteratorConfig()
                    .set_check_mem_overlap(true)
                    .promote_inputs_to_common_dtype(true)
                    .cast_common_dtype_to_outputs(true)
                    .enforce_safe_casting_to_output(true)
                    .promote_integer_inputs_to_float(true)
                    .add_output(result)
                    .add_input(self)
                    .add_input(accumu)
                    .build();
    IPEX_DISPATCH_ALL_TYPES_AND_COMPLEX_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        iter.dtype(),
        "mul_scalar_add",
        [&]() {
          using accscalar_t = acc_type<scalar_t>;
          auto alpha_scalar = alpha.to<accscalar_t>();
          auto other_scalar = other.to<accscalar_t>();
          mul_scalar_add_functor<scalar_t, accscalar_t> f(
              alpha_scalar, other_scalar);
          dpcpp_kernel_for_tensor_iter(iter, f);
        });
    result = iter.output();
  }
  return result;
}

Tensor mul_scalar_add_autocast(
    const Tensor& self,
    Scalar other,
    const Tensor& accumu,
    Scalar alpha) {
  c10::impl::ExcludeDispatchKeyGuard no_autocast(c10::DispatchKey::AutocastXPU);
  auto to_type = promote_type(
      get_lower_precision_fp_from_device_type(c10::DeviceType::XPU),
      c10::DeviceType::XPU,
      self,
      accumu);
  return mul_scalar_add(
      cached_cast(to_type, self, c10::DeviceType::XPU),
      other,
      cached_cast(to_type, accumu, c10::DeviceType::XPU),
      alpha);
}

Tensor mul_add(
    const Tensor& self,
    const Tensor& other,
    const Tensor& accumu,
    Scalar alpha) {
  Tensor result;
  torch_ipex::xpu::COMPUTE_ENG real_eng = choose_compute_eng(
      torch_ipex::xpu::COMPUTE_ENG::BASIC, self, other, accumu);
  if (torch_ipex::xpu::COMPUTE_ENG::ONEDNN == real_eng) {
    result = AtenIpexTypeXPU::mul(self, other);
    result = AtenIpexTypeXPU::add(result, accumu, alpha);
  } else {
    auto iter = TensorIteratorConfig()
                    .set_check_mem_overlap(true)
                    .promote_inputs_to_common_dtype(true)
                    .cast_common_dtype_to_outputs(true)
                    .enforce_safe_casting_to_output(true)
                    .promote_integer_inputs_to_float(true)
                    .add_output(result)
                    .add_input(self)
                    .add_input(other)
                    .add_input(accumu)
                    .build();
    impl::mul_add_kernel_dpcpp(iter, alpha);
    result = iter.output();
  }
  return result;
}

Tensor mul_add_autocast(
    const Tensor& self,
    const Tensor& other,
    const Tensor& accumu,
    Scalar alpha) {
  c10::impl::ExcludeDispatchKeyGuard no_autocast(c10::DispatchKey::AutocastXPU);
  auto to_type = promote_type(
      get_lower_precision_fp_from_device_type(c10::DeviceType::XPU),
      c10::DeviceType::XPU,
      self,
      other,
      accumu);
  return mul_add(
      cached_cast(to_type, self, c10::DeviceType::XPU),
      cached_cast(to_type, other, c10::DeviceType::XPU),
      cached_cast(to_type, accumu, c10::DeviceType::XPU),
      alpha);
}

template <typename scalar_t>
struct PackedAddKernelFunctor {
  union packed_bf16 {
    unsigned short s[2];
    float f;
  };

  void operator()(sycl::item<1> item) const {
    int64_t gid = item.get_linear_id();
    auto MSB_p = MSB_data;
    auto LSB_p = LSB_data;
    auto gw_p = gw_data;

    packed_bf16 p16;
    p16.s[0] = LSB_p[gid];
    p16.s[1] = MSB_p[gid];
    p16.f += lr * (float)(gw_p[gid]);
    LSB_p[gid] = p16.s[0];
    MSB_p[gid] = p16.s[1];
  }
  PackedAddKernelFunctor(
      unsigned short* __restrict__ MSB_data_,
      unsigned short* __restrict__ LSB_data_,
      const at::BFloat16* __restrict__ gw_data_,
      int num_elem_,
      float lr_)
      : MSB_data(MSB_data_),
        LSB_data(LSB_data_),
        gw_data(gw_data_),
        num_elem(num_elem_),
        lr(lr_) {}

 private:
  unsigned short* __restrict__ MSB_data;
  unsigned short* __restrict__ LSB_data;
  const at::BFloat16* __restrict__ gw_data;
  int num_elem;
  float lr;
};

template <typename scalar_t>
static inline void packed_add_kernel(
    unsigned short* __restrict__ w_MSB,
    unsigned short* __restrict__ w_LSB,
    const at::BFloat16* __restrict__ gw,
    int num_elem,
    float lr) {
  auto& dpcpp_queue = dpcppGetCurrentQueue();

  auto cgf = DPCPP_Q_CGF(cgh) {
    auto MSB_data = w_MSB;
    auto LSB_data = w_LSB;
    auto gw_data = gw;
    PackedAddKernelFunctor<scalar_t> kfn(
        MSB_data, LSB_data, gw_data, num_elem, lr);
    cgh.parallel_for<decltype(kfn)>(sycl::range<1>(num_elem), kfn);
  };
  DPCPP_Q_SUBMIT(dpcpp_queue, cgf);
}

template <typename scalar_t, typename accscalar_t>
struct SparsePackedAddKernelFunctor {
  union packed_bf16 {
    unsigned short s[2];
    float f;
  };

  void operator()(sycl::nd_item<2> item) const {
    auto MSB_p = w_MSB;
    auto LSB_p = w_LSB;
    auto uniqueOffsets_ptr = uniqueOffsets;
    auto origIndices_ptr = origIndices;
    auto values_ptr = values;
    auto indices1D_ptr = indices1D;
    // auto newValues_ptr = newValues_data;

    int seg = item.get_global_id()[0];

    if (seg < newNnz) {
      const int newValueRow = seg * stride;
      const int begin = uniqueOffsets_ptr[seg];
      const int end = (seg < newNnz - 1) ? uniqueOffsets_ptr[seg + 1] : nnz;
      const int featureDim = item.get_global_id()[1];

      accscalar_t tmp = 0;
      for (int row = begin; row < end; row++) {
        const int valueRow = ((int)origIndices_ptr[row]) * stride;
        if (featureDim < stride) {
          tmp += static_cast<accscalar_t>(values_ptr[valueRow + featureDim]);
        }
      }
      if (featureDim < stride) {
        const int weight_index = indices1D_ptr[seg] * stride + featureDim;
        packed_bf16 p16;
        p16.s[0] = LSB_p[weight_index];
        p16.s[1] = MSB_p[weight_index];
        p16.f += lr * (float)(tmp);
        LSB_p[weight_index] = p16.s[0];
        MSB_p[weight_index] = p16.s[1];
        // newValues_ptr[newValueRow + featureDim] =
        // static_cast<scalar_t>(tmp);
      }
    }
  }
  SparsePackedAddKernelFunctor(
      unsigned short* __restrict__ w_MSB_,
      unsigned short* __restrict__ w_LSB_,
      const at::BFloat16* __restrict__ values_,
      int64_t* indices1D_,
      int64_t* origIndices_,
      int64_t* uniqueOffsets_,
      int64_t stride_,
      int64_t nnz_,
      float lr_,
      int64_t newNnz_)
      : w_MSB(w_MSB_),
        w_LSB(w_LSB_),
        values(values_),
        indices1D(indices1D_),
        origIndices(origIndices_),
        uniqueOffsets(uniqueOffsets_),
        stride(stride_),
        nnz(nnz_),
        lr(lr_),
        newNnz(newNnz_) {}

 private:
  unsigned short* __restrict__ w_MSB;
  unsigned short* __restrict__ w_LSB;
  const at::BFloat16* __restrict__ values;
  int64_t* indices1D;
  int64_t* origIndices;
  int64_t* uniqueOffsets;
  int64_t stride;
  int64_t nnz;
  float lr;
  int64_t newNnz;
};

struct sparse_packed_add_kernel_functor {
  template <typename T>
  auto operator()(T lhs, T rhs) const {
    return lhs == rhs;
  }
};

template <typename scalar_t>
static inline void sparse_packed_add_kernel(
    unsigned short* __restrict__ w_MSB,
    unsigned short* __restrict__ w_LSB,
    const at::BFloat16* __restrict__ values,
    int64_t* indices1D,
    int64_t* origIndices,
    int64_t* uniqueOffsets,
    int64_t stride,
    int64_t nnz,
    float lr) {
  using accscalar_t = acc_type<scalar_t>;

  auto& dpcpp_queue = dpcppGetCurrentQueue();
  int64_t newNnz;
  auto indices1D_end = indices1D;
  auto uniqueOffsets_end = uniqueOffsets;
  sparse_packed_add_kernel_functor f;
  std::tie(indices1D_end, uniqueOffsets_end) =
      torch_ipex::xpu::pstl::unique_with_zip<int64_t, int64_t, int64_t>(
          indices1D, indices1D + nnz, uniqueOffsets, f);
  newNnz = std::distance(indices1D, indices1D_end);

  const int num_group_0 = CeilDiv(newNnz, (int64_t)4);
  const int num_group_1 = CeilDiv(stride, (int64_t)64);

  auto cgf = DPCPP_Q_CGF(cgh) {
    // auto newValues_data = newValues.data_ptr<scalar_t>();
    SparsePackedAddKernelFunctor<scalar_t, accscalar_t> kfn(
        w_MSB,
        w_LSB,
        values,
        indices1D,
        origIndices,
        uniqueOffsets,
        stride,
        nnz,
        lr,
        newNnz);

    // kick off kernel
    cgh.parallel_for<decltype(kfn)>(
        sycl::nd_range<2>(
            sycl::range<2>(num_group_0 * 4, num_group_1 * 64),
            sycl::range<2>(4, 64)),
        kfn);
  };
  DPCPP_Q_SUBMIT(dpcpp_queue, cgf);
}

Tensor packed_add(
    at::Tensor& top_half,
    at::Tensor& bot_half,
    const at::Tensor& grad,
    double alpha) {
  RECORD_FUNCTION(
      "packed_add", std::vector<c10::IValue>({top_half, bot_half, grad}));
  if (grad.is_sparse()) {
    Tensor values = grad._values();
    Tensor indices = grad._indices();
    int64_t nDim = top_half.dim();
    int64_t nDimI = grad.sparse_dim();
    const int64_t nnz = grad._nnz();
    Tensor indices1D =
        AtenIpexTypeSparseXPU::flatten_indices(indices, grad.sizes(), 0);
    int64_t view_rows = 1;
    int64_t view_columns = 1;
    for (int i = 0; i < nDimI; i++) {
      view_rows *= top_half.size(i);
    }
    for (int i = nDimI; i < nDim; i++) {
      view_columns *= top_half.size(i);
    }

    Tensor top_half_view = top_half.view({view_rows, view_columns});
    Tensor bot_half_view = bot_half.view({view_rows, view_columns});
    values = values.contiguous();
    int64_t stride =
        torch_ipex::xpu::dpcpp::detail::prod_intlist(values.sizes().slice(1));

    Tensor uniqueOffsets = at::arange(0, {nnz}, indices.options());
    Tensor new_indices, origIndices;
    std::tie(new_indices, origIndices) =
        at::AtenIpexTypeXPU::sort(indices1D, 0, false);

    IPEX_DISPATCH_FLOATING_TYPES_AND(
        at::ScalarType::BFloat16,
        top_half.scalar_type(),
        "sparse_packed_add_kernel",
        [&]() {
          sparse_packed_add_kernel<scalar_t>(
              (unsigned short*)top_half.data_ptr<scalar_t>(),
              (unsigned short*)bot_half.data_ptr<scalar_t>(),
              values.data_ptr<at::BFloat16>(),
              new_indices.data_ptr<int64_t>(),
              origIndices.data_ptr<int64_t>(),
              uniqueOffsets.data_ptr<int64_t>(),
              stride,
              nnz,
              static_cast<float>(alpha));
        });
  } else {
    IPEX_DISPATCH_FLOATING_TYPES_AND(
        at::ScalarType::BFloat16,
        top_half.scalar_type(),
        "packed_add_kernel",
        [&]() {
          packed_add_kernel<scalar_t>(
              (unsigned short*)top_half.data_ptr<scalar_t>(),
              (unsigned short*)bot_half.data_ptr<scalar_t>(),
              grad.data_ptr<at::BFloat16>(),
              top_half.numel(),
              static_cast<float>(alpha));
        });
  }
  return top_half;
}

} // namespace AtenIpexTypeXPU
} // namespace at

namespace {
IPEX_LIBRARY_FRAGMENT() {
  IPEX_OP_REGISTER_DISPATCH("mul_add", mul_add, c10::DispatchKey::XPU);
  IPEX_OP_REGISTER_DISPATCH(
      "mul_add", mul_add_autocast, c10::DispatchKey::AutocastXPU);
  IPEX_OP_REGISTER_DISPATCH(
      "mul_add.Scalar_Tensor", mul_scalar_add, c10::DispatchKey::XPU);
  IPEX_OP_REGISTER_DISPATCH(
      "mul_add.Scalar_Tensor",
      mul_scalar_add_autocast,
      c10::DispatchKey::AutocastXPU);
  IPEX_OP_REGISTER_DISPATCH(
      "mul_add.Tensor_Scalar", mul_add_scalar, c10::DispatchKey::XPU);
  IPEX_OP_REGISTER_DISPATCH(
      "mul_add.Tensor_Scalar",
      mul_add_scalar_autocast,
      c10::DispatchKey::AutocastXPU);
  IPEX_OP_REGISTER_DISPATCH(
      "mul_add.Scalar_Scalar", mul_scalar_add_scalar, c10::DispatchKey::XPU);
  IPEX_OP_REGISTER_DISPATCH(
      "mul_add.Scalar_Scalar",
      mul_scalar_add_scalar_autocast,
      c10::DispatchKey::AutocastXPU);
  IPEX_OP_REGISTER_DISPATCH(
      "packed_add", at::AtenIpexTypeXPU::packed_add, c10::DispatchKey::XPU);
  IPEX_OP_REGISTER_DISPATCH(
      "packed_add",
      at::AtenIpexTypeXPU::packed_add,
      c10::DispatchKey::SparseXPU)
  IPEX_OP_REGISTER_DISPATCH(
      "silu_mul", at::AtenIpexTypeXPU::silu_mul, c10::DispatchKey::XPU)
  IPEX_OP_REGISTER_DISPATCH(
      "gelu_mul", at::AtenIpexTypeXPU::gelu_mul, c10::DispatchKey::XPU)
  IPEX_OP_REGISTER_DISPATCH(
      "silu_and_mul", at::AtenIpexTypeXPU::silu_and_mul, c10::DispatchKey::XPU)
  IPEX_OP_REGISTER_DISPATCH(
      "gelu_and_mul", at::AtenIpexTypeXPU::gelu_and_mul, c10::DispatchKey::XPU)
}
} // namespace
