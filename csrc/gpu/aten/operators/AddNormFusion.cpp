#include <ATen/ATen.h>
#include <ATen/Config.h>
#include <ATen/NativeFunctions.h>
#include <ATen/autocast_mode.h>
#include <oneDNN/oneDNN.h>
#include "Norm.h"
#include "comm/RegistrationDeclarations.h"
#include "utils/CustomOperatorRegistration.h"

using namespace torch_ipex::xpu::dpcpp;
using namespace at::AtenIpexTypeXPU::normalization;

namespace at {
namespace AtenIpexTypeXPU {
using autocast::cached_cast;

// Decalre the rms_norm_fwd from RMSNorm.cpp for naive implementation fallback
std::tuple<Tensor, Tensor> rms_norm_fw(
    const Tensor& input,
    at::IntArrayRef normalized_shape,
    const Tensor& weight,
    double epsilon);

namespace impl {

constexpr int float4_size = sizeof(float) * 4;

// Disabled tensors shape check for host overhead optimization
// Only for customized layernorm
inline std::pair<int64_t, int64_t> fast_check_layer_norm_inputs(
    const Tensor& input,
    IntArrayRef normalized_shape,
    const Tensor& weight /* optional */,
    const Tensor& bias /* optional */) {
  const int normalized_ndim = normalized_shape.size();

  const auto input_shape = input.sizes();
  const auto input_ndim = input.dim();

  const int axis = input_ndim - normalized_ndim;
  const int64_t M =
      c10::multiply_integers(input_shape.cbegin(), input_shape.cbegin() + axis);
  const int64_t N =
      c10::multiply_integers(input_shape.cbegin() + axis, input_shape.cend());

  return std::make_pair(M, N);
}

template <
    typename scalar_t,
    typename mean_t,
    typename weight_t,
    typename index_t,
    typename accscalar_t,
    typename vec_t,
    typename weight_vec_t,
    int Num,
    int vec_size,
    bool one_moment = false,
    bool add_back = false>
struct FusedNormKernel1Functor {
  [[intel::reqd_sub_group_size(SIMD)]] void operator()(
      sycl::nd_item<1> item_id) const {
    auto group_id = item_id.get_group(0);
    auto local_id = item_id.get_local_id(0);
    index_t group_offset = group_id * Plane;

    vec_t X_val[Num];
    accscalar_t sum1 = 0;
    accscalar_t sum2 = 0;
    for (int i = 0; i < Num; ++i) {
      index_t plane_offset = (i * workgroup_size + local_id) * vec_size;
      if (plane_offset < Plane) {
        X_val[i] =
            *(reinterpret_cast<vec_t*>(X_data + group_offset + plane_offset));
        if (add1_data != nullptr && add2_data != nullptr) {
          vec_t add1_val = *(reinterpret_cast<vec_t*>(
              add1_data + group_offset + plane_offset));
          vec_t add2_val = *(reinterpret_cast<vec_t*>(
              add2_data + group_offset + plane_offset));
#pragma unroll
          for (int v = 0; v < vec_size; ++v) {
            X_val[i][v] += add1_val[v] + add2_val[v];
            if constexpr (!one_moment) {
              sum1 += static_cast<accscalar_t>(X_val[i][v]);
            }
            sum2 += Numerics<accscalar_t>::pow(X_val[i][v], 2);
          }
          if constexpr (add_back) {
            *(reinterpret_cast<vec_t*>(
                add1_data + group_offset + plane_offset)) = X_val[i];
          }
        } else if (add1_data != nullptr) {
          vec_t add1_val = *(reinterpret_cast<vec_t*>(
              add1_data + group_offset + plane_offset));
#pragma unroll
          for (int v = 0; v < vec_size; ++v) {
            X_val[i][v] += add1_val[v];
            if constexpr (!one_moment) {
              sum1 += static_cast<accscalar_t>(X_val[i][v]);
            }
            sum2 += Numerics<accscalar_t>::pow(X_val[i][v], 2);
          }
          if constexpr (add_back) {
            *(reinterpret_cast<vec_t*>(
                add1_data + group_offset + plane_offset)) = X_val[i];
          }
        } else {
#pragma unroll(vec_size)
          for (int v = 0; v < vec_size; ++v) {
            if constexpr (!one_moment) {
              sum1 += static_cast<accscalar_t>(X_val[i][v]);
            }
            sum2 += Numerics<accscalar_t>::pow(X_val[i][v], 2);
          }
        }
      }
    }

    if constexpr (one_moment) {
      sum2 = sycl::reduce_over_group(
          item_id.get_group(), sum2, sycl::plus<accscalar_t>());
      if (local_id == 0) {
        accscalar_t scale = static_cast<accscalar_t>(Plane);
        local_sum2[0] = Numerics<accscalar_t>::rsqrt(
            sum2 < 0 ? 0 : sum2 / scale + static_cast<accscalar_t>(eps));
      }
    } else {
      norm_group_reduce<accscalar_t>(
          item_id,
          sub_group_num,
          sum1,
          sum2,
          local_sum1,
          local_sum2,
          [](accscalar_t a, accscalar_t b) { return a + b; });
      if (local_id == 0) {
        accscalar_t scale = static_cast<accscalar_t>(Plane);
        sum2 = (sum2 - sum1 * sum1 / scale) / scale;
        sum1 = sum1 / scale;
        local_sum1[0] = sum1;
        local_sum2[0] = Numerics<accscalar_t>::rsqrt(
            sum2 < 0 ? 0 : sum2 + static_cast<accscalar_t>(eps));
      }
    }

    item_id.barrier();

    auto mean_val = one_moment ? 0 : local_sum1[0];
    auto var_val = local_sum2[0];
    for (int i = 0; i < Num; ++i) {
      index_t plane_offset = (i * workgroup_size + local_id) * vec_size;
      if (plane_offset < Plane) {
        weight_vec_t gamma_val, beta_val;
        if (gamma_data != nullptr) {
          gamma_val =
              *(reinterpret_cast<weight_vec_t*>(gamma_data + plane_offset));
        }
        if (beta_data != nullptr) {
          beta_val =
              *(reinterpret_cast<weight_vec_t*>(beta_data + plane_offset));
        }

#pragma unroll(vec_size)
        for (int v = 0; v < vec_size; ++v) {
          if (gamma_data != nullptr && beta_data != nullptr) {
            X_val[i][v] = static_cast<accscalar_t>(gamma_val[v]) *
                    (var_val *
                     static_cast<accscalar_t>(X_val[i][v] - mean_val)) +
                static_cast<accscalar_t>(beta_val[v]);
          } else if (gamma_data != nullptr) {
            X_val[i][v] = static_cast<accscalar_t>(gamma_val[v]) *
                (var_val * static_cast<accscalar_t>(X_val[i][v] - mean_val));
          } else if (beta_data != nullptr) {
            X_val[i][v] =
                (var_val * static_cast<accscalar_t>(X_val[i][v] - mean_val)) +
                static_cast<accscalar_t>(beta_val[v]);
          } else {
            X_val[i][v] =
                (var_val * static_cast<accscalar_t>(X_val[i][v] - mean_val));
          }
        }
        *(reinterpret_cast<vec_t*>(Y_data + group_offset + plane_offset)) =
            X_val[i];
      }
    }
  }
  FusedNormKernel1Functor(
      scalar_t* add1_data_,
      scalar_t* add2_data_,
      scalar_t* X_data_,
      weight_t* gamma_data_,
      weight_t* beta_data_,
      scalar_t* Y_data_,
      float eps_,
      int Plane_,
      int workgroup_size_,
      int sub_group_num_,
      dpcpp_local_acc_t<accscalar_t> local_sum1_,
      dpcpp_local_acc_t<accscalar_t> local_sum2_)
      : add1_data(add1_data_),
        add2_data(add2_data_),
        X_data(X_data_),
        gamma_data(gamma_data_),
        beta_data(beta_data_),
        Y_data(Y_data_),
        eps(eps_),
        Plane(Plane_),
        workgroup_size(workgroup_size_),
        sub_group_num(sub_group_num_),
        local_sum1(local_sum1_),
        local_sum2(local_sum2_) {}

 private:
  scalar_t* add1_data;
  scalar_t* add2_data;
  scalar_t* X_data;
  weight_t* gamma_data;
  weight_t* beta_data;
  scalar_t* Y_data;
  float eps;
  int Plane;
  int workgroup_size;
  int sub_group_num;
  dpcpp_local_acc_t<accscalar_t> local_sum1;
  dpcpp_local_acc_t<accscalar_t> local_sum2;
};

template <
    typename scalar_t,
    typename mean_t,
    typename weight_t,
    typename index_t,
    int vec_size,
    bool one_moment = false,
    bool add_back = false>
void fused_norm_kernel1(
    scalar_t* add1_data,
    scalar_t* add2_data,
    scalar_t* X_data,
    weight_t* gamma_data,
    weight_t* beta_data,
    scalar_t* Y_data,
    float eps,
    int BS,
    int Plane) {
  constexpr int max_vec_size = float4_size / sizeof(scalar_t);
  constexpr int reg_num_per_item = max_vec_size << 1;
  constexpr int Num = std::max(reg_num_per_item / vec_size, 1);

  auto& dpcpp_queue = dpcppGetCurrentQueue();
  auto dev_id = dpcppGetDeviceIdOfCurrentQueue();
  int workgroup_size = dpcppMaxWorkGroupSize(dev_id);
  while ((workgroup_size >> 1) * vec_size >= Plane) {
    workgroup_size = workgroup_size >> 1;
  }
  int workgroup_num = BS;
  int sub_group_num = workgroup_size / SIMD;

  using accscalar_t = acc_type<scalar_t>;
  using vec_t = at::native::Memory::aligned_vector_loop<scalar_t, vec_size>;
  using weight_vec_t =
      at::native::Memory::aligned_vector_loop<weight_t, vec_size>;
  sycl::range<1> local_range{workgroup_size};
  sycl::range<1> global_range{workgroup_num * workgroup_size};

  auto cgf = DPCPP_Q_CGF(cgh) {
    dpcpp_local_acc_t<accscalar_t> local_sum1(sub_group_num, cgh);
    dpcpp_local_acc_t<accscalar_t> local_sum2(sub_group_num, cgh);
    FusedNormKernel1Functor<
        scalar_t,
        mean_t,
        weight_t,
        index_t,
        accscalar_t,
        vec_t,
        weight_vec_t,
        Num,
        vec_size,
        one_moment,
        add_back>
        kfn(add1_data,
            add2_data,
            X_data,
            gamma_data,
            beta_data,
            Y_data,
            eps,
            Plane,
            workgroup_size,
            sub_group_num,
            local_sum1,
            local_sum2);
    cgh.parallel_for<decltype(kfn)>(
        sycl::nd_range<1>(
            sycl::range<1>(global_range), sycl::range<1>(local_range)),
        kfn);
  };
  DPCPP_Q_SUBMIT(dpcpp_queue, cgf);
}

template <
    typename scalar_t,
    typename mean_t,
    typename weight_t,
    bool one_moment = false,
    bool add_back = false>
void launch_vectorized_fused_norm_kernel1(
    scalar_t* add1_data,
    scalar_t* add2_data,
    scalar_t* X_data,
    weight_t* gamma_data,
    weight_t* beta_data,
    scalar_t* Y_data,
    float eps,
    int64_t M,
    int64_t N,
    int vec_size,
    bool can_use_32bit_index) {
#define vectorized_fused_norm_kernel1(vec_size)                                \
  {                                                                            \
    using vec_t = at::native::Memory::aligned_vector_loop<scalar_t, vec_size>; \
    constexpr int align_bytes = alignof(vec_t);                                \
    int input_start = ((uint64_t)X_data) % align_bytes / sizeof(scalar_t);     \
    int output_start = ((uint64_t)Y_data) % align_bytes / sizeof(scalar_t);    \
    if (input_start == 0 && output_start == 0 && N % vec_size == 0) {          \
      if (can_use_32bit_index) {                                               \
        impl::fused_norm_kernel1<                                              \
            scalar_t,                                                          \
            mean_t,                                                            \
            weight_t,                                                          \
            uint32_t,                                                          \
            vec_size,                                                          \
            one_moment,                                                        \
            add_back>(                                                         \
            add1_data,                                                         \
            add2_data,                                                         \
            X_data,                                                            \
            gamma_data,                                                        \
            beta_data,                                                         \
            Y_data,                                                            \
            eps,                                                               \
            M,                                                                 \
            N);                                                                \
      } else {                                                                 \
        impl::fused_norm_kernel1<                                              \
            scalar_t,                                                          \
            mean_t,                                                            \
            weight_t,                                                          \
            uint64_t,                                                          \
            vec_size,                                                          \
            one_moment,                                                        \
            add_back>(                                                         \
            add1_data,                                                         \
            add2_data,                                                         \
            X_data,                                                            \
            gamma_data,                                                        \
            beta_data,                                                         \
            Y_data,                                                            \
            eps,                                                               \
            M,                                                                 \
            N);                                                                \
      }                                                                        \
    }                                                                          \
  }
  switch (vec_size) {
    case 8: {
      vectorized_fused_norm_kernel1(8);
      break;
    }
    case 4: {
      vectorized_fused_norm_kernel1(4);
      break;
    }
    case 2: {
      vectorized_fused_norm_kernel1(2);
      break;
    }
    default: {
      vectorized_fused_norm_kernel1(1);
      break;
    }
  }
}

template <
    typename scalar_t,
    typename mean_t,
    typename weight_t,
    bool one_moment,
    bool add_back>
bool LayerNormKernelImplInternal(
    const Tensor& add1,
    const Tensor& add2,
    const Tensor& X,
    const Tensor& gamma,
    const Tensor& beta,
    int64_t M,
    int64_t N,
    acc_type<scalar_t> eps,
    Tensor& Y) {
  constexpr int max_vec_size = float4_size / sizeof(scalar_t);
  constexpr int reg_num_per_item = max_vec_size * 2;
  auto& dpcpp_queue = dpcppGetCurrentQueue();
  auto dev_id = dpcppGetDeviceIdOfCurrentQueue();
  int workgroup_size = dpcppMaxWorkGroupSize(dev_id);
  bool fast_path_success = (N <= workgroup_size * reg_num_per_item);
  if (!fast_path_success)
    return fast_path_success;

  // decide vec_size
  int vec_size = max_vec_size;
  while ((vec_size >> 1) * workgroup_size >= N) {
    vec_size = vec_size >> 1;
  }
  // fallback if N is not divisible for vec_size
  if (N % vec_size != 0) {
    return false;
  }
  scalar_t* X_data = X.data_ptr<scalar_t>();
  scalar_t* Y_data = Y.data_ptr<scalar_t>();
  weight_t* gamma_data = gamma.defined() ? gamma.data_ptr<weight_t>() : nullptr;
  weight_t* beta_data = beta.defined() ? beta.data_ptr<weight_t>() : nullptr;
  scalar_t* add1_data = add1.defined() ? add1.data_ptr<scalar_t>() : nullptr;
  scalar_t* add2_data = add2.defined() ? add2.data_ptr<scalar_t>() : nullptr;

  bool can_use_32bit_index = canUse32BitIndexMath(X);
  launch_vectorized_fused_norm_kernel1<
      scalar_t,
      mean_t,
      weight_t,
      one_moment,
      add_back>(
      add1_data,
      add2_data,
      X_data,
      gamma_data,
      beta_data,
      Y_data,
      eps,
      M,
      N,
      vec_size,
      can_use_32bit_index);
  return true;
}

template <bool one_moment = false, bool add_back = false>
bool LayerNormKernelImpl(
    const Tensor& add1,
    const Tensor& add2,
    const Tensor& X,
    const Tensor& gamma,
    const Tensor& beta,
    int64_t M,
    int64_t N,
    double eps,
    Tensor& Y) {
  bool success = false;
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      X.scalar_type(),
      "LayerNormKernelImpl",
      [&]() {
        success = LayerNormKernelImplInternal<
            scalar_t,
            scalar_t,
            scalar_t,
            one_moment,
            add_back>(
            add1,
            add2,
            X,
            gamma,
            beta,
            M,
            N,
            static_cast<acc_type<scalar_t>>(eps),
            Y);
      });
  return success;
}
} // namespace impl

Tensor add_add_layer_norm(
    const Tensor& add1,
    const Tensor& add2,
    const Tensor& input,
    at::IntArrayRef normalized_shape,
    const c10::optional<at::Tensor>& weight_opt,
    const c10::optional<at::Tensor>& bias_opt,
    double epsilon,
    bool add_back = false) {
  c10::MaybeOwned<Tensor> weight_maybe_owned =
      at::borrow_from_optional_tensor(weight_opt);
  const Tensor& weight = *weight_maybe_owned;
  c10::MaybeOwned<Tensor> bias_maybe_owned =
      at::borrow_from_optional_tensor(bias_opt);
  const Tensor& bias = *bias_maybe_owned;
  auto M_N =
      impl::fast_check_layer_norm_inputs(input, normalized_shape, weight, bias);
  auto M = M_N.first;
  auto N = M_N.second;

  int numel = input.numel();
  Tensor output = at::empty(input.sizes(), input.options());
  if (input.numel() != 0) {
    Tensor input_ = to_plain_if_needed(input).contiguous();
    Tensor weight_ =
        weight.defined() ? to_plain_if_needed(weight).contiguous() : weight;
    Tensor bias_ =
        bias.defined() ? to_plain_if_needed(bias).contiguous() : bias;
    bool can_be_fused = true;

    bool fast_path_success = false;
    if (can_be_fused) {
      if (add_back) {
        fast_path_success = impl::LayerNormKernelImpl<false, true>(
            add1, add2, input_, weight_, bias_, M, N, epsilon, output);
      } else {
        fast_path_success = impl::LayerNormKernelImpl<false, false>(
            add1, add2, input_, weight_, bias_, M, N, epsilon, output);
      }
    }
    if (!(can_be_fused && fast_path_success)) {
      if (add1.defined() && add2.defined()) {
        if (add_back) {
          add1.add_(input_).add_(add2);
          input_ = add1;
        } else {
          input_ = input_ + add1 + add2;
        }
      } else if (add1.defined()) {
        if (add_back) {
          add1.add_(input_);
          input_ = add1;
        } else
          input_ = add1 + input_;
      }
      output = std::get<0>(at::AtenIpexTypeXPU::native_layer_norm(
          input_, normalized_shape, weight_opt, bias_opt, epsilon));
    }
  }
  return output.reshape(input.sizes());
}

Tensor add_layer_norm(
    const Tensor& add1,
    const Tensor& input,
    at::IntArrayRef normalized_shape,
    const c10::optional<at::Tensor>& weight_opt,
    const c10::optional<at::Tensor>& bias_opt,
    double epsilon,
    bool add_back = false) {
  RECORD_FUNCTION("add_layer_norm", std::vector<c10::IValue>({input}));
  return add_add_layer_norm(
      add1,
      Tensor(),
      input,
      normalized_shape,
      weight_opt,
      bias_opt,
      epsilon,
      add_back);
}

Tensor fast_layer_norm(
    const Tensor& input,
    at::IntArrayRef normalized_shape,
    const c10::optional<at::Tensor>& weight_opt,
    const c10::optional<at::Tensor>& bias_opt,
    double epsilon) {
  RECORD_FUNCTION("fast_layer_norm", std::vector<c10::IValue>({input}));
  return add_add_layer_norm(
      Tensor(),
      Tensor(),
      input,
      normalized_shape,
      weight_opt,
      bias_opt,
      epsilon);
}

Tensor fast_layer_norm_autocast(
    const Tensor& input,
    at::IntArrayRef normalized_shape,
    const c10::optional<at::Tensor>& weight_opt,
    const c10::optional<at::Tensor>& bias_opt,
    double epsilon) {
  c10::impl::ExcludeDispatchKeyGuard no_autocast(c10::DispatchKey::AutocastXPU);
  auto to_type = input.scalar_type();
  if (input.scalar_type() == at::ScalarType::Half ||
      weight_opt->scalar_type() == at::ScalarType::Half ||
      bias_opt->scalar_type() == at::ScalarType::Half) {
    to_type = at::ScalarType::Half;
  } else if (
      input.scalar_type() == at::ScalarType::BFloat16 ||
      weight_opt->scalar_type() == at::ScalarType::BFloat16 ||
      bias_opt->scalar_type() == at::ScalarType::BFloat16) {
    to_type = at::ScalarType::BFloat16;
  }
  return fast_layer_norm(
      cached_cast(to_type, input, c10::DeviceType::XPU),
      normalized_shape,
      cached_cast(to_type, *weight_opt, c10::DeviceType::XPU),
      cached_cast(to_type, *bias_opt, c10::DeviceType::XPU),
      epsilon);
}

Tensor add_add_rms_norm(
    const Tensor& add1,
    const Tensor& add2,
    const Tensor& input,
    at::IntArrayRef normalized_shape,
    const c10::optional<at::Tensor>& weight_opt,
    const c10::optional<at::Tensor>& bias_opt,
    double epsilon,
    bool add_back = false) {
  c10::MaybeOwned<Tensor> weight_maybe_owned =
      at::borrow_from_optional_tensor(weight_opt);
  const Tensor& weight = *weight_maybe_owned;
  c10::MaybeOwned<Tensor> bias_maybe_owned =
      at::borrow_from_optional_tensor(bias_opt);
  const Tensor& bias = *bias_maybe_owned;

  auto M_N =
      impl::fast_check_layer_norm_inputs(input, normalized_shape, weight, bias);
  auto M = M_N.first;
  auto N = M_N.second;
  int numel = input.numel();
  Tensor output = at::empty(input.sizes(), input.options());
  if (input.numel()) {
    Tensor input_ = to_plain_if_needed(input).contiguous();
    Tensor weight_ =
        weight.defined() ? to_plain_if_needed(weight).contiguous() : weight;
    Tensor bias_ =
        bias.defined() ? to_plain_if_needed(bias).contiguous() : bias;
    bool can_be_fused = true;
    bool fast_path_success = false;
    if (can_be_fused) {
      if (add_back)
        fast_path_success = impl::LayerNormKernelImpl<true, true>(
            add1, add2, input_, weight_, bias_, M, N, epsilon, output);
      else
        fast_path_success = impl::LayerNormKernelImpl<true, false>(
            add1, add2, input_, weight_, bias_, M, N, epsilon, output);
    }
    if (!(can_be_fused && fast_path_success)) {
      if (add1.defined() && add2.defined()) {
        if (add_back) {
          add1.add_(input_).add_(add2);
          input_ = add1;
        } else {
          input_ = input_ + add1 + add2;
        }
      } else if (add1.defined()) {
        if (add_back) {
          add1.add_(input_);
          input_ = add1;
        } else
          input_ = add1 + input_;
      }
      output =
          std::get<0>(rms_norm_fw(input_, normalized_shape, weight_, epsilon));
    }
  }
  return output.reshape(input.sizes());
}

Tensor add_rms_norm(
    const Tensor& add1,
    const Tensor& input,
    at::IntArrayRef normalized_shape,
    const c10::optional<at::Tensor>& weight_opt,
    const c10::optional<at::Tensor>& bias_opt,
    double epsilon,
    bool add_back = false) {
  RECORD_FUNCTION("add_rms_norm", std::vector<c10::IValue>({input}));
  return add_add_rms_norm(
      add1,
      Tensor(),
      input,
      normalized_shape,
      weight_opt,
      bias_opt,
      epsilon,
      add_back);
}

Tensor fast_rms_norm(
    const Tensor& input,
    at::IntArrayRef normalized_shape,
    const c10::optional<at::Tensor>& weight_opt,
    const c10::optional<at::Tensor>& bias_opt,
    double epsilon) {
  RECORD_FUNCTION("fast_rms_norm", std::vector<c10::IValue>({input}));
  return add_add_rms_norm(
      Tensor(),
      Tensor(),
      input,
      normalized_shape,
      weight_opt,
      bias_opt,
      epsilon);
}

namespace {
IPEX_LIBRARY_FRAGMENT() {
  IPEX_OP_REGISTER_DISPATCH(
      "add_add_layer_norm", add_add_layer_norm, c10::DispatchKey::XPU);
}
IPEX_LIBRARY_FRAGMENT() {
  IPEX_OP_REGISTER_DISPATCH(
      "add_layer_norm", add_layer_norm, c10::DispatchKey::XPU);
}
IPEX_LIBRARY_FRAGMENT() {
  IPEX_OP_REGISTER_DISPATCH(
      "fast_layer_norm", fast_layer_norm, c10::DispatchKey::XPU);
}
IPEX_LIBRARY_FRAGMENT() {
  IPEX_OP_REGISTER_DISPATCH(
      "fast_layer_norm",
      fast_layer_norm_autocast,
      c10::DispatchKey::AutocastXPU);
}
IPEX_LIBRARY_FRAGMENT() {
  IPEX_OP_REGISTER_DISPATCH(
      "add_add_rms_norm", add_add_rms_norm, c10::DispatchKey::XPU);
}
IPEX_LIBRARY_FRAGMENT() {
  IPEX_OP_REGISTER_DISPATCH(
      "add_rms_norm", add_rms_norm, c10::DispatchKey::XPU);
}
IPEX_LIBRARY_FRAGMENT() {
  IPEX_OP_REGISTER_DISPATCH(
      "fast_rms_norm", fast_rms_norm, c10::DispatchKey::XPU);
}
} // namespace

} // namespace AtenIpexTypeXPU
} // namespace at
