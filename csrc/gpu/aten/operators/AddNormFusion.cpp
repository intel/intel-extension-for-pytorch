#include <ATen/ATen.h>
#include <ATen/Config.h>
#include <ATen/NativeFunctions.h>

#include <oneDNN/oneDNN.h>
#include "Norm.h"
#include "comm/RegistrationDeclarations.h"
#include "utils/CustomOperatorRegistration.h"

using namespace xpu::dpcpp;
using namespace at::AtenIpexTypeXPU::normalization;

namespace at {
namespace AtenIpexTypeXPU {

constexpr int float4_size = sizeof(float) * 4;

template <
    typename scalar_t,
    typename mean_t,
    typename weight_t,
    typename index_t,
    int vec_size,
    bool one_moment = false>
void fused_norm_kernel1(
    scalar_t* add1_data,
    scalar_t* add2_data,
    scalar_t* X_data,
    mean_t* mean_data,
    mean_t* var_data,
    weight_t* gamma_data,
    weight_t* beta_data,
    scalar_t* Y_data,
    float eps,
    int BS,
    int Plane) {
  constexpr int max_vec_size = float4_size / sizeof(scalar_t);
  constexpr int reg_num_per_item = max_vec_size * 2;
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
    cgh.parallel_for(
        sycl::nd_range<1>(
            sycl::range<1>(global_range), sycl::range<1>(local_range)),
        [=](sycl::nd_item<1> item_id) [[intel::reqd_sub_group_size(SIMD)]] {
          auto group_id = item_id.get_group(0);
          auto local_id = item_id.get_local_id(0);
          index_t group_offset = group_id * Plane;

          vec_t X_val[Num];
          accscalar_t sum1 = 0;
          accscalar_t sum2 = 0;

          for (int i = 0; i < Num; ++i) {
            index_t plane_offset = (i * workgroup_size + local_id) * vec_size;
            if (plane_offset < Plane) {
              X_val[i] = *(reinterpret_cast<vec_t*>(
                  X_data + group_offset + plane_offset));

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
              } else if (add1_data != nullptr) {
                vec_t add1_val = *(reinterpret_cast<vec_t*>(
                    add1_data + group_offset + plane_offset));
                for (int v = 0; v < vec_size; ++v) {
                  X_val[i][v] += add1_val[v];
                  if constexpr (!one_moment) {
                    sum1 += static_cast<accscalar_t>(X_val[i][v]);
                  }
                  sum2 += Numerics<accscalar_t>::pow(X_val[i][v], 2);
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
                gamma_val = *(
                    reinterpret_cast<weight_vec_t*>(gamma_data + plane_offset));
              }
              if (beta_data != nullptr) {
                beta_val = *(
                    reinterpret_cast<weight_vec_t*>(beta_data + plane_offset));
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
                      (var_val *
                       static_cast<accscalar_t>(X_val[i][v] - mean_val));
                } else if (beta_data != nullptr) {
                  X_val[i][v] =
                      (var_val *
                       static_cast<accscalar_t>(X_val[i][v] - mean_val)) +
                      static_cast<accscalar_t>(beta_val[v]);
                } else {
                  X_val[i][v] =
                      (var_val *
                       static_cast<accscalar_t>(X_val[i][v] - mean_val));
                }
              }
              *(reinterpret_cast<vec_t*>(
                  Y_data + group_offset + plane_offset)) = X_val[i];
            }
          }
        });
  };
  DPCPP_Q_SUBMIT(dpcpp_queue, cgf);
}

template <
    typename scalar_t,
    typename mean_t,
    typename weight_t,
    bool one_moment = false>
void launch_vectorized_fused_norm_kernel1(
    scalar_t* add1_data,
    scalar_t* add2_data,
    scalar_t* X_data,
    mean_t* mean_data,
    mean_t* var_data,
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
        fused_norm_kernel1<                                                    \
            scalar_t,                                                          \
            mean_t,                                                            \
            weight_t,                                                          \
            uint32_t,                                                          \
            vec_size,                                                          \
            one_moment>(                                                       \
            add1_data,                                                         \
            add2_data,                                                         \
            X_data,                                                            \
            mean_data,                                                         \
            var_data,                                                          \
            gamma_data,                                                        \
            beta_data,                                                         \
            Y_data,                                                            \
            eps,                                                               \
            M,                                                                 \
            N);                                                                \
      } else {                                                                 \
        fused_norm_kernel1<                                                    \
            scalar_t,                                                          \
            mean_t,                                                            \
            weight_t,                                                          \
            uint32_t,                                                          \
            vec_size,                                                          \
            one_moment>(                                                       \
            add1_data,                                                         \
            add2_data,                                                         \
            X_data,                                                            \
            mean_data,                                                         \
            var_data,                                                          \
            gamma_data,                                                        \
            beta_data,                                                         \
            Y_data,                                                            \
            eps,                                                               \
            M,                                                                 \
            N);                                                                \
      }                                                                        \
      break;                                                                   \
    }                                                                          \
  }

  switch (vec_size) {
    case 8: {
      vectorized_fused_norm_kernel1(8);
    }
    case 4: {
      vectorized_fused_norm_kernel1(4);
    }
    case 2: {
      vectorized_fused_norm_kernel1(2);
    }
    default: {
      vectorized_fused_norm_kernel1(1);
    }
  }
}

template <
    typename scalar_t,
    typename mean_t,
    typename weight_t,
    bool one_moment>
bool LayerNormKernelImplInternal(
    const Tensor& add1,
    const Tensor& add2,
    const Tensor& X,
    const Tensor& gamma,
    const Tensor& beta,
    int64_t M,
    int64_t N,
    acc_type<scalar_t> eps,
    Tensor& Y,
    Tensor& mean,
    Tensor& rstd) {
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

  scalar_t* add1_data = add1.defined() ? add1.data_ptr<scalar_t>() : nullptr;
  scalar_t* add2_data = add2.defined() ? add2.data_ptr<scalar_t>() : nullptr;
  scalar_t* X_data = X.data_ptr<scalar_t>();
  scalar_t* Y_data = Y.data_ptr<scalar_t>();
  mean_t* mean_data = mean.data_ptr<mean_t>();
  mean_t* var_data = rstd.data_ptr<mean_t>();
  weight_t* gamma_data = gamma.defined() ? gamma.data_ptr<weight_t>() : nullptr;
  weight_t* beta_data = beta.defined() ? beta.data_ptr<weight_t>() : nullptr;

  bool can_use_32bit_index = canUse32BitIndexMath(X);
  launch_vectorized_fused_norm_kernel1<scalar_t, mean_t, weight_t, one_moment>(
      add1_data,
      add2_data,
      X_data,
      mean_data,
      var_data,
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

template <bool one_moment = false>
bool LayerNormKernelImpl(
    const Tensor& add1,
    const Tensor& add2,
    const Tensor& X,
    const Tensor& gamma,
    const Tensor& beta,
    int64_t M,
    int64_t N,
    double eps,
    Tensor& Y,
    Tensor& mean,
    Tensor& rstd) {
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      X.scalar_type(),
      "LayerNormKernelImpl",
      [&]() {
        if (gamma.scalar_type() == kFloat) {
          mean = at::empty({M}, X.options().dtype(kFloat));
          rstd = at::empty({M}, X.options().dtype(kFloat));
          return LayerNormKernelImplInternal<
              scalar_t,
              float,
              float,
              one_moment>(
              add1,
              add2,
              X,
              gamma,
              beta,
              M,
              N,
              static_cast<acc_type<scalar_t>>(eps),
              Y,
              mean,
              rstd);
        } else {
          mean = at::empty({M}, X.options());
          rstd = at::empty({M}, X.options());
          return LayerNormKernelImplInternal<
              scalar_t,
              scalar_t,
              scalar_t,
              one_moment>(
              add1,
              add2,
              X,
              gamma,
              beta,
              M,
              N,
              static_cast<acc_type<scalar_t>>(eps),
              Y,
              mean,
              rstd);
        }
      });
  return true;
}

std::tuple<Tensor, Tensor, Tensor> add_add_layer_norm(
    const Tensor& add1,
    const Tensor& add2,
    const Tensor& input,
    at::IntArrayRef normalized_shape,
    const c10::optional<at::Tensor>& weight_opt,
    const c10::optional<at::Tensor>& bias_opt,
    double epsilon) {
  c10::MaybeOwned<Tensor> weight_maybe_owned =
      at::borrow_from_optional_tensor(weight_opt);
  const Tensor& weight = *weight_maybe_owned;
  c10::MaybeOwned<Tensor> bias_maybe_owned =
      at::borrow_from_optional_tensor(bias_opt);
  const Tensor& bias = *bias_maybe_owned;

  auto M_N = _check_layer_norm_inputs(input, normalized_shape, weight, bias);
  auto M = M_N.first;
  auto N = M_N.second;

  int numel = input.numel();
  Tensor output = at::empty(input.sizes(), input.options());
  Tensor mean, rstd;
  if (input.numel() != 0) {
    Tensor input_ = (input.dim() == 1) ? input.reshape({M, N}) : input;
    Tensor output_ = (output.dim() == 1) ? output.reshape({M, N}) : output;
    Tensor weight_ =
        (weight.defined() && weight.dim() == 1) ? weight.reshape({N}) : weight;
    Tensor bias_ =
        (bias.defined() && bias.dim() == 1) ? bias.reshape({N}) : bias;

    Tensor add1_ =
        add1.defined() ? to_plain_if_needed(add1).contiguous() : add1;
    Tensor add2_ =
        add2.defined() ? to_plain_if_needed(add2).contiguous() : add2;
    input_ = to_plain_if_needed(input_).contiguous();
    weight_ =
        weight_.defined() ? to_plain_if_needed(weight_).contiguous() : weight_;
    bias_ = bias_.defined() ? to_plain_if_needed(bias_).contiguous() : bias_;
    bool can_be_fused = true;
    if (add1.defined() &&
        (input_.sizes() != add1_.sizes() || input_.dtype() != add1_.dtype())) {
      can_be_fused = false;
    }
    if (add2.defined() &&
        (input_.sizes() != add2_.sizes() || input_.dtype() != add2_.dtype())) {
      can_be_fused = false;
    }

    bool fast_path_success = false;
    if (can_be_fused) {
      fast_path_success = LayerNormKernelImpl(
          add1_,
          add2_,
          input_,
          weight_,
          bias_,
          M,
          N,
          epsilon,
          output,
          mean,
          rstd);
    }
    if (!(can_be_fused && fast_path_success)) {
      input_ = add1_.defined() ? input_ + add1_ : input_;
      input_ = add2_.defined() ? input_ + add2_ : input_;
      std::tuple<Tensor, Tensor, Tensor>(output, mean, rstd) =
          at::AtenIpexTypeXPU::native_layer_norm(
              input_, normalized_shape, weight_opt, bias_opt, epsilon);
    }
  }
  return std::make_tuple(output.reshape(input.sizes()), mean, rstd);
}

std::tuple<Tensor, Tensor, Tensor> add_layer_norm(
    const Tensor& add1,
    const Tensor& add2,
    const Tensor& input,
    at::IntArrayRef normalized_shape,
    const c10::optional<at::Tensor>& weight_opt,
    const c10::optional<at::Tensor>& bias_opt,
    double epsilon) {
  RECORD_FUNCTION("add_layer_norm", std::vector<c10::IValue>({input}));
  return add_add_layer_norm(
      add1, Tensor(), input, normalized_shape, weight_opt, bias_opt, epsilon);
}

std::tuple<Tensor, Tensor, Tensor> fast_layer_norm(
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

std::tuple<Tensor, Tensor, Tensor> add_add_rms_norm(
    const Tensor& add1,
    const Tensor& add2,
    const Tensor& input,
    at::IntArrayRef normalized_shape,
    const c10::optional<at::Tensor>& weight_opt,
    const c10::optional<at::Tensor>& bias_opt,
    double epsilon) {
  c10::MaybeOwned<Tensor> weight_maybe_owned =
      at::borrow_from_optional_tensor(weight_opt);
  const Tensor& weight = *weight_maybe_owned;
  c10::MaybeOwned<Tensor> bias_maybe_owned =
      at::borrow_from_optional_tensor(bias_opt);
  const Tensor& bias = *bias_maybe_owned;

  auto M_N = _check_layer_norm_inputs(input, normalized_shape, weight, bias);
  auto M = M_N.first;
  auto N = M_N.second;

  int numel = input.numel();
  Tensor output = at::empty(input.sizes(), input.options());
  Tensor mean, rstd;
  if (input.numel()) {
    Tensor input_ = (input.dim() == 1) ? input.reshape({M, N}) : input;
    Tensor output_ = (output.dim() == 1) ? output.reshape({M, N}) : output;
    Tensor weight_ =
        (weight.defined() && weight.dim() == 1) ? weight.reshape({N}) : weight;
    Tensor bias_ =
        (bias.defined() && bias.dim() == 1) ? bias.reshape({N}) : bias;

    Tensor add1_ =
        add1.defined() ? to_plain_if_needed(add1).contiguous() : add1;
    Tensor add2_ =
        add2.defined() ? to_plain_if_needed(add2).contiguous() : add2;
    input_ = to_plain_if_needed(input_).contiguous();
    weight_ =
        weight_.defined() ? to_plain_if_needed(weight_).contiguous() : weight_;
    bias_ = bias_.defined() ? to_plain_if_needed(bias_).contiguous() : bias_;

    bool can_be_fused = true;
    if (add1.defined() &&
        (input_.sizes() != add1_.sizes() || input_.dtype() != add1_.dtype())) {
      can_be_fused = false;
    }
    if (add2.defined() &&
        (input_.sizes() != add2_.sizes() || input_.dtype() != add2_.dtype())) {
      can_be_fused = false;
    }

    bool fast_path_success = false;
    if (can_be_fused) {
      fast_path_success = LayerNormKernelImpl<true>(
          add1_,
          add2_,
          input_,
          weight_,
          bias_,
          M,
          N,
          epsilon,
          output,
          mean,
          rstd);
    }
    if (!(can_be_fused && fast_path_success)) {
      input_ = add1_.defined() ? input_ + add1_ : input_;
      input_ = add2_.defined() ? input_ + add2_ : input_;
      std::tuple<Tensor, Tensor, Tensor>(output, mean, rstd) =
          at::AtenIpexTypeXPU::native_layer_norm(
              input_, normalized_shape, weight_opt, bias_opt, epsilon);
    }
  }
  return std::make_tuple(output.reshape(input.sizes()), mean, rstd);
}

std::tuple<Tensor, Tensor, Tensor> add_rms_norm(
    const Tensor& add1,
    const Tensor& add2,
    const Tensor& input,
    at::IntArrayRef normalized_shape,
    const c10::optional<at::Tensor>& weight_opt,
    const c10::optional<at::Tensor>& bias_opt,
    double epsilon) {
  RECORD_FUNCTION("add_rms_norm", std::vector<c10::IValue>({input}));
  return add_add_layer_norm(
      add1, Tensor(), input, normalized_shape, weight_opt, bias_opt, epsilon);
}

std::tuple<Tensor, Tensor, Tensor> fast_rms_norm(
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
