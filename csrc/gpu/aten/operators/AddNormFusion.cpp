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

inline std::pair<int64_t, int64_t> _check_layer_norm_inputs(
    const Tensor& input,
    IntArrayRef normalized_shape,
    const Tensor& weight /* optional */,
    const Tensor& bias /* optional */) {
  const int normalized_ndim = normalized_shape.size();
  TORCH_CHECK(
      normalized_ndim >= 1,
      "Expected normalized_shape to be at least 1-dimensional, i.e., ",
      "containing at least one element, but got normalized_shape = ",
      normalized_shape);
  TORCH_CHECK(
      !weight.defined() || weight.sizes().equals(normalized_shape),
      "Expected weight to be of same shape as normalized_shape, but got ",
      "weight of shape ",
      weight.sizes(),
      " and normalized_shape = ",
      normalized_shape);
  TORCH_CHECK(
      !bias.defined() || bias.sizes().equals(normalized_shape),
      "Expected bias to be of same shape as normalized_shape, but got ",
      "bias of shape ",
      bias.sizes(),
      " and normalized_shape = ",
      normalized_shape);
  const auto input_shape = input.sizes();
  const auto input_ndim = input.dim();

  if (input_ndim < normalized_ndim ||
      !input_shape.slice(input_ndim - normalized_ndim)
           .equals(normalized_shape)) {
    std::stringstream ss;
    ss << "Given normalized_shape=" << normalized_shape
       << ", expected input with shape [*";
    for (auto size : normalized_shape) {
      ss << ", " << size;
    }
    ss << "], but got input of size" << input_shape;
    AT_ERROR(ss.str());
  }

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
  constexpr int float4_size = sizeof(float) * 4;
  constexpr int max_vec_size = float4_size / sizeof(scalar_t);
  constexpr int reg_num_per_item = max_vec_size * 4;
  constexpr int Num1 = reg_num_per_item / vec_size;
  constexpr int Num = 1;
  int workgroup_size = 1024;
  int workgroup_num = BS;
  int sub_group_num = workgroup_size / SIMD;

  using accscalar_t = acc_type<scalar_t>;
  using vec_t = at::native::Memory::aligned_vector_loop<scalar_t, vec_size>;
  using weight_vec_t =
      at::native::Memory::aligned_vector_loop<weight_t, vec_size>;
  sycl::range<3> local_range{1, 1, workgroup_size};
  sycl::range<3> global_range{workgroup_num, 1, workgroup_size};

  auto& dpcpp_queue = dpcppGetCurrentQueue();
  auto cgf = DPCPP_Q_CGF(cgh) {
    dpcpp_local_acc_t<accscalar_t> local_sum1(sub_group_num, cgh);
    dpcpp_local_acc_t<accscalar_t> local_sum2(sub_group_num, cgh);
    cgh.parallel_for(
        sycl::nd_range<3>(
            sycl::range<3>(global_range), sycl::range<3>(local_range)),
        [=](sycl::nd_item<3> item_id) [[intel::reqd_sub_group_size(SIMD)]] {
          auto group_id = item_id.get_group(0);
          auto local_id = item_id.get_local_id(2);
          index_t group_offset = group_id * Plane;

          vec_t X_val[Num];
          accscalar_t sum1 = 0;
          accscalar_t sum2 = 0;
          for (int i = 0; i < Num; ++i) {
            index_t plane_offset = (i * workgroup_size + local_id) * vec_size;
            if (plane_offset < Plane) {
              if (add1_data != nullptr && add2_data != nullptr) {
                X_val[i] = *(reinterpret_cast<vec_t*>(
                    X_data + group_offset + plane_offset));
                vec_t add1_val = *(reinterpret_cast<vec_t*>(
                    add1_data + group_offset + plane_offset));
                vec_t add2_val = *(reinterpret_cast<vec_t*>(
                    add2_data + group_offset + plane_offset));
#pragma unroll
                for (int v = 0; v < vec_size; ++v) {
                  X_val[i][v] += add1_val[v] + add2_val[v];
                  sum1 += static_cast<accscalar_t>(X_val[i][v]);
                  sum2 += static_cast<accscalar_t>(X_val[i][v]) *
                      static_cast<accscalar_t>(X_val[i][v]);
                }
              } else {
                X_val[i] = *(reinterpret_cast<vec_t*>(
                    X_data + group_offset + plane_offset));
#pragma unroll
                for (int v = 0; v < vec_size; ++v) {
                  sum1 += static_cast<accscalar_t>(X_val[i][v]);
                  sum2 += static_cast<accscalar_t>(X_val[i][v]) *
                      static_cast<accscalar_t>(X_val[i][v]);
                }
              }
            }
          }

          if constexpr (one_moment) {
            sum1 = sycl::reduce_over_group(
                item_id.get_group(), sum1, sycl::plus<accscalar_t>());
          } else {
            norm_group_reduce<accscalar_t>(
                item_id,
                sub_group_num,
                sum1,
                sum2,
                local_sum1,
                local_sum2,
                [](accscalar_t a, accscalar_t b) { return a + b; });
          }

          if (local_id == 0) {
            accscalar_t scale = static_cast<accscalar_t>(Plane);
            sum2 = (sum2 - sum1 * sum1 / scale) / scale;
            sum1 = sum1 / scale;
            local_sum1[group_id] = static_cast<mean_t>(sum1);
            local_sum2[group_id] =
                static_cast<mean_t>(Numerics<accscalar_t>::rsqrt(
                    sum2 < 0 ? 0 : sum2 + static_cast<accscalar_t>(eps)));
          }
          item_id.barrier();

          mean_t mean_val = local_sum1[group_id];
          mean_t var_val = local_sum2[group_id];
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

#pragma unroll
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
            }
            *(reinterpret_cast<vec_t*>(Y_data + group_offset + plane_offset)) =
                X_val[i];
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
    int64_t N) {
  // decide vec_size

  /*constexpr int float4_size = sizeof(float) * 4;
constexpr int max_vec_size = float4_size / sizeof(scalar_t);
int vec_size = max_vec_size;
using vec_t = at::native::Memory::aligned_vector_loop<scalar_t, max_vec_size>;
constexpr int align_bytes = alignof(vec_t);
int input_start = ((uint64_t)X_data) % align_bytes / sizeof(scalar_t);
int output_start = ((uint64_t)Y_data) % align_bytes / sizeof(scalar_t);

if (input_start > 0 || output_start > 0) {
  vec_size = 1;
}*/
  int vec_size = 4;
  bool can_use_32bit_index = true;
#define vectorized_fused_norm_kernel1(vec_size) \
  {                                             \
    if (can_use_32bit_index) {                  \
      fused_norm_kernel1<                       \
          scalar_t,                             \
          mean_t,                               \
          weight_t,                             \
          uint32_t,                             \
          vec_size,                             \
          one_moment>(                          \
          add1_data,                            \
          add2_data,                            \
          X_data,                               \
          mean_data,                            \
          var_data,                             \
          gamma_data,                           \
          beta_data,                            \
          Y_data,                               \
          eps,                                  \
          M,                                    \
          N);                                   \
    } else {                                    \
      fused_norm_kernel1<                       \
          scalar_t,                             \
          mean_t,                               \
          weight_t,                             \
          uint32_t,                             \
          vec_size,                             \
          one_moment>(                          \
          add1_data,                            \
          add2_data,                            \
          X_data,                               \
          mean_data,                            \
          var_data,                             \
          gamma_data,                           \
          beta_data,                            \
          Y_data,                               \
          eps,                                  \
          M,                                    \
          N);                                   \
    }                                           \
    break;                                      \
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

template <typename scalar_t, typename mean_t, typename weight_t>
void LayerNormKernelImplInternal(
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
  TORCH_CHECK(X.numel() == M * N);
  TORCH_CHECK(!gamma.defined() || gamma.numel() == N);
  TORCH_CHECK(!beta.defined() || beta.numel() == N);

  scalar_t* add1_data = add1.defined() ? add1.data_ptr<scalar_t>() : nullptr;
  scalar_t* add2_data = add2.defined() ? add2.data_ptr<scalar_t>() : nullptr;
  scalar_t* X_data = X.data_ptr<scalar_t>();
  scalar_t* Y_data = Y.data_ptr<scalar_t>();
  mean_t* mean_data = mean.data_ptr<mean_t>();
  mean_t* var_data = rstd.data_ptr<mean_t>();
  weight_t* gamma_data = gamma.defined() ? gamma.data_ptr<weight_t>() : nullptr;
  weight_t* beta_data = beta.defined() ? beta.data_ptr<weight_t>() : nullptr;

  auto config = NormConfig(M, N, 1, sizeof(scalar_t));
  bool can_use_32bit_index = canUse32BitIndexMath(X);

  launch_vectorized_fused_norm_kernel1<scalar_t, mean_t, weight_t>(
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
      N);
}

void LayerNormKernelImpl(
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
          LayerNormKernelImplInternal<scalar_t, float, float>(
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
          LayerNormKernelImplInternal<scalar_t, scalar_t, scalar_t>(
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
  if (input.numel() != 0 && add1.numel() == numel && add2.numel() == numel) {
    Tensor input_ = (input.dim() == 1) ? input.reshape({M, N}) : input;
    Tensor output_ = (output.dim() == 1) ? output.reshape({M, N}) : output;
    Tensor weight_ =
        (weight.defined() && weight.dim() == 1) ? weight.reshape({N}) : weight;
    Tensor bias_ =
        (bias.defined() && bias.dim() == 1) ? bias.reshape({N}) : bias;

    Tensor add1_ = to_plain_if_needed(add1).contiguous();
    Tensor add2_ = to_plain_if_needed(add2).contiguous();
    input_ = to_plain_if_needed(input_).contiguous();
    weight_ =
        weight_.defined() ? to_plain_if_needed(weight_).contiguous() : weight_;
    bias_ = bias_.defined() ? to_plain_if_needed(bias_).contiguous() : bias_;

    // if (N < 8192 && N % 4 == 0) {
    if (N == 4096) {
      LayerNormKernelImpl(
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
    } else {
      input_ += add1_ + add2_;
      std::tuple<Tensor, Tensor, Tensor>(output, mean, rstd) =
          at::AtenIpexTypeXPU::native_layer_norm(
              input_, normalized_shape, weight_opt, bias_opt, epsilon);
    }
  }
  return std::make_tuple(output.reshape(input.sizes()), mean, rstd);
}

std::tuple<Tensor, Tensor, Tensor> fast_layer_norm(
    const Tensor& input,
    at::IntArrayRef normalized_shape,
    const c10::optional<at::Tensor>& weight_opt,
    const c10::optional<at::Tensor>& bias_opt,
    double epsilon) {
  RECORD_FUNCTION("fast_layer_norm", std::vector<c10::IValue>({input}));
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

    Tensor add1_;
    Tensor add2_;
    input_ = to_plain_if_needed(input_).contiguous();
    weight_ =
        weight_.defined() ? to_plain_if_needed(weight_).contiguous() : weight_;
    bias_ = bias_.defined() ? to_plain_if_needed(bias_).contiguous() : bias_;

    // if (N < 8192 && N % 4 == 0) {
    if (N == 4096) {
      LayerNormKernelImpl(
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
    } else {
      input_ += add1_ + add2_;
      std::tuple<Tensor, Tensor, Tensor>(output, mean, rstd) =
          at::AtenIpexTypeXPU::native_layer_norm(
              input_, normalized_shape, weight_opt, bias_opt, epsilon);
    }
  }
  return std::make_tuple(output.reshape(input.sizes()), mean, rstd);
}

namespace {
IPEX_LIBRARY_FRAGMENT() {
  IPEX_OP_REGISTER_DISPATCH(
      "add_add_layer_norm", add_add_layer_norm, c10::DispatchKey::XPU);
}
IPEX_LIBRARY_FRAGMENT() {
  IPEX_OP_REGISTER_DISPATCH(
      "fast_layer_norm", fast_layer_norm, c10::DispatchKey::XPU);
}
} // namespace

} // namespace AtenIpexTypeXPU
} // namespace at
