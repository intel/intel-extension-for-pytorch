#include <ATen/ATen.h>
#include <ATen/record_function.h>
#include <oneDNN/oneDNN.h>
#include <runtime/Utils.h>
#include <sycl/sycl.hpp>
#include <utils/oneMKLUtils.h>
#include "comm/ATDispatch.h"
#include "utils/ComputeEngine.h"

namespace at {
namespace AtenIpexTypeXPU {

namespace RmsNormVllmImpl {
template <typename scalar_t>
class rms_norm_kernel {
 public:
  rms_norm_kernel(
      scalar_t* out_,
      const scalar_t* input_,
      const int64_t input_stride_,
      const scalar_t* weight_,
      const float epsilon_,
      const int num_tokens_,
      const int hidden_size_,
      sycl::local_accessor<float, 1> s_variance_)
      : out(out_),
        input(input_),
        input_stride(input_stride_),
        weight(weight_),
        epsilon(epsilon_),
        num_tokens(num_tokens_),
        hidden_size(hidden_size_),
        s_variance(s_variance_) {}

  void operator() [[sycl::reqd_sub_group_size(32)]] (
      const sycl::nd_item<3>& item_ct1) const {
    float* s_variance_ptr =
        s_variance.template get_multi_ptr<sycl::access::decorated::no>().get();
    float variance = 0.0f;

    for (int idx = item_ct1.get_local_id(2); idx < hidden_size;
         idx += item_ct1.get_local_range(2)) {
      const float x = (float)input[item_ct1.get_group(2) * input_stride + idx];
      variance += x * x;
    }

    variance = sycl::reduce_over_group(
        sycl::ext::oneapi::this_work_item::get_work_group<3>(),
        variance,
        sycl::plus<>());
    if (item_ct1.get_local_id(2) == 0) {
      *s_variance_ptr = sycl::rsqrt(variance / hidden_size + epsilon);
    }

    item_ct1.barrier(sycl::access::fence_space::local_space);

    for (int idx = item_ct1.get_local_id(2); idx < hidden_size;
         idx += item_ct1.get_local_range(2)) {
      float x = (float)input[item_ct1.get_group(2) * hidden_size + idx];
      out[item_ct1.get_group(2) * input_stride + idx] =
          ((scalar_t)(x * (*s_variance_ptr))) * weight[idx];
    }
  }

 private:
  scalar_t* __restrict__ out; // [..., hidden_size]
  const scalar_t* __restrict__ input; // [..., hidden_size]
  const int64_t input_stride;
  const scalar_t* __restrict__ weight; // [hidden_size]
  const float epsilon;
  const int num_tokens;
  const int hidden_size;
  sycl::local_accessor<float, 1> s_variance;
};
} // namespace RmsNormVllmImpl

void rms_norm_vllm(Tensor& out, Tensor& input, Tensor& weight, double epsilon) {
  int hidden_size = input.size(-1);
  int num_tokens = input.numel() / hidden_size;
  int64_t input_stride = input.stride(-2);
  auto out_ptr = out.data_ptr();
  auto input_ptr = input.data_ptr();
  auto weight_ptr = weight.data_ptr();
  sycl::range<3> grid(1, 1, num_tokens);
  sycl::range<3> block(1, 1, std::min(hidden_size, 1024));
  auto range = sycl::nd_range<3>(grid * block, block);
  auto& queue = dpcppGetCurrentQueue();

#define RMS_NORM_KERNEL_CALL(TYPE)                                       \
  {                                                                      \
    using scalar_t = TYPE;                                               \
    using Kernel = RmsNormVllmImpl::rms_norm_kernel<scalar_t>;           \
    auto cgf = DPCPP_Q_CGF(cgh) {                                        \
      sycl::local_accessor<float, 1> s_variance(sycl::range<1>(1), cgh); \
      Kernel task(                                                       \
          reinterpret_cast<scalar_t*>(out_ptr),                          \
          reinterpret_cast<scalar_t*>(input_ptr),                        \
          input_stride,                                                  \
          reinterpret_cast<scalar_t*>(weight_ptr),                       \
          epsilon,                                                       \
          num_tokens,                                                    \
          hidden_size,                                                   \
          s_variance);                                                   \
      cgh.parallel_for(range, task);                                     \
    };                                                                   \
    DPCPP_Q_SUBMIT(queue, cgf);                                          \
  }

  at::ScalarType stype = input.scalar_type();
  if (stype == at::kBFloat16) {
    RMS_NORM_KERNEL_CALL(sycl::ext::oneapi::bfloat16)
  } else if (stype == at::kHalf) {
    RMS_NORM_KERNEL_CALL(sycl::half)
  } else if (stype == at::kFloat) {
    RMS_NORM_KERNEL_CALL(float)
  } else {
    TORCH_CHECK(false, "Unknow dtype of rms_norm_vllm");
  }

#undef RMS_NORM_KERNEL_CALL
}

namespace FusedAddRmsNormVllmImpl {
template <typename scalar_t>
class fused_add_rms_norm_kernel {
 public:
  fused_add_rms_norm_kernel(
      scalar_t* __restrict__ input_, // [..., hidden_size]
      scalar_t* __restrict__ residual_, // [..., hidden_size]
      const int64_t input_stride_,
      const scalar_t* __restrict__ weight_, // [hidden_size]
      const float epsilon_,
      const int num_tokens_,
      const int hidden_size_,
      sycl::local_accessor<float, 1> s_variance_)
      : input(input_),
        residual(residual_),
        input_stride(input_stride_),
        weight(weight_),
        epsilon(epsilon_),
        num_tokens(num_tokens_),
        hidden_size(hidden_size_),
        s_variance(s_variance_) {}

  void operator() [[sycl::reqd_sub_group_size(32)]] (
      const sycl::nd_item<3>& item_ct1) const {
    float* s_variance_ptr =
        s_variance.template get_multi_ptr<sycl::access::decorated::no>().get();
    float variance = 0.0f;

    for (int idx = item_ct1.get_local_id(2); idx < hidden_size;
         idx += item_ct1.get_local_range(2)) {
      scalar_t z = (scalar_t)input[item_ct1.get_group(2) * input_stride + idx];
      z += residual[item_ct1.get_group(2) * hidden_size + idx];
      float x = (float)z;
      variance += x * x;
      residual[item_ct1.get_group(2) * hidden_size + idx] = z;
    }

    variance = sycl::reduce_over_group(
        sycl::ext::oneapi::this_work_item::get_work_group<3>(),
        variance,
        sycl::plus<>());
    if (item_ct1.get_local_id(2) == 0) {
      *s_variance_ptr = sycl::rsqrt(variance / hidden_size + epsilon);
    }

    item_ct1.barrier(sycl::access::fence_space::local_space);

    for (int idx = item_ct1.get_local_id(2); idx < hidden_size;
         idx += item_ct1.get_local_range(2)) {
      float x = (float)residual[item_ct1.get_group(2) * hidden_size + idx];
      input[item_ct1.get_group(2) * input_stride + idx] =
          ((scalar_t)(x * (*s_variance_ptr))) * weight[idx];
    }
  }

 private:
  scalar_t* __restrict__ input; // [..., hidden_size]
  scalar_t* __restrict__ residual; // [..., hidden_size]
  const int64_t input_stride;
  const scalar_t* __restrict__ weight; // [hidden_size]
  const float epsilon;
  const int num_tokens;
  const int hidden_size;
  sycl::local_accessor<float, 1> s_variance; // local memory for variance
};
} // namespace FusedAddRmsNormVllmImpl

void fused_add_rms_norm_vllm(
    Tensor& input,
    Tensor& residual,
    Tensor& weight,
    double epsilon) {
  int hidden_size = input.size(-1);
  int num_tokens = input.numel() / hidden_size;
  auto input_ptr = input.data_ptr();
  auto residual_ptr = residual.data_ptr();
  auto weight_ptr = weight.data_ptr();
  int64_t input_stride = input.stride(-2);
  sycl::range<3> grid(1, 1, num_tokens);
  sycl::range<3> block(1, 1, std::min(hidden_size, 1024));
  auto range = sycl::nd_range<3>(grid * block, block);
  auto& queue = dpcppGetCurrentQueue();

#define FUSED_ADD_RMS_NORM_KERNEL_CALL(TYPE)                             \
  {                                                                      \
    using scalar_t = TYPE;                                               \
    using Kernel =                                                       \
        FusedAddRmsNormVllmImpl::fused_add_rms_norm_kernel<scalar_t>;    \
    auto cgf = DPCPP_Q_CGF(cgh) {                                        \
      sycl::local_accessor<float, 1> s_variance(sycl::range<1>(1), cgh); \
      Kernel task(                                                       \
          reinterpret_cast<scalar_t*>(input_ptr),                        \
          reinterpret_cast<scalar_t*>(residual_ptr),                     \
          input_stride,                                                  \
          reinterpret_cast<scalar_t*>(weight_ptr),                       \
          epsilon,                                                       \
          num_tokens,                                                    \
          hidden_size,                                                   \
          s_variance);                                                   \
      cgh.parallel_for(range, task);                                     \
    };                                                                   \
    DPCPP_Q_SUBMIT(queue, cgf);                                          \
  }

  at::ScalarType stype = input.scalar_type();
  if (stype == at::kBFloat16) {
    FUSED_ADD_RMS_NORM_KERNEL_CALL(sycl::ext::oneapi::bfloat16)
  } else if (stype == at::kHalf) {
    FUSED_ADD_RMS_NORM_KERNEL_CALL(sycl::half)
  } else if (stype == at::kFloat) {
    FUSED_ADD_RMS_NORM_KERNEL_CALL(float)
  } else {
    TORCH_CHECK(false, "Unknow dtype of fused_add_rms_norm_vllm");
  }

#undef FUSED_ADD_RMS_NORM_KERNEL_CALL
}

} // namespace AtenIpexTypeXPU
} // namespace at

namespace {
IPEX_LIBRARY_FRAGMENT() {
  IPEX_OP_REGISTER("rms_norm_vllm", at::AtenIpexTypeXPU::rms_norm_vllm);
  IPEX_OP_REGISTER(
      "fused_add_rms_norm_vllm", at::AtenIpexTypeXPU::fused_add_rms_norm_vllm);
}
} // namespace
