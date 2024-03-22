#include "cast.h"
#include <runtime/Utils.h>
#include "../comm/ATDispatch.h"
#include "../comm/Numerics.h"
#include "utils/CustomOperatorRegistration.h"

using namespace torch_ipex::xpu::dpcpp;

namespace at {
namespace AtenIpexTypeXPU {

namespace detail {

struct Empty {};

struct Identity {
  float operator()(float value, int offset, const Empty&) {
    return value;
  }
};

struct DequantizeParam {
  const float* scale_inv;
};

struct Dequantize {
  float operator()(float value, int offset, const DequantizeParam& param) {
    return value * (*(param.scale_inv));
  }
};

} // namespace detail

namespace impl {

template <
    typename ComputeType,
    typename Param,
    typename OP,
    typename InputType,
    typename OutputType>
struct ElementWiseKernelFunctor {
  void operator()(sycl::nd_item<1> item) const {
    auto in_data = in_data_;
    auto out_data = out_data_;
    auto id = item.get_global_linear_id();
    ComputeType max = 0;
    if (id < num_elements_) {
      ComputeType s = 0;
      if (is_fp8<OutputType>::value) {
        if (scale_ != nullptr) {
          s = *scale_;
          if (id == 0 && scale_inv_ != nullptr) {
            *scale_inv_ = sycl::native::recip(s);
          }
        }
      }
      ComputeType val = static_cast<ComputeType>(in_data[id]);
      ComputeType temp = OP()(val, id, p_);
      if (is_fp8<OutputType>::value && amax_ != nullptr) {
        max = sycl::fmax(sycl::fabs(temp), max);
        temp = temp * s;
      }
      out_data[id] = (OutputType)(temp);
    }
    if (is_fp8<OutputType>::value && amax_ != nullptr) {
      auto group_max = sycl::reduce_over_group(
          item.get_group(), max, sycl::maximum<ComputeType>());
      if (item.get_local_linear_id() == 0) {
        auto atm = sycl::atomic_ref<
            float,
            sycl::memory_order::relaxed,
            sycl::memory_scope::device,
            sycl::access::address_space::global_space>(*amax_);
        atm.fetch_max(group_max);
      }
    }
  }
  ElementWiseKernelFunctor(
      const InputType* in_data,
      OutputType* out_data,
      const ComputeType* scale,
      ComputeType* scale_inv,
      ComputeType* amax,
      Param p,
      int num_elements)
      : in_data_(in_data),
        out_data_(out_data),
        scale_(scale),
        scale_inv_(scale_inv),
        amax_(amax),
        p_(p),
        num_elements_(num_elements) {}

 private:
  const InputType* in_data_;
  OutputType* out_data_;
  const ComputeType* scale_;
  ComputeType* scale_inv_;
  ComputeType* amax_;
  Param p_;
  int num_elements_;
};

template <
    typename ComputeType,
    typename Param,
    typename OP,
    typename InputType,
    typename OutputType>
void ElementWiseKernel(
    const InputType* in_,
    OutputType* out_,
    const ComputeType* scale_,
    ComputeType* scale_inv_,
    ComputeType* amax_,
    Param p_,
    int num_elements_) {
  // TODO: do vectorized load/store for better perf
  auto& dpcpp_queue = dpcppGetCurrentQueue();
  auto dev_id = dpcppGetDeviceIdOfCurrentQueue();

  auto maxGroupsize = dpcppMaxWorkGroupSize(dev_id);
  auto wgroup_size = std::min(num_elements_, (int)maxGroupsize);
  int wgroup_num = CeilDiv((int64_t)num_elements_, (int64_t)wgroup_size);
  auto cgf = DPCPP_Q_CGF(cgh) {
    auto in_data_ = in_;
    auto out_data_ = out_;
    ElementWiseKernelFunctor<ComputeType, Param, OP, InputType, OutputType> kfn(
        in_data_, out_data_, scale_, scale_inv_, amax_, p_, num_elements_);

    cgh.parallel_for<decltype(kfn)>(
        sycl::nd_range<1>(wgroup_num * wgroup_size, wgroup_size), kfn);
  };
  DPCPP_Q_SUBMIT(dpcpp_queue, cgf);
}

template <typename T, bool HighPrecisionOut = false>
void fp8_quantize(
    const Tensor& input,
    Tensor& output,
    int fp8_format,
    float* amax,
    const float* scale,
    float* scale_inv) {
  IPEX_TYPE_SWITCH_FP8ONLY(
      fp8_format, type, int num_elements = input.numel();
      auto& dpcpp_queue = dpcppGetCurrentQueue();
      auto dev_id = dpcppGetDeviceIdOfCurrentQueue();

      auto wgroup_size = dpcppMaxWorkGroupSize(dev_id);
      int wgroup_num = CeilDiv((int64_t)num_elements, (int64_t)wgroup_size);
      impl::ElementWiseKernel<float, detail::Empty, detail::Identity, T, type>(
          input.data_ptr<T>(),
          output.data_ptr<type>(),
          scale,
          scale_inv,
          amax,
          {},
          num_elements););
}

template <typename Tout>
void fp8_dequantize(
    const Tensor& input,
    Tensor& output,
    int fp8_format,
    const float* scale_inv) {
  int num_elements = input.numel();
  IPEX_TYPE_SWITCH_FP8ONLY(fp8_format, scalar_t, detail::DequantizeParam P;
                           P.scale_inv = scale_inv;
                           impl::ElementWiseKernel<
                               float,
                               detail::DequantizeParam,
                               detail::Dequantize,
                               scalar_t,
                               Tout>(
                               FP8_PTR(input.data_ptr<scalar_t>(), scalar_t),
                               output.data_ptr<Tout>(),
                               nullptr,
                               nullptr,
                               nullptr,
                               P,
                               num_elements););
}

} // namespace impl

at::ScalarType convert_to_dtype(int64_t format) {
  switch (format) {
    case Float8Format::kFloat8_E5M2:
      return ScalarType::Float8_e5m2;
    case Float8Format::kFloat8_E4M3:
      return ScalarType::Float8_e4m3fn;
    default:
      TORCH_CHECK(false, "undefined format.\n");
  }
}

void fp8_quantize_op(
    const Tensor& input,
    Tensor& output,
    int fp8_format,
    float* amax,
    const float* scale,
    float* scale_inv) {
  IPEX_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::BFloat16,
      at::ScalarType::Half,
      input.scalar_type(),
      "fp8_quantize_op",
      [&] {
        impl::fp8_quantize<scalar_t>(
            input, output, fp8_format, amax, scale, scale_inv);
      });
}

void fp8_dequantize_op(
    const Tensor& input,
    Tensor& output,
    int64_t itype,
    const float* scale_inv) {
  IPEX_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::BFloat16,
      at::ScalarType::Half,
      output.scalar_type(),
      "fp8_dequantize_op",
      [&] { impl::fp8_dequantize<scalar_t>(input, output, itype, scale_inv); });
}

at::Tensor cast_to_fp8(
    const at::Tensor& input_,
    const at::Tensor& scale,
    at::Tensor& amax,
    at::Tensor& scale_inv,
    int64_t fp8_tensor,
    int64_t otype) {
  at::ScalarType out_type = convert_to_dtype(otype);
  auto input = input_.is_contiguous() ? input_ : input_.contiguous();
  auto output = at::empty(input.sizes(), input.options().dtype(out_type));
  fp8_quantize_op(
      input,
      output,
      otype,
      amax[fp8_tensor].data_ptr<float>(),
      scale[fp8_tensor].data_ptr<float>(),
      scale_inv[fp8_tensor].data_ptr<float>());
  return output;
}

at::Tensor cast_from_fp8(
    const at::Tensor& input_,
    const at::Tensor& scale_inv,
    int64_t fp8_tensor,
    int64_t itype,
    ScalarType otype) {
  auto input = input_.is_contiguous() ? input_ : input_.contiguous();
  auto output = at::empty(input.sizes(), input.options().dtype(otype));
  fp8_dequantize_op(
      input, output, itype, scale_inv[fp8_tensor].data_ptr<float>());
  return output;
}

IPEX_LIBRARY_FRAGMENT() {
  IPEX_OP_REGISTER_DISPATCH(
      "cast_to_fp8.xpu",
      at::AtenIpexTypeXPU::cast_to_fp8,
      c10::DispatchKey::XPU);
}

IPEX_LIBRARY_FRAGMENT() {
  IPEX_OP_REGISTER_DISPATCH(
      "cast_from_fp8.xpu",
      at::AtenIpexTypeXPU::cast_from_fp8,
      c10::DispatchKey::XPU);
}

} // namespace AtenIpexTypeXPU
} // namespace at
