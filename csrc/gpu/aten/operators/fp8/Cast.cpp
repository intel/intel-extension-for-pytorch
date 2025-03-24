/*******************************************************************************
 * Copyright (C) 2024 Intel Corporation
 * This software and the related documents are Intel copyrighted materials,
 * and your use of them is governed by the express license under which they
 * were provided to you ("License"). Unless the License provides otherwise,
 * you may not use, modify, copy, publish, distribute, disclose or transmit
 * this software or the related documents without Intel's prior written
 * permission. This software and the related documents are provided as is,
 * with no express or implied warranties, other than those that are expressly
 * stated in the License.
 *******************************************************************************
 */
#include "Cast.h"
#include <runtime/Utils.h>
#include <runtime/XPUGraphPhiloxGenerator.h>
#include <climits>
#include "../comm/ATDispatch.h"
#include "../comm/Numerics.h"
#include "ATen/xpu/XPUGeneratorImpl.h"
#include "utils/CustomOperatorRegistration.h"

using namespace sycl;
using namespace torch_ipex::xpu::dpcpp;

namespace at {
namespace AtenIpexTypeXPU {
namespace detail {

struct Empty {};

template <class T>
struct Identity {
  T operator()(T value, const Empty&) {
    return value;
  }
};

template <class T>
struct DequantizeParam {
  const T* scale_inv;
};

template <class T>
struct Dequantize {
  T operator()(T value, const DequantizeParam<T>& param) {
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
struct FP8CastingKernelFunctor {
  void operator()(sycl::nd_item<1> item) const {
    auto in_data = in_data_;
    auto out_data = out_data_;
    auto id = item.get_global_linear_id();
    float max = 0;
    if (id < num_elements_) {
      ComputeType s = 0;
      ComputeType val = static_cast<ComputeType>(in_data[id]);
      ComputeType temp = OP()(val, p_);
      if (is_amax_ && is_fp8<OutputType>::value && amax_ != nullptr) {
        max = sycl::fmax(sycl::fabs((float)temp), max);
      }
      if (is_quantize_ && is_fp8<OutputType>::value) {
        if (scale_ != nullptr) {
          s = *scale_;
          if (id == 0 && scale_inv_ != nullptr) {
            *scale_inv_ = sycl::native::recip(s);
          }
        }
        temp = temp * s;
      }
      out_data[id] = (OutputType)(temp);
    }
    if (is_amax_ && is_fp8<OutputType>::value && amax_ != nullptr) {
      auto group_max = sycl::reduce_over_group(
          item.get_group(), max, sycl::maximum<float>());
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
  FP8CastingKernelFunctor(
      const InputType* in_data,
      OutputType* out_data,
      const ComputeType* scale,
      ComputeType* scale_inv,
      float* amax,
      Param p,
      int num_elements,
      bool is_amax,
      bool is_quantize)
      : in_data_(in_data),
        out_data_(out_data),
        scale_(scale),
        scale_inv_(scale_inv),
        amax_(amax),
        p_(p),
        num_elements_(num_elements),
        is_amax_(is_amax),
        is_quantize_(is_quantize) {}

 private:
  const InputType* in_data_;
  OutputType* out_data_;
  const ComputeType* scale_;
  ComputeType* scale_inv_;
  float* amax_;
  Param p_;
  int num_elements_;
  bool is_amax_;
  bool is_quantize_;
};

template <typename ComputeType, typename Param, typename OP, typename InputType>
struct FP8CastingHybridKernelFunctor {
  void operator()(sycl::nd_item<1> item) const {
    auto in_data = in_data_;
    auto out_data_e4m3 = out_data_e4m3_;
    auto out_data_e5m2 = out_data_e5m2_;
    auto id = item.get_global_linear_id();
    float max = 0;
    if (id < num_elements_) {
      ComputeType val = static_cast<ComputeType>(in_data[id]);
      ComputeType temp = OP()(val, p_);
      ComputeType temp_143 = temp;
      ComputeType temp_152 = temp;
      if (is_amax_ && amax_ != nullptr) {
        max = sycl::fmax(sycl::fabs((float)temp), max);
      }
      if (is_quantize_143_ && scale_143_ != nullptr) {
        ComputeType s_143 = *scale_143_;
        temp_143 = temp * s_143;
      }
      if (is_quantize_152_ && scale_152_ != nullptr) {
        ComputeType s_152 = *scale_152_;
        temp_152 = temp * s_152;
      }
      out_data_e4m3[id] = (fp8e4m3)(temp_143);
      out_data_e5m2[id] = (fp8e5m2)(temp_152);
    }
    if (is_amax_ && amax_ != nullptr) {
      auto group_max = sycl::reduce_over_group(
          item.get_group(), max, sycl::maximum<float>());
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
  FP8CastingHybridKernelFunctor(
      const InputType* in_data,
      fp8e4m3* out_data_e4m3,
      fp8e5m2* out_data_e5m2,
      const ComputeType* scale_143,
      const ComputeType* scale_152,
      float* amax,
      Param p,
      int num_elements,
      bool is_amax,
      bool is_quantize_152,
      bool is_quantize_143)
      : in_data_(in_data),
        out_data_e4m3_(out_data_e4m3),
        out_data_e5m2_(out_data_e5m2),
        scale_143_(scale_143),
        scale_152_(scale_152),
        amax_(amax),
        p_(p),
        num_elements_(num_elements),
        is_amax_(is_amax),
        is_quantize_152_(is_quantize_152),
        is_quantize_143_(is_quantize_143) {}

 private:
  const InputType* in_data_;
  fp8e4m3* out_data_e4m3_;
  fp8e5m2* out_data_e5m2_;
  const ComputeType* scale_143_;
  const ComputeType* scale_152_;
  float* amax_;
  Param p_;
  int num_elements_;
  bool is_amax_;
  bool is_quantize_152_;
  bool is_quantize_143_;
};

template <
    typename ComputeType,
    typename Param,
    typename OP,
    typename InputType,
    typename OutputType>
void launch_fp8_casting_kernel(
    const InputType* in_,
    OutputType* out_,
    const ComputeType* scale_,
    ComputeType* scale_inv_,
    float* amax_,
    Param p_,
    int num_elements_,
    bool is_amax_ = true,
    bool is_quantize_ = true) {
  // TODO: do vectorized load/store for better perf
  auto& dpcpp_queue = dpcppGetCurrentQueue();
  auto dev_id = dpcppGetDeviceIdOfCurrentQueue();

  auto maxGroupsize = dpcppMaxWorkGroupSize(dev_id);
  auto wgroup_size = std::min(num_elements_, (int)maxGroupsize);
  int wgroup_num = CeilDiv((int64_t)num_elements_, (int64_t)wgroup_size);
  auto cgf = DPCPP_Q_CGF(cgh) {
    auto in_data_ = in_;
    auto out_data_ = out_;
    FP8CastingKernelFunctor<ComputeType, Param, OP, InputType, OutputType> kfn(
        in_data_,
        out_data_,
        scale_,
        scale_inv_,
        amax_,
        p_,
        num_elements_,
        is_amax_,
        is_quantize_);

    cgh.parallel_for<decltype(kfn)>(
        sycl::nd_range<1>(wgroup_num * wgroup_size, wgroup_size), kfn);
  };
  DPCPP_Q_SUBMIT(dpcpp_queue, cgf);
}

template <typename ComputeType, typename Param, typename OP, typename InputType>
void launch_fp8_casting_hybrid_kernel(
    const InputType* in_,
    fp8e4m3* out_e4m3_,
    fp8e5m2* out_e5m2_,
    const ComputeType* scale_152_,
    const ComputeType* scale_143_,
    float* amax_,
    Param p_,
    int num_elements_,
    bool is_amax_,
    bool is_quantize_152_,
    bool is_quantize_143_) {
  // TODO: do vectorized load/store for better perf
  auto& dpcpp_queue = dpcppGetCurrentQueue();
  auto dev_id = dpcppGetDeviceIdOfCurrentQueue();

  auto maxGroupsize = dpcppMaxWorkGroupSize(dev_id);
  auto wgroup_size = std::min(num_elements_, (int)maxGroupsize);
  int wgroup_num = CeilDiv((int64_t)num_elements_, (int64_t)wgroup_size);
  auto cgf = DPCPP_Q_CGF(cgh) {
    auto in_data_ = in_;
    auto out_data_e4m3 = out_e4m3_;
    auto out_data_e5m2 = out_e5m2_;
    FP8CastingHybridKernelFunctor<ComputeType, Param, OP, InputType> kfn(
        in_data_,
        out_data_e4m3,
        out_data_e5m2,
        scale_143_,
        scale_152_,
        amax_,
        p_,
        num_elements_,
        is_amax_,
        is_quantize_152_,
        is_quantize_143_);

    cgh.parallel_for<decltype(kfn)>(
        sycl::nd_range<1>(wgroup_num * wgroup_size, wgroup_size), kfn);
  };
  DPCPP_Q_SUBMIT(dpcpp_queue, cgf);
}

template <typename T> //, bool HighPrecisionOut = false>
void fp8_quantize(
    const Tensor& input,
    Tensor& output,
    int fp8_format,
    void* amax,
    const void* scale,
    void* scale_inv,
    bool is_SR,
    bool is_amax,
    bool is_quantize) {
  int num_elements = input.numel();

  if (is_SR) {
    TORCH_CHECK(
        false,
        "FP8 Stochastic Rounding depend on the ESIMD kernels by currently so we removed it.\n");
  } else {
    IPEX_TYPE_SWITCH_FP8ONLY(fp8_format,
                             type,
                             launch_fp8_casting_kernel<
                                 T,
                                 detail::Empty,
                                 detail::Identity<T>,
                                 T,
                                 type>(
                                 input.data_ptr<T>(),
                                 output.data_ptr<type>(),
                                 (T*)scale,
                                 (T*)scale_inv,
                                 (float*)amax,
                                 {},
                                 num_elements,
                                 is_amax,
                                 is_quantize););
  }
}

template <typename T> //, bool HighPrecisionOut = false>
void fp8_quantize_hybrid(
    const Tensor& input,
    Tensor& output_e4m3,
    Tensor& output_e5m2,
    void* amax,
    const void* scale_152,
    const void* scale_143,
    bool is_SR,
    bool is_amax,
    bool is_quantize_152,
    bool is_quantize_143) {
  int num_elements = input.numel();

  if (is_SR) {
    TORCH_CHECK(
        false,
        "FP8 Stochastic Rounding depend on the ESIMD kernels by currently so we removed it.\n");
  } else {
    launch_fp8_casting_hybrid_kernel<T, detail::Empty, detail::Identity<T>, T>(
        input.data_ptr<T>(),
        output_e4m3.data_ptr<fp8e4m3>(),
        output_e5m2.data_ptr<fp8e5m2>(),
        (T*)scale_152,
        (T*)scale_143,
        (float*)amax,
        {},
        num_elements,
        is_amax,
        is_quantize_152,
        is_quantize_143);
  }
}

template <typename Tout>
void fp8_dequantize(
    const Tensor& input,
    Tensor& output,
    int fp8_format,
    const void* scale_inv,
    bool is_dequantize) {
  int num_elements = input.numel();
  if (is_dequantize) {
    IPEX_TYPE_SWITCH_FP8ONLY(
        fp8_format, scalar_t, detail::DequantizeParam<Tout> P;
        P.scale_inv = (Tout*)scale_inv;
        launch_fp8_casting_kernel<
            Tout,
            detail::DequantizeParam<Tout>,
            detail::Dequantize<Tout>,
            scalar_t,
            Tout>(
            FP8_PTR(input.data_ptr<scalar_t>(), scalar_t),
            output.data_ptr<Tout>(),
            nullptr,
            nullptr,
            nullptr,
            P,
            num_elements,
            false,
            false););
  } else {
    IPEX_TYPE_SWITCH_FP8ONLY(fp8_format,
                             scalar_t,
                             launch_fp8_casting_kernel<
                                 Tout,
                                 detail::Empty,
                                 detail::Identity<Tout>,
                                 scalar_t,
                                 Tout>(
                                 FP8_PTR(input.data_ptr<scalar_t>(), scalar_t),
                                 output.data_ptr<Tout>(),
                                 nullptr,
                                 nullptr,
                                 nullptr,
                                 {},
                                 num_elements,
                                 false,
                                 false););
  }
}

} // namespace impl

// namespace torch_ipex::xpu {
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

int64_t convert_from_dtype(at::ScalarType dtype) {
  switch (dtype) {
    case ScalarType::Float8_e5m2:
      return Float8Format::kFloat8_E5M2;
    case ScalarType::Float8_e4m3fn:
      return Float8Format::kFloat8_E4M3;
    default:
      TORCH_CHECK(false, "undefined FP8 Dtype.\n");
  }
}

void fp8_quantize_op(
    const Tensor& input,
    Tensor& output,
    int fp8_format,
    void* amax,
    const void* scale,
    void* scale_inv,
    bool is_SR,
    bool is_amax,
    bool is_quantize) {
  IPEX_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::BFloat16,
      at::ScalarType::Half,
      input.scalar_type(),
      "fp8_quantize",
      [&] {
        impl::fp8_quantize<scalar_t>(
            input,
            output,
            fp8_format,
            amax,
            scale,
            scale_inv,
            is_SR,
            is_amax,
            is_quantize);
      });
}

void fp8_quantize_hybrid_op(
    const Tensor& input,
    Tensor& output_e4m3,
    Tensor& output_e5m2,
    void* amax,
    const void* scale_152,
    const void* scale_143,
    bool is_SR,
    bool is_amax,
    bool is_quantize_152,
    bool is_quantize_143) {
  IPEX_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::BFloat16,
      at::ScalarType::Half,
      input.scalar_type(),
      "fp8_quantize_hybrid",
      [&] {
        impl::fp8_quantize_hybrid<scalar_t>(
            input,
            output_e4m3,
            output_e5m2,
            amax,
            scale_152,
            scale_143,
            is_SR,
            is_amax,
            is_quantize_152,
            is_quantize_143);
      });
}

void fp8_dequantize_op(
    const Tensor& input,
    Tensor& output,
    int64_t itype,
    const void* scale_inv,
    bool is_dequantize) {
  IPEX_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::BFloat16,
      at::ScalarType::Half,
      output.scalar_type(),
      "fp8_dequantize_op",
      [&] {
        impl::fp8_dequantize<scalar_t>(
            input, output, itype, scale_inv, is_dequantize);
      });
}

at::Tensor cast_to_fp8(
    at::Tensor& input_,
    at::Tensor& scale_,
    at::Tensor& amax,
    at::Tensor& scale_inv_,
    int64_t fp8_tensor,
    int64_t otype) {
  at::ScalarType out_type = convert_to_dtype(otype);
  auto input = input_.is_contiguous() ? input_ : input_.contiguous();
  auto output = at::empty(input.sizes(), input.options().dtype(out_type));
  Tensor scale, scale_inv;
  if (input_.dtype() != scale_[fp8_tensor].dtype()) {
    scale = scale_[fp8_tensor].to(input_.dtype());
  } else {
    scale = scale_[fp8_tensor];
  }
  if (input_.dtype() != scale_inv_[fp8_tensor].dtype()) {
    scale_inv = scale_inv_[fp8_tensor].to(input_.dtype());
  } else {
    scale_inv = scale_inv_[fp8_tensor];
  }
  fp8_quantize_op(
      input,
      output,
      otype,
      amax[fp8_tensor].data_ptr(),
      scale.data_ptr(),
      scale_inv.data_ptr());
  scale_inv_[fp8_tensor] = scale_inv;
  return output;
}

std::tuple<at::Tensor, at::Tensor> cast_to_fp8_tensor(
    const at::Tensor& input_,
    const c10::optional<at::Tensor>& scale_,
    bool stochastic_rounding,
    bool is_amax,
    int64_t otype,
    const c10::optional<std::vector<int64_t>>& scale_shape) {
  TORCH_CHECK(
      !scale_shape.has_value(),
      "The scale_shape will only be used for scalar_list scale\n");
  auto input = input_.is_contiguous() ? input_ : input_.contiguous();
  at::ScalarType out_type = convert_to_dtype(otype);
  auto output = at::empty(input.sizes(), input.options().dtype(out_type));
  bool is_quantize = scale_.has_value() ? true : false;
  Tensor scale;
  if (is_quantize) {
    if (input_.dtype() != scale_.value().dtype()) {
      scale = scale_.value().to(input_.dtype());
    } else {
      scale = scale_.value();
    }
  }

  Tensor amax = is_amax ? at::zeros({}, at::dtype(at::kFloat).device(at::kXPU))
                        : at::empty({}, at::dtype(at::kFloat).device(at::kXPU));
  fp8_quantize_op(
      input,
      output,
      otype,
      amax.data_ptr(),
      is_quantize ? scale.data_ptr() : nullptr,
      nullptr,
      stochastic_rounding,
      is_amax,
      is_quantize);
  return std::make_tuple(output, amax);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> cast_to_fp8_hybrid(
    const at::Tensor& input_,
    const c10::optional<at::Tensor>& scale_152_,
    const c10::optional<at::Tensor>& scale_143_,
    bool stochastic_rounding,
    bool is_amax) {
  auto input = input_.is_contiguous() ? input_ : input_.contiguous();
  auto output_e5m2 =
      at::empty(input.sizes(), input.options().dtype(ScalarType::Float8_e5m2));
  auto output_e4m3 = at::empty(
      input.sizes(), input.options().dtype(ScalarType::Float8_e4m3fn));
  Tensor scale_152, scale_143;
  bool is_quantize_152 = scale_152_.has_value() ? true : false;
  bool is_quantize_143 = scale_143_.has_value() ? true : false;
  if (is_quantize_152) {
    if (input_.dtype() != scale_152_.value().dtype()) {
      scale_152 = scale_152_.value().to(input_.dtype());
    } else {
      scale_152 = scale_152_.value();
    }
  }
  if (is_quantize_143) {
    if (input_.dtype() != scale_143_.value().dtype()) {
      scale_143 = scale_143_.value().to(input_.dtype());
    } else {
      scale_143 = scale_143_.value();
    }
  }
  Tensor amax = is_amax ? at::zeros({}, at::dtype(at::kFloat).device(at::kXPU))
                        : at::empty({}, at::dtype(at::kFloat).device(at::kXPU));
  fp8_quantize_hybrid_op(
      input,
      output_e4m3,
      output_e5m2,
      amax.data_ptr<float>(),
      is_quantize_152 ? scale_152.data_ptr() : nullptr,
      is_quantize_143 ? scale_143.data_ptr() : nullptr,
      stochastic_rounding,
      is_amax,
      is_quantize_152,
      is_quantize_143);
  return std::make_tuple(output_e5m2, output_e4m3, amax);
}

std::tuple<at::Tensor, at::Tensor> cast_to_fp8_scalar(
    const at::Tensor& input_,
    const c10::optional<c10::Scalar>& scale_,
    bool stochastic_rounding,
    bool is_amax,
    int64_t otype,
    const c10::optional<std::vector<int64_t>>& scale_shape) {
  TORCH_CHECK(
      !scale_shape.has_value(),
      "The scale_shape will only be used for scalar_list scale\n");
  auto input = input_.is_contiguous() ? input_ : input_.contiguous();
  at::ScalarType out_type = convert_to_dtype(otype);
  auto output = at::empty(input.sizes(), input.options().dtype(out_type));
  bool is_quantize = scale_.has_value() ? true : false;
  Tensor scale_tensor_ = at::empty({}, input_.options());
  Tensor scale_tensor;
  if (is_quantize) {
    scale_tensor_.fill_(scale_.value());
    if (input_.dtype() != scale_tensor_.dtype()) {
      scale_tensor = scale_tensor_.to(input_.dtype());
    } else {
      scale_tensor = scale_tensor_;
    }
  }

  Tensor amax = is_amax ? at::zeros({}, at::dtype(at::kFloat).device(at::kXPU))
                        : at::empty({}, at::dtype(at::kFloat).device(at::kXPU));
  fp8_quantize_op(
      input,
      output,
      otype,
      amax.data_ptr(),
      is_quantize ? scale_tensor.data_ptr() : nullptr,
      nullptr,
      stochastic_rounding,
      is_amax,
      is_quantize);
  return std::make_tuple(output, amax);
}

std::tuple<at::Tensor, at::Tensor> cast_to_fp8_list(
    const at::Tensor& input_,
    const c10::optional<std::vector<c10::Scalar>>& scale_,
    bool stochastic_rounding,
    bool is_amax,
    int64_t otype,
    const c10::optional<std::vector<int64_t>>& scale_shape) {
  TORCH_CHECK(
      false,
      "The scale scalar list type will be used for channel-wise and block-wise quantization, which still work in progress by currently\n");
  // TODO: implement channel-wise and block-wise quantization
}

at::Tensor cast_to_fp8_out(
    const at::Tensor& input_,
    const at::Tensor& scale_,
    at::Tensor output,
    at::Tensor& amax,
    at::Tensor& scale_inv_,
    int64_t fp8_tensor,
    int64_t otype) {
  at::ScalarType out_type = convert_to_dtype(otype);
  auto input = input_.is_contiguous() ? input_ : input_.contiguous();
  Tensor scale, scale_inv;
  if (input_.dtype() != scale_[fp8_tensor].dtype()) {
    scale = scale_[fp8_tensor].to(input_.dtype());
  } else {
    scale = scale_[fp8_tensor];
  }
  if (input_.dtype() != scale_inv_[fp8_tensor].dtype()) {
    scale_inv = scale_inv_[fp8_tensor].to(input_.dtype());
  } else {
    scale_inv = scale_inv_[fp8_tensor];
  }
  fp8_quantize_op(
      input,
      output,
      otype,
      amax[fp8_tensor].data_ptr(),
      scale[fp8_tensor].data_ptr(),
      scale_inv[fp8_tensor].data_ptr());
  scale_inv_[fp8_tensor] = scale_inv;
  return output;
}

at::Tensor cast_from_fp8(
    at::Tensor& input_,
    at::Tensor& scale_inv_,
    int64_t fp8_tensor,
    int64_t itype,
    ScalarType otype) {
  auto input = input_.is_contiguous() ? input_ : input_.contiguous();
  auto output = at::empty(input.sizes(), input.options().dtype(otype));
  Tensor scale_inv;
  if (output.dtype() != scale_inv_[fp8_tensor].dtype()) {
    scale_inv = scale_inv_[fp8_tensor].to(output.dtype());
  } else {
    scale_inv = scale_inv_[fp8_tensor];
  }
  fp8_dequantize_op(input, output, itype, scale_inv.data_ptr());
  return output;
}

at::Tensor cast_from_fp8_tensor(
    const at::Tensor& input_,
    const c10::optional<at::Tensor>& scale_inv_,
    ScalarType otype) {
  Tensor scale_inv;
  auto input = input_.is_contiguous() ? input_ : input_.contiguous();
  auto output = at::empty(input.sizes(), input.options().dtype(otype));
  auto itype = convert_from_dtype(input.scalar_type());
  bool is_dequantize = scale_inv_.has_value() ? true : false;
  if (is_dequantize) {
    if (output.dtype() != scale_inv_.value().dtype()) {
      scale_inv = scale_inv_.value().to(output.dtype());
    } else {
      scale_inv = scale_inv_.value();
    }
  }
  fp8_dequantize_op(
      input,
      output,
      itype,
      is_dequantize ? scale_inv.data_ptr() : nullptr,
      is_dequantize);
  return output;
}

at::Tensor cast_from_fp8_scalar(
    at::Tensor& input_,
    const c10::optional<c10::Scalar>& scale_inv,
    ScalarType otype) {
  auto input = input_.is_contiguous() ? input_ : input_.contiguous();
  auto output = at::empty(input.sizes(), input.options().dtype(otype));
  auto itype = convert_from_dtype(input.scalar_type());
  bool is_dequantize = scale_inv.has_value() ? true : false;
  Tensor scale_inv_tensor_ = at::empty({}, output.options());
  Tensor scale_inv_tensor;
  if (is_dequantize) {
    scale_inv_tensor_.fill_(scale_inv.value());
    if (output.dtype() != scale_inv_tensor_.dtype()) {
      scale_inv_tensor = scale_inv_tensor_.to(output.dtype());
    } else {
      scale_inv_tensor = scale_inv_tensor_;
    }
  }
  fp8_dequantize_op(
      input,
      output,
      itype,
      is_dequantize ? scale_inv_tensor.data_ptr() : nullptr,
      is_dequantize);
  return output;
}

at::Tensor cast_from_fp8_list(
    at::Tensor& input_,
    const c10::optional<std::vector<c10::Scalar>>& scale_inv,
    ScalarType otype) {
  TORCH_CHECK(
      false,
      "The scale scalar list type will be used for channel-wise and block-wise de-quantization, which still work in progress by currently\n");
  // TODO: implement channel-wise and block-wise de-quantization
}

IPEX_LIBRARY_FRAGMENT() {
  IPEX_OP_REGISTER_DISPATCH(
      "cast_to_fp8", at::AtenIpexTypeXPU::cast_to_fp8, c10::DispatchKey::XPU);
}

IPEX_LIBRARY_FRAGMENT() {
  IPEX_OP_REGISTER_DISPATCH(
      "cast_to_fp8.Tensor",
      at::AtenIpexTypeXPU::cast_to_fp8_tensor,
      c10::DispatchKey::XPU);
}

IPEX_LIBRARY_FRAGMENT() {
  IPEX_OP_REGISTER_DISPATCH(
      "cast_to_fp8_hybrid.Tensor",
      at::AtenIpexTypeXPU::cast_to_fp8_hybrid,
      c10::DispatchKey::XPU);
}

IPEX_LIBRARY_FRAGMENT() {
  IPEX_OP_REGISTER_DISPATCH(
      "cast_from_fp8",
      at::AtenIpexTypeXPU::cast_from_fp8,
      c10::DispatchKey::XPU);
}

IPEX_LIBRARY_FRAGMENT() {
  IPEX_OP_REGISTER_DISPATCH(
      "cast_from_fp8.Tensor",
      at::AtenIpexTypeXPU::cast_from_fp8_tensor,
      c10::DispatchKey::XPU);
}

IPEX_LIBRARY_FRAGMENT() {
  IPEX_OP_REGISTER_DISPATCH(
      "cast_to_fp8.Scalar",
      at::AtenIpexTypeXPU::cast_to_fp8_scalar,
      c10::DispatchKey::XPU);
}

IPEX_LIBRARY_FRAGMENT() {
  IPEX_OP_REGISTER_DISPATCH(
      "cast_from_fp8.Scalar",
      at::AtenIpexTypeXPU::cast_from_fp8_scalar,
      c10::DispatchKey::XPU);
}

IPEX_LIBRARY_FRAGMENT() {
  IPEX_OP_REGISTER_DISPATCH(
      "cast_to_fp8.List",
      at::AtenIpexTypeXPU::cast_to_fp8_list,
      c10::DispatchKey::XPU);
}

IPEX_LIBRARY_FRAGMENT() {
  IPEX_OP_REGISTER_DISPATCH(
      "cast_from_fp8.List",
      at::AtenIpexTypeXPU::cast_from_fp8_list,
      c10::DispatchKey::XPU);
}

} // namespace AtenIpexTypeXPU
} // namespace at
