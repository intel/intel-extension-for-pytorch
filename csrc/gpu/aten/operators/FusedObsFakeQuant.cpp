#include <ATen/ATen.h>
#include <iostream>
#include <tuple>
#include "DistributionTemplates.h"
#include "comm/Numerics.h"
using namespace xpu::dpcpp;

namespace at {
namespace AtenIpexTypeXPU {

void ChooseQuantizationParamsKernelImpl(
    const int64_t* fake_quant_on,
    const float* x_min,
    const float* x_max,
    int32_t qmin,
    int32_t qmax,
    int size,
    bool preserve_sparsity,
    float* scale,
    int32_t* zero_point,
    sycl::nd_item<1>& item) {
  int i = item.get_group(0) * item.get_local_range(0) + item.get_local_id(0);

  if (i < size && *fake_quant_on == 1) {
    float min_val = x_min[i];
    float max_val = x_max[i];

    if (min_val < 0 && max_val > 0 && preserve_sparsity) {
      int symmetric_qmin = -((qmax - qmin) / 2 + 1);
      int symmetric_qmax = (qmax - qmin) / 2;

      float max_scale = Numerics<float>::max(
          Numerics<float>::fabs(min_val / symmetric_qmin),
          Numerics<float>::fabs(max_val / symmetric_qmax));
      min_val = max_scale * symmetric_qmin;
      max_val = max_scale * symmetric_qmax;
    }

    // We extend the [min, max] interval to ensure that it contains 0.
    // Otherwise, we would not meet the requirement that 0 be an exactly
    // representable value.
    min_val = Numerics<float>::min(min_val, 0.f);
    max_val = Numerics<float>::max(max_val, 0.f);
    scale[i] = (max_val - min_val) / (qmax - qmin);

    // Moving this check outside this function would result in extra Device to
    // Host copy of the min and max val which would result in a perf hit.
    if (scale[i] == 0.0f || Numerics<float>::isinf(1.0f / scale[i])) {
      scale[i] = 0.1;
    }

    float zero_point_from_min = qmin - min_val / scale[i];
    float zero_point_from_max = qmax - max_val / scale[i];
    float zero_point_from_min_error =
        Numerics<float>::abs(qmin) + Numerics<float>::abs(min_val / scale[i]);
    float zero_point_from_max_error =
        Numerics<float>::abs(qmax) + Numerics<float>::abs(max_val / scale[i]);
    float initial_zero_point =
        zero_point_from_min_error < zero_point_from_max_error
        ? zero_point_from_min
        : zero_point_from_max;

    // Note: preserve_sparsity here means symmetric quantization.
    // for symmetric quantization, we force zero_point
    // to be a middle value between qmin and qmax.
    // If either min or max is 0, then we just use 0 as zero_point.
    if (min_val < 0 && max_val > 0 && preserve_sparsity) {
      initial_zero_point = static_cast<float>(qmin + qmax) / 2;
    }
    // Now we need to nudge the zero point to be an integer
    // (our zero points are integer, and this is motivated by the
    // requirement to be able to represent the real value "0" exactly as a
    // quantized value, which is required in multiple places, for example in
    // Im2col with zero padding).
    int32_t nudged_zero_point = 0;
    if (initial_zero_point < qmin) {
      nudged_zero_point = qmin;
    } else if (initial_zero_point > qmax) {
      nudged_zero_point = qmax;
    } else {
      nudged_zero_point = std::nearbyint(initial_zero_point);
    }
    zero_point[i] = nudged_zero_point;
  }
}

void MovingAverageMinMax(
    const int64_t* observer_on,
    const float* x_min,
    const float* x_max,
    float* running_min,
    float* running_max,
    const float averaging_const,
    const int size,
    sycl::nd_item<1>& item) {
  int i = item.get_group(0) * item.get_local_range(0) + item.get_local_id(0);

  if (*observer_on == 1) {
    if (i < size) {
      float curr_min = x_min[i];
      float curr_max = x_max[i];

      float adjusted_min = Numerics<float>::isinf(running_min[i])
          ? curr_min
          : (running_min[i]) + averaging_const * (curr_min - (running_min[i]));

      float adjusted_max = Numerics<float>::isinf(running_max[i])
          ? curr_max
          : (running_max[i]) + averaging_const * (curr_max - (running_max[i]));

      running_min[i] = adjusted_min;
      running_max[i] = adjusted_max;
    }
  }
}

void _calculate_moving_average(
    const at::Tensor& x,
    const at::Tensor& observer_on,
    at::Tensor& running_min,
    at::Tensor& running_max,
    const float averaging_const,
    const int64_t size,
    bool per_row_fake_quant) {
  auto& sycl_queue = dpcppGetCurrentQueue();
  auto execution_policy = calc_execution_policy(size);
  auto counter_offset = std::get<0>(execution_policy);
  auto num_groups = std::get<1>(execution_policy);
  auto group_size = std::get<2>(execution_policy);

  at::Tensor x_min, x_max;

  int64_t* observer_on_data = observer_on.data_ptr<int64_t>();
  float* running_min_data = running_min.data_ptr<float>();
  float* running_max_data = running_max.data_ptr<float>();

  if (per_row_fake_quant) {
    std::tie(x_min, x_max) = at::_aminmax(x, 1);
    float* x_min_data = x_min.data_ptr<float>();
    float* x_max_data = x_max.data_ptr<float>();

    // Moving Average Min/Max observer for activations
    auto cgf = DPCPP_Q_CGF(cgh) {
      auto kfn = DPCPP_Q_KFN(sycl::nd_item<1> item) {
        MovingAverageMinMax(
            observer_on_data,
            x_min_data,
            x_max_data,
            running_min_data,
            running_max_data,
            averaging_const,
            size,
            item);
      };
      cgh.parallel_for(
          sycl::nd_range<1>(num_groups * group_size, group_size), kfn);
    };
    DPCPP_Q_SUBMIT(sycl_queue, cgf);

  } else {
    std::tie(x_min, x_max) = at::_aminmax(x);
    float* x_min_data = x_min.data_ptr<float>();
    float* x_max_data = x_max.data_ptr<float>();

    // Moving Average Min/Max observer for activations
    auto cgf = DPCPP_Q_CGF(cgh) {
      auto kfn = DPCPP_Q_KFN(sycl::nd_item<1> item) {
        MovingAverageMinMax(
            observer_on_data,
            x_min_data,
            x_max_data,
            running_min_data,
            running_max_data,
            averaging_const,
            size,
            item);
      };
      cgh.parallel_for(sycl::nd_range<1>(num_groups * group_size, 1), kfn);
    };
    DPCPP_Q_SUBMIT(sycl_queue, cgf);
  }
}

void _calc_moving_avg_qparams_helper(
    const at::Tensor& x,
    const at::Tensor fake_quant_on,
    at::Tensor& running_min,
    at::Tensor& running_max,
    float* scale_ptr,
    int32_t* zp_ptr,
    int32_t qmin,
    int32_t qmax,
    bool symmetric_quant,
    const int64_t size,
    bool per_row_fq = false) {
  auto& sycl_queue = dpcppGetCurrentQueue();
  auto execution_policy = calc_execution_policy(size);
  auto counter_offset = std::get<0>(execution_policy);
  auto num_groups = std::get<1>(execution_policy);
  auto group_size = std::get<2>(execution_policy);

  int64_t* fake_quant_on_data = fake_quant_on.data_ptr<int64_t>();
  if (per_row_fq) {
    float* running_min_data = running_min.data_ptr<float>();
    float* running_max_data = running_max.data_ptr<float>();

    auto cgf = DPCPP_Q_CGF(cgh) {
      auto kfn = DPCPP_Q_KFN(sycl::nd_item<1> item) {
        ChooseQuantizationParamsKernelImpl(
            fake_quant_on_data,
            running_min_data,
            running_max_data,
            qmin,
            qmax,
            size,
            symmetric_quant, // preserve_sparsity
            scale_ptr,
            zp_ptr,
            item);
      };
      cgh.parallel_for(
          sycl::nd_range<1>(num_groups * group_size, group_size), kfn);
    };
    DPCPP_Q_SUBMIT(sycl_queue, cgf);

  } else {
    float* running_min_data = running_min.data_ptr<float>();
    float* running_max_data = running_max.data_ptr<float>();

    auto cgf = DPCPP_Q_CGF(cgh) {
      auto kfn = DPCPP_Q_KFN(sycl::nd_item<1> item) {
        ChooseQuantizationParamsKernelImpl(
            fake_quant_on_data,
            running_min_data,
            running_max_data,
            qmin,
            qmax,
            1, // size
            symmetric_quant, // preserve_sparsity
            scale_ptr,
            zp_ptr,
            item);
      };
      cgh.parallel_for(sycl::nd_range<1>(num_groups * group_size, 1), kfn);
    };
    DPCPP_Q_SUBMIT(sycl_queue, cgf);
  }
}

std::tuple<at::Tensor, at::Tensor> _fused_moving_avg_obs_fq_helper(
    const at::Tensor& x,
    const at::Tensor& observer_on,
    const at::Tensor& fake_quant_on,
    at::Tensor& running_min,
    at::Tensor& running_max,
    at::Tensor& scale,
    at::Tensor& zero_point,
    double averaging_const,
    int64_t quant_min,
    int64_t quant_max,
    int64_t ch_axis,
    bool per_row_fake_quant,
    bool symmetric_quant) {
  TORCH_CHECK(
      ch_axis < x.dim(),
      "Error in fused_moving_avg_obs_fq_helper: ch_axis must be < "
      "self.dim()");

  const auto x_contig = x.contiguous();
  // Calculate the size of the dimension we need to quantize over,
  // For per-channel quant we default to axis 0, since it is only for
  // weight quantization currently.
  int64_t size = 1;

  if (per_row_fake_quant) {
    at::Tensor y = x;
    if (x.dim() != 2) {
      auto res = DimVector(x.sizes());
      std::iota(res.begin(), res.end(), 0);
      res[ch_axis] = 0;
      res[0] = ch_axis;

      y = x.permute(res);
      y = y.flatten(1);
    }
    size = x.size(ch_axis);
    if (running_min.numel() == 0) {
      running_min.resize_(size).fill_(Numerics<float>::upper_bound());
      running_max.resize_(size).fill_(Numerics<float>::lower_bound());
      scale.resize_(size);
      zero_point.resize_(size);
    }
    _calculate_moving_average(
        y,
        observer_on,
        running_min,
        running_max,
        averaging_const,
        size,
        per_row_fake_quant);
  } else {
    _calculate_moving_average(
        x_contig,
        observer_on,
        running_min,
        running_max,
        averaging_const,
        size,
        per_row_fake_quant);
  }

  float* scale_ptr = scale.data_ptr<float>();
  int32_t* zp_ptr = zero_point.data_ptr<int32_t>();

  _calc_moving_avg_qparams_helper(
      x_contig,
      fake_quant_on,
      running_min,
      running_max,
      scale_ptr,
      zp_ptr,
      quant_min,
      quant_max,
      symmetric_quant,
      size,
      per_row_fake_quant);

  if (per_row_fake_quant) {
    if (fake_quant_on.item().toInt()) {
      return at::fake_quantize_per_channel_affine_cachemask(
          x, scale, zero_point, 0, quant_min, quant_max);
    } else {
      auto mask = at::ones_like(x, at::kBool, MemoryFormat::Preserve);
      return std::make_tuple(x.clone(), mask);
    }
  } else {
    return at::_fake_quantize_per_tensor_affine_cachemask_tensor_qparams(
        x, scale, zero_point, fake_quant_on, quant_min, quant_max);
  }
}

} // namespace AtenIpexTypeXPU
} // namespace at
