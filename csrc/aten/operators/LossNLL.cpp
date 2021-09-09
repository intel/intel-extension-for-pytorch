#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/Functions.h>
#include <ATen/TensorUtils.h>
#include <ATen/core/Reduction.h>

#include <core/Memory.h>
#include <core/TensorImplUtils.h>
#include <core/detail/TensorInfo.h>
#include <runtime/Utils.h>
#include <utils/DPCPP.h>
#include "Reduce.h"
#include "comm/ATDispatch.h"
#include "comm/AccumulateType.h"
#include "comm/Atomics.h"
#include "comm/Numerics.h"
#include "comm/SimpelReduce.h"

using namespace xpu::dpcpp::detail;
using namespace xpu::dpcpp;

namespace at {
namespace AtenIpexTypeXPU {
namespace impl {

template <typename scalar_t>
void ClassNLLCriterion_updateOutput(
    const Tensor& input,
    const Tensor& target,
    Tensor& output,
    const Tensor& weights,
    Tensor& total_weight,
    int64_t reduction,
    int64_t ignore_index) {
  TORCH_CHECK(
      input.dim() > 0 && input.dim() <= 2, "input tensor should be 1D or 2D");
  TORCH_CHECK(
      target.dim() == 1,
      "1D target tensor expected, multi-target not supported");
  TORCH_CHECK(
      input.size(0) == target.size(0),
      "size mismatch (got input: ",
      input.sizes(),
      ", target: ",
      target.sizes(),
      ")")

  int n_dims = input.dim();
  int n_classes = input.size(-1);
  ignore_index -= 0;

  int64_t batch_size = input.size(0);
  int64_t num_targets = target.size(0);
  int64_t target_stride = target.stride(0);

  TORCH_CHECK(
      !weights.defined() || weights.numel() == n_classes,
      "weight tensor should be defined either for all ",
      n_classes,
      " classes or no classes"
      " but got weight tensor of shape: ",
      weights.sizes());

  if (reduction == at::Reduction::None && n_dims == 2) {
    output.resize_({batch_size});

    auto weights_cont = weights.defined() ? weights.contiguous() : weights;

    auto& queue = dpcppGetCurrentQueue();
    auto dev_id = dpcppGetDeviceIdOfCurrentQueue();
    int64_t local_size = dpcppMaxWorkGroupSize(dev_id);
    bool has_weights = weights.defined()
        ? true
        : false; // dpcpp kernel can not accept host pointer

    auto output_stride_0 = output.stride(0);
    auto input_stride_0 = input.stride(0);
    auto input_stride_1 = input.stride(1);
    auto cgf = DPCPP_Q_CGF(cgh) {
      auto input_data = input.data_ptr<scalar_t>();
      auto target_data = target.data_ptr<int64_t>();
      auto weights_data = has_weights
          ? weights_cont.data_ptr<scalar_t>()
          : input_data; // use the input as the dummy data.
      auto output_data = output.data_ptr<scalar_t>();
      auto kfn = DPCPP_Q_KFN(DPCPP::item<1> item_id) {
        auto input_ptr = input_data;
        auto target_ptr = target_data;
        auto weights_ptr = has_weights ? weights_data : NULL;
        auto output_ptr = output_data;
        auto local_item_id = item_id.get_id(0);
        for (int i = local_item_id; i < batch_size; i += local_size) {
          int cur_target = target_ptr[i * target_stride];
          if (cur_target >= 0 && cur_target < n_classes)
            if (cur_target == ignore_index) {
              output_ptr[i * output_stride_0] = 0.0f;
              continue;
            }
          scalar_t cur_weight = has_weights ? weights_ptr[cur_target]
                                            : static_cast<scalar_t>(1.0f);
          output_ptr[i * output_stride_0] =
              -static_cast<scalar_t>(
                  input_ptr[i * input_stride_0 + cur_target * input_stride_1]) *
              cur_weight;
        }
      };

      cgh.parallel_for(DPCPP::range<1>(local_size), kfn);
    };

    DPCPP_Q_SUBMIT(queue, cgf);
    return;
  }

  output.resize_({});
  total_weight.resize_({});

  auto input_cont = input.contiguous();
  auto weights_cont = weights.defined() ? weights.contiguous() : weights;
  auto target_cont = target.contiguous();

  scalar_t* _input_data = input_cont.data_ptr<scalar_t>();
  scalar_t* _weights_data =
      weights.defined() ? weights_cont.data_ptr<scalar_t>() : NULL;
  int64_t* _target_data = target_cont.data_ptr<int64_t>();
  scalar_t* _output_data = output.data_ptr<scalar_t>();
  scalar_t* _total_weight_data = total_weight.data_ptr<scalar_t>();
  bool has_weights = _weights_data != NULL ? true : false;
  auto& queue = dpcppGetCurrentQueue();

  if (input_cont.dim() == 1 || input_cont.dim() == 0) {
    int64_t local_size = 1;

    auto cgf = DPCPP_Q_CGF(cgh) {
      auto input_data = _input_data;
      auto weights_data = has_weights
          ? _weights_data
          : input_data; // use the input as the dummy data.
      auto target_data = _target_data;
      auto total_weight_data = _total_weight_data;
      auto output_data = _output_data;
      auto kfn = DPCPP_Q_KFN(DPCPP::item<1> item_id) {
        auto input_ptr = input_data;
        auto target_ptr = target_data;
        auto weights_ptr = has_weights ? weights_data : NULL;
        auto total_weight_ptr = total_weight_data;
        auto output_ptr = output_data;
        // auto local_item_id = item_id.get_id(0);
        int cur_target = target_ptr[0];
        if (cur_target != ignore_index) {
          total_weight_ptr[0] = has_weights ? weights_ptr[cur_target]
                                            : static_cast<scalar_t>(1.0f);
          output_ptr[0] = -static_cast<scalar_t>(input_ptr[cur_target]) *
              static_cast<scalar_t>(total_weight_ptr[0]);
        }
        if (reduction == at::Reduction::Mean && total_weight_ptr[0]) {
          output_ptr[0] /= total_weight_ptr[0];
        }
      };
      cgh.parallel_for(DPCPP::range<1>(local_size), kfn);
    };

    DPCPP_Q_SUBMIT(queue, cgf);
  } else if (input.dim() == 2) {
    int64_t batch_size = input.size(0);
    int n_target = input.size(1);
    auto dev_id = dpcppGetDeviceIdOfCurrentQueue();
    int64_t local_size = dpcppMaxWorkGroupSize(dev_id);

    auto cgf = DPCPP_Q_CGF(cgh) {
      auto input_data = _input_data;
      auto weights_data = has_weights
          ? _weights_data
          : input_data; // use the input as the dummy data.
      auto target_data = _target_data;
      auto total_weight_data = _total_weight_data;
      auto output_data = _output_data;
      auto local_output_acc = dpcpp_local_acc_t<scalar_t>(local_size, cgh);
      auto local_total_weight_acc =
          dpcpp_local_acc_t<scalar_t>(local_size, cgh);

      auto kfn = DPCPP_Q_KFN(DPCPP::nd_item<1> item_id) {
        auto input_ptr = input_data;
        auto target_ptr = target_data;
        auto weights_ptr = has_weights ? weights_data : NULL;
        auto total_weight_ptr = total_weight_data;
        auto output_ptr = output_data;
        int64_t local_id = item_id.get_local_id(0);
        local_output_acc[local_id] = 0.0;
        local_total_weight_acc[local_id] = 0.0;
        for (int i = local_id; i < batch_size; i += local_size) {
          int cur_target = target_ptr[i];
          if (cur_target != ignore_index) {
            scalar_t cur_weight = has_weights ? weights_ptr[cur_target]
                                              : static_cast<scalar_t>(1.0f);
            local_total_weight_acc[local_id] += cur_weight;
            local_output_acc[local_id] -=
                static_cast<scalar_t>(input_ptr[i * n_target + cur_target]) *
                static_cast<scalar_t>(cur_weight);
          }
        }

        // reduce
        for (int64_t i = (local_size >> 1); i > 0; i >>= 1) {
          item_id.barrier(dpcpp_global_and_local_fence);
          if (local_id < i) {
            local_total_weight_acc[local_id] +=
                local_total_weight_acc[local_id + i];
            local_output_acc[local_id] += local_output_acc[local_id + i];
          }
        }
        item_id.barrier(dpcpp_global_and_local_fence);

        output_ptr[0] = local_output_acc[0];
        total_weight_ptr[0] = local_total_weight_acc[0];
        if (reduction == at::Reduction::Mean && total_weight_ptr[0]) {
          output_ptr[0] /= total_weight_ptr[0];
        }
      };
      cgh.parallel_for(
          DPCPP::nd_range<1>(
              DPCPP::range<1>(local_size), DPCPP::range<1>(local_size)),
          kfn);
    };

    DPCPP_Q_SUBMIT(queue, cgf);
  }
}

template <typename scalar_t>
void ClassNLLCriterion_updateGradInput(
    const Tensor& input,
    const Tensor& target,
    const Tensor& gradOutput,
    Tensor& gradInput,
    int64_t reduction,
    const Tensor& weights,
    const Tensor& total_weight,
    int64_t ignore_index) {
  TORCH_CHECK(
      target.dim() == 1,
      "1D target tensor expected, multi-target not supported");

  int n_dims = input.dim();
  int n_classes = input.size(-1);

  gradInput.resize_as_(input);
  gradInput.zero_();
  TORCH_CHECK(gradInput.is_contiguous(), "gradInput must be contiguous");

  TORCH_CHECK(
      input.defined() && (n_dims <= 2 && n_dims > 0),
      "input tensor should be 1D or 2D");

  int64_t batch_size = input.size(0);
  int64_t num_targets = target.size(0);
  int64_t target_stride = target.stride(0);

  TORCH_CHECK(
      batch_size == num_targets,
      "mismatch between the batch size of input and that of target")

  TORCH_CHECK(
      !weights.defined() || weights.numel() == input.size(-1),
      "weight tensor should be defined either for all or no classes");

  if (reduction == at::Reduction::None && n_dims == 2) {
    check_dim_size(gradOutput, 1, 0, batch_size);
    auto weights_cont = weights.defined() ? weights.contiguous() : weights;

    auto& queue = dpcppGetCurrentQueue();
    auto dev_id = dpcppGetDeviceIdOfCurrentQueue();
    int64_t local_size = dpcppMaxWorkGroupSize(dev_id);
    int64_t global_size =
        ((batch_size + local_size - 1) / local_size) * local_size;
    bool has_weights = weights.defined() ? true : false;
    DPCPP::buffer<uint8_t, 1> dummy_buffer(DPCPP::range<1>(1));

    auto gradInput_stride_0 = gradInput.stride(0);
    auto gradInput_stride_1 = gradInput.stride(1);
    auto gradOutput_stride_0 = gradOutput.stride(0);
    auto cgf = DPCPP_Q_CGF(cgh) {
      auto target_data = target.data_ptr<int64_t>();
      auto gradOutput_data = gradOutput.data_ptr<scalar_t>();
      auto weights_data = has_weights
          ? weights_cont.data_ptr<scalar_t>()
          : gradOutput_data; // Use gradOutput handler as dummy weights
      auto gradInput_data = gradInput.data_ptr<scalar_t>();
      auto kfn = DPCPP_Q_KFN(DPCPP::nd_item<1> item_id) {
        auto target_ptr = target_data;
        auto gradOutput_ptr = gradOutput_data;
        auto weights_ptr = has_weights ? weights_data : NULL;
        auto gradInput_ptr = gradInput_data;

        auto local_id = item_id.get_local_id(0);
        auto group_id = item_id.get_group(0);

        for (int i = group_id * local_size + local_id; i < batch_size;
             i += item_id.get_global_range(0)) {
          int cur_target = target_ptr[i * target_stride];
          if (cur_target == ignore_index) {
            continue;
          }
          scalar_t cur_weight = has_weights ? weights_ptr[cur_target]
                                            : static_cast<scalar_t>(1.0f);
          gradInput_ptr
              [i * gradInput_stride_0 + cur_target * gradInput_stride_1] =
                  -cur_weight *
              static_cast<scalar_t>(gradOutput_ptr[i * gradOutput_stride_0]);
        }
      };

      cgh.parallel_for(
          DPCPP::nd_range<1>(
              DPCPP::range<1>(global_size), DPCPP::range<1>(local_size)),
          kfn);
    };

    DPCPP_Q_SUBMIT(queue, cgf);
    return;
  }

  auto weights_cont = weights.defined() ? weights.contiguous() : weights;
  auto target_cont = target.contiguous();
  bool has_weights = weights.defined() ? true : false;

  TORCH_CHECK(
      gradOutput.dim() <= 1 && gradOutput.numel() == 1,
      "Expected a single element grad_output tensor, but got: ",
      gradOutput.sizes());

  auto& queue = dpcppGetCurrentQueue();
  if (input.dim() == 1) {
    auto cgf = DPCPP_Q_CGF(cgh) {
      auto gradOutput_data = gradOutput.data_ptr<scalar_t>();
      auto weights_data = has_weights
          ? weights_cont.data_ptr<scalar_t>()
          : gradOutput_data; // Use gradOutput handler as dummy weights
      auto gradInput_data = gradInput.data_ptr<scalar_t>();
      auto target_data = target_cont.data_ptr<int64_t>();
      auto total_weight_data = total_weight.data_ptr<scalar_t>();
      auto kfn = DPCPP_Q_KFN(DPCPP::item<1> item_id) {
        auto gradOutput_ptr = gradOutput_data;
        auto weights_ptr = has_weights ? weights_data : NULL;
        auto gradInput_ptr = gradInput_data;
        auto target_ptr = target_data;
        auto total_weight_ptr = total_weight_data;

        if (*total_weight_ptr <= 0)
          return;
        scalar_t norm = (reduction == at::Reduction::Mean)
            ? (static_cast<scalar_t>(1) /
               static_cast<scalar_t>(*total_weight_ptr))
            : static_cast<scalar_t>(1);
        int t = (int)*target_ptr;
        if (t != (int)ignore_index) {
          gradInput_ptr[t] =
              -(has_weights ? weights_ptr[t] : static_cast<scalar_t>(1)) *
              norm * gradOutput_ptr[0];
        }
      };
      cgh.parallel_for(DPCPP::range<1>(1), kfn);
    };
    DPCPP_Q_SUBMIT(queue, cgf);
  } else {
    int nframe = input.size(0);
    int ndim = input.size(1);
    int64_t local_size = 32;
    auto cgf = DPCPP_Q_CGF(cgh) {
      auto gradOutput_data = gradOutput.data_ptr<scalar_t>();
      auto weights_data = has_weights
          ? weights_cont.data_ptr<scalar_t>()
          : gradOutput_data; // use the gradOutput handler as dummy weights
      auto gradInput_data = gradInput.data_ptr<scalar_t>();
      auto target_data = target_cont.data_ptr<int64_t>();
      auto total_weight_data = total_weight.data_ptr<scalar_t>();
      auto kfn = DPCPP_Q_KFN(DPCPP::item<1> item_id) {
        auto gradOutput_ptr = gradOutput_data;
        auto weights_ptr = has_weights ? weights_data : NULL;
        auto gradInput_ptr = gradInput_data;
        auto target_ptr = target_data;
        auto total_weight_ptr = total_weight_data;

        auto local_item_id = item_id.get_id(0);

        if (*total_weight_ptr <= 0)
          return;
        int i, t;
        scalar_t norm = (reduction == at::Reduction::Mean)
            ? (static_cast<scalar_t>(1.0f) /
               static_cast<scalar_t>(*total_weight_ptr))
            : static_cast<scalar_t>(1);
        for (i = local_item_id; i < nframe; i += local_size) {
          t = (int)target_ptr[i];
          if (t != (int)ignore_index) {
            // assert(t >= 0 && t < n_classes)
            gradInput_ptr[i * ndim + t] =
                -(has_weights ? weights_ptr[t] : static_cast<scalar_t>(1)) *
                norm * gradOutput_ptr[0];
          }
        }
      };
      cgh.parallel_for(DPCPP::range<1>(local_size), kfn);
    };

    DPCPP_Q_SUBMIT(queue, cgf);
  }
}

void spatial_class_nll_criterion_shape_check(
    const Tensor& self,
    const Tensor& target,
    const Tensor& weights) {
  TORCH_CHECK(
      target.dim() == 3,
      1,
      "only batches of spatial targets supported (3D tensors)"
      " but got targets of size: : ",
      target.sizes());
  TORCH_CHECK(
      self.dim() == 4,
      2,
      "only batches of spatial inputs supported (4D tensors), "
      "but got input of size: ",
      self.sizes());
  TORCH_CHECK(
      self.size(0) == target.size(0) && self.size(2) == target.size(1) &&
          self.size(3) == target.size(2),
      "input and target batch or spatial sizes don't match: target ",
      target.sizes(),
      ", input ",
      self.sizes());
  if (weights.defined()) {
    TORCH_CHECK(
        weights.numel() == self.size(1),
        "weight tensor should be defined either for all or no classes");
  }
}

void spatial_class_nll_criterion_grad_output_no_reduce_shape_check(
    const Tensor& grad_output,
    const Tensor& target) {
  TORCH_CHECK(
      grad_output.dim() == 3,
      2,
      "grad_output must have same dimension as target (3) but got dimension: ",
      grad_output.sizes());
  TORCH_CHECK(
      grad_output.size(0) == target.size(0) &&
          grad_output.size(1) == target.size(1) &&
          grad_output.size(2) == target.size(2),
      "gradOutput sizes don't match target sizes: target ",
      target.sizes(),
      ", grad_output ",
      grad_output.sizes());
}

template <typename scalar_t>
void spatial_class_nll_criterion_update_output_no_reduce_kernel(
    const Tensor& self,
    const Tensor& target,
    Tensor& output,
    const Tensor& weight,
    int64_t ignore_index) {
  int64_t batch_size = self.size(0);
  int64_t H = self.size(2);
  int64_t W = self.size(3);
  int64_t count = batch_size * H * W;

  TensorInfo<scalar_t, uint64_t> self_info =
      getTensorInfo<scalar_t, uint64_t>(self);
  int dst_dim = self_info.collapseDims(1);
  self_info.reduceDim(dst_dim);
  TensorInfo<long, uint64_t> target_info =
      getTensorInfo<long, uint64_t>(target);
  target_info.collapseDims();
  TensorInfo<scalar_t, uint64_t> output_info =
      getTensorInfo<scalar_t, uint64_t>(output);
  output_info.collapseDims();
  TensorInfo<scalar_t, uint64_t> weight_info =
      getTensorInfo<scalar_t, uint64_t>(weight);
  weight_info.collapseDims();

  auto& dpcpp_queue = dpcppGetCurrentQueue();

  auto cgf = DPCPP_Q_CGF(cgh) {
    auto out_data = output.data_ptr<scalar_t>();
    auto self_data = self.data_ptr<scalar_t>();
    auto target_data = target.data_ptr<long>();
    auto weight_data = weight.data_ptr<scalar_t>();

    auto kfn = DPCPP_Q_KFN(DPCPP::item<1> item_id) {
      auto out_ptr = out_data;
      auto self_ptr = self_data;
      auto target_ptr = target_data;
      auto weight_ptr = weight_data;

      auto index = item_id.get_linear_id();
      auto target_offset =
          IndexToOffset<long, uint64_t>::get(index, target_info);
      auto output_offset =
          IndexToOffset<scalar_t, uint64_t>::get(index, output_info);

      int64_t cur_target = target_ptr[target_offset];
      if (cur_target == ignore_index) {
        out_ptr[output_offset] = ScalarConvert<int, scalar_t>::to(0);
      } else {
        auto self_offset =
            IndexToOffset<scalar_t, uint64_t>::get(index, self_info);
        auto weight_offset =
            IndexToOffset<scalar_t, uint64_t>::get(cur_target, weight_info);

        auto self_slice_ptr =
            self_ptr + cur_target * self_info.strides[dst_dim];
        scalar_t value = self_slice_ptr[self_offset];
        scalar_t weight = weight_ptr[weight_offset];
        out_ptr[output_offset] = -value * weight;
      }
    };

    cgh.parallel_for(DPCPP::range</*dim=*/1>(count), kfn);
  };

  DPCPP_Q_SUBMIT(dpcpp_queue, cgf);
}

template <typename scalar_t, typename accscalar_t>
void spatial_class_nll_criterion_update_output_kernel(
    Tensor& output,
    Tensor& total_weight,
    const Tensor& self,
    const Tensor& target,
    const Tensor& weight,
    int64_t reduction,
    int64_t ignore_index) {
  auto numel = target.numel();
  auto& dpcpp_queue = dpcppGetCurrentQueue();
  auto dev_id = dpcppGetDeviceIdOfCurrentQueue();
  auto wgroup_size = dpcppMaxWorkGroupSize(dev_id);
  auto cu_num = dpcppMaxComputeUnitSize(dev_id);
  auto num_groups = (numel - 1) / wgroup_size + 1;
  num_groups = std::min(decltype(num_groups)(cu_num), num_groups);

  TensorInfo<scalar_t, uint64_t> self_info =
      getTensorInfo<scalar_t, uint64_t>(self);
  int dst_dim = self_info.collapseDims(1);
  self_info.reduceDim(dst_dim);
  TensorInfo<long, uint64_t> target_info =
      getTensorInfo<long, uint64_t>(target);
  target_info.collapseDims();
  TensorInfo<scalar_t, uint64_t> weight_info =
      getTensorInfo<scalar_t, uint64_t>(weight);
  weight_info.collapseDims();

  auto cgf = DPCPP_Q_CGF(cgh) {
    auto out_data = output.data_ptr<scalar_t>();
    auto total_weight_data = total_weight.data_ptr<scalar_t>();
    auto self_data = self.data_ptr<scalar_t>();
    auto target_data = target.data_ptr<long>();
    auto weight_data = weight.data_ptr<scalar_t>();
    DPCPP::accessor<accscalar_t, 1, dpcpp_rw_mode, DPCPP::access::target::local>
        partial_sums(wgroup_size, cgh);
    DPCPP::accessor<accscalar_t, 1, dpcpp_rw_mode, DPCPP::access::target::local>
        partial_weight(wgroup_size, cgh);

    auto kfn = DPCPP_Q_KFN(DPCPP::nd_item<1> item_id) {
      auto out_ptr = out_data;
      auto total_weight_ptr = total_weight_data;
      auto self_ptr = self_data;
      auto target_ptr = target_data;
      auto weight_ptr = weight_data;

      auto local_idx = item_id.get_local_linear_id();
      auto global_idx = item_id.get_global_linear_id();
      auto global_range_size = item_id.get_global_range().size();
      auto num_combine = (numel + global_range_size - 1) / global_range_size;

      accscalar_t input_sum = 0;
      accscalar_t acc_weight = 0;
      for (uint64_t i = 0; i < num_combine; ++i) {
        auto global_shift = global_idx + i * global_range_size;
        if (global_shift < numel) {
          auto target_offset =
              IndexToOffset<long, uint64_t>::get(global_shift, target_info);
          int64_t cur_target = target_ptr[target_offset];

          if (cur_target != ignore_index) {
            auto weight_offset =
                IndexToOffset<scalar_t, uint64_t>::get(cur_target, weight_info);
            scalar_t weight = weight_ptr[weight_offset];

            auto self_offset =
                IndexToOffset<scalar_t, uint64_t>::get(global_shift, self_info);
            auto self_slice_ptr =
                self_ptr + cur_target * self_info.strides[dst_dim];
            scalar_t value = self_slice_ptr[self_offset];
            input_sum -= value * weight;
            acc_weight += weight;
          }
        }
      }
      partial_sums[local_idx] = input_sum;
      partial_weight[local_idx] = acc_weight;

      simple_reduce(item_id, partial_sums, [](accscalar_t a, accscalar_t b) {
        return Numerics<accscalar_t>::add(a, b);
      });

      simple_reduce(item_id, partial_weight, [](accscalar_t a, accscalar_t b) {
        return Numerics<accscalar_t>::add(a, b);
      });

      if (local_idx == 0) {
        atomicAdd(
            (dpcpp_global_ptr_pt<scalar_t>)total_weight_ptr,
            ScalarConvert<accscalar_t, scalar_t>::to(partial_weight[0]));
        atomicAdd(
            (dpcpp_global_ptr_pt<scalar_t>)out_ptr,
            ScalarConvert<accscalar_t, scalar_t>::to(partial_sums[0]));
      }
    };

    cgh.parallel_for(
        DPCPP::nd_range<1>(
            DPCPP::range<1>(num_groups * wgroup_size),
            DPCPP::range<1>(wgroup_size)),
        kfn);
  };

  DPCPP_Q_SUBMIT(dpcpp_queue, cgf);

  if (reduction == at::Reduction::Mean) {
    auto cgf = DPCPP_Q_CGF(cgh) {
      auto out_data = output.data_ptr<scalar_t>();
      auto total_weight_data = total_weight.data_ptr<scalar_t>();

      auto kfn = DPCPP_Q_KFN(DPCPP::item<1> item_id) {
        auto out_ptr = out_data;
        auto total_weight_ptr = total_weight_data;
        if (total_weight_ptr[0] != 0) {
          out_ptr[0] = Numerics<scalar_t>::div(out_ptr[0], total_weight_ptr[0]);
        }
      };

      cgh.parallel_for(DPCPP::range</*dim=*/1>(1), kfn);
    };

    DPCPP_Q_SUBMIT(dpcpp_queue, cgf);
  }
}

template <typename scalar_t>
void spatial_class_nll_criterion_update_grad_input_no_reduce_kernel(
    const Tensor& target,
    const Tensor& grad_output,
    Tensor grad_input,
    const Tensor& weight,
    int64_t ignore_index) {
  int64_t batch_size = target.size(0);
  int64_t H = target.size(1);
  int64_t W = target.size(2);
  int64_t count = batch_size * H * W;

  TensorInfo<scalar_t, uint64_t> grad_input_info =
      getTensorInfo<scalar_t, uint64_t>(grad_input);
  int dst_dim = grad_input_info.collapseDims(1);
  grad_input_info.reduceDim(dst_dim);
  TensorInfo<long, uint64_t> target_info =
      getTensorInfo<long, uint64_t>(target);
  target_info.collapseDims();
  TensorInfo<scalar_t, uint64_t> grad_output_info =
      getTensorInfo<scalar_t, uint64_t>(grad_output);
  grad_output_info.collapseDims();
  TensorInfo<scalar_t, uint64_t> weight_info =
      getTensorInfo<scalar_t, uint64_t>(weight);
  weight_info.collapseDims();

  auto& dpcpp_queue = dpcppGetCurrentQueue();

  auto cgf = DPCPP_Q_CGF(cgh) {
    auto grad_input_data = grad_input.data_ptr<scalar_t>();
    auto grad_output_data = grad_output.data_ptr<scalar_t>();
    auto target_data = target.data_ptr<long>();
    auto weight_data = weight.data_ptr<scalar_t>();

    auto kfn = DPCPP_Q_KFN(DPCPP::item<1> item_id) {
      auto grad_input_ptr = grad_input_data;
      auto grad_output_ptr = grad_output_data;
      auto target_ptr = target_data;
      auto weight_ptr = weight_data;

      auto index = item_id.get_linear_id();
      auto target_offset =
          IndexToOffset<long, uint64_t>::get(index, target_info);

      int64_t cur_target = target_ptr[target_offset];
      if (cur_target != ignore_index) {
        auto grad_output_offset =
            IndexToOffset<scalar_t, uint64_t>::get(index, grad_output_info);
        auto weight_offset =
            IndexToOffset<scalar_t, uint64_t>::get(cur_target, weight_info);

        scalar_t value = grad_output_ptr[grad_output_offset];
        scalar_t weight = weight_ptr[weight_offset];

        auto grad_input_offset =
            IndexToOffset<scalar_t, uint64_t>::get(index, grad_input_info);
        auto grad_input_slice_ptr =
            grad_input_ptr + cur_target * grad_input_info.strides[dst_dim];

        grad_input_slice_ptr[grad_input_offset] = -value * weight;
      }
    };

    cgh.parallel_for(DPCPP::range</*dim=*/1>(count), kfn);
  };

  DPCPP_Q_SUBMIT(dpcpp_queue, cgf);
}

template <typename scalar_t>
void spatial_class_nll_criterion_update_grad_input_kernel(
    Tensor& grad_input,
    const Tensor& grad_output,
    const Tensor& target,
    const Tensor& weight,
    const Tensor& total_weight,
    int64_t reduction,
    int64_t ignore_index) {
  int64_t batch_size = target.size(0);
  int64_t H = target.size(1);
  int64_t W = target.size(2);
  int64_t count = batch_size * H * W;

  TensorInfo<scalar_t, uint64_t> grad_input_info =
      getTensorInfo<scalar_t, uint64_t>(grad_input);
  int dst_dim = grad_input_info.collapseDims(1);
  grad_input_info.reduceDim(dst_dim);
  TensorInfo<long, uint64_t> target_info =
      getTensorInfo<long, uint64_t>(target);
  target_info.collapseDims();
  TensorInfo<scalar_t, uint64_t> weight_info =
      getTensorInfo<scalar_t, uint64_t>(weight);
  weight_info.collapseDims();

  auto& dpcpp_queue = dpcppGetCurrentQueue();

  auto cgf = DPCPP_Q_CGF(cgh) {
    auto grad_input_data = grad_input.data_ptr<scalar_t>();
    auto grad_output_data = grad_output.data_ptr<scalar_t>();
    auto target_data = target.data_ptr<long>();
    auto weight_data = weight.data_ptr<scalar_t>();
    auto total_weight_data = total_weight.data_ptr<scalar_t>();

    auto kfn = DPCPP_Q_KFN(DPCPP::item<1> item_id) {
      auto total_weight_ptr = total_weight_data;
      scalar_t total_weight = total_weight_ptr[0];
      if (total_weight <= 0)
        return;

      auto target_ptr = target_data;

      auto index = item_id.get_linear_id();
      auto target_offset =
          IndexToOffset<long, uint64_t>::get(index, target_info);

      int64_t cur_target = target_ptr[target_offset];
      if (cur_target != ignore_index) {
        auto grad_input_ptr = grad_input_data;
        auto grad_output_ptr = grad_output_data;
        auto weight_ptr = weight_data;
        auto weight_offset =
            IndexToOffset<scalar_t, uint64_t>::get(cur_target, weight_info);
        scalar_t weight = weight_ptr[weight_offset];

        scalar_t norm = (reduction == at::Reduction::Mean)
            ? (ScalarConvert<int, scalar_t>::to(1) / total_weight)
            : ScalarConvert<int, scalar_t>::to(1);

        auto grad_input_offset =
            IndexToOffset<scalar_t, uint64_t>::get(index, grad_input_info);
        auto grad_input_slice_ptr =
            grad_input_ptr + cur_target * grad_input_info.strides[dst_dim];

        grad_input_slice_ptr[grad_input_offset] =
            -weight * norm * grad_output_ptr[0];
      }
    };

    cgh.parallel_for(DPCPP::range</*dim=*/1>(count), kfn);
  };

  DPCPP_Q_SUBMIT(dpcpp_queue, cgf);
}

} // namespace impl

// namespace AtenIpexTypeXPU
std::tuple<Tensor&, Tensor&> nll_loss_forward_out(
    Tensor& output,
    Tensor& total_weight,
    const Tensor& self,
    const Tensor& target,
    const Tensor& weight,
    int64_t reduction,
    int64_t ignore_index) {
  IPEX_DISPATCH_ALL_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      self.scalar_type(),
      "ClassNLLCriterion_updateOutput",
      [&]() {
        impl::ClassNLLCriterion_updateOutput<scalar_t>(
            self,
            target,
            output,
            weight,
            total_weight,
            reduction,
            ignore_index);
      });

  return std::tuple<Tensor&, Tensor&>(output, total_weight);
}

std::tuple<at::Tensor, at::Tensor> nll_loss_forward(
    const Tensor& self,
    const Tensor& target,
    const Tensor& weight,
    int64_t reduction,
    int64_t ignore_index) {
  auto output = at::empty({0}, self.options());
  auto total_weight = at::empty({0}, self.options());

  IPEX_DISPATCH_ALL_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      self.scalar_type(),
      "ClassNLLCriterion_updateOutput",
      [&]() {
        impl::ClassNLLCriterion_updateOutput<scalar_t>(
            self,
            target,
            output,
            weight,
            total_weight,
            reduction,
            ignore_index);
      });

  return std::tuple<Tensor&, Tensor&>(output, total_weight);
}

Tensor& nll_loss_backward_out(
    Tensor& grad_input,
    const Tensor& grad_output,
    const Tensor& self,
    const Tensor& target,
    const Tensor& weight,
    int64_t reduction,
    int64_t ignore_index,
    const Tensor& total_weight) {
  IPEX_DISPATCH_ALL_TYPES_AND(
      at::ScalarType::BFloat16,
      self.scalar_type(),
      "ClassNLLCriterion_updateGradInput",
      [&]() {
        impl::ClassNLLCriterion_updateGradInput<scalar_t>(
            self,
            target,
            grad_output,
            grad_input,
            reduction,
            weight,
            total_weight,
            ignore_index);
      });
  return grad_input;
}

Tensor nll_loss_backward(
    const Tensor& grad_output,
    const Tensor& self,
    const Tensor& target,
    const Tensor& weight,
    int64_t reduction,
    int64_t ignore_index,
    const Tensor& total_weight) {
  auto grad_input = at::zeros_like(self, c10::MemoryFormat::Contiguous);

  IPEX_DISPATCH_ALL_TYPES_AND(
      at::ScalarType::BFloat16,
      self.scalar_type(),
      "ClassNLLCriterion_updateGradInput",
      [&]() {
        impl::ClassNLLCriterion_updateGradInput<scalar_t>(
            self,
            target,
            grad_output,
            grad_input,
            reduction,
            weight,
            total_weight,
            ignore_index);
      });
  return grad_input;
}

Tensor& nll_loss2d_backward_out(
    Tensor& grad_input,
    const Tensor& grad_output,
    const Tensor& self,
    const Tensor& target,
    const Tensor& weight,
    int64_t reduction,
    int64_t ignore_index,
    const Tensor& total_weight) {
  impl::spatial_class_nll_criterion_shape_check(self, target, weight);
  grad_input.resize_(self.sizes()).fill_(0);

  if (weight.defined()) {
    TORCH_CHECK(
        IsOnSameDevice({self, target, weight, grad_input, total_weight}),
        "Some of weight/gradient/input tensors are located on different GPUs. Please move them to a single one.");
  } else {
    TORCH_CHECK(
        IsOnSameDevice({self, target, grad_input, total_weight}),
        "Some of weight/gradient/input tensors are located on different GPUs. Please move them to a single one.");
  }

  int64_t batch_size = self.size(0);
  int64_t H = self.size(2);
  int64_t W = self.size(3);
  int64_t count = batch_size * H * W;

  if (count == 0) {
    // This guards from unnecessary operations and launching kernel with 0
    // blocks.
    return grad_input;
  }

  if (reduction == at::Reduction::None) {
    impl::spatial_class_nll_criterion_grad_output_no_reduce_shape_check(
        grad_output, target);

    IPEX_DISPATCH_FLOATING_TYPES_AND(
        at::ScalarType::BFloat16,
        self.scalar_type(),
        "nll_loss2d_dpcpp_backward",
        [&]() {
          impl::spatial_class_nll_criterion_update_grad_input_no_reduce_kernel<
              scalar_t>(
              target,
              grad_output,
              grad_input,
              weight.defined() ? weight
                               : at::ones({self.size(1)}, self.options()),
              ignore_index);
        });
  } else {
    IPEX_DISPATCH_FLOATING_TYPES_AND(
        at::ScalarType::BFloat16,
        self.scalar_type(),
        "nll_loss2d_dpcpp_backward",
        [&]() {
          impl::spatial_class_nll_criterion_update_grad_input_kernel<scalar_t>(
              grad_input,
              grad_output,
              target,
              weight.defined() ? weight
                               : at::ones({self.size(1)}, self.options()),
              total_weight,
              reduction,
              ignore_index);
        });
  }

  return grad_input;
}

Tensor nll_loss2d_backward(
    const Tensor& grad_output,
    const Tensor& self,
    const Tensor& target,
    const Tensor& weight,
    int64_t reduction,
    int64_t ignore_index,
    const Tensor& total_weight) {
  Tensor grad_input = at::empty({0}, grad_output.options());
  at::AtenIpexTypeXPU::nll_loss2d_backward_out(
      grad_input,
      grad_output,
      self,
      target,
      weight,
      reduction,
      ignore_index,
      total_weight);
  return grad_input;
}

std::tuple<Tensor&, Tensor&> nll_loss2d_forward_out(
    Tensor& output,
    Tensor& total_weight,
    const Tensor& self,
    const Tensor& target,
    const Tensor& weight,
    int64_t reduction,
    int64_t ignore_index) {
  impl::spatial_class_nll_criterion_shape_check(self, target, weight);
  output.resize_({});
  total_weight.resize_({});

  if (weight.defined()) {
    TORCH_CHECK(
        IsOnSameDevice({self, target, weight, output, total_weight}),
        "Some of weight/gradient/input tensors are located on different GPUs. Please move them to a single one.");
  } else {
    TORCH_CHECK(
        IsOnSameDevice({self, target, output, total_weight}),
        "Some of weight/gradient/input tensors are located on different GPUs. Please move them to a single one.");
  }

  int64_t batch_size = self.size(0);
  int64_t H = self.size(2);
  int64_t W = self.size(3);
  int64_t count = batch_size * H * W;

  if (count != 0) {
    // This guards from unnecessary operations and launching kernel with 0
    // blocks.
    if (reduction == at::Reduction::None) {
      output.resize_({batch_size, H, W});

      IPEX_DISPATCH_FLOATING_TYPES_AND(
          at::ScalarType::BFloat16,
          self.scalar_type(),
          "nll_loss2d_dpcpp_forward",
          [&]() {
            impl::spatial_class_nll_criterion_update_output_no_reduce_kernel<
                scalar_t>(
                self,
                target,
                output,
                weight.defined() ? weight
                                 : at::ones({self.size(1)}, self.options()),
                ignore_index);
          });
    } else {
      output.fill_(0);
      total_weight.fill_(0);
      IPEX_DISPATCH_FLOATING_TYPES_AND(
          at::ScalarType::BFloat16,
          self.scalar_type(),
          "nll_loss2d_dpcpp_forward",
          [&]() {
            using accscalar_t = acc_type<scalar_t>;
            impl::spatial_class_nll_criterion_update_output_kernel<
                scalar_t,
                accscalar_t>(
                output,
                total_weight,
                self,
                target,
                weight.defined() ? weight
                                 : at::ones({self.size(1)}, self.options()),
                reduction,
                ignore_index);
          });
    }
  }

  return std::tuple<Tensor&, Tensor&>{output, total_weight};
}

std::tuple<Tensor, Tensor> nll_loss2d_forward(
    const Tensor& self,
    const Tensor& target,
    const Tensor& weight,
    int64_t reduction,
    int64_t ignore_index) {
  Tensor output = at::empty({0}, self.options());
  Tensor total_weight = at::empty({0}, self.options());
  at::AtenIpexTypeXPU::nll_loss2d_forward_out(
      output, total_weight, self, target, weight, reduction, ignore_index);
  return std::tuple<Tensor, Tensor>{output, total_weight};
}

} // namespace AtenIpexTypeXPU
} // namespace at
