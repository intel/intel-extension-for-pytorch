#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/Functions.h>
#include <ATen/TensorUtils.h>
#include <ATen/core/Reduction.h>

#include <core/DPCPP.h>
#include <core/DPCPPUtils.h>
#include <core/Memory.h>
#include <core/TensorImplUtils.h>

#include <ATen/aten_ipex_type_dpcpp.h>

DPCPP_DEF_K2(updateOutputName, typename scalar_t);
DPCPP_DEF_K2(updateOutputKernel1Name, typename scalar_t);
DPCPP_DEF_K2(updateOutputKernelName, typename scalar_t);

DPCPP_DEF_K2(updateGradInputName, typename scalar_t);
DPCPP_DEF_K2(updateGradInputKernel1Name, typename scalar_t);
DPCPP_DEF_K2(updateGradInputKernelName, typename scalar_t);

using namespace at::dpcpp;

namespace at {
namespace AtenIpexTypeDPCPP {
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

    auto queue = dpcppGetCurrentQueue();
    int64_t local_size =
        queue.get_device()
            .template get_info<DPCPP::info::device::max_work_group_size>();
    bool has_weights = weights.defined()
        ? true
        : false; // dpcpp kernel can not accept host pointer
    DPCPP::buffer<uint8_t, 1> dummy_buffer(DPCPP::range<1>(1));

    auto output_stride_0 = output.stride(0);
    auto input_stride_0 = input.stride(0);
    auto input_stride_1 = input.stride(1);
    auto cgf = DPCPP_Q_CGF(cgh) {
      auto input_acc =
          DPCPPAccessor<dpcpp_r_mode>(cgh, input.data_ptr<scalar_t>());
      auto target_acc =
          DPCPPAccessor<dpcpp_r_mode>(cgh, target.data_ptr<int64_t>());
      auto weights_acc = has_weights
          ? DPCPPAccessor<dpcpp_r_mode>(cgh, weights_cont.data_ptr<scalar_t>())
          : DPCPPAccessor<dpcpp_r_mode>(cgh, dummy_buffer); // dummy weights
      auto output_acc =
          DPCPPAccessor<dpcpp_w_mode>(cgh, output.data_ptr<scalar_t>());
      auto kfn = DPCPP_Q_KFN(DPCPP::item<1> item_id) {
        auto input_ptr = input_acc.template get_pointer<scalar_t>();
        auto target_ptr = target_acc.template get_pointer<int64_t>();
        auto weights_ptr =
            has_weights ? weights_acc.template get_pointer<scalar_t>() : NULL;
        auto output_ptr = output_acc.template get_pointer<scalar_t>();
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
              -input_ptr[i * input_stride_0 + cur_target * input_stride_1] *
              cur_weight;
        }
      };

      cgh.parallel_for<DPCPP_K(updateOutputName, scalar_t)>(
          DPCPP::range<1>(local_size), kfn);
    };

    DPCPP_Q_ASYNC_SUBMIT(queue, cgf);
    return;
  }

  output.resize_({});
  total_weight.resize_({});

  auto input_cont = input.contiguous();
  auto weights_cont = weights.defined() ? weights.contiguous() : weights;
  auto target_cont = target.contiguous();

  scalar_t* input_data = input_cont.data_ptr<scalar_t>();
  scalar_t* weights_data =
      weights.defined() ? weights_cont.data_ptr<scalar_t>() : NULL;
  int64_t* target_data = target_cont.data_ptr<int64_t>();
  scalar_t* output_data = output.data_ptr<scalar_t>();
  scalar_t* total_weight_data = total_weight.data_ptr<scalar_t>();
  bool has_weights = weights_data != NULL ? true : false;
  auto queue = dpcppGetCurrentQueue();

  if (input_cont.dim() == 1 || input_cont.dim() == 0) {
    int64_t local_size = 1;
    DPCPP::buffer<uint8_t, 1> dummy_buffer(DPCPP::range<1>(1));

    auto cgf = DPCPP_Q_CGF(cgh) {
      auto input_acc = DPCPPAccessor<dpcpp_r_mode>(cgh, input_data);
      auto weights_acc = has_weights
          ? DPCPPAccessor<dpcpp_r_mode>(cgh, weights_data)
          : DPCPPAccessor<dpcpp_r_mode>(cgh, dummy_buffer); // dummy weights
      auto target_acc = DPCPPAccessor<dpcpp_r_mode>(cgh, target_data);
      auto total_weight_acc =
          DPCPPAccessor<dpcpp_w_mode>(cgh, total_weight_data);
      auto output_acc = DPCPPAccessor<dpcpp_w_mode>(cgh, output_data);
      auto kfn = DPCPP_Q_KFN(DPCPP::item<1> item_id) {
        auto input_ptr = input_acc.template get_pointer<scalar_t>();
        auto target_ptr = target_acc.template get_pointer<int64_t>();
        auto weights_ptr =
            has_weights ? weights_acc.template get_pointer<scalar_t>() : NULL;
        auto total_weight_ptr =
            total_weight_acc.template get_pointer<scalar_t>();
        auto output_ptr = output_acc.template get_pointer<scalar_t>();
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
      cgh.parallel_for<DPCPP_K(updateOutputKernel1Name, scalar_t)>(
          DPCPP::range<1>(local_size), kfn);
    };

    DPCPP_Q_ASYNC_SUBMIT(queue, cgf);
  } else if (input.dim() == 2) {
    int64_t batch_size = input.size(0);
    int n_target = input.size(1);
    int64_t local_size =
        queue.get_device()
            .template get_info<DPCPP::info::device::max_work_group_size>();
    DPCPP::buffer<uint8_t, 1> dummy_buffer(DPCPP::range<1>(1));

    auto cgf = DPCPP_Q_CGF(cgh) {
      auto input_acc = DPCPPAccessor<dpcpp_r_mode>(cgh, input_data);
      auto weights_acc = has_weights
          ? DPCPPAccessor<dpcpp_r_mode>(cgh, weights_data)
          : DPCPPAccessor<dpcpp_r_mode>(cgh, dummy_buffer); // Dummy weight
      auto target_acc = DPCPPAccessor<dpcpp_r_mode>(cgh, target_data);
      auto total_weight_acc =
          DPCPPAccessor<dpcpp_r_mode>(cgh, total_weight_data);
      auto output_acc = DPCPPAccessor<dpcpp_r_mode>(cgh, output_data);
      auto local_output_acc = dpcpp_local_acc_t<scalar_t>(local_size, cgh);
      auto local_total_weight_acc =
          dpcpp_local_acc_t<scalar_t>(local_size, cgh);

      auto kfn = DPCPP_Q_KFN(DPCPP::nd_item<1> item_id) {
        auto input_ptr = input_acc.template get_pointer<scalar_t>();
        auto target_ptr = target_acc.template get_pointer<int64_t>();
        auto weights_ptr =
            has_weights ? weights_acc.template get_pointer<scalar_t>() : NULL;
        auto total_weight_ptr =
            total_weight_acc.template get_pointer<scalar_t>();
        auto output_ptr = output_acc.template get_pointer<scalar_t>();
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
      cgh.parallel_for<DPCPP_K(updateOutputKernelName, scalar_t)>(
          DPCPP::nd_range<1>(DPCPP::range<1>(local_size), DPCPP::range<1>(local_size)), kfn);
    };

    DPCPP_Q_ASYNC_SUBMIT(queue, cgf);
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

    auto queue = dpcppGetCurrentQueue();
    int64_t local_size =
        queue.get_device()
            .template get_info<DPCPP::info::device::max_work_group_size>();
    int64_t global_size =
        ((batch_size + local_size - 1) / local_size) * local_size;
    bool has_weights = weights.defined() ? true : false;
    DPCPP::buffer<uint8_t, 1> dummy_buffer(DPCPP::range<1>(1));

    auto gradInput_stride_0 = gradInput.stride(0);
    auto gradInput_stride_1 = gradInput.stride(1);
    auto gradOutput_stride_0 = gradOutput.stride(0);
    auto cgf = DPCPP_Q_CGF(cgh) {
      auto target_acc =
          DPCPPAccessor<dpcpp_r_mode>(cgh, target.data_ptr<int64_t>());
      auto gradOutput_acc =
          DPCPPAccessor<dpcpp_r_mode>(cgh, gradOutput.data_ptr<scalar_t>());
      auto weights_acc = has_weights
          ? DPCPPAccessor<dpcpp_r_mode>(cgh, weights_cont.data_ptr<scalar_t>())
          : DPCPPAccessor<dpcpp_r_mode>(cgh, dummy_buffer); // dummy weights
      auto gradInput_acc =
          DPCPPAccessor<dpcpp_w_mode>(cgh, gradInput.data_ptr<scalar_t>());
      auto kfn = DPCPP_Q_KFN(DPCPP::nd_item<1> item_id) {
        auto target_ptr = target_acc.template get_pointer<int64_t>();
        auto gradOutput_ptr = gradOutput_acc.template get_pointer<scalar_t>();
        auto weights_ptr =
            has_weights ? weights_acc.template get_pointer<scalar_t>() : NULL;
        auto gradInput_ptr = gradInput_acc.template get_pointer<scalar_t>();

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
                  -cur_weight * gradOutput_ptr[i * gradOutput_stride_0];
        }
      };

      cgh.parallel_for<DPCPP_K(updateGradInputName, scalar_t)>(
          DPCPP::nd_range<1>(
              DPCPP::range<1>(global_size), DPCPP::range<1>(local_size)),
          kfn);
    };

    DPCPP_Q_ASYNC_SUBMIT(queue, cgf);
    return;
  }

  auto weights_cont = weights.defined() ? weights.contiguous() : weights;
  auto target_cont = target.contiguous();
  bool has_weights = weights.defined() ? true : false;

  TORCH_CHECK(
      gradOutput.dim() <= 1 && gradOutput.numel() == 1,
      "Expected a single element grad_output tensor, but got: ",
      gradOutput.sizes());

  auto queue = dpcppGetCurrentQueue();
  if (input.dim() == 1) {
    DPCPP::buffer<uint8_t, 1> dummy_buffer(DPCPP::range<1>(1));

    auto cgf = DPCPP_Q_CGF(cgh) {
      auto gradOutput_acc =
          DPCPPAccessor<dpcpp_r_mode>(cgh, gradOutput.data_ptr<scalar_t>());
      auto weights_acc = has_weights
          ? DPCPPAccessor<dpcpp_r_mode>(cgh, weights_cont.data_ptr<scalar_t>())
          : DPCPPAccessor<dpcpp_r_mode>(cgh, dummy_buffer); // dummy weights
      auto gradInput_acc =
          DPCPPAccessor<dpcpp_w_mode>(cgh, gradInput.data_ptr<scalar_t>());
      auto target_acc =
          DPCPPAccessor<dpcpp_r_mode>(cgh, target_cont.data_ptr<int64_t>());
      auto total_weight_acc =
          DPCPPAccessor<dpcpp_r_mode>(cgh, total_weight.data_ptr<scalar_t>());
      auto kfn = DPCPP_Q_KFN(DPCPP::item<1> item_id) {
        auto gradOutput_ptr = gradOutput_acc.template get_pointer<scalar_t>();
        auto weights_ptr =
            has_weights ? weights_acc.template get_pointer<scalar_t>() : NULL;
        auto gradInput_ptr = gradInput_acc.template get_pointer<scalar_t>();
        auto target_ptr = target_acc.template get_pointer<int64_t>();
        auto total_weight_ptr =
            total_weight_acc.template get_pointer<scalar_t>();

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
      cgh.parallel_for<DPCPP_K(updateGradInputKernel1Name, scalar_t)>(
          DPCPP::range<1>(1), kfn);
    };
    DPCPP_Q_ASYNC_SUBMIT(queue, cgf);
  } else {
    int nframe = input.size(0);
    int ndim = input.size(1);
    int64_t local_size = 32;
    DPCPP::buffer<uint8_t, 1> dummy_buffer(DPCPP::range<1>(1));
    auto cgf = DPCPP_Q_CGF(cgh) {
      auto gradOutput_acc =
          DPCPPAccessor<dpcpp_r_mode>(cgh, gradOutput.data_ptr<scalar_t>());
      auto weights_acc = has_weights
          ? DPCPPAccessor<dpcpp_r_mode>(cgh, weights_cont.data_ptr<scalar_t>())
          : DPCPPAccessor<dpcpp_r_mode>(cgh, dummy_buffer); // dummy weights
      auto gradInput_acc =
          DPCPPAccessor<dpcpp_w_mode>(cgh, gradInput.data_ptr<scalar_t>());
      auto target_acc =
          DPCPPAccessor<dpcpp_r_mode>(cgh, target_cont.data_ptr<int64_t>());
      auto total_weight_acc =
          DPCPPAccessor<dpcpp_r_mode>(cgh, total_weight.data_ptr<scalar_t>());
      auto kfn = DPCPP_Q_KFN(DPCPP::item<1> item_id) {
        auto gradOutput_ptr = gradOutput_acc.template get_pointer<scalar_t>();
        auto weights_ptr =
            has_weights ? weights_acc.template get_pointer<scalar_t>() : NULL;
        auto gradInput_ptr = gradInput_acc.template get_pointer<scalar_t>();
        auto target_ptr = target_acc.template get_pointer<int64_t>();
        auto total_weight_ptr =
            total_weight_acc.template get_pointer<scalar_t>();

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
      cgh.parallel_for<DPCPP_K(updateGradInputKernelName, scalar_t)>(
          DPCPP::range<1>(local_size), kfn);
    };

    DPCPP_Q_ASYNC_SUBMIT(queue, cgf);
  }
}

} // namespace impl

// namespace AtenIpexTypeDPCPP
std::tuple<Tensor&, Tensor&> nll_loss_forward_out(
    Tensor& output,
    Tensor& total_weight,
    const Tensor& self,
    const Tensor& target,
    const Tensor& weight,
    int64_t reduction,
    int64_t ignore_index) {
  AT_DISPATCH_ALL_TYPES(
      self.scalar_type(), "ClassNLLCriterion_updateOutput", [&]() {
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

  AT_DISPATCH_ALL_TYPES(
      self.scalar_type(), "ClassNLLCriterion_updateOutput", [&]() {
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
  AT_DISPATCH_ALL_TYPES(
      self.scalar_type(), "ClassNLLCriterion_updateGradInput", [&]() {
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

  AT_DISPATCH_ALL_TYPES(
      self.scalar_type(), "ClassNLLCriterion_updateGradInput", [&]() {
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

} // namespace AtenIpexTypeDPCPP
} // namespace at
