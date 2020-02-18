#include <ATen/core/Reduction.h>
#include <ATen/Functions.h>
#include <ATen/Dispatch.h>

#include <core/SYCL.h>
#include <core/SYCLMemory.h>
#include <core/SYCLUtils.h>
#include <core/TensorImplUtils.h>

#include <ATen/aten_ipex_type_dpcpp.h>


DP_DEF_K2(updateOutputName, typename scalar_t);
DP_DEF_K2(updateOutputKernel1Name, typename scalar_t);
DP_DEF_K2(updateOutputKernelName, typename scalar_t);

// TODO:
#if ClassNLLCriterion_updateGradInput_PORTED
class updateGradInput_no_reduce_KernelName;
class updateGradInput_kernel1Name {};
class updateGradInput_kernelName {};
#endif

using namespace at::native;
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
  if (target.dim() > 1) {
    TORCH_CHECK(false, "multi-target not supported");
  }

  int n_dims = input.dim() == 0 ? 1 : input.dim();
  int n_classes =
      TensorImpl_sizeLegacyNoScalars(TensorImpl_Unwrap(input), n_dims - 1);
  ignore_index -= 0;

  TORCH_CHECK(input.defined() && (n_dims <= 2 && n_dims > 0),
      "non-empty vector or matrix expected");

  int64_t batch_size = n_dims == 1 ? 1 :
      TensorImpl_sizeLegacyNoScalars(TensorImpl_Unwrap(input), 0);
  int64_t num_targets =
      TensorImpl_sizeLegacyNoScalars(TensorImpl_Unwrap(target), 0);
  int64_t target_stride = target.dim() == 0 ? 1 : target.stride(0);

  TORCH_CHECK(batch_size == num_targets,
      "mismatch between the batch size of input (%ld) and that of target (%ld)",
      batch_size, num_targets);

  if (weights.defined() && weights.numel() != n_classes)
    TORCH_CHECK(false, "weight tensor should be defined either for all %d classes or no classes"
        " but got incorrect weight tensor of shape", n_classes);

  if (reduction == at::Reduction::None && n_dims == 2) {
    at::AtenIpexTypeDPCPP::resize_(
        output, {batch_size}, c10::MemoryFormat::Contiguous);

    auto weights_cont = weights.defined() ? weights.contiguous() : weights;

    auto queue = c10::sycl::syclGetCurrentQueue();
    int64_t local_size = queue.get_device(). template get_info<DP::info::device::max_work_group_size>();
    bool has_weights = weights.defined() ? true : false; //sycl kernel can not accept host pointer
    DP::buffer<uint8_t, 1> dummy_buffer(DP::range<1>(1));

    auto output_stride_0 = output.stride(0);
    auto input_stride_0 = input.stride(0);
    auto input_stride_1 = input.stride(1);
    auto cgf = DP_Q_CGF(cgh) {
      auto input_acc = c10::sycl::SYCLAccessor<dp_r_mode>(cgh, input.data<scalar_t>());
      auto target_acc = c10::sycl::SYCLAccessor<dp_r_mode>(cgh, target.data<int64_t>());
      auto weights_acc = has_weights ? c10::sycl::SYCLAccessor<dp_r_mode>(cgh, weights_cont.data<scalar_t>()) :
                                       c10::sycl::SYCLAccessor<dp_r_mode>(cgh, dummy_buffer); // dummy weights
      auto output_acc = c10::sycl::SYCLAccessor<dp_w_mode>(cgh, output.data<scalar_t>());
      auto kfn = DP_Q_KFN(DP::item<1> item_id) {
        auto input_ptr = input_acc.template get_pointer<scalar_t>();
        auto target_ptr = target_acc.template get_pointer<int64_t>();
        auto weights_ptr = has_weights ? weights_acc.template get_pointer<scalar_t>() : NULL;
        auto output_ptr = output_acc.template get_pointer<scalar_t>();
        auto local_item_id = item_id.get_id(0);
        for (int i = local_item_id; i < batch_size; i += local_size) {
          int cur_target = target_ptr[i * target_stride];
          if (cur_target >= 0 && cur_target < n_classes)
            if (cur_target == ignore_index) {
              output_ptr[i * output_stride_0] = 0.0f;
              continue;
            }
          scalar_t cur_weight = has_weights ? weights_ptr[cur_target] : static_cast<scalar_t>(1.0f);
          output_ptr[i * output_stride_0] =
              -input_ptr[i * input_stride_0 + cur_target * input_stride_1] * cur_weight;
        }
      };

      cgh.parallel_for<DP_K(updateOutputName, scalar_t)>(DP::range<1>(local_size), kfn);
    };

    DP_Q_ASYNC_SUBMIT(queue, cgf);
    return;
  }

  at::AtenIpexTypeDPCPP::resize_(output, {1}, c10::MemoryFormat::Contiguous);
  at::AtenIpexTypeDPCPP::resize_(total_weight, {1}, c10::MemoryFormat::Contiguous);

  auto input_cont = input.contiguous();
  auto weights_cont = weights.defined() ? weights.contiguous() : weights;
  auto target_cont = target.contiguous();

  scalar_t *input_data = input_cont.data<scalar_t>();
  scalar_t *weights_data = weights.defined() ? weights_cont.data<scalar_t>() : NULL;
  int64_t  *target_data = target_cont.data<int64_t>();
  scalar_t *output_data = output.data<scalar_t>();
  scalar_t *total_weight_data = total_weight.data<scalar_t>();
  bool has_weights = weights_data != NULL ? true : false;
  auto queue = c10::sycl::syclGetCurrentQueue();

  if (input_cont.dim() == 1 || input_cont.dim() == 0) {
    int64_t local_size = 1;
    DP::buffer<uint8_t, 1> dummy_buffer(DP::range<1>(1));

    auto cgf = DP_Q_CGF(cgh) {
      auto input_acc = c10::sycl::SYCLAccessor<dp_r_mode>(cgh, input_data);
      auto weights_acc = has_weights ? c10::sycl::SYCLAccessor<dp_r_mode>(cgh, weights_data) :
          c10::sycl::SYCLAccessor<dp_r_mode>(cgh, dummy_buffer); // dummy weights
      auto target_acc = c10::sycl::SYCLAccessor<dp_r_mode>(cgh, target_data);
      auto total_weight_acc = c10::sycl::SYCLAccessor<dp_w_mode>(cgh, total_weight_data);
      auto output_acc = c10::sycl::SYCLAccessor<dp_w_mode>(cgh, output_data);
      auto kfn = DP_Q_KFN(DP::item<1> item_id) {
        auto input_ptr = input_acc.template get_pointer<scalar_t>();
        auto target_ptr = target_acc.template get_pointer<int64_t>();
        auto weights_ptr = has_weights ? weights_acc.template get_pointer<scalar_t>() : NULL;
            auto total_weight_ptr = total_weight_acc.template get_pointer<scalar_t>();
            auto output_ptr = output_acc.template get_pointer<scalar_t>();
        // auto local_item_id = item_id.get_id(0);
        int cur_target = target_ptr[0];
            if (cur_target != ignore_index) {
        total_weight_ptr[0] = has_weights ? weights_ptr[cur_target] : static_cast<scalar_t>(1.0f);
        output_ptr[0] = -static_cast<scalar_t>(input_ptr[cur_target]) * static_cast<scalar_t>(total_weight_ptr[0]);
        }
        if (reduction == at::Reduction::Mean && total_weight_ptr[0]) {
          output_ptr[0] /= total_weight_ptr[0];
        }
      };
      cgh.parallel_for<DP_K(updateOutputKernel1Name, scalar_t)>(
          cl::sycl::range<1>(local_size), kfn);
    };

    DP_Q_ASYNC_SUBMIT(queue, cgf);
  } else if (input.dim() == 2) {
    int batch_size = input.size(0);
    int n_target = input.size(1);
    size_t local_size = queue.get_device().template get_info<DP::info::device::max_work_group_size>();
    DP::buffer<uint8_t, 1> dummy_buffer(DP::range<1>(1));

    auto cgf = DP_Q_CGF(cgh) {
      auto input_acc = c10::sycl::SYCLAccessor<dp_r_mode>(cgh, input_data);
      auto weights_acc = has_weights ? c10::sycl::SYCLAccessor<dp_r_mode>(cgh, weights_data)
                                     : c10::sycl::SYCLAccessor<dp_r_mode>(cgh, dummy_buffer); // Dummy weight
      auto target_acc = c10::sycl::SYCLAccessor<dp_r_mode>(cgh, target_data);
      auto total_weight_acc = c10::sycl::SYCLAccessor<dp_r_mode>(cgh, total_weight_data);
      auto output_acc = c10::sycl::SYCLAccessor<dp_r_mode>(cgh, output_data);
      auto local_output_acc = dp_local_acc_t<scalar_t>(local_size, cgh);
      auto local_total_weight_acc = dp_local_acc_t<scalar_t>(local_size, cgh);

      auto kfn = DP_Q_KFN(DP::item<1> item_id) {
        auto input_ptr = input_acc.template get_pointer<scalar_t>();
        auto target_ptr = target_acc.template get_pointer<int64_t>();
        auto weights_ptr = has_weights ? weights_acc.template get_pointer<scalar_t>() : NULL;
        auto total_weight_ptr = total_weight_acc.template get_pointer<scalar_t>();
        auto output_ptr = output_acc.template get_pointer<scalar_t>();
        int64_t local_id = item_id.get_id(0);
        local_output_acc[local_id] = 0.0;
        local_total_weight_acc[local_id] = 0.0;
        for (int i = local_id; i < batch_size; i += local_size) {
          int cur_target = target_ptr[i];
          if (cur_target != ignore_index) {
            scalar_t cur_weight = has_weights ? weights_ptr[cur_target] : static_cast<scalar_t>(1.0f);
            local_total_weight_acc[local_id] += cur_weight;
            local_output_acc[local_id] -=
                static_cast<scalar_t>(input_ptr[i * n_target + cur_target]) *
                                      static_cast<scalar_t>(cur_weight);
          }
        }

        // reduce
        for (int64_t i = (local_size >> 1); i > 0; i >>= 1) {
          if (local_id < i) {
            local_total_weight_acc[local_id] += local_total_weight_acc[local_id + i];
            local_output_acc[local_id] += local_output_acc[local_id + i];
          }
        }

        output_ptr[0] = local_output_acc[0];
        total_weight_ptr[0] = local_total_weight_acc[0];
        if (reduction == at::Reduction::Mean && total_weight_ptr[0]) {
          output_ptr[0] /= total_weight_ptr[0];
        }
      };
      cgh.parallel_for<DP_K(updateOutputKernelName, scalar_t)>(
          cl::sycl::range<1>{local_size}, kfn);
    };

    DP_Q_ASYNC_SUBMIT(queue, cgf);
  }
}

// TODO:
#if ClassNLLCriterion_updateGradInput_PORTED
void ClassNLLCriterion_updateGradInput(
    THSYCLState *state,
    THSYCLTensor *input,
    THSYCLIndexTensor *target,
    THSYCLTensor *gradOutput,
    THSYCLTensor *gradInput,
    int64_t reduction,
    THSYCLTensor *weights,       // [OPTIONAL]
    THSYCLTensor *total_weight,
    int64_t ignore_index) {
  static const auto write_mode = cl::sycl::access::mode::discard_write;
  static const auto dp_r_mode = cl::sycl::access::mode::read;
  if (THSYCLIndexTensor_(nDimensionLegacyNoScalars)(state, target) > 1) {
    THError("multi-target not supported");
  } 

  int n_dims = THSYCLTensor_(nDimensionLegacyNoScalars)(state, input);
  int n_classes = THSYCLTensor_(size)(state, input, n_dims - 1); 

  THSYCLTensor_(resizeAs)(state, gradInput, input);
  THSYCLTensor_(zero)(state, gradInput);
  THArgCheck(THSYCLTensor_(isContiguous)(state, gradInput), 4, "gradInput must be contiguous");

  if (weights) {
    THSYCLNN_assertSameGPU(
      state, 5, weights, input, target, gradInput, total_weight
    );  
  }
  else {
    THSYCLNN_assertSameGPU(
      state, 4, input, target, gradInput, total_weight
    );  
  }

  THArgCheck(!input->is_empty() && (n_dims <= 2 && n_dims > 0), 2, "non-empty vector or matrix expected");

  int64_t batch_size = n_dims == 1 ? 1 : THSYCLTensor_(size)(state, input, 0); 
  int64_t num_targets = THSyclLongTensor_sizeLegacyNoScalars(state, target, 0); 
  THArgCheck(batch_size == num_targets,
      2, "mismatch between the batch size of input (%ld) and that of target (%ld)",
      batch_size, num_targets);

  if (weights && THSYCLTensor_(nElement)(state, weights) != n_classes) {
    THError("weight tensor should be defined either for all or no classes");
  }

  if (reduction == at::Reduction::None && n_dims == 2) {
    THSYCLNN_check_dim_size(state, gradOutput, 1, 0, batch_size);
    if (weights) {
      weights = THSYCLTensor_(newContiguous)(state, weights);
    }
    auto& sycl_queue = c10::sycl::getCurrentSYCLStream().sycl_queue();
    int64_t local_size = sycl_queue.get_device(). template get_info<cl::sycl::info::device::max_work_group_size>();
    int64_t global_size = ((batch_size + local_size -1) /local_size ) * local_size;
    bool has_weights = weights ? true: false;
    DP::buffer<uint8_t, 1> dummy_buffer(DP::range<1>(1));
    sycl_queue.submit([&](cl::sycl::handler &cgh) {
      auto target_device_tensor = THSYCLDeviceTensor<1, THSYCLIndex_t,dp_r_mode>(state, target, cgh);
    auto gradOutput_device_tensor = THSYCLDeviceTensor<1, scalar_t,dp_r_mode>(state, gradOutput, cgh);
      auto weights_acc = has_weights ? c10::sycl::SYCLAccessor<dp_r_mode>(cgh, THSYCLTensor_(data)(state, weights)) :
                                       c10::sycl::SYCLAccessor<dp_r_mode>(cgh, dummy_buffer); // dummy weights
 
    auto gradInput_device_tensor = THSYCLDeviceTensor<2, scalar_t, write_mode>(state, gradInput, cgh);
    cgh.parallel_for<updateGradInput_no_reduce_KernelName>(cl::sycl::nd_range<1>(cl::sycl::range<1>(global_size), cl::sycl::range<1>(local_size)),
    [=](cl::sycl::nd_item<1> item_id) {
      auto local_id = item_id.get_local_id(0);
        auto group_id = item_id.get_group(0);
        auto weights_ptr = has_weights ? weights_acc.template get_pointer<scalar_t>() : NULL;
  
    for(int i = group_id * local_size + local_id; i < batch_size; i += item_id.get_global_range(0)) {
      int cur_target = THSyclLongTensor_fastGetLegacy1dNoScalars(target_device_tensor, i);
      if (cur_target == ignore_index) {
        continue;
      }
      scalar_t weight = has_weights ? weights_ptr[cur_target] : ScalarConvert<int, scalar_t>::to(1);;
      THSYCLTensor_(fastSet2d)(gradInput_device_tensor, i, cur_target, -weight * THSYCLTensor_(fastGetLegacy1dNoScalars)(gradOutput_device_tensor, i));
    }
      });
  }); 

    if (weights) {
      THSYCLTensor_(free)(state, weights);
    }
    return;
  }

  weights = weights ? THSYCLTensor_(newContiguous)(state, weights) : NULL;
  target = THSYCLIndexTensor_(newContiguous)(state, target);

  THSYCLNN_check_dim_size(state, gradOutput, 1, 0, 1);
  scalar_t *gradOutput_data = THSYCLTensor_(data)(state, gradOutput);
  scalar_t *weights_data = weights ? THSYCLTensor_(data)(state, weights) : NULL;
  scalar_t *gradInput_data = THSYCLTensor_(data)(state, gradInput);
  THSYCLIndex_t  *target_data = THSYCLIndexTensor_(data)(state, target);
  scalar_t *total_weight_data = THSYCLTensor_(data)(state, total_weight);
  bool has_weights = weights_data != NULL ? true : false;
  if (THSYCLTensor_(nDimensionLegacyNoScalars)(state, input) == 1) {
    auto &sycl_queue = c10::sycl::getCurrentSYCLStream().sycl_queue();
    DP::buffer<uint8_t, 1> dummy_buffer(DP::range<1>(1));
    sycl_queue.submit([&](cl::sycl::handler &cgh) {
      auto gradInput_acc = c10::sycl::SYCLAccessor<write_mode>(cgh, gradInput_data);
    auto gradOutput_acc = c10::sycl::SYCLAccessor<dp_r_mode>(cgh, gradOutput_data);
    auto weights_acc = has_weights ? c10::sycl::SYCLAccessor<dp_r_mode>(cgh, weights_data) :
                     c10::sycl::SYCLAccessor<dp_r_mode>(cgh, dummy_buffer); // dummy weights
      auto target_acc = c10::sycl::SYCLAccessor<dp_r_mode>(cgh, target_data);
    auto total_weight_acc = c10::sycl::SYCLAccessor<dp_r_mode>(cgh, total_weight_data);
    cgh.single_task<updateGradInput_kernel1Name>([=]() {
       auto gradInput_ptr = gradInput_acc.template get_pointer<scalar_t>();
     auto gradOutput_ptr = gradOutput_acc.template get_pointer<scalar_t>();
     auto weights_ptr = has_weights ? weights_acc.template get_pointer<scalar_t>() : NULL;
     auto target_ptr = target_acc.template get_pointer<THSYCLIndex_t>();
     auto total_weight_ptr = total_weight_acc.template get_pointer<scalar_t>();
     if (*total_weight_ptr <= 0)
    return;
     scalar_t norm = (reduction == at::Reduction::Mean) ? (ScalarConvert<int, scalar_t>::to(1) / static_cast<scalar_t>(*total_weight_ptr)) : ScalarConvert<int, scalar_t>::to(1);
     int t = (int)*target_ptr;
     if (t != (int) ignore_index) {
       gradInput_ptr[t] = -(has_weights ? weights_ptr[t] : ScalarConvert<int, scalar_t>::to(1)) * norm * gradOutput_ptr[0];
     }
    });
  });
  } else {
  int nframe = THSYCLTensor_(size)(state, input, 0);
  int ndim = THSYCLTensor_(size)(state, input, 1);
  auto &sycl_queue = c10::sycl::getCurrentSYCLStream().sycl_queue();
  DP::buffer<uint8_t, 1> dummy_buffer(DP::range<1>(1));
  sycl_queue.submit([&](cl::sycl::handler &cgh) {
    auto gradInput_acc = c10::sycl::SYCLAccessor<write_mode>(cgh, gradInput_data);
    auto gradOutput_acc = c10::sycl::SYCLAccessor<dp_r_mode>(cgh, gradOutput_data);
    auto target_acc = c10::sycl::SYCLAccessor<dp_r_mode>(cgh, target_data);
    auto weights_acc = has_weights ? c10::sycl::SYCLAccessor<dp_r_mode>(cgh, weights_data) :
                                       c10::sycl::SYCLAccessor<dp_r_mode>(cgh, dummy_buffer); // dummy weights
    auto total_weight_acc = c10::sycl::SYCLAccessor<dp_r_mode>(cgh, total_weight_data);
    int64_t local_size = 32;
    cgh.parallel_for<updateGradInput_kernelName>(cl::sycl::range<1>(local_size),
    [=](cl::sycl::item<1> item_id) {
        auto gradInput_ptr = gradInput_acc.template get_pointer<scalar_t>();
        auto gradOutput_ptr = gradOutput_acc.template get_pointer<scalar_t>();
        auto weights_ptr = has_weights ? weights_acc.template get_pointer<scalar_t>() : NULL;
        auto target_ptr = target_acc.template get_pointer<THSYCLIndex_t>();
        auto total_weight_ptr = total_weight_acc.template get_pointer<scalar_t>();

    auto local_item_id = item_id.get_id(0);

      if (*total_weight_ptr <= 0)
      return;
    int i, t;
    scalar_t norm = (reduction == at::Reduction::Mean) ? (static_cast<scalar_t>(1.0f) / static_cast<scalar_t>(*total_weight_ptr)) : ScalarConvert<int, scalar_t>::to(1);
      for (i = local_item_id; i <  nframe; i += local_size) {
      t = (int)target_ptr[i];
      if (t != (int) ignore_index) {
            // assert(t >= 0 && t < n_classes)
        gradInput_ptr[i * ndim + t] = -(has_weights? weights_ptr[t] : ScalarConvert<int, scalar_t>::to(1)) * norm * gradOutput_ptr[0];

          }
    }
    });
  });
  }
  if (weights)
    THSYCLTensor_(free)(state, weights);
  THSYCLIndexTensor_(free)(state, target);
}
#endif

} // namespace impl

// namespace AtenIpexTypeDPCPP
std::tuple<at::Tensor,at::Tensor>
nll_loss_forward(const Tensor & self, const Tensor & target,
    const Tensor & weight, int64_t reduction, int64_t ignore_index) {
  auto output = at::empty({0}, self.options());
  auto total_weight = at::empty({0}, self.options());

  AT_DISPATCH_ALL_TYPES(self.scalar_type(), "ClassNLLCriterion_updateOutput",
      [&]() {
        impl::ClassNLLCriterion_updateOutput<scalar_t>(
            self, target, output, weight, total_weight, reduction, ignore_index);
      }
  );

  auto output_ = TensorImpl_Unwrap(output);
  auto total_weight_ = TensorImpl_Unwrap(total_weight);
  output_->maybe_zero_dim(reduction != Reduction::None || self.dim() == 0);
  total_weight_->maybe_zero_dim(true);

  return std::tuple<Tensor &, Tensor &>(output, total_weight);
}

} // namespace AtenIpexTypeDPCPP
} // namespace at
