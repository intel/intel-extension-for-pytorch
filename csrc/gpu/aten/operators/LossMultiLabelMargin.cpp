#include <ATen/ATen.h>
#include <ATen/Functions.h>
#include <ATen/TensorUtils.h>
#include <ATen/core/Reduction.h>
#include <ATen/native/TensorIterator.h>

#include <core/Device.h>
#include <core/Memory.h>
#include <runtime/Utils.h>
#include <utils/DPCPP.h>

#include "comm/ATDispatch.h"
#include "comm/AccumulateType.h"
#include "comm/RegistrationDeclarations.h"
#include "comm/SYCLGroupAlgorithm.h"

#include "Loops.h"

using namespace xpu::dpcpp;

namespace at {
namespace AtenIpexTypeXPU {
namespace impl {

template <typename scalar_t>
void multilabel_margin_loss_forward_kernel(
    scalar_t* output,
    scalar_t* input,
    int64_t* target,
    scalar_t* is_target,
    int nframe,
    int dim,
    bool size_average) {
  auto& queue = dpcppGetCurrentQueue();
  auto dev_id = dpcppGetDeviceIdOfCurrentQueue();
  int work_group_size = dpcppMaxWorkGroupSize(dev_id);

  using acc_t = acc_type<scalar_t>;

  auto cgf = DPCPP_Q_CGF(cgh) {
    auto smem = dpcpp_local_acc_t<acc_t>(work_group_size, cgh);

    auto kfn = DPCPP_Q_KFN(sycl::nd_item<1> item_id) {
      int local_item_id = item_id.get_local_id(0);
      int global_item_id = item_id.get_group(0);
      int local_range = item_id.get_local_range(0);

      scalar_t* input_ptr = input + global_item_id * dim;
      int64_t* target_ptr = target + global_item_id * dim;
      scalar_t* output_ptr = output + global_item_id;
      scalar_t* is_target_ptr = is_target + global_item_id * dim;

      // zero is_target
      for (int d = local_item_id; d < dim; d += local_range) {
        is_target_ptr[d] = false;
      }
      item_id.barrier(dpcpp_global_fence);

      // mark targets in is_target
      if (local_item_id == 0) {
        for (int dt = 0; dt < dim; dt++) {
          int target_idx = target_ptr[dt];
          if (target_idx < 0) {
            break;
          }
          is_target_ptr[target_idx] = true;
        }
      }
      item_id.barrier(dpcpp_global_fence);

      scalar_t sum = 0.0f;
      for (int dt = 0; dt < dim; dt++) {
        // next target:
        int target_idx = target_ptr[dt];
        if (target_idx < 0) {
          break;
        }

        // current value for target
        scalar_t input_target = input_ptr[target_idx];

        // compare to all inputs (multithreaded):
        for (int d = local_item_id; d < dim; d += local_range) {
          // contribute to loss only if not a target
          if (!is_target_ptr[d]) {
            scalar_t z = 1.0f - input_target + input_ptr[d];
            if (z > 0.0f) {
              sum += z;
            }
          }
        }
      }

      acc_t total_sum = 0.0f;
      total_sum = GroupReduceSum(
          item_id, static_cast<acc_t>(sum), IPEXGetLocalAccPointer(smem));

      if (local_item_id == 0) {
        if (size_average) {
          *output_ptr = static_cast<scalar_t>((total_sum / dim) / nframe);
        } else {
          *output_ptr = static_cast<scalar_t>(total_sum / dim);
        }
      }
    };

    cgh.parallel_for(
        sycl::nd_range<1>(
            sycl::range<1>(nframe * work_group_size),
            sycl::range<1>(work_group_size)),
        kfn);
  };

  DPCPP_Q_SUBMIT(queue, cgf);
}

template <typename scalar_t>
void multilabel_margin_loss_backward_kernel(
    scalar_t* grad_input,
    scalar_t* grad_output,
    scalar_t* input,
    int64_t* target,
    scalar_t* is_target,
    int nframe,
    int dim,
    bool size_average,
    bool reduce) {
  auto& queue = dpcppGetCurrentQueue();
  auto dev_id = dpcppGetDeviceIdOfCurrentQueue();
  int work_group_size = dpcppMaxWorkGroupSize(dev_id);

  using acc_t = acc_type<scalar_t>;

  auto cgf = DPCPP_Q_CGF(cgh) {
    auto smem = dpcpp_local_acc_t<acc_t>(work_group_size, cgh);
    auto kfn = DPCPP_Q_KFN(sycl::nd_item<1> item) {
      int local_id = item.get_local_id(0);
      int group_id = item.get_group(0);
      int local_range = item.get_local_range(0);

      scalar_t* input_k = input + group_id * dim;
      scalar_t* grad_input_k = grad_input + group_id * dim;
      int64_t* target_k = target + group_id * dim;
      scalar_t* is_target_k = is_target + group_id * dim;
      scalar_t* grad_output_k = grad_output;

      if (!reduce) {
        grad_output_k += group_id;
      }

      // gain:
      scalar_t g = static_cast<scalar_t>(
          size_average && reduce ? 1.0f / static_cast<float>(nframe * dim)
                                 : 1.0f / static_cast<float>(dim));

      // iterate over targets
      for (int dt = 0; dt < dim; dt++) {
        // next target:
        int target_idx = static_cast<int>(target_k[dt]);
        if (target_idx < 0) {
          break;
        }

        // current value for target
        scalar_t input_target_k = input_k[target_idx];

        // compare to all inputs (multithreaded):
        float sum = 0.0f;
        for (int d = local_id; d < dim; d += local_range) {
          // contribute to loss only if not a target
          if (is_target_k[d]) {
            scalar_t z = 1.0f - input_target_k + input_k[d];
            if (z > 0.0f) {
              sum -= g;
              grad_input_k[d] += g;
            }
          }
        }
        item.barrier(dpcpp_global_fence);

        acc_t total_sum = 0.0f;
        total_sum = GroupReduceSum(
            item, static_cast<acc_t>(sum), IPEXGetLocalAccPointer(smem));
        if (local_id == 0) {
          grad_input_k[target_idx] += static_cast<scalar_t>(total_sum);
        }
      }

      item.barrier(dpcpp_global_fence);

      for (int d = local_id; d < dim; d += local_range) {
        grad_input_k[d] *= *grad_output_k;
      }
    };

    cgh.parallel_for(
        sycl::nd_range<1>(
            sycl::range<1>(nframe * work_group_size),
            sycl::range<1>(work_group_size)),
        kfn);
  };
  DPCPP_Q_SUBMIT(queue, cgf);
}

} // namespace impl

void check_shape(const Tensor& input, const Tensor& target) {
  int64_t ndims = input.dim();
  bool valid_inputs = (ndims == 2 && input.size(1) != 0) ||
      (ndims == 1 && input.size(0) != 0) || (ndims == 0);
  TORCH_CHECK(
      valid_inputs,
      "Expected non-empty vector or matrix with optional 0-dim batch size, but got: ",
      input.sizes());

  if (ndims <= 1) {
    int dim = input.dim() == 0 ? 1 : input.size(0);
    TORCH_CHECK(
        valid_inputs && target.dim() <= 1 && target.numel() == dim,
        "inconsistent target size: ",
        target.sizes(),
        " for input of size: ",
        input.sizes());
  } else if (ndims == 2) {
    int nframe = input.size(0);
    int dim = input.size(1);
    TORCH_CHECK(
        valid_inputs && target.dim() == 2 && target.size(0) == nframe &&
            target.size(1) == dim,
        "inconsistent target size: ",
        target.sizes(),
        " for input of size: ",
        input.sizes());
  } else {
    TORCH_CHECK(false, "Expected input of ndims <= 2, but got ndims: ", ndims);
  }
}

void multilabel_margin_loss_forward_out_xpu_template(
    const Tensor& input,
    const Tensor& target,
    int64_t reduction,
    Tensor& output,
    Tensor& is_target) {
  check_shape(input, target);
  if (input.numel() == 0) {
    return;
  }

  auto input_ = input.contiguous();
  auto target_ = target.contiguous();
  auto is_target_ = is_target.contiguous();
  is_target_.resize_as_(target);

  if (input.dim() <= 1) {
    int dim = input.dim() == 0 ? 1 : input.size(0);
    output.resize_({});
    IPEX_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        input.scalar_type(),
        "multilabel_margin_loss_forward_kernel",
        [&] {
          at::AtenIpexTypeXPU::impl::multilabel_margin_loss_forward_kernel(
              output.data_ptr<scalar_t>(),
              input_.data_ptr<scalar_t>(),
              target_.data_ptr<int64_t>(),
              is_target_.data_ptr<scalar_t>(),
              1,
              dim,
              reduction == at::Reduction::Mean);
        });
  } else if (input.dim() == 2) {
    int nframe = input.size(0);
    int dim = input.size(1);
    auto output_tmp = at::empty({input_.size(0)}, input_.options());
    if (reduction != at::Reduction::None) {
      output.resize_({});
      IPEX_DISPATCH_FLOATING_TYPES_AND2(
          at::ScalarType::Half,
          at::ScalarType::BFloat16,
          input.scalar_type(),
          "multilabel_margin_loss_forward_kernel",
          [&] {
            at::AtenIpexTypeXPU::impl::multilabel_margin_loss_forward_kernel(
                output_tmp.data_ptr<scalar_t>(),
                input_.data_ptr<scalar_t>(),
                target_.data_ptr<int64_t>(),
                is_target_.data_ptr<scalar_t>(),
                nframe,
                dim,
                reduction == at::Reduction::Mean);
          });
      at::sum_out(
          output,
          output_tmp,
          at::IntArrayRef(std::vector<int64_t>{}),
          false,
          output.scalar_type());

    } else {
      output.resize_({input.size(0)});
      IPEX_DISPATCH_FLOATING_TYPES_AND2(
          at::ScalarType::Half,
          at::ScalarType::BFloat16,
          input.scalar_type(),
          "multilabel_margin_loss_forward_kernel",
          [&] {
            at::AtenIpexTypeXPU::impl::multilabel_margin_loss_forward_kernel(
                output.data_ptr<scalar_t>(),
                input_.data_ptr<scalar_t>(),
                target_.data_ptr<int64_t>(),
                is_target_.data_ptr<scalar_t>(),
                nframe,
                dim,
                false);
          });
    }
  } else {
    TORCH_CHECK(
        false,
        "Expected 2D input with optional zero batch dim, or 1D input with non-zero dims, but got sizes: ",
        input.sizes());
  }
}

Tensor& multilabel_margin_loss_out(
    Tensor& out,
    const Tensor& self,
    const Tensor& target,
    int64_t reduction) {
  Tensor is_target = at::empty({0}, self.options().dtype(at::kBool));
  IPEX_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      self.scalar_type(),
      "multilabel_margin_loss_out",
      [&] {
        multilabel_margin_loss_forward_out_xpu_template(
            self, target, reduction, out, is_target);
      });
  return out;
}

Tensor multilabel_margin_loss(
    const Tensor& self,
    const Tensor& target,
    int64_t reduction) {
  Tensor out = at::empty({0}, self.options());
  return at::AtenIpexTypeXPU::multilabel_margin_loss_out(
      out, self, target, reduction);
}

std::tuple<Tensor&, Tensor&> multilabel_margin_loss_forward_out(
    const Tensor& self,
    const Tensor& target,
    int64_t reduction,
    Tensor& output,
    Tensor& is_target) {
  multilabel_margin_loss_forward_out_xpu_template(
      self, target, reduction, output, is_target);
  return std::tuple<Tensor&, Tensor&>(output, is_target);
}

std::tuple<Tensor, Tensor> multilabel_margin_loss_forward(
    const Tensor& self,
    const Tensor& target,
    int64_t reduction) {
  Tensor output = at::empty({0}, self.options());
  Tensor is_target = at::empty({0}, self.options());
  multilabel_margin_loss_forward_out_xpu_template(
      self, target, reduction, output, is_target);
  return std::make_tuple(output, is_target);
}

void multilabel_margin_loss_backward_xpu_out_template(
    const Tensor& grad_output,
    const Tensor& input,
    const Tensor& target,
    int64_t reduction,
    const Tensor& is_target,
    Tensor& grad_input) {
  check_shape(input, target);
  auto input_ = input.contiguous();
  if (input_.numel() == 0) {
    return;
  }

  grad_input.resize_as_(input);
  auto target_ = target.contiguous();
  auto is_target_ = is_target.contiguous();
  auto grad_output_ = grad_output.contiguous();

  if (grad_input.dim() <= 1) {
    int dim = grad_input.dim() == 0 ? 1 : grad_input.size(0);
    int target_size = target_.dim() == 0 ? 1 : target_.size(0);
    TORCH_CHECK(
        (target_.numel() != 0) && (target_.dim() <= 1) && (target_size == dim),
        "inconsistent target size");
    TORCH_CHECK(
        target_.sizes() == is_target_.sizes(), "inconsistent is_target size");

    IPEX_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        input.scalar_type(),
        "multilabel_margin_loss_backward_kernel",
        [&] {
          impl::multilabel_margin_loss_backward_kernel<scalar_t>(
              grad_input.data_ptr<scalar_t>(),
              grad_output_.data_ptr<scalar_t>(),
              input_.data_ptr<scalar_t>(),
              target_.data_ptr<int64_t>(),
              is_target_.data_ptr<scalar_t>(),
              1,
              dim,
              reduction == at::Reduction::Mean,
              reduction != at::Reduction::None);
        });
  } else if (grad_input.dim() == 2) {
    int nframe = grad_input.size(0);
    int dim = grad_input.size(1);
    TORCH_CHECK(
        (input_.size(1) != 0) && (target_.dim() == 2) &&
            (target_.size(0) == nframe) && (target_.size(1) == dim),
        "inconsistent target size");
    TORCH_CHECK(
        target_.sizes() == is_target_.sizes(), "inconsistent is_target size");

    IPEX_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        input.scalar_type(),
        "multilabel_margin_loss_backward_kernel",
        [&] {
          impl::multilabel_margin_loss_backward_kernel<scalar_t>(
              grad_input.data_ptr<scalar_t>(),
              grad_output_.data_ptr<scalar_t>(),
              input_.data_ptr<scalar_t>(),
              target_.data_ptr<int64_t>(),
              is_target_.data_ptr<scalar_t>(),
              grad_input.size(0),
              grad_input.size(1),
              reduction == at::Reduction::Mean,
              reduction != at::Reduction::None);
        });
  } else {
    TORCH_CHECK(
        false,
        "Expected 2D input with optional zero batch dim, or 1D input with non-zero dims, but got sizes: ",
        grad_input.sizes());
  }
}

Tensor& multilabel_margin_loss_backward_out(
    const Tensor& grad_output,
    const Tensor& self,
    const Tensor& target,
    int64_t reduction,
    const Tensor& is_target,
    Tensor& grad_input) {
  multilabel_margin_loss_backward_xpu_out_template(
      grad_input, self, target, reduction, is_target, grad_input);
  return grad_input;
}

Tensor multilabel_margin_loss_backward(
    const Tensor& grad_output,
    const Tensor& self,
    const Tensor& target,
    int64_t reduction,
    const Tensor& is_target) {
  Tensor grad_input = at::zeros_like(self, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  multilabel_margin_loss_backward_xpu_out_template(
      grad_output, self, target, reduction, is_target, grad_input);
  return grad_input;
}

} // namespace AtenIpexTypeXPU
} // namespace at
