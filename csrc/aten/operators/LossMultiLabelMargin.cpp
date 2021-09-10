#include <ATen/ATen.h>
#include <ATen/AtenIpexTypeXPU.h>
#include <ATen/Functions.h>
#include <ATen/TensorUtils.h>
#include <ATen/core/Reduction.h>
#include <ATen/native/TensorIterator.h>

#include <core/Memory.h>
#include <core/TensorImplUtils.h>
#include <runtime/Utils.h>
#include <utils/DPCPP.h>

#include "comm/ATDispatch.h"

#include "Loops.h"

using namespace xpu::dpcpp;

namespace at {
namespace AtenIpexTypeXPU {
namespace impl {

template <typename scalar_t>
void MultilabelMarginCriterion_updateOutput(
    Tensor& output,
    const Tensor& input,
    const Tensor& target,
    int64_t reduction,
    Tensor& is_target) {
  auto target_arg = TensorArg(target, "target", 2);

  const auto ndims = input.dim();

  TORCH_CHECK(
      input.numel() > 0 && ndims <= 2,
      "non-empty vector or matrix expected, got size: ",
      input.sizes());

  int64_t nframe, dim;
  if (ndims <= 1) {
    nframe = 1;
    dim = (ndims == 0) ? 1 : input.size(0);
    TORCH_CHECK(
        target.numel() > 0 && target.dim() <= 1 && target.numel() == dim,
        "inconsistent size ",
        target.sizes(),
        " for ",
        target_arg);
  } else {
    nframe = input.size(0);
    dim = input.size(1);
    TORCH_CHECK(
        target.numel() > 0 && target.dim() == 2 && target.size(0) == nframe &&
            target.size(1) == dim,
        "inconsistent size ",
        target.sizes(),
        " for ",
        target_arg);
  }

  TORCH_CHECK(
      target.min().item<int64_t>() >= -1, target_arg, " is out of range");
  TORCH_CHECK(
      target.max().item<int64_t>() < dim, target_arg, "is out of range");

  auto input_contiguous = input.contiguous();
  auto target_contiguous = target.contiguous();

  is_target.resize_as_(target);
  TORCH_CHECK(is_target.is_contiguous(), "is_target must be contiguous");
  is_target.zero_();

  if (reduction != Reduction::None || target.dim() <= 1) {
    output.resize_({});
  } else {
    output.resize_({nframe});
  }

  auto& queue = dpcppGetCurrentQueue();
  auto dev_id = dpcppGetDeviceIdOfCurrentQueue();
  int64_t local_size = dpcppMaxWorkGroupSize(dev_id);

  auto cgf = DPCPP_Q_CGF(cgh) {
    auto input_data = input_contiguous.data_ptr<scalar_t>();
    auto target_data = target_contiguous.data_ptr<int64_t>();
    auto output_data = output.data_ptr<scalar_t>();
    auto is_target_data = is_target.data_ptr<scalar_t>();
    auto local_output_data = dpcpp_local_acc_t<scalar_t>(local_size, cgh);

    if (reduction == Reduction::None && output.dim() > 0) {
      auto kfn = DPCPP_Q_KFN(DPCPP::item<1> item_id) {
        auto input_ptr = input_data;
        auto target_ptr = target_data;
        auto output_ptr = output_data;
        auto is_target_ptr = is_target_data;
        auto local_item_id = item_id.get_id(0);
        for (int i = local_item_id; i < nframe; i += local_size) {
          scalar_t sum = 0;
          for (int64_t ddt = 0; ddt < dim; ddt++) {
            auto target_idx = target_ptr[i * dim + ddt];
            if (target_idx < 0)
              break;
            is_target_ptr[i * dim + target_idx] = 1;
          }
          for (int64_t dt = 0; dt < dim; dt++) {
            auto target_idx = target_ptr[i * dim + dt];
            if (target_idx < 0)
              break;

            auto input_target = input_ptr[i * dim + target_idx];
            for (int64_t d = 0; d < dim; d++) {
              if (!is_target_ptr[i * dim + d]) {
                scalar_t z = 1.0 - input_target + input_ptr[i * dim + d];
                if (z > 0)
                  sum += z;
              }
            }
          }
          sum /= dim;
          output_ptr[i] = sum;
        }
      };
      cgh.parallel_for(DPCPP::range<1>(local_size), kfn);
    } else {
      auto kfn = DPCPP_Q_KFN(DPCPP::nd_item<1> item_id) {
        auto input_ptr = input_data;
        auto target_ptr = target_data;
        auto output_ptr = output_data;
        auto is_target_ptr = is_target_data;
        auto local_item_id = item_id.get_local_id(0);
        local_output_data[local_item_id] = 0.0;
        for (int i = local_item_id; i < nframe; i += local_size) {
          scalar_t sum = 0;
          for (int64_t ddt = 0; ddt < dim; ddt++) {
            auto target_idx = target_ptr[i * dim + ddt];
            if (target_idx < 0)
              break;
            is_target_ptr[i * dim + target_idx] = 1;
          }
          for (int64_t dt = 0; dt < dim; dt++) {
            auto target_idx = target_ptr[i * dim + dt];
            if (target_idx < 0)
              break;

            auto input_target = input_ptr[i * dim + target_idx];
            for (int64_t d = 0; d < dim; d++) {
              if (!is_target_ptr[i * dim + d]) {
                scalar_t z = 1.0 - input_target + input_ptr[i * dim + d];
                if (z > 0)
                  sum += z;
              }
            }
          }
          sum /= dim;
          if (reduction == Reduction::Mean)
            sum /= nframe;
          local_output_data[local_item_id] += sum;
        }

        // reduce
        for (int64_t i = (local_size >> 1); i > 0; i >>= 1) {
          item_id.barrier(dpcpp_global_and_local_fence);
          if (local_item_id < i)
            local_output_data[local_item_id] +=
                local_output_data[local_item_id + i];
        }
        item_id.barrier(dpcpp_global_and_local_fence);
        output_ptr[0] = local_output_data[0];
      };
      cgh.parallel_for(
          DPCPP::nd_range<1>(
              DPCPP::range<1>(local_size), DPCPP::range<1>(local_size)),
          kfn);
    }
  };

  DPCPP_Q_SUBMIT(queue, cgf);
}

template <typename scalar_t>
void MultilabelMarginCriterion_updateGradInput(
    Tensor& grad_input,
    const Tensor& grad_output,
    const Tensor& input,
    const Tensor& target,
    int64_t reduction,
    const Tensor& is_target) {
  auto target_arg = TensorArg(target, "target", 3);
  auto is_target_arg = TensorArg(is_target, "is_target", 5);

  const auto ndims = input.dim();

  TORCH_CHECK(
      input.numel() > 0 && ndims <= 2,
      "non-empty vector or matrix expected, got size: ",
      input.sizes());

  int64_t nframe, dim;
  if (ndims <= 1) {
    nframe = 1;
    dim = (ndims == 0) ? 1 : input.size(0);
    TORCH_CHECK(
        target.numel() > 0 && target.dim() <= 1 && target.numel() == dim,
        "inconsistent size ",
        target.sizes(),
        " for ",
        target_arg);
  } else {
    nframe = input.size(0);
    dim = input.size(1);
    TORCH_CHECK(
        target.numel() > 0 && target.dim() == 2 && target.size(0) == nframe &&
            target.size(1) == dim,
        "inconsistent size ",
        target.sizes(),
        " for ",
        target_arg);
  }
  IsOnSameDevice(
      "multilabel_margin_loss_backward_out", target_arg, is_target_arg);

  TORCH_CHECK(
      target.min().item<int64_t>() >= -1, target_arg, " is out of range");
  TORCH_CHECK(
      target.max().item<int64_t>() < dim, target_arg, "is out of range");

  auto input_contiguous = input.contiguous();
  auto target_contiguous = target.contiguous();
  auto is_target_contiguous = is_target.contiguous();

  grad_input.resize_as_(input);
  TORCH_CHECK(grad_input.is_contiguous(), "grad_input must be contiguous");
  grad_input.zero_();

  auto is_target_cont_arg =
      TensorArg(is_target_contiguous, "is_target_cont", 5);
  TORCH_CHECK(
      is_target_contiguous.min().item<scalar_t>() >= 0,
      is_target_cont_arg,
      " is out of range");
  TORCH_CHECK(
      is_target_contiguous.max().item<scalar_t>() <= 1,
      is_target_cont_arg,
      " is out of range");

  scalar_t g = static_cast<scalar_t>(
      reduction == Reduction::Mean ? 1. / (nframe * dim) : 1. / dim);

  auto& queue = dpcppGetCurrentQueue();
  auto dev_id = dpcppGetDeviceIdOfCurrentQueue();
  int64_t local_size = dpcppMaxWorkGroupSize(dev_id);

  auto cgf = DPCPP_Q_CGF(cgh) {
    auto grad_input_data = grad_input.data_ptr<scalar_t>();
    auto grad_output_data = grad_output.data_ptr<scalar_t>();
    auto input_data = input_contiguous.data_ptr<scalar_t>();
    auto target_data = target_contiguous.data_ptr<int64_t>();
    auto is_target_data = is_target_contiguous.data_ptr<scalar_t>();

    auto kfn = DPCPP_Q_KFN(DPCPP::item<1> item_id) {
      auto grad_input_ptr = grad_input_data;
      auto grad_output_ptr = grad_output_data;
      auto input_ptr = input_data;
      auto target_ptr = target_data;
      auto is_target_ptr = is_target_data;
      auto local_item_id = item_id.get_id(0);

      for (int i = local_item_id; i < nframe; i += local_size) {
        for (int64_t dt = 0; dt < dim; dt++) {
          auto target_idx = target_ptr[i * dim + dt];
          if (target_idx < 0)
            break;

          auto input_target = input_ptr[i * dim + target_idx];
          for (int64_t d = 0; d < dim; d++) {
            if (!is_target_ptr[i * dim + d]) {
              scalar_t z = 1.0 - input_target + input_ptr[i * dim + d];
              if (z > 0) {
                grad_input_ptr[i * dim + target_idx] -= g;
                grad_input_ptr[i * dim + d] += g;
              }
            }
          }
        }
        for (int64_t d = 0; d < dim; d++)
          grad_input_ptr[i * dim + d] *= (reduction == Reduction::None)
              ? grad_output_ptr[i]
              : grad_output_ptr[0];
      }
    };
    cgh.parallel_for(DPCPP::range<1>(local_size), kfn);
  };

  DPCPP_Q_SUBMIT(queue, cgf);
}

} // namespace impl

Tensor& multilabel_margin_loss_out(
    Tensor& out,
    const Tensor& self,
    const Tensor& target,
    int64_t reduction) {
  Tensor is_target = at::empty({0}, self.options());
  IPEX_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      self.scalar_type(),
      "multilabel_margin_loss_out",
      [&] {
        impl::MultilabelMarginCriterion_updateOutput<scalar_t>(
            out, self, target, reduction, is_target);
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
    Tensor& output,
    Tensor& is_target,
    const Tensor& self,
    const Tensor& target,
    int64_t reduction) {
  IPEX_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      self.scalar_type(),
      "multilabel_margin_loss_forward_out",
      [&] {
        impl::MultilabelMarginCriterion_updateOutput<scalar_t>(
            output, self, target, reduction, is_target);
      });
  return std::tuple<Tensor&, Tensor&>(output, is_target);
}

std::tuple<Tensor, Tensor> multilabel_margin_loss_forward(
    const Tensor& self,
    const Tensor& target,
    int64_t reduction) {
  Tensor output = at::empty({0}, self.options());
  Tensor is_target = at::empty({0}, self.options());
  return at::AtenIpexTypeXPU::multilabel_margin_loss_forward_out(
      output, is_target, self, target, reduction);
}

Tensor& multilabel_margin_loss_backward_out(
    Tensor& grad_input,
    const Tensor& grad_output,
    const Tensor& self,
    const Tensor& target,
    int64_t reduction,
    const Tensor& is_target) {
  IPEX_DISPATCH_FLOATING_TYPES_AND(
      at::ScalarType::BFloat16,
      self.scalar_type(),
      "multilabel_margin_loss_backward_out",
      [&] {
        impl::MultilabelMarginCriterion_updateGradInput<scalar_t>(
            grad_input, grad_output, self, target, reduction, is_target);
      });
  return grad_input;
}

Tensor multilabel_margin_loss_backward(
    const Tensor& grad_output,
    const Tensor& self,
    const Tensor& target,
    int64_t reduction,
    const Tensor& is_target) {
  Tensor grad_input = at::empty({0}, self.options());
  return at::AtenIpexTypeXPU::multilabel_margin_loss_backward_out(
      grad_input, grad_output, self, target, reduction, is_target);
}

} // namespace AtenIpexTypeXPU
} // namespace at
