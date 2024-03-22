#include <ATen/ATen.h>
#include <ATen/Functions.h>
#include <ATen/TensorUtils.h>
#include <ATen/core/Reduction.h>
#include <ATen/native/TensorIterator.h>

#include <core/Memory.h>
#include <runtime/Utils.h>
#include <utils/DPCPP.h>

#include "comm/ATDispatch.h"
#include "comm/RegistrationDeclarations.h"

#include "Loops.h"

using namespace torch_ipex::xpu::dpcpp;

namespace at {
namespace AtenIpexTypeXPU {
namespace impl {

template <typename scalar_t>
struct MultiMarginCriterionUpdateOutputKernelFunctor {
  void operator()(sycl::item<1> item_id) const {
    auto input_ptr = input_data;
    auto target_ptr = target_data;
    auto output_ptr = output_data;
    auto weights_ptr = has_weights ? weights_data : NULL;
    auto local_item_id = item_id.get_id(0);
    for (int i = local_item_id; i < nframe; i += local_size) {
      scalar_t sum = 0;
      auto target_idx = target_ptr[i];
      auto input_target = input_ptr[i * dim + target_idx];
      for (auto d = 0; d < dim; d++) {
        scalar_t z = margin_ - input_target + input_ptr[i * dim + d];
        if (d == target_idx)
          continue;
        if (z > 0) {
          scalar_t h = (p_ == 1) ? z : z * z;
          if (weights_ptr)
            h *= weights_ptr[target_idx];
          sum += h;
        }
      }
      sum /= dim;
      output_ptr[i] = sum;
    }
  }
  MultiMarginCriterionUpdateOutputKernelFunctor(
      scalar_t* input_data_,
      int64_t* target_data_,
      scalar_t* output_data_,
      scalar_t* weights_data_,
      bool has_weights_,
      int64_t nframe_,
      int64_t dim_,
      int64_t local_size_,
      const scalar_t margin__,
      const int p__)
      : input_data(input_data_),
        target_data(target_data_),
        output_data(output_data_),
        weights_data(weights_data_),
        has_weights(has_weights_),
        nframe(nframe_),
        dim(dim_),
        local_size(local_size_),
        margin_(margin__),
        p_(p__) {}

 private:
  scalar_t* input_data;
  int64_t* target_data;
  scalar_t* output_data;
  scalar_t* weights_data;
  bool has_weights;
  int64_t nframe;
  int64_t dim;
  int64_t local_size;
  const scalar_t margin_;
  const int p_;
};

template <typename scalar_t>
struct MultiMarginCriterionUpdateOutputKernelFunctor2 {
  void operator()(sycl::nd_item<1> item_id) const {
    auto input_ptr = input_data;
    auto target_ptr = target_data;
    auto output_ptr = output_data;
    auto weights_ptr = has_weights ? weights_data : NULL;
    auto local_item_id = item_id.get_local_id(0);
    local_output_data[local_item_id] = 0.0;
    for (int i = local_item_id; i < nframe; i += local_size) {
      scalar_t sum = 0;
      auto target_idx = target_ptr[i];
      auto input_target = input_ptr[i * dim + target_idx];
      for (auto d = 0; d < dim; d++) {
        scalar_t z = margin_ - input_target + input_ptr[i * dim + d];
        if (d == target_idx)
          continue;
        if (z > 0) {
          scalar_t h = (p_ == 1) ? z : z * z;
          if (weights_ptr)
            h *= weights_ptr[target_idx];
          sum += h;
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
  }
  MultiMarginCriterionUpdateOutputKernelFunctor2(
      scalar_t* input_data_,
      int64_t* target_data_,
      scalar_t* output_data_,
      scalar_t* weights_data_,
      dpcpp_local_acc_t<scalar_t> local_output_data_,
      bool has_weights_,
      int64_t nframe_,
      int64_t dim_,
      int64_t local_size_,
      const scalar_t margin__,
      const int p__,
      int64_t reduction_)
      : input_data(input_data_),
        target_data(target_data_),
        output_data(output_data_),
        weights_data(weights_data_),
        local_output_data(local_output_data_),
        has_weights(has_weights_),
        nframe(nframe_),
        dim(dim_),
        local_size(local_size_),
        margin_(margin__),
        p_(p__),
        reduction(reduction_) {}

 private:
  scalar_t* input_data;
  int64_t* target_data;
  scalar_t* output_data;
  scalar_t* weights_data;
  dpcpp_local_acc_t<scalar_t> local_output_data;
  bool has_weights;
  int64_t nframe;
  int64_t dim;
  int64_t local_size;
  const scalar_t margin_;
  const int p_;
  int64_t reduction;
};

template <typename scalar_t>
void MultiMarginCriterion_updateOutput(
    Tensor& output,
    const Tensor& input,
    const Tensor& target,
    Scalar p,
    Scalar margin,
    const c10::optional<at::Tensor>& weights_optional,
    int64_t reduction) {
  c10::MaybeOwned<Tensor> weight_maybe_owned =
      at::borrow_from_optional_tensor(weights_optional);
  const Tensor& weights = *weight_maybe_owned;
  const auto ndims = input.dim();
  TORCH_CHECK(
      input.numel() > 0 && ndims <= 2,
      "non-empty vector or matrix expected, got size: ",
      input.sizes());

  const int p_ = p.toInt();
  const scalar_t margin_ = margin.to<scalar_t>();
  TORCH_CHECK(p_ == 1 || p_ == 2, "only p == 1 and p == 2 supported");

  int64_t nframe, dim;
  if (ndims <= 1) {
    nframe = 1;
    dim = (ndims == 0) ? 1 : input.size(0);
  } else {
    nframe = input.size(0);
    dim = input.size(1);
  }

  TORCH_CHECK(
      target.numel() > 0 && target.dim() <= 1 && target.numel() == nframe,
      "inconsistent target size, got: ",
      target.sizes());

  if (reduction == Reduction::None && target.dim() > 0) {
    output.resize_({nframe});
  } else {
    output.resize_({});
  }

  auto input_contiguous = input.contiguous();
  auto target_contiguous = target.contiguous();
  auto weights_contiguous =
      (weights.defined()) ? weights.contiguous() : weights;

  bool has_weights = weights.defined() ? true : false;

  auto& queue = dpcppGetCurrentQueue();
  auto dev_id = dpcppGetDeviceIdOfCurrentQueue();
  int64_t local_size = dpcppMaxWorkGroupSize(dev_id);

  auto cgf = DPCPP_Q_CGF(cgh) {
    auto input_data = input_contiguous.data_ptr<scalar_t>();
    auto target_data = target_contiguous.data_ptr<int64_t>();
    auto output_data = output.data_ptr<scalar_t>();
    auto weights_data = has_weights
        ? weights_contiguous.data_ptr<scalar_t>()
        : input_data; // use the input_data handler as dummy weights
    auto local_output_data = dpcpp_local_acc_t<scalar_t>(local_size, cgh);

    if (reduction == Reduction::None && output.dim() > 0) {
      MultiMarginCriterionUpdateOutputKernelFunctor<scalar_t> kfn(
          input_data,
          target_data,
          output_data,
          weights_data,
          has_weights,
          nframe,
          dim,
          local_size,
          margin_,
          p_);
      cgh.parallel_for<decltype(kfn)>(sycl::range<1>(local_size), kfn);
    } else {
      MultiMarginCriterionUpdateOutputKernelFunctor2<scalar_t> kfn(
          input_data,
          target_data,
          output_data,
          weights_data,
          local_output_data,
          has_weights,
          nframe,
          dim,
          local_size,
          margin_,
          p_,
          reduction);
      cgh.parallel_for<decltype(kfn)>(
          sycl::nd_range<1>(
              sycl::range<1>(local_size), sycl::range<1>(local_size)),
          kfn);
    }
  };

  DPCPP_Q_SUBMIT(queue, cgf);
}

template <typename scalar_t>
struct MultiMarginCriterionUpdateGradInputKernelFunctor {
  void operator()(sycl::item<1> item_id) const {
    auto grad_input_ptr = grad_input_data;
    auto grad_output_ptr = grad_output_data;
    auto input_ptr = input_data;
    auto target_ptr = target_data;
    auto weights_ptr = has_weights ? weights_data : NULL;
    auto local_item_id = item_id.get_id(0);

    for (int i = local_item_id; i < nframe; i += local_size) {
      auto target_idx = target_ptr[i];
      auto input_target = input_ptr[i * dim + target_idx];
      scalar_t grad_input_target = 0;
      for (auto d = 0; d < dim; d++) {
        scalar_t z = margin_ - input_target + input_ptr[i * dim + d];
        if (d == target_idx)
          continue;
        if (z > 0) {
          scalar_t h = (p_ == 1) ? g : 2 * g * z;
          if (weights_ptr)
            h *= weights_ptr[target_idx];
          grad_input_target -= h;
          grad_input_ptr[i * dim + d] = h;
        } else
          grad_input_ptr[i * dim + d] = 0;
      }
      grad_input_ptr[i * dim + target_idx] = grad_input_target;

      for (auto d = 0; d < dim; d++)
        grad_input_ptr[i * dim + d] *= reduction == Reduction::None
            ? grad_output_ptr[i]
            : grad_output_ptr[0];
    }
  }
  MultiMarginCriterionUpdateGradInputKernelFunctor(
      int64_t reduction_,
      const int p_,
      scalar_t margin_,
      int64_t nframe_,
      int64_t dim_,
      bool has_weights_,
      scalar_t g_,
      int64_t local_size_,
      scalar_t* grad_input_data_,
      scalar_t* grad_output_data_,
      scalar_t* input_data_,
      int64_t* target_data_,
      scalar_t* weights_data_)
      : reduction(reduction_),
        p_(p_),
        margin_(margin_),
        nframe(nframe_),
        dim(dim_),
        has_weights(has_weights_),
        g(g_),
        local_size(local_size_),
        grad_input_data(grad_input_data_),
        grad_output_data(grad_output_data_),
        input_data(input_data_),
        target_data(target_data_),
        weights_data(weights_data_) {}

 private:
  int64_t reduction;
  const int p_;
  const scalar_t margin_;
  int64_t nframe;
  int64_t dim;
  bool has_weights;
  scalar_t g;
  int64_t local_size;
  scalar_t* grad_input_data;
  scalar_t* grad_output_data;
  scalar_t* input_data;
  int64_t* target_data;
  scalar_t* weights_data;
};

template <typename scalar_t>
void MultiMarginCriterion_updateGradInput(
    Tensor& grad_input,
    const Tensor& grad_output,
    const Tensor& input,
    const Tensor& target,
    Scalar p,
    Scalar margin,
    const c10::optional<at::Tensor>& weights_optional,
    int64_t reduction) {
  c10::MaybeOwned<Tensor> weight_maybe_owned =
      at::borrow_from_optional_tensor(weights_optional);
  const Tensor& weights = *weight_maybe_owned;
  const auto ndims = input.dim();
  TORCH_CHECK(
      input.numel() > 0 && ndims <= 2,
      "non-empty vector or matrix expected, got size: ",
      input.sizes());

  const int p_ = p.toInt();
  const scalar_t margin_ = margin.to<scalar_t>();
  TORCH_CHECK(p_ == 1 || p_ == 2, "only p == 1 and p == 2 supported");

  int64_t nframe, dim;
  if (ndims <= 1) {
    nframe = 1;
    dim = (ndims == 0) ? 1 : input.size(0);
  } else {
    nframe = input.size(0);
    dim = input.size(1);
  }

  TORCH_CHECK(
      target.numel() > 0 && target.dim() <= 1 && target.numel() == nframe,
      "inconsistent target size, got: ",
      target.sizes());

  grad_input.resize_as_(input);
  TORCH_CHECK(grad_input.is_contiguous(), "grad_input must be contiguous");

  auto input_contiguous = input.contiguous();
  auto target_contiguous = target.contiguous();
  auto weights_contiguous =
      (weights.defined()) ? weights.contiguous() : weights;

  bool has_weights = weights.defined() ? true : false;
  scalar_t g = (reduction == Reduction::Mean)
      ? static_cast<scalar_t>(1. / (nframe * dim))
      : static_cast<scalar_t>(1. / dim);

  auto& queue = dpcppGetCurrentQueue();
  auto dev_id = dpcppGetDeviceIdOfCurrentQueue();
  int64_t local_size = dpcppMaxWorkGroupSize(dev_id);

  auto cgf = DPCPP_Q_CGF(cgh) {
    auto grad_input_data = grad_input.data_ptr<scalar_t>();
    auto grad_output_data = grad_output.data_ptr<scalar_t>();
    auto input_data = input_contiguous.data_ptr<scalar_t>();
    auto target_data = target_contiguous.data_ptr<int64_t>();
    auto weights_data = has_weights
        ? weights_contiguous.data_ptr<scalar_t>()
        : input_data; // use the input_data handler as dummy weights

    MultiMarginCriterionUpdateGradInputKernelFunctor<scalar_t> kfn(
        reduction,
        p_,
        margin_,
        nframe,
        dim,
        has_weights,
        g,
        local_size,
        grad_input_data,
        grad_output_data,
        input_data,
        target_data,
        weights_data);
    cgh.parallel_for<decltype(kfn)>(sycl::range<1>(local_size), kfn);
  };

  DPCPP_Q_SUBMIT(queue, cgf);
}

} // namespace impl

Tensor& multi_margin_loss_out(
    const Tensor& self,
    const Tensor& target,
    const Scalar& p,
    const Scalar& margin,
    const c10::optional<at::Tensor>& weight,
    int64_t reduction,
    Tensor& out) {
  IPEX_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      self.scalar_type(),
      "multi_margin_loss_out",
      [&] {
        impl::MultiMarginCriterion_updateOutput<scalar_t>(
            out, self, target, p, margin, weight, reduction);
      });
  return out;
}

Tensor multi_margin_loss(
    const Tensor& self,
    const Tensor& target,
    const Scalar& p,
    const Scalar& margin,
    const c10::optional<at::Tensor>& weight,
    int64_t reduction) {
  Tensor out = at::empty({0}, self.options());
  return at::AtenIpexTypeXPU::multi_margin_loss_out(
      self, target, p, margin, weight, reduction, out);
}

Tensor& multi_margin_loss_backward_out(
    const Tensor& grad_output,
    const Tensor& self,
    const Tensor& target,
    const Scalar& p,
    const Scalar& margin,
    const c10::optional<at::Tensor>& weight,
    int64_t reduction,
    Tensor& grad_input) {
  IPEX_DISPATCH_FLOATING_TYPES_AND(
      at::ScalarType::BFloat16,
      self.scalar_type(),
      "multi_margin_loss_backward_out",
      [&] {
        impl::MultiMarginCriterion_updateGradInput<scalar_t>(
            grad_input,
            grad_output,
            self,
            target,
            p,
            margin,
            weight,
            reduction);
      });
  return grad_input;
}

Tensor multi_margin_loss_backward(
    const Tensor& grad_output,
    const Tensor& self,
    const Tensor& target,
    const Scalar& p,
    const Scalar& margin,
    const c10::optional<at::Tensor>& weight,
    int64_t reduction) {
  Tensor grad_input = at::empty({0}, self.options());
  return at::AtenIpexTypeXPU::multi_margin_loss_backward_out(
      grad_output, self, target, p, margin, weight, reduction, grad_input);
}

} // namespace AtenIpexTypeXPU
} // namespace at
