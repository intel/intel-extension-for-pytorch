#include <ATen/ATen.h>
#include <ATen/Functions.h>
#include <ATen/native/Activation.h>
#include <ATen/record_function.h>

#include <core/Memory.h>
#include <core/TensorImplUtils.h>
#include <runtime/Utils.h>
#include <utils/DPCPP.h>
#include "comm/ApplyUtils.h"
#include "comm/Numerics.h"

#include "comm/ATDispatch.h"
#include "comm/Numerics.h"

using namespace xpu::dpcpp;
using namespace xpu::dpcpp::detail;

namespace at {
namespace AtenIpexTypeXPU {
namespace impl {

// Here is the adamW link,
// https://github.com/huggingface/transformers/blob/master/src/transformers/optimization.py
// scalar_t = weight dtype = grad dtype(BF16 or FP16)
template <typename scalar_t>
static void ComputeAdamWeightDecayKernel(
    Tensor& master_weight,
    Tensor& weight,
    Tensor& grad,
    Tensor& avg,
    Tensor& avg_sq,
    double step,
    double lr,
    double eps,
    double beta1,
    double beta2,
    double weight_decay,
    const bool correct_bias) {
  auto& dpcpp_queue = dpcppGetCurrentQueue();
  auto total_threads = master_weight.numel();

  auto cgf = DPCPP_Q_CGF(cgh) {
    auto master_weight_ptr = master_weight.data_ptr<float>();
    auto weight_ptr = weight.data_ptr<scalar_t>();
    auto grad_ptr = grad.data_ptr<scalar_t>();
    auto exp_avg_ptr = avg.data_ptr<float>();
    auto exp_avg_sq_ptr = avg_sq.data_ptr<float>();
    cgh.parallel_for(DPCPP::range<1>(total_threads), [=](DPCPP::item<1> item) {
      auto id = item.get_id(0);

      // master weight grad should be fp32 to involve in computation to keep
      // acc.
      auto grad_elm = static_cast<float>(grad_ptr[id]);

      // exp_avg
      auto exp_ele = exp_avg_ptr[id];
      exp_ele = exp_ele * beta1 + grad_elm * (1.0 - beta1);
      exp_avg_ptr[id] = exp_ele;

      // exp_avg_sq
      auto exp_avg_ele = exp_avg_sq_ptr[id];
      exp_avg_ele = exp_avg_ele * beta2 + grad_elm * grad_elm * (1.0 - beta2);
      exp_avg_sq_ptr[id] = exp_avg_ele;

      // denom
      auto denom = Numerics<float>::sqrt(exp_avg_ele) + eps;

      auto step_size = lr;
      if (correct_bias) {
        step_size = step_size *
            Numerics<float>::sqrt(1.0 - Numerics<float>::pow(beta2, step)) /
            (1.0 - Numerics<float>::pow(beta1, step));
      }

      // p.data.addcdiv_(exp_avg, denom, value=-step_size)
      auto master_weight_elem = master_weight_ptr[id];
      master_weight_elem = master_weight_elem - step_size * (exp_ele / denom);

      // p.data.add_(p.data, alpha=(-group["lr"] * group["weight_decay"]))
      if (weight_decay > 0.0) {
        master_weight_elem =
            master_weight_elem - master_weight_elem * (lr * weight_decay);
      }

      // update master weight fp32
      master_weight_ptr[id] = master_weight_elem;

      // update real weight bf16
      weight_ptr[id] = static_cast<scalar_t>(master_weight_elem);
    });
  };
  DPCPP_Q_SUBMIT(dpcpp_queue, cgf);
}

} // namespace impl

Tensor& fused_adamW(
    Tensor& master_grad_input,
    Tensor& grad_input,
    Tensor& grad,
    Tensor& avg,
    Tensor& avg_sq,
    int64_t step,
    double lr,
    double eps,
    double beta1,
    double beta2,
    double weight_decay,
    const bool correct_bias) {
  RECORD_FUNCTION(
      "fused_adamW",
      std::vector<c10::IValue>({master_grad_input, avg, avg_sq}));
  auto master_w = master_grad_input.contiguous();
  auto w = grad_input.contiguous();
  auto gw = grad.contiguous();

  // scalar_t = weight dtype = grad dtype
  IPEX_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      w.scalar_type(),
      "apply_adam_weight_decay_dpcpp",
      [&] {
        impl::ComputeAdamWeightDecayKernel<scalar_t>(
            master_w,
            w,
            gw,
            avg,
            avg_sq,
            step,
            lr,
            eps,
            beta1,
            beta2,
            weight_decay,
            correct_bias);
      });
  return master_grad_input;
}

} // namespace AtenIpexTypeXPU
} // namespace at
