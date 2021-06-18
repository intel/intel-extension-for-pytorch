#include <ATen/ATen.h>
#include <ATen/Functions.h>
#include <ATen/record_function.h>
#include <ATen/native/Activation.h>

#include <core/Memory.h>
#include <core/TensorImplUtils.h>
#include <utils/DPCPP.h>
#include <runtime/DPCPPUtils.h>
#include "comm/ApplyUtils.h"

#include "comm/Numerics.h"
#include "comm/ATDispatch.h"

using namespace xpu::dpcpp::detail;

namespace at {
namespace AtenIpexTypeXPU {
namespace impl {

template <typename scalar_t>
class adamw_dpcpp_kernel{};

template <typename scalar_t>
static void ComputeAdamWeightDecayKernel (
    const Tensor& grad_input,
    Tensor &out,
    const Tensor& avg,
    const Tensor& avg_sq,
    double step,
    double lr,
    double eps,
    double beta1,
    double beta2,
    double weight_decay,
    const bool correct_bias){

    auto& dpcpp_queue = xpu::dpcpp::getCurrentDPCPPStream().dpcpp_queue();
    auto total_threads = grad_input.numel();

    auto cgf = DPCPP_Q_CGF(cgh) {
        auto grad_input_ptr = grad_input.data_ptr<scalar_t>();
        auto avg_ptr = avg.data_ptr<scalar_t>();
        auto avg_sq_ptr = avg_sq.data_ptr<scalar_t>();
        auto out_ptr = out.data_ptr<scalar_t>();
        cgh.parallel_for<adamw_dpcpp_kernel<scalar_t>>(
            DPCPP::range<1>(total_threads), [=](DPCPP::item<1> itemId) {
                auto id = itemId.get_id(0);

                auto alpha = lr;
                if(correct_bias){
                    alpha = alpha * DPCPP::sqrt(1.0 - DPCPP::pow(beta2,step)) /
                    (1.0 - DPCPP::pow(beta1,step));
                }
                auto wd_sub = 1.0 - weight_decay * lr;

                auto grad_ele = grad_input_ptr[id];
                auto m_ele = avg_ptr[id];
                auto v_ele = avg_sq_ptr[id];
                auto m = m_ele + (grad_ele - m_ele) * (1.0 - beta1);  //exp_avg
                auto v = v_ele + (grad_ele * grad_ele - v_ele) * (1.0 - beta2);  //exp_avg_sq
                auto var_ele = wd_sub * out_ptr[id] -
                            (m * alpha) / (DPCPP::sqrt(v) + eps);
                out_ptr[id] = var_ele;
        });
    };
    DPCPP_Q_ASYNC_SUBMIT(dpcpp_queue, cgf);
}

} // namespace impl

Tensor& fused_adamW(
    Tensor& grad_input,
    const Tensor& avg,
    const Tensor& avg_sq,
    int64_t step,
    double lr,
    double eps,
    double beta1,
    double beta2,
    double weight_decay,
    const bool correct_bias
    ){
    RECORD_FUNCTION("fused_adamW", std::vector<c10::IValue>({grad_input, avg, avg_sq}));
    auto gI_ = grad_input.contiguous();
    Tensor output = at::empty_like(gI_);
    IPEX_DISPATCH_FLOATING_TYPES_AND_HALF(
        grad_input.scalar_type(), "apply_adam_weight_decay_dpcpp", [&] {
            impl::ComputeAdamWeightDecayKernel<scalar_t>(
                gI_,
                output,
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
    return grad_input;
}

} // namespace AtenIpexTypeXPU
} // namespace at
