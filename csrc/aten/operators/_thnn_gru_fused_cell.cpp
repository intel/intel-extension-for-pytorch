
#include <ATen/ATen.h>
#include <core/detail/IndexUtils.h>
#include <core/detail/TensorInfo.h>
#include <runtime/Utils.h>
#include <stdlib.h>
#include "comm/ATDispatch.h"
#include "comm/Numerics.h"
#include "comm/RegistrationDeclarations.h" // important

using namespace xpu::dpcpp::detail;
using namespace xpu::dpcpp;
using namespace at::native;

static constexpr int64_t GRU_WORKSPACE_MULTIPLIER = 5;

namespace at {
namespace AtenIpexTypeXPU {

namespace impl {
// forward dpcpp implementation
template <typename T>
static inline void fuse_ops_kernel(
    const T* igates,
    const T* hgates,
    const T* ibias,
    const T* hbias,
    const T* hidden,
    T* output,
    T* workspace,
    const int64_t feature_size,
    const int64_t batch_size,
    bool has_bias) {
  auto& dpcpp_queue = dpcppGetCurrentQueue();

  auto dev_id = dpcppGetDeviceIdOfCurrentQueue();
  int64_t max_group_size = dpcppMaxWorkGroupSize(dev_id);

  const int64_t COL = feature_size;
  const int64_t ROW = batch_size;
  int64_t GROUP_SIZE = std::min(feature_size, max_group_size);

  const sycl::range<1> global_size{ROW * GROUP_SIZE};
  const sycl::range<1> local_size{GROUP_SIZE};

  auto cgf = DPCPP_Q_CGF(cgh) {
    cgh.parallel_for(
        sycl::nd_range<1>(global_size, local_size), [=](auto itemId) {
          int64_t lid = itemId.get_local_id(0);
          int64_t gid = itemId.get_group(0);
          // this kernel is used to implemente the small op fusion in GRUCell.
          // The small ops include
          // (1) reset_gate = sigmoid(W_i0x+b_i0 + W_h0h+b_h0)
          // (2) input_gate = sigmoid(W_i1x+b_i1+ w_h1h+b_h1)
          // (3) new_gate = tanh(W_i2x+b_i2 + reset_gate*(W_h2h+b_h2))
          // (4) output = (1-input_gate)*new_gate + input_gate*hidden
          // All above operations are elementwise and cannot be implemented in
          // parallel as the sequential dependence. The igates or hgates here
          // are blocks of 3 three tensors after matmul. So one can access them
          // respectively by the pointer with constant shift of COL.
          int64_t gate0_index = gid * COL * 3;
          int64_t gate1_index = gate0_index + COL;
          int64_t gate2_index = gate1_index + COL;
          int64_t workspace0_index = gid * COL * 5;
          int64_t workspace1_index = workspace0_index + COL;
          int64_t workspace2_index = workspace1_index + COL;
          int64_t workspace3_index = workspace2_index + COL;
          int64_t workspace4_index = workspace3_index + COL;
          for (int64_t loc = lid; loc < COL; loc += GROUP_SIZE) {
            int64_t index = gid * COL + loc;
            if (has_bias) {
              T reset_gate = 1.0 /
                  (1.0 +
                   Numerics<T>::exp(
                       -(igates[gate0_index + loc] + hgates[gate0_index + loc] +
                         ibias[loc] + hbias[loc])));

              T input_gate = 1.0 /
                  (1.0 +
                   Numerics<T>::exp(
                       -(igates[gate1_index + loc] + hgates[loc + gate1_index] +
                         ibias[loc + COL] + hbias[loc + COL])));

              T hn_bn = hgates[loc + gate2_index] + hbias[loc + 2 * COL];

              T new_gate = Numerics<T>::tanh(
                  igates[gate2_index + loc] + ibias[loc + 2 * COL] +
                  reset_gate * hn_bn);

              output[index] =
                  (1.0 - input_gate) * new_gate + input_gate * hidden[index];
              // save for the backward computation
              workspace[loc + workspace0_index] = reset_gate;
              workspace[workspace1_index + loc] = input_gate;
              workspace[workspace2_index + loc] = new_gate;
              workspace[workspace3_index + loc] = hidden[index];
              workspace[workspace4_index + loc] = hn_bn;
            } else {
              T reset_gate = 1.0 /
                  (1.0 +
                   Numerics<T>::exp(-(
                       igates[gate0_index + loc] + hgates[gate0_index + loc])));

              T input_gate = 1.0 /
                  (1.0 +
                   Numerics<T>::exp(-(
                       igates[gate1_index + loc] + hgates[loc + gate1_index])));

              T hn_bn = hgates[loc + gate2_index];

              T new_gate = Numerics<T>::tanh(
                  igates[gate2_index + loc] + reset_gate * hn_bn);

              output[index] =
                  (1.0 - input_gate) * new_gate + input_gate * hidden[index];
              // save for the backward computation
              workspace[loc + workspace0_index] = reset_gate;
              workspace[workspace1_index + loc] = input_gate;
              workspace[workspace2_index + loc] = new_gate;
              workspace[workspace3_index + loc] = hidden[index];
              workspace[workspace4_index + loc] = hn_bn;
            }
          }
        });
  };
  DPCPP_Q_SUBMIT(dpcpp_queue, cgf);
}

// backward dpcpp implementation
template <typename T>
static inline void fuse_ops_kernel_backward(
    const T* grad_hy,
    const T* workspace,
    T* grad_input_gates,
    T* grad_hidden_gates,
    T* grad_hx,
    const int64_t feature_size,
    const int64_t batch_size) {
  auto& dpcpp_queue = dpcppGetCurrentQueue();
  auto dev_id = dpcppGetDeviceIdOfCurrentQueue();
  int64_t max_group_size = dpcppMaxWorkGroupSize(dev_id);
  const int64_t COL = feature_size;
  const int64_t ROW = batch_size;
  int64_t GROUP_SIZE = std::min(feature_size, max_group_size);

  const sycl::range<1> global_size{ROW * GROUP_SIZE};
  const sycl::range<1> local_size{GROUP_SIZE};
  // all input gates have the same shape
  // all intermdiate variables are precessed in this function
  auto cgf = DPCPP_Q_CGF(cgh) {
    cgh.parallel_for(
        sycl::nd_range<1>(global_size, local_size), [=](auto itemId) {
          int64_t lid = itemId.get_local_id(0);
          int64_t gid = itemId.get_group(0);
          int64_t grad_0 = gid * COL * 3;
          int64_t grad_1 = grad_0 + COL;
          int64_t grad_2 = grad_1 + COL;
          // the bwd process is accomplished with elementwise operation
          // the storage order of the workspace is vital and one can according
          // to their name to access different blocks of a whole tensor. For
          // example, workspacer_index denotes the reset_gates cache in fwd and
          // vice versa.
          int64_t workspacer_index = gid * COL * 5; // r
          int64_t workspacez_index = workspacer_index + COL; // z
          int64_t workspacen_index = workspacez_index + COL; // n
          int64_t workspaceh_index = workspacen_index + COL; // h
          int64_t workspacehbn_index = workspaceh_index + COL; // hbn
          for (int64_t loc = lid; loc < COL; loc += GROUP_SIZE) {
            int64_t index_ = gid * COL + loc;

            // grad_input_1 = A*(1-z)*(1-n^2)*hn*(1-r)*r
            grad_input_gates[grad_0 + loc] = grad_hy[index_] *
                (1.0 - workspace[workspacez_index + loc]) *
                (1.0 -
                 workspace[workspacen_index + loc] *
                     workspace[workspacen_index + loc]) *
                workspace[workspacehbn_index + loc] *
                (1.0 - workspace[workspacer_index + loc]) *
                workspace[workspacer_index + loc];
            // grad_input_2 = A*(h-n)*(1-z)*z
            grad_input_gates[grad_1 + loc] = grad_hy[index_] *
                (workspace[workspaceh_index + loc] -
                 workspace[workspacen_index + loc]) *
                (1.0 - workspace[workspacez_index + loc]) *
                workspace[workspacez_index + loc];
            // grad_input_3 = A*(1-z)*(1-n^2)
            grad_input_gates[grad_2 + loc] = grad_hy[index_] *
                (1 - workspace[workspacez_index + loc]) *
                (1 -
                 workspace[workspacen_index + loc] *
                     workspace[workspacen_index + loc]);
            // grad_hidden_1 = A*(1-z)*(1-n^2)*hn*(1-r)*r
            grad_hidden_gates[grad_0 + loc] = grad_hy[index_] *
                (1.0 - workspace[workspacez_index + loc]) *
                (1.0 -
                 workspace[workspacen_index + loc] *
                     workspace[workspacen_index + loc]) *
                workspace[workspacehbn_index + loc] *
                (1.0 - workspace[workspacer_index + loc]) *
                workspace[workspacer_index + loc];
            // grad_hidden_2 = A*(h-n)*(1-z)*z
            grad_hidden_gates[grad_1 + loc] = grad_hy[index_] *
                (workspace[workspaceh_index + loc] -
                 workspace[workspacen_index + loc]) *
                (1 - workspace[workspacez_index + loc]) *
                workspace[workspacez_index + loc];
            // grad_hidden_3 = A*(1-z)*(1-n^2)*r
            grad_hidden_gates[grad_2 + loc] = grad_hy[index_] *
                (1.0 - workspace[workspacez_index + loc]) *
                (1.0 -
                 workspace[workspacen_index + loc] *
                     workspace[workspacen_index + loc]) *
                workspace[workspacer_index + loc];
            grad_hx[index_] =
                grad_hy[index_] * workspace[workspacez_index + loc];
          }
        });
  };
  DPCPP_Q_SUBMIT(dpcpp_queue, cgf);
}

} // namespace impl

std::tuple<Tensor, Tensor> _thnn_fused_gru_cell(
    const Tensor& input_gates,
    const Tensor& hidden_gates,
    const Tensor& hx,
    const c10::optional<Tensor>& input_bias_opt,
    const c10::optional<Tensor>& hidden_bias_opt) {
  c10::MaybeOwned<Tensor> input_bias_maybe_owned =
      at::borrow_from_optional_tensor(input_bias_opt);
  const Tensor& input_bias = *input_bias_maybe_owned;
  const Tensor& hidden_bias =
      c10::value_or_else(hidden_bias_opt, [] { return Tensor(); });

  auto batched_input = true;
  auto feature_size = hx.size(1);
  auto batch_size = hx.size(0);

  if (hx.dim() == 1) {
    // no batch input
    auto batched_input = false;
    auto feature_size = hx.size(0);
    hx.resize_({1, hx.size(0)});
    auto batch_size = 1;
  }

  at::Tensor workspace = at::empty(
      {hx.size(0), hx.size(1) * GRU_WORKSPACE_MULTIPLIER}, hx.options());

  auto hy = at::empty_like(hx, LEGACY_CONTIGUOUS_MEMORY_FORMAT);

  IPEX_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      input_gates.scalar_type(),
      "_thnn_fused_gru_cell",
      [&] {
        bool has_bias = input_bias.defined() || hidden_bias.defined();
        scalar_t* input_bias_ptr =
            input_bias.defined() ? input_bias.data_ptr<scalar_t>() : NULL;
        scalar_t* hidden_bias_ptr =
            hidden_bias.defined() ? hidden_bias.data_ptr<scalar_t>() : NULL;
        impl::fuse_ops_kernel<scalar_t>(
            input_gates.data_ptr<scalar_t>(),
            hidden_gates.data_ptr<scalar_t>(),
            input_bias_ptr,
            hidden_bias_ptr,
            hx.data_ptr<scalar_t>(),
            hy.data_ptr<scalar_t>(),
            workspace.data_ptr<scalar_t>(),
            feature_size,
            batch_size,
            has_bias);
      });

  if (!batched_input) {
    hy.resize_({feature_size});
    return std::make_tuple(hy, workspace);
  }
  return std::make_tuple(hy, workspace);
}

// backward process
std::tuple<Tensor, Tensor, Tensor, Tensor, Tensor> _thnn_fused_gru_cell_backward(
    const Tensor& grad_hy,
    const Tensor& workspace,
    bool has_bias) {
  int64_t hidden_size = workspace.size(1) / GRU_WORKSPACE_MULTIPLIER;
  int64_t batch_size = workspace.size(0);

  // grad_hy input is not contiguous in GRUCell bwd. But we hope it contiguous
  // to save much time. One can access the grad_hy without congtiguous operation
  // with information of current timestep. Then, one can access the grad_hy by
  // the pointer directly i.e. index_grad_hy = row*(seq_len*hidden) +
  // timestep*hidden + loc. this operation requires knowing the seq_len and
  // current timestep. but one can fuse all GRU components such as the different
  // layers and matmul and smallop in a GRUCEll. Then, the freedom degree is
  // larger without concerning the current API implementations.
  auto grad_hy_ = grad_hy;
  if (grad_hy_.is_contiguous() == false) {
    grad_hy_ = grad_hy_.contiguous();
  }

  at::Tensor grad_input_gates =
      at::empty({batch_size, hidden_size * 3}, workspace.options());

  at::Tensor grad_hidden_gates =
      at::empty({batch_size, hidden_size * 3}, workspace.options());

  auto grad_hx = at::empty_like(grad_hy_, workspace.options());

  IPEX_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      grad_hy.scalar_type(),
      "_thnn_fused_gru_cell_backward",
      [&] {
        impl::fuse_ops_kernel_backward<scalar_t>(
            grad_hy_.data_ptr<scalar_t>(),
            workspace.data_ptr<scalar_t>(),
            grad_input_gates.data_ptr<scalar_t>(),
            grad_hidden_gates.data_ptr<scalar_t>(),
            grad_hx.data_ptr<scalar_t>(),
            hidden_size,
            batch_size);
      });

  at::Tensor grad_input_bias, grad_hidden_bias;

  if (has_bias) {
    grad_input_bias = grad_input_gates.sum(0, false);
    grad_hidden_bias = grad_hidden_gates.sum(0, false);
  }

  return std::make_tuple(
      grad_input_gates,
      grad_hidden_gates,
      grad_hx,
      grad_input_bias,
      grad_hidden_bias);
}

} // namespace AtenIpexTypeXPU
} // namespace at
