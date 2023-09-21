#include "../../GRU.h"

#include "backward_input.h"
#include "backward_weight.h"

namespace xpu {
namespace xetla {

void gru_backward_data(
    void* layer_err_ptr,
    void* y_err_ptr,
    void* x_grad_ptr,
    void* bpi0_ptr,
    void* bpi1_ptr,
    void* partial_grad_ptr,
    void* x0_grad_ptr,
    void* reset_gate_ptr,
    void* input_gate_ptr,
    void* new_gate_ptr,
    void* hgate_2_ptr,
    void* hidden_ptr,
    void* i_weights,
    void* h_weights,
    void* mask_ptr,
    const int batch_size,
    const int input_size,
    const int hidden_size,
    const int sequence_length,
    const int layer_size,
    const float dropout,
    sycl::queue& Queue) {
  if (hidden_size > 0 && hidden_size <= 256)
    gru_backward_data_impl<m512k92n256_bpi>(
        layer_err_ptr,
        y_err_ptr,
        x_grad_ptr,
        bpi0_ptr,
        bpi1_ptr,
        partial_grad_ptr,
        x0_grad_ptr,
        reset_gate_ptr,
        input_gate_ptr,
        new_gate_ptr,
        hgate_2_ptr,
        hidden_ptr,
        i_weights,
        h_weights,
        mask_ptr,
        batch_size,
        input_size,
        hidden_size,
        sequence_length,
        layer_size,
        dropout,
        Queue);
  else if (hidden_size <= 1024)
    gru_backward_data_impl<m512k379n681_bpi>(
        layer_err_ptr,
        y_err_ptr,
        x_grad_ptr,
        bpi0_ptr,
        bpi1_ptr,
        partial_grad_ptr,
        x0_grad_ptr,
        reset_gate_ptr,
        input_gate_ptr,
        new_gate_ptr,
        hgate_2_ptr,
        hidden_ptr,
        i_weights,
        h_weights,
        mask_ptr,
        batch_size,
        input_size,
        hidden_size,
        sequence_length,
        layer_size,
        dropout,
        Queue);
  else
    assert(0); // Currently a few shapes are supported
}

void gru_backward_weight(
    void* err0_ptr,
    void* err1_ptr,
    void* layer_ptr,
    void* hidden_ptr,
    void* w_i_ptr, /// [r, z, n ]
    void* w_h_ptr, /// [r, z, n ]
    void* bias0_ptr,
    void* bias1_ptr,
    const int batch_size,
    const int input_size,
    const int hidden_size,
    const int sequence_length,
    const int layer_size,
    sycl::queue& Queue) {
  if (hidden_size > 0 && hidden_size <= 1024)
    gru_backward_weight_impl<m512k92n256_bpk>(
        err0_ptr,
        err1_ptr,
        layer_ptr,
        hidden_ptr,
        w_i_ptr, /// [r, z, n ]
        w_h_ptr, /// [r, z, n ]
        bias0_ptr,
        bias1_ptr,
        batch_size,
        input_size,
        hidden_size,
        sequence_length,
        layer_size,
        Queue);
  else
    assert(0); // Currently a few shapes are supported
}

} // namespace xetla
} // namespace xpu
