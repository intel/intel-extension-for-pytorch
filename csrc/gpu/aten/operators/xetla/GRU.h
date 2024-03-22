#pragma once
#include <sycl/sycl.hpp>
#include "xetla_kernel_api.h"

namespace torch_ipex::xpu::xetla {

XETLA_KERNEL_API cgfs_t gru_forward(
    void* layer_ptr,
    void* hx_ptr,
    void* i_weights,
    void* h_weights,
    void* i_biases,
    void* h_biases,
    void* layer_out_ptr,
    void* hidden_out_ptr,
    void* mask_ptr,
    void* dropout_buffer,
    void* workspace_ptr,
    void* reset_gate_ptr,
    void* input_gate_ptr,
    void* new_gate_ptr,
    void* hgate_2_ptr,
    const int batch_size,
    const int input_size,
    const int hidden_size,
    const int sequence_size,
    const int layer_size);

XETLA_KERNEL_API cgfs_t gru_backward_data(
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
    const float dropout);

XETLA_KERNEL_API cgfs_t gru_backward_weight(
    void* err0_ptr,
    void* err1_ptr,
    void* layer_ptr,
    void* hidden_ptr,
    void* w_i_ptr,
    void* w_h_ptr,
    void* bias0_ptr,
    void* bias1_ptr,
    const int batch_size,
    const int input_size,
    const int hidden_size,
    const int sequence_length,
    const int layer_size);
} // namespace torch_ipex::xpu::xetla
