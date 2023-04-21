#include "../../GRU.h"

#include "forward.h"

namespace xpu {
namespace xetla {

void gru_forward(
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
    const int layer_size,
    sycl::queue& Queue) {
  if (hidden_size <= 256)
    gru_forward_impl<m512k92n256_fwd>(
        layer_ptr,
        hx_ptr,
        i_weights,
        h_weights,
        i_biases,
        h_biases,
        layer_out_ptr,
        hidden_out_ptr,
        mask_ptr,
        dropout_buffer,
        workspace_ptr,
        reset_gate_ptr,
        input_gate_ptr,
        new_gate_ptr,
        hgate_2_ptr,
        batch_size,
        input_size,
        hidden_size,
        sequence_size,
        layer_size,
        Queue);
  else if (hidden_size <= 1024)
    gru_forward_impl<m512k379n681_fwd>(
        layer_ptr,
        hx_ptr,
        i_weights,
        h_weights,
        i_biases,
        h_biases,
        layer_out_ptr,
        hidden_out_ptr,
        mask_ptr,
        dropout_buffer,
        workspace_ptr,
        reset_gate_ptr,
        input_gate_ptr,
        new_gate_ptr,
        hgate_2_ptr,
        batch_size,
        input_size,
        hidden_size,
        sequence_size,
        layer_size,
        Queue);
  else
    assert(0); // Currently need add shape conig manually
}

} // namespace xetla
} // namespace xpu
