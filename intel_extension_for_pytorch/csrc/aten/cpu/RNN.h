#include <ATen/Tensor.h>
#include <torch/csrc/autograd/custom_function.h>

#include "csrc/cpu/ideep/ideep.hpp"

#include <vector>

namespace torch_ipex {
namespace cpu {

class IPEXLSTMOp : public torch::autograd::Function<IPEXLSTMOp> {
 public:
  // forward function without autograd overhead, will go this way when only do
  // forward
  static std::vector<at::Tensor> _forward(
      const at::Tensor& input,
      const at::Tensor& w0,
      const at::Tensor& w1,
      const at::Tensor& w2,
      const at::Tensor& w3,
      const at::Tensor& hx_,
      const at::Tensor& cx_,
      bool reverse,
      at::IntArrayRef batch_sizes,
      int64_t mode,
      int64_t hidden_size,
      int64_t num_layers,
      bool has_biases,
      bool bidirectional,
      bool batch_first,
      bool train);
  static std::vector<at::Tensor> forward(
      torch::autograd::AutogradContext* ctx,
      const at::Tensor& input,
      const at::Tensor& w0,
      const at::Tensor& w1,
      const at::Tensor& w2,
      const at::Tensor& w3,
      const at::Tensor& hx_,
      const at::Tensor& cx_,
      bool reverse,
      at::IntArrayRef batch_sizes,
      int64_t mode,
      int64_t hidden_size,
      int64_t num_layers,
      bool has_biases,
      bool bidirectional,
      bool batch_first,
      bool train);

  static torch::autograd::tensor_list backward(
      torch::autograd::AutogradContext* ctx,
      torch::autograd::tensor_list grad_outputs);
};

std::vector<at::Tensor> ipex_lstm_layer(
    const at::Tensor& input,
    const at::Tensor& weight0,
    const at::Tensor& weight1,
    const at::Tensor& weight2,
    const at::Tensor& weight3,
    const at::Tensor& hx_,
    const at::Tensor& cx_,
    bool reverse,
    at::IntArrayRef batch_sizes,
    int64_t mode,
    int64_t hidden_size,
    int64_t num_layers,
    bool has_biases,
    bool bidirectional,
    bool batch_first,
    bool train);

std::vector<at::Tensor> ipex_lstm_layer_backward(
    const at::Tensor& input,
    const at::Tensor& weight0,
    const at::Tensor& weight1,
    const at::Tensor& weight2,
    const at::Tensor& weight3,
    const at::Tensor& hx_,
    const at::Tensor& cx_tmp,
    const at::Tensor& output,
    const at::Tensor& hy_,
    const at::Tensor& cy_,
    const at::Tensor& grad_output,
    const at::Tensor& grad_hy,
    const at::Tensor& grad_cy,
    bool reverse,
    int64_t mode,
    int64_t hidden_size,
    int64_t num_layers,
    bool has_biases,
    bool train,
    bool bidirectional,
    at::IntArrayRef batch_sizes,
    bool batch_first,
    const at::Tensor& workspace);

std::vector<at::Tensor> ipex_lstm_layer_forward(
    const at::Tensor& input,
    const at::Tensor& w1,
    const at::Tensor& w2,
    const at::Tensor& w3,
    const at::Tensor& w4,
    const at::Tensor& hx_,
    const at::Tensor& cx_,
    bool reverse,
    at::IntArrayRef batch_sizes,
    int64_t mode,
    int64_t hidden_size,
    int64_t num_layers,
    bool has_biases,
    bool bidirectional,
    bool batch_first,
    bool train);

static std::tuple<at::Tensor, at::Tensor, at::Tensor> ipex_lstm(
    const at::Tensor& input,
    std::vector<at::Tensor> hx,
    std::vector<at::Tensor> params,
    bool has_biases,
    int64_t num_layers,
    double dropout_p,
    bool train,
    bool bidirectional,
    bool batch_first);

} // namespace cpu
} // namespace torch_ipex
