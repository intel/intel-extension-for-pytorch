#include <torch/script.h>
#include <torch/custom_class.h>
#include "csrc/utils/ipex_op_profile.h"
#include "csrc/utils/library.h"
#include "LSTM.h"

namespace torch_ipex {
namespace cpu {

DEFINE_DISPATCH(optimized_lstm_batch_kernel_stub);

std::tuple<at::Tensor, at::Tensor, at::Tensor> OptimizedLstmBatch(
    const at::Tensor& input,
    const std::vector<at::Tensor>& hx,
    const std::vector<at::Tensor>& params,
    int64_t input_size,
    int64_t hidden_size,
    int64_t proj_size,
    bool has_biases,
    int64_t num_layers,
    bool bidirectional,
    bool batch_first) {
  return optimized_lstm_batch_kernel_stub(
      kCPU, input, hx, params, input_size, hidden_size, proj_size,
      has_biases, num_layers, bidirectional, batch_first);
}

DEFINE_DISPATCH(optimized_lstm_packed_kernel_stub);

std::tuple<at::Tensor, at::Tensor, at::Tensor> OptimizedLstmPacked(
    const at::Tensor& input,
    const at::Tensor& batch_sizes,
    const std::vector<at::Tensor>& hx,
    const std::vector<at::Tensor>& params,
    int64_t input_size,
    int64_t hidden_size,
    int64_t proj_size,
    bool has_biases,
    int64_t num_layers,
    bool bidirectional) {
  return optimized_lstm_packed_kernel_stub(
      kCPU, input, batch_sizes, hx, params, input_size, hidden_size,
      proj_size, has_biases, num_layers, bidirectional);
}

} // namespace cpu
} // namespace torch_ipex

namespace torch_ipex {
std::tuple<at::Tensor, at::Tensor, at::Tensor> optimized_lstm_batch(
    const at::Tensor& input,
    const std::vector<at::Tensor>& hx,
    const std::vector<at::Tensor>& params,
    int64_t input_size,
    int64_t hidden_size,
    int64_t proj_size,
    bool has_biases,
    int64_t num_layers,
    bool bidirectional,
    bool batch_first) {
  IPEX_RECORD_FUNCTION("optimized_lstm_batch", c10::ArrayRef<c10::IValue>({}));

#if defined(IPEX_DISP_OP)
  printf("optimized_lstm_batch\n");
#endif
  auto result = cpu::OptimizedLstmBatch(
      input,
      hx,
      params,
      input_size,
      hidden_size,
      proj_size,
      has_biases,
      num_layers,
      bidirectional,
      batch_first);
  return result;
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> optimized_lstm_packed(
    const at::Tensor& input,
    const at::Tensor& batch_sizes,
    const std::vector<at::Tensor>& hx,
    const std::vector<at::Tensor>& params,
    int64_t input_size,
    int64_t hidden_size,
    int64_t proj_size,
    bool has_biases,
    int64_t num_layers,
    bool bidirectional) {
  IPEX_RECORD_FUNCTION("optimized_lstm_packed", c10::ArrayRef<c10::IValue>({}));

#if defined(IPEX_DISP_OP)
  printf("optimized_lstm_packed\n");
#endif
  auto result = cpu::OptimizedLstmPacked(
      input,
      batch_sizes,
      hx,
      params,
      input_size,
      hidden_size,
      proj_size,
      has_biases,
      num_layers,
      bidirectional);
  return result;
}
} // namespace torch_ipex

namespace {
TORCH_LIBRARY_FRAGMENT(torch_ipex, m) {
  m.def(
      "optimized_lstm_batch(Tensor input, Tensor[] hx, Tensor[] params, int "
      "input_size, int hidden_size, int proj_size, bool has_biases, int num_layers, bool "
      "bidirectional, bool batch_first) -> (Tensor, Tensor, Tensor)",
      torch_ipex::optimized_lstm_batch);
  m.impl("optimized_lstm_batch", c10::DispatchKey::CPU, torch_ipex::optimized_lstm_batch);
  m.def(
      "optimized_lstm_packed(Tensor input, Tensor batch_sizes, Tensor[] hx, Tensor[] params, int "
      "input_size, int hidden_size, int proj_size, bool has_biases, int num_layers, bool "
      "bidirectional) -> (Tensor, Tensor, Tensor)",
      torch_ipex::optimized_lstm_packed);
  m.impl("optimized_lstm_packed", c10::DispatchKey::CPU, torch_ipex::optimized_lstm_packed);
}

} // namespace