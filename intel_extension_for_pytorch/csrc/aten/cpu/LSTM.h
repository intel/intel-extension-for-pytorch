#pragma once
#include <ATen/Tensor.h>
#include <csrc/dyndisp/DispatchStub.h>
#include <immintrin.h>
#include <c10/core/impl/alloc_cpu.h>


namespace torch_ipex {
namespace cpu {

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
    bool batch_first);

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
    bool bidirectional);

using optimized_lstm_batch_kernel_fn = std::tuple<at::Tensor, at::Tensor, at::Tensor> (*)(
    const at::Tensor& input,
    const std::vector<at::Tensor>& hx,
    const std::vector<at::Tensor>& params,
    int64_t input_size,
    int64_t hidden_size,
    int64_t proj_size,
    bool has_biases,
    int64_t num_layers,
    bool bidirectional,
    bool batch_first);
DECLARE_DISPATCH(optimized_lstm_batch_kernel_fn, optimized_lstm_batch_kernel_stub);

using optimized_lstm_packed_kernel_fn = std::tuple<at::Tensor, at::Tensor, at::Tensor> (*)(
    const at::Tensor& input,
    const at::Tensor& batch_sizes,
    const std::vector<at::Tensor>& hx,
    const std::vector<at::Tensor>& params,
    int64_t input_size,
    int64_t hidden_size,
    int64_t proj_size,
    bool has_biases,
    int64_t num_layers,
    bool bidirectional);
DECLARE_DISPATCH(optimized_lstm_packed_kernel_fn, optimized_lstm_packed_kernel_stub);

} // namespace cpu
} // namespace torch_ipex
