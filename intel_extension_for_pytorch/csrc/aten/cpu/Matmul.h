#pragma once

#include <ATen/Tensor.h>
#include <torch/csrc/autograd/FunctionsManual.h>
#include <torch/csrc/autograd/custom_function.h>
#include <vector>

namespace torch {
namespace autograd {
namespace generated {

namespace ipex {
using IndexRange = std::pair<size_t, size_t>;
// A simple way to imperatively compute index ranges for slots
// that have been flattened
struct IndexRangeGenerator {
  IndexRange range(size_t range_size) {
    i += range_size;
    return {i - range_size, i};
  }
  size_t size() {
    return i;
  }

 private:
  size_t i = 0;
};

struct TORCH_API BmmBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "BmmBackward0";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    mat2_.reset_data();
  }

  SavedVariable self_;
  SavedVariable mat2_;
};

} // namespace ipex
} // namespace generated
} // namespace autograd
} // namespace torch

namespace torch_ipex {
namespace cpu {

at::Tensor matmul_cpu(const at::Tensor& self, const at::Tensor& mat2);
at::Tensor& matmul_out_cpu(
    const at::Tensor& self,
    const at::Tensor& mat2,
    at::Tensor& out);
at::Tensor matmul_onednn(const at::Tensor& self, const at::Tensor& mat2);
at::Tensor& matmul_onednn(
    at::Tensor& out,
    const at::Tensor& self,
    const at::Tensor& mat2);
} // namespace cpu
} // namespace torch_ipex
