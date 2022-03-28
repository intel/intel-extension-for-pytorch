#pragma once

#include <ATen/Tensor.h>
#include <torch/csrc/autograd/custom_function.h>
#include <vector>

namespace torch {
namespace autograd {
namespace generated {
struct TORCH_API MmBackward0 : public torch::autograd::TraceableFunction {
  using TraceableFunction::TraceableFunction;
  torch::autograd::variable_list apply(
      torch::autograd::variable_list&& grads) override;
  std::string name() const override {
    return "MmBackward0";
  }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    mat2_.reset_data();
  }

  torch::autograd::SavedVariable self_;
  std::vector<int64_t> mat2_sizes;
  std::vector<int64_t> mat2_strides;
  std::vector<int64_t> self_sizes;
  std::vector<int64_t> self_strides;
  torch::autograd::SavedVariable mat2_;
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
