#include <iostream>
#include <memory>
#include <torch/script.h>
#include <c10/xpu/XPUStream.h>
#include <ATen/ATen.h>
#include <CL/sycl.hpp>

using namespace sycl;

int main(int argc, const char* argv[]) {
  torch::jit::script::Module module;
  try {
    module = torch::jit::load(argv[1]);
  }
  catch (const c10::Error& e) {
    std::cerr << "error loading the model\n";
    return -1;
  }
  std::cout << "load model done " << std::endl;
  module.to(at::kXPU);

  std::vector<torch::jit::IValue> inputs;
  c10::xpu::XPUStream stream = c10::xpu::getCurrentXPUStream();
  auto options = at::TensorOptions().dtype(at::kFloat).device(stream.device());
  float *input_ptr = malloc_device<float>(224 * 224 * 3, stream);
  auto input = torch::from_blob(
      input_ptr,
      {1, 3, 224, 224},
      nullptr,
      options,
      {stream.device()});
  std::cout << "input tensor created from usm " << std::endl;
  inputs.push_back(input);

  at::IValue output = module.forward(inputs);
  torch::Tensor output_tensor;
  output_tensor = output.toTensor();
  std::cout << output_tensor.slice(/*dim=*/1, /*start=*/0, /*end=*/5) << std::endl;
  std::cout << "Execution finished" << std::endl;

  return 0;
}
