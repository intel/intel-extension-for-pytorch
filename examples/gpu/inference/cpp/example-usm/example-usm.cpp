#include <iostream>
#include <memory>
#include <torch/script.h>
#include <ipex.h>
#include <CL/sycl.hpp>

using namespace sycl;
using namespace xpu::dpcpp;

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
  queue sycl_queue=queue(gpu_selector());
  float *input_ptr = malloc_device<float>(224 * 224 * 3, sycl_queue);
  auto input = fromUSM(input_ptr, at::ScalarType::Float, {1, 3, 224, 224}, c10::nullopt, -1).to(at::kXPU);
  std::cout << "input tensor created from usm " << std::endl;
  inputs.push_back(input);

  at::IValue output = module.forward(inputs);
  torch::Tensor output_tensor;
  output_tensor = output.toTensor();
  std::cout << output_tensor.slice(/*dim=*/1, /*start=*/0, /*end=*/5) << std::endl;

  return 0;
}
