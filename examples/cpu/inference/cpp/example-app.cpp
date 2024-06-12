#include <torch/script.h>
#include <iostream>
#include <memory>

int main(int argc, const char* argv[]) {
  torch::jit::script::Module module;
  try {
    module = torch::jit::load(argv[1]);
  } catch (const c10::Error& e) {
    std::cerr << "error loading the model\n";
    return -1;
  }

  std::vector<torch::jit::IValue> inputs;
  torch::Tensor input = torch::rand({1, 3, 224, 224});
  inputs.push_back(input);

  at::Tensor output = module.forward(inputs).toTensor();
  std::cout << output.slice(/*dim=*/1, /*start=*/0, /*end=*/5) << std::endl;
  std::cout << "Execution finished" << std::endl;

  return 0;
}
