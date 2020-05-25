#include <torch/script.h>
#include "ExtendOPs.h"

static auto registry =
    torch::RegisterOperators()
       .op("torch_ipex::linear",
          [](const at::Tensor& input, const at::Tensor& weight, const at::Tensor& bias) {
          return torch_ipex::AtenIpexTypeExt::linear(input, weight, bias);
        });


