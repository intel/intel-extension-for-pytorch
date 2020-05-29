#include <torch/script.h>
#include "ExtendOPs.h"

static auto registry =
    torch::RegisterOperators()
       .op("torch_ipex::linear", &torch_ipex::AtenIpexTypeExt::linear)
       .op("torch_ipex::max_pool2d", [](const at::Tensor& self, c10::List<int64_t> kernel_size,
          c10::List<int64_t> stride, c10::List<int64_t> padding, c10::List<int64_t> dilation, bool ceil_mode=false){
          return torch_ipex::AtenIpexTypeExt::max_pooling(self, kernel_size.vec(), stride.vec(), padding.vec(), dilation.vec(), ceil_mode);
        })
       .op("torch_ipex::adaptive_avg_pool2d", [](const at::Tensor&self, c10::List<int64_t> output_size) {
          return torch_ipex::AtenIpexTypeExt::adaptive_avg_pool2d(self, output_size.vec());
        });

