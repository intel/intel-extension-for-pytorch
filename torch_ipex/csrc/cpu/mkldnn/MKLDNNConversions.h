#pragma once

#include <ATen/ATen.h>

namespace torch_ipex { namespace cpu {

at::Tensor mkldnn_to_dense(const at::Tensor & self, c10::optional<at::ScalarType> dtype=c10::nullopt);

}}