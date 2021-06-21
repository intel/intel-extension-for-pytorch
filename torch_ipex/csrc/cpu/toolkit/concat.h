#pragma once
#include <ATen/Tensor.h>
#include <ATen/Dispatch.h>
#include <ATen/Parallel.h>
#include <c10/core/ScalarType.h>

namespace toolkit {
  at::Tensor concat_all_continue(std::vector<at::Tensor> tensors, int64_t dim);
}
