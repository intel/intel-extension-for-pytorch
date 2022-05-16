#include "csrc/jit/cpu/passes/utils.h"

// Check if the memory format of the tensor is ChannelsLast(3d)
bool is_channelslast(c10::TensorType tensor) {
  TORCH_CHECK(tensor.dim().has_value());
  int64_t dim = tensor.dim().value();
  std::vector<int64_t> sizes(dim);
  std::vector<int64_t> strides(dim);
  for (int64_t i = 0; i < dim; ++i) {
    TORCH_CHECK(
        tensor.sizes()[i].has_value() && tensor.strides()[i].has_value());
    sizes[i] = tensor.sizes()[i].value();
    strides[i] = tensor.strides()[i].value();
  }
  return (
      c10::is_channels_last_strides_2d(sizes, strides) ||
      c10::is_channels_last_strides_3d(sizes, strides));
}

// Check if the memory format of the tensor is Contiguous
bool is_contiguous(c10::TensorTypePtr tensor) {
  auto tensor_contiguous = tensor->contiguous();
  bool is_contiguous = tensor_contiguous->strides() == tensor->strides();
  return is_contiguous;
}
