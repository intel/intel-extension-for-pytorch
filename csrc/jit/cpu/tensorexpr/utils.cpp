#include "utils.h"

#include <torch/csrc/jit/tensorexpr/exceptions.h>

namespace torch_ipex {
namespace jit {
namespace cpu {
namespace tensorexpr {

c10::MemoryFormat deduce_memory_format(
    c10::IntArrayRef strides,
    c10::IntArrayRef dims) {
  if (strides.size() == 4 && strides[3] == dims[1] && strides[1] == 1l) {
    return c10::MemoryFormat::ChannelsLast;
  }
  return c10::MemoryFormat::Contiguous;
}

c10::MemoryFormat deduce_memory_format(
    const std::vector<int64_t>& strides,
    const std::vector<int64_t>& dims) {
  return deduce_memory_format(
      c10::IntArrayRef(strides), c10::IntArrayRef(dims));
}

std::vector<at::Tensor> constructTensors(
    int64_t bufs_num,
    void** buf_data,
    int64_t* buf_ranks,
    int64_t* buf_dims,
    int64_t* buf_strides,
    int8_t* buf_dtypes) {
  std::vector<void*> buf_data_vec;
  std::vector<std::vector<int64_t>> buf_dims_vec;
  std::vector<std::vector<int64_t>> buf_strides_vec;
  std::vector<c10::ScalarType> buf_dtypes_vec;
  int64_t buf_dims_idx = 0;
  int64_t buf_strides_idx = 0;
  for (const auto i : c10::irange(bufs_num)) {
    buf_data_vec.push_back(buf_data[i]);
    buf_dims_vec.emplace_back();
    buf_strides_vec.emplace_back();
    for (const auto dim : c10::irange(buf_ranks[i])) {
      (void)dim;
      buf_dims_vec[i].push_back(buf_dims[buf_dims_idx++]);
      buf_strides_vec[i].push_back(buf_strides[buf_strides_idx++]);
    }
    buf_dtypes_vec.push_back(static_cast<c10::ScalarType>(buf_dtypes[i]));
  }

  std::vector<at::Tensor> tensors;
  for (const auto i : c10::irange(buf_data_vec.size())) {
    auto options = at::TensorOptions()
                       // NOLINTNEXTLINE
                       .dtype(buf_dtypes_vec[i])
                       .layout(at::kStrided)
                       .device(at::kCPU) // TODO: support GPUs too
                       .memory_format(deduce_memory_format(
                           // NOLINTNEXTLINE
                           buf_strides_vec[i],
                           // NOLINTNEXTLINE
                           buf_dims_vec[i]))
                       .requires_grad(false);
    auto tensor = at::from_blob(
        // NOLINTNEXTLINE
        buf_data_vec[i],
        buf_dims_vec[i],
        buf_strides_vec[i],
        options);
    tensors.emplace_back(tensor);
  }
  return tensors;
}

pytnnc::ExprHandle constant(const pytnnc::ArgValue& v) {
  if (auto s = c10::get_if<pytnnc::VarHandle>(&v)) {
    return *s;
  } else if (auto d = c10::get_if<double>(&v)) {
    return pytnnc::DoubleImm::make(*d);
  } else if (auto i = c10::get_if<int64_t>(&v)) {
    return pytnnc::LongImm::make(*i);
  } else if (auto b = c10::get_if<bool>(&v)) {
    return pytnnc::BoolImm::make(*b);
  } else if (c10::get_if<pytnnc::ArgNone>(&v)) {
    // This is just a placeholder so we don't throw.  None-handling
    // is operator-specific and should be handled properly in
    // the operator-specific lowering code.
    return pytnnc::IntImm::make(0);
  } else {
    throw pytnnc::unsupported_dtype(
        "Trying to convert unsupported dtype to constant");
  }
}

} // namespace tensorexpr
} // namespace cpu
} // namespace jit
} // namespace torch_ipex
