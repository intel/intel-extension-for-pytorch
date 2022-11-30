#pragma once

#include <tensor/Context.h>

namespace at {
namespace AtenIpexTypeXPU {

struct DPCPPTensorContext;

at::Tensor empty_opaque_tensor(
    DPCPPTensorContext::Meta meta,
    const TensorOptions& options,
    c10::optional<MemoryFormat> optional_memory_format);

at::Tensor empty_opaque_qtensor(
    DPCPPTensorContext::Meta meta,
    c10::optional<MemoryFormat> optional_memory_format,
    QuantizerPtr quantizer);

at::Tensor to_plain_if_needed(const Tensor& tensor);

at::Tensor to_plain_if_needed_(const Tensor& tensor);

std::vector<at::Tensor> to_plain_if_needed(TensorList tensor);

std::vector<at::Tensor> to_plain_if_needed(MaterializedITensorListRef tensors);

} // namespace AtenIpexTypeXPU
} // namespace at
