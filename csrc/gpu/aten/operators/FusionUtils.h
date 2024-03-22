#pragma once
#include <oneDNN/oneDNN.h>

namespace at {
namespace AtenIpexTypeXPU {
torch_ipex::xpu::oneDNN::Attr unary_attr_with_arg(
    c10::string_view unary,
    torch::List<c10::optional<at::Scalar>> scalars,
    c10::optional<c10::string_view> algorithm,
    torch_ipex::xpu::oneDNN::Attr attr);

torch_ipex::xpu::oneDNN::Attr string_to_unary_attr(
    torch_ipex::xpu::oneDNN::Attr attr);

torch_ipex::xpu::oneDNN::Attr construct_unary_attr(
    c10::string_view unary,
    torch::List<c10::optional<at::Scalar>> scalars,
    c10::optional<c10::string_view> algorithm,
    torch_ipex::xpu::oneDNN::Attr attr);

torch_ipex::xpu::oneDNN::Attr construct_binary_attr(
    c10::string_view binary,
    c10::optional<at::Scalar> alpha,
    const Tensor& other,
    torch_ipex::xpu::oneDNN::Attr attr);

} // namespace AtenIpexTypeXPU
} // namespace at
