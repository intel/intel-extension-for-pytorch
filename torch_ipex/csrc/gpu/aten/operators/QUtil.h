#include <ATen/ATen.h>
#include <ATen/Config.h>
#include <ATen/NativeFunctions.h>
#include <ATen/cpp_custom_type_hack.h>

namespace at {
namespace AtenIpexTypeQuantizedXPU {

struct PackedConvWeightQDPCPP {
  Tensor weight;
  c10::optional<Tensor> bias;
};

struct PackedLinearWeightQDPCPP {
  Tensor weight;
  c10::optional<Tensor> bias;
};

} // namespace AtenIpexTypeQuantizedXPU
} // namespace at
