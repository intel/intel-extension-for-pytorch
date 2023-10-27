#pragma once

namespace sdp {

// The same definition as PyTorch
// We define here because head file in PyTorch is not exposed
enum class SDPBackend {
  error = -1,
  math = 0,
  flash_attention = 1,
  efficient_attention = 2
};

} // namespace sdp