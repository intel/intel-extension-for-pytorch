#include <ATen/Tensor.h>

namespace torch_ipex {

class AtenIpexTypeDefault {
public:
  static at::Tensor add(const at::Tensor & self, const at::Tensor & other, at::Scalar alpha);
  static at::Tensor ones(at::IntArrayRef size, const at::TensorOptions & options);
  static at::Tensor empty(at::IntArrayRef size, const at::TensorOptions & options, c10::optional<at::MemoryFormat> memory_format);
};

void RegisterAtenTypeFunctions();
} // namespace torch_ipe