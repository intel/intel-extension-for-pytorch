#include "torch_dpcpp/csrc/aten_ipex_type.h"

#include <ATen/Context.h>

#include <mutex>

#include "torch_dpcpp/csrc/aten_ipex_type_default.h"
#include "torch_dpcpp/csrc/version.h"

namespace torch_ipex {

namespace {

void AtenInitialize() {
  RegisterAtenTypeFunctions();
}

}  // namespace

void AtenIpexType::InitializeAtenBindings() {
  static std::once_flag once;
  std::call_once(once, []() { AtenInitialize(); });
}

} // namespace torch_ipe
