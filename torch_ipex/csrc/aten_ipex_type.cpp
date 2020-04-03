#include "aten_ipex_type.h"

#include <ATen/Context.h>

#include <mutex>

#include "aten_ipex_type_default.h"
#include "aten_ipex_sparse_type_default.h"
#include "version.h"

namespace torch_ipex {

namespace {

void AtenInitialize() {
  RegisterAtenTypeFunctions();
  RegisterAtenTypeSparseFunctions();
}

}  // namespace

void AtenIpexType::InitializeAtenBindings() {
  static std::once_flag once;
  std::call_once(once, []() { AtenInitialize(); });
}

} // namespace torch_ipe
