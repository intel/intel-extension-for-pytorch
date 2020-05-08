#include "aten_ipex_type.h"

#include <ATen/Context.h>

#include <mutex>

#include "cpu/DenseOPs.h"
#include "cpu/SparseOPs.h"
#include "version.h"

namespace torch_ipex {

namespace {

void AtenInitialize() {
  cpu::RegisterIpexDenseOPs();
  cpu::RegisterIpexSparseOPs();
}

}  // namespace

void AtenIpexType::InitializeAtenBindings() {
  static std::once_flag once;
  std::call_once(once, []() { AtenInitialize(); });
}

} // namespace torch_ipe
