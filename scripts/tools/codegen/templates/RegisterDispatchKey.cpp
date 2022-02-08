// required for old g++ to compile PRId64 macros, see
// https://github.com/pytorch/pytorch/issues/3571
// for context
#ifndef __STDC_FORMAT_MACROS
#define __STDC_FORMAT_MACROS
#endif

// clang-format off

// ${generated_comment}

#include <c10/core/TensorImpl.h>
#include <c10/core/Allocator.h>
#include <ATen/DeviceGuard.h>
#include <ATen/NativeFunctions.h>
#include <ATen/NamedTensorUtils.h>
#include <ATen/Utils.h>
#include <ATen/WrapDimUtils.h>
#include <ATen/Dispatch.h>
#include <c10/util/ExclusivelyOwned.h>
#include <c10/util/Half.h>
#include <c10/core/TensorImpl.h>
#include <c10/core/UndefinedTensorImpl.h>
#include <c10/util/Optional.h>
#include <ATen/Tensor.h>
#include <ATen/Functions.h>
#include <ATen/native/Resize.h>

#include <cstddef>
#include <functional>
#include <memory>
#include <utility>

#include <ATen/Config.h>
#include <ATen/core/op_registration/adaption.h>
#include <torch/library.h>
#include <intrinsic/ipex_intrinsic.h>
$extra_cuda_headers
$external_backend_headers
$namespaced_headers

namespace at {

// NB: TORCH_LIBRARY_IMPL must be in an anonymous namespace to avoid
// ambiguity with conflicting identifiers that may have been defined in
// at namespace already.
namespace {

class IpexSimpleTrace {
public:
  IpexSimpleTrace(const char * name) : _name(name) {
    indent++;
    gindex++;
    print_indent();
    printf("step into %s (%d)\n", _name, gindex);
    fflush(stdout);
  }
  ~IpexSimpleTrace() {
    print_indent();
    printf("step out of %s\n", _name);
    fflush(stdout);
    indent--;
  }
private:
  void print_indent() {
    for (int i = 0; i < indent*2; ++i) {
      printf(" ");
    }
  }
  static int indent;
  static int gindex;
  const char * _name;
};

int IpexSimpleTrace::indent = -1;
int IpexSimpleTrace::gindex = -1;

${dispatch_helpers}

${dispatch_anonymous_definitions}

TORCH_LIBRARY_IMPL(aten, ${DispatchKey}, m) {
  ${dispatch_registrations}
}

} // anonymous namespace

} // namespace at
