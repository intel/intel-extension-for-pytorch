#pragma once
#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <c10/util/intrusive_ptr.h>
#include <c10/core/UndefinedTensorImpl.h>
#include <c10/core/impl/LocalDispatchKeySet.h>
#include <torch/library.h>
#include <torch/csrc/jit/frontend/tracer.h>
#include "utils.h"
#include "autocast_mode.h"

namespace torch_ipex {
namespace autocast {
namespace verbose {

using at::IntArrayRef;
using at::Tensor;
using at::TensorList;
using namespace c10;

#if defined(ENABLE_AUTOCAST_VERBOSE)
extern thread_local string current_op_name;
string get_current_op_name();
void set_current_op_name(const string& name);

class OpNameGuard{
public:
  explicit OpNameGuard(const string& name)
    : previor_op_name(get_current_op_name()) {
    set_current_op_name(name);
  }

  ~OpNameGuard(){
      set_current_op_name(previor_op_name);
  }

  OpNameGuard() = delete;
  OpNameGuard(const OpNameGuard &) = delete;
  OpNameGuard &operator=(const OpNameGuard &) = delete;
  OpNameGuard(const OpNameGuard &&) = delete;
  OpNameGuard &operator=(const OpNameGuard &&) = delete;

private:
  const string previor_op_name;
};

inline void autocast_verbose(at::ScalarType dtype, const Tensor& arg) {
  std::cout << "autocast verbose, operation name: " << current_op_name
            << ", tensor shape: " << arg.sizes()
            << ", src dtype: " << scalarTypeName(arg.scalar_type())
            << ", dst dtype: " << scalarTypeName(dtype) << std::endl;
  return;
}
#endif

}
}
}