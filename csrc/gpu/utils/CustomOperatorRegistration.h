#pragma once
#include <ATen/ATen.h>
#include <ATen/native/quantized/PackedParams.h>
#include <oneDNN/oneDNN.h>
#include <oneapi/dnnl/dnnl.hpp>
#include "aten/tensor/OpaqueTensorFactories.h"
#include "torch/library.h"

namespace at {
namespace AtenIpexTypeXPU {

namespace {
template <typename T, int N>
struct TypeSelector {
  template <typename... Args>
  void extract_type(Args... args) {
    return;
  }

  template <typename... Args>
  void extract_type(T& type, Args... args) {
    container_.push_back(type);
    extract_type(args...);
  }

  template <typename U, typename... Args>
  void extract_type(U type, Args... args) {
    extract_type(args...);
  }

  at::ArrayRef<T> retrive_types() {
    return at::ArrayRef<T>(container_.begin(), container_.end());
  }

  at::SmallVector<T, N> container_;
};
} // namespace

template <
    typename Signature,
    Signature* Func,
    bool to_plain,
    bool device_guard,
    typename Ret,
    typename TypeList>
struct IpexFunctionWarpper_ {};

template <
    typename Signature,
    Signature* Func,
    bool to_plain,
    typename Ret,
    typename... Args>
struct IpexFunctionWarpper_<
    Signature,
    Func,
    to_plain,
    false,
    Ret,
    guts::typelist::typelist<Args...>> {
  static Ret call(Args... args) {
    if constexpr (to_plain) {
      TypeSelector<at::Tensor, sizeof...(args)> selector;
      selector.extract_type(args...);
      std::for_each(
          selector.retrive_types().begin(),
          selector.retrive_types().end(),
          AtenIpexTypeXPU::to_plain_if_needed_);
    }
    return (*Func)(args...);
  }
};

template <
    typename Signature,
    Signature* Func,
    bool to_plain,
    typename Ret,
    typename... Args>
struct IpexFunctionWarpper_<
    Signature,
    Func,
    to_plain,
    true,
    Ret,
    guts::typelist::typelist<Args...>> {
  static Ret call(Args... args) {
    TypeSelector<at::Tensor, sizeof...(args)> selector;
    selector.extract_type(args...);
    const OptionalDeviceGuard dev_guard(device_of(selector.retrive_types()));
    if constexpr (to_plain) {
      std::for_each(
          selector.retrive_types().begin(),
          selector.retrive_types().end(),
          AtenIpexTypeXPU::to_plain_if_needed_);
    }
    return (*Func)(args...);
  }
};

template <typename Signature, Signature* Func, bool to_plain, bool device_guard>
struct IpexFunctionWarpper {
  using type = IpexFunctionWarpper_<
      Signature,
      Func,
      to_plain,
      device_guard,
      typename guts::function_traits<Signature>::return_type,
      typename guts::function_traits<Signature>::parameter_types>;
};
template <typename Func>
c10::FunctionSchema constructFunctionSchema(const char* name, Func&& func) {
  std::unique_ptr<c10::FunctionSchema> infer_schema =
      c10::detail::inferFunctionSchemaFromFunctor<std::decay_t<Func>>();
  auto parse_name = torch::jit::parseSchemaOrName(name);
  c10::OperatorName op_name = std::move(parse_name).left();
  c10::FunctionSchema s = infer_schema->cloneWithName(
      std::move(op_name.name), std::move(op_name.overload_name));
  s.setAliasAnalysis(c10::AliasAnalysisKind::CONSERVATIVE);
  return s;
}
/*
*************************************************
IPEX_OP_REGISTER_OVERLOAD
This macro is used to register the functions which have its overload version.
Considering that overload functions tends to have same function name and
different signatures, function schema and signature should be explicitly
specified in this scenario. Here's one example: For two overload function
mul_add, here is its declaration:

==========================================================================
  Tensor mul_add(Tensor tensor1, Tensor mul, Tensor add, Scalar scale);
--------------------------------------------------------------------------
  Tensor mul_add(Tensor tensor1, Scalar mul, Tensor add, Scalar scale);
==========================================================================

The macro should be written in below way to register these two different
functions:

==========================================================================
  IPEX_OP_REGISTER_OVERLOAD(
    "mul_add(Tensor tensor1, Tensor mul, Tensor add, Scalar scale) -> Tensor",
    "mul_add",
    Tensor (Tensor, Tensor, Tensor, Scalar),
    mul_add);
---------------------------------------------------------
  IPEX_OP_REGISTER_OVERLOAD(
    "mul_add(Tensor tensor1, Scalar mul, Tensor add, Scalar scale) -> Tensor",
    "mul_add",
    Tensor (Tensor, Scalar, Tensor, Scalar),
    mul_add);
===========================================================================
*/
#define IPEX_OP_REGISTER_OVERLOAD(SCHEMA, NAME, SIGNATURE, Func)             \
  m.def(TORCH_SELECTIVE_SCHEMA("torch_ipex::" SCHEMA));                      \
  m.impl(                                                                    \
      TORCH_SELECTIVE_NAME("torch_ipex::" NAME),                             \
      &AtenIpexTypeXPU::IpexFunctionWarpper<SIGNATURE, &Func, false, true>:: \
          type::call);

#define IPEX_OP_REGISTER_NEED_PLAIN_OVERLOAD(SCHEMA, NAME, SIGNATURE, Func) \
  m.def(TORCH_SELECTIVE_SCHEMA("torch_ipex::" SCHEMA));                     \
  m.impl(                                                                   \
      TORCH_SELECTIVE_NAME("torch_ipex::" NAME),                            \
      &AtenIpexTypeXPU::IpexFunctionWarpper<SIGNATURE, &Func, true, true>:: \
          type::call);

/*
***************************************************************************
IPEX_OP_REGISTER | IPEX_OP_REGISTER_TO_PLAIN
This macro is used to register ops into torch_ipex library. Through this macro,
function schema and signature will automatically be infered from function
prototype. However, it is worth to note that this macro will not works on
overload functions(see IPEX_OP_REGISTER_OVERLOAD). Here is some examples for
register ipex operators:

For operator
===========================================================================
Tensor mul_add(Tensor tensor1, Tensor mul, Tensor add, Scalar scale);
===========================================================================

We can register this op like follow:
===========================================================================
IPEX_LIBRARY_FRAGMENT() {
  IPEX_OP_REGISTER("mul_add", mul_add);
}
And if this op does not support oneDNN's block format memory layout for tensor.
It would be necessary for developer to register it specificly by adopting the
macro IPEX_OP_REGISTER_NEED_PLAIN. In this way, all the tensor passed to this
operator will automatically convert to normal tensor layout when execution.

*/
#define IPEX_OP_REGISTER(NAME, Func)                                      \
  m.def(AtenIpexTypeXPU::constructFunctionSchema(NAME, Func));            \
  m.impl(                                                                 \
      TORCH_SELECTIVE_NAME("torch_ipex::" NAME),                          \
      &AtenIpexTypeXPU::                                                  \
          IpexFunctionWarpper<decltype(Func), &Func, false, true>::type:: \
              call);

#define IPEX_OP_REGISTER_NEED_PLAIN(NAME, Func)                \
  m.def(AtenIpexTypeXPU::constructFunctionSchema(NAME, Func)); \
  m.impl(                                                      \
      TORCH_SELECTIVE_NAME("torch_ipex::" NAME),               \
      &AtenIpexTypeXPU::                                       \
          IpexFunctionWarpper<decltype(Func), &Func, true, true>::type::call);

#define IPEX_LIBRARY_FRAGMENT() TORCH_LIBRARY_FRAGMENT(torch_ipex, m)

} // namespace AtenIpexTypeXPU
} // namespace at