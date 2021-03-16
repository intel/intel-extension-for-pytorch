#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>

#include <c10/util/intrusive_ptr.h>
#include <c10/core/impl/LocalDispatchKeySet.h>
#include <torch/library.h>
#include <torch/csrc/jit/frontend/tracer.h>

#include <iostream>
#include <exception>
#include "autocast_mode.h"

namespace torch_ipex {
namespace autocast {

namespace {
using weakref_type = c10::weak_intrusive_ptr<TensorImpl, UndefinedTensorImpl>;
using val_type = std::tuple<weakref_type, Tensor>;
thread_local std::unordered_map<TensorImpl *, val_type> cached_casts;

thread_local int nesting = 0;

thread_local int current_target_dtype_priority = FP32_DTYPE_PRIORITY;
}  // namespace

std::map<at::ScalarType, int> dtype_priority_map = {
  {at::kChar, INT8_DTYPE_PRIORITY},
  {at::kBFloat16, BF16_DTYPE_PRIORITY},
  {at::kFloat, FP32_DTYPE_PRIORITY},
};

std::map<int, at::ScalarType> inv_dtype_priority_map = flip_map<at::ScalarType>(dtype_priority_map);

bool is_autocast_enabled() {
  return c10::impl::tls_is_dispatch_key_included(c10::DispatchKey::AutocastCPU);
}

void set_autocast_enabled(bool new_enabled) {
  c10::impl::tls_set_dispatch_key_included(c10::DispatchKey::AutocastCPU,
                                           new_enabled);
}

at::ScalarType get_autocast_dtype() {
  return inv_dtype_priority_map[current_target_dtype_priority];
}

void set_autocast_dtype(at::ScalarType dtype) {
  current_target_dtype_priority = dtype_priority_map[dtype];
}

int autocast_increment_nesting() { return ++nesting; }

int autocast_decrement_nesting() { return --nesting; }

void clear_autocast_cache() { cached_casts.clear(); }

Tensor cpu_cached_cast(at::ScalarType to_type, const Tensor& arg) {
  if (is_eligible_cpu(arg) && (arg.scalar_type() != to_type)) {
    bool can_try_cache =
        (to_type == at::kBFloat16 && arg.scalar_type() == at::kFloat &&
        arg.requires_grad() && arg.is_leaf() && !arg.is_view() && !torch::jit::tracer::isTracing()); //Leslie Disable cache when we use the jit mode

    if (can_try_cache) {
      auto it = cached_casts.find(arg.unsafeGetTensorImpl());
      if (it != cached_casts.end()) {
        return std::get<1>(it->second);
      }
    }
    auto casted_arg = arg;
    if (to_type == at::kBFloat16) {
      //casted_arg = arg.to(at::kBFloat16).to_mkldnn();
      casted_arg = arg.to(at::kBFloat16);
      // casted_arg = arg.to_mkldnn(at::kBFloat16);
    } else {
      //casted_arg = arg.to_dense().to(at::kFloat);
      casted_arg = arg.to(at::kFloat);
      // casted_arg = arg.to_dense(at::kFloat);
    }
    if (can_try_cache) {
      cached_casts.emplace(
          arg.unsafeGetTensorImpl(),
          val_type{weakref_type(arg.getIntrusivePtr()), casted_arg});
    }
    return casted_arg;
  } else {
    return arg;
  }
}

template <class Redispatch, Redispatch* F>
int get_op_capability() {
  return FP32_DTYPE_PRIORITY; // fp32
}

template <>
int get_op_capability<Tensor (const Tensor &, const Tensor &, const c10::optional<Tensor>&, IntArrayRef, IntArrayRef, IntArrayRef, int64_t), 
           at::conv2d>() {
    return INT8_DTYPE_PRIORITY; // int8
}

template<DtypeCastPolicy policy, class Redispatch, Redispatch* F, class Ret, class ArgList> struct CPU_WrapFunction_ {};

template<DtypeCastPolicy policy,
         class Registered, // The signature for which we're registering.  The dispatcher's calling code invokes our
                           // registered functions with arguments matching Registered, so we register
                           // WrapFunction_::call methods with a matching signature to properly field those arguments.
                           // guts::function_traits below extracts return_type and parameter_types from Registered,
                           // which WrapFunction_ templates above use to declare their call methods.
         class Redispatch, // The signature for the function we're redispatching to.  In most cases this is the same
                           // as Registered, but for some ops (for example, ops where we append a dtype) it's useful
                           // to redispatch to a function with a different signature.
         Redispatch* F>    // The actual function we're redispatching to.
struct CPU_WrapFunction final {
  using type = CPU_WrapFunction_<
      policy, Redispatch, F, typename guts::function_traits<Registered>::return_type,
      typename guts::function_traits<Registered>::parameter_types>;
};

template<class Redispatch, Redispatch* F, class Ret, class... Args>
struct CPU_WrapFunction_<DtypeCastPolicy::user_defined_dtype, Redispatch, F, Ret,
                              guts::typelist::typelist<Args...>> {
  static Ret call(Args... args) {
    c10::impl::ExcludeDispatchKeyGuard no_autocastCPU(DispatchKey::AutocastCPU);

    int dtype_priority = get_op_capability<Redispatch, F>();

    dtype_priority = dtype_priority>current_target_dtype_priority
                    ?current_target_dtype_priority
                    :dtype_priority;

    return (*F)(cpu_cached_cast(inv_dtype_priority_map[dtype_priority],
                                args)...);
  }
};

template<class Redispatch, Redispatch* F, class Ret, class... Args>
struct CPU_WrapFunction_<DtypeCastPolicy::fp32, Redispatch, F, Ret,
                         guts::typelist::typelist<Args...>> {
  static Ret call(Args... args) {
    c10::impl::ExcludeDispatchKeyGuard no_autocastCPU(DispatchKey::AutocastCPU);
    return (*F)(cpu_cached_cast(at::kFloat, args)...);
  }
};

#define ADD_NS(RAW_OP) at::RAW_OP

#define KERNEL_CPU(FUNC, REGISTER_NAME, SIGNATURE, PRE_DEFINED_POLICY) \
    m.impl(TORCH_SELECTIVE_NAME("aten::" REGISTER_NAME), \
      &CPU_WrapFunction<DtypeCastPolicy::PRE_DEFINED_POLICY, SIGNATURE, SIGNATURE, &FUNC>::type::call);

/*void generic_wrapper_fallback(const c10::OperatorHandle& op, torch::jit::Stack* stack) {
  c10::impl::ExcludeDispatchKeyGuard no_autocastCPU(DispatchKey::AutocastCPU);
  if(int8_enabled()){
    //todo: int8 casted policy
  }
  op.callBoxed(stack);
  return;
}*/

TORCH_LIBRARY_IMPL(_, AutocastCPU, m) {
  m.fallback(torch::CppFunction::makeFallthrough());
  //m.fallback(torch::CppFunction::makeFromBoxedFunction<&generic_wrapper_fallback>());
}

TORCH_LIBRARY_IMPL(aten, AutocastCPU, m){
  KERNEL_CPU(ADD_NS(conv2d), "conv2d", Tensor (const Tensor &, const Tensor &, const c10::optional<Tensor>&, IntArrayRef, IntArrayRef, IntArrayRef, int64_t), user_defined_dtype)
  KERNEL_CPU(ADD_NS(_log_softmax), "_log_softmax", Tensor (const Tensor &, int64_t, bool), fp32)
  KERNEL_CPU(ADD_NS(batch_norm), "batch_norm", Tensor (const Tensor &, const c10::optional<Tensor>&, const c10::optional<Tensor>&, const c10::optional<Tensor>&, const c10::optional<Tensor>&, bool, double, double, bool), fp32)
}

}  // namespace autocast
}  // namespace torch_ipex

