#include "autocast_mode.h"

#include "library.h"

#include <exception>
#include <iostream>

namespace torch_ipex {
namespace autocast {

namespace {

using weakref_type =
    c10::weak_intrusive_ptr<c10::TensorImpl, c10::UndefinedTensorImpl>;
using val_type = std::tuple<weakref_type, at::Tensor>;
thread_local std::unordered_map<c10::TensorImpl*, val_type> cached_casts;

thread_local at::ScalarType current_target_dtype = at::kBFloat16;
} // namespace


at::ScalarType get_autocast_dtype() {
  return current_target_dtype;
}

void set_autocast_dtype(at::ScalarType dtype) {
  current_target_dtype = dtype;
}

void clear_autocast_cache() {
  cached_casts.clear();
}

Tensor cpu_cached_cast(at::ScalarType to_type, const Tensor& arg) {
  if (is_eligible_cpu(arg) && (arg.scalar_type() != to_type)) {
    bool can_try_cache =
        (to_type == at::kBFloat16 && arg.scalar_type() == at::kFloat &&
         arg.requires_grad() && arg.is_leaf() && !arg.is_view() &&
         at::autocast::is_autocast_cache_enabled());

    if (can_try_cache) {
      auto it = cached_casts.find(arg.unsafeGetTensorImpl());
      if (it != cached_casts.end()) {
        return std::get<1>(it->second);
      }
    }
    auto casted_arg = arg;
    if (arg.scalar_type() == at::kFloat && to_type == at::kBFloat16) {
      // This path works for fp32 to bf16
      casted_arg = arg.to(at::kBFloat16);
      // casted_arg = arg.to_mkldnn(at::kBFloat16);
    } else if (arg.scalar_type() == at::kBFloat16 && to_type == at::kFloat) {
      // This path works for bf16 to fp32
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

template <
    DtypeCastPolicy policy,
    class Redispatch,
    Redispatch* F,
    class Ret,
    class ArgList>
struct CPU_WrapFunction_ {};

template <
    DtypeCastPolicy policy,
    class Registered, // The signature for which we're registering.  The
                      // dispatcher's calling code invokes our registered
                      // functions with arguments matching Registered, so we
                      // register WrapFunction_::call methods with a matching
                      // signature to properly field those arguments.
                      // guts::function_traits below extracts return_type and
                      // parameter_types from Registered, which WrapFunction_
                      // templates above use to declare their call methods.
    class Redispatch, // The signature for the function we're redispatching to.
                      // In most cases this is the same as Registered, but for
                      // some ops (for example, ops where we append a dtype)
                      // it's useful to redispatch to a function with a
                      // different signature.
    Redispatch* F> // The actual function we're redispatching to.
struct CPU_WrapFunction final {
  using type = CPU_WrapFunction_<
      policy,
      Redispatch,
      F,
      typename guts::function_traits<Registered>::return_type,
      typename guts::function_traits<Registered>::parameter_types>;
};

// DtypeCastPolicy::user_defined_dtype
template <class Redispatch, Redispatch* F, class Ret, class... Args>
struct CPU_WrapFunction_<
    DtypeCastPolicy::user_defined_dtype,
    Redispatch,
    F,
    Ret,
    guts::typelist::typelist<Args...>> {
  static Ret call(Args... args) {
    c10::impl::ExcludeDispatchKeyGuard no_autocastCPU(DispatchKey::AutocastCPU);
    return (*F)(cpu_cached_cast(current_target_dtype, args)...);
  }
};

// DtypeCastPolicy::fp32
template <class Redispatch, Redispatch* F, class Ret, class... Args>
struct CPU_WrapFunction_<
    DtypeCastPolicy::fp32,
    Redispatch,
    F,
    Ret,
    guts::typelist::typelist<Args...>> {
  static Ret call(Args... args) {
    c10::impl::ExcludeDispatchKeyGuard no_autocastCPU(DispatchKey::AutocastCPU);
    return (*F)(cpu_cached_cast(at::kFloat, args)...);
  }
};

#define TUPLE_TWO_TENSORS std::tuple<Tensor, Tensor>
#define ADD_NS(RAW_OP) at::RAW_OP

#define MAKE_REGISTER_FUNC(FUNC, NAME, SIG, CAST_POLICY)                   \
  IPEX_TORCH_LIBRARY_IMPL(aten, AutocastCPU, m) {                          \
    m.impl(                                                                \
        TORCH_SELECTIVE_NAME("aten::" NAME),                               \
        &CPU_WrapFunction<DtypeCastPolicy::CAST_POLICY, SIG, SIG, &FUNC>:: \
            type::call);                                                   \
  }                                                                        \

// user_defined_dtype a.k.a WhiteList
MAKE_REGISTER_FUNC(
    ADD_NS(group_norm),
    "group_norm",
    Tensor(
        const Tensor&,
        int64_t,
        const c10::optional<Tensor>&,
        const c10::optional<Tensor>&,
        double,
        bool),
    user_defined_dtype)

MAKE_REGISTER_FUNC(
    ADD_NS(_native_multi_head_attention),
    "_native_multi_head_attention",
    TUPLE_TWO_TENSORS(
        const Tensor&,
        const Tensor&,
        const Tensor&,
        int64_t,
        int64_t,
        const Tensor&,
        const Tensor&,
        const Tensor&,
        const Tensor&,
        const c10::optional<Tensor>&,
        bool,
        bool,
        c10::optional<int64_t>),
    user_defined_dtype)

// fp32 cast policy a.k.a BlackList
MAKE_REGISTER_FUNC(ADD_NS(mish), "mish", Tensor(const Tensor&), fp32)

#undef TUPLE_TWO_TENSORS
} // namespace autocast
} // namespace torch_ipex
