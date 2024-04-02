#include "autocast_mode.h"
#include <exception>
#include "library.h"

namespace torch_ipex {
namespace autocast {

at::ScalarType get_autocast_dtype() {
  return at::autocast::get_autocast_cpu_dtype();
}

Tensor cpu_cached_cast(at::ScalarType to_type, const Tensor& arg) {
  return at::autocast::cached_cast(to_type, arg, c10::DeviceType::CPU);
}

template <
    DtypeCastPolicy bf16_cast_policy,
    DtypeCastPolicy fp16_cast_policy,
    class Redispatch,
    Redispatch* F,
    class Ret,
    class ArgList>
struct CPU_WrapFunction_ {};

template <
    DtypeCastPolicy bf16_cast_policy, // Cast policy for BF16.
    DtypeCastPolicy fp16_cast_policy, // Cast policy for FP16.
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
      bf16_cast_policy,
      fp16_cast_policy,
      Redispatch,
      F,
      typename guts::function_traits<Registered>::return_type,
      typename guts::function_traits<Registered>::parameter_types>;
};

template <
    DtypeCastPolicy bf16_cast_policy,
    DtypeCastPolicy fp16_cast_policy,
    class Redispatch,
    Redispatch* F,
    class Ret,
    class... Args>
struct CPU_WrapFunction_<
    bf16_cast_policy,
    fp16_cast_policy,
    Redispatch,
    F,
    Ret,
    guts::typelist::typelist<Args...>> {
  static Ret call(Args... args) {
    c10::impl::ExcludeDispatchKeyGuard no_autocastCPU(DispatchKey::AutocastCPU);
    auto set_type = get_autocast_dtype();
    auto policy =
        (set_type == at::kBFloat16) ? bf16_cast_policy : fp16_cast_policy;
    switch (policy) {
      case DtypeCastPolicy::user_defined_dtype:
        return (*F)(cpu_cached_cast(set_type, args)...);
      case DtypeCastPolicy::fp32:
        return (*F)(cpu_cached_cast(at::kFloat, args)...);
      case DtypeCastPolicy::promote:
        return (*F)(cpu_cached_cast(
            promote_type(get_autocast_dtype(), args...), args)...);
      default:
        return (*F)(args...);
    }
  }
};

#define ATEN_FN2(op_name, overload) at::_ops::op_name##_##overload::call
#define ATEN_FN(op_name) at::_ops::op_name::call

// BF16_CAST_POLICY: cast policy for BF16
// FP16_CAST_POLICY: cast policy for FP16
#define MAKE_REGISTER_FUNC_TWO_POLICIES(     \
    OP, BF16_CAST_POLICY, FP16_CAST_POLICY)  \
  m.impl(                                    \
      TORCH_SELECTIVE_NAME("aten::" #OP),    \
      &CPU_WrapFunction<                     \
          DtypeCastPolicy::BF16_CAST_POLICY, \
          DtypeCastPolicy::FP16_CAST_POLICY, \
          decltype(ATEN_FN(OP)),             \
          decltype(ATEN_FN(OP)),             \
          &ATEN_FN(OP)>::type::call);

#define MAKE_REGISTER_FUNC2_TWO_POLICIES(               \
    OP, OVERLOAD, BF16_CAST_POLICY, FP16_CAST_POLICY)   \
  m.impl(                                               \
      TORCH_SELECTIVE_NAME("aten::" #OP "." #OVERLOAD), \
      &CPU_WrapFunction<                                \
          DtypeCastPolicy::BF16_CAST_POLICY,            \
          DtypeCastPolicy::FP16_CAST_POLICY,            \
          decltype(ATEN_FN2(OP, OVERLOAD)),             \
          decltype(ATEN_FN2(OP, OVERLOAD)),             \
          &ATEN_FN2(OP, OVERLOAD)>::type::call);

IPEX_TORCH_LIBRARY_IMPL(aten, AutocastCPU, m) {
  // low precision policy for bf16 and fp32 cast policy for fp16
  MAKE_REGISTER_FUNC_TWO_POLICIES(_addmm_activation, user_defined_dtype, fp32)
  MAKE_REGISTER_FUNC_TWO_POLICIES(
      _transform_bias_rescale_qkv, user_defined_dtype, user_defined_dtype)

  // bf16 and fallthrough
  MAKE_REGISTER_FUNC_TWO_POLICIES(group_norm, user_defined_dtype, fallthrough)
  // fp32 and fp32 cast policies
  MAKE_REGISTER_FUNC_TWO_POLICIES(avg_pool3d, fp32, fp32)
  MAKE_REGISTER_FUNC_TWO_POLICIES(adaptive_avg_pool3d, fp32, fp32)
  MAKE_REGISTER_FUNC_TWO_POLICIES(_adaptive_avg_pool3d, fp32, fp32)

  // promote cast policies
  MAKE_REGISTER_FUNC_TWO_POLICIES(cat, promote, promote)
  MAKE_REGISTER_FUNC_TWO_POLICIES(stack, promote, promote)
  MAKE_REGISTER_FUNC_TWO_POLICIES(index_copy, promote, promote)
  MAKE_REGISTER_FUNC2_TWO_POLICIES(index_copy, dimname, promote, promote)
}

} // namespace autocast
} // namespace torch_ipex
