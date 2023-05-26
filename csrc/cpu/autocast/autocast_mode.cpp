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
        (to_type == current_target_dtype && arg.scalar_type() == at::kFloat &&
         arg.requires_grad() && arg.is_leaf() && !arg.is_view() &&
         at::autocast::is_autocast_cache_enabled());

    if (can_try_cache) {
      auto it = cached_casts.find(arg.unsafeGetTensorImpl());
      if (it != cached_casts.end()) {
        return std::get<1>(it->second);
      }
    }
    auto casted_arg = arg;
    if (arg.scalar_type() == at::kFloat && to_type == current_target_dtype) {
      // This path works for fp32 to bf16
      casted_arg = arg.to(current_target_dtype);
      // casted_arg = arg.to_mkldnn(at::kBFloat16);
    } else if (
        arg.scalar_type() == current_target_dtype && to_type == at::kFloat) {
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
  MAKE_REGISTER_FUNC_TWO_POLICIES(conv1d, user_defined_dtype, fp32)
  MAKE_REGISTER_FUNC_TWO_POLICIES(conv2d, user_defined_dtype, fp32)
  MAKE_REGISTER_FUNC_TWO_POLICIES(conv3d, user_defined_dtype, fp32)
  MAKE_REGISTER_FUNC_TWO_POLICIES(bmm, user_defined_dtype, fp32)
  MAKE_REGISTER_FUNC_TWO_POLICIES(mm, user_defined_dtype, fp32)
  MAKE_REGISTER_FUNC_TWO_POLICIES(baddbmm, user_defined_dtype, fp32)
  MAKE_REGISTER_FUNC_TWO_POLICIES(addmm, user_defined_dtype, fp32)
  MAKE_REGISTER_FUNC_TWO_POLICIES(addbmm, user_defined_dtype, fp32)
  MAKE_REGISTER_FUNC_TWO_POLICIES(linear, user_defined_dtype, fp32)
  MAKE_REGISTER_FUNC2_TWO_POLICIES(
      _convolution, deprecated, user_defined_dtype, fp32)
  MAKE_REGISTER_FUNC_TWO_POLICIES(matmul, user_defined_dtype, fp32)
  MAKE_REGISTER_FUNC_TWO_POLICIES(conv_tbc, user_defined_dtype, fp32)
  MAKE_REGISTER_FUNC_TWO_POLICIES(conv_transpose1d, user_defined_dtype, fp32)
  MAKE_REGISTER_FUNC2_TWO_POLICIES(
      conv_transpose2d, input, user_defined_dtype, fp32)
  MAKE_REGISTER_FUNC2_TWO_POLICIES(
      conv_transpose3d, input, user_defined_dtype, fp32)
  MAKE_REGISTER_FUNC_TWO_POLICIES(group_norm, user_defined_dtype, fp32)
  MAKE_REGISTER_FUNC_TWO_POLICIES(
      _native_multi_head_attention, user_defined_dtype, fp32)
  MAKE_REGISTER_FUNC_TWO_POLICIES(
      _transform_bias_rescale_qkv, user_defined_dtype, fp32)

  // fp32 and fp32 cast policies
  MAKE_REGISTER_FUNC_TWO_POLICIES(avg_pool3d, fp32, fp32)
  MAKE_REGISTER_FUNC_TWO_POLICIES(adaptive_avg_pool3d, fp32, fp32)
  MAKE_REGISTER_FUNC_TWO_POLICIES(_adaptive_avg_pool3d, fp32, fp32)

  // fallthrough and fp32 cast policies
  MAKE_REGISTER_FUNC_TWO_POLICIES(batch_norm, fallthrough, fp32)
  MAKE_REGISTER_FUNC_TWO_POLICIES(avg_pool1d, fallthrough, fp32)
  MAKE_REGISTER_FUNC_TWO_POLICIES(avg_pool2d, fallthrough, fp32)
  MAKE_REGISTER_FUNC_TWO_POLICIES(max_pool1d, fallthrough, fp32)
  MAKE_REGISTER_FUNC_TWO_POLICIES(max_pool2d, fallthrough, fp32)
  MAKE_REGISTER_FUNC_TWO_POLICIES(max_pool3d, fallthrough, fp32)
  MAKE_REGISTER_FUNC_TWO_POLICIES(layer_norm, fallthrough, fp32)
  MAKE_REGISTER_FUNC_TWO_POLICIES(bernoulli, fallthrough, fp32)
  MAKE_REGISTER_FUNC2_TWO_POLICIES(bernoulli, p, fallthrough, fp32)
  MAKE_REGISTER_FUNC_TWO_POLICIES(dropout, fallthrough, fp32)
  MAKE_REGISTER_FUNC_TWO_POLICIES(topk, fallthrough, fp32)
  MAKE_REGISTER_FUNC_TWO_POLICIES(cumsum, fallthrough, fp32)
  MAKE_REGISTER_FUNC2_TWO_POLICIES(cumsum, dimname, fallthrough, fp32)
  MAKE_REGISTER_FUNC_TWO_POLICIES(
      scaled_dot_product_attention, fallthrough, fp32)
  MAKE_REGISTER_FUNC_TWO_POLICIES(
      _scaled_dot_product_attention, fallthrough, fp32)
  MAKE_REGISTER_FUNC_TWO_POLICIES(
      _scaled_dot_product_attention_math, fallthrough, fp32)
  MAKE_REGISTER_FUNC_TWO_POLICIES(addcdiv, fallthrough, fp32)
  MAKE_REGISTER_FUNC_TWO_POLICIES(addcmul, fallthrough, fp32)
  MAKE_REGISTER_FUNC2_TWO_POLICIES(softmax, int, fallthrough, fp32)
  MAKE_REGISTER_FUNC2_TWO_POLICIES(softmax, Dimname, fallthrough, fp32)
  MAKE_REGISTER_FUNC2_TWO_POLICIES(log_softmax, int, fallthrough, fp32)
  MAKE_REGISTER_FUNC2_TWO_POLICIES(log_softmax, Dimname, fallthrough, fp32)
  MAKE_REGISTER_FUNC_TWO_POLICIES(upsample_linear1d, fallthrough, fp32)
  MAKE_REGISTER_FUNC_TWO_POLICIES(upsample_bilinear2d, fallthrough, fp32)
  MAKE_REGISTER_FUNC_TWO_POLICIES(_upsample_bilinear2d_aa, fallthrough, fp32)
  MAKE_REGISTER_FUNC_TWO_POLICIES(upsample_bicubic2d, fallthrough, fp32)
  MAKE_REGISTER_FUNC_TWO_POLICIES(_upsample_bicubic2d_aa, fallthrough, fp32)
  MAKE_REGISTER_FUNC_TWO_POLICIES(upsample_trilinear3d, fallthrough, fp32)
  MAKE_REGISTER_FUNC_TWO_POLICIES(upsample_nearest1d, fallthrough, fp32)
  MAKE_REGISTER_FUNC_TWO_POLICIES(_upsample_nearest_exact1d, fallthrough, fp32)
  MAKE_REGISTER_FUNC_TWO_POLICIES(upsample_nearest2d, fallthrough, fp32)
  MAKE_REGISTER_FUNC_TWO_POLICIES(_upsample_nearest_exact2d, fallthrough, fp32)
  MAKE_REGISTER_FUNC_TWO_POLICIES(upsample_nearest3d, fallthrough, fp32)
  MAKE_REGISTER_FUNC_TWO_POLICIES(_upsample_nearest_exact3d, fallthrough, fp32)
  MAKE_REGISTER_FUNC2_TWO_POLICIES(upsample_linear1d, vec, fallthrough, fp32)
  MAKE_REGISTER_FUNC2_TWO_POLICIES(upsample_bilinear2d, vec, fallthrough, fp32)
  MAKE_REGISTER_FUNC2_TWO_POLICIES(
      _upsample_bilinear2d_aa, vec, fallthrough, fp32)
  MAKE_REGISTER_FUNC2_TWO_POLICIES(upsample_trilinear3d, vec, fallthrough, fp32)
  MAKE_REGISTER_FUNC2_TWO_POLICIES(upsample_bicubic2d, vec, fallthrough, fp32)
  MAKE_REGISTER_FUNC2_TWO_POLICIES(
      _upsample_bicubic2d_aa, vec, fallthrough, fp32)
  MAKE_REGISTER_FUNC2_TWO_POLICIES(upsample_nearest1d, vec, fallthrough, fp32)
  MAKE_REGISTER_FUNC2_TWO_POLICIES(
      _upsample_nearest_exact1d, vec, fallthrough, fp32)
  MAKE_REGISTER_FUNC2_TWO_POLICIES(upsample_nearest2d, vec, fallthrough, fp32)
  MAKE_REGISTER_FUNC2_TWO_POLICIES(
      _upsample_nearest_exact2d, vec, fallthrough, fp32)
  MAKE_REGISTER_FUNC2_TWO_POLICIES(upsample_nearest3d, vec, fallthrough, fp32)
  MAKE_REGISTER_FUNC2_TWO_POLICIES(
      _upsample_nearest_exact3d, vec, fallthrough, fp32)

  MAKE_REGISTER_FUNC_TWO_POLICIES(adaptive_avg_pool1d, fallthrough, fp32)
  MAKE_REGISTER_FUNC_TWO_POLICIES(adaptive_avg_pool2d, fallthrough, fp32)
  MAKE_REGISTER_FUNC_TWO_POLICIES(_adaptive_avg_pool2d, fallthrough, fp32)

  // promote cast policies
  MAKE_REGISTER_FUNC_TWO_POLICIES(cat, promote, promote)
  MAKE_REGISTER_FUNC_TWO_POLICIES(stack, promote, promote)
  MAKE_REGISTER_FUNC_TWO_POLICIES(index_copy, promote, promote)
  MAKE_REGISTER_FUNC2_TWO_POLICIES(index_copy, dimname, promote, promote)
}

} // namespace autocast
} // namespace torch_ipex
