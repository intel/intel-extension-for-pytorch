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
            promote_type(get_autocast_dtype(), DeviceType::CPU, args...),
            args)...);
      default:
        return (*F)(args...);
    }
  }
};

#define TUPLE_TWO_TENSORS std::tuple<Tensor, Tensor>
#define ADD_NS(RAW_OP) at::RAW_OP

// BF16_CAST_POLICY: cast policy for BF16
// FP16_CAST_POLICY: cast policy for FP16
#define MAKE_REGISTER_FUNC_TWO_POLICIES(                 \
    FUNC, NAME, SIG, BF16_CAST_POLICY, FP16_CAST_POLICY) \
  m.impl(                                                \
      TORCH_SELECTIVE_NAME("aten::" NAME),               \
      &CPU_WrapFunction<                                 \
          DtypeCastPolicy::BF16_CAST_POLICY,             \
          DtypeCastPolicy::FP16_CAST_POLICY,             \
          SIG,                                           \
          SIG,                                           \
          &FUNC>::type::call);

IPEX_TORCH_LIBRARY_IMPL(aten, AutocastCPU, m) {
  // low precision policy for bf16 and fp32 cast policy for fp16
  MAKE_REGISTER_FUNC_TWO_POLICIES(
      ADD_NS(conv1d),
      "conv1d",
      Tensor(
          const Tensor&,
          const Tensor&,
          const c10::optional<Tensor>&,
          IntArrayRef,
          IntArrayRef,
          IntArrayRef,
          int64_t),
      user_defined_dtype,
      fp32)
  MAKE_REGISTER_FUNC_TWO_POLICIES(
      ADD_NS(conv2d),
      "conv2d",
      Tensor(
          const Tensor&,
          const Tensor&,
          const c10::optional<Tensor>&,
          IntArrayRef,
          IntArrayRef,
          IntArrayRef,
          int64_t),
      user_defined_dtype,
      fp32)
  MAKE_REGISTER_FUNC_TWO_POLICIES(
      ADD_NS(conv3d),
      "conv3d",
      Tensor(
          const Tensor&,
          const Tensor&,
          const c10::optional<Tensor>&,
          IntArrayRef,
          IntArrayRef,
          IntArrayRef,
          int64_t),
      user_defined_dtype,
      fp32)
  MAKE_REGISTER_FUNC_TWO_POLICIES(
      ADD_NS(bmm),
      "bmm",
      Tensor(const Tensor&, const Tensor&),
      user_defined_dtype,
      fp32)
  MAKE_REGISTER_FUNC_TWO_POLICIES(
      ADD_NS(mm),
      "mm",
      Tensor(const Tensor&, const Tensor&),
      user_defined_dtype,
      fp32)
  MAKE_REGISTER_FUNC_TWO_POLICIES(
      ADD_NS(baddbmm),
      "baddbmm",
      Tensor(
          const Tensor&,
          const Tensor&,
          const Tensor&,
          const Scalar&,
          const Scalar&),
      user_defined_dtype,
      fp32)
  MAKE_REGISTER_FUNC_TWO_POLICIES(
      ADD_NS(addmm),
      "addmm",
      Tensor(
          const Tensor&,
          const Tensor&,
          const Tensor&,
          const Scalar&,
          const Scalar&),
      user_defined_dtype,
      fp32)
  MAKE_REGISTER_FUNC_TWO_POLICIES(
      ADD_NS(addbmm),
      "addbmm",
      Tensor(
          const Tensor&,
          const Tensor&,
          const Tensor&,
          const Scalar&,
          const Scalar&),
      user_defined_dtype,
      fp32)
  MAKE_REGISTER_FUNC_TWO_POLICIES(
      ADD_NS(linear),
      "linear",
      Tensor(const Tensor&, const Tensor&, const c10::optional<Tensor>&),
      user_defined_dtype,
      fp32)
  MAKE_REGISTER_FUNC_TWO_POLICIES(
      ADD_NS(_convolution),
      "_convolution.deprecated",
      Tensor(
          const Tensor&,
          const Tensor&,
          const c10::optional<Tensor>&,
          IntArrayRef,
          IntArrayRef,
          IntArrayRef,
          bool,
          IntArrayRef,
          int64_t,
          bool,
          bool,
          bool),
      user_defined_dtype,
      fp32)
  MAKE_REGISTER_FUNC_TWO_POLICIES(
      ADD_NS(matmul),
      "matmul",
      Tensor(const Tensor&, const Tensor&),
      user_defined_dtype,
      fp32)
  MAKE_REGISTER_FUNC_TWO_POLICIES(
      ADD_NS(conv_tbc),
      "conv_tbc",
      Tensor(const Tensor&, const Tensor&, const Tensor&, int64_t),
      user_defined_dtype,
      fp32)
  MAKE_REGISTER_FUNC_TWO_POLICIES(
      ADD_NS(conv_transpose1d),
      "conv_transpose1d",
      Tensor(
          const Tensor&,
          const Tensor&,
          const c10::optional<Tensor>&,
          IntArrayRef,
          IntArrayRef,
          IntArrayRef,
          int64_t,
          IntArrayRef),
      user_defined_dtype,
      fp32)
  MAKE_REGISTER_FUNC_TWO_POLICIES(
      ADD_NS(conv_transpose2d),
      "conv_transpose2d.input",
      Tensor(
          const Tensor&,
          const Tensor&,
          const c10::optional<Tensor>&,
          IntArrayRef,
          IntArrayRef,
          IntArrayRef,
          int64_t,
          IntArrayRef),
      user_defined_dtype,
      fp32)
  MAKE_REGISTER_FUNC_TWO_POLICIES(
      ADD_NS(conv_transpose3d),
      "conv_transpose3d.input",
      Tensor(
          const Tensor&,
          const Tensor&,
          const c10::optional<Tensor>&,
          IntArrayRef,
          IntArrayRef,
          IntArrayRef,
          int64_t,
          IntArrayRef),
      user_defined_dtype,
      fp32)

  MAKE_REGISTER_FUNC_TWO_POLICIES(
      ADD_NS(group_norm),
      "group_norm",
      Tensor(
          const Tensor&,
          int64_t,
          const c10::optional<Tensor>&,
          const c10::optional<Tensor>&,
          double,
          bool),
      user_defined_dtype,
      fp32)

  MAKE_REGISTER_FUNC_TWO_POLICIES(
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
      user_defined_dtype,
      fp32)

  // fp32 and fp32 cast policies a.k.a BlackList
  MAKE_REGISTER_FUNC_TWO_POLICIES(
      ADD_NS(mish), "mish", Tensor(const Tensor&), fp32, fp32)

  // fallthrough and fp32 cast policies
  MAKE_REGISTER_FUNC_TWO_POLICIES(
      ADD_NS(batch_norm),
      "batch_norm",
      Tensor(
          const Tensor&,
          const c10::optional<Tensor>&,
          const c10::optional<Tensor>&,
          const c10::optional<Tensor>&,
          const c10::optional<Tensor>&,
          bool,
          double,
          double,
          bool),
      fallthrough,
      fp32)
  MAKE_REGISTER_FUNC_TWO_POLICIES(
      ADD_NS(avg_pool1d),
      "avg_pool1d",
      Tensor(const Tensor&, IntArrayRef, IntArrayRef, IntArrayRef, bool, bool),
      fallthrough,
      fp32)
  MAKE_REGISTER_FUNC_TWO_POLICIES(
      ADD_NS(avg_pool3d),
      "avg_pool3d",
      Tensor(
          const Tensor&,
          IntArrayRef,
          IntArrayRef,
          IntArrayRef,
          bool,
          bool,
          c10::optional<int64_t>),
      fallthrough,
      fp32)
  MAKE_REGISTER_FUNC_TWO_POLICIES(
      ADD_NS(avg_pool2d),
      "avg_pool2d",
      Tensor(
          const Tensor&,
          IntArrayRef,
          IntArrayRef,
          IntArrayRef,
          bool,
          bool,
          c10::optional<int64_t>),
      fallthrough,
      fp32)
  MAKE_REGISTER_FUNC_TWO_POLICIES(
      ADD_NS(max_pool1d),
      "max_pool1d",
      Tensor(
          const Tensor&,
          IntArrayRef,
          IntArrayRef,
          IntArrayRef,
          IntArrayRef,
          bool),
      fallthrough,
      fp32)
  MAKE_REGISTER_FUNC_TWO_POLICIES(
      ADD_NS(max_pool2d),
      "max_pool2d",
      Tensor(
          const Tensor&,
          IntArrayRef,
          IntArrayRef,
          IntArrayRef,
          IntArrayRef,
          bool),
      fallthrough,
      fp32)
  MAKE_REGISTER_FUNC_TWO_POLICIES(
      ADD_NS(max_pool3d),
      "max_pool3d",
      Tensor(
          const Tensor&,
          IntArrayRef,
          IntArrayRef,
          IntArrayRef,
          IntArrayRef,
          bool),
      fallthrough,
      fp32)
  MAKE_REGISTER_FUNC_TWO_POLICIES(
      ADD_NS(layer_norm),
      "layer_norm",
      Tensor(
          const Tensor&,
          IntArrayRef,
          const c10::optional<Tensor>&,
          const c10::optional<Tensor>&,
          double,
          bool),
      fallthrough,
      fp32)
  MAKE_REGISTER_FUNC_TWO_POLICIES(
      ADD_NS(bernoulli),
      "bernoulli",
      Tensor(const Tensor&, c10::optional<at::Generator>),
      fallthrough,
      fp32)
  MAKE_REGISTER_FUNC_TWO_POLICIES(
      ADD_NS(bernoulli),
      "bernoulli.p",
      Tensor(const Tensor&, double, c10::optional<at::Generator>),
      fallthrough,
      fp32)
  MAKE_REGISTER_FUNC_TWO_POLICIES(
      ADD_NS(dropout),
      "dropout",
      Tensor(const Tensor&, double, bool),
      fallthrough,
      fp32)
  MAKE_REGISTER_FUNC_TWO_POLICIES(
      ADD_NS(gelu),
      "gelu",
      Tensor(const Tensor&, c10::string_view),
      fallthrough,
      fp32)
  MAKE_REGISTER_FUNC_TWO_POLICIES(
      ADD_NS(tanh), "tanh", Tensor(const Tensor&), fallthrough, fp32)

  // promote cast policies
  MAKE_REGISTER_FUNC_TWO_POLICIES(
      ADD_NS(cat), "cat", Tensor(TensorList, int64_t), promote, promote)
  MAKE_REGISTER_FUNC_TWO_POLICIES(
      ADD_NS(stack), "stack", Tensor(TensorList, int64_t), promote, promote)
  MAKE_REGISTER_FUNC_TWO_POLICIES(
      ADD_NS(index_copy),
      "index_copy",
      Tensor(const Tensor&, int64_t, const Tensor&, const Tensor&),
      promote,
      promote)
  MAKE_REGISTER_FUNC_TWO_POLICIES(
      ADD_NS(index_copy),
      "index_copy.dimname",
      Tensor(const Tensor&, at::Dimname, const Tensor&, const Tensor&),
      promote,
      promote)
}
#undef TUPLE_TWO_TENSORS

} // namespace autocast
} // namespace torch_ipex
