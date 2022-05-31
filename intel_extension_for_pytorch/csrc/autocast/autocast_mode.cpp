#include "autocast_mode.h"
#include "autocast_kernel.hpp"

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

thread_local int nesting = 0;

thread_local at::ScalarType current_target_dtype = at::kFloat;
thread_local bool int8_enabled = false;
thread_local bool llga_fp32_bf16_enabled = false;
} // namespace

bool is_quantization_enabled() {
  return int8_enabled;
}
void set_quantization_enabled(bool new_enabled) {
  int8_enabled = new_enabled;
}

bool is_llga_fp32_bf16_enabled() {
  return llga_fp32_bf16_enabled;
}
void set_llga_fp32_bf16_enabled(bool new_enabled) {
  llga_fp32_bf16_enabled = new_enabled;
}

at::ScalarType get_autocast_dtype() {
  return current_target_dtype;
}

void set_autocast_dtype(at::ScalarType dtype) {
  current_target_dtype = dtype;
}

int autocast_increment_nesting() {
  return ++nesting;
}

int autocast_decrement_nesting() {
  return --nesting;
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

#define ADD_NS(RAW_OP) at::RAW_OP

#define MAKE_REGISTER_FUNC(FUNC, NAME, SIG, CAST_POLICY)                   \
  IPEX_TORCH_LIBRARY_IMPL(aten, AutocastCPU, m) {                          \
    m.impl(                                                                \
        TORCH_SELECTIVE_NAME("aten::" NAME),                               \
        &CPU_WrapFunction<DtypeCastPolicy::CAST_POLICY, SIG, SIG, &FUNC>:: \
            type::call);                                                   \
  }                                                                        \
  template <>                                                              \
  std::string get_op_name<SIG, FUNC>() {                                   \
    return NAME;                                                           \
  }

// user_defined_dtype a.k.a WhiteList
MAKE_REGISTER_FUNC(
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
    user_defined_dtype)
MAKE_REGISTER_FUNC(
    ADD_NS(bmm),
    "bmm",
    Tensor(const Tensor&, const Tensor&),
    user_defined_dtype)
MAKE_REGISTER_FUNC(
    ADD_NS(mm),
    "mm",
    Tensor(const Tensor&, const Tensor&),
    user_defined_dtype)
MAKE_REGISTER_FUNC(
    ADD_NS(baddbmm),
    "baddbmm",
    Tensor(
        const Tensor&,
        const Tensor&,
        const Tensor&,
        const Scalar&,
        const Scalar&),
    user_defined_dtype)
MAKE_REGISTER_FUNC(
    ADD_NS(addmm),
    "addmm",
    Tensor(
        const Tensor&,
        const Tensor&,
        const Tensor&,
        const Scalar&,
        const Scalar&),
    user_defined_dtype)
MAKE_REGISTER_FUNC(
    ADD_NS(addbmm),
    "addbmm",
    Tensor(
        const Tensor&,
        const Tensor&,
        const Tensor&,
        const Scalar&,
        const Scalar&),
    user_defined_dtype)
MAKE_REGISTER_FUNC(
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
    user_defined_dtype)
MAKE_REGISTER_FUNC(
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
    user_defined_dtype)

// fp32 cast policy a.k.a BlackList
MAKE_REGISTER_FUNC(
    ADD_NS(linalg_matrix_rank),
    "linalg_matrix_rank.atol_rtol_tensor",
    Tensor(
        const Tensor&,
        const c10::optional<at::Tensor>&,
        const c10::optional<at::Tensor>&,
        bool),
    fp32)
MAKE_REGISTER_FUNC(
    ADD_NS(linalg_matrix_rank),
    "linalg_matrix_rank.atol_rtol_float",
    Tensor(const Tensor&, c10::optional<double>, c10::optional<double>, bool),
    fp32)
MAKE_REGISTER_FUNC(ADD_NS(mish), "mish", Tensor(const Tensor&), fp32)

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

IPEX_TORCH_LIBRARY_IMPL(aten, AutocastCPU, m) {
  // for int8 path
  m.impl(
      TORCH_SELECTIVE_NAME("aten::conv2d"),
      TORCH_FN((&torch_ipex::autocast::conv2d)));
  m.impl(
      TORCH_SELECTIVE_NAME("aten::conv3d"),
      TORCH_FN((&torch_ipex::autocast::conv3d)));
  m.impl(
      TORCH_SELECTIVE_NAME("aten::conv_transpose3d.input"),
      TORCH_FN((&torch_ipex::autocast::conv_transpose3d)));
  m.impl(
      TORCH_SELECTIVE_NAME("aten::_convolution"),
      TORCH_FN((&torch_ipex::autocast::_convolution)));
  m.impl(
      TORCH_SELECTIVE_NAME("aten::_convolution.deprecated"),
      TORCH_FN((&torch_ipex::autocast::_convolution_deprecated)));
  m.impl(
      TORCH_SELECTIVE_NAME("aten::batch_norm"),
      TORCH_FN((&torch_ipex::autocast::batch_norm)));
  // m.impl(TORCH_SELECTIVE_NAME("aten::linear"),
  // TORCH_FN((&torch_ipex::autocast::linear)));
  m.impl(
      TORCH_SELECTIVE_NAME("aten::max_pool2d"),
      TORCH_FN((&torch_ipex::autocast::max_pool2d)));
  m.impl(
      TORCH_SELECTIVE_NAME("aten::adaptive_avg_pool2d"),
      TORCH_FN((&torch_ipex::autocast::adaptive_avg_pool2d)));
  m.impl(
      TORCH_SELECTIVE_NAME("aten::avg_pool2d"),
      TORCH_FN((&torch_ipex::autocast::avg_pool2d)));
  m.impl(
      TORCH_SELECTIVE_NAME("aten::adaptive_max_pool2d"),
      TORCH_FN((&torch_ipex::autocast::adaptive_max_pool2d)));
  m.impl(
      TORCH_SELECTIVE_NAME("aten::upsample_nearest1d"),
      TORCH_FN((&torch_ipex::autocast::upsample_nearest1d)));
  m.impl(
      TORCH_SELECTIVE_NAME("aten::upsample_nearest1d.vec"),
      TORCH_FN((&torch_ipex::autocast::upsample_nearest1d_vec)));
  m.impl(
      TORCH_SELECTIVE_NAME("aten::upsample_nearest2d"),
      TORCH_FN((&torch_ipex::autocast::upsample_nearest2d)));
  m.impl(
      TORCH_SELECTIVE_NAME("aten::upsample_nearest2d.vec"),
      TORCH_FN((&torch_ipex::autocast::upsample_nearest2d_vec)));
  m.impl(
      TORCH_SELECTIVE_NAME("aten::upsample_nearest3d"),
      TORCH_FN((&torch_ipex::autocast::upsample_nearest3d)));
  m.impl(
      TORCH_SELECTIVE_NAME("aten::upsample_nearest3d.vec"),
      TORCH_FN((&torch_ipex::autocast::upsample_nearest3d_vec)));
  m.impl(
      TORCH_SELECTIVE_NAME("aten::upsample_linear1d"),
      TORCH_FN((&torch_ipex::autocast::upsample_linear1d)));
  m.impl(
      TORCH_SELECTIVE_NAME("aten::upsample_linear1d.vec"),
      TORCH_FN((&torch_ipex::autocast::upsample_linear1d_vec)));
  m.impl(
      TORCH_SELECTIVE_NAME("aten::upsample_bilinear2d"),
      TORCH_FN((&torch_ipex::autocast::upsample_bilinear2d)));
  m.impl(
      TORCH_SELECTIVE_NAME("aten::upsample_bilinear2d.vec"),
      TORCH_FN((&torch_ipex::autocast::upsample_bilinear2d_vec)));
  m.impl(
      TORCH_SELECTIVE_NAME("aten::upsample_trilinear3d"),
      TORCH_FN((&torch_ipex::autocast::upsample_trilinear3d)));
  m.impl(
      TORCH_SELECTIVE_NAME("aten::upsample_trilinear3d.vec"),
      TORCH_FN((&torch_ipex::autocast::upsample_trilinear3d_vec)));
  m.impl(
      TORCH_SELECTIVE_NAME("aten::relu"),
      TORCH_FN((&torch_ipex::autocast::relu)));
  m.impl(
      TORCH_SELECTIVE_NAME("aten::relu_"),
      TORCH_FN((&torch_ipex::autocast::relu_)));
  m.impl(
      TORCH_SELECTIVE_NAME("aten::sigmoid"),
      TORCH_FN((&torch_ipex::autocast::sigmoid)));
  m.impl(
      TORCH_SELECTIVE_NAME("aten::linear"),
      TORCH_FN((&torch_ipex::autocast::linear)));
  m.impl(
      TORCH_SELECTIVE_NAME("aten::add_.Tensor"),
      TORCH_FN((&torch_ipex::autocast::add_tensor_)));
  m.impl(
      TORCH_SELECTIVE_NAME("aten::add.Tensor"),
      TORCH_FN((&torch_ipex::autocast::add_tensor)));
  m.impl(
      TORCH_SELECTIVE_NAME("aten::dropout"),
      TORCH_FN((&torch_ipex::autocast::dropout)));
  m.impl(
      TORCH_SELECTIVE_NAME("aten::gelu"),
      TORCH_FN((&torch_ipex::autocast::gelu)));
  m.impl(
      TORCH_SELECTIVE_NAME("aten::lstm.input"),
      TORCH_FN((&torch_ipex::autocast::lstm_aten)));
  m.impl(
      TORCH_SELECTIVE_NAME("aten::flatten.using_ints"),
      TORCH_FN((&torch_ipex::autocast::flatten)));
  m.impl(
      TORCH_SELECTIVE_NAME("aten::matmul"),
      TORCH_FN((&torch_ipex::autocast::matmul)));
}

} // namespace autocast
} // namespace torch_ipex