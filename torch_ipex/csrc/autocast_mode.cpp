#include "autocast_mode.h"
#include "autocast_kernel.hpp"
#include "autocast_verbose.h"

#include "library.h"

#include <exception>
#include <iostream>

namespace torch_ipex {
namespace autocast {

namespace {

using weakref_type =
    c10::weak_intrusive_ptr<c10::TensorImpl, c10::UndefinedTensorImpl>;
using val_type = std::tuple<weakref_type, at::Tensor>;
thread_local std::unordered_map<c10::TensorImpl *, val_type> cached_casts;

thread_local int nesting = 0;

thread_local at::ScalarType current_target_dtype = at::kFloat;
thread_local bool int8_enabled = false;
} // namespace

bool is_quantization_enabled() { return int8_enabled; }
void set_quantization_enabled(bool new_enabled) { int8_enabled = new_enabled; }

at::ScalarType get_autocast_dtype() { return current_target_dtype; }

void set_autocast_dtype(at::ScalarType dtype) { current_target_dtype = dtype; }

int autocast_increment_nesting() { return ++nesting; }

int autocast_decrement_nesting() { return --nesting; }

void clear_autocast_cache() { cached_casts.clear(); }

Tensor cpu_cached_cast(at::ScalarType to_type, const Tensor &arg) {
  if (is_eligible_cpu(arg) && (arg.scalar_type() != to_type)) {
    bool can_try_cache =
        !at::GradMode::is_enabled() &&
        (to_type == at::kBFloat16 && arg.scalar_type() == at::kFloat &&
         arg.requires_grad() && arg.is_leaf() && !arg.is_view() &&
         !torch::jit::tracer::isTracing()); // Disable cache in jit mode

    if (can_try_cache) {
      auto it = cached_casts.find(arg.unsafeGetTensorImpl());
      if (it != cached_casts.end()) {
        return std::get<1>(it->second);
      }
    }
    auto casted_arg = arg;
    if (arg.scalar_type() == at::kFloat && to_type == at::kBFloat16) {
      // This path works for fp32 to bf16
#if defined(ENABLE_AUTOCAST_VERBOSE)
      verbose::autocast_verbose(to_type, arg);
#endif
      casted_arg = arg.to(at::kBFloat16);
      // casted_arg = arg.to_mkldnn(at::kBFloat16);
    } else if (arg.scalar_type() == at::kBFloat16 && to_type == at::kFloat) {
      // This path works for bf16 to fp32
#if defined(ENABLE_AUTOCAST_VERBOSE)
      verbose::autocast_verbose(to_type, arg);
#endif
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

template <DtypeCastPolicy policy, class Redispatch, Redispatch *F, class Ret,
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
    Redispatch *F> // The actual function we're redispatching to.
struct CPU_WrapFunction final {
  using type = CPU_WrapFunction_<
      policy, Redispatch, F,
      typename guts::function_traits<Registered>::return_type,
      typename guts::function_traits<Registered>::parameter_types>;
};

// DtypeCastPolicy::user_defined_dtype
template <class Redispatch, Redispatch *F, class Ret, class... Args>
struct CPU_WrapFunction_<DtypeCastPolicy::user_defined_dtype, Redispatch, F,
                         Ret, guts::typelist::typelist<Args...>> {
  static Ret call(Args... args) {
    c10::impl::ExcludeDispatchKeyGuard no_autocastCPU(DispatchKey::AutocastCPU);
#if defined(ENABLE_AUTOCAST_VERBOSE)
    verbose::OpNameGuard op_name(get_op_name<Redispatch, F>());
#endif
    return (*F)(cpu_cached_cast(current_target_dtype, args)...);
  }
};

// DtypeCastPolicy::fp32
template <class Redispatch, Redispatch *F, class Ret, class... Args>
struct CPU_WrapFunction_<DtypeCastPolicy::fp32, Redispatch, F, Ret,
                         guts::typelist::typelist<Args...>> {
  static Ret call(Args... args) {
    c10::impl::ExcludeDispatchKeyGuard no_autocastCPU(DispatchKey::AutocastCPU);
#if defined(ENABLE_AUTOCAST_VERBOSE)
    verbose::OpNameGuard op_name(get_op_name<Redispatch, F>());
#endif
    return (*F)(cpu_cached_cast(at::kFloat, args)...);
  }
};

// DtypeCastPolicy::promote
template <class Redispatch, Redispatch *F, class Ret, class... Args>
struct CPU_WrapFunction_<DtypeCastPolicy::promote, Redispatch, F, Ret,
                         guts::typelist::typelist<Args...>> {
  static Ret call(Args... args) {
    c10::impl::ExcludeDispatchKeyGuard no_autocastCPU(DispatchKey::AutocastCPU);
    auto to_type = promote_type(at::kBFloat16, args...);
#if defined(ENABLE_AUTOCAST_VERBOSE)
    verbose::OpNameGuard op_name(get_op_name<Redispatch, F>());
#endif
    return (*F)(cpu_cached_cast(to_type, args)...);
  }
};

#define ADD_NS(RAW_OP) at::RAW_OP

#define KERNEL_CPU(FUNC, REGISTER_NAME, SIGNATURE, PRE_DEFINED_POLICY)         \
  m.impl(TORCH_SELECTIVE_NAME("aten::" REGISTER_NAME),                         \
         &CPU_WrapFunction<DtypeCastPolicy::PRE_DEFINED_POLICY, SIGNATURE,     \
                           SIGNATURE, &FUNC>::type::call);

#define TUPLE_TWO_TENSORS std::tuple<Tensor, Tensor>

#define TUPLE_THREE_TENSORS std::tuple<Tensor, Tensor, Tensor>

#define TUPLE_FOUR_TENSORS std::tuple<Tensor, Tensor, Tensor, Tensor>

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
MAKE_REGISTER_FUNC(ADD_NS(conv1d), "conv1d",
                   Tensor(const Tensor &, const Tensor &,
                          const c10::optional<Tensor> &, IntArrayRef,
                          IntArrayRef, IntArrayRef, int64_t),
                   user_defined_dtype)
MAKE_REGISTER_FUNC(ADD_NS(bmm), "bmm", Tensor(const Tensor &, const Tensor &),
                   user_defined_dtype)
MAKE_REGISTER_FUNC(ADD_NS(mm), "mm", Tensor(const Tensor &, const Tensor &),
                   user_defined_dtype)
MAKE_REGISTER_FUNC(ADD_NS(baddbmm), "baddbmm",
                   Tensor(const Tensor &, const Tensor &, const Tensor &,
                          const Scalar &, const Scalar &),
                   user_defined_dtype)
MAKE_REGISTER_FUNC(ADD_NS(addmm), "addmm",
                   Tensor(const Tensor &, const Tensor &, const Tensor &,
                          const Scalar &, const Scalar &),
                   user_defined_dtype)
MAKE_REGISTER_FUNC(ADD_NS(addbmm), "addbmm",
                   Tensor(const Tensor &, const Tensor &, const Tensor &,
                          const Scalar &, const Scalar &),
                   user_defined_dtype)
// fp32 cast policy a.k.a BlackList
MAKE_REGISTER_FUNC(ADD_NS(avg_pool1d), "avg_pool1d",
                   Tensor(const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef,
                          bool, bool),
                   fp32)
MAKE_REGISTER_FUNC(ADD_NS(avg_pool3d), "avg_pool3d",
                   Tensor(const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef,
                          bool, bool, c10::optional<int64_t>),
                   fp32)
MAKE_REGISTER_FUNC(ADD_NS(binary_cross_entropy), "binary_cross_entropy",
                   Tensor(const Tensor &, const Tensor &,
                          const c10::optional<Tensor> &, int64_t),
                   fp32)
MAKE_REGISTER_FUNC(ADD_NS(binary_cross_entropy_with_logits),
                   "binary_cross_entropy_with_logits",
                   Tensor(const Tensor &, const Tensor &,
                          const c10::optional<Tensor> &,
                          const c10::optional<Tensor> &, int64_t),
                   fp32)
MAKE_REGISTER_FUNC(
    ADD_NS(hardshrink),
    "hardshrink",
    Tensor(const Tensor&, const Scalar&),
    fp32)
MAKE_REGISTER_FUNC(
    ADD_NS(elu),
    "elu",
    Tensor(const Tensor&, const Scalar&, const Scalar&, const Scalar&),
    fp32)
MAKE_REGISTER_FUNC(ADD_NS(glu), "glu", Tensor(const Tensor&, int64_t), fp32)
MAKE_REGISTER_FUNC(
    ADD_NS(hardsigmoid),
    "hardsigmoid",
    Tensor(const Tensor&),
    fp32)
MAKE_REGISTER_FUNC(ADD_NS(hardswish), "hardswish", Tensor(const Tensor&), fp32)
MAKE_REGISTER_FUNC(
    ADD_NS(log_sigmoid),
    "log_sigmoid",
    Tensor(const Tensor&),
    fp32)
MAKE_REGISTER_FUNC(
    ADD_NS(log_sigmoid_forward),
    "log_sigmoid_forward",
    TUPLE_TWO_TENSORS(const Tensor&),
    fp32)
MAKE_REGISTER_FUNC(
    ADD_NS(prelu),
    "prelu",
    Tensor(const Tensor&, const Tensor&),
    fp32)
MAKE_REGISTER_FUNC(ADD_NS(selu), "selu", Tensor(const Tensor&), fp32)
MAKE_REGISTER_FUNC(
    ADD_NS(celu),
    "celu",
    Tensor(const Tensor&, const Scalar&),
    fp32)
MAKE_REGISTER_FUNC(
    ADD_NS(softplus),
    "softplus",
    Tensor(const Tensor&, const Scalar&, const Scalar&),
    fp32)
MAKE_REGISTER_FUNC(
    ADD_NS(softshrink),
    "softshrink",
    Tensor(const Tensor&, const Scalar&),
    fp32)
MAKE_REGISTER_FUNC(
    ADD_NS(instance_norm),
    "instance_norm",
    Tensor(
        const at::Tensor& input,
        const c10::optional<at::Tensor>&,
        const c10::optional<at::Tensor>&,
        const c10::optional<at::Tensor>&,
        const c10::optional<at::Tensor>&,
        bool,
        double,
        double,
        bool),
    fp32)
MAKE_REGISTER_FUNC(
    ADD_NS(multinomial),
    "multinomial",
    Tensor(const Tensor&, int64_t, bool, c10::optional<at::Generator>),
    fp32)
MAKE_REGISTER_FUNC(
    ADD_NS(poisson),
    "poisson",
    Tensor(const Tensor&, c10::optional<at::Generator>),
    fp32)
MAKE_REGISTER_FUNC(
    ADD_NS(grid_sampler),
    "grid_sampler",
    Tensor(const Tensor&, const Tensor&, int64_t, int64_t, bool),
    fp32)
MAKE_REGISTER_FUNC(
    ADD_NS(grid_sampler_2d),
    "grid_sampler_2d",
    Tensor(const Tensor&, const Tensor&, int64_t, int64_t, bool),
    fp32)
MAKE_REGISTER_FUNC(
    ADD_NS(_grid_sampler_2d_cpu_fallback),
    "_grid_sampler_2d_cpu_fallback",
    Tensor(const Tensor&, const Tensor&, int64_t, int64_t, bool),
    fp32)
MAKE_REGISTER_FUNC(
    ADD_NS(grid_sampler_3d),
    "grid_sampler_3d",
    Tensor(const Tensor&, const Tensor&, int64_t, int64_t, bool),
    fp32)
MAKE_REGISTER_FUNC(
    ADD_NS(atan2),
    "atan2",
    Tensor(const Tensor&, const Tensor&),
    fp32)
MAKE_REGISTER_FUNC(
    ADD_NS(logaddexp),
    "logaddexp",
    Tensor(const Tensor&, const Tensor&),
    fp32)
MAKE_REGISTER_FUNC(
    ADD_NS(logaddexp2),
    "logaddexp2",
    Tensor(const Tensor&, const Tensor&),
    fp32)
MAKE_REGISTER_FUNC(ADD_NS(logdet), "logdet", Tensor(const Tensor&), fp32)
MAKE_REGISTER_FUNC(
    ADD_NS(linalg_slogdet),
    "linalg_slogdet",
    TUPLE_TWO_TENSORS(const Tensor&),
    fp32)
MAKE_REGISTER_FUNC(
    ADD_NS(linalg_det),
    "linalg_det",
    Tensor(const Tensor&),
    fp32)
MAKE_REGISTER_FUNC(ADD_NS(polar), "polar",
                   Tensor(const Tensor &, const Tensor &), fp32)
MAKE_REGISTER_FUNC(ADD_NS(fmod), "fmod.Tensor",
                   Tensor(const Tensor &, const Tensor &), fp32)
MAKE_REGISTER_FUNC(ADD_NS(fmod), "fmod.Scalar",
                   Tensor(const Tensor &, const Scalar &), fp32)
MAKE_REGISTER_FUNC(ADD_NS(prod), "prod",
                   Tensor(const Tensor &, c10::optional<at::ScalarType>), fp32)
MAKE_REGISTER_FUNC(ADD_NS(prod), "prod.dim_int",
                   Tensor(const Tensor &, int64_t, bool,
                          c10::optional<at::ScalarType>),
                   fp32)
MAKE_REGISTER_FUNC(ADD_NS(prod), "prod.dim_Dimname",
                   Tensor(const Tensor &, at::Dimname, bool,
                          c10::optional<at::ScalarType>),
                   fp32)
MAKE_REGISTER_FUNC(ADD_NS(quantile), "quantile",
                   Tensor(const Tensor &, const Tensor &,
                          c10::optional<int64_t>, bool),
                   fp32)
MAKE_REGISTER_FUNC(ADD_NS(quantile), "quantile.scalar",
                   Tensor(const Tensor &, double, c10::optional<int64_t>, bool),
                   fp32)
MAKE_REGISTER_FUNC(ADD_NS(quantile), "quantile.new",
                   Tensor(const Tensor &, const Tensor &,
                          c10::optional<int64_t>, bool, c10::string_view),
                   fp32)
MAKE_REGISTER_FUNC(ADD_NS(quantile), "quantile.new_scalar",
                   Tensor(const Tensor &, double, c10::optional<int64_t>, bool,
                          c10::string_view),
                   fp32)
MAKE_REGISTER_FUNC(ADD_NS(nanquantile), "nanquantile",
                   Tensor(const Tensor &, const Tensor &,
                          c10::optional<int64_t>, bool),
                   fp32)
MAKE_REGISTER_FUNC(ADD_NS(nanquantile), "nanquantile.scalar",
                   Tensor(const Tensor &, double, c10::optional<int64_t>, bool),
                   fp32)
MAKE_REGISTER_FUNC(ADD_NS(nanquantile), "nanquantile.new",
                   Tensor(const Tensor &, const Tensor &,
                          c10::optional<int64_t>, bool, c10::string_view),
                   fp32)
MAKE_REGISTER_FUNC(ADD_NS(nanquantile), "nanquantile.new_scalar",
                   Tensor(const Tensor &, double, c10::optional<int64_t>, bool,
                          c10::string_view),
                   fp32)
MAKE_REGISTER_FUNC(ADD_NS(stft), "stft",
                   Tensor(const Tensor &, int64_t, c10::optional<int64_t>,
                          c10::optional<int64_t>, const c10::optional<Tensor> &,
                          bool, c10::optional<bool>, c10::optional<bool>),
                   fp32)
MAKE_REGISTER_FUNC(ADD_NS(cdist), "cdist",
                   Tensor(const Tensor &, const Tensor &, double,
                          c10::optional<int64_t>),
                   fp32)
MAKE_REGISTER_FUNC(ADD_NS(cumprod), "cumprod",
                   Tensor(const Tensor &, int64_t,
                          c10::optional<at::ScalarType>),
                   fp32)
MAKE_REGISTER_FUNC(ADD_NS(cumprod), "cumprod.dimname",
                   Tensor(const Tensor &, at::Dimname,
                          c10::optional<at::ScalarType>),
                   fp32)
MAKE_REGISTER_FUNC(ADD_NS(cumsum), "cumsum",
                   Tensor(const Tensor &, int64_t,
                          c10::optional<at::ScalarType>),
                   fp32)
MAKE_REGISTER_FUNC(ADD_NS(cumsum), "cumsum.dimname",
                   Tensor(const Tensor &, at::Dimname,
                          c10::optional<at::ScalarType>),
                   fp32)
MAKE_REGISTER_FUNC(ADD_NS(diag), "diag", Tensor(const Tensor &, int64_t), fp32)
MAKE_REGISTER_FUNC(ADD_NS(diagflat), "diagflat",
                   Tensor(const Tensor &, int64_t), fp32)
MAKE_REGISTER_FUNC(ADD_NS(histc), "histc",
                   Tensor(const Tensor &, int64_t, const at::Scalar &,
                          const at::Scalar &),
                   fp32)
MAKE_REGISTER_FUNC(ADD_NS(logcumsumexp), "logcumsumexp",
                   Tensor(const Tensor &, int64_t), fp32)
MAKE_REGISTER_FUNC(ADD_NS(trace), "trace", Tensor(const Tensor &), fp32)
MAKE_REGISTER_FUNC(ADD_NS(vander), "vander",
                   Tensor(const Tensor &, c10::optional<int64_t>, bool), fp32)
MAKE_REGISTER_FUNC(ADD_NS(view_as_complex), "view_as_complex",
                   Tensor(const Tensor &), fp32)
MAKE_REGISTER_FUNC(ADD_NS(cholesky), "cholesky", Tensor(const Tensor &, bool),
                   fp32)
MAKE_REGISTER_FUNC(ADD_NS(cholesky_inverse), "cholesky_inverse",
                   Tensor(const Tensor &, bool), fp32)
MAKE_REGISTER_FUNC(ADD_NS(cholesky_solve), "cholesky_solve",
                   Tensor(const Tensor &, const Tensor &, bool), fp32)
MAKE_REGISTER_FUNC(ADD_NS(inverse), "inverse", Tensor(const Tensor &), fp32)
MAKE_REGISTER_FUNC(ADD_NS(lu_solve), "lu_solve",
                   Tensor(const Tensor &, const Tensor &, const Tensor &), fp32)
MAKE_REGISTER_FUNC(ADD_NS(matrix_rank), "matrix_rank",
                   Tensor(const Tensor &, bool), fp32)
MAKE_REGISTER_FUNC(ADD_NS(orgqr), "orgqr",
                   Tensor(const Tensor &, const Tensor &), fp32)
MAKE_REGISTER_FUNC(ADD_NS(ormqr), "ormqr",
                   Tensor(const Tensor &, const Tensor &, const Tensor &, bool,
                          bool),
                   fp32)
MAKE_REGISTER_FUNC(ADD_NS(pinverse), "pinverse", Tensor(const Tensor &, double),
                   fp32)
MAKE_REGISTER_FUNC(
    ADD_NS(max_pool1d),
    "max_pool1d",
    Tensor(
        const Tensor&,
        IntArrayRef,
        IntArrayRef,
        IntArrayRef,
        IntArrayRef,
        bool),
    fp32)
MAKE_REGISTER_FUNC(ADD_NS(max_pool3d), "max_pool3d",
                   Tensor(const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef,
                          IntArrayRef, bool),
                   fp32)
MAKE_REGISTER_FUNC(ADD_NS(max_unpool2d), "max_unpool2d",
                   Tensor(const Tensor &, const Tensor &, IntArrayRef), fp32)
MAKE_REGISTER_FUNC(ADD_NS(max_unpool3d), "max_unpool3d",
                   Tensor(const Tensor &, const Tensor &, IntArrayRef,
                          IntArrayRef, IntArrayRef),
                   fp32)
MAKE_REGISTER_FUNC(
    ADD_NS(adaptive_avg_pool1d),
    "adaptive_avg_pool1d",
    Tensor(const Tensor&, IntArrayRef),
    fp32)
MAKE_REGISTER_FUNC(ADD_NS(adaptive_avg_pool3d), "adaptive_avg_pool3d",
                   Tensor(const Tensor &, IntArrayRef), fp32)
MAKE_REGISTER_FUNC(ADD_NS(reflection_pad1d), "reflection_pad1d",
                   Tensor(const Tensor &, IntArrayRef), fp32)
MAKE_REGISTER_FUNC(ADD_NS(reflection_pad2d), "reflection_pad2d",
                   Tensor(const Tensor &, IntArrayRef), fp32)
MAKE_REGISTER_FUNC(
    ADD_NS(reflection_pad3d),
    "reflection_pad3d",
    Tensor(const Tensor&, IntArrayRef),
    fp32)
MAKE_REGISTER_FUNC(ADD_NS(replication_pad1d), "replication_pad1d",
                   Tensor(const Tensor &, IntArrayRef), fp32)
MAKE_REGISTER_FUNC(ADD_NS(replication_pad2d), "replication_pad2d",
                   Tensor(const Tensor &, IntArrayRef), fp32)
MAKE_REGISTER_FUNC(ADD_NS(replication_pad3d), "replication_pad3d",
                   Tensor(const Tensor &, IntArrayRef), fp32)
MAKE_REGISTER_FUNC(ADD_NS(group_norm), "group_norm",
                   Tensor(const Tensor &, int64_t,
                          const c10::optional<Tensor> &,
                          const c10::optional<Tensor> &, double, bool),
                   fp32)
MAKE_REGISTER_FUNC(ADD_NS(mse_loss), "mse_loss",
                   Tensor(const Tensor &, const Tensor &, int64_t), fp32)
MAKE_REGISTER_FUNC(ADD_NS(ctc_loss), "ctc_loss.IntList",
                   Tensor(const Tensor &, const Tensor &, IntArrayRef,
                          IntArrayRef, int64_t, int64_t, bool),
                   fp32)
MAKE_REGISTER_FUNC(ADD_NS(ctc_loss), "ctc_loss.Tensor",
                   Tensor(const Tensor &, const Tensor &, const Tensor &,
                          const Tensor &, int64_t, int64_t, bool),
                   fp32)
MAKE_REGISTER_FUNC(ADD_NS(kl_div), "kl_div",
                   Tensor(const Tensor &, const Tensor &, int64_t, bool), fp32)
MAKE_REGISTER_FUNC(ADD_NS(multilabel_margin_loss), "multilabel_margin_loss",
                   Tensor(const Tensor &, const Tensor &, int64_t), fp32)
MAKE_REGISTER_FUNC(ADD_NS(fft_fft), "fft_fft",
                   Tensor(const Tensor &, c10::optional<int64_t>, int64_t,
                          c10::optional<c10::string_view>),
                   fp32)
MAKE_REGISTER_FUNC(ADD_NS(fft_ifft), "fft_ifft",
                   Tensor(const Tensor &, c10::optional<int64_t>, int64_t,
                          c10::optional<c10::string_view>),
                   fp32)
MAKE_REGISTER_FUNC(ADD_NS(fft_fft2), "fft_fft2",
                   Tensor(const Tensor &, c10::optional<at::IntArrayRef>,
                          at::IntArrayRef, c10::optional<c10::string_view>),
                   fp32)
MAKE_REGISTER_FUNC(ADD_NS(fft_ifft2), "fft_ifft2",
                   Tensor(const Tensor &, c10::optional<at::IntArrayRef>,
                          at::IntArrayRef, c10::optional<c10::string_view>),
                   fp32)
MAKE_REGISTER_FUNC(ADD_NS(fft_fftn), "fft_fftn",
                   Tensor(const Tensor &, c10::optional<at::IntArrayRef>,
                          c10::optional<at::IntArrayRef>,
                          c10::optional<c10::string_view>),
                   fp32)
MAKE_REGISTER_FUNC(ADD_NS(fft_ifftn), "fft_ifftn",
                   Tensor(const Tensor &, c10::optional<at::IntArrayRef>,
                          c10::optional<at::IntArrayRef>,
                          c10::optional<c10::string_view>),
                   fp32)
MAKE_REGISTER_FUNC(ADD_NS(fft_rfft), "fft_rfft",
                   Tensor(const Tensor &, c10::optional<int64_t>, int64_t,
                          c10::optional<c10::string_view>),
                   fp32)
MAKE_REGISTER_FUNC(ADD_NS(fft_irfft), "fft_irfft",
                   Tensor(const Tensor &, c10::optional<int64_t>, int64_t,
                          c10::optional<c10::string_view>),
                   fp32)
MAKE_REGISTER_FUNC(ADD_NS(fft_rfft2), "fft_rfft2",
                   Tensor(const Tensor &, c10::optional<at::IntArrayRef>,
                          at::IntArrayRef, c10::optional<c10::string_view>),
                   fp32)
MAKE_REGISTER_FUNC(ADD_NS(fft_irfft2), "fft_irfft2",
                   Tensor(const Tensor &, c10::optional<at::IntArrayRef>,
                          at::IntArrayRef, c10::optional<c10::string_view>),
                   fp32)
MAKE_REGISTER_FUNC(ADD_NS(fft_rfftn), "fft_rfftn",
                   Tensor(const Tensor &, c10::optional<at::IntArrayRef>,
                          c10::optional<at::IntArrayRef>,
                          c10::optional<c10::string_view>),
                   fp32)
MAKE_REGISTER_FUNC(ADD_NS(fft_irfftn), "fft_irfftn",
                   Tensor(const Tensor &, c10::optional<at::IntArrayRef>,
                          c10::optional<at::IntArrayRef>,
                          c10::optional<c10::string_view>),
                   fp32)
MAKE_REGISTER_FUNC(ADD_NS(fft_hfft), "fft_hfft",
                   Tensor(const Tensor &, c10::optional<int64_t>, int64_t,
                          c10::optional<c10::string_view>),
                   fp32)
MAKE_REGISTER_FUNC(ADD_NS(fft_ihfft), "fft_ihfft",
                   Tensor(const Tensor &, c10::optional<int64_t>, int64_t,
                          c10::optional<c10::string_view>),
                   fp32)
MAKE_REGISTER_FUNC(ADD_NS(conv_tbc), "conv_tbc",
                   Tensor(const Tensor &, const Tensor &, const Tensor &,
                          int64_t),
                   fp32)
MAKE_REGISTER_FUNC(
    ADD_NS(linalg_pinv),
    "linalg_pinv",
    Tensor(const Tensor&, double, bool),
    fp32)
MAKE_REGISTER_FUNC(
    ADD_NS(linalg_pinv),
    "linalg_pinv.rcond_tensor",
    Tensor(const Tensor&, const Tensor&, bool),
    fp32)
MAKE_REGISTER_FUNC(ADD_NS(linalg_matrix_norm), "linalg_matrix_norm",
                   Tensor(const Tensor &, const at::Scalar &, at::IntArrayRef,
                          bool, c10::optional<at::ScalarType>),
                   fp32)
MAKE_REGISTER_FUNC(ADD_NS(linalg_matrix_norm), "linalg_matrix_norm.str_ord",
                   Tensor(const Tensor &, c10::string_view, at::IntArrayRef, bool,
                          c10::optional<at::ScalarType>),
                   fp32)
MAKE_REGISTER_FUNC(ADD_NS(linalg_cond), "linalg_cond",
                   Tensor(const Tensor &, const c10::optional<at::Scalar> &),
                   fp32)
MAKE_REGISTER_FUNC(ADD_NS(linalg_cond), "linalg_cond.p_str",
                   Tensor(const Tensor &, c10::string_view), fp32)
MAKE_REGISTER_FUNC(
    ADD_NS(linalg_matrix_rank),
    "linalg_matrix_rank",
    Tensor(const Tensor &, const c10::optional<double>, bool),
    fp32)
MAKE_REGISTER_FUNC(
    ADD_NS(linalg_matrix_rank),
    "linalg_matrix_rank.tol_tensor",
    Tensor(const Tensor &, const Tensor &, bool),
    fp32)
MAKE_REGISTER_FUNC(ADD_NS(linalg_solve), "linalg_solve",
                   Tensor(const Tensor &, const Tensor &), fp32)
MAKE_REGISTER_FUNC(
    ADD_NS(linalg_cholesky),
    "linalg_cholesky",
    Tensor(const Tensor&, bool),
    fp32)
MAKE_REGISTER_FUNC(ADD_NS(linalg_svdvals), "linalg_svdvals",
                   Tensor(const Tensor &), fp32)
MAKE_REGISTER_FUNC(ADD_NS(linalg_eigvals), "linalg_eigvals",
                   Tensor(const Tensor &), fp32)
MAKE_REGISTER_FUNC(ADD_NS(linalg_eigvalsh), "linalg_eigvalsh",
                   Tensor(const Tensor &, c10::string_view), fp32)
MAKE_REGISTER_FUNC(ADD_NS(linalg_inv), "linalg_inv", Tensor(const Tensor &),
                   fp32)
MAKE_REGISTER_FUNC(ADD_NS(linalg_householder_product),
                   "linalg_householder_product",
                   Tensor(const Tensor &, const Tensor &), fp32)
MAKE_REGISTER_FUNC(ADD_NS(linalg_tensorinv), "linalg_tensorinv",
                   Tensor(const Tensor &, int64_t), fp32)
MAKE_REGISTER_FUNC(ADD_NS(linalg_tensorsolve), "linalg_tensorsolve",
                   Tensor(const Tensor &, const Tensor &,
                          c10::optional<at::IntArrayRef>),
                   fp32)
MAKE_REGISTER_FUNC(ADD_NS(fake_quantize_per_tensor_affine),
                   "fake_quantize_per_tensor_affine",
                   Tensor(const Tensor &, double, int64_t, int64_t, int64_t),
                   fp32)
MAKE_REGISTER_FUNC(
    ADD_NS(fake_quantize_per_channel_affine),
    "fake_quantize_per_channel_affine",
    Tensor(
        const Tensor&,
        const Tensor&,
        const Tensor&,
        int64_t,
        int64_t,
        int64_t),
    fp32)
MAKE_REGISTER_FUNC(ADD_NS(cummax), "cummax",
                   TUPLE_TWO_TENSORS(const Tensor &, int64_t), fp32)
MAKE_REGISTER_FUNC(ADD_NS(cummax), "cummax.dimname",
                   TUPLE_TWO_TENSORS(const Tensor &, at::Dimname), fp32)
MAKE_REGISTER_FUNC(ADD_NS(cummin), "cummin",
                   TUPLE_TWO_TENSORS(const Tensor &, int64_t), fp32)
MAKE_REGISTER_FUNC(ADD_NS(cummin), "cummin.dimname",
                   TUPLE_TWO_TENSORS(const Tensor &, at::Dimname), fp32)
MAKE_REGISTER_FUNC(ADD_NS(eig), "eig", TUPLE_TWO_TENSORS(const Tensor &, bool),
                   fp32)
MAKE_REGISTER_FUNC(ADD_NS(geqrf), "geqrf", TUPLE_TWO_TENSORS(const Tensor &),
                   fp32)
MAKE_REGISTER_FUNC(ADD_NS(lstsq), "lstsq",
                   TUPLE_TWO_TENSORS(const Tensor &, const Tensor &), fp32)
MAKE_REGISTER_FUNC(ADD_NS(_lu_with_info), "_lu_with_info",
                   TUPLE_THREE_TENSORS(const Tensor &, bool, bool), fp32)
MAKE_REGISTER_FUNC(ADD_NS(qr), "qr", TUPLE_TWO_TENSORS(const Tensor &, bool),
                   fp32)
MAKE_REGISTER_FUNC(ADD_NS(solve), "solve",
                   TUPLE_TWO_TENSORS(const Tensor &, const Tensor &), fp32)
MAKE_REGISTER_FUNC(ADD_NS(svd), "svd",
                   TUPLE_THREE_TENSORS(const Tensor &, bool, bool), fp32)
MAKE_REGISTER_FUNC(ADD_NS(symeig), "symeig",
                   TUPLE_TWO_TENSORS(const Tensor &, bool, bool), fp32)
MAKE_REGISTER_FUNC(ADD_NS(triangular_solve), "triangular_solve",
                   TUPLE_TWO_TENSORS(const Tensor &, const Tensor &, bool, bool,
                                     bool),
                   fp32)
MAKE_REGISTER_FUNC(ADD_NS(fractional_max_pool2d), "fractional_max_pool2d",
                   TUPLE_TWO_TENSORS(const Tensor &, IntArrayRef, IntArrayRef,
                                     const Tensor &),
                   fp32)
MAKE_REGISTER_FUNC(ADD_NS(fractional_max_pool3d), "fractional_max_pool3d",
                   TUPLE_TWO_TENSORS(const Tensor &, IntArrayRef, IntArrayRef,
                                     const Tensor &),
                   fp32)
MAKE_REGISTER_FUNC(ADD_NS(adaptive_max_pool1d), "adaptive_max_pool1d",
                   TUPLE_TWO_TENSORS(const Tensor &, IntArrayRef), fp32)
MAKE_REGISTER_FUNC(ADD_NS(adaptive_max_pool3d), "adaptive_max_pool3d",
                   TUPLE_TWO_TENSORS(const Tensor &, IntArrayRef), fp32)
MAKE_REGISTER_FUNC(ADD_NS(multilabel_margin_loss_forward),
                   "multilabel_margin_loss_forward",
                   TUPLE_TWO_TENSORS(const Tensor &, const Tensor &, int64_t),
                   fp32)
MAKE_REGISTER_FUNC(ADD_NS(linalg_qr), "linalg_qr",
                   TUPLE_TWO_TENSORS(const Tensor &, c10::string_view), fp32)
MAKE_REGISTER_FUNC(
    ADD_NS(linalg_cholesky_ex),
    "linalg_cholesky_ex",
    TUPLE_TWO_TENSORS(const Tensor&, bool, bool),
    fp32)
MAKE_REGISTER_FUNC(ADD_NS(linalg_svd), "linalg_svd",
                   TUPLE_THREE_TENSORS(const Tensor &, bool), fp32)
MAKE_REGISTER_FUNC(ADD_NS(linalg_eig), "linalg_eig",
                   TUPLE_TWO_TENSORS(const Tensor &), fp32)
MAKE_REGISTER_FUNC(ADD_NS(linalg_eigh), "linalg_eigh",
                   TUPLE_TWO_TENSORS(const Tensor &, c10::string_view), fp32)
MAKE_REGISTER_FUNC(ADD_NS(linalg_lstsq), "linalg_lstsq",
                   TUPLE_FOUR_TENSORS(const Tensor &, const Tensor &,
                                      c10::optional<double>,
                                      c10::optional<c10::string_view>),
                   fp32)
MAKE_REGISTER_FUNC(ADD_NS(linalg_inv_ex), "linalg_inv_ex",
                   TUPLE_TWO_TENSORS(const Tensor &, bool), fp32)
MAKE_REGISTER_FUNC(
    ADD_NS(special_i1),
    "special_i1",
    Tensor(const Tensor&),
    fp32)
MAKE_REGISTER_FUNC(
    ADD_NS(special_i1e),
    "special_i1e",
    Tensor(const Tensor&),
    fp32)
MAKE_REGISTER_FUNC(ADD_NS(pdist), "pdist", Tensor(const Tensor&, double), fp32)

// promote
MAKE_REGISTER_FUNC(ADD_NS(cat), "cat", Tensor(TensorList, int64_t), promote)
MAKE_REGISTER_FUNC(ADD_NS(stack), "stack", Tensor(TensorList, int64_t), promote)
MAKE_REGISTER_FUNC(ADD_NS(index_copy), "index_copy",
                   Tensor(const Tensor &, int64_t, const Tensor &,
                          const Tensor &),
                   promote)
MAKE_REGISTER_FUNC(ADD_NS(index_copy), "index_copy.dimname",
                   Tensor(const Tensor &, at::Dimname, const Tensor &,
                          const Tensor &),
                   promote)

#undef TUPLE_TWO_TENSORS
#undef TUPLE_THREE_TENSORS
#undef TUPLE_FOUR_TENSORS
#undef MAKE_REGISTER_FUNC

IPEX_TORCH_LIBRARY_IMPL(aten, AutocastCPU, m) {
  // for int8 path
  m.impl(TORCH_SELECTIVE_NAME("aten::conv2d"),
         TORCH_FN((&torch_ipex::autocast::conv2d)));
  m.impl(TORCH_SELECTIVE_NAME("aten::conv3d"),
         TORCH_FN((&torch_ipex::autocast::conv3d)));
  m.impl(TORCH_SELECTIVE_NAME("aten::conv_transpose3d.input"),
         TORCH_FN((&torch_ipex::autocast::conv_transpose3d)));
  m.impl(TORCH_SELECTIVE_NAME("aten::_convolution"),
         TORCH_FN((&torch_ipex::autocast::_convolution)));
  m.impl(TORCH_SELECTIVE_NAME("aten::_convolution.deprecated"),
         TORCH_FN((&torch_ipex::autocast::_convolution_deprecated)));
  m.impl(TORCH_SELECTIVE_NAME("aten::batch_norm"),
         TORCH_FN((&torch_ipex::autocast::batch_norm)));
  // m.impl(TORCH_SELECTIVE_NAME("aten::linear"),
  // TORCH_FN((&torch_ipex::autocast::linear)));
  m.impl(TORCH_SELECTIVE_NAME("aten::max_pool2d"),
         TORCH_FN((&torch_ipex::autocast::max_pool2d)));
  m.impl(TORCH_SELECTIVE_NAME("aten::adaptive_avg_pool2d"),
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
  m.impl(TORCH_SELECTIVE_NAME("aten::relu"),
         TORCH_FN((&torch_ipex::autocast::relu)));
  m.impl(TORCH_SELECTIVE_NAME("aten::relu_"),
         TORCH_FN((&torch_ipex::autocast::relu_)));
  m.impl(TORCH_SELECTIVE_NAME("aten::sigmoid"),
         TORCH_FN((&torch_ipex::autocast::sigmoid)));
  m.impl(TORCH_SELECTIVE_NAME("aten::linear"),
         TORCH_FN((&torch_ipex::autocast::linear)));
  m.impl(TORCH_SELECTIVE_NAME("aten::add_.Tensor"),
         TORCH_FN((&torch_ipex::autocast::add_tensor_)));
  m.impl(TORCH_SELECTIVE_NAME("aten::add.Tensor"),
         TORCH_FN((&torch_ipex::autocast::add_tensor)));
  m.impl(TORCH_SELECTIVE_NAME("aten::dropout"),
         TORCH_FN((&torch_ipex::autocast::dropout)));
  m.impl(TORCH_SELECTIVE_NAME("aten::gelu"),
         TORCH_FN((&torch_ipex::autocast::gelu)));
  m.impl(TORCH_SELECTIVE_NAME("aten::lstm.input"),
         TORCH_FN((&torch_ipex::autocast::lstm_aten)));
  m.impl(TORCH_SELECTIVE_NAME("aten::flatten.using_ints"),
         TORCH_FN((&torch_ipex::autocast::flatten)));
  m.impl(
      TORCH_SELECTIVE_NAME("aten::matmul"),
      TORCH_FN((&torch_ipex::autocast::matmul)));
}

} // namespace autocast
} // namespace torch_ipex
