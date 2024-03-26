#include <ATen/ATen.h>
#include <ATen/native/TensorIterator.h>

#include <ATen/xpu/XPUGeneratorImpl.h>
#include "comm/ATDispatch.h"
#include "comm/AccumulateType.h"
#include "comm/RegistrationDeclarations.h"

#include "DistributionTemplates.h"
#include "Distributions.h"
#include "Random.h"
#include "RandomEngine.h"
#include "comm/ApplyUtils.h"

namespace at {
namespace AtenIpexTypeXPU {

static inline double lgamma_integer(int a) {
  double s;
  double t;
  double fa = fabs((float)a);
  double sum;

  if (a > 8) {
    /* Stirling approximation; coefficients from Hart et al, "Computer
     * Approximations", Wiley 1968. Approximation 5404.
     */
    s = 1.0 / fa;
    t = s * s;
    sum = -0.1633436431e-2;
    sum = sum * t + 0.83645878922e-3;
    sum = sum * t - 0.5951896861197e-3;
    sum = sum * t + 0.793650576493454e-3;
    sum = sum * t - 0.277777777735865004e-2;
    sum = sum * t + 0.833333333333331018375e-1;
    sum = sum * s + 0.918938533204672;
    s = 0.5 * logf(fa);
    t = fa - 0.5;
    s = s * t;
    t = s - fa;
    s = s + sum;
    t = t + s;
    return t;
  } else {
    switch (a) {
      case 1:
        return 0.000000000000000000e-1;
      case 2:
        return 0.000000000000000000e-1;
      case 3:
        return 6.931471805599453094e-1;
      case 4:
        return 1.791759469228055001e0;
      case 5:
        return 3.178053830347945620e0;
      case 6:
        return 4.787491742782045994e0;
      case 7:
        return 6.579251212010100995e0;
      case 8:
        return 8.525161361065414300e0;
      default:
        return 1.060460290274525023e1;
    }
  }
}

/* Computes regularized gamma function:  gammainc(a,x)/gamma(a) */
static inline float pgammainc(float a, float x) {
  float t, alpha, beta;

  /* First level parametrization constants */
  float ma1 = 1.43248035075540910f, ma2 = 0.12400979329415655f,
        ma3 = 0.00025361074907033f, mb1 = 0.21096734870196546f,
        mb2 = 1.97381164089999420f, mb3 = 0.94201734077887530f;

  /* Second level parametrization constants (depends only on a) */

  alpha = 1 / sqrtf(a - ma2);
  alpha = ma1 * alpha + ma3;
  beta = 1 / sqrtf(a - mb2);
  beta = mb1 * beta + mb3;

  /* Final approximation (depends on a and x) */

  t = a - x;
  t = alpha * t - beta;
  t = 1.0f + expf(t);
  t = t * t;
  t = 1 / t;

  /* Negative a,x or a,x=NAN requires special handling */
  // t = !(x > 0 && a >= 0) ? 0.0 : t;
  return t;
}

/* Computes inverse of pgammainc */
static inline float pgammaincinv(float a, float y) {
  float t, alpha, beta;

  /* First level parametrization constants */

  float ma1 = 1.43248035075540910f, ma2 = 0.12400979329415655f,
        ma3 = 0.00025361074907033f, mb1 = 0.21096734870196546f,
        mb2 = 1.97381164089999420f, mb3 = 0.94201734077887530f;

  /* Second level parametrization constants (depends only on a) */

  alpha = 1.0f / sqrtf(a - ma2);
  alpha = ma1 * alpha + ma3;
  beta = 1.0f / sqrtf(a - mb2);
  beta = mb1 * beta + mb3;

  /* Final approximation (depends on a and y) */

  t = 1.0f / sqrtf(y) - 1.0f;
  t = logf(t);
  t = beta + t;
  t = -t * (1 / alpha) + a;
  /* Negative a,x or a,x=NAN requires special handling */
  // t = !(y > 0 && a >= 0) ? 0.0 : t;
  return t;
}

/* Rejection Method for Poisson distribution based on gammainc approximation */
static inline float rand_poisson_gammainc(
    AtenIpexTypeXPU::randStatePhilox4_32_10_t* state,
    float lambda) {
  float y, x, t, z, v;
  float logl = logf(lambda);
  while (true) {
    y = rand_uniform(state);
    x = pgammaincinv(lambda, y);
    x = floorf(x);
    z = rand_uniform(state);
    v = (pgammainc(lambda, x + 1.0f) - pgammainc(lambda, x)) * 1.3f;
    z = z * v;
    t = (float)expf(
        -lambda + x * logl - (float)lgamma_integer((int)(1.0f + x)));
    if ((z < t) && (v >= 1e-20))
      break;
  }
  return (unsigned int)x;
}

// Donald E. Knuth Seminumerical Algorithms. The Art of Computer Programming,
// Volume 2
static inline float rand_poisson_knuth(
    AtenIpexTypeXPU::randStatePhilox4_32_10_t* state,
    float lambda) {
  unsigned int k = 0;
  float p = expf(lambda);
  do {
    k++;
    p *= rand_uniform(state);
  } while (p > 1.0);
  return k - 1;
}

static inline double rand_poisson(
    AtenIpexTypeXPU::randStatePhilox4_32_10_t* state,
    double lambda) {
  if (lambda < 64)
    return rand_poisson_knuth(state, (float)lambda);
  if (lambda > 4000)
    return (
        float)((sqrtf(lambda) * rand_normal_double(state)) + lambda + 0.5); // Round to nearest
  return rand_poisson_gammainc(state, (float)lambda);
}

template <typename scalar_t>
struct poisson_sycl_kernel_functor {
  void operator()(
      AtenIpexTypeXPU::randStatePhilox4_32_10_t& state,
      scalar_t& ret,
      scalar_t lambda) const {
    ret = rand_poisson(&state, lambda);
  }
};

template <typename scalar_t>
void poisson_sycl_kernel(
    at::Tensor& ret,
    const at::Tensor& lambda,
    PhiloxState philox_args) {
  at::TensorIterator iter =
      at::TensorIteratorConfig().add_output(ret).add_input(lambda).build();
  poisson_sycl_kernel_functor<scalar_t> f;
  distribution_unary_kernel<scalar_t, scalar_t>(iter, philox_args, f);
}

Tensor poisson(const Tensor& lambda, c10::optional<Generator> gen_) {
  auto gen = get_generator_or_default<at::XPUGeneratorImpl>(
      gen_, at::xpu::detail::getDefaultXPUGenerator());
  std::pair<uint64_t, uint64_t> seeds;
  {
    std::lock_guard<std::mutex> lock(gen->mutex_);
    seeds = gen->philox_engine_inputs(10);
  }
  PhiloxState rng_engine_inputs(std::get<0>(seeds), std::get<1>(seeds));
  Tensor ret = at::empty(lambda.sizes(), lambda.options());
  IPEX_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      ret.scalar_type(),
      "poisson_sycl",
      [&]() { poisson_sycl_kernel<scalar_t>(ret, lambda, rng_engine_inputs); });
  return ret;
}

} // namespace AtenIpexTypeXPU
} // namespace at