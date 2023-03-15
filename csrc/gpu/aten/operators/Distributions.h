#pragma once

#include <core/Generator.h>
#include <runtime/Utils.h>
#include "Loops.h"
#include "Random.h"
#include "comm/Math.h"
#include "comm/Numerics.h"

#define compat_exp std::exp
#define compat_ceil std::ceil
#define compat_floor std::floor
#define compat_log std::log
#define compat_pow std::pow
#define compat_sqrt std::sqrt
#define compat_tan std::tan
#define compat_abs std::abs
#define compat_log1p std::log1p

namespace at {
namespace AtenIpexTypeXPU {

template <
    typename scalar_t,
    typename accscalar_t,
    typename dist_t,
    typename transform_t>
void distribution_elementwise_grid_stride_kernel(
    at::TensorIterator& iter,
    int numel,
    std::pair<uint64_t, uint64_t> seeds,
    const dist_t dist_func,
    const transform_t transform_func) {
  constexpr int unroll_factor = sizeof(accscalar_t) <= 4 ? 4 : 2;
  auto& sycl_queue = dpcppGetCurrentQueue();
  int group_items = dpcppMaxWorkGroupSize(dpcppGetDeviceIdOfCurrentQueue());
  int group_work_size = group_items * unroll_factor;
  int num_groups = (numel + group_work_size - 1) / group_work_size;
  if (iter.is_trivial_1d()) {
    auto strides = iter.get_inner_strides();
    int stride0 = strides[0];
    auto cgf = DPCPP_Q_CGF(cgh) {
      auto out_data = (char*)iter.data_ptr(0);
      auto kfn = DPCPP_Q_KFN(sycl::nd_item<1> item) {
        int gid = item.get_group(0);
        int tid = item.get_local_id(0);
        RandomState<Philox4_32_10> state(
            seeds.first, gid * group_items + tid, seeds.second);
        int sample_id = gid * group_work_size + tid;
#pragma unroll
        for (int i = 0; i < unroll_factor; i++) {
          if (sample_id >= numel)
            return;
          auto rand = dist_func(&state);
          accscalar_t r = ScalarConvert<scalar_t, accscalar_t>::to(rand);
          scalar_t ret = transform_func(r);
          auto offset = sample_id * stride0;
          scalar_t* out = (scalar_t*)(out_data + offset);
          *out = ret;
          sample_id += group_items;
        }
      };
      cgh.parallel_for(
          sycl::nd_range<1>(
              sycl::range<1>(num_groups * group_items),
              sycl::range<1>(group_items)),
          kfn);
    };
    DPCPP_Q_SUBMIT(sycl_queue, cgf);
  } else {
    auto offset_calc = make_offset_calculator<1>(iter);
    auto cgf = DPCPP_Q_CGF(cgh) {
      auto out_data = (char*)iter.data_ptr(0);
      auto kfn = DPCPP_Q_KFN(sycl::nd_item<1> item) {
        int gid = item.get_group(0);
        int tid = item.get_local_id(0);
        RandomState<Philox4_32_10> state(
            seeds.first, gid * group_items + tid, seeds.second);
        int sample_id = gid * group_work_size + tid;
#pragma unroll
        for (int i = 0; i < unroll_factor; i++) {
          if (sample_id >= numel)
            return;
          auto rand = dist_func(&state);
          accscalar_t r = ScalarConvert<scalar_t, accscalar_t>::to(rand);
          scalar_t ret = transform_func(r);
          auto offset = offset_calc.get(sample_id)[0];
          scalar_t* out = (scalar_t*)(out_data + offset);
          *out = ret;
          sample_id += group_items;
        }
      };
      cgh.parallel_for(
          sycl::nd_range<1>(
              sycl::range<1>(num_groups * group_items),
              sycl::range<1>(group_items)),
          kfn);
    };
    DPCPP_Q_SUBMIT(sycl_queue, cgf);
  }
}

template <
    typename scalar_t,
    typename accscalar_t,
    typename dist_t,
    typename transform_t>
void distribution_nullary_kernel(
    at::TensorIterator& iter,
    xpu::dpcpp::DPCPPGeneratorImpl* gen,
    const dist_t& dist_func,
    const transform_t transform_func) {
  int64_t numel = iter.numel();
  if (numel == 0) {
    return;
  }

  std::pair<uint64_t, uint64_t> rng_engine_inputs;
  {
    // See Note [Acquire lock when using random generators]
    std::lock_guard<std::mutex> lock(gen->mutex_);
    rng_engine_inputs = gen->philox_engine_inputs(1);
  }

  if (!iter.can_use_32bit_indexing()) {
    for (auto& sub_iter : iter.with_32bit_indexing()) {
      distribution_nullary_kernel<scalar_t, accscalar_t>(
          sub_iter, gen, dist_func, transform_func);
    }
    return;
  }

  distribution_elementwise_grid_stride_kernel<scalar_t, accscalar_t>(
      iter, numel, rng_engine_inputs, dist_func, transform_func);
}

/*
 * This function is derived from the implementation of the digamma function in
 * the Cephes Math Library. See note [3-Clause BSD License for the Cephes Math
 * Library] in ATen/native/Math.h.
 */
template <typename scalar_t, typename accscalar_t>
static inline scalar_t digamma_one(scalar_t x) {
  constexpr accscalar_t PSI_10 = 2.25175258906672110764;
  if (x == 0) {
    return INFINITY;
  }
  accscalar_t additional_summand = 0;
  int x_is_integer = x == compat_floor(x);
  if (x < 0) {
    if (x_is_integer) {
      return INFINITY;
    }
    // it is more standard to write this as recursion, but
    // nvcc does not like that
    additional_summand = -c10::pi<scalar_t> / compat_tan(c10::pi<scalar_t> * x);
    x = 1 - x;
  }

  // Push x to be >= 10
  accscalar_t result = 0;
  while (x < 10) {
    result -= 1 / x;
    x += 1;
  }
  if (x == 10) {
    return result + PSI_10 + additional_summand;
  }

  // Compute asymptotic digamma
  static const accscalar_t A[] = {
      8.33333333333333333333E-2,
      -2.10927960927960927961E-2,
      7.57575757575757575758E-3,
      -4.16666666666666666667E-3,
      3.96825396825396825397E-3,
      -8.33333333333333333333E-3,
      8.33333333333333333333E-2,
  };

  accscalar_t y = 0;
  if (x < 1.0e17f) {
    accscalar_t z = 1.0 / (x * x);
    y = z * polevl<accscalar_t>(z, A, 6);
  }
  return static_cast<scalar_t>(
      result + compat_log(x) - (0.5f / x) - y + additional_summand);
}

// Approximate reparameterized gradient of Beta(x,alpha,beta) wrt alpha.
// Assumes x is close to zero and uses a Taylor expansion.
template <typename scalar_t, typename accscalar_t>
static inline scalar_t _beta_grad_alpha_small(
    scalar_t x,
    scalar_t alpha,
    scalar_t beta) {
  const scalar_t factor = digamma_one<scalar_t, accscalar_t>(alpha) -
      digamma_one<scalar_t, accscalar_t>(alpha + beta) - compat_log(x);
  scalar_t numer = 1;
  scalar_t series = numer / alpha * (factor + 1 / alpha);
  for (int i = 1; i <= 10; ++i) {
    scalar_t casted_i = static_cast<scalar_t>(i);
    numer *= (casted_i - beta) * x / casted_i;
    const scalar_t denom = alpha + casted_i;
    series += numer / denom * (factor + 1 / denom);
  }
  const scalar_t result = x * compat_pow(1 - x, -beta) * series;
  return Numerics<scalar_t>::isnan(result) ? static_cast<scalar_t>(0.f)
                                           : result;
}

// Approximate reparameterized gradient of Beta(x,alpha,beta) wrt beta.
// Assumes x is close to zero and uses a Taylor expansion.
template <typename scalar_t, typename accscalar_t>
static inline scalar_t _beta_grad_beta_small(
    scalar_t x,
    scalar_t alpha,
    scalar_t beta) {
  const scalar_t factor = digamma_one<scalar_t, accscalar_t>(alpha + beta) -
      digamma_one<scalar_t, accscalar_t>(beta);
  scalar_t numer = 1, betas = 1, dbetas = 0, series = factor / alpha;
  for (int i = 1; i <= 8; ++i) {
    scalar_t casted_i = static_cast<scalar_t>(i);
    numer *= -x / casted_i;
    dbetas = dbetas * (beta - casted_i) + betas;
    betas = betas * (beta - casted_i);
    series += numer / (alpha + casted_i) * (dbetas + factor * betas);
  }
  const scalar_t result = -compat_pow(1 - x, 1 - beta) * series;
  return Numerics<scalar_t>::isnan(result) ? static_cast<scalar_t>(0.f)
                                           : result;
}

// Approximate reparameterized gradient of Beta(x,alpha,beta) wrt alpha.
// Assumes alpha and beta are both large and uses a Rice saddle point expansion.
// To ensure numerical stability, this computation is performed at higher
// precision.
template <typename scalar_t, typename accscalar_t>
static inline scalar_t _beta_grad_alpha_mid(
    accscalar_t x,
    accscalar_t alpha,
    accscalar_t beta) {
  const accscalar_t total = alpha + beta;
  const accscalar_t mean = alpha / total;
  const accscalar_t std = compat_sqrt(alpha * beta / (total + 1)) / total;
  if (mean - 0.1f * std <= x && x <= mean + 0.1f * std) {
    // Avoid the singularity at x = mean.
    const accscalar_t poly = 47 * x * (beta * beta) * (beta * beta) +
        alpha *
            ((43 + 20 * (16 + 27 * beta) * x) * (beta * beta) * beta +
             alpha *
                 (3 * (59 + 180 * beta - 90 * x) * (beta * beta) +
                  alpha *
                      ((453 + 1620 * beta * (1 - x) - 455 * x) * beta +
                       alpha * (8 * (1 - x) * (135 * beta - 11)))));
    const accscalar_t prefactor_num =
        (1 + 12 * alpha) * (1 + 12 * beta) / (total * total);
    const accscalar_t prefactor_den =
        12960 * alpha * alpha * alpha * beta * beta * (1 + 12 * total);
    return prefactor_num / (1 - x) * poly / prefactor_den;
  }
  const accscalar_t prefactor = -x / compat_sqrt(2 * alpha * beta / total);
  const accscalar_t stirling =
      (1 + 1 / (12 * alpha) + 1 / (288 * alpha * alpha)) *
      (1 + 1 / (12 * beta) + 1 / (288 * beta * beta)) /
      (1 + 1 / (12 * total) + 1 / (288 * total * total));
  const accscalar_t term1_num = 2 * (alpha * alpha) * (x - 1) +
      alpha * beta * (x - 1) - x * (beta * beta);
  const accscalar_t axbx = alpha * (x - 1) + beta * x;
  const accscalar_t term1_den = compat_sqrt(2 * alpha / beta) *
      compat_pow(total, static_cast<accscalar_t>(1.5f)) * axbx * axbx;
  const accscalar_t term1 = term1_num / term1_den;
  const accscalar_t term2 = 0.5f * compat_log(alpha / (total * x));
  const accscalar_t term3_num = compat_sqrt(8 * alpha * beta / total);
  const accscalar_t term3_den = beta * x + alpha * (x - 1);
  const accscalar_t term3 = term3_num / term3_den;
  const accscalar_t term4_base = beta * compat_log(beta / (total * (1 - x))) +
      alpha * compat_log(alpha / (total * x));
  const accscalar_t term4 =
      compat_pow(term4_base, static_cast<accscalar_t>(-1.5f));
  const accscalar_t term1234 =
      term1 + term2 * (term3 + (x < mean ? term4 : -term4));
  return static_cast<scalar_t>(stirling * prefactor * term1234);
}

// Computes a scaled reparameterized gradient
//   -(d/dalpha cdf(x;alpha,beta)) / pdf(x;alpha,beta) / (1-x)
// for random number x drawn from a Beta distribution Beta(alpha,beta).
// This function inputs total=alpha+beta to make it easy to implement
// Dirichlet reparameterized gradients in terms of Betas.
template <typename scalar_t, typename accscalar_t>
static inline scalar_t dirichlet_grad_one(
    scalar_t x,
    scalar_t alpha,
    scalar_t total) {
  accscalar_t x_ = static_cast<accscalar_t>(x);
  accscalar_t alpha_ = static_cast<accscalar_t>(alpha);
  accscalar_t total_ = static_cast<accscalar_t>(total);

  const scalar_t beta = total - alpha;
  const accscalar_t beta_ = total_ - alpha_;
  const scalar_t boundary = total * x * (1 - x);

  // Use an asymptotic approximation for x close to 0.
  if (x <= 0.5f && boundary < 2.5f) {
    return _beta_grad_alpha_small<scalar_t, accscalar_t>(x, alpha, beta);
  }

  // Use an asymptotic approximation for x close to 1.
  if (x >= 0.5f && boundary < 0.75f) {
    return -_beta_grad_beta_small<scalar_t, accscalar_t>(1 - x, beta, alpha);
  }

  // Use an asymptotic approximation when alpha and (total - alpha) are both
  // large.
  if (alpha > 6 && beta > 6) {
    return _beta_grad_alpha_mid<scalar_t, accscalar_t>(x_, alpha_, beta_);
  }

  // Use a rational correction to an analytic approximation.
  static const accscalar_t c[2][3][3][4] = {
      {{{1.003668233, -0.01061107488, -0.0657888334, 0.01201642863},
        {0.6336835991, -0.3557432599, 0.05486251648, -0.001465281033},
        {-0.03276231906, 0.004474107445, 0.002429354597, -0.0001557569013}},
       {{0.221950385, -0.3187676331, 0.01799915743, 0.01074823814},
        {-0.2951249643, 0.06219954479, 0.01535556598, 0.001550077057},
        {0.02155310298, 0.004170831599, 0.001292462449, 6.976601077e-05}},
       {{-0.05980841433, 0.008441916499, 0.01085618172, 0.002319392565},
        {0.02911413504, 0.01400243777, -0.002721828457, 0.000751041181},
        {0.005900514878, -0.001936558688, -9.495446725e-06, 5.385558597e-05}}},
      {{{1, -0.02924021934, -0.04438342661, 0.007285809825},
        {0.6357567472, -0.3473456711, 0.05454656494, -0.002407477521},
        {-0.03301322327, 0.004845219414, 0.00231480583, -0.0002307248149}},
       {{0.5925320577, -0.1757678135, 0.01505928619, 0.000564515273},
        {0.1014815858, -0.06589186703, 0.01272886114, -0.0007316646956},
        {-0.007258481865, 0.001096195486, 0.0003934994223, -4.12701925e-05}},
       {{0.06469649321, -0.0236701437, 0.002902096474, -5.896963079e-05},
        {0.001925008108, -0.002869809258, 0.0008000589141, -6.063713228e-05},
        {-0.0003477407336,
         6.959756487e-05,
         1.097287507e-05,
         -1.650964693e-06}}},
  };
  const accscalar_t u = compat_log(x_);
  const accscalar_t a = compat_log(alpha_) - u;
  const accscalar_t b = compat_log(total_) - a;
  const accscalar_t pow_u[3] = {1, u, u * u};
  const accscalar_t pow_a[3] = {1, a, a * a};
  accscalar_t p = 0.0;
  accscalar_t q = 0.0;
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      const accscalar_t ua = pow_u[i] * pow_a[j];
      p += ua *
          (c[0][i][j][0] +
           b * (c[0][i][j][1] + b * (c[0][i][j][2] + b * c[0][i][j][3])));
      q += ua *
          (c[1][i][j][0] +
           b * (c[1][i][j][1] + b * (c[1][i][j][2] + b * c[1][i][j][3])));
    }
  }
  const accscalar_t approx = x_ *
      (digamma_one<scalar_t, accscalar_t>(total_) -
       digamma_one<scalar_t, accscalar_t>(alpha_)) /
      beta_;
  return static_cast<scalar_t>(p / q * approx);
}

} // namespace AtenIpexTypeXPU
} // namespace at
