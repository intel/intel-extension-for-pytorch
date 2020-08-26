#include <ATen/core/PhiloxRNGEngine.h>

#include <core/DPCPP.h>
#include <core/Memory.h>

using namespace DPCPP;
using namespace at::dpcpp;

#define RANDOM_NUM (200000)
#define NUM_PER_RND (624)

#define _MERSENNE_STATE_N 624
#define _MERSENNE_STATE_M 397

#define MT_WMASK 0xFFFFFFFFU
#define MATRIX_A 0x9908b0dfUL /* constant vector a */
#define UMASK 0x80000000UL /* most significant w-r bits */
#define LMASK 0x7fffffffUL /* least significant r bits */
#define MIXBITS(u, v) (((u)&UMASK) | ((v)&LMASK))
#define TWIST(u, v) ((MIXBITS(u, v) >> 1) ^ ((v)&1UL ? MATRIX_A : 0UL))

static const double DOUBLE_DIVISOR = 18446744073709551616.0;
static const float FLOAT_DIVISOR = 4294967296.0f;

template <typename scalar_t>
class MTRandomEngine {
public:
  MTRandomEngine(
      scalar_t* ptr,
      int64_t ele_num,
      uint64_t seed,
      scalar_t min,
      scalar_t max) {
    m_ptr = ptr;
    m_ele_num = ele_num;
    m_seed = seed;
    m_real_min = static_cast<scalar_t>(min);
    m_real_max = static_cast<scalar_t>(max);
  }

  // mersenne_twister_engine
  void generate_random_numbers(
      const int& global_linear_id,
      const int& seed_offset,
      uint32_t* mt) {
    mt[0] = m_seed + global_linear_id * 100 + seed_offset;
    for (int i = 1; i < NUM_PER_RND; i++) {
      mt[i] = (1812433253U * (mt[i - 1] ^ (mt[i - 1] >> 30)) + i) & MT_WMASK;
    }

    uint32_t* p = mt;
    for (int j = (_MERSENNE_STATE_N - _MERSENNE_STATE_M + 1); --j; p++)
      *p = p[_MERSENNE_STATE_M] ^ TWIST(p[0], p[1]);

    for (int j = _MERSENNE_STATE_M; --j; p++)
      *p = p[_MERSENNE_STATE_M - _MERSENNE_STATE_N] ^ TWIST(p[0], p[1]);

    *p = p[_MERSENNE_STATE_M - _MERSENNE_STATE_N] ^ TWIST(p[0], mt[0]);

    for (int i = 0; i < NUM_PER_RND; i++) {
      uint32_t& y = mt[i];
      y ^= (y >> 11);
      y ^= (y << 7) & 0x9d2c5680UL;
      y ^= (y << 15) & 0xefc60000UL;
      y ^= (y >> 18);
    }
  }


 protected:
  scalar_t* m_ptr;
  int64_t m_ele_num;
  uint64_t m_seed;
  scalar_t m_real_min;
  scalar_t m_real_max;
};

using Philox4_32_10 = at::Philox4_32_10;
template <typename scalar_t, typename engine_t = Philox4_32_10>
class RandomState final {
public:
  template <
    typename = std::enable_if<std::is_same<Philox4_32_10, engine_t>::value>>
  RandomState(
    uint64_t seed = 67280421310721,
    uint64_t subsequence = 0,
    uint64_t offset = 0)
    : engine(seed, subsequence, offset){};

  RandomState() = delete;
  RandomState(const RandomState&) = delete;
  RandomState& operator=(const RandomState&) = delete;
  RandomState(RandomState&&) = default;
  RandomState& operator=(RandomState&&) = default;

  float uniform() {
    uint32_t y = engine();

    return ((float)y + 1.0f) / FLOAT_DIVISOR;
  }

private:
  engine_t engine;
};

template <typename scalar_t>
class RandomEngine : public MTRandomEngine<scalar_t> {
public:
  RandomEngine(
    scalar_t* ptr,
    int64_t ele_num,
    uint64_t seed,
    scalar_t min,
    scalar_t max)
    : MTRandomEngine<scalar_t>(ptr, ele_num, seed, min, max){
  };

  void operator()(nd_item<1> item) {
#ifdef __SYCL_DEVICE_ONLY__
    assert(0 && "not implemented random engine type");
#else
    throw std::runtime_error("not implemented random engine type");
#endif
  }
};

template <>
class RandomEngine<at::Half> : public MTRandomEngine<at::Half>  {
public:
  RandomEngine(
    at::Half* ptr,
    int64_t ele_num,
    uint64_t seed,
    at::Half min,
    at::Half max)
    : MTRandomEngine<at::Half>(ptr, ele_num, seed, min, max){
  };

  void operator()(nd_item<1> item) {
    uint32_t mt[NUM_PER_RND];

    int g_id = item.get_global_linear_id();
    generate_random_numbers(g_id, 0, mt);

    for (int i = 0; i < NUM_PER_RND; i++) {
      int idx = i + g_id * NUM_PER_RND;

      if (idx >= m_ele_num)
        return;
      uint32_t y = mt[i];

      float tmp = m_real_min +
                  ((float)y + 1.0f) / FLOAT_DIVISOR * (m_real_max - m_real_min);
      m_ptr[idx] = static_cast<unsigned short>(static_cast<at::Half>(tmp));
    }
  }
};

template <>
class RandomEngine<at::BFloat16> : public MTRandomEngine<at::BFloat16>  {
public:
  RandomEngine(
    at::BFloat16* ptr,
    int64_t ele_num,
    uint64_t seed,
    at::BFloat16 min,
    at::BFloat16 max)
    : MTRandomEngine<at::BFloat16>(ptr, ele_num, seed, min, max){
  };

  void operator()(nd_item<1> item) {
    uint32_t mt[NUM_PER_RND];

    int g_id = item.get_global_linear_id();
    generate_random_numbers(g_id, 0, mt);

    for (int i = 0; i < NUM_PER_RND; i++) {
      int idx = i + g_id * NUM_PER_RND;

      if (idx >= m_ele_num)
        return;
      uint32_t y = mt[i];

      float tmp = m_real_min +
                  ((float)y + 1.0f) / FLOAT_DIVISOR * (m_real_max - m_real_min);
      m_ptr[idx] = static_cast<unsigned short>(static_cast<at::BFloat16>(tmp));
    }
  }
};
template <>
class RandomEngine<float> : public MTRandomEngine<float> {
public:
  RandomEngine(
    float* ptr,
    int64_t ele_num,
    uint64_t seed,
    float min,
    float max)
    : MTRandomEngine<float>(ptr, ele_num, seed, min, max){
  };

  void operator()(nd_item<1> item) {
    uint32_t mt[NUM_PER_RND];

    int g_id = item.get_global_linear_id();
    generate_random_numbers(g_id, 0, mt);

    for (int i = 0; i < NUM_PER_RND; i++) {
      int idx = i + g_id * NUM_PER_RND;

      if (idx >= m_ele_num)
        return;
      uint32_t y = mt[i];
      m_ptr[idx] = m_real_min +
                   ((float)y + 1.0f) / FLOAT_DIVISOR * (m_real_max - m_real_min);
    }
  }
};

template <>
class RandomEngine<double> : public MTRandomEngine<double> {
public:
  RandomEngine(
    double* ptr,
    int64_t ele_num,
    uint64_t seed,
    double min,
    double max)
    : MTRandomEngine<double>(ptr, ele_num, seed, min, max){
  };

  void operator()(nd_item<1> item) {
    uint32_t mt_hi[NUM_PER_RND];
    uint32_t mt_lo[NUM_PER_RND];

    int g_id = item.get_global_linear_id();
    generate_random_numbers(g_id, 0, mt_hi);
    generate_random_numbers(g_id, 50, mt_lo);

    for (int i = 0; i < NUM_PER_RND; i++) {
      int idx = i + g_id * NUM_PER_RND;
      if (idx >= m_ele_num)
        return;
      uint64_t y = (((uint64_t)mt_hi[i]) << 32) | mt_lo[i];
      m_ptr[idx] = m_real_min +
                   ((double)y + 1.0) / DOUBLE_DIVISOR * (m_real_max - m_real_min);
    }
  }
};
//
//
//template <>
//class MTRandomEngine<c10::BFloat16, 1> : public MTRandomEngine<c10::BFloat16, 0> {
//
//};

template <typename scalar_t, typename accumulate_t>
class NormalRandomFiller {
 public:
  NormalRandomFiller(
      scalar_t* ptr,
      int64_t compute_num,
      double stdv,
      double mean)
      : m_ptr(ptr) {
    m_compute_num = compute_num;
    m_stdv = stdv;
    m_mean = mean;
  }

  void operator()(nd_item<1> item) {
    int g_id = item.get_global_linear_id();

    if (g_id < m_compute_num) {
      const accumulate_t u1 = 1 - m_ptr[g_id]; // [0, 1) -> (0, 1] for log.
      const accumulate_t u2 = m_ptr[g_id + m_compute_num];

      const accumulate_t radius = DPCPP::sqrt(-2 * DPCPP::log(u1));
      const accumulate_t theta = 2.0f * M_PI * u2;

      m_ptr[g_id] = radius * DPCPP::cos(theta) * m_stdv + m_mean;
      m_ptr[g_id + m_compute_num] =
          radius * DPCPP::sin(theta) * m_stdv + m_mean;
    }
  }

 private:
  scalar_t* m_ptr;
  accumulate_t m_stdv;
  accumulate_t m_mean;
  int64_t m_compute_num;
};
