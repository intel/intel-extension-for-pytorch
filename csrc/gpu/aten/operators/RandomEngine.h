#pragma once

#include <ATen/core/DistributionsHelper.h>
#include <aten/operators/MemoryAccess.h>
#include <utils/DPCPP.h>

#include "comm/Numerics.h"

namespace at {
namespace AtenIpexTypeXPU {

#define EXTRA_FLAG_NORMAL 0x00000001

template <typename T>
struct alignas(sizeof(T) * 2) rand_vec2 {
  union {
    T val[2];
    struct {
      T x, y;
    };
  };
  inline rand_vec2() {}
  inline rand_vec2(T x_, T y_) : x(x_), y(y_) {}
};

template <typename T>
struct alignas(sizeof(T) * 4) rand_vec4 {
  union {
    T val[4];
    struct {
      T x, y, z, w;
    };
  };
  inline rand_vec4() {}
  inline rand_vec4(T x_, T y_, T z_, T w_) : x(x_), y(y_), z(z_), w(w_) {}
};

typedef rand_vec4<float> float4;
typedef rand_vec2<float> float2;
typedef rand_vec2<double> double2;
typedef rand_vec4<uint32_t> uint4;
typedef rand_vec2<uint32_t> uint2;

#define PHILOX_W32_0 (0x9E3779B9)
#define PHILOX_W32_1 (0xBB67AE85)
#define PHILOX_M4x32_0 (0xD2511F53)
#define PHILOX_M4x32_1 (0xCD9E8D57)

typedef struct randStatePhilox4_32_10 {
  uint4 ctr;
  uint4 output;
  uint2 key;
  unsigned int STATE;
  int boxmuller_flag;
  int boxmuller_flag_double;
  float boxmuller_extra;
  double boxmuller_extra_double;
} randStatePhilox4_32_10_t;

static inline void Philox_State_Incr(
    randStatePhilox4_32_10_t* s,
    unsigned long long n) {
  unsigned int nlo = (unsigned int)(n);
  unsigned int nhi = (unsigned int)(n >> 32);
  s->ctr.x += nlo;
  if (s->ctr.x < nlo)
    nhi++;
  s->ctr.y += nhi;
  if (nhi <= s->ctr.y)
    return;
  if (++s->ctr.z)
    return;
  ++s->ctr.w;
}

static inline void Philox_State_Incr(randStatePhilox4_32_10_t* s) {
  if (++s->ctr.x)
    return;
  if (++s->ctr.y)
    return;
  if (++s->ctr.z)
    return;
  ++s->ctr.w;
}

static inline void Philox_State_Incr_hi(
    randStatePhilox4_32_10_t* s,
    unsigned long long n) {
  unsigned int nlo = (unsigned int)(n);
  unsigned int nhi = (unsigned int)(n >> 32);
  s->ctr.z += nlo;
  if (s->ctr.z < nlo)
    nhi++;
  s->ctr.w += nhi;
}

static inline unsigned int mulhilo32(
    unsigned int a,
    unsigned int b,
    unsigned int* hip) {
  *hip = sycl::mul_hi(a, b);
  return a * b;
}

static inline uint4 _philox4x32round(uint4 ctr, uint2 key) {
  unsigned int hi0;
  unsigned int hi1;
  unsigned int lo0 = mulhilo32(PHILOX_M4x32_0, ctr.x, &hi0);
  unsigned int lo1 = mulhilo32(PHILOX_M4x32_1, ctr.z, &hi1);
  uint4 ret = {hi1 ^ ctr.y ^ key.x, lo1, hi0 ^ ctr.w ^ key.y, lo0};
  return ret;
}

static inline uint4 rand_Philox4x32_10(uint4 c, uint2 k) {
  c = _philox4x32round(c, k); // 1
  k.x += PHILOX_W32_0;
  k.y += PHILOX_W32_1;
  c = _philox4x32round(c, k); // 2
  k.x += PHILOX_W32_0;
  k.y += PHILOX_W32_1;
  c = _philox4x32round(c, k); // 3
  k.x += PHILOX_W32_0;
  k.y += PHILOX_W32_1;
  c = _philox4x32round(c, k); // 4
  k.x += PHILOX_W32_0;
  k.y += PHILOX_W32_1;
  c = _philox4x32round(c, k); // 5
  k.x += PHILOX_W32_0;
  k.y += PHILOX_W32_1;
  c = _philox4x32round(c, k); // 6
  k.x += PHILOX_W32_0;
  k.y += PHILOX_W32_1;
  c = _philox4x32round(c, k); // 7
  k.x += PHILOX_W32_0;
  k.y += PHILOX_W32_1;
  c = _philox4x32round(c, k); // 8
  k.x += PHILOX_W32_0;
  k.y += PHILOX_W32_1;
  c = _philox4x32round(c, k); // 9
  k.x += PHILOX_W32_0;
  k.y += PHILOX_W32_1;
  return _philox4x32round(c, k); // 10
}

static inline void skipahead_sequence(
    unsigned long long n,
    randStatePhilox4_32_10_t* state) {
  Philox_State_Incr_hi(state, n);
  state->output = rand_Philox4x32_10(state->ctr, state->key);
}

static inline void skipahead(
    unsigned long long n,
    randStatePhilox4_32_10_t* state) {
  state->STATE += (n & 3);
  n /= 4;
  if (state->STATE > 3) {
    n += 1;
    state->STATE -= 4;
  }
  Philox_State_Incr(state, n);
  state->output = rand_Philox4x32_10(state->ctr, state->key);
}

static inline void rand_init(
    unsigned long long seed,
    unsigned long long subsequence,
    unsigned long long offset,
    randStatePhilox4_32_10_t* state) {
  state->ctr.x = 0;
  state->ctr.y = 0;
  state->ctr.z = 0;
  state->ctr.w = 0;
  state->key.x = (unsigned int)seed;
  state->key.y = (unsigned int)(seed >> 32);
  state->STATE = 0;
  skipahead_sequence(subsequence, state);
  skipahead(offset, state);
}

static inline unsigned int rand(randStatePhilox4_32_10_t* state) {
  // Maintain the invariant: output[STATE] is always "good" and
  //  is the next value to be returned by curand.
  unsigned int ret;
  switch (state->STATE++) {
    default:
      ret = state->output.x;
      break;
    case 1:
      ret = state->output.y;
      break;
    case 2:
      ret = state->output.z;
      break;
    case 3:
      ret = state->output.w;
      break;
  }
  if (state->STATE == 4) {
    Philox_State_Incr(state);
    state->output = rand_Philox4x32_10(state->ctr, state->key);
    state->STATE = 0;
  }
  return ret;
}

static inline uint4 rand4(randStatePhilox4_32_10_t* state) {
  uint4 r;
  uint4 tmp = state->output;
  Philox_State_Incr(state);
  state->output = rand_Philox4x32_10(state->ctr, state->key);
  switch (state->STATE) {
    case 0:
      return tmp;
    case 1:
      r.x = tmp.y;
      r.y = tmp.z;
      r.z = tmp.w;
      r.w = state->output.x;
      break;
    case 2:
      r.x = tmp.z;
      r.y = tmp.w;
      r.z = state->output.x;
      r.w = state->output.y;
      break;
    case 3:
      r.x = tmp.w;
      r.y = state->output.x;
      r.z = state->output.y;
      r.w = state->output.z;
      break;
    default:
      // NOT possible but needed to avoid compiler warnings
      return tmp;
  }
  return r;
}

#define RAND_2POW32_INV (2.3283064e-10f)
#define RAND_2POW32_INV_2PI (2.3283064e-10f * 6.2831855f)
#define RAND_2POW53_INV_DOUBLE (1.1102230246251565e-16)
#define RAND_PI_DOUBLE (3.1415926535897932)

// =================== uniform ===================

static inline float _rand_uniform(unsigned int x) {
  return x * RAND_2POW32_INV + (RAND_2POW32_INV / 2.0f);
}

static inline float _rand_uniform(unsigned long long x) {
  unsigned int t;
  t = (unsigned int)(x >> 32);
  return t * RAND_2POW32_INV + (RAND_2POW32_INV / 2.0f);
}

static inline float rand_uniform(randStatePhilox4_32_10_t* state) {
  return _rand_uniform(rand(state));
}

static inline float4 rand_uniform4(randStatePhilox4_32_10_t* state) {
  auto x = rand4(state);
  float4 y;
  y.x = x.x * RAND_2POW32_INV + (RAND_2POW32_INV / 2.0f);
  y.y = x.y * RAND_2POW32_INV + (RAND_2POW32_INV / 2.0f);
  y.z = x.z * RAND_2POW32_INV + (RAND_2POW32_INV / 2.0f);
  y.w = x.w * RAND_2POW32_INV + (RAND_2POW32_INV / 2.0f);
  return y;
}

static inline double _rand_uniform_double_hq(unsigned int x, unsigned int y) {
  unsigned long long z =
      (unsigned long long)x ^ ((unsigned long long)y << (53 - 32));
  return z * RAND_2POW53_INV_DOUBLE + (RAND_2POW53_INV_DOUBLE / 2.0);
}

static inline double2 rand_uniform2_double(randStatePhilox4_32_10_t* state) {
  auto _x = rand4(state);
  double2 result;
  result.x = _rand_uniform_double_hq(_x.x, _x.y);
  result.y = _rand_uniform_double_hq(_x.z, _x.w);
  return result;
}

// =================== normal ===================

static inline float2 _rand_box_muller(unsigned int x, unsigned int y) {
  float2 result;
  float u = x * RAND_2POW32_INV + (RAND_2POW32_INV / 2);
  float v = y * RAND_2POW32_INV_2PI + (RAND_2POW32_INV_2PI / 2);
  float s = Numerics<float>::sqrt(-2.0f * Numerics<float>::log(u));
  result.x = Numerics<float>::sin(v);
  result.y = Numerics<float>::cos(v);
  result.x *= s;
  result.y *= s;
  return result;
}

template <typename R>
static inline float4 rand_box_muller4(R* state) {
  float4 result;
  float2 _result;
  auto x = rand4(state);
  _result = _rand_box_muller(x.x, x.y);
  result.x = _result.x;
  result.y = _result.y;
  _result = _rand_box_muller(x.z, x.w);
  result.z = _result.x;
  result.w = _result.y;
  return result;
}

static inline double2 _rand_box_muller_double(
    unsigned int x0,
    unsigned int x1,
    unsigned int y0,
    unsigned int y1) {
  double2 result;
  unsigned long long zx =
      (unsigned long long)x0 ^ ((unsigned long long)x1 << (53 - 32));
  double u = zx * RAND_2POW53_INV_DOUBLE + (RAND_2POW53_INV_DOUBLE / 2.0);
  unsigned long long zy =
      (unsigned long long)y0 ^ ((unsigned long long)y1 << (53 - 32));
  double v = zy * (RAND_2POW53_INV_DOUBLE * 2.0) + RAND_2POW53_INV_DOUBLE;
  double s = Numerics<double>::sqrt(-2.0 * Numerics<double>::log(u));

  result.x = Numerics<double>::sin(v * RAND_PI_DOUBLE);
  result.y = Numerics<double>::cos(v * RAND_PI_DOUBLE);
  result.x *= s;
  result.y *= s;

  return result;
}

static inline float rand_normal(randStatePhilox4_32_10_t* state) {
  if (state->boxmuller_flag != EXTRA_FLAG_NORMAL) {
    unsigned int x, y;
    x = rand(state);
    y = rand(state);
    float2 v = _rand_box_muller(x, y);
    state->boxmuller_extra = v.y;
    state->boxmuller_flag = EXTRA_FLAG_NORMAL;
    return v.x;
  }
  state->boxmuller_flag = 0;
  return state->boxmuller_extra;
}

static inline float4 rand_normal4(randStatePhilox4_32_10_t* state) {
  return rand_box_muller4(state);
}

static inline double2 rand_normal2_double(randStatePhilox4_32_10_t* state) {
  double2 result;
  auto x = rand4(state);
  result = _rand_box_muller_double(x.x, x.y, x.z, x.w);
  return result;
}

} // namespace AtenIpexTypeXPU
} // namespace at
