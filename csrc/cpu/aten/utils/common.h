#pragma once

#include <ATen/ATen.h>
#include <ATen/Parallel.h>

#if defined(_OPENMP)
#include <omp.h>
#endif

namespace {

inline void check_shape(
    const at::Tensor& a,
    const at::Tensor& b,
    const char* a_name,
    const char* b_name) {
  TORCH_CHECK(
      a.dim() == b.dim(),
      a_name,
      ".dim() != ",
      b_name,
      ".dim(). ",
      a.dim(),
      " vs ",
      b.dim());
  for (int i = 0; i < a.dim(); ++i) {
    TORCH_CHECK(
        a.size(i) == b.size(i),
        a_name,
        ".size(",
        i,
        ") != ",
        b_name,
        ".size(",
        i,
        ")");
  }
}

inline constexpr uint32_t pack_u16(uint16_t a, uint16_t b) {
  return (uint32_t(a) << 16) | uint32_t(b);
}

#define TORCH_UNUSED(x) (void)(x)

#define CHECK_GQA_HEAD_DIVISIBLE(num_qo_heads, num_kv_heads) \
  TORCH_CHECK(                                               \
      num_qo_heads % num_kv_heads == 0,                      \
      "num_qo_heads(",                                       \
      num_qo_heads,                                          \
      ") must be divisible by num_kv_heads(",                \
      num_kv_heads,                                          \
      ")")

#define CHECK_CPU(x) \
  TORCH_CHECK(x.device().type() == at::kCPU, #x " must be a CPU tensor")

#define CHECK_CONTIGUOUS(x) \
  TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_LAST_DIM_CONTIGUOUS(x)            \
  TORCH_CHECK(                                  \
      x.strides()[x.strides().size() - 1] == 1, \
      #x "must be contiguous at last dimention")

#define CHECK_INPUT(x) \
  CHECK_CPU(x);        \
  CHECK_CONTIGUOUS(x)
#define CHECK_LAST_DIM_CONTIGUOUS_INPUT(x) \
  CHECK_CPU(x);                            \
  CHECK_LAST_DIM_CONTIGUOUS(x)

#define CHECK_DIM(d, x) \
  TORCH_CHECK(x.dim() == d, #x " must be a " #d "D tensor")

#define CHECK_SHAPE(a, b) check_shape(a, b, #a, #b)

#define CHECK_EQ(a, b) \
  TORCH_CHECK((a) == (b), "CHECK_EQ(" #a ", " #b ") failed. ", a, " vs ", b)

#define CHECK_GE(a, b) \
  TORCH_CHECK((a) >= (b), "CHECK_GE(" #a ", " #b ") failed. ", a, " vs ", b)

// parallel routines
constexpr int GRAIN_SIZE = 1024;

template <
    typename T,
    typename std::enable_if<std::is_integral<T>::value, int>::type = 0>
inline T div_up(T x, T y) {
  return (x + y - 1) / y;
}

template <typename T>
inline void balance211(T n, T nth, T ith, T& n_start, T& n_end) {
#if 0
    // onednn partition pattern
    T& n_my = n_end;
    if (nth <= 1 || n == 0) {
        n_start = 0;
        n_my = n;
    } else {
        T n1 = div_up(n, nth);
        T n2 = n1 - 1;
        T T1 = n - n2 * nth;
        n_my = ith < T1 ? n1 : n2;
        n_start = ith <= T1 ? ith*n1 : T1 * n1 + (ith - T1) * n2;
    }
    n_end += n_start;
#else
  // pytorch aten partition pattern
  T n_my = div_up(n, nth);
  n_start = ith * n_my;
  n_end = std::min(n_start + n_my, n);
#endif
}

template <typename func_t>
inline void parallel_for(int n, const func_t& f) {
#if defined(_OPENMP)
#pragma omp parallel
  {
    int nth = omp_get_num_threads();
    int ith = omp_get_thread_num();
    int tbegin, tend;
    balance211(n, nth, ith, tbegin, tend);
    f(tbegin, tend);
  }
#else
  f(0, n);
#endif
}

// data indexing for dimension collapse
template <typename T>
inline T data_index_init(T offset) {
  return offset;
}

template <typename T, typename... Args>
inline T data_index_init(T offset, T& x, const T& X, Args&&... args) {
  offset = data_index_init(offset, std::forward<Args>(args)...);
  x = offset % X;
  return offset / X;
}

inline bool data_index_step() {
  return true;
}

template <typename T, typename... Args>
inline bool data_index_step(T& x, const T& X, Args&&... args) {
  if (data_index_step(std::forward<Args>(args)...)) {
    x = ((x + 1) == X) ? 0 : (x + 1);
    return x == 0;
  }
  return false;
}

// forced unroll for perf critical path

#if __has_attribute(always_inline)
#define ALWAYS_INLINE __attribute__((__always_inline__)) inline
#else
#define ALWAYS_INLINE inline
#endif

// TODO: remove me
#define STRINGIFY(x) #x // Turns the argument into a string
#define TOSTRING(x) STRINGIFY(x) // Handles nested macros

template <int n>
struct Unroll {
  template <typename Func, typename... Args>
  ALWAYS_INLINE void operator()(const Func& f, Args... args) const {
    Unroll<n - 1>{}(f, args...);
    f(std::integral_constant<int, n - 1>{}, args...);
  }
};

template <>
struct Unroll<1> {
  template <typename Func, typename... Args>
  ALWAYS_INLINE void operator()(const Func& f, Args... args) const {
    f(std::integral_constant<int, 0>{}, args...);
  }
};

} // anonymous namespace