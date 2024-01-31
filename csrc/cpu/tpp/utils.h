#ifndef _PCL_UTILS_H_
#define _PCL_UTILS_H_

#include <ATen/record_function.h>
#include <torch/csrc/autograd/VariableTypeUtils.h>
//#include <torch/extension.h>

#ifdef _WIN32
#include <intrin.h>
#include <stdint.h>
#include <stdexcept>
#endif

#include <iostream>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#else

#define omp_get_max_threads() 1
#define omp_get_num_threads() 1
#define omp_get_thread_num() 0
#endif

#define MAX_THREADS 640
#define ALIGNDOWN(N, A) ((N) & ~((A)-1))
#define PCL_ASSERT(cond, x...) \
  do {                         \
    if (!(cond)) {             \
      printf(x);               \
      fflush(stdout);          \
      exit(1);                 \
    }                          \
  } while (0)
#define TLA_ASSERT(cond, x...) \
  do {                         \
    if (!(cond)) {             \
      printf(x);               \
      printf("\n");            \
      fflush(stdout);          \
      exit(1);                 \
    }                          \
  } while (0)

namespace torch_ipex {
namespace tpp {

typedef at::BFloat16 bfloat16;
typedef at::Half half;

#define DECL_VLA_PTR(type, name, dims, ptr) type(*name) dims = (type(*) dims)ptr
/*
  Fix issue with clang build: 'cannot initialize a variable of type X with an
  rvalue of type X'. Keep the original code as backup:
*/
#ifdef __clang__
#define DECL_VLA_PTR_PT(type, name, dims, t) \
  auto name = (type(*) dims)(t.data_ptr<type>())
#else
#define DECL_VLA_PTR_PT(type, name, dims, t) \
  type(*name) dims = (type(*) dims)(t.data_ptr<type>())
#endif

#ifdef _WIN32
struct drand48_data {
  uint16_t __x[3] = {0}; /* Current state.  */
  uint16_t __old_x[3] = {0}; /* Old state.  */
  uint16_t __c = 0; /* Additive const. in congruential formula.  */
  uint16_t __init = 0; /* Flag for initializing.  */
  uint64_t __a = 0; /* Factor in congruential formula.  */
};

int srand48_r(uint64_t seed_val, struct drand48_data* buffer) {
  throw std::runtime_error("not implemented.");

  return 0;
}
#endif

// defined in init.cpp
extern double ifreq;
extern thread_local unsigned int* rng_state;
extern thread_local struct drand48_data drng_state; // For non AVX512 version
unsigned int* get_rng_state();
void init_libxsmm();
void xsmm_manual_seed(unsigned int seed);

#ifdef __x86_64__
#ifdef _WIN32
inline uint64_t rdtsc() {
  return __rdtsc();
}
#else
static __inline__ unsigned long long rdtsc(void) {
  unsigned hi, lo;
  __asm__ __volatile__("rdtsc" : "=a"(lo), "=d"(hi));
  return ((unsigned long long)lo) | (((unsigned long long)hi) << 32);
}
#endif
#elif defined(__aarch64__)
static __inline__ unsigned long long rdtsc(void) {
  unsigned long long val;

  /*
   * According to ARM DDI 0487F.c, from Armv8.0 to Armv8.5 inclusive, the
   * system counter is at least 56 bits wide; from Armv8.6, the counter
   * must be 64 bits wide.  So the system counter could be less than 64
   * bits wide and it is attributed with the flag 'cap_user_time_short'
   * is true.
   */
  asm volatile("mrs %0, cntvct_el0" : "=r"(val));

  return val;
}
#else
#error "Unsupported architecture for rdtsc"
#endif
inline double getFreq() {
  uint64_t s = rdtsc();
  uint64_t e = rdtsc();
  return (e - s) * 1.0;
}

inline double getTime() {
  return rdtsc() * ifreq;
}

inline int guess_mpi_rank() {
  const char* env_names[] = {
      "RANK", "PMI_RANK", "OMPI_COMM_WORLD_RANK", "MV2_COMM_WORLD_RANK"};
  static int guessed_rank = -1;
  if (guessed_rank >= 0)
    return guessed_rank;
  for (int i = 0; i < 4; i++) {
    if (getenv(env_names[i]) != NULL) {
      int r = atoi(getenv(env_names[i]));
      if (r >= 0) {
        printf("My guessed rank = %d\n", r);
        guessed_rank = r;
        return guessed_rank;
      }
    }
  }
  guessed_rank = 0;
  return guessed_rank;
}

template <int maxlen>
class SafePrint {
 public:
  SafePrint() {}
  template <typename... Types>
  int operator()(Types... vars) {
    if (len < maxlen) {
      int l = snprintf(&buf[len], 2 * maxlen - len, vars...);
      len += l;
      if (len > maxlen) {
        print();
      }
      return l;
    }
    return 0;
  }
  void print() {
    printf("%s", buf);
    len = 0;
  }

 private:
  char buf[2 * maxlen];
  int len = 0;
};

template <typename T, typename index_t = int64_t>
class VLAAccessorBase {
 public:
  typedef T* PtrType;

  VLAAccessorBase(PtrType data_, const index_t* strides_)
      : data_(data_), strides_(strides_) {}

 protected:
  PtrType data_;
  const index_t* strides_;
};

template <typename T, std::size_t N, typename index_t = int64_t>
class VLAAccessor : public VLAAccessorBase<T, index_t> {
 public:
  typedef T* PtrType;

  VLAAccessor(PtrType data_, const index_t* strides_)
      : VLAAccessorBase<T, index_t>(data_, strides_) {}

  VLAAccessor<T, N - 1, index_t> operator[](index_t i) {
    return VLAAccessor<T, N - 1, index_t>(
        this->data_ + this->strides_[0] * i, this->strides_ + 1);
  }

  const VLAAccessor<T, N - 1, index_t> operator[](index_t i) const {
    return VLAAccessor<T, N - 1, index_t>(
        this->data_ + this->strides_[0] * i, this->strides_ + 1);
  }
};

#if 1
template <typename T, typename index_t>
class VLAAccessor<T, 1, index_t> : public VLAAccessorBase<T, index_t> {
 public:
  typedef T* PtrType;

  VLAAccessor(PtrType data_, const index_t* strides_)
      : VLAAccessorBase<T, index_t>(data_, strides_) {}
  T* operator[](index_t i) {
    return this->data_ + i * this->strides_[0];
  }
  const T* operator[](index_t i) const {
    return this->data_ + i * this->strides_[0];
  }
};
#endif
template <typename T, typename index_t>
class VLAAccessor<T, 0, index_t> : public VLAAccessorBase<T, index_t> {
 public:
  typedef T* PtrType;

  VLAAccessor(PtrType data_, const index_t* strides_)
      : VLAAccessorBase<T, index_t>(data_, strides_) {}
  T& operator[](index_t i) {
    return this->data_[i];
  }
  const T& operator[](index_t i) const {
    return this->data_[i];
  }
  operator T*() {
    return this->data_;
  }
  operator const T*() const {
    return this->data_;
  }
};

template <typename T, std::size_t N, typename index_t = int64_t>
class VLAPtr {
 public:
  VLAPtr(T* data_, const index_t (&sizes)[N]) : data_(data_) {
    strides[N - 1] = sizes[N - 1];
    for (long i = N - 2; i >= 0; i--)
      strides[i] = strides[i + 1] * sizes[i];
  }
  VLAAccessor<T, N - 1, index_t> operator[](index_t i) {
    return VLAAccessor<T, N - 1, index_t>(data_ + i * strides[0], strides + 1);
  }
  operator bool() {
    return data_ != nullptr;
  }

 protected:
  index_t strides[N];
  T* data_;
};

#if 1
template <typename T>
class VLAPtr<T, 1, int64_t> {
 public:
  typedef int64_t index_t;
  VLAPtr(T* data_, const index_t (&sizes)[1]) : data_(data_) {
    strides[0] = sizes[0];
  }
  T* operator[](index_t i) {
    return data_ + i * strides[0];
  }
  operator bool() {
    return data_ != nullptr;
  }

 protected:
  index_t strides[1];
  T* data_;
};
#endif

typedef int64_t index_t;

template <typename T, std::size_t N> //, typename index_t = int64_t>
VLAPtr<T, N, index_t> GetVLAPtr(T* data_, const index_t (&list)[N]) {
  return VLAPtr<T, N, index_t>(data_, list);
}

template <typename T>
inline T* pt_get_data_ptr(at::Tensor t) {
  if (!t.is_contiguous()) {
    std::cout << "Warning: Tensor t " << t.sizes() << " is not contiguous"
              << std::endl;
  }
  return t.data_ptr<T>();
}

template <typename T, std::size_t N> //, typename index_t = int64_t>
VLAPtr<T, N, index_t> GetVLAPtr(at::Tensor t, const index_t (&sizes)[N]) {
  if (!t.defined()) {
    return VLAPtr<T, N, index_t>(nullptr, sizes);
  }
  return VLAPtr<T, N, index_t>(pt_get_data_ptr<T>(t), sizes);
}
template <typename T>
T* GetVLAPtr(at::Tensor t) {
  return pt_get_data_ptr<T>(t);
}

inline int env2int(const char* env_name, int def_val = 0) {
  int val = def_val;
  auto env = getenv(env_name);
  if (env)
    val = atoi(env);
  // printf("Using %s = %d\n", env_name, val);
  return val;
}

} // namespace tpp
} // namespace torch_ipex
#endif //_PCL_UTILS_H_
