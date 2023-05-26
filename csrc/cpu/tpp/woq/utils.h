/******************************************************************************
 * Copyright (c) 2022 Intel Corporation - All rights reserved.                *
 *                                                                            *
 * For information on the license, see the LICENSE file.                      *
 * Further information: https://github.com/libxsmm/tpp-pytorch-extension/     *
 * SPDX-License-Identifier: BSD-3-Clause                                      *
 ******************************************************************************/
/* Author: Dhiraj Kalamkar (Intel Corp.)
 ******************************************************************************/

#ifndef _TPP_UTILS_H_
#define _TPP_UTILS_H_

#include <cxxabi.h>
#include <iostream>
#include <typeinfo>
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
#define TPP_ASSERT(cond, x...) \
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

#define DECL_VLA_PTR(type, name, dims, ptr) type(*name) dims = (type(*) dims)ptr
#define DECL_VLA_PTR_PT(type, name, dims, t) \
  type(*name) dims = (type(*) dims)(pt_get_data_ptr<type>(t))

extern double ifreq; // defined in init.cpp

#if 0
// Defined in xsmm.cpp
extern thread_local unsigned int* rng_state;
extern thread_local struct drand48_data drng_state; // For non AVX512 version
unsigned int* get_rng_state();
#endif

template <typename T>
inline std::string get_class_name() {
  auto cname = abi::__cxa_demangle(typeid(T).name(), 0, 0, NULL);
  std::string name(cname);
  free(cname);
  return name;
}

#ifdef __x86_64__
static __inline__ unsigned long long rdtsc(void) {
  unsigned hi, lo;
  __asm__ __volatile__("rdtsc" : "=a"(lo), "=d"(hi));
  return ((unsigned long long)lo) | (((unsigned long long)hi) << 32);
}
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
  long long int s = rdtsc();
  // sleep(1);
  long long int e = rdtsc();
  return (e - s) * 1.0;
}

inline double getTime() {
  return rdtsc() * ifreq;
}

inline int env2int(const char* env_name, int def_val = 0) {
  int val = def_val;
  auto env = getenv(env_name);
  if (env)
    val = atoi(env);
  // printf("Using %s = %d\n", env_name, val);
  return val;
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

// A class for forced loop unrolling at compile time
// These macro utils and the small gemm intrinsics kernels are implemented
// based on the initial code by pujiang.he@intel.com.
template <int i>
struct compile_time_for {
  template <typename Lambda, typename... Args>
  inline static void op(const Lambda& function, Args... args) {
    compile_time_for<i - 1>::op(function, std::forward<Args>(args)...);
    function(std::integral_constant<int, i - 1>{}, std::forward<Args>(args)...);
  }
};
template <>
struct compile_time_for<1> {
  template <typename Lambda, typename... Args>
  inline static void op(const Lambda& function, Args... args) {
    function(std::integral_constant<int, 0>{}, std::forward<Args>(args)...);
  }
};
template <>
struct compile_time_for<0> {
  // 0 loops, do nothing
  template <typename Lambda, typename... Args>
  inline static void op(const Lambda& function, Args... args) {}
};

#endif //_TPP_UTILS_H_
