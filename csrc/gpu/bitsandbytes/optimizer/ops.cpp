#include "ops.h"
#include <ATen/ATen.h>
#include <ATen/NestedTensorImpl.h>
#include <ATen/native/nested/NestedTensorUtils.h>
#include <stdio.h>
#include <sycl/sycl.hpp>
#include "optimizer.h"

typedef at::Half half;
typedef at::BFloat16 bfloat16;

#define BLOCKSIZE_2STATE 256
#define NUM_2STATE 1
#define BLOCKSIZE_1STATE 256
#define NUM_1STATE 1

template <typename T, int OPTIMIZER>
void optimizerStatic8bitBlockwise(
    T* p,
    T* g,
    unsigned char* state1,
    unsigned char* state2,
    float beta1,
    float beta2,
    float beta3,
    float alpha,
    float eps,
    int step,
    float lr,
    float* quantiles1,
    float* quantiles2,
    float* absmax1,
    float* absmax2,
    float weight_decay,
    const float gnorm_scale,
    bool skip_zeros,
    int n) {
  int num_blocks = 0;
  switch (OPTIMIZER) {
    case ADAM:
      num_blocks = n / BLOCKSIZE_2STATE;
      num_blocks = n % BLOCKSIZE_2STATE == 0 ? num_blocks : num_blocks + 1;

      sycl::queue* q = at::getCurrentSYCLStream();
      q->submit([&](sycl::handler& cgh) {
        kOptimizerStatic8bit2StateBlockwise<
            T,
            OPTIMIZER,
            BLOCKSIZE_2STATE,
            NUM_2STATE>
            fn(p,
               g,
               state1,
               state2,
               beta1,
               beta2,
               beta3,
               alpha,
               eps,
               step,
               lr,
               quantiles1,
               quantiles2,
               absmax1,
               absmax2,
               weight_decay,
               gnorm_scale,
               skip_zeros,
               n);
        cgh.parallel_for(
            sycl::nd_range<1>(
                num_blocks * (BLOCKSIZE_2STATE / NUM_2STATE),
                BLOCKSIZE_2STATE / NUM_2STATE),
            fn);
      });
      break;
  }
}

template <typename T>
void percentileClipping(T* g, float* gnorm_vec, int step, const int n) {
  int num_blocks = n / 2048;
  num_blocks = n % 2048 == 0 ? num_blocks : num_blocks + 1;
  sycl::queue* q = at::getCurrentSYCLStream();
  q->memset(&gnorm_vec[step % 100], 0, 1 * sizeof(float)).wait();
  q->submit([&](sycl::handler& cgh) {
    kPercentileClipping<T, 2048, 4> fn(g, gnorm_vec, step, n);
    cgh.parallel_for(sycl::nd_range<1>(num_blocks * 512, 512), fn);
  });
}

template <typename T, int DATA_TYPE>
void dequantizeBlockwise(
    float* code,
    unsigned char* A,
    float* absmax,
    T* out,
    int blocksize,
    const int n) {
  int num_blocks = n / blocksize;
  num_blocks = n % blocksize == 0 ? num_blocks : num_blocks + 1;
  int tile_size = (DATA_TYPE > 0) ? 1024 : 512;
  blocksize = (DATA_TYPE > 0) ? blocksize / 2 : blocksize;
  sycl::queue* q = at::getCurrentSYCLStream();
  q->submit([&](sycl::handler& cgh) {
    kDequantizeBlockwise<T, 512, 64, 8, DATA_TYPE> fn(
        code, A, absmax, out, blocksize, n);
    cgh.parallel_for(
        sycl::nd_range<1>(((n + tile_size - 1) / tile_size) * 64, 64), fn);
  });
}

#define MAKE_optimizerStatic8bitBlockwise(gtype, optim_name)     \
  template void optimizerStatic8bitBlockwise<gtype, optim_name>( \
      gtype * p,                                                 \
      gtype * g,                                                 \
      unsigned char* state1,                                     \
      unsigned char* state2,                                     \
      float beta1,                                               \
      float beta2,                                               \
      float beta3,                                               \
      float alpha,                                               \
      float eps,                                                 \
      int step,                                                  \
      float lr,                                                  \
      float* quantiles1,                                         \
      float* quantiles2,                                         \
      float* absmax1,                                            \
      float* absmax2,                                            \
      float weight_decay,                                        \
      const float gnorm_scale,                                   \
      bool skip_zeros,                                           \
      int n);

MAKE_optimizerStatic8bitBlockwise(half, ADAM);
MAKE_optimizerStatic8bitBlockwise(float, ADAM);
MAKE_optimizerStatic8bitBlockwise(bfloat16, ADAM);

template void percentileClipping(
    float* g,
    float* gnorm_vec,
    int step,
    const int n);
template void percentileClipping(
    half* g,
    float* gnorm_vec,
    int step,
    const int n);

template void dequantizeBlockwise<float, General8bit>(
    float* code,
    unsigned char* A,
    float* absmax,
    float* out,
    int blocksize,
    const int n);

template void dequantizeBlockwise<half, General8bit>(
    float* code,
    unsigned char* A,
    float* absmax,
    half* out,
    int blocksize,
    const int n);

template void dequantizeBlockwise<bfloat16, General8bit>(
    float* code,
    unsigned char* A,
    float* absmax,
    bfloat16* out,
    int blocksize,
    const int n);