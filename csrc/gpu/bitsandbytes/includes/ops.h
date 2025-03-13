#include "context.h"

typedef enum Optimizer_t {
  ADAM = 0,
  MOMENTUM = 1,
  RMSPROP = 2,
  LARS = 3,
  ADAGRAD = 4,
  LION = 5,
  ADEMAMIX = 6
} Optimizer_t;

typedef enum DataType_t {
  General8bit = 0,
  FP4 = 1,
  NF4 = 2,
} DataType_t;

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
    int n);

template <typename T>
void percentileClipping(T* g, float* gnorm_vec, int step, const int n);

template <typename T, int DATA_TYPE>
void dequantizeBlockwise(
    float* code,
    unsigned char* A,
    float* absmax,
    T* out,
    int block_size,
    const int n);
