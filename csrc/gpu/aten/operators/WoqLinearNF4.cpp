#include <ATen/ATen.h>
#include <core/Device.h>
#include <core/Memory.h>
#include <float.h>
#include <oneDNN/oneDNN.h>
#include <runtime/Utils.h>
#include <torch/torch.h>
#include <utils/DPCPP.h>
#include <iostream>
#include "BlasImpl.h"

#include "utils/CustomOperatorRegistration.h"

using namespace torch_ipex::xpu::dpcpp;

namespace at {
namespace AtenIpexTypeXPU {

typedef enum DataType_t {
  WOQ_DTYPE_INT8 = 1,
  WOQ_DTYPE_INT4 = 2,
  WOQ_DTYPE_NF4 = 3,
} DataType_t;

static const float lookup_table[16] = {
    -1.0f,
    -0.6961928009986877f,
    -0.5250730514526367f,
    -0.39491748809814453f,
    -0.28444138169288635f,
    -0.18477343022823334f,
    -0.09105003625154495f,
    0.0f,
    0.07958029955625534f,
    0.16093020141124725f,
    0.24611230194568634f,
    0.33791524171829224f,
    0.44070982933044434f,
    0.5626170039176941f,
    0.7229568362236023f,
    1.0f};

template <typename T>
inline T dDequantizeNF4(uint8_t val) {
  return lookup_table[val]; // val < 16
}

template <
    typename T1,
    typename T2,
    int TILE_SIZE,
    int THREADS,
    int NUM_PER_TH,
    int DATA_TYPE>
void kDequantizeBlockwise_kernel(
    float* code,
    uint8_t* A,
    T1* absmax,
    T2* out,
    const int blocksize,
    const int n,
    sycl::nd_item<1>& item) {
  const int base_idx = (item.get_group(0) * TILE_SIZE);

  uint8_t qvals[NUM_PER_TH]; // quantized weight
  T2 vals[NUM_PER_TH * 2]; // dequantized weight

  float* qvals_f = reinterpret_cast<float*>(qvals);
  float* vals_f = reinterpret_cast<float*>(vals);

  T1 local_abs_max =
      absmax[(base_idx + item.get_local_id(0) * NUM_PER_TH) / (blocksize)];

  // load A to qvals
  float* A_f = reinterpret_cast<float*>(
      &A[(base_idx + item.get_local_id(0) * NUM_PER_TH)]);
#pragma unroll
  for (int j = 0; j < NUM_PER_TH / (sizeof(float) / sizeof(uint8_t)); j++) {
    qvals_f[j] = A_f[j];
  }

#pragma unroll
  for (int j = 0; j < NUM_PER_TH; j++) {
    // unpack to val and dequant
    vals[j * 2] =
        static_cast<T2>(dDequantizeNF4<T1>(qvals[j] & 0x0F) * local_abs_max);
    vals[j * 2 + 1] =
        static_cast<T2>(dDequantizeNF4<T1>(qvals[j] >> 4) * local_abs_max);
  }

  // write to output
  float* out_f = reinterpret_cast<float*>(
      &out[base_idx * 2 + item.get_local_id(0) * NUM_PER_TH * 2]);
#pragma unroll
  for (int j = 0; j < NUM_PER_TH * 2 / (sizeof(float) / sizeof(T2)); j++) {
    out_f[j] = vals_f[j];
  }
}

template <
    typename T1,
    typename T2,
    int TILE_SIZE,
    int THREADS,
    int NUM_PER_TH,
    int DATA_TYPE>
struct kDequantizeBlockwiseFunctor {
  void operator()(sycl::nd_item<1> item) const {
    kDequantizeBlockwise_kernel<
        T1,
        T2,
        TILE_SIZE,
        THREADS,
        NUM_PER_TH,
        DATA_TYPE>(code, A, absmax, out, blocksize, n, item);
  }

  kDequantizeBlockwiseFunctor(
      float* code_,
      uint8_t* A_,
      T1* absmax_,
      T2* out_,
      const int blocksize_,
      const int n_)
      : code(code_),
        A(A_),
        absmax(absmax_),
        out(out_),
        blocksize(blocksize_),
        n(n_) {}

 private:
  float* code;
  uint8_t* A;
  T1* absmax;
  T2* out;
  const int blocksize;
  const int n;
};

template <typename T1, typename T2, int DATA_TYPE>
void dequantizeBlockwise(
    float* code,
    uint8_t* A,
    T1* absmax,
    T2* out,
    int blocksize,
    const int n) {
  auto& sycl_queue = dpcppGetCurrentQueue();

  const int work_group_size = 128;
  const int num_per_th = 4;
  const int tile_size = work_group_size * num_per_th;
  const int work_group_num = (n + tile_size - 1) / tile_size / 2;

  auto cgf = DPCPP_Q_CGF(cgh) {
    kDequantizeBlockwiseFunctor<
        T1,
        T2,
        tile_size,
        work_group_size,
        num_per_th,
        DATA_TYPE>
        kfn(code, A, absmax, out, blocksize / 2, n);

    cgh.parallel_for<decltype(kfn)>(
        sycl::nd_range<1>(
            sycl::range<1>(work_group_size * work_group_num),
            sycl::range<1>(work_group_size)),
        kfn);
  };
  DPCPP_Q_SUBMIT(sycl_queue, cgf);
}

at::Tensor dequantize_4bit(
    const at::Tensor& qweight,
    const c10::string_view& weight_dtype,
    const std::vector<int64_t>& weight_shape,
    const at::Tensor& weight_scales,
    const c10::optional<at::Tensor>& weight_zeros,
    int64_t group_size) {
  static const std::map<c10::string_view, int64_t> WOQ_DTYPE_MAP = {
      {"int8", WOQ_DTYPE_INT8},
      {"int4", WOQ_DTYPE_INT4},
      {"nf4", WOQ_DTYPE_NF4},
  };
  TORCH_CHECK(
      WOQ_DTYPE_MAP.find(weight_dtype) != WOQ_DTYPE_MAP.end(),
      "Unsupported weight dtype: ",
      weight_dtype);
  TORCH_CHECK(
      WOQ_DTYPE_MAP.at(weight_dtype) == WOQ_DTYPE_NF4,
      "Only NF4 is supported Now!");
  int64_t n = weight_shape[0] * weight_shape[1];
  auto dqout_shape = weight_shape;
  at::Tensor dq_output = at::zeros(
      dqout_shape, weight_scales.options().dtype(weight_scales.scalar_type()));
  // Output dtype is set the same as input activation
  uint8_t* qweight_ptr = (uint8_t*)qweight.data_ptr();
  if (dq_output.scalar_type() == at::ScalarType::Float) {
    float* absmax_ptr = (float*)weight_scales.data_ptr();
    auto dq_output_ptr = (float*)dq_output.data_ptr();
    dequantizeBlockwise<float, float, WOQ_DTYPE_NF4>(
        NULL, qweight_ptr, absmax_ptr, dq_output_ptr, group_size, n);
  } else if (dq_output.scalar_type() == at::ScalarType::Half) {
    auto dq_output_ptr = (at::Half*)dq_output.data_ptr();
    at::Half* absmax_ptr = (at::Half*)weight_scales.data_ptr();
    dequantizeBlockwise<at::Half, at::Half, WOQ_DTYPE_NF4>(
        NULL, qweight_ptr, absmax_ptr, dq_output_ptr, group_size, n);
  } else if (dq_output.scalar_type() == at::ScalarType::BFloat16) {
    auto dq_output_ptr = (at::BFloat16*)dq_output.data_ptr();
    at::BFloat16* absmax_ptr = (at::BFloat16*)weight_scales.data_ptr();
    dequantizeBlockwise<at::BFloat16, at::BFloat16, WOQ_DTYPE_NF4>(
        NULL, qweight_ptr, absmax_ptr, dq_output_ptr, group_size, n);
  }
  return dq_output;
}

at::Tensor woq_linear(
    const at::Tensor& input,
    const at::Tensor& qweight,
    const c10::string_view& weight_dtype,
    const std::vector<int64_t>& weight_shape,
    const at::Tensor& weight_scales,
    const c10::optional<at::Tensor>& weight_zeros,
    const c10::optional<at::Tensor>& bias,
    const c10::optional<at::Tensor>& g_idx,
    int64_t group_size,
    int64_t lowp_mode,
    int64_t act_quant_mode,
    const c10::optional<at::Tensor>& compensation) {
  static const std::map<c10::string_view, int64_t> WOQ_DTYPE_MAP = {
      {"int8", WOQ_DTYPE_INT8},
      {"int4", WOQ_DTYPE_INT4},
      {"nf4", WOQ_DTYPE_NF4},
  };

  TORCH_CHECK(
      WOQ_DTYPE_MAP.find(weight_dtype) != WOQ_DTYPE_MAP.end(),
      "Unsupported weight dtype: ",
      weight_dtype);
  TORCH_CHECK(
      WOQ_DTYPE_MAP.at(weight_dtype) == WOQ_DTYPE_NF4,
      "Only NF4 is supported Now!");

  // step 1: dequant weight
  int64_t n = weight_shape[0] * weight_shape[1];

  auto dqout_shape = weight_shape;
  at::Tensor dq_output =
      at::zeros(dqout_shape, input.options().dtype(input.scalar_type()));

  // Output dtype is set the same as input activation
  uint8_t* qweight_ptr = (uint8_t*)qweight.data_ptr();

#define DEQUANTIZE_BLOCKWISE(absmax_type, dq_output_type)                  \
  do {                                                                     \
    absmax_type* absmax_ptr = (absmax_type*)weight_scales.data_ptr();      \
    dq_output_type* dq_output_ptr = (dq_output_type*)dq_output.data_ptr(); \
    dequantizeBlockwise<absmax_type, dq_output_type, WOQ_DTYPE_NF4>(       \
        NULL, qweight_ptr, absmax_ptr, dq_output_ptr, group_size, n);      \
  } while (0)

  if (weight_scales.scalar_type() == at::ScalarType::Float) {
    if (dq_output.scalar_type() == at::ScalarType::Float) {
      DEQUANTIZE_BLOCKWISE(float, float);
    } else if (dq_output.scalar_type() == at::ScalarType::Half) {
      DEQUANTIZE_BLOCKWISE(float, at::Half);
    } else if (dq_output.scalar_type() == at::ScalarType::BFloat16) {
      DEQUANTIZE_BLOCKWISE(float, at::BFloat16);
    }
  } else if (weight_scales.scalar_type() == at::ScalarType::Half) {
    if (dq_output.scalar_type() == at::ScalarType::Float) {
      DEQUANTIZE_BLOCKWISE(at::Half, float);
    } else if (dq_output.scalar_type() == at::ScalarType::Half) {
      DEQUANTIZE_BLOCKWISE(at::Half, at::Half);
    } else if (dq_output.scalar_type() == at::ScalarType::BFloat16) {
      DEQUANTIZE_BLOCKWISE(at::Half, at::BFloat16);
    }
  } else if (weight_scales.scalar_type() == at::ScalarType::BFloat16) {
    if (dq_output.scalar_type() == at::ScalarType::Float) {
      DEQUANTIZE_BLOCKWISE(at::BFloat16, float);
    } else if (dq_output.scalar_type() == at::ScalarType::Half) {
      DEQUANTIZE_BLOCKWISE(at::BFloat16, at::Half);
    } else if (dq_output.scalar_type() == at::ScalarType::BFloat16) {
      DEQUANTIZE_BLOCKWISE(at::BFloat16, at::BFloat16);
    }
  }
#undef DEQUANTIZE_BLOCKWISE
  // step 2: OneDNN gemm
  Attr attr;
  bool is_fused;
  Tensor result;
  impl::matmul_fusion_variants(result, input, dq_output, false, attr, is_fused);

  return result;
}

} // namespace AtenIpexTypeXPU
} // namespace at

namespace {
IPEX_LIBRARY_FRAGMENT() {
  IPEX_OP_REGISTER_DISPATCH(
      "dequantize_4bit.xpu",
      AtenIpexTypeXPU::dequantize_4bit,
      c10::DispatchKey::XPU);
  IPEX_OP_REGISTER_DISPATCH(
      "woq_linear.xpu", at::AtenIpexTypeXPU::woq_linear, c10::DispatchKey::XPU);
}

} // namespace
