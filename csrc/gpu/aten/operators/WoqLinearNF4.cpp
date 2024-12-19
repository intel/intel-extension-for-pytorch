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

template <typename T>
T dDequantizeNF4(uint8_t val) {
  if ((val & 0b1000) == 8)
    if ((val & 0b0100) == 4) // 1
      if ((val & 0b0010) == 2) // 11
        if ((val & 0b0001) == 1) // 111
          return 1.0f;
        else
          return 0.7229568362236023f;
      else if ((val & 0b0001) == 1) // 110
        return 0.5626170039176941f;
      else
        return 0.44070982933044434f;
    else if ((val & 0b0010) == 2) // 10
      if ((val & 0b0001) == 1) // 101
        return 0.33791524171829224f;
      else
        return 0.24611230194568634f;
    else if ((val & 0b0001) == 1) // 100
      return 0.16093020141124725f;
    else
      return 0.07958029955625534f;

  else if ((val & 0b0100) == 4) // 0
    if ((val & 0b0010) == 2) // 01
      if ((val & 0b0001) == 1) // 011
        return 0.0f;
      else
        return -0.09105003625154495f;
    else if ((val & 0b0001) == 1) // 010
      return -0.18477343022823334f;
    else
      return -0.28444138169288635f;
  else if ((val & 0b0010) == 2) // 00
    if ((val & 0b0001) == 1) // 001
      return -0.39491748809814453f;
    else
      return -0.5250730514526367f;
  else if ((val & 0b0001) == 1) // 000
    return -0.6961928009986877f;
  else
    return -1.0f;
}

template <typename T, int TILE_SIZE, int THREADS, int NUM_PER_TH, int DATA_TYPE>
void kDequantizeBlockwise_kernel(
    float* code,
    uint8_t* A,
    T* absmax,
    T* out,
    const int blocksize,
    const int n,
    sycl::nd_item<1>& item) {
  const int n_load = (item.get_group_range(0) * TILE_SIZE);

  const int base_idx = (item.get_group(0) * TILE_SIZE);

  uint8_t qvals[NUM_PER_TH]; // quantized weight
  T vals[NUM_PER_TH * 2]; // dequantized weight

  T local_abs_max = -FLT_MAX;

  for (unsigned int i = base_idx; i < n_load;
       i += item.get_group_range(0) * TILE_SIZE) {
    local_abs_max =
        absmax[(i + item.get_local_id(0) * NUM_PER_TH) / (blocksize)];

#pragma unroll NUM_PER_TH
    for (int j = 0; j < NUM_PER_TH; j++) {
      // load A to qvals
      qvals[j] = A[(i + item.get_local_id(0) * NUM_PER_TH + j)];

      // unpack to val and dequant
      vals[j * 2] = dDequantizeNF4<T>(qvals[j] & 0x0F) * local_abs_max;
      vals[j * 2 + 1] = dDequantizeNF4<T>(qvals[j] >> 4) * local_abs_max;

      // write to output
      out[i * 2 + item.get_local_id(0) * NUM_PER_TH * 2 + j * 2] =
          dDequantizeNF4<T>(qvals[j] & 0x0F) * local_abs_max;
      out[i * 2 + item.get_local_id(0) * NUM_PER_TH * 2 + j * 2 + 1] =
          dDequantizeNF4<T>(qvals[j] >> 4) * local_abs_max;
    }
  }
}

template <typename T, int TILE_SIZE, int THREADS, int NUM_PER_TH, int DATA_TYPE>
struct kDequantizeBlockwiseFunctor {
  void operator()(sycl::nd_item<1> item) const {
    kDequantizeBlockwise_kernel<T, TILE_SIZE, THREADS, NUM_PER_TH, DATA_TYPE>(
        code, A, absmax, out, blocksize, n, item);
  }

  kDequantizeBlockwiseFunctor(
      float* code_,
      uint8_t* A_,
      T* absmax_,
      T* out_,
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
  T* absmax;
  T* out;
  const int blocksize;
  const int n;
};

template <typename T, int DATA_TYPE>
void dequantizeBlockwise(
    float* code,
    uint8_t* A,
    T* absmax,
    T* out,
    int blocksize,
    const int n) {
  auto& sycl_queue = dpcppGetCurrentQueue();
  auto dev_id = dpcppGetDeviceIdOfCurrentQueue();

  int num_blocks = n / blocksize;
  num_blocks = n % blocksize == 0 ? num_blocks : num_blocks + 1;
  int tile_size = 1024;
  int work_group_num = (n + tile_size - 1) / tile_size;
  int work_group_size = 64;

  auto cgf = DPCPP_Q_CGF(cgh) {
    kDequantizeBlockwiseFunctor<T, 512, 64, 8, DATA_TYPE> kfn(
        code, A, absmax, out, blocksize / 2, n);
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
    dequantizeBlockwise<float, WOQ_DTYPE_NF4>(
        NULL, qweight_ptr, absmax_ptr, dq_output_ptr, group_size, n);
  } else if (dq_output.scalar_type() == at::ScalarType::Half) {
    auto dq_output_ptr = (at::Half*)dq_output.data_ptr();
    at::Half* absmax_ptr = (at::Half*)weight_scales.data_ptr();
    dequantizeBlockwise<at::Half, WOQ_DTYPE_NF4>(
        NULL, qweight_ptr, absmax_ptr, dq_output_ptr, group_size, n);
  } else if (dq_output.scalar_type() == at::ScalarType::BFloat16) {
    auto dq_output_ptr = (at::BFloat16*)dq_output.data_ptr();
    at::BFloat16* absmax_ptr = (at::BFloat16*)weight_scales.data_ptr();
    dequantizeBlockwise<at::BFloat16, WOQ_DTYPE_NF4>(
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
  at::Tensor dq_output = at::zeros(
      dqout_shape, weight_scales.options().dtype(weight_scales.scalar_type()));

  // Output dtype is set the same as input activation
  uint8_t* qweight_ptr = (uint8_t*)qweight.data_ptr();

  if (dq_output.scalar_type() == at::ScalarType::Float) {
    float* absmax_ptr = (float*)weight_scales.data_ptr();
    auto dq_output_ptr = (float*)dq_output.data_ptr();
    dequantizeBlockwise<float, WOQ_DTYPE_NF4>(
        NULL, qweight_ptr, absmax_ptr, dq_output_ptr, group_size, n);

  }

  else if (dq_output.scalar_type() == at::ScalarType::Half) {
    auto dq_output_ptr = (at::Half*)dq_output.data_ptr();
    at::Half* absmax_ptr = (at::Half*)weight_scales.data_ptr();
    dequantizeBlockwise<at::Half, WOQ_DTYPE_NF4>(
        NULL, qweight_ptr, absmax_ptr, dq_output_ptr, group_size, n);

  }

  else if (dq_output.scalar_type() == at::ScalarType::BFloat16) {
    auto dq_output_ptr = (at::BFloat16*)dq_output.data_ptr();
    at::BFloat16* absmax_ptr = (at::BFloat16*)weight_scales.data_ptr();
    dequantizeBlockwise<at::BFloat16, WOQ_DTYPE_NF4>(
        NULL, qweight_ptr, absmax_ptr, dq_output_ptr, group_size, n);
  }

  // step 2: OneDNN gemm
  dq_output = dq_output.to(input.scalar_type());
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
