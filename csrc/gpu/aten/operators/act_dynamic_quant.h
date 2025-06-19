#pragma once
#include <ATen/ATen.h>
#include <algorithm>
#include "comm/ATDispatch.h"
#include "comm/Numerics.h"
#include "utils/CustomOperatorRegistration.h"
#include "utils/DPCPP.h"

namespace at {
namespace AtenIpexTypeXPU {

enum class ActQuantScheme : int8_t {
  UNQUANT_A = -1,
  QUANT_A_PER_TENSOR = 0,
  QUANT_A_PER_TENSOR_SYM = 1,
  QUANT_A_PER_M = 2,
  QUANT_A_PER_M_SYM = 3,
  QUANT_A_PER_K_BLOCK = 4,
  QUANT_A_PER_K_BLOCK_SYM = 5,
  QUANT_A_PER_M_K_BLOCK = 6,
  QUANT_A_PER_M_K_BLOCK_SYM = 7
};

static bool is_sym_quant(int64_t scheme) {
  return (scheme & 1) == 1;
}

template <typename scalar_t>
struct alignas(8) vec4_t {
  scalar_t x;
  scalar_t y;
  scalar_t z;
  scalar_t w;
};

using MinMaxPair = sycl::vec<float, 2>;

template <typename scalar_t>
MinMaxPair thread_minmax_vec(
    scalar_t const* input,
    int64_t const num_elems,
    int const tid,
    int const step) {
  // Vectorized input/output to better utilize memory bandwidth.
  vec4_t<scalar_t> const* vectorized_in =
      reinterpret_cast<vec4_t<scalar_t> const*>(input);

  int64_t const num_vec_elems = num_elems >> 2;
  float max_val = Numerics<float>::lower_bound();
  float min_val = Numerics<float>::upper_bound();

#pragma unroll 4
  for (int64_t i = tid; i < num_vec_elems; i += step) {
    vec4_t<scalar_t> in_vec = vectorized_in[i];
    max_val = sycl::fmax(max_val, (float)in_vec.x);
    max_val = sycl::fmax(max_val, (float)in_vec.y);
    max_val = sycl::fmax(max_val, (float)in_vec.z);
    max_val = sycl::fmax(max_val, (float)in_vec.w);

    min_val = sycl::fmin(min_val, (float)in_vec.x);
    min_val = sycl::fmin(min_val, (float)in_vec.y);
    min_val = sycl::fmin(min_val, (float)in_vec.z);
    min_val = sycl::fmin(min_val, (float)in_vec.w);
  }

  // Handle the remaining elements if num_elems is not divisible by 4
  for (int64_t i = num_vec_elems * 4 + tid; i < num_elems; i += step) {
    max_val = sycl::fmax(max_val, (float)input[i]);
    min_val = sycl::fmin(min_val, (float)input[i]);
  }

  return {min_val, max_val};
}

// fp16 --> int8, per-token quant
template <typename src_t, typename dst_t>
struct DynamicPerTokenQuantActFunctor {
  void operator()(sycl::nd_item<1> item) const {
    int tid = item.get_local_id(0);
    int local_range = item.get_local_range(0);
    int group_id = item.get_group(0);
    auto& min_cache =
        *sycl::ext::oneapi::group_local_memory_for_overwrite<float[1024]>(
            item.get_group());
    auto& max_cache =
        *sycl::ext::oneapi::group_local_memory_for_overwrite<float[1024]>(
            item.get_group());

    int offset = group_id * k_;
    int scale_offset = group_id;
    src_t const* token_input = &src_[offset];
    dst_t* token_output = &dst_[offset];
    src_t* token_scales_ptr = &scales_ptr_[scale_offset];
    int32_t* token_zp_ptr = &zp_ptr_[scale_offset];

    bool const can_vectorize = k_ % 4 == 0;
    MinMaxPair min_max = {
        Numerics<float>::upper_bound(), Numerics<float>::lower_bound()};
    if (can_vectorize) {
      min_max = thread_minmax_vec(token_input, k_, tid, local_range);
    } else {
      for (int i = tid; i < k_; i += local_range) {
        min_max[0] = sycl::fmin(min_max[0], (float)token_input[i]);
        min_max[1] = sycl::fmax(min_max[1], (float)token_input[i]);
      }
    }
    min_cache[tid] = min_max[0];
    max_cache[tid] = min_max[1];
    group_barrier(item.get_group());

    int cur_range = local_range;
    int ib = (local_range + 1) / 2;
    while (ib != 0) {
      if (tid < ib && (tid + ib < cur_range) &&
          max_cache[tid + ib] > max_cache[tid]) {
        max_cache[tid] = max_cache[tid + ib];
      }
      if (tid < ib && (tid + ib < cur_range) &&
          min_cache[tid + ib] < min_cache[tid]) {
        min_cache[tid] = min_cache[tid + ib];
      }
      group_barrier(item.get_group());
      ib = (ib > 1 ? (ib + 1) / 2 : ib / 2);
      cur_range = (cur_range + 1) / 2;
    }

    if (tid == 0) {
      float scale_ = is_sym_quant_
          ? (sycl::fmax(sycl::fabs(min_cache[0]), sycl::fabs(max_cache[0])) /
             (float)qmax)
          : (max_cache[0] - min_cache[0]) / (float)qmax;
      token_scales_ptr[0] = static_cast<src_t>(scale_);
      int32_t zp_ = is_sym_quant_ ? 0
                                  : static_cast<int32_t>(-std::nearbyint(
                                        min_cache[0] / token_scales_ptr[0]));
      token_zp_ptr[0] = zp_;
    }
    group_barrier(item.get_group());

    for (int i = tid; i < k_; i += local_range) {
      auto qout = std::nearbyint(
          static_cast<float>(token_input[i]) /
              static_cast<float>(token_scales_ptr[0]) +
          static_cast<float>(token_zp_ptr[0]));
      qout = static_cast<float>(qout);
      qout = (qout > (float)qmax ? (float)qmax : qout);
      qout = (qout < (float)qmin ? (float)qmin : qout);
      token_output[i] = static_cast<dst_t>(qout);
    }
  }

  DynamicPerTokenQuantActFunctor(
      src_t* src,
      dst_t* dst,
      src_t* scales_ptr,
      int32_t* zp_ptr,
      int32_t hidden_size,
      bool is_sym_quant)
      : src_(src),
        dst_(dst),
        scales_ptr_(scales_ptr),
        zp_ptr_(zp_ptr),
        k_(hidden_size),
        is_sym_quant_(is_sym_quant) {
    qmin = is_sym_quant ? -128 : 0;
    qmax = is_sym_quant ? 127 : 255;
  }

 private:
  src_t* src_; // m x k
  dst_t* dst_; // m x k
  src_t* scales_ptr_; // m x 1
  int32_t* zp_ptr_; // m x 1
  int32_t k_;
  bool is_sym_quant_;
  int qmin;
  int qmax;
};

// compute scale and zp
template <typename scalar_t>
struct GetPerTensorScaleZPFunctor {
  void operator()() const {
    float min = static_cast<float>(*min_ptr_);
    float max = static_cast<float>(*max_ptr_);

    float scale_ = is_sym_quant_
        ? (sycl::fmax(sycl::fabs(min), sycl::fabs(max)) / (float)qmax)
        : (max - min) / (float)qmax;
    if (scale_ == 0.0f || sycl::isinf(1.0f / scale_)) {
      scale_ = 0.1f; // avoid zero scale
    }
    scale_ptr_[0] = static_cast<scalar_t>(scale_);
    int32_t zp_ = is_sym_quant_
        ? 0
        : static_cast<int32_t>(-std::nearbyint(min / scale_ptr_[0]));
    zp_ptr_[0] = zp_;
  }

  GetPerTensorScaleZPFunctor(
      scalar_t* min_ptr,
      scalar_t* max_ptr,
      scalar_t* scale_ptr,
      int32_t* zp_ptr,
      bool is_sym_quant)
      : min_ptr_(min_ptr),
        max_ptr_(max_ptr),
        scale_ptr_(scale_ptr),
        zp_ptr_(zp_ptr),
        is_sym_quant_(is_sym_quant) {
    qmin = is_sym_quant ? -128 : 0;
    qmax = is_sym_quant ? 127 : 255;
  }

 private:
  scalar_t* min_ptr_;
  scalar_t* max_ptr_;
  scalar_t* scale_ptr_;
  int32_t* zp_ptr_;
  bool is_sym_quant_;
  int qmin;
  int qmax;
};

// fp16 --> int8, per-token quant
template <typename src_t, typename dst_t>
struct DynamicPerTensorQuantFunctor {
  void operator()(sycl::nd_item<1> item) const {
    int local_range = item.get_local_range(0);
    int group_range = item.get_group_range(0);
    // quantize
    int tid = item.get_global_linear_id();
    for (int i = tid; i < num_elements_; i += local_range * group_range) {
      auto qout = std::nearbyint(
          static_cast<float>(src_[i]) / static_cast<float>(scale_ptr_[0]) +
          static_cast<float>(zp_ptr_[0]));
      qout = static_cast<float>(qout);
      qout = (qout > (float)qmax ? (float)qmax : qout);
      qout = (qout < (float)qmin ? (float)qmin : qout);
      dst_[i] = static_cast<dst_t>(qout);
    }
  }

  DynamicPerTensorQuantFunctor(
      src_t* src,
      dst_t* dst,
      src_t* scale_ptr,
      int32_t* zp_ptr,
      int64_t num_elements,
      bool is_sym_quant)
      : src_(src),
        dst_(dst),
        scale_ptr_(scale_ptr),
        zp_ptr_(zp_ptr),
        num_elements_(num_elements),
        is_sym_quant_(is_sym_quant) {
    qmin = is_sym_quant ? -128 : 0;
    qmax = is_sym_quant ? 127 : 255;
  }

 private:
  src_t* src_; // m x k
  dst_t* dst_; // m x k
  src_t* scale_ptr_; // m x 1
  int32_t* zp_ptr_; // m x 1
  int64_t num_elements_;
  bool is_sym_quant_;
  int qmin;
  int qmax;
};

std::tuple<Tensor, Tensor, Tensor> dynamic_per_token_quant(
    const Tensor& input,
    bool use_sym_quant);

std::tuple<Tensor, Tensor, Tensor> dynamic_per_tensor_quant(
    const Tensor& input,
    bool use_sym_quant);

} // namespace AtenIpexTypeXPU
} // namespace at