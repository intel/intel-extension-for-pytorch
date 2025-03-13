#include <ATen/ATen.h>
#include <ATen/DeviceGuard.h>
#include <ATen/record_function.h>
#include <c10/core/ScalarType.h>
#include <c10/util/Float8_e4m3fn.h>
#include <c10/util/Float8_e5m2.h>
#include <core/Device.h>
#include <core/detail/ListUtils.h>
#include <runtime/Utils.h>
#include <torch/library.h>
#include <utils/DPCPP.h>
#include <map>
#include "comm/ATDispatch.h"
#include "comm/AccumulateType.h"
#include "comm/ApplyUtils.h"
#include "comm/Numerics.h"
#include "utils/CustomOperatorRegistration.h"

#include <aten/operators/MemoryAccess.h> // clang-format off

#include <core/Memory.h>
namespace at {
namespace AtenIpexTypeXPU {

namespace cache_op {

void swap_blocks(
    at::Tensor& src,
    at::Tensor& dst,
    at::Tensor& block_map // [num_pairs, 2]
) {
  at::Device src_device = src.device();
  at::Device dst_device = dst.device();
  at::DeviceGuard device_guard(src.device());
  dpcppMemcpyKind cpy_kind;
  if (src_device.is_xpu() && dst_device.is_cpu()) {
    cpy_kind = dpcppMemcpyKind::DeviceToHost;
  } else if (src_device.is_cpu() && dst_device.is_xpu()) {
    cpy_kind = dpcppMemcpyKind::HostToDevice;
  } else if (src_device.is_xpu() && dst_device.is_xpu()) {
    cpy_kind = dpcppMemcpyKind::DeviceToDevice;
  } else {
    TORCH_CHECK(false, "Invalid device combination");
  }
  char* src_ptr = static_cast<char*>(src.data_ptr());
  char* dst_ptr = static_cast<char*>(dst.data_ptr());
  const int64_t block_size = src.element_size() * src[0].numel();
  int64_t* block_map_data = block_map.data_ptr<int64_t>();
  int pair_number = block_map.size(0);
  for (int i = 0; i < pair_number; ++i) {
    int64_t src_block_number = block_map_data[i * 2];
    int64_t dst_block_number = block_map_data[i * 2 + 1];
    int64_t src_offset = src_block_number * block_size;
    int64_t dst_offset = dst_block_number * block_size;
    dpcppMemcpy(
        dst_ptr + dst_offset, src_ptr + src_offset, block_size, cpy_kind);
  }
  return;
}

template <typename scalar_t, typename vec_type, int vec_size>
struct CopyBlockFunctor {
  CopyBlockFunctor(
      int64_t* key_cache,
      int64_t* value_cache,
      int64_t* mappings,
      int copy_number)
      : key_caches_(key_cache),
        value_caches_(value_cache),
        mappings_(mappings),
        copy_number_(copy_number) {}

  void operator()(const sycl::nd_item<2> item_id) const {
    int64_t local_id = item_id.get_local_id(0);
    int64_t local_range = item_id.get_local_range(0);
    int64_t group_id_x = item_id.get_group(0);
    int64_t group_id_y = item_id.get_group(1);
    scalar_t* k_block_ptr =
        reinterpret_cast<scalar_t*>(key_caches_[group_id_y]);
    scalar_t* v_block_ptr =
        reinterpret_cast<scalar_t*>(value_caches_[group_id_y]);
    int64_t src_offset = mappings_[2 * group_id_x] * copy_number_;
    int64_t dst_offset = mappings_[2 * group_id_x + 1] * copy_number_;
    int vector_cpy_num = copy_number_ / vec_size;
    int vector_cpy_total = vector_cpy_num * vec_size;
    int scalar_cpy_num = copy_number_ % vec_size;
    for (int i = local_id; i < vector_cpy_num; i += local_range) {
      reinterpret_cast<vec_type*>(k_block_ptr + dst_offset)[i] =
          reinterpret_cast<vec_type*>(k_block_ptr + src_offset)[i];
    }
    for (int i = local_id; i < scalar_cpy_num; i += local_range) {
      k_block_ptr[dst_offset + i + vector_cpy_total] =
          k_block_ptr[src_offset + i + vector_cpy_total];
    }
    for (int i = local_id; i < vector_cpy_num; i += local_range) {
      reinterpret_cast<vec_type*>(v_block_ptr + dst_offset)[i] =
          reinterpret_cast<vec_type*>(v_block_ptr + src_offset)[i];
    }
    for (int i = local_id; i < scalar_cpy_num; i += local_range) {
      v_block_ptr[dst_offset + i + vector_cpy_total] =
          v_block_ptr[src_offset + i + vector_cpy_total];
    }
  }

 private:
  int64_t* key_caches_;
  int64_t* value_caches_;
  int64_t* mappings_;
  int copy_number_;
};

template <typename scalar_t>
void dpcpp_copy_block_kernel(
    int64_t* key_caches,
    int64_t* value_caches,
    int64_t* mappings,
    std::vector<int64_t>& k_cache_vec,
    std::vector<int64_t>& v_cache_vec,
    int copy_number,
    int num_pairs,
    int num_layers) {
  auto& queue = dpcppGetCurrentQueue();
  auto dev_id = dpcppGetDeviceIdOfCurrentQueue();
  int vec_size = 16;
  for (int i = 0; i < k_cache_vec.size(); ++i) {
    auto k_vec_size = at::native::Memory::can_vectorize_up_to<scalar_t>(
        dev_id, reinterpret_cast<char*>(k_cache_vec[i]));
    auto v_vec_size = at::native::Memory::can_vectorize_up_to<scalar_t>(
        dev_id, reinterpret_cast<char*>(v_cache_vec[i]));
    vec_size = std::min(std::min(k_vec_size, v_vec_size), vec_size);
  }
  int max_wg_size = dpcppMaxWorkGroupSize(dev_id);
  int wg_size = std::min(max_wg_size, copy_number / vec_size);
#define COPY_BLOCK_KERNEL(vec_sz)                                             \
  using vec_type = at::native::Memory::aligned_vector_loop<scalar_t, vec_sz>; \
  auto cgf = DPCPP_Q_CGF(cgh) {                                               \
    auto kfn = CopyBlockFunctor<scalar_t, vec_type, vec_sz>(                  \
        key_caches, value_caches, mappings, copy_number);                     \
    cgh.parallel_for(                                                         \
        sycl::nd_range<2>(                                                    \
            sycl::range<2>(num_pairs * wg_size, num_layers),                  \
            sycl::range<2>(wg_size, 1)),                                      \
        kfn);                                                                 \
  };                                                                          \
  DPCPP_Q_SUBMIT(queue, cgf);

  switch (vec_size) {
    case 16: {
      COPY_BLOCK_KERNEL(16)
      break;
    }
    case 8: {
      COPY_BLOCK_KERNEL(8)
      break;
    }
    case 4: {
      COPY_BLOCK_KERNEL(4)
      break;
    }
    case 2: {
      COPY_BLOCK_KERNEL(2)
      break;
    }
    case 1: {
      COPY_BLOCK_KERNEL(1)
      break;
    }
    default:
      TORCH_INTERNAL_ASSERT(
          false,
          "Unexpected vectorization size for copy blocks. vec size ",
          vec_size);
  }
#undef COPY_BLOCK_KERNEL
}

void copy_blocks(
    c10::ArrayRef<at::Tensor>
        key_caches, // [num_layers, num_blocks, num_head, head_size, block_size]
    c10::ArrayRef<at::Tensor> value_caches, // [num_layers, num_blocks,
                                            // num_head, head_size, block_size]
    at::Tensor& block_mapping // [num_pairs, 2]
) {
  int num_layers = key_caches.size();
  TORCH_CHECK(num_layers == value_caches.size());
  if (num_layers == 0)
    return;
  at::Device k_cache_device = key_caches[0].device();
  at::Device v_cache_device = value_caches[0].device();
  at::Device map_device = block_mapping.device();
  TORCH_CHECK(
      k_cache_device.is_xpu() && v_cache_device.is_xpu() &&
      map_device.is_xpu());
  at::DeviceGuard device_guard(key_caches[0].device());
  std::vector<int64_t> k_cache_ptr{};
  std::vector<int64_t> v_cache_ptr{};
  for (int i = 0; i < key_caches.size(); ++i) {
    k_cache_ptr.push_back(reinterpret_cast<int64_t>(key_caches[i].data_ptr()));
    v_cache_ptr.push_back(
        reinterpret_cast<int64_t>(value_caches[i].data_ptr()));
  }
  at::Tensor k_cache_ptr_tensor =
      at::from_blob(k_cache_ptr.data(), {num_layers}, at::kLong)
          .to(k_cache_device);
  at::Tensor v_cache_ptr_tensor =
      at::from_blob(v_cache_ptr.data(), {num_layers}, at::kLong)
          .to(v_cache_device);

  int64_t copy_number = key_caches[0][0].numel();
  IPEX_DISPATCH_ALL_TYPES_AND2(
      at::kHalf, at::kBFloat16, key_caches[0].scalar_type(), "copy_block", [&] {
        dpcpp_copy_block_kernel<scalar_t>(
            k_cache_ptr_tensor.data_ptr<int64_t>(),
            v_cache_ptr_tensor.data_ptr<int64_t>(),
            block_mapping.data_ptr<int64_t>(),
            k_cache_ptr,
            v_cache_ptr,
            copy_number,
            block_mapping.size(0),
            num_layers);
      });
}

enum class Fp8KVCacheDataType {
  kAuto = 0,
  kFp8E4M3 = 1,
  kFp8E5M2 = 2,
};

enum class ReshapeAndCacheOp {
  kChunkedPrefill = 0,
  kPagedAttention = 1,
};
template <
    typename scalar_t,
    typename cache_t,
    typename slot_type,
    Fp8KVCacheDataType kv_dt>
struct ReshapeAndCache {
  ReshapeAndCache(
      const scalar_t* key,
      const scalar_t* value,
      cache_t* key_cache,
      cache_t* value_cache,
      const slot_type* slot_mapping,
      int num_tokens,
      const int key_stride,
      const int value_stride,
      const int num_head,
      const int head_size,
      const int block_size,
      const int x,
      const float k_scale,
      const float v_scale)
      : key_(key),
        value_(value),
        key_cache_(key_cache),
        value_cache_(value_cache),
        slot_mapping_(slot_mapping),
        num_tokens_(num_tokens),
        key_stride_(key_stride),
        value_stride_(value_stride),
        num_head_(num_head),
        head_size_(head_size),
        block_size_(block_size),
        x_(x),
        k_scale(k_scale),
        v_scale(v_scale) {}

  void operator()(const sycl::nd_item<1> item_id) const {
    int group_idx = item_id.get_group(0);
    int local_idx = item_id.get_local_id(0);
    int local_range = item_id.get_local_range(0);
    int slot_idx = slot_mapping_[group_idx];
    if (slot_idx < 0)
      return;

    const int block_idx = slot_idx / block_size_;
    const int block_offset = slot_idx % block_size_;
    const int n = num_head_ * head_size_;
    for (int i = local_idx; i < n; i += local_range) {
      const int src_key_idx = group_idx * key_stride_ + i;
      const int src_value_idx = group_idx * value_stride_ + i;
      const int head_idx = i / head_size_;
      const int head_offset = i % head_size_;
      const int x_idx = head_offset / x_;
      const int x_offset = head_offset % x_;
      const int dst_key_idx = block_idx * n * block_size_ +
          head_idx * head_size_ * block_size_ + x_idx * block_size_ * x_ +
          block_offset * x_ + x_offset;
      const int dst_value_idx = block_idx * n * block_size_ +
          head_idx * head_size_ * block_size_ + head_offset * block_size_ +
          block_offset;
      scalar_t tgt_key = key_[src_key_idx];
      scalar_t tgt_value = value_[src_value_idx];
      if constexpr (kv_dt == Fp8KVCacheDataType::kFp8E5M2) {
        key_cache_[dst_key_idx] =
            static_cast<at::Float8_e5m2>(tgt_key * k_scale);
        value_cache_[dst_value_idx] =
            static_cast<at::Float8_e5m2>(tgt_value * v_scale);
      } else if constexpr (kv_dt == Fp8KVCacheDataType::kFp8E4M3) {
        key_cache_[dst_key_idx] =
            static_cast<at::Float8_e4m3fn>(tgt_key * k_scale);
        value_cache_[dst_value_idx] =
            static_cast<at::Float8_e4m3fn>(tgt_value * v_scale);
      } else { // kv_dt == Fp8KVCacheDataType::kAuto
        key_cache_[dst_key_idx] = tgt_key;
        value_cache_[dst_value_idx] = tgt_value;
      }
    }
  }

 private:
  const scalar_t* key_;
  const scalar_t* value_;
  cache_t* key_cache_;
  cache_t* value_cache_;
  const slot_type* slot_mapping_;
  int num_tokens_;
  const int key_stride_;
  const int value_stride_;
  const int num_head_;
  const int head_size_;
  const int block_size_;
  const int x_;
  const float k_scale;
  const float v_scale;
};

template <
    typename scalar_t,
    typename cache_t,
    typename slot_type,
    Fp8KVCacheDataType kv_dt>
struct ReshapeAndCacheFlash {
  ReshapeAndCacheFlash(
      const scalar_t* key,
      const scalar_t* value,
      cache_t* key_cache,
      cache_t* value_cache,
      const slot_type* slot_mapping,
      int num_tokens,
      const int block_stride,
      const int key_stride,
      const int value_stride,
      const int num_head,
      const int head_size,
      const int block_size,
      const float k_scale,
      const float v_scale)
      : key_(key),
        value_(value),
        key_cache_(key_cache),
        value_cache_(value_cache),
        slot_mapping_(slot_mapping),
        num_tokens_(num_tokens),
        block_stride_(block_stride),
        key_stride_(key_stride),
        value_stride_(value_stride),
        num_head_(num_head),
        head_size_(head_size),
        block_size_(block_size),
        k_scale(k_scale),
        v_scale(v_scale) {}

  void operator()(const sycl::nd_item<1> item_id) const {
    int group_idx = item_id.get_group(0);
    int local_idx = item_id.get_local_id(0);
    int local_range = item_id.get_local_range(0);
    int slot_idx = slot_mapping_[group_idx];
    if (slot_idx < 0)
      return;

    const int block_idx = slot_idx / block_size_;
    const int block_offset = slot_idx % block_size_;
    const int n = num_head_ * head_size_;
    for (int i = local_idx; i < n; i += local_range) {
      const int src_key_idx = group_idx * key_stride_ + i;
      const int src_value_idx = group_idx * value_stride_ + i;
      const int head_idx = i / head_size_;
      const int head_offset = i % head_size_;
      const int dst_idx = block_idx * block_stride_ + block_offset * n +
          head_idx * head_size_ + head_offset;
      scalar_t tgt_key = key_[src_key_idx];
      scalar_t tgt_value = value_[src_value_idx];
      if constexpr (kv_dt == Fp8KVCacheDataType::kFp8E4M3) {
        key_cache_[dst_idx] = static_cast<at::Float8_e4m3fn>(tgt_key * k_scale);
        value_cache_[dst_idx] =
            static_cast<at::Float8_e4m3fn>(tgt_value * v_scale);
      } else if constexpr (kv_dt == Fp8KVCacheDataType::kFp8E5M2) {
        key_cache_[dst_idx] = static_cast<at::Float8_e5m2>(tgt_key * k_scale);
        value_cache_[dst_idx] =
            static_cast<at::Float8_e5m2>(tgt_value * v_scale);
      } else {
        key_cache_[dst_idx] = tgt_key;
        value_cache_[dst_idx] = tgt_value;
      }
    }
  }

 private:
  const scalar_t* key_;
  const scalar_t* value_;
  cache_t* key_cache_;
  cache_t* value_cache_;
  const slot_type* slot_mapping_;
  int num_tokens_;
  const int block_stride_;
  const int key_stride_;
  const int value_stride_;
  const int num_head_;
  const int head_size_;
  const int block_size_;
  const float k_scale;
  const float v_scale;
};

template <
    typename scalar_t,
    typename cache_t,
    typename slot_type,
    Fp8KVCacheDataType kv_dt>
void dpcpp_reshape_and_cache_kernel(
    const scalar_t* key,
    const scalar_t* value,
    cache_t* key_cache,
    cache_t* value_cache,
    const slot_type* slot_mapping,
    int num_tokens,
    const int key_stride,
    const int value_stride,
    const int num_head,
    const int head_size,
    const int block_size,
    const int x,
    const float k_scale,
    const float v_scale) {
  auto& queue = dpcppGetCurrentQueue();
  auto dev_id = dpcppGetDeviceIdOfCurrentQueue();
  int max_wg_size = dpcppMaxWorkGroupSize(dev_id);
  int wg = std::min(max_wg_size, int(num_head * head_size));
  auto cgf = DPCPP_Q_CGF(cgh) {
    auto kfn = ReshapeAndCache<scalar_t, cache_t, slot_type, kv_dt>(
        key,
        value,
        key_cache,
        value_cache,
        slot_mapping,
        num_tokens,
        key_stride,
        value_stride,
        num_head,
        head_size,
        block_size,
        x,
        k_scale,
        v_scale);
    cgh.parallel_for(
        sycl::nd_range<1>(sycl::range<1>(num_tokens * wg), sycl::range<1>(wg)),
        kfn);
  };
  DPCPP_Q_SUBMIT(queue, cgf);
}

template <
    typename scalar_t,
    typename cache_t,
    typename slot_type,
    Fp8KVCacheDataType kv_dt>
void dpcpp_reshape_and_cache_flash_kernel(
    const scalar_t* key,
    const scalar_t* value,
    cache_t* key_cache,
    cache_t* value_cache,
    const slot_type* slot_mapping,
    int num_tokens,
    const int block_stride,
    const int key_stride,
    const int value_stride,
    const int num_head,
    const int head_size,
    const int block_size,
    const float k_scale,
    const float v_scale) {
  auto& queue = dpcppGetCurrentQueue();
  auto dev_id = dpcppGetDeviceIdOfCurrentQueue();
  int max_wg_size = dpcppMaxWorkGroupSize(dev_id);
  int wg = std::min(max_wg_size, int(num_head * head_size));
  auto cgf = DPCPP_Q_CGF(cgh) {
    auto kfn = ReshapeAndCacheFlash<scalar_t, cache_t, slot_type, kv_dt>(
        key,
        value,
        key_cache,
        value_cache,
        slot_mapping,
        num_tokens,
        block_stride,
        key_stride,
        value_stride,
        num_head,
        head_size,
        block_size,
        k_scale,
        v_scale);
    cgh.parallel_for(
        sycl::nd_range<1>(sycl::range<1>(num_tokens * wg), sycl::range<1>(wg)),
        kfn);
  };
  DPCPP_Q_SUBMIT(queue, cgf);
}

// The following macro is used to dispatch the conversion function based on
// the data type of the key and value cache. The FN is a macro that calls a
// function with template<typename scalar_t, typename cache_t,
// Fp8KVCacheDataType kv_dt>.
#define DISPATCH_BY_KV_CACHE_DTYPE(SRC_DTYPE, KV_DTYPE, FN)                    \
  if (KV_DTYPE == "auto") {                                                    \
    if (SRC_DTYPE == at::ScalarType::Float) {                                  \
      FN(float, float, Fp8KVCacheDataType::kAuto);                             \
    } else if (SRC_DTYPE == at::ScalarType::Half) {                            \
      FN(at::Half, at::Half, Fp8KVCacheDataType::kAuto);                       \
    } else if (SRC_DTYPE == at::ScalarType::BFloat16) {                        \
      FN(at::BFloat16, at::BFloat16, Fp8KVCacheDataType::kAuto);               \
    } else {                                                                   \
      TORCH_CHECK(false, "Unsupported input type of kv cache: ", SRC_DTYPE);   \
    }                                                                          \
  } else {                                                                     \
    if (KV_DTYPE == "fp8" || KV_DTYPE == "fp8_e4m3") {                         \
      if (SRC_DTYPE == at::ScalarType::Float) {                                \
        FN(float, at::Float8_e4m3fn, Fp8KVCacheDataType::kFp8E4M3);            \
      } else if (SRC_DTYPE == at::ScalarType::Half) {                          \
        FN(at::Half, at::Float8_e4m3fn, Fp8KVCacheDataType::kFp8E4M3);         \
      } else if (SRC_DTYPE == at::ScalarType::BFloat16) {                      \
        FN(at::BFloat16, at::Float8_e4m3fn, Fp8KVCacheDataType::kFp8E4M3);     \
      } else {                                                                 \
        TORCH_CHECK(false, "Unsupported input type of kv cache: ", SRC_DTYPE); \
      }                                                                        \
    } else if (KV_DTYPE == "fp8_e5m2") {                                       \
      if (SRC_DTYPE == at::ScalarType::Float) {                                \
        FN(float, at::Float8_e5m2, Fp8KVCacheDataType::kFp8E5M2);              \
      } else if (SRC_DTYPE == at::ScalarType::Half) {                          \
        FN(at::Half, at::Float8_e5m2, Fp8KVCacheDataType::kFp8E5M2);           \
      } else if (SRC_DTYPE == at::ScalarType::BFloat16) {                      \
        FN(at::BFloat16, at::Float8_e5m2, Fp8KVCacheDataType::kFp8E5M2);       \
      } else {                                                                 \
        TORCH_CHECK(false, "Unsupported input type of kv cache: ", SRC_DTYPE); \
      }                                                                        \
    } else {                                                                   \
      TORCH_CHECK(false, "Unsupported data type of kv cache: ", KV_DTYPE);     \
    }                                                                          \
  }

// KV_T is the stored data type of kv-cache.
// CACHE_T is the data type of key and value tensors.
// KV_DTYPE is the real data type of kv-cache.
#define CALL_RESHAPE_AND_CACHE(KV_T, CACHE_T, KV_DTYPE)                       \
  IPEX_DISPATCH_ALL_TYPES_AND2(                                               \
      at::kHalf, at::kBFloat16, key.scalar_type(), "reshape_and_cache", [&] { \
        if (slot_map.scalar_type() == at::kLong)                              \
          dpcpp_reshape_and_cache_kernel<KV_T, CACHE_T, int64_t, KV_DTYPE>(   \
              reinterpret_cast<KV_T*>(key.data_ptr()),                        \
              reinterpret_cast<KV_T*>(value.data_ptr()),                      \
              reinterpret_cast<CACHE_T*>(key_cache.data_ptr()),               \
              reinterpret_cast<CACHE_T*>(value_cache.data_ptr()),             \
              slot_map.data_ptr<int64_t>(),                                   \
              num_tokens,                                                     \
              key_stride,                                                     \
              value_stride,                                                   \
              num_heads,                                                      \
              head_size,                                                      \
              block_size,                                                     \
              x,                                                              \
              k_scale,                                                        \
              v_scale);                                                       \
        else                                                                  \
          dpcpp_reshape_and_cache_kernel<KV_T, CACHE_T, int32_t, KV_DTYPE>(   \
              reinterpret_cast<KV_T*>(key.data_ptr()),                        \
              reinterpret_cast<KV_T*>(value.data_ptr()),                      \
              reinterpret_cast<CACHE_T*>(key_cache.data_ptr()),               \
              reinterpret_cast<CACHE_T*>(value_cache.data_ptr()),             \
              slot_map.data_ptr<int32_t>(),                                   \
              num_tokens,                                                     \
              key_stride,                                                     \
              value_stride,                                                   \
              num_heads,                                                      \
              head_size,                                                      \
              block_size,                                                     \
              x,                                                              \
              k_scale,                                                        \
              v_scale);                                                       \
      });

// KV_T is the stored data type of kv-cache.
// CACHE_T is the data type of key and value tensors.
// KV_DTYPE is the real data type of kv-cache.
#define CALL_RESHAPE_AND_CACHE_FLASH(KV_T, CACHE_T, KV_DTYPE)     \
  IPEX_DISPATCH_ALL_TYPES_AND2(                                   \
      at::kHalf,                                                  \
      at::kBFloat16,                                              \
      key.scalar_type(),                                          \
      "reshape_and_cache_flash",                                  \
      [&] {                                                       \
        if (slot_map.scalar_type() == at::kLong)                  \
          dpcpp_reshape_and_cache_flash_kernel<                   \
              KV_T,                                               \
              CACHE_T,                                            \
              int64_t,                                            \
              KV_DTYPE>(                                          \
              reinterpret_cast<KV_T*>(key.data_ptr()),            \
              reinterpret_cast<KV_T*>(value.data_ptr()),          \
              reinterpret_cast<CACHE_T*>(key_cache.data_ptr()),   \
              reinterpret_cast<CACHE_T*>(value_cache.data_ptr()), \
              slot_map.data_ptr<int64_t>(),                       \
              num_tokens,                                         \
              block_stride,                                       \
              key_stride,                                         \
              value_stride,                                       \
              num_heads,                                          \
              head_size,                                          \
              block_size,                                         \
              k_scale,                                            \
              v_scale);                                           \
        else                                                      \
          dpcpp_reshape_and_cache_flash_kernel<                   \
              KV_T,                                               \
              CACHE_T,                                            \
              int32_t,                                            \
              KV_DTYPE>(                                          \
              reinterpret_cast<KV_T*>(key.data_ptr()),            \
              reinterpret_cast<KV_T*>(value.data_ptr()),          \
              reinterpret_cast<CACHE_T*>(key_cache.data_ptr()),   \
              reinterpret_cast<CACHE_T*>(value_cache.data_ptr()), \
              slot_map.data_ptr<int32_t>(),                       \
              num_tokens,                                         \
              block_stride,                                       \
              key_stride,                                         \
              value_stride,                                       \
              num_heads,                                          \
              head_size,                                          \
              block_size,                                         \
              k_scale,                                            \
              v_scale);                                           \
      });

void reshape_and_cache(
    at::Tensor& key, // [num_tokens, num_heads, head_size]
    at::Tensor& value, // [num_tokens, num_heads, head_size]
    at::Tensor&
        key_cache, // [num_blocks, num_heads, head_size/x, block_size, x]
    at::Tensor& value_cache, // [num_blocks, num_heads, head_size, block_size]
    at::Tensor& slot_map, // [num_tokens]
    const std::string& kv_cache_dtype,
    const double k_scale,
    const double v_scale) {
  at::DeviceGuard device_guard(key.device());
  int num_tokens = slot_map.size(0);
  int num_heads = key.size(1);
  int head_size = key.size(2);
  int block_size = key_cache.size(3);
  int x = key_cache.size(4);

  int key_stride = key.stride(0);
  int value_stride = value.stride(0);
  DISPATCH_BY_KV_CACHE_DTYPE(
      key.dtype(), kv_cache_dtype, CALL_RESHAPE_AND_CACHE)
}

void reshape_and_cache_flash(
    at::Tensor& key, // [num_tokens, num_heads, head_size]
    at::Tensor& value, // [num_tokens, num_heads, head_size]
    at::Tensor& key_cache, // [num_blocks, block_size, num_heads, head_size]
    at::Tensor& value_cache, // [num_blocks, block_size, num_heads, head_size]
    at::Tensor& slot_map, // [num_tokens]
    const std::string& kv_cache_dtype,
    const double k_scale,
    const double v_scale) {
  at::DeviceGuard device_guard(key.device());
  int num_tokens = slot_map.size(0);
  int num_heads = key.size(1);
  int head_size = key.size(2);
  int block_size = key_cache.size(1);

  int key_stride = key.stride(0);
  int value_stride = value.stride(0);
  int block_stride = key_cache.stride(0);
  TORCH_CHECK(key_cache.stride(0) == value_cache.stride(0));
  DISPATCH_BY_KV_CACHE_DTYPE(
      key.dtype(), kv_cache_dtype, CALL_RESHAPE_AND_CACHE_FLASH)
}

IPEX_LIBRARY_FRAGMENT() {
  IPEX_OP_REGISTER_DISPATCH("swap_blocks", swap_blocks, c10::DispatchKey::XPU);
  IPEX_OP_REGISTER_DISPATCH("copy_blocks", copy_blocks, c10::DispatchKey::XPU);
  IPEX_OP_REGISTER_DISPATCH(
      "reshape_and_cache", reshape_and_cache, c10::DispatchKey::XPU);
  IPEX_OP_REGISTER_DISPATCH(
      "reshape_and_cache_flash",
      reshape_and_cache_flash,
      c10::DispatchKey::XPU);
}
} // namespace cache_op
} // namespace AtenIpexTypeXPU
} // namespace at
