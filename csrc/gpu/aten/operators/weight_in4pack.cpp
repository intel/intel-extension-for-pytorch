#include <ATen/ATen.h>
#include <torch/torch.h>

#include <core/Device.h>
#include <core/Memory.h>
#include <runtime/Utils.h>
#include <utils/DPCPP.h>

#include <aten/operators/MemoryAccess.h>
#include "Loops.h"
#include "comm/ATDispatch.h"
#include "comm/AccumulateType.h"
#include "comm/Atomics.h"
#include "comm/Numerics.h"

namespace at {
namespace AtenIpexTypeXPU {

namespace impl {

template <typename vec_t, typename in_t, typename out_t, int vec_len>
struct WeightToInt4packKernelFunctor {
  WeightToInt4packKernelFunctor(
      out_t* res_ptr_,
      in_t* in_ptr_,
      const int N_,
      const int K_,
      const int fold_len_,
      const uint64_t stride_k_)
      : res_ptr(res_ptr_),
        in_ptr(in_ptr_),
        N(N_),
        K(K_),
        fold_len(fold_len_),
        stride_k(stride_k_) {}
  void operator()(sycl::nd_item<2> item) const {
    uint64_t g_row = item.get_global_id()[0];
    uint64_t g_loc = item.get_global_id()[1];
    uint64_t k_index = g_loc * vec_len;
    if ((g_row < N) && (k_index < K)) {
      vec_t even =
          *(reinterpret_cast<vec_t*>(in_ptr + g_row * fold_len * K + k_index));
      vec_t odd = *(reinterpret_cast<vec_t*>(
          in_ptr + g_row * fold_len * K + K + k_index));
#pragma unroll
      for (int i = 0; i < vec_len; i++) {
        res_ptr[(k_index + i) * N + g_row] =
            (((out_t)(odd[i])) << 4) | ((out_t)(even[i]));
      }
    }
  }

 private:
  out_t* res_ptr;
  in_t* in_ptr;
  int N;
  int K;
  int fold_len;
  uint64_t stride_k;
};

template <typename in_t, typename out_t, int vec_len>
struct WeightToInt4packKernelFunctor<int32_t, in_t, out_t, vec_len> {
  WeightToInt4packKernelFunctor(
      out_t* res_ptr_,
      in_t* in_ptr_,
      const int N_,
      const int K_,
      const int fold_len_,
      const uint64_t stride_k_)
      : res_ptr(res_ptr_),
        in_ptr(in_ptr_),
        N(N_),
        K(K_),
        fold_len(fold_len_),
        stride_k(stride_k_) {}
  void operator()(sycl::nd_item<2> item) const {
    uint64_t g_row = item.get_global_id()[0];
    uint64_t g_loc = item.get_global_id()[1];
#pragma unroll
    for (int i = 0; i < vec_len; i++) {
      uint64_t k_index = g_loc + i * stride_k;
      if ((g_row < N) && (k_index < K)) {
        out_t even = (out_t)(in_ptr[g_row * fold_len * K + k_index]);
        out_t odd = (out_t)(in_ptr[g_row * fold_len * K + K + k_index]);
        res_ptr[k_index * N + g_row] = ((odd << 4) | even);
      }
    }
  }

 private:
  out_t* res_ptr;
  in_t* in_ptr;
  int N;
  int K;
  int fold_len;
  uint64_t stride_k;
};

void get_group_param(
    int N,
    int K,
    int fold_len,
    uint64_t& global_row,
    uint64_t& global_col,
    uint64_t& local_row,
    uint64_t& local_col,
    uint64_t& float_len,
    uint64_t& stride_k) {
  auto dev_id = dpcppGetDeviceIdOfCurrentQueue();
  int64_t maxWGSize = dpcppMaxWorkGroupSize(dev_id);
  auto max_group = dpcppMaxWorkItemsPerTile(dev_id) / maxWGSize;

  auto* dev_prop =
      at::xpu::getDeviceProperties(dpcppGetDeviceIdOfCurrentQueue());
  auto sub_group_size = dev_prop->sub_group_sizes;
  int SIMD = sub_group_size[1];
  if ((SIMD == SIMD32) && (K <= 16)) {
    SIMD = SIMD16;
    float_len = 1;
    local_col = SIMD;
    local_row = maxWGSize / local_col;
    global_col = 1;
    global_row = CeilDiv(static_cast<uint64_t>(N / fold_len), local_row);
    stride_k = global_col * local_col;
    return;
  }
  if (K > 16 && K <= 32) {
    float_len = 1;
  } else if (K > 32 && K <= 64) {
    float_len = 2;
  } else {
    float_len = 4;
  }
  local_col = SIMD;
  local_row = maxWGSize / local_col;
  global_col = CeilDiv(static_cast<uint64_t>(K), local_col * float_len);
  global_row = CeilDiv(static_cast<uint64_t>(N / fold_len), local_row);
  stride_k = global_col * local_col;
  return;
}

void weight_to_int4pack_kernel(
    const Tensor& weight_packed,
    const Tensor& weight,
    int N,
    int K,
    int fold_len) {
  using in_t = int32_t;
  using out_t = uint8_t;
  // int fold_len = 2;
  auto& queue = dpcppGetCurrentQueue();
  uint64_t global_row = 0, global_col = 0, local_row = 0, local_col = 0,
           float_len = 0, stride_k = 0;
  get_group_param(
      N,
      K,
      fold_len,
      global_row,
      global_col,
      local_row,
      local_col,
      float_len,
      stride_k);
  auto weight_packed_data = reinterpret_cast<out_t*>(weight_packed.data_ptr());
  const auto weight_data = weight.data_ptr<in_t>();

#define VEC_WEIGHT_PACK_KERNEL_FUNC(vec_len, vec_t)                           \
  {                                                                           \
    auto cgf = DPCPP_Q_CGF(cgh) {                                             \
      WeightToInt4packKernelFunctor<vec_t, in_t, out_t, vec_len> kernel(      \
          weight_packed_data,                                                 \
          weight_data,                                                        \
          N / fold_len,                                                       \
          K,                                                                  \
          fold_len,                                                           \
          stride_k);                                                          \
      cgh.parallel_for(                                                       \
          sycl::nd_range<2>(                                                  \
              sycl::range<2>(global_row * local_row, global_col * local_col), \
              sycl::range<2>(local_row, local_col)),                          \
          kernel);                                                            \
    };                                                                        \
    DPCPP_Q_SUBMIT(queue, cgf);                                               \
  }

#define WEIGHT_PACK_KERNEL(vec_len)                                       \
  {                                                                       \
    using vec_t = at::native::Memory::aligned_vector_loop<in_t, vec_len>; \
    constexpr int align_bytes = alignof(vec_t);                           \
    int in_start = ((uint64_t)weight_data) % align_bytes / sizeof(in_t);  \
    if (in_start == 0 && K % vec_len == 0) {                              \
      VEC_WEIGHT_PACK_KERNEL_FUNC(vec_len, vec_t);                        \
    } else {                                                              \
      VEC_WEIGHT_PACK_KERNEL_FUNC(vec_len, in_t);                         \
    }                                                                     \
  }

  switch (float_len) {
    case 1: {
      WEIGHT_PACK_KERNEL(1);
      break;
    }
    case 2: {
      WEIGHT_PACK_KERNEL(2);
      break;
    }
    case 4: {
      WEIGHT_PACK_KERNEL(4);
      break;
    }
    default:
      TORCH_INTERNAL_ASSERT(
          false,
          "Unexpected vectorization size for weight_to_int4pack. vec size ",
          float_len);
  }
}
} // namespace impl

Tensor _convert_weight_to_int4pack(const Tensor& in, int64_t innerKTiles) {
  TORCH_CHECK(in.dim() == 2, __func__, " : expect weight to be 2D tensor.");
  TORCH_CHECK(in.dtype() == at::kInt, __func__, " : expect weight to be kInt.");
  TORCH_CHECK(
      innerKTiles == 2 || innerKTiles == 4 || innerKTiles == 8,
      __func__,
      " : innerKTiles need to be 2, 4, or 8, got ",
      innerKTiles);

  auto weight = in.contiguous();
  auto N = weight.size(0);
  auto K = weight.size(1);

  // Create fake shapes for cpu. The meta registration in dynamo requires
  // operator has the same output shape for each device. So creating a fake
  // shape {N / 8, K / (16 * innerKTiles), 32, innerKTiles / 2}
  constexpr int64_t kNTileSize = 8;
  constexpr int64_t kKTileSize = 16;
  auto nTiles = (N + kNTileSize - 1) / kNTileSize;

  TORCH_CHECK(N % 16 == 0, __func__, " : expect N to be dividable by 16");
  const int64_t kSuperKTileSize = kKTileSize * innerKTiles;
  TORCH_CHECK(
      K % kSuperKTileSize == 0,
      __func__,
      " : epxect K to be dividable by ",
      kSuperKTileSize);
  auto kSuperTiles = (K + kSuperKTileSize - 1) / kSuperKTileSize;

  auto weight_packed = at::empty(
      {nTiles, kSuperTiles, 32, innerKTiles / 2},
      at::TensorOptions().dtype(at::kInt).device(in.device()));

  impl::weight_to_int4pack_kernel(weight_packed, weight, N, K, 2);
  return weight_packed;
}

} // namespace AtenIpexTypeXPU
} // namespace at
