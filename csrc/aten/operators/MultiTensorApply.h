#pragma once
#include <ATen/ATen.h>
#include <iostream>
#include <vector>
#include "Loops.h"
#include "MemoryAccess.h"
// #include "autograd/function.h"
#include "c10/core/ScalarType.h"
#include "runtime/CachingHostAllocator.h"
#include "runtime/Memory.h"

namespace at {
namespace AtenIpexTypeXPU {
// Instruction level Parallelism, namely vec size
static constexpr int64_t kILP = 4;
// element number for one thread execution
static constexpr int64_t kElementPerThread = 128;

template <typename T>
bool is_aligned(T* p) {
  return ((uint64_t)p) % (kILP * sizeof(T)) == 0;
}

template <typename T>
void load_store(T* dst, T* src, int dst_offset, int src_offset) {
  using LT = at::native::Memory::aligned_vector_loop<T, kILP>;
  ((LT*)dst)[dst_offset] = ((LT*)src)[src_offset];
}

template <int n>
struct TLMetaForAddress {
  void* addresses[n];
  int numel_to_tensor;
};

struct TLMetaForWG {
  unsigned char wg_to_tensor;
  int wg_to_chunk;
};

template <typename T, typename Y, typename U, typename... ArgTypes>
void multi_tensor_apply_kernel(
    T tlAddressMeta,
    Y tlWGMeta,
    U callable,
    int global_size,
    ArgTypes... args) {
  auto& dpcpp_queue = dpcppGetCurrentQueue();
  auto dev_id = dpcppGetDeviceIdOfCurrentQueue();
  int64_t max_wg_size = dpcppMaxWorkGroupSize(dev_id);
  int64_t kChunkSize = max_wg_size * kElementPerThread;
  auto cgf = DPCPP_Q_CGF(__cgh) {
    auto kfn = DPCPP_Q_KFN(sycl::nd_item<1> item_id) {
      callable(kChunkSize, tlAddressMeta, tlWGMeta, item_id, args...);
    };
    __cgh.parallel_for(
        sycl::nd_range<1>(
            sycl::range<1>(global_size * max_wg_size),
            sycl::range<1>(max_wg_size)),
        kfn);
  };
  DPCPP_Q_SUBMIT(dpcpp_queue, cgf);
}

template <int depth, typename T, typename... ArgTypes>
void multi_tensor_apply(
    std::vector<std::vector<at::Tensor>>& tensor_lists,
    T callable,
    ArgTypes... args) {
  TORCH_CHECK(
      tensor_lists.size() == depth,
      "Number of tensor lists has to match he depth");
  size_t n_tensors = tensor_lists[0].size();
  using scalar_vals_t = typename T::opmath_t;
  auto dev_id = xpu::dpcpp::dpcppGetDeviceIdOfCurrentQueue();
  int64_t max_wg_size = xpu::dpcpp::dpcppMaxWorkGroupSize(dev_id);
  int64_t kChunkSize = kElementPerThread * max_wg_size;
  // TensorListMetaData<depth>* tensorListMeta = nullptr;
  auto addressStorage = at::empty(
      {sizeof(TLMetaForAddress<depth>) * n_tensors},
      tensor_lists[0][0].options().dtype(at::kByte));
  auto metaAddressInput =
      static_cast<TLMetaForAddress<depth>*>(addressStorage.data_ptr());
  TLMetaForAddress<depth>* tlAddress = nullptr;

  // SYCL capture parameter size is limited and still gets functionality
  // issue in DPCPP implementation when capture parameter size is large.
  // There might be some unkonw computation error when the passed
  // structure is too large in ATSP (>512 Byte) or ATSM(> 1MB). In this
  // implementation, metadata is copied into device by H2D.
  xpu::dpcpp::CachingHostAllocator::Instance()->malloc(
      (void**)&tlAddress, sizeof(TLMetaForAddress<depth>) * n_tensors);
  uint64_t totalWG = 0;

  // this loop record all the tensor address and numel info.
  for (size_t t = 0; t < n_tensors; ++t) {
    auto numel = tensor_lists[0][t].numel();
    tlAddress[t].numel_to_tensor = numel;
    totalWG += (numel + kChunkSize - 1) / kChunkSize;
    for (int d = 0; d < depth; ++d) {
      tlAddress[t].addresses[d] = tensor_lists[d][t].data_ptr();
    }
  }
  xpu::dpcpp::memcpyHostToDevice(
      metaAddressInput,
      tlAddress,
      sizeof(TLMetaForAddress<depth>) * n_tensors,
      /*async*/ true);
  xpu::dpcpp::CachingHostAllocator::Instance()->release(tlAddress);

  auto wgMetaStorage = at::empty(
      {sizeof(TLMetaForWG) * totalWG},
      tensor_lists[0][0].options().dtype(at::kByte));
  auto metaWGInput = static_cast<TLMetaForWG*>(wgMetaStorage.data_ptr());
  TLMetaForWG* tlWGMeta = nullptr;

  xpu::dpcpp::CachingHostAllocator::Instance()->malloc(
      (void**)&tlWGMeta, sizeof(TLMetaForWG) * totalWG);
  uint64_t posWG = 0;
  // this loop record the correspond tensor and chunk info for each work group.
  for (size_t t = 0; t < n_tensors; ++t) {
    auto numel = tensor_lists[0][t].numel();
    auto chunkForWG = (numel + kChunkSize - 1) / kChunkSize;
    for (size_t chunkId = 0; chunkId < chunkForWG; ++chunkId, ++posWG) {
      tlWGMeta[posWG].wg_to_tensor = t;
      tlWGMeta[posWG].wg_to_chunk = chunkId;
    }
  }
  TORCH_CHECK(
      posWG == totalWG,
      "Work group index dose not equal to the allocated memory size, segment fault might occur");
  xpu::dpcpp::memcpyHostToDevice(
      metaWGInput, tlWGMeta, sizeof(TLMetaForWG) * totalWG, /*async*/ true);
  xpu::dpcpp::CachingHostAllocator::Instance()->release(tlWGMeta);

  multi_tensor_apply_kernel(
      metaAddressInput, metaWGInput, callable, totalWG, args...);
}

} // namespace AtenIpexTypeXPU
} // namespace at
