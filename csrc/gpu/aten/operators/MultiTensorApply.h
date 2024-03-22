#pragma once
#include <ATen/ATen.h>
#include <iostream>
#include <vector>
#include "Loops.h"
#include "MemoryAccess.h"
// #include "autograd/function.h"
#include <aten/core/HostAllocator.h>
#include "c10/core/ScalarType.h"
#include "runtime/Memory.h"

namespace at {
namespace AtenIpexTypeXPU {
// Instruction level Parallelism, namely vec size
static constexpr int64_t kILP = 4;
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

template <typename scalar_vals_t, int n>
struct TLMetaForAddressScalar {
  void* addresses[n];
  uint32_t numel_to_tensor;
  scalar_vals_t scalar_vals;
};

template <int n>
struct TLMetaForAddress {
  void* addresses[n];
  uint32_t numel_to_tensor;
};

template <int n>
struct TLFusedMetaForAddress {
  void* addresses[n];
  uint32_t numel_to_tensor;
  void* state_steps_addresses;
};

struct TLMetaForWG {
  uint32_t wg_to_tensor;
  uint32_t wg_to_chunk;
};

template <typename T, typename Y, typename U, typename... ArgTypes>
struct MultiTensorApplyKernelFunctor {
  void operator()(sycl::nd_item<1> item_id) const {
    // Expand the tuple elements manually and call the callable
    expandAndCall(item_id, std::index_sequence_for<ArgTypes...>());
  }
  MultiTensorApplyKernelFunctor(
      int64_t kChunkSize_,
      T tlAddressMeta_,
      Y tlWGMeta_,
      U callable_,
      ArgTypes... args_)
      : kChunkSize(kChunkSize_),
        tlAddressMeta(tlAddressMeta_),
        tlWGMeta(tlWGMeta_),
        callable(callable_),
        args(std::make_tuple(args_...)) {}

 private:
  template <std::size_t... Indices>
  void expandAndCall(sycl::nd_item<1> item_id, std::index_sequence<Indices...>)
      const {
    // Call the callable with expanded tuple elements
    callable(
        kChunkSize,
        tlAddressMeta,
        tlWGMeta,
        item_id,
        std::get<Indices>(args)...);
  }

  int64_t kChunkSize;
  T tlAddressMeta;
  Y tlWGMeta;
  U callable;
  std::tuple<ArgTypes...> args;
};

template <
    bool fused_kernel,
    typename T,
    typename Y,
    typename U,
    typename... ArgTypes>
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
  if constexpr (fused_kernel) {
    max_wg_size = dpcppMaxWorkItemsPerEU(dev_id);
    kChunkSize = max_wg_size * kILP;
  }

  auto cgf = DPCPP_Q_CGF(__cgh) {
    MultiTensorApplyKernelFunctor<T, Y, U, ArgTypes...> kfn(
        kChunkSize, tlAddressMeta, tlWGMeta, callable, args...);
    __cgh.parallel_for<decltype(kfn)>(
        sycl::nd_range<1>(
            sycl::range<1>(global_size * max_wg_size),
            sycl::range<1>(max_wg_size)),
        kfn);
  };
  DPCPP_Q_SUBMIT(dpcpp_queue, cgf);
}

template <int depth, typename scalar_t, typename T, typename... ArgTypes>
void multi_tensor_apply(
    std::vector<std::vector<at::Tensor>>& tensor_lists,
    at::ArrayRef<Scalar> scalars,
    T callable,
    ArgTypes... args) {
  TORCH_CHECK(
      tensor_lists.size() == depth,
      "Number of tensor lists has to match he depth");
  size_t n_tensors = tensor_lists[0].size();
  using scalar_vals_t = typename T::opmath_t;
  auto dev_id = torch_ipex::xpu::dpcpp::dpcppGetDeviceIdOfCurrentQueue();
  int64_t max_wg_size = torch_ipex::xpu::dpcpp::dpcppMaxWorkGroupSize(dev_id);
  int64_t kChunkSize = kElementPerThread * max_wg_size;
  // TensorListMetaData<depth>* tensorListMeta = nullptr;
  auto addressStorage = at::empty(
      {sizeof(TLMetaForAddressScalar<scalar_t, depth>) * n_tensors},
      tensor_lists[0][0].options().dtype(at::kByte));
  auto metaAddressInput = static_cast<TLMetaForAddressScalar<scalar_t, depth>*>(
      addressStorage.data_ptr());
  TLMetaForAddressScalar<scalar_t, depth>* tlAddress = nullptr;

  // SYCL capture parameter size is limited and still gets functionality
  // issue in DPCPP implementation when capture parameter size is large.
  // There might be some unkonw computation error when the passed
  // structure is too large in ATSP (>512 Byte) or ATSM(> 1MB). In this
  // implementation, metadata is copied into device by H2D.
  tlAddress =
      (TLMetaForAddressScalar<scalar_t, depth>*)
          torch_ipex::xpu::dpcpp::HostAllocator::Instance()
              ->raw_allocate(
                  sizeof(TLMetaForAddressScalar<scalar_t, depth>) * n_tensors);
  uint64_t totalWG = 0;

  // this loop record all the tensor address and numel info.
  for (size_t t = 0; t < n_tensors; ++t) {
    auto numel = tensor_lists[0][t].numel();
    tlAddress[t].numel_to_tensor = numel;
    tlAddress[t].scalar_vals = scalars[t].to<scalar_t>();
    totalWG += (numel + kChunkSize - 1) / kChunkSize;
    for (int d = 0; d < depth; ++d) {
      tlAddress[t].addresses[d] = tensor_lists[d][t].data_ptr();
    }
  }
  torch_ipex::xpu::dpcpp::memcpyHostToDevice(
      metaAddressInput,
      tlAddress,
      sizeof(TLMetaForAddressScalar<scalar_t, depth>) * n_tensors,
      /*async*/ true);

  auto wgMetaStorage = at::empty(
      {sizeof(TLMetaForWG) * totalWG},
      tensor_lists[0][0].options().dtype(at::kByte));
  auto metaWGInput = static_cast<TLMetaForWG*>(wgMetaStorage.data_ptr());
  TLMetaForWG* tlWGMeta = nullptr;

  tlWGMeta = (TLMetaForWG*)torch_ipex::xpu::dpcpp::HostAllocator::Instance()
                 ->raw_allocate(sizeof(TLMetaForWG) * totalWG);
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
  torch_ipex::xpu::dpcpp::memcpyHostToDevice(
      metaWGInput, tlWGMeta, sizeof(TLMetaForWG) * totalWG, /*async*/ true);

  multi_tensor_apply_kernel<false>(
      metaAddressInput, metaWGInput, callable, totalWG, args...);

  // free
  torch_ipex::xpu::dpcpp::HostAllocator::Instance()->release(tlAddress);
  torch_ipex::xpu::dpcpp::HostAllocator::Instance()->release(tlWGMeta);
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
  auto dev_id = torch_ipex::xpu::dpcpp::dpcppGetDeviceIdOfCurrentQueue();
  int64_t max_wg_size = torch_ipex::xpu::dpcpp::dpcppMaxWorkGroupSize(dev_id);
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
  tlAddress =
      (TLMetaForAddress<depth>*)
          torch_ipex::xpu::dpcpp::HostAllocator::Instance()
              ->raw_allocate(sizeof(TLMetaForAddress<depth>) * n_tensors);
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
  torch_ipex::xpu::dpcpp::memcpyHostToDevice(
      metaAddressInput,
      tlAddress,
      sizeof(TLMetaForAddress<depth>) * n_tensors,
      /*async*/ true);

  auto wgMetaStorage = at::empty(
      {sizeof(TLMetaForWG) * totalWG},
      tensor_lists[0][0].options().dtype(at::kByte));
  auto metaWGInput = static_cast<TLMetaForWG*>(wgMetaStorage.data_ptr());
  TLMetaForWG* tlWGMeta = nullptr;

  tlWGMeta = (TLMetaForWG*)torch_ipex::xpu::dpcpp::HostAllocator::Instance()
                 ->raw_allocate(sizeof(TLMetaForWG) * totalWG);
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
  torch_ipex::xpu::dpcpp::memcpyHostToDevice(
      metaWGInput, tlWGMeta, sizeof(TLMetaForWG) * totalWG, /*async*/ true);

  multi_tensor_apply_kernel<false>(
      metaAddressInput, metaWGInput, callable, totalWG, args...);
  // free
  torch_ipex::xpu::dpcpp::HostAllocator::Instance()->release(tlAddress);
  torch_ipex::xpu::dpcpp::HostAllocator::Instance()->release(tlWGMeta);
}

template <int depth, typename T, typename... ArgTypes>
void multi_tensor_apply_for_fused_optimizer(
    std::vector<std::vector<at::Tensor>>& tensor_lists,
    at::TensorList state_steps,
    T callable,
    ArgTypes... args) {
  TORCH_CHECK(
      tensor_lists.size() == depth,
      "Number of tensor lists has to match the depth");
  const auto n_tensors = tensor_lists[0].size();
  // FusedOptimizerTensorListMetadata<depth> tensorListMeta;
  auto dev_id = torch_ipex::xpu::dpcpp::dpcppGetDeviceIdOfCurrentQueue();
  int64_t wg_size = dpcppMaxWorkItemsPerEU(dev_id);
  int64_t kChunkSize = wg_size * kILP;
  auto addressStorage = at::empty(
      {sizeof(TLFusedMetaForAddress<depth>) * n_tensors},
      tensor_lists[0][0].options().dtype(at::kByte));
  auto metaFusedAddressInput =
      static_cast<TLFusedMetaForAddress<depth>*>(addressStorage.data_ptr());
  TLFusedMetaForAddress<depth>* tlAddress = nullptr;

  // SYCL capture parameter size is limited and still gets functionality
  // issue in DPCPP implementation when capture parameter size is large.
  // There might be some unkonw computation error when the passed
  // structure is too large in ATSP (>512 Byte) or ATSM(> 1MB). In this
  // implementation, metadata is copied into device by H2D.
  tlAddress =
      (TLFusedMetaForAddress<depth>*)
          torch_ipex::xpu::dpcpp::HostAllocator::Instance()
              ->raw_allocate(sizeof(TLFusedMetaForAddress<depth>) * n_tensors);
  uint64_t totalWG = 0;

  // this loop record all the tensor address and numel info.
  for (size_t t = 0; t < n_tensors; ++t) {
    auto numel = tensor_lists[0][t].numel();
    tlAddress[t].numel_to_tensor = numel;
    tlAddress[t].state_steps_addresses = state_steps[t].data_ptr();
    totalWG += (numel + kChunkSize - 1) / kChunkSize;
    for (int d = 0; d < depth; ++d) {
      tlAddress[t].addresses[d] = tensor_lists[d][t].data_ptr();
    }
  }

  torch_ipex::xpu::dpcpp::memcpyHostToDevice(
      metaFusedAddressInput,
      tlAddress,
      sizeof(TLFusedMetaForAddress<depth>) * n_tensors,
      /*async*/ true);

  auto wgMetaStorage = at::empty(
      {sizeof(TLMetaForWG) * totalWG},
      tensor_lists[0][0].options().dtype(at::kByte));
  auto metaWGInput = static_cast<TLMetaForWG*>(wgMetaStorage.data_ptr());
  TLMetaForWG* tlWGMeta = nullptr;

  tlWGMeta = (TLMetaForWG*)torch_ipex::xpu::dpcpp::HostAllocator::Instance()
                 ->raw_allocate(sizeof(TLMetaForWG) * totalWG);
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
  torch_ipex::xpu::dpcpp::memcpyHostToDevice(
      metaWGInput, tlWGMeta, sizeof(TLMetaForWG) * totalWG, /*async*/ true);

  multi_tensor_apply_kernel<true>(
      metaFusedAddressInput, metaWGInput, callable, totalWG, args...);
  // free
  torch_ipex::xpu::dpcpp::HostAllocator::Instance()->release(tlAddress);
  torch_ipex::xpu::dpcpp::HostAllocator::Instance()->release(tlWGMeta);
}

} // namespace AtenIpexTypeXPU
} // namespace at
