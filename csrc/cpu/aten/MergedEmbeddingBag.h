#ifndef MERGEDEMB_H
#define MERGEDEMB_H
#include <ATen/AccumulateType.h>
#include <ATen/Tensor.h>
#include <dyndisp/DispatchStub.h>
#include <torch/all.h>
#include "utils/robin_hood.h"

namespace torch_ipex {
namespace cpu {

namespace {
using namespace at;
enum PoolingMode { SUM = 0, MEAN = 1 };

template <class T>
class EMBROW {
 public:
  T* data = nullptr;
  std::vector<T> arr;
  int32_t length;

  EMBROW(int32_t len) {
    length = len;
    arr.resize(length);
    data = &arr[0];
    memset(data, 0, len * sizeof(T));
  }
};

template <class T, int32_t emb_dim>
class EMBROWFixLen {
 public:
  T* data = nullptr;
  T arr[emb_dim];

  EMBROWFixLen(int32_t len) {
    data = &arr[0];
    memset(data, 0, emb_dim * sizeof(T));
  }
};

/**
 * EmbeddingRowCache is used for 2 purpose:
 * (1) For low precision data type, we need accumulate grads or lookup results
 * on FP32 data type. EmbeddingRowCache is where we hold the accumulate results.
 * (2) For backward and row-wise distributed merged emb, we wish to represent
 * final results with sparse representation, in this case we do wish to allocate
 * a large contiguous buffer to store the results, then we store them in
 * EmbeddingRowCache with smaller memory usage.
 *
 * EmbeddingRowCache contains var length EmbRow hash map and Fixed length EmbRow
 * with len=64, 128, 256 And handle different lenght inside EmbeddingRowCache
 * without expose len info to users.
 *
 * The robin_hood::unordered_map<int64_t, T*> _cached_ptr is used because user
 * need to iterate this cache and we wish to provide a unfied api to return the
 * iterated object without introducing length info in user code.
 *
 * Why we need fixed length EmbRow:
 *     We will allocate memory to hold emb row very frequently during Embedding
 * FW/BW, we wish to allocate the memory on stack by using temporal varalble
 * instead of allocating them in heap for performance consideration. So we use C
 * array to hold fixed length and use std::vector to hold var lenght
 * (std::vector will use memory on heap).
 *
 * How to use:
 *
 * T* find(int32_t Key)
 *    Return the data-ptr for row-id "key" if key is in EmbeddingRowCache,
 *    else return nullptr
 * T* emplace(const int64_t key, int32_t emb_dim)
 *    When Key is not in EmbeddingRowCache, create a row and set the value to
 *    zero, emplace this (key, row) in EmbeddingRowCache and return the data-ptr
 *    for this row
 * T* emplace(const int64_t key, T* data, int32_t emb_dim)
 *    When Key is not in EmbeddingRowCache, create a row and set the value equal
 *    with T* data, emplace this (key, row) in EmbeddingRowCache and return the
 *    data-ptr for this row
 *
 * int64_t size()
 *    return the cache size
 *
 * const robin_hood::unordered_map<int64_t, T*> cache()
 *    return the hash map of EmbeddingRowCache to iterate purpose
 */
template <class T>
class EmbeddingRowCache {
  robin_hood::unordered_map<int64_t, EMBROW<T>> _cache;
  robin_hood::unordered_map<int64_t, EMBROWFixLen<T, 64>> _cache64;
  robin_hood::unordered_map<int64_t, EMBROWFixLen<T, 128>> _cache128;
  robin_hood::unordered_map<int64_t, EMBROWFixLen<T, 256>> _cache256;
  robin_hood::unordered_map<int64_t, T*> _cached_ptr;
  int32_t _emb_dim = -1;

 public:
  T* find(int64_t key) {
    auto find = _cached_ptr.find(key);
    if (find == _cached_ptr.end())
      return nullptr;
    return find->second;
  }

  T* emplace(const int64_t key, int32_t emb_dim) {
    _emb_dim = emb_dim;
    T* ptr = nullptr;
    switch (_emb_dim) {
      case 64:
        ptr = _cache64.emplace(key, 64).first->second.data;
        break;
      case 128:
        ptr = _cache128.emplace(key, 128).first->second.data;
        break;
      case 256:
        ptr = _cache256.emplace(key, 256).first->second.data;
        break;
      default:
        ptr = _cache.emplace(key, emb_dim).first->second.data;
    }
    _cached_ptr.emplace(key, ptr);
    return ptr;
  }

  T* emplace(const int64_t key, T* data, int32_t emb_dim) {
    T* ptr = emplace(key, emb_dim);
    memcpy(ptr, data, sizeof(T) * emb_dim);
    return ptr;
  }

  int64_t size() {
    return _cached_ptr.size();
  }

  const robin_hood::unordered_map<int64_t, T*> cache() const {
    return _cached_ptr;
  }
};

struct SGDArgs {
  SGDArgs(const TensorList& bf16_trail_, float weight_decay_, float lr_)
      : bf16_trail(bf16_trail_), weight_decay(weight_decay_), lr(lr_) {}

  TensorList bf16_trail;
  float weight_decay;
  float lr;
};

struct AdaGradArgs {
  AdaGradArgs(
      const TensorList& bf16_trail_,
      const TensorList& hessian_,
      float eps_,
      float lr_)
      : bf16_trail(bf16_trail_), hessian(hessian_), eps(eps_), lr(lr_) {}

  TensorList bf16_trail;
  TensorList hessian;
  float eps;
  float lr;
};

template <typename data_t, typename acc_t, typename optimizer_args_t>
class EmbeddingGradUpdate {};

template <typename data_t, typename acc_t>
class EmbeddingGradUpdate<data_t, acc_t, SGDArgs> {
 public:
  static void update(
      data_t* weight,
      const EmbeddingRowCache<acc_t>& ewc,
      const SGDArgs& args,
      const int32_t table_id,
      const int64_t emb_dim);
};

template <typename data_t, typename acc_t>
class EmbeddingGradUpdate<data_t, acc_t, AdaGradArgs> {
 public:
  static void update(
      data_t* weight,
      const EmbeddingRowCache<acc_t>& ewc,
      const AdaGradArgs& args,
      const int32_t table_id,
      const int64_t emb_dim);
};

std::vector<Tensor> merged_embeddingbag_forward_cpu_kernel_impl(
    const std::vector<Tensor>& weights,
    const TensorList& indices,
    const TensorList& offsets,
    const int64_t pooling_mode,
    const bool include_last_offsets);

std::vector<Tensor> merged_embeddingbag_backward_cpu_kernel_impl(
    const TensorList& grad_outs_,
    const TensorList& weights,
    const TensorList& indices,
    const TensorList& offsets,
    const int64_t pooling_mode,
    const bool include_last_offsets);

void merged_embeddingbag_backward_sgd_cpu_kernel_impl(
    const TensorList& grad_outs_,
    const TensorList& weights,
    const TensorList& indices,
    const TensorList& offsets,
    const int64_t pooling_mode,
    const bool include_last_offsets,
    const TensorList& bf16_trail,
    const double weight_decay,
    const double lr);

void merged_embeddingbag_backward_adagrad_cpu_kernel_impl(
    const TensorList& grad_outs_,
    const TensorList& weights,
    const TensorList& indices,
    const TensorList& offsets,
    const int64_t pooling_mode,
    const bool include_last_offsets,
    const TensorList& hessian,
    const TensorList& bf16_trail,
    const double eps,
    const double lr);

std::tuple<std::vector<Tensor>, std::vector<Tensor>, std::vector<Tensor>>
mergedemb_distribute_forward_local_kernel_impl(
    const Tensor& weight,
    const std::vector<int64_t> row_offset,
    const TensorList& indices,
    const TensorList& offset,
    const int64_t rank,
    const int64_t world_size,
    const bool include_last_offsets);

std::tuple<std::vector<Tensor>, std::vector<Tensor>, std::vector<Tensor>>
mergedemb_distribute_backward_local_kernel_impl(
    const Tensor& grad,
    const std::vector<int64_t> row_offset,
    const TensorList& indices,
    const TensorList& offset,
    const int64_t rank,
    const int64_t world_size,
    const bool include_last_offsets);

void mergedemb_distribute_forward_merge_cpu(
    Tensor& output,
    const TensorList& idx,
    const TensorList& val,
    const TensorList& ofs,
    const int64_t num_emb);

void mergedemb_distribute_backward_merge_adagrad_update_cpu(
    const TensorList& idx,
    const TensorList& val,
    const TensorList& ofs,
    Tensor& weights,
    Tensor& hessian,
    Tensor& weight_trail,
    const float lr,
    const float eps);

} // namespace

using merged_embeddingbag_forward_cpu_kernel_fn = std::vector<Tensor> (*)(
    const std::vector<Tensor>&,
    const TensorList&,
    const TensorList&,
    const int64_t,
    const bool);
IPEX_DECLARE_DISPATCH(
    merged_embeddingbag_forward_cpu_kernel_fn,
    merged_embeddingbag_forward_cpu_kernel_stub);

using merged_embeddingbag_backward_cpu_kernel_fn = std::vector<Tensor> (*)(
    const TensorList&,
    const TensorList&,
    const TensorList&,
    const TensorList&,
    const int64_t,
    const bool);
IPEX_DECLARE_DISPATCH(
    merged_embeddingbag_backward_cpu_kernel_fn,
    merged_embeddingbag_backward_cpu_kernel_stub);

using merged_embeddingbag_backward_sgd_cpu_kernel_fn = void (*)(
    const TensorList&,
    const TensorList&,
    const TensorList&,
    const TensorList&,
    const int64_t,
    const bool,
    const TensorList&,
    const double,
    const double);
IPEX_DECLARE_DISPATCH(
    merged_embeddingbag_backward_sgd_cpu_kernel_fn,
    merged_embeddingbag_backward_sgd_cpu_kernel_stub);

using merged_embeddingbag_backward_adagrad_cpu_kernel_fn = void (*)(
    const TensorList&,
    const TensorList&,
    const TensorList&,
    const TensorList&,
    const int64_t,
    const bool,
    const TensorList&,
    const TensorList&,
    const double,
    const double);
IPEX_DECLARE_DISPATCH(
    merged_embeddingbag_backward_adagrad_cpu_kernel_fn,
    merged_embeddingbag_backward_adagrad_cpu_kernel_stub);

using mergedemb_distribute_forward_local_kernel_fn = std::
    tuple<std::vector<Tensor>, std::vector<Tensor>, std::vector<Tensor>> (*)(
        const Tensor&,
        const std::vector<int64_t>,
        const TensorList&,
        const TensorList&,
        const int64_t,
        const int64_t,
        const bool);
IPEX_DECLARE_DISPATCH(
    mergedemb_distribute_forward_local_kernel_fn,
    mergedemb_distribute_forward_local_kernel_stub);

using mergedemb_distribute_forward_merge_kernel_fn = void (*)(
    Tensor&,
    const TensorList&,
    const TensorList&,
    const TensorList&,
    const int64_t);
IPEX_DECLARE_DISPATCH(
    mergedemb_distribute_forward_merge_kernel_fn,
    mergedemb_distribute_forward_merge_kernel_stub);

using mergedemb_distribute_backward_local_kernel_fn = std::
    tuple<std::vector<Tensor>, std::vector<Tensor>, std::vector<Tensor>> (*)(
        const Tensor&,
        const std::vector<int64_t>,
        const TensorList&,
        const TensorList&,
        const int64_t,
        const int64_t,
        const bool);
IPEX_DECLARE_DISPATCH(
    mergedemb_distribute_backward_local_kernel_fn,
    mergedemb_distribute_backward_local_kernel_stub);

using mergedemb_distribute_backward_merge_adagrad_update_fn = void (*)(
    const TensorList&,
    const TensorList&,
    const TensorList&,
    Tensor&,
    Tensor&,
    Tensor&,
    const double,
    const double);
IPEX_DECLARE_DISPATCH(
    mergedemb_distribute_backward_merge_adagrad_update_fn,
    mergedemb_distribute_backward_merge_adagrad_update_stub);

} // namespace cpu
} // namespace torch_ipex

#endif