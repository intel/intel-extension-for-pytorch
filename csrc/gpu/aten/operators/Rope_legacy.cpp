#include <ATen/ATen.h>
#include <ATen/core/Array.h>

#include <ATen/OpMathType.h>
#include <core/MemoryFormat.h>
#include <core/detail/IndexUtils.h>
#include <runtime/Utils.h>
#include <utils/DPCPP.h>
#include <cstdint>

#include "Reduce.h"
#include "comm/ATDispatch.h"
#include "comm/AccumulateType.h"

#include "comm/Numerics.h"
#include "utils/CustomOperatorRegistration.h"

using namespace torch_ipex::xpu::dpcpp;

namespace at {
namespace AtenIpexTypeXPU {

enum class EmbeddingAlgorithm { RotateHalf = 0, RotateInterleave = 1 };

template <
    typename scalar_t,
    int N,
    typename emb_scalar_t,
    EmbeddingAlgorithm Algo>
struct RotaryEmbedding {};

template <
    typename scalar_t,
    int N,
    typename emb_scalar_t,
    int noutput,
    int sin_offset,
    int cos_offset,
    typename OffsetCalculatorType>
struct RotaryEmbeddingKernelFunctor {
  void operator()(sycl::nd_item<2> item_id) const {
    auto item_idx = item_id.get_local_id(1);
    auto item_range = item_id.get_local_range(1);
    auto group_idx = item_id.get_group(1);
    auto group_id = item_id.get_group(0);
    auto sg = item_id.get_sub_group();

    for (int group_num = group_idx; group_num < total_group_num;
         group_num += max_group_num) {
      for (int i = item_idx; i < problem_size; i += item_range) {
#pragma unroll
        for (int j = 0; j < noutput; ++j) {
          scalar_t* output_ptr = static_cast<scalar_t*>(data_ptr[j]);
          scalar_t* input_ptr = static_cast<scalar_t*>(data_ptr[j + noutput]);
          emb_scalar_t* sin_ptr =
              static_cast<emb_scalar_t*>(data_ptr[sin_offset]);
          emb_scalar_t* cos_ptr =
              static_cast<emb_scalar_t*>(data_ptr[cos_offset]);
          auto global_offset = group_num * problem_size + i;
          const auto offset = offset_calc.get(global_offset);
          scalar_t val = *(input_ptr + offset[j + noutput]);
          scalar_t scale = i % 2 == 0 ? -1 : 1;
          scalar_t shift_val = sycl::permute_group_by_xor(sg, val, 1) * scale;
          float sin_val = static_cast<float>(*(sin_ptr + offset[sin_offset]));
          float cos_val = static_cast<float>(*(cos_ptr + offset[cos_offset]));
          *(output_ptr + offset[j]) =
              (scalar_t)((float)shift_val * sin_val + (float)val * cos_val);
        }
      }
    }
  }
  RotaryEmbeddingKernelFunctor(
      int64_t problem_size_,
      int64_t max_group_num_,
      int64_t total_group_num_,
      OffsetCalculatorType offset_calc_,
      void** data_ptr_)
      : problem_size(problem_size_),
        max_group_num(max_group_num_),
        total_group_num(total_group_num_),
        offset_calc(offset_calc_) {
    for (int i = 0; i < N; ++i) {
      data_ptr[i] = data_ptr_[i];
    }
  }

 private:
  int64_t problem_size;
  int64_t max_group_num;
  int64_t total_group_num;
  OffsetCalculatorType offset_calc;
  void* data_ptr[N];
};

template <typename scalar_t, int N, typename emb_scalar_t>
struct RotaryEmbedding<
    scalar_t,
    N,
    emb_scalar_t,
    EmbeddingAlgorithm::RotateInterleave> {
  void call(
      TensorIteratorBase& iter,
      int64_t problem_size,
      int64_t total_size) {
    auto& dpcpp_queue = dpcppGetCurrentQueue();
    auto dev_id = dpcppGetDeviceIdOfCurrentQueue();
    int64_t max_wg_size = dpcppMaxWorkGroupSize(dev_id);
    auto wg_size = std::min(max_wg_size, problem_size);

    int64_t max_group_num = dpcppMaxWorkItemsPerTile(dev_id) / wg_size;
    int64_t total_group_num = (total_size + problem_size - 1) / problem_size;
    max_group_num = std::min(max_group_num, total_group_num);
    auto offset_calc = make_element_offset_calculator<N>(iter);
    constexpr int noutput = (N - 2) / 2;
    constexpr int sin_offset = N - 2;
    constexpr int cos_offset = N - 1;
    void* data_ptr[N];
    for (int i = 0; i < N; ++i) {
      data_ptr[i] = iter.data_ptr(i);
    }
    TORCH_INTERNAL_ASSERT(2 * noutput + 2 == iter.ntensors());
    auto cgf = DPCPP_Q_CGF(cgh) {
      RotaryEmbeddingKernelFunctor<
          scalar_t,
          N,
          emb_scalar_t,
          noutput,
          sin_offset,
          cos_offset,
          decltype(offset_calc)>
          kfn(problem_size,
              max_group_num,
              total_group_num,
              offset_calc,
              data_ptr);
      cgh.parallel_for<decltype(kfn)>(
          sycl::nd_range<2>(
              sycl::range<2>({1, max_group_num * wg_size}),
              sycl::range<2>({1, wg_size})),
          kfn);
    };
    DPCPP_Q_SUBMIT(dpcpp_queue, cgf);
  }
};

template <
    typename scalar_t,
    int N,
    typename emb_scalar_t,
    int noutput,
    typename OffsetCalculatorType>
struct RotaryEmbeddingKernelFunctor2 {
  void operator()(sycl::nd_item<2> item_id) const {
    auto item_idx = item_id.get_local_id(1);
    auto group_idx = item_id.get_group(1);
    auto group_id = item_id.get_group(0);

    for (int group_num = group_idx; group_num < total_group_num;
         group_num += max_group_num) {
      for (int64_t i = item_idx; i < problem_half; i += wg_size) {
#pragma unroll
        for (int j = 0; j < noutput; ++j) {
          scalar_t* output_ptr = static_cast<scalar_t*>(data_ptr[j]);
          scalar_t* input_ptr = static_cast<scalar_t*>(data_ptr[j + noutput]);
          int64_t global_offset1 = group_num * problem_size + i;
          int64_t global_offset2 = global_offset1 + problem_half;
          const auto offset1 = offset_calc.get(global_offset1);
          const auto offset2 = offset_calc.get(global_offset2);
          float x1 = static_cast<float>(*(input_ptr + offset1[j + noutput]));
          float x2 = static_cast<float>(*(input_ptr + offset2[j + noutput]));
          float rotate_x1 = -x2;
          float rotate_x2 = x1;
          float sin_val = static_cast<float>(*(sin_ptr + offset1[2 * noutput]));
          float cos_val =
              static_cast<float>(*(cos_ptr + offset1[2 * noutput + 1]));
          float sin_val_half =
              static_cast<float>(*(sin_ptr + offset2[2 * noutput]));
          float cos_val_half =
              static_cast<float>(*(cos_ptr + offset2[2 * noutput + 1]));
          *(output_ptr + offset1[j]) =
              static_cast<scalar_t>(x1 * cos_val + rotate_x1 * sin_val);
          *(output_ptr + offset2[j]) = static_cast<scalar_t>(
              x2 * cos_val_half + rotate_x2 * sin_val_half);
        }
      }
    }
  }
  RotaryEmbeddingKernelFunctor2(
      int64_t problem_size_,
      int64_t problem_half_,
      int64_t wg_size_,
      int64_t max_group_num_,
      int64_t total_group_num_,
      OffsetCalculatorType offset_calc_,
      void** data_ptr_,
      emb_scalar_t* sin_ptr_,
      emb_scalar_t* cos_ptr_)
      : problem_size(problem_size_),
        problem_half(problem_half_),
        wg_size(wg_size_),
        max_group_num(max_group_num_),
        total_group_num(total_group_num_),
        offset_calc(offset_calc_),
        sin_ptr(sin_ptr_),
        cos_ptr(cos_ptr_) {
    for (int i = 0; i < N; ++i) {
      data_ptr[i] = data_ptr_[i];
    }
  }

 private:
  int64_t problem_size;
  int64_t problem_half;
  int64_t wg_size;
  int64_t max_group_num;
  int64_t total_group_num;
  OffsetCalculatorType offset_calc;
  void* data_ptr[N];
  emb_scalar_t* sin_ptr;
  emb_scalar_t* cos_ptr;
};

template <typename scalar_t, int N, typename emb_scalar_t>
struct RotaryEmbedding<
    scalar_t,
    N,
    emb_scalar_t,
    EmbeddingAlgorithm::RotateHalf> {
  void call(
      TensorIteratorBase& iter,
      int64_t problem_size,
      int64_t total_size) {
    auto& dpcpp_queue = dpcppGetCurrentQueue();
    auto dev_id = dpcppGetDeviceIdOfCurrentQueue();
    int64_t max_wg_size = dpcppMaxWorkGroupSize(dev_id);
    int64_t problem_half = problem_size / 2;
    int64_t wg_size = std::min(max_wg_size, problem_half);

    int64_t max_group_num = dpcppMaxWorkItemsPerTile(dev_id) / wg_size;
    int64_t total_group_num = (total_size + problem_size - 1) / problem_size;
    max_group_num = std::min(max_group_num, total_group_num);
    auto offset_calc = make_element_offset_calculator<N>(iter);
    constexpr int noutput = (N - 2) / 2;

    void* data_ptr[N];
    for (int i = 0; i < N; ++i) {
      data_ptr[i] = iter.data_ptr(i);
    }
    emb_scalar_t* sin_ptr = static_cast<emb_scalar_t*>(iter.data_ptr(N - 2));
    emb_scalar_t* cos_ptr = static_cast<emb_scalar_t*>(iter.data_ptr(N - 1));

    TORCH_INTERNAL_ASSERT(2 * noutput + 2 == iter.ntensors());
    TORCH_INTERNAL_ASSERT(2 * noutput + 2 == iter.ntensors());
    auto cgf = DPCPP_Q_CGF(cgh) {
      RotaryEmbeddingKernelFunctor2<
          scalar_t,
          N,
          emb_scalar_t,
          noutput,
          decltype(offset_calc)>
          kfn(problem_size,
              problem_half,
              wg_size,
              max_group_num,
              total_group_num,
              offset_calc,
              data_ptr,
              sin_ptr,
              cos_ptr);

      cgh.parallel_for<decltype(kfn)>(
          sycl::nd_range<2>(
              sycl::range<2>({1, max_group_num * wg_size}),
              sycl::range<2>({1, wg_size})),
          kfn);
    };
    DPCPP_Q_SUBMIT(dpcpp_queue, cgf);
  }
};

enum class RotaryCheck {
  CheckDim = 0,
  CheckProblemSize = 1,
};

template <RotaryCheck check>
struct RotaryEmbeddingCheck {};

template <>
struct RotaryEmbeddingCheck<RotaryCheck::CheckDim> {
  template <typename... Args>
  void call(int64_t sin_dim, Tensor& input_or_output, Args... args) {
    TORCH_CHECK(sin_dim == input_or_output.ndimension());
    call(sin_dim, args...);
  }
  void call(int64_t sin_dim) {
    return;
  }
};

template <>
struct RotaryEmbeddingCheck<RotaryCheck::CheckProblemSize> {
  template <typename... Args>
  void call(int64_t problem_size, Tensor& input_or_output, Args... args) {
    int64_t ndim = input_or_output.ndimension();
    TORCH_CHECK(
        problem_size == input_or_output.size(ndim - 1),
        "The problem size of all tensor should be equal");
    TORCH_CHECK(
        !(input_or_output.size(ndim - 1) & 1),
        "The problem size should be divisible by 2");
    call(problem_size, args...);
  }

  void call(int64_t sin_dim) {
    return;
  }
};

template <int total_size, int cur_size>
struct BuildTensorIterConfigFromArgs {
  template <typename... Args>
  void call(TensorIteratorConfig& config, Tensor& tensor, Args... args) {
    BuildTensorIterConfigFromArgs<total_size, cur_size - 1>().call(
        config, args...);
    // The first half should be input and the last half should be output
    if constexpr ((total_size >> 1) < cur_size) {
      config.add_input(tensor);
    } else {
      config.add_output(tensor);
    }
  }

  void call(TensorIteratorConfig& config) {
    return;
  }
};

template <EmbeddingAlgorithm Algo, typename... Args>
void apply_rotary_embedding(
    const Tensor& sin,
    const Tensor& cos,
    Args... args) {
  int64_t sin_dim = sin.ndimension();
  int64_t cos_dim = cos.ndimension();
  int64_t sin_prob_size = sin.size(sin_dim - 1);
  int64_t cos_prob_size = cos.size(cos_dim - 1);
  TORCH_CHECK(
      sin_prob_size == cos_prob_size,
      "The problem size of sin and cos should be same in rotary embedding");
  TORCH_CHECK(
      sin_dim == cos_dim,
      "The dimension of sin and cos should be the same in rotary embedding");
  RotaryEmbeddingCheck<RotaryCheck::CheckDim>().call(sin_dim, args...);
  RotaryEmbeddingCheck<RotaryCheck::CheckProblemSize>().call(
      sin.size(sin_dim - 1), args...);
  auto config = TensorIteratorConfig();
  BuildTensorIterConfigFromArgs<sizeof...(args), sizeof...(args)>().call(
      config, args...);
  auto iter =
      config.add_input(sin).add_input(cos).check_all_same_dtype(false).build();
  IPEX_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      iter.input_dtype(),
      "apply_rotary_embedding",
      [&]() {
        if (sin.scalar_type() == at::kFloat)
          RotaryEmbedding<scalar_t, sizeof...(args) + 2, float, Algo>().call(
              iter, sin_prob_size, iter.numel());
        else
          RotaryEmbedding<scalar_t, sizeof...(args) + 2, scalar_t, Algo>().call(
              iter, sin_prob_size, iter.numel());
      });
}

template <
    typename scalar_t,
    EmbeddingAlgorithm algo = EmbeddingAlgorithm::RotateHalf,
    bool has_cache_offset = true>
struct RotaryEmbeddingBatched {
  void operator()(sycl::nd_item<1> item) const {
    using accscalar = at::opmath_type<scalar_t>;
    int32_t token_id = item.get_group(0);
    int32_t start_idx = item.get_local_id(0);
    int64_t pos = positions_[token_id];
    int64_t cos_sin_offset = 0;
    if constexpr (has_cache_offset)
      int64_t cos_sin_offset = cos_sin_cache_offsets_[token_id];
    scalar_t* cos_cache = cos_sin_cache_ + (pos + cos_sin_offset) * rot_dim_;
    int32_t embed_dim = rot_dim_ / 2;
    scalar_t* sin_cache = cos_cache + embed_dim;
    scalar_t* query_base = query_ + token_id * query_stride_;
    scalar_t* key_base = key_ + token_id * key_stride_;
    int32_t q_num = num_heads_ * embed_dim;
    int32_t k_num = num_kv_heads_ * embed_dim;
    int32_t local_range = item.get_local_range(0);
    for (int i = start_idx; i < q_num; i += local_range) {
      int32_t head_id = i / embed_dim;
      int32_t rot_offset = i % embed_dim;
      scalar_t* query_st = query_base + head_id * head_size_;
      int x_idx, y_idx;
      accscalar cos_val, sin_val;
      if constexpr (algo == EmbeddingAlgorithm::RotateHalf) {
        x_idx = rot_offset;
        y_idx = rot_offset + embed_dim;
        cos_val = cos_cache[x_idx];
        sin_val = sin_cache[x_idx];
      } else {
        x_idx = 2 * rot_offset;
        y_idx = 2 * rot_offset + 1;
        cos_val = cos_cache[x_idx / 2];
        sin_val = sin_cache[x_idx / 2];
      }
      accscalar x = query_st[x_idx];
      accscalar y = query_st[y_idx];
      query_st[x_idx] = x * cos_val - y * sin_val;
      query_st[y_idx] = x * sin_val + y * cos_val;
      if (i < k_num) {
        scalar_t* key_st = key_base + head_id * head_size_;
        x = key_st[x_idx];
        y = key_st[y_idx];
        key_st[x_idx] = x * cos_val - y * sin_val;
        key_st[y_idx] = x * sin_val + y * cos_val;
      }
    }
  }

  int64_t* positions_;
  scalar_t* cos_sin_cache_;
  int64_t* cos_sin_cache_offsets_;
  scalar_t* query_;
  scalar_t* key_;
  int num_heads_;
  int num_kv_heads_;
  int query_stride_;
  int key_stride_;
  int64_t head_size_;
  int64_t rot_dim_;
};

namespace DSRotaryEmbedding {
template <typename T, int64_t rotary_dim, bool is_neox>
struct FusedDSRotaryEmbeddingQK {
  static constexpr int sg_size = 16;
  static constexpr int64_t sg_no = 1;
  FusedDSRotaryEmbeddingQK(
      const int64_t* positions,
      const T* query,
      const T* key,
      const int64_t* offsets,
      const T* cos_sin_cache,
      T* query_out,
      T* key_out,
      const int64_t batch,
      const int64_t q_num_head,
      const int64_t k_num_head,
      const int64_t head_size,
      const int64_t q_num_head_d,
      const int64_t q_batch_d,
      const int64_t k_num_head_d,
      const int64_t k_batch_d)
      : positions(positions),
        query(query),
        key(key),
        offsets(offsets),
        cos_sin_cache(cos_sin_cache),
        query_out(query_out),
        key_out(key_out),
        batch(batch),
        q_num_head(q_num_head),
        k_num_head(k_num_head),
        head_size(head_size),
        q_num_head_d(q_num_head_d),
        q_batch_d(q_batch_d),
        k_num_head_d(k_num_head_d),
        k_batch_d(k_batch_d) {}

  static inline sycl::nd_range<3> get_nd_range(
      const int64_t batch,
      const int64_t q_num_head,
      const int64_t k_num_head) {
    const int64_t sg_per_heads = divup(q_num_head + k_num_head, sg_size);
    // const int64_t thd_per_heads = sg_per_heads * sg_size;
    sycl::range<3> local(1, sg_per_heads, sg_size);
    sycl::range<3> global(batch, sg_per_heads, sg_size);
    return sycl::nd_range<3>(global, local);
  }

  void rotary_emb_kern(
      const int64_t position,
      const T* pe,
      const T* cos_sin_cache,
      T* res) const {
    constexpr int64_t half_rotary_dim = rotary_dim / 2;
    constexpr int64_t vec_2_len = 2;
    using v2_type = sycl::vec<T, vec_2_len>;
    const int64_t cache_idx = position * rotary_dim;
    const T* cos_cache_offset = &cos_sin_cache[cache_idx];
    const T* sin_cache_offset = cos_cache_offset + half_rotary_dim;
    if constexpr (is_neox) {
      // repeat & rotate mul add
      for (int64_t i = 0; i < half_rotary_dim; ++i) {
        int64_t j = i + half_rotary_dim;
        T cv = cos_cache_offset[i];
        T sv = sin_cache_offset[i];
        res[i] = pe[i] * cv - pe[j] * sv;
        res[j] = pe[j] * cv + pe[i] * sv;
      }
    } else {
      // interleave & rotate mul add, unfortunately no prefetch in sycl
      const v2_type* pe_2 = reinterpret_cast<const v2_type*>(pe);
      v2_type* res_2 = reinterpret_cast<v2_type*>(res);
      for (int64_t h = 0; h < half_rotary_dim; ++h) {
        T c = cos_cache_offset[h];
        T s = sin_cache_offset[h];
        v2_type c2 = {c, c};
        v2_type s2 = {s, s};
        v2_type t = pe_2[h];
        v2_type* dst = &res_2[h];
        v2_type tr = {-t[1], t[0]};
        *dst = t * c2 + tr * s2;
      }
    }
  }

  [[sycl::reqd_sub_group_size(sg_size)]] void operator()(
      sycl::nd_item<3> idx) const {
    int64_t batch_idx = idx.get_global_id(0);
    int64_t sg_idx = idx.get_local_id(1);
    int64_t local_id = idx.get_global_id(2);
    int64_t head_idx = sg_idx * sg_size + local_id;
    int64_t qo_idx = batch_idx * q_num_head * head_size + head_idx * head_size;
    int64_t ko_idx = batch_idx * k_num_head * head_size +
        (head_idx - q_num_head) * head_size;
    int64_t qi_idx = batch_idx * q_batch_d + head_idx * q_num_head_d;
    int64_t ki_idx =
        batch_idx * k_batch_d + (head_idx - q_num_head) * k_num_head_d;
    if (head_idx < q_num_head) {
      rotary_emb_kern(
          positions[batch_idx],
          &query[qi_idx],
          cos_sin_cache,
          &query_out[qo_idx]);
    } else if (head_idx < q_num_head + k_num_head) {
      rotary_emb_kern(
          positions[batch_idx], &key[ki_idx], cos_sin_cache, &key_out[ko_idx]);
    }
  }

  const int64_t* positions;
  const T* query;
  const T* key;
  const int64_t* offsets;
  const T* cos_sin_cache;
  T* query_out;
  T* key_out;
  const int64_t batch;
  const int64_t q_num_head;
  const int64_t k_num_head;
  const int64_t head_size;
  const int64_t q_num_head_d;
  const int64_t q_batch_d;
  const int64_t k_num_head_d;
  const int64_t k_batch_d;
};

template <typename T, int64_t rotary_dim, bool is_neox>
void launch_rotary_embedding(
    sycl::queue& Q,
    const int64_t* positions,
    const T* query,
    const T* key,
    const int64_t* offsets,
    const T* cos_sin_cache,
    T* query_out,
    T* key_out,
    const int64_t batch,
    const int64_t q_num_head,
    const int64_t k_num_head,
    const int64_t head_size,
    const int64_t q_num_head_d,
    const int64_t q_batch_d,
    const int64_t k_num_head_d,
    const int64_t k_batch_d) {
  using Kernel = FusedDSRotaryEmbeddingQK<T, rotary_dim, is_neox>;
  auto range = Kernel::get_nd_range(batch, q_num_head, k_num_head);
  auto cgf = DPCPP_Q_CGF(cgh) {
    Kernel task(
        positions,
        query,
        key,
        offsets,
        cos_sin_cache,
        query_out,
        key_out,
        batch,
        q_num_head,
        k_num_head,
        head_size,
        q_num_head_d,
        q_batch_d,
        k_num_head_d,
        k_batch_d);
    cgh.parallel_for(range, task);
  };
  DPCPP_Q_SUBMIT(Q, cgf);
}

template <typename T>
using LAUNCH_FUNC = void (*)(
    sycl::queue&,
    const int64_t*,
    const T*,
    const T*,
    const int64_t*,
    const T*,
    T*,
    T*,
    const int64_t,
    const int64_t,
    const int64_t,
    const int64_t,
    const int64_t,
    const int64_t,
    const int64_t,
    const int64_t);

#define DEFINE_DS_ROTEMB_FUNC(T, n, b) &launch_rotary_embedding<T, n, b>

template <typename T>
void launch_rotary_embedding(
    const int64_t* positions,
    const T* query,
    const T* key,
    const int64_t* offsets,
    const T* cos_sin_cache,
    T* query_out,
    T* key_out,
    const int64_t batch,
    const int64_t q_num_head,
    const int64_t k_num_head,
    const int64_t head_size,
    const int64_t rotary_dim,
    bool is_neox_style,
    const int64_t q_num_head_d,
    const int64_t q_batch_d,
    const int64_t k_num_head_d,
    const int64_t k_batch_d) {
  auto& queue = dpcppGetCurrentQueue();

  constexpr int dim_size = 5;
  constexpr std::array<int, dim_size> allowed_dim = {32, 64, 96, 128, 256};
  int rot_idx = -1;
  int neox_idx = is_neox_style ? 1 : 0;
  for (int i = 0; i < allowed_dim.size(); ++i) {
    if (allowed_dim[i] == rotary_dim) {
      rot_idx = i;
    }
  }
  TORCH_CHECK(
      rot_idx >= 0,
      "wrong values for rotary_dim (%ld) only support 32,64,96,128,256\n",
      rotary_dim);
  TORCH_CHECK(
      rotary_dim == head_size,
      "rotary_dim (%ld)should be equal to head_size (%ld)",
      rotary_dim,
      head_size);
  int funcIndex = neox_idx * allowed_dim.size() + rot_idx;
  constexpr int func_size = dim_size * 2;
  static constexpr std::array<LAUNCH_FUNC<T>, func_size> launch_funcs = {
      DEFINE_DS_ROTEMB_FUNC(T, 32, false),
      DEFINE_DS_ROTEMB_FUNC(T, 64, false),
      DEFINE_DS_ROTEMB_FUNC(T, 96, false),
      DEFINE_DS_ROTEMB_FUNC(T, 128, false),
      DEFINE_DS_ROTEMB_FUNC(T, 256, false),
      DEFINE_DS_ROTEMB_FUNC(T, 32, true),
      DEFINE_DS_ROTEMB_FUNC(T, 64, true),
      DEFINE_DS_ROTEMB_FUNC(T, 96, true),
      DEFINE_DS_ROTEMB_FUNC(T, 128, true),
      DEFINE_DS_ROTEMB_FUNC(T, 256, true),
  };
  launch_funcs[funcIndex](
      queue,
      positions,
      query,
      key,
      offsets,
      cos_sin_cache,
      query_out,
      key_out,
      batch,
      q_num_head,
      k_num_head,
      head_size,
      q_num_head_d,
      q_batch_d,
      k_num_head_d,
      k_batch_d);
}
} // namespace DSRotaryEmbedding

void rotary_embedding_batched(
    const Tensor& positions, //[batch_size, seqlen] or [num_tokens]
    const Tensor& query, // [(bs, seq)/num_tokens, num_head * head_dim]
    const Tensor& key, // [(bs, seq)/num_tokens, num_kv_head * head_dim]
    int64_t head_size,
    const Tensor& cos_sin_cache, // [max_position, rot_dim]
    bool is_neox,
    int64_t rot_dim,
    Tensor& cos_sin_cache_offsets // [num_tokens]
) {
  int64_t num_tokens = positions.view(-1).size(0);
  int64_t num_heads = query.size(-1) / head_size;
  int64_t num_kv_heads = key.size(-1) / head_size;
  int64_t query_stride = query.stride(-2);
  int64_t key_stride = key.stride(-2);
  auto queue = dpcppGetCurrentQueue();
  auto dev_id = dpcppGetDeviceIdOfCurrentQueue();
  int64_t max_wg_size = dpcppMaxWorkGroupSize(dev_id);
  int64_t max_group_num = dpcppMaxWorkItemsPerTile(dev_id) / max_wg_size;
  int64_t num_groups = num_tokens;
  int64_t group_size = std::min(num_heads * rot_dim / 2, max_wg_size);

  IPEX_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      query.scalar_type(),
      "rotary_embedding_batched",
      [=]() {
        auto cgf = DPCPP_Q_CGF(cgh) {
          if (is_neox) {
            RotaryEmbeddingBatched<scalar_t, EmbeddingAlgorithm::RotateHalf>
                kernel = {
                    .positions_ = positions.data_ptr<int64_t>(),
                    .cos_sin_cache_ = cos_sin_cache.data_ptr<scalar_t>(),
                    .cos_sin_cache_offsets_ =
                        cos_sin_cache_offsets.data_ptr<int64_t>(),
                    .query_ = query.data_ptr<scalar_t>(),
                    .key_ = key.data_ptr<scalar_t>(),
                    .num_heads_ = num_heads,
                    .num_kv_heads_ = num_kv_heads,
                    .query_stride_ = query_stride,
                    .key_stride_ = key_stride,
                    .head_size_ = head_size,
                    .rot_dim_ = rot_dim,
                };
            cgh.parallel_for<decltype(kernel)>(
                sycl::nd_range<1>(
                    sycl::range<1>(num_groups * group_size),
                    sycl::range<1>(group_size)),
                kernel);
          } else {
            RotaryEmbeddingBatched<
                scalar_t,
                EmbeddingAlgorithm::RotateInterleave>
                kernel = {
                    .positions_ = positions.data_ptr<int64_t>(),
                    .cos_sin_cache_ = cos_sin_cache.data_ptr<scalar_t>(),
                    .cos_sin_cache_offsets_ =
                        cos_sin_cache_offsets.data_ptr<int64_t>(),
                    .query_ = query.data_ptr<scalar_t>(),
                    .key_ = key.data_ptr<scalar_t>(),
                    .num_heads_ = num_heads,
                    .num_kv_heads_ = num_kv_heads,
                    .query_stride_ = query_stride,
                    .key_stride_ = key_stride,
                    .head_size_ = head_size,
                    .rot_dim_ = rot_dim,
                };
            cgh.parallel_for<decltype(kernel)>(
                sycl::nd_range<1>(
                    sycl::range<1>(num_groups * group_size),
                    sycl::range<1>(group_size)),
                kernel);
          }
        };
        DPCPP_Q_SUBMIT(dpcppGetCurrentQueue(), cgf);
      });
}

void rotary_embedding_legacy(
    const Tensor& positions, //[batch_size, seqlen] or [num_tokens]
    const Tensor& query, // [(bs, seq)/num_tokens, num_head * head_dim]
    const Tensor& key, // [(bs, seq)/num_tokens, num_kv_head * head_dim]
    int64_t head_size,
    const Tensor& cos_sin_cache, // [max_position, rot_dim]
    bool is_neox,
    int64_t rot_dim) {
  int64_t num_tokens = positions.view(-1).size(0);
  int64_t num_heads = query.size(-1) / head_size;
  int64_t num_kv_heads = key.size(-1) / head_size;
  int64_t query_stride = query.stride(-2);
  int64_t key_stride = key.stride(-2);
  auto queue = dpcppGetCurrentQueue();
  auto dev_id = dpcppGetDeviceIdOfCurrentQueue();
  int64_t max_wg_size = dpcppMaxWorkGroupSize(dev_id);
  int64_t max_group_num = dpcppMaxWorkItemsPerTile(dev_id) / max_wg_size;
  int64_t num_groups = num_tokens;
  int64_t group_size = std::min(max_wg_size, query.size(-1));

  IPEX_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      query.scalar_type(),
      "rotary_embedding_legacy",
      [=]() {
        auto cgf = DPCPP_Q_CGF(cgh) {
          if (is_neox) {
            RotaryEmbeddingBatched<
                scalar_t,
                EmbeddingAlgorithm::RotateHalf,
                false>
                kernel = {
                    .positions_ = positions.data_ptr<int64_t>(),
                    .cos_sin_cache_ = cos_sin_cache.data_ptr<scalar_t>(),
                    .cos_sin_cache_offsets_ = nullptr,
                    .query_ = query.data_ptr<scalar_t>(),
                    .key_ = key.data_ptr<scalar_t>(),
                    .num_heads_ = num_heads,
                    .num_kv_heads_ = num_kv_heads,
                    .query_stride_ = query_stride,
                    .key_stride_ = key_stride,
                    .head_size_ = head_size,
                    .rot_dim_ = rot_dim,
                };
            cgh.parallel_for<decltype(kernel)>(
                sycl::nd_range<1>(
                    sycl::range<1>(num_groups * group_size),
                    sycl::range<1>(group_size)),
                kernel);
          } else {
            RotaryEmbeddingBatched<
                scalar_t,
                EmbeddingAlgorithm::RotateInterleave,
                false>
                kernel = {
                    .positions_ = positions.data_ptr<int64_t>(),
                    .cos_sin_cache_ = cos_sin_cache.data_ptr<scalar_t>(),
                    .cos_sin_cache_offsets_ = nullptr,
                    .query_ = query.data_ptr<scalar_t>(),
                    .key_ = key.data_ptr<scalar_t>(),
                    .num_heads_ = num_heads,
                    .num_kv_heads_ = num_kv_heads,
                    .query_stride_ = query_stride,
                    .key_stride_ = key_stride,
                    .head_size_ = head_size,
                    .rot_dim_ = rot_dim,
                };
            cgh.parallel_for<decltype(kernel)>(
                sycl::nd_range<1>(
                    sycl::range<1>(num_groups * group_size),
                    sycl::range<1>(group_size)),
                kernel);
          }
        };
        DPCPP_Q_SUBMIT(dpcppGetCurrentQueue(), cgf);
      });
}

void apply_rotary_embedding_two(
    const Tensor& query,
    const Tensor& sin,
    const Tensor& cos,
    Tensor& query_out) {
  apply_rotary_embedding<EmbeddingAlgorithm::RotateInterleave>(
      sin, cos, query, query_out);
}

void apply_rotary_embedding_two_qk(
    const Tensor& query,
    const Tensor& key,
    const Tensor& sin,
    const Tensor& cos,
    Tensor& query_out,
    Tensor& key_out) {
  apply_rotary_embedding<EmbeddingAlgorithm::RotateInterleave>(
      sin, cos, query, key, query_out, key_out);
}

void apply_rotary_embedding_half(
    const Tensor& query,
    const Tensor& sin,
    const Tensor& cos,
    Tensor& query_out) {
  apply_rotary_embedding<EmbeddingAlgorithm::RotateHalf>(
      sin, cos, query, query_out);
}

void apply_rotary_embedding_half_qk(
    const Tensor& query,
    const Tensor& key,
    const Tensor& sin,
    const Tensor& cos,
    Tensor& query_out,
    Tensor& key_out) {
  apply_rotary_embedding<EmbeddingAlgorithm::RotateHalf>(
      sin, cos, query, key, query_out, key_out);
}

/**
 * @brief Perform deepseek rotary embedding with q&k.
 * @param positions index of embedding [batch]
 * @param query query to be processed [batch, num_head, head_dim]
 * @param key key to be processed [batch, num_head, head_dim]
 * @param offsets optional tensor for offset with position
 * @param cos_sin_cache shared cache with cos/sin
 * @param is_neox_style choose interleave or half.
 * @return A tuple of tensors (query_out, key_out).
 */
std::tuple<at::Tensor, at::Tensor> ds_rotary_embedding_qk(
    const Tensor& positions,
    const Tensor& query,
    const Tensor& key,
    const c10::optional<at::Tensor>& offsets_opt,
    const Tensor& cos_sin_cache,
    int64_t rotary_dim,
    bool is_neox_style) {
  auto query_out = at::empty_like(query);
  auto key_out = at::empty_like(key);

  auto q_shape = query.sizes();
  auto q_stride = query.strides();
  int64_t head_size = q_shape[2];
  int64_t q_num_head = q_shape[1];
  int64_t batch = q_shape[0];
  int64_t q_num_head_d = q_stride[1];
  int64_t q_batch_d = q_stride[0];
  auto k_shape = key.sizes();
  auto k_stride = key.strides();
  int64_t k_num_head = k_shape[1];
  int64_t k_num_head_d = k_stride[1];
  int64_t k_batch_d = k_stride[0];
  if (is_neox_style) {
    query_out = query_out.reshape({1, batch, q_num_head, head_size});
    key_out = key_out.reshape({1, batch, k_num_head, head_size});
  }
  TORCH_CHECK(
      cos_sin_cache.sizes()[1] == head_size,
      "Rotary dim doesn't match query head_size");
  TORCH_CHECK(
      cos_sin_cache.sizes()[1] == k_shape[2],
      "Rotary dim doesn't match key head_size");
  const c10::MaybeOwned<Tensor> offsets_maybe_owned =
      at::borrow_from_optional_tensor(offsets_opt);
  const Tensor& offsets = *offsets_maybe_owned;
  auto offsets_ptr = offsets.defined() ? offsets.data_ptr() : nullptr;
  if (query.scalar_type() == at::kBFloat16) {
    using scalar_t = sycl::ext::oneapi::bfloat16;
    DSRotaryEmbedding::launch_rotary_embedding<scalar_t>(
        reinterpret_cast<int64_t*>(positions.data_ptr()),
        reinterpret_cast<scalar_t*>(query.data_ptr()),
        reinterpret_cast<scalar_t*>(key.data_ptr()),
        reinterpret_cast<int64_t*>(offsets_ptr),
        reinterpret_cast<scalar_t*>(cos_sin_cache.data_ptr()),
        reinterpret_cast<scalar_t*>(query_out.data_ptr()),
        reinterpret_cast<scalar_t*>(key_out.data_ptr()),
        batch,
        q_num_head,
        k_num_head,
        head_size,
        rotary_dim,
        is_neox_style,
        q_num_head_d,
        q_batch_d,
        k_num_head_d,
        k_batch_d);
  } else if (query.scalar_type() == at::kHalf) {
    using scalar_t = sycl::half;
    DSRotaryEmbedding::launch_rotary_embedding<scalar_t>(
        reinterpret_cast<int64_t*>(positions.data_ptr()),
        reinterpret_cast<scalar_t*>(query.data_ptr()),
        reinterpret_cast<scalar_t*>(key.data_ptr()),
        reinterpret_cast<int64_t*>(offsets_ptr),
        reinterpret_cast<scalar_t*>(cos_sin_cache.data_ptr()),
        reinterpret_cast<scalar_t*>(query_out.data_ptr()),
        reinterpret_cast<scalar_t*>(key_out.data_ptr()),
        batch,
        q_num_head,
        k_num_head,
        head_size,
        rotary_dim,
        is_neox_style,
        q_num_head_d,
        q_batch_d,
        k_num_head_d,
        k_batch_d);
  } else {
    IPEX_DISPATCH_FLOATING_TYPES(
        query.scalar_type(), "ds_rotary_embedding_qk", [&]() {
          DSRotaryEmbedding::launch_rotary_embedding<scalar_t>(
              reinterpret_cast<int64_t*>(positions.data_ptr()),
              reinterpret_cast<scalar_t*>(query.data_ptr()),
              reinterpret_cast<scalar_t*>(key.data_ptr()),
              reinterpret_cast<int64_t*>(offsets_ptr),
              reinterpret_cast<scalar_t*>(cos_sin_cache.data_ptr()),
              reinterpret_cast<scalar_t*>(query_out.data_ptr()),
              reinterpret_cast<scalar_t*>(key_out.data_ptr()),
              batch,
              q_num_head,
              k_num_head,
              head_size,
              rotary_dim,
              is_neox_style,
              q_num_head_d,
              q_batch_d,
              k_num_head_d,
              k_batch_d);
        });
  }
  return {query_out, key_out};
}

namespace {
IPEX_LIBRARY_FRAGMENT() {
  IPEX_OP_REGISTER_DISPATCH(
      "apply_rotary_embedding_two_qk",
      apply_rotary_embedding_two_qk,
      c10::DispatchKey::XPU);

  IPEX_OP_REGISTER_DISPATCH(
      "apply_rotary_embedding_two",
      apply_rotary_embedding_two,
      c10::DispatchKey::XPU);

  IPEX_OP_REGISTER_DISPATCH(
      "apply_rotary_embedding_half",
      apply_rotary_embedding_half,
      c10::DispatchKey::XPU);

  IPEX_OP_REGISTER_DISPATCH(
      "apply_rotary_embedding_half_qk",
      apply_rotary_embedding_half_qk,
      c10::DispatchKey::XPU);

  IPEX_OP_REGISTER_DISPATCH(
      "rotary_embedding_batched",
      rotary_embedding_batched,
      c10::DispatchKey::XPU);

  IPEX_OP_REGISTER_DISPATCH(
      "rotary_embedding_legacy",
      rotary_embedding_legacy,
      c10::DispatchKey::XPU);

  IPEX_OP_REGISTER_DISPATCH(
      "ds_rotary_embedding_qk", ds_rotary_embedding_qk, c10::DispatchKey::XPU);
}
} // namespace

} // namespace AtenIpexTypeXPU
} // namespace at
