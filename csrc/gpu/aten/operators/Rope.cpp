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
#include "comm/RegistrationDeclarations.h"

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
          scalar_t shift_val = sg.shuffle_xor(val, 1) * scale;
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

void rotary_embedding(
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
      "rotary_embedding",
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
      "rotary_embedding", rotary_embedding, c10::DispatchKey::XPU);
}
} // namespace

} // namespace AtenIpexTypeXPU
} // namespace at
