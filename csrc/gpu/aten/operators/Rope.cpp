#include <ATen/ATen.h>
#include <ATen/core/Array.h>

#include <ATen/OpMathType.h>
#include <core/MemoryFormat.h>
#include <core/detail/IndexUtils.h>
#include <runtime/Utils.h>
#include <utils/DPCPP.h>
#include <cstdint>

#include "comm/ATDispatch.h"

#include "utils/CustomOperatorRegistration.h"

namespace at {
namespace AtenIpexTypeXPU {

namespace vllm {

template <typename scalar_t, bool IS_NEOX>
inline void apply_token_rotary_embedding(
    scalar_t* __restrict__ arr,
    const scalar_t* __restrict__ cos_ptr,
    const scalar_t* __restrict__ sin_ptr,
    int rot_offset,
    int embed_dim) {
  int x_index, y_index;
  scalar_t cos, sin;
  if constexpr (IS_NEOX) {
    // GPT-NeoX style rotary embedding.
    x_index = rot_offset;
    y_index = embed_dim + rot_offset;
    cos = cos_ptr[x_index];
    sin = sin_ptr[x_index];
  } else {
    // GPT-J style rotary embedding.
    x_index = 2 * rot_offset;
    y_index = 2 * rot_offset + 1;
    cos = cos_ptr[x_index / 2];
    sin = sin_ptr[x_index / 2];
  }

  const scalar_t x = arr[x_index];
  const scalar_t y = arr[y_index];
  arr[x_index] = x * cos - y * sin;
  arr[y_index] = y * cos + x * sin;
}

template <typename scalar_t, bool IS_NEOX>
inline void apply_rotary_embedding(
    scalar_t* __restrict__ query, // [batch_size, seq_len, num_heads,
                                  // head_size] or [num_tokens, num_heads,
                                  // head_size]
    scalar_t* __restrict__ key, // nullptr or
                                // [batch_size, seq_len, num_kv_heads,
                                // head_size] or [num_tokens, num_kv_heads,
                                // head_size]
    const scalar_t* cache_ptr,
    const int head_size,
    const int num_heads,
    const int num_kv_heads,
    const int rot_dim,
    const int token_idx,
    const int64_t query_stride,
    const int64_t key_stride,
    const int64_t head_stride,
    const sycl::nd_item<3>& item_ct1) {
  const int embed_dim = rot_dim / 2;
  const scalar_t* cos_ptr = cache_ptr;
  const scalar_t* sin_ptr = cache_ptr + embed_dim;

  const int nq = num_heads * embed_dim;
  for (int i = item_ct1.get_local_id(2); i < nq;
       i += item_ct1.get_local_range(2)) {
    const int head_idx = i / embed_dim;
    const int64_t token_head =
        token_idx * query_stride + head_idx * head_stride;
    const int rot_offset = i % embed_dim;
    apply_token_rotary_embedding<scalar_t, IS_NEOX>(
        query + token_head, cos_ptr, sin_ptr, rot_offset, embed_dim);
  }

  if (key != nullptr) {
    const int nk = num_kv_heads * embed_dim;
    for (int i = item_ct1.get_local_id(2); i < nk;
         i += item_ct1.get_local_range(2)) {
      const int head_idx = i / embed_dim;
      const int64_t token_head =
          token_idx * key_stride + head_idx * head_stride;
      const int rot_offset = i % embed_dim;
      apply_token_rotary_embedding<scalar_t, IS_NEOX>(
          key + token_head, cos_ptr, sin_ptr, rot_offset, embed_dim);
    }
  }
}

template <typename scalar_t, bool IS_NEOX>
class rotary_embedding_kernel {
 public:
  rotary_embedding_kernel(
      const int64_t* __restrict__ positions_, // [batch_size, seq_len] or
                                              // [num_tokens]
      scalar_t* __restrict__ query_, // [batch_size, seq_len, num_heads,
                                     // head_size] or [num_tokens, num_heads,
                                     // head_size]
      scalar_t* __restrict__ key_, // nullptr or
                                   // [batch_size, seq_len, num_kv_heads,
      // head_size] or [num_tokens, num_kv_heads,
      // head_size]
      const scalar_t* __restrict__ cos_sin_cache_, // [max_position, 2, rot_dim
                                                   // // 2]
      const int rot_dim_,
      const int64_t query_stride_,
      const int64_t key_stride_,
      const int64_t head_stride_,
      const int num_heads_,
      const int num_kv_heads_,
      const int head_size_)
      : positions(positions_),
        query(query_),
        key(key_),
        cos_sin_cache(cos_sin_cache_),
        rot_dim(rot_dim_),
        query_stride(query_stride_),
        key_stride(key_stride_),
        head_stride(head_stride_),
        num_heads(num_heads_),
        num_kv_heads(num_kv_heads_),
        head_size(head_size_) {}

  void operator() [[sycl::reqd_sub_group_size(32)]] (
      const sycl::nd_item<3>& item_ct1) const {
    // Each thread block is responsible for one token.
    const int token_idx = item_ct1.get_group(2);
    int64_t pos = positions[token_idx];
    const scalar_t* cache_ptr = cos_sin_cache + pos * rot_dim;

    apply_rotary_embedding<scalar_t, IS_NEOX>(
        query,
        key,
        cache_ptr,
        head_size,
        num_heads,
        num_kv_heads,
        rot_dim,
        token_idx,
        query_stride,
        key_stride,
        head_stride,
        item_ct1);
  }

 private:
  const int64_t* __restrict__ positions; // [batch_size, seq_len] or
                                         // [num_tokens]
  scalar_t* __restrict__ query; // [batch_size, seq_len, num_heads,
                                // head_size] or [num_tokens, num_heads,
                                // head_size]
  scalar_t* __restrict__ key; // nullptr or
                              // [batch_size, seq_len, num_kv_heads,
                              // head_size] or [num_tokens, num_kv_heads,
                              // head_size]
  const scalar_t* __restrict__ cos_sin_cache; // [max_position, 2, rot_dim //
                                              // 2]
  const int rot_dim;
  const int64_t query_stride;
  const int64_t key_stride;
  const int64_t head_stride;
  const int num_heads;
  const int num_kv_heads;
  const int head_size;
};

} // namespace vllm

template <typename scalar_t>
void call_rotary_embedding_kernel(
    at::Tensor& positions,
    at::Tensor& query,
    std::optional<at::Tensor> key,
    int64_t head_size,
    at::Tensor& cos_sin_cache, // [max_position, rot_dim]
    bool is_neox) {
  // num_tokens = batch_size * seq_len
  int64_t num_tokens = positions.numel();
  int positions_ndim = positions.dim();

  // Make sure num_tokens dim is consistent across positions, query, and key
  TORCH_CHECK(
      positions_ndim == 1 || positions_ndim == 2,
      "positions must have shape [num_tokens] or [batch_size, seq_len]");
  if (positions_ndim == 1) {
    TORCH_CHECK(
        query.size(0) == positions.size(0) &&
            (!key.has_value() || key->size(0) == positions.size(0)),
        "query, key and positions must have the same number of tokens");
  }
  if (positions_ndim == 2) {
    TORCH_CHECK(
        query.size(0) == positions.size(0) &&
            (!key.has_value() || key->size(0) == positions.size(0)) &&
            query.size(1) == positions.size(1) &&
            (!key.has_value() || key->size(1) == positions.size(1)),
        "query, key and positions must have the same batch_size and seq_len");
  }

  // Make sure head_size is valid for query and key
  // hidden_size = num_heads * head_size
  int query_hidden_size = query.numel() / num_tokens;
  int key_hidden_size = key.has_value() ? key->numel() / num_tokens : 0;
  TORCH_CHECK(query_hidden_size % head_size == 0);
  TORCH_CHECK(key_hidden_size % head_size == 0);

  // Make sure query and key have consistent number of heads
  int num_heads = query_hidden_size / head_size;
  int num_kv_heads = key.has_value() ? key_hidden_size / head_size : num_heads;
  TORCH_CHECK(num_heads % num_kv_heads == 0);

  int rot_dim = cos_sin_cache.size(1);
  int seq_dim_idx = positions_ndim - 1;
  int64_t query_stride = query.stride(seq_dim_idx);
  int64_t key_stride = key.has_value() ? key->stride(seq_dim_idx) : 0;
  // Determine head stride: for [*, heads, head_size] use stride of last dim;
  // for flat [*, heads*head_size], heads blocks are contiguous of size
  // head_size
  int query_ndim = query.dim();
  int64_t head_stride =
      (query_ndim == positions_ndim + 2) ? query.stride(-2) : head_size;

  sycl::range<3> grid(1, 1, num_tokens);
  sycl::range<3> block(1, 1, std::min<int64_t>(num_heads * rot_dim / 2, 512));

  at::DeviceGuard device_guard(query.device());
  auto& queue = torch_ipex::xpu::dpcpp::dpcppGetCurrentQueue();
  if (is_neox) {
    queue.submit([&](sycl::handler& cgh) {
      cgh.parallel_for(
          sycl::nd_range<3>(grid * block, block),
          vllm::rotary_embedding_kernel<scalar_t, true>(
              reinterpret_cast<int64_t*>(positions.data_ptr()),
              reinterpret_cast<scalar_t*>(query.data_ptr()),
              key.has_value() ? reinterpret_cast<scalar_t*>(key->data_ptr())
                              : static_cast<scalar_t*>(nullptr),
              reinterpret_cast<scalar_t*>(cos_sin_cache.data_ptr()),
              rot_dim,
              query_stride,
              key_stride,
              head_stride,
              num_heads,
              num_kv_heads,
              head_size));
    });
  } else {
    queue.submit([&](sycl::handler& cgh) {
      cgh.parallel_for(
          sycl::nd_range<3>(grid * block, block),
          vllm::rotary_embedding_kernel<scalar_t, false>(
              reinterpret_cast<int64_t*>(positions.data_ptr()),
              reinterpret_cast<scalar_t*>(query.data_ptr()),
              key.has_value() ? reinterpret_cast<scalar_t*>(key->data_ptr())
                              : static_cast<scalar_t*>(nullptr),
              reinterpret_cast<scalar_t*>(cos_sin_cache.data_ptr()),
              rot_dim,
              query_stride,
              key_stride,
              head_stride,
              num_heads,
              num_kv_heads,
              head_size));
    });
  }
}

void rotary_embedding(
    at::Tensor& positions, // [batch_size, seq_len] or [num_tokens]
    at::Tensor& query, // [batch_size, seq_len, num_heads * head_size] or
                       // [num_tokens, num_heads * head_size] or
                       // [batch_size, seq_len, num_heads, head_size] or
                       // [num_tokens, num_heads, head_size]
    std::optional<at::Tensor> key,
    // null or
    // [batch_size, seq_len, num_kv_heads * head_size] or
    // [num_tokens, num_kv_heads * head_size] or
    // [batch_size, seq_len, num_heads, head_size] or
    // [num_tokens, num_heads, head_size]
    int64_t head_size,
    at::Tensor& cos_sin_cache, // [max_position, rot_dim]
    bool is_neox) {
  if (query.scalar_type() == at::kBFloat16) {
    using scalar_t = sycl::ext::oneapi::bfloat16;
    call_rotary_embedding_kernel<scalar_t>(
        positions, query, key, head_size, cos_sin_cache, is_neox);
  } else if (query.scalar_type() == at::kHalf) {
    using scalar_t = sycl::half;
    call_rotary_embedding_kernel<scalar_t>(
        positions, query, key, head_size, cos_sin_cache, is_neox);
  } else if (query.scalar_type() == at::kFloat) {
    call_rotary_embedding_kernel<float>(
        positions, query, key, head_size, cos_sin_cache, is_neox);
  } else {
    TORCH_CHECK(false, "rotary_embedding only support bf16, fp16, float input.")
  }
}

namespace {
IPEX_LIBRARY_FRAGMENT() {
  IPEX_OP_REGISTER_DISPATCH(
      "rotary_embedding", rotary_embedding, c10::DispatchKey::XPU);
}
} // namespace

} // namespace AtenIpexTypeXPU
} // namespace at
