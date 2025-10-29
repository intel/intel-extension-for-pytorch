#include <ATen/ATen.h>
#include <ATen/record_function.h>
#include <oneDNN/oneDNN.h>
#include <runtime/Utils.h>
#include <sycl/sycl.hpp>
#include <utils/oneMKLUtils.h>
#include "comm/ATDispatch.h"
#include "utils/ComputeEngine.h"

namespace at {
namespace AtenIpexTypeXPU {

#define CEILDIV(x, y) (((x) + (y)-1) / (y))
// Round a up to the next multiple of b. The caller is responsible for making
// sure that b is non-zero
template <typename T>
inline constexpr T round_to_next_multiple_of(T a, T b) {
  return a % b == 0 ? a : ((a / b) + 1) * b;
}

namespace batched_moe_align_block_size_impl {
// Note num_threads needs to be 1024 for BlockScan Reduction in the kernel.
static constexpr int32_t num_threads = 1024;
static constexpr int32_t num_blocks = 1;

class batched_moe_align_block_size_kernel {
 private:
  sycl::local_accessor<int32_t, 1> slm;
  int32_t const num_batches;
  int32_t const max_tokens_per_batch;
  int32_t const block_size;
  int32_t const* __restrict__ batch_num_tokens;
  int32_t* __restrict__ sorted_ids;
  int32_t* __restrict__ block_ids;
  int32_t* __restrict__ num_tokens_post_pad;

 public:
  batched_moe_align_block_size_kernel(
      sycl::local_accessor<int32_t, 1>& slm,
      int32_t const num_batches,
      int32_t const max_tokens_per_batch,
      int32_t const block_size,
      int32_t const* __restrict__ batch_num_tokens,
      int32_t* __restrict__ sorted_ids,
      int32_t* __restrict__ block_ids,
      int32_t* __restrict__ num_tokens_post_pad)
      : slm(slm),
        num_batches(num_batches),
        max_tokens_per_batch(max_tokens_per_batch),
        block_size(block_size),
        batch_num_tokens(batch_num_tokens),
        sorted_ids(sorted_ids),
        block_ids(block_ids),
        num_tokens_post_pad(num_tokens_post_pad) {}

  void operator()(sycl::nd_item<1> item) const {
    // TODO: This is a naive implementation. Could be optimized.
    auto group = item.get_group();
    auto local_id_x = item.get_local_id(0);
    auto local_range = item.get_local_range(0);
    auto group_range = item.get_group_range(0);

    int32_t* temp_storage = static_cast<int32_t*>(
        slm.template get_multi_ptr<sycl::access::decorated::no>().get());

    size_t const batch_id = local_id_x;
    size_t const stride = local_range * group_range;
    int32_t const num_blocks_per_batch =
        CEILDIV(max_tokens_per_batch, block_size);
    int32_t const sorted_ids_size =
        num_blocks_per_batch * num_batches * block_size;
    int32_t const block_ids_size = sorted_ids_size / block_size;
    int32_t const SENTINEL =
        num_batches * max_tokens_per_batch; // To denote invalid entries.
    // Initialize sorted_ids
    for (size_t i = local_id_x; i < sorted_ids_size; i += stride) {
      sorted_ids[i] = SENTINEL;
    }
    // Initialize expert_ids with -1
    for (size_t i = local_id_x; i < block_ids_size; i += stride) {
      block_ids[i] = -1;
    }

    int32_t b_num_tokens = 0;
    if (batch_id < num_batches) {
      b_num_tokens = batch_num_tokens[batch_id];
    }
    int32_t const ceil_b_num_tokens =
        CEILDIV(b_num_tokens, block_size) * block_size;

    // Compute prefix sum over token counts per expert
    temp_storage[local_id_x] = ceil_b_num_tokens;
    item.barrier(sycl::access::fence_space::local_space);

    int cumsum_val;
    sycl::joint_exclusive_scan(
        item.get_group(),
        temp_storage,
        temp_storage + 1024,
        temp_storage,
        0,
        sycl::plus<int>{});
    cumsum_val = temp_storage[local_id_x];

    bool const is_last_batch = batch_id == (num_batches - 1);
    if (is_last_batch) {
      *num_tokens_post_pad = cumsum_val + ceil_b_num_tokens;
    }

    if (batch_id < num_batches) {
      int32_t const batch_offset = batch_id * max_tokens_per_batch;
      for (size_t i = 0; i < b_num_tokens; ++i) {
        sorted_ids[cumsum_val + i] = batch_offset + i;
      }

      int32_t const block_start = cumsum_val / block_size;
      int32_t const num_blocks = ceil_b_num_tokens / block_size;
      for (size_t i = 0; i < num_blocks; ++i) {
        block_ids[block_start + i] = batch_id;
      }
    }
  }
};
} // namespace batched_moe_align_block_size_impl

namespace moe_align_block_size_impl {
static constexpr int WARP_SIZE = 32;
template <typename scalar_t>
class moe_align_block_size_kernel {
 private:
  sycl::local_accessor<int32_t, 1> slm;
  const scalar_t* __restrict__ topk_ids;
  int32_t* __restrict__ sorted_token_ids;
  int32_t* __restrict__ expert_ids;
  int32_t* __restrict__ total_tokens_post_pad;
  int32_t num_experts;
  int32_t padded_num_experts;
  int32_t experts_per_warp;
  int32_t block_size;
  size_t numel;
  int32_t* __restrict__ cumsum;
  int32_t max_num_tokens_padded;

 public:
  moe_align_block_size_kernel(
      sycl::local_accessor<int32_t, 1>& slm,
      const scalar_t* __restrict__ topk_ids,
      int32_t* __restrict__ sorted_token_ids,
      int32_t* __restrict__ expert_ids,
      int32_t* __restrict__ total_tokens_post_pad,
      int32_t num_experts,
      int32_t padded_num_experts,
      int32_t experts_per_warp,
      int32_t block_size,
      size_t numel,
      int32_t* __restrict__ cumsum,
      int32_t max_num_tokens_padded)
      : slm(slm),
        topk_ids(topk_ids),
        sorted_token_ids(sorted_token_ids),
        expert_ids(expert_ids),
        total_tokens_post_pad(total_tokens_post_pad),
        num_experts(num_experts),
        padded_num_experts(padded_num_experts),
        experts_per_warp(experts_per_warp),
        block_size(block_size),
        numel(numel),
        cumsum(cumsum),
        max_num_tokens_padded(max_num_tokens_padded) {}

  void operator()(sycl::nd_item<1> item) const {
    auto group = item.get_group();
    auto local_id_x = item.get_local_id(0);
    auto local_range = item.get_local_range(0);

    int32_t* temp_storage = static_cast<int32_t*>(
        slm.template get_multi_ptr<sycl::access::decorated::no>().get());

    int32_t* shared_counts = temp_storage + 1024;

    // Initialize sorted_token_ids with numel
    for (size_t it = local_id_x; it < max_num_tokens_padded;
         it += local_range) {
      sorted_token_ids[it] = numel;
    }

    const int warp_id = local_id_x / WARP_SIZE;
    const int my_expert_start = warp_id * experts_per_warp;

    for (int i = 0; i < experts_per_warp; ++i) {
      if (my_expert_start + i < padded_num_experts) {
        shared_counts[warp_id * experts_per_warp + i] = 0;
      }
    }

    item.barrier(sycl::access::fence_space::local_space);

    const size_t tid = local_id_x;
    const size_t stride = local_range;

    for (size_t i = tid; i < numel; i += stride) {
      int expert_id = topk_ids[i];
      if (expert_id >= num_experts) {
        continue;
      }
      int warp_idx = expert_id / experts_per_warp;
      int expert_offset = expert_id % experts_per_warp;
      int idx = warp_idx * experts_per_warp + expert_offset;
      sycl::atomic_ref<
          int,
          sycl::memory_order::relaxed,
          sycl::memory_scope::device,
          sycl::access::address_space::local_space>
          atomic_count(shared_counts[idx]);
      atomic_count.fetch_add(1);
    }

    item.barrier(sycl::access::fence_space::local_space);

    // Compute prefix sum over token counts per expert
    int expert_count = 0;
    int expert_id = local_id_x;
    if (expert_id < num_experts) {
      int warp_idx = expert_id / experts_per_warp;
      int expert_offset = expert_id % experts_per_warp;
      expert_count = shared_counts[warp_idx * experts_per_warp + expert_offset];
      expert_count = CEILDIV(expert_count, block_size) * block_size;
    }

    temp_storage[local_id_x] = expert_count;
    item.barrier(sycl::access::fence_space::local_space);

    int cumsum_val;
    sycl::joint_exclusive_scan(
        item.get_group(),
        temp_storage,
        temp_storage + 1024,
        temp_storage,
        0,
        sycl::plus<int>{});
    cumsum_val = temp_storage[local_id_x];
    if (expert_id <= num_experts) {
      cumsum[expert_id] = cumsum_val;
    }

    if (expert_id == num_experts) {
      *total_tokens_post_pad = cumsum_val;
    }

    item.barrier(sycl::access::fence_space::local_space);

    if (local_id_x < num_experts) {
      for (int i = cumsum[local_id_x]; i < cumsum[local_id_x + 1];
           i += block_size) {
        expert_ids[i / block_size] = local_id_x;
      }
    }

    // Fill remaining expert_ids with 0
    const size_t fill_start_idx = cumsum[num_experts] / block_size + local_id_x;
    const size_t expert_ids_size = CEILDIV(max_num_tokens_padded, block_size);
    for (size_t i = fill_start_idx; i < expert_ids_size; i += local_range) {
      expert_ids[i] = 0;
    }
  }
};

template <typename scalar_t>
class count_and_sort_expert_tokens_kernel {
 private:
  const scalar_t* __restrict__ topk_ids;
  int32_t* __restrict__ sorted_token_ids;
  int32_t* __restrict__ cumsum_buffer;
  size_t numel;
  int32_t num_experts;

 public:
  count_and_sort_expert_tokens_kernel(
      const scalar_t* __restrict__ topk_ids,
      int32_t* __restrict__ sorted_token_ids,
      int32_t* __restrict__ cumsum_buffer,
      size_t numel,
      int32_t num_experts)
      : topk_ids(topk_ids),
        sorted_token_ids(sorted_token_ids),
        cumsum_buffer(cumsum_buffer),
        numel(numel),
        num_experts(num_experts) {}

  void operator()(sycl::nd_item<1> item) const {
    const size_t tid = item.get_global_linear_id();
    const size_t stride = item.get_global_range(0);

    for (size_t i = tid; i < numel; i += stride) {
      int32_t expert_id = topk_ids[i];
      if (expert_id >= num_experts) {
        continue;
      }

      auto atomic_count = sycl::atomic_ref<
          int,
          sycl::memory_order::relaxed,
          sycl::memory_scope::device,
          sycl::access::address_space::global_space>(
          *(cumsum_buffer + expert_id));
      int32_t rank_post_pad = atomic_count.fetch_add(1);

      sorted_token_ids[rank_post_pad] = i;
    }
  }
};

template <typename scalar_t>
class moe_align_block_size_small_batch_expert_kernel {
 private:
  sycl::local_accessor<int32_t, 1> slm;
  const scalar_t* __restrict__ topk_ids;
  int32_t* __restrict__ sorted_token_ids;
  int32_t* __restrict__ expert_ids;
  int32_t* __restrict__ total_tokens_post_pad;
  int32_t num_experts;
  int32_t block_size;
  size_t numel;
  int32_t max_num_tokens_padded;

 public:
  moe_align_block_size_small_batch_expert_kernel(
      sycl::local_accessor<int32_t, 1>& slm,
      const scalar_t* __restrict__ topk_ids,
      int32_t* __restrict__ sorted_token_ids,
      int32_t* __restrict__ expert_ids,
      int32_t* __restrict__ total_tokens_post_pad,
      int32_t num_experts,
      int32_t block_size,
      size_t numel,
      int32_t max_num_tokens_padded)
      : slm(slm),
        topk_ids(topk_ids),
        sorted_token_ids(sorted_token_ids),
        expert_ids(expert_ids),
        total_tokens_post_pad(total_tokens_post_pad),
        num_experts(num_experts),
        block_size(block_size),
        numel(numel),
        max_num_tokens_padded(max_num_tokens_padded) {}
  void operator()(sycl::nd_item<1> item) const {
    auto group = item.get_group();
    auto local_id_x = item.get_local_id(0);
    auto local_range = item.get_local_range(0);

    // Initialize sorted_token_ids with numel
    for (size_t it = local_id_x; it < max_num_tokens_padded;
         it += local_range) {
      sorted_token_ids[it] = numel;
    }

    const size_t tid = local_id_x;
    const size_t stride = local_range;

    void* slm_ptr = static_cast<void*>(
        slm.template get_multi_ptr<sycl::access::decorated::no>().get());
    int32_t* cumsum = reinterpret_cast<int32_t*>(slm_ptr);
    int32_t* tokens_cnts = cumsum + num_experts + 1;

    for (int i = 0; i < num_experts; ++i) {
      tokens_cnts[(local_id_x + 1) * num_experts + i] = 0;
    }

    for (size_t i = tid; i < numel; i += stride) {
      ++tokens_cnts[(local_id_x + 1) * num_experts + topk_ids[i]];
    }

    item.barrier(sycl::access::fence_space::local_space);

    if (local_id_x < num_experts) {
      tokens_cnts[local_id_x] = 0;
      for (int i = 1; i <= local_range; ++i) {
        tokens_cnts[i * num_experts + local_id_x] +=
            tokens_cnts[(i - 1) * num_experts + local_id_x];
      }
    }

    item.barrier(sycl::access::fence_space::local_space);

    if (local_id_x == 0) {
      cumsum[0] = 0;
      for (int i = 1; i <= num_experts; ++i) {
        cumsum[i] = cumsum[i - 1] +
            CEILDIV(
                tokens_cnts[local_range * num_experts + i - 1], block_size) *
                block_size;
      }
      *total_tokens_post_pad = static_cast<int32_t>(cumsum[num_experts]);
    }

    item.barrier(sycl::access::fence_space::local_space);

    if (local_id_x < num_experts) {
      for (int i = cumsum[local_id_x]; i < cumsum[local_id_x + 1];
           i += block_size) {
        expert_ids[i / block_size] = local_id_x;
      }
    }

    // Fill remaining expert_ids with 0
    const size_t fill_start_idx = cumsum[num_experts] / block_size + local_id_x;
    const size_t expert_ids_size = CEILDIV(max_num_tokens_padded, block_size);
    for (size_t i = fill_start_idx; i < expert_ids_size; i += local_range) {
      expert_ids[i] = 0;
    }

    for (size_t i = tid; i < numel; i += stride) {
      int32_t expert_id = topk_ids[i];
      int32_t rank_post_pad =
          tokens_cnts[local_id_x * num_experts + expert_id] + cumsum[expert_id];
      sorted_token_ids[rank_post_pad] = i;
      ++tokens_cnts[local_id_x * num_experts + expert_id];
    }
  }
};
} // namespace moe_align_block_size_impl

// taken from
// https://github.com/sgl-project/sglang/blob/8b5f83ed3b7d2a49ad5c5cd5aa61c5d502f47dbc
void moe_align_block_size(
    at::Tensor topk_ids,
    int64_t num_experts,
    int64_t block_size,
    at::Tensor sorted_token_ids,
    at::Tensor experts_ids,
    at::Tensor num_tokens_post_pad) {
  auto& queue = dpcppGetCurrentQueue();

  constexpr int32_t WARP_SIZE = 32;
  int64_t padded_num_experts =
      ((num_experts + WARP_SIZE - 1) / WARP_SIZE) * WARP_SIZE;
  int experts_per_warp = WARP_SIZE;
  int threads = 1024;
  threads = ((threads + WARP_SIZE - 1) / WARP_SIZE) * WARP_SIZE;

  // BlockScan uses 1024 threads and assigns one thread per expert.
  TORCH_CHECK(
      padded_num_experts < 1024, "padded_num_experts must be less than 1024");

  IPEX_DISPATCH_INTEGRAL_TYPES(
      topk_ids.scalar_type(), "moe_align_block_size_kernel", [&] {
        // calc needed amount of shared mem for `cumsum` tensors
        auto options_int =
            at::TensorOptions().dtype(at::kInt).device(topk_ids.device());
        at::Tensor cumsum_buffer = at::empty({num_experts + 1}, options_int);
        bool small_batch_expert_mode =
            (topk_ids.numel() < 1024) && (num_experts <= 64);

        if (small_batch_expert_mode) {
          const int32_t threads = (int32_t)num_experts > WARP_SIZE
              ? (int32_t)num_experts
              : WARP_SIZE;
          const int32_t shared_mem_size =
              ((threads + 1) * num_experts + (num_experts + 1)) *
              sizeof(int32_t);

          sycl::range<1> grid1(1);
          sycl::range<1> block1(threads);
          using small_batch_expert_kernel = moe_align_block_size_impl::
              moe_align_block_size_small_batch_expert_kernel<scalar_t>;

          auto cgf = DPCPP_Q_CGF(cgh) {
            sycl::local_accessor<int32_t, 1> slm(
                sycl::range<1>(shared_mem_size / sizeof(int32_t)), cgh);
            cgh.parallel_for(
                sycl::nd_range<1>(grid1 * block1, block1),
                small_batch_expert_kernel(
                    slm,
                    topk_ids.data_ptr<scalar_t>(),
                    sorted_token_ids.data_ptr<int32_t>(),
                    experts_ids.data_ptr<int32_t>(),
                    num_tokens_post_pad.data_ptr<int32_t>(),
                    num_experts,
                    block_size,
                    topk_ids.numel(),
                    sorted_token_ids.size(0)));
          };
          DPCPP_Q_SUBMIT(queue, cgf);
        } else {
          sycl::range<1> grid1(1);
          sycl::range<1> block1(threads);
          using align_kernel =
              moe_align_block_size_impl::moe_align_block_size_kernel<scalar_t>;

          size_t num_warps = CEILDIV(padded_num_experts, experts_per_warp);
          size_t shared_mem_num = 1024 + num_warps * experts_per_warp;

          auto cgf1 = DPCPP_Q_CGF(cgh) {
            sycl::local_accessor<int32_t, 1> slm(
                sycl::range<1>(shared_mem_num), cgh);
            cgh.parallel_for(
                sycl::nd_range<1>(grid1 * block1, block1),
                align_kernel(
                    slm,
                    topk_ids.data_ptr<scalar_t>(),
                    sorted_token_ids.data_ptr<int32_t>(),
                    experts_ids.data_ptr<int32_t>(),
                    num_tokens_post_pad.data_ptr<int32_t>(),
                    num_experts,
                    padded_num_experts,
                    experts_per_warp,
                    block_size,
                    topk_ids.numel(),
                    cumsum_buffer.data_ptr<int32_t>(),
                    sorted_token_ids.size(0)));
          };
          DPCPP_Q_SUBMIT(queue, cgf1);

          const int block_threads = std::min(256, (int)threads);
          const int num_blocks =
              (topk_ids.numel() + block_threads - 1) / block_threads;
          const int max_blocks = 65535;
          const int actual_blocks = std::min(num_blocks, max_blocks);

          sycl::range<1> grid2(actual_blocks);
          sycl::range<1> block2(block_threads);
          using sort_kernel =
              moe_align_block_size_impl::count_and_sort_expert_tokens_kernel<
                  scalar_t>;

          auto cgf2 = DPCPP_Q_CGF(cgh) {
            cgh.parallel_for(
                sycl::nd_range<1>(grid2 * block2, block2),
                sort_kernel(
                    topk_ids.data_ptr<scalar_t>(),
                    sorted_token_ids.data_ptr<int32_t>(),
                    cumsum_buffer.data_ptr<int32_t>(),
                    topk_ids.numel(),
                    num_experts));
          };
          DPCPP_Q_SUBMIT(queue, cgf2);
        }
      });
}

void batched_moe_align_block_size(
    int64_t max_tokens_per_batch,
    int64_t block_size,
    at::Tensor const& batch_num_tokens,
    at::Tensor sorted_ids,
    at::Tensor batch_ids,
    at::Tensor num_tokens_post_pad) {
  namespace batched_kernel = batched_moe_align_block_size_impl;

  auto& queue = dpcppGetCurrentQueue();
  int32_t const B = batch_num_tokens.size(0);
  int32_t const num_blocks_per_batch =
      round_to_next_multiple_of(max_tokens_per_batch, block_size) / block_size;
  int32_t const num_blocks = num_blocks_per_batch * B;
  int64_t const sorted_ids_size = num_blocks * block_size;

  TORCH_CHECK(sorted_ids.size(0) == sorted_ids_size);
  TORCH_CHECK(batch_ids.size(0) == sorted_ids_size / block_size);
  TORCH_CHECK(num_tokens_post_pad.size(0) == 1);
  TORCH_CHECK(B <= batched_kernel::num_threads);

  sycl::range<1> grid(batched_kernel::num_blocks);
  sycl::range<1> block(batched_kernel::num_threads);

  auto cgf = DPCPP_Q_CGF(cgh) {
    sycl::local_accessor<int32_t, 1> slm(sycl::range<1>(1024), cgh);
    cgh.parallel_for(
        sycl::nd_range<1>(grid * block, block),
        batched_kernel::batched_moe_align_block_size_kernel(
            slm,
            B,
            max_tokens_per_batch,
            block_size,
            batch_num_tokens.data_ptr<int32_t>(),
            sorted_ids.data_ptr<int32_t>(),
            batch_ids.data_ptr<int32_t>(),
            num_tokens_post_pad.data_ptr<int32_t>()));
  };
  DPCPP_Q_SUBMIT(queue, cgf);
}

#undef CEILDIV
} // namespace AtenIpexTypeXPU
} // namespace at

namespace {
IPEX_LIBRARY_FRAGMENT() {
  IPEX_OP_REGISTER(
      "moe_align_block_size.moe", at::AtenIpexTypeXPU::moe_align_block_size);
  IPEX_OP_REGISTER(
      "batched_moe_align_block_size.moe",
      at::AtenIpexTypeXPU::batched_moe_align_block_size);
}
} // namespace
