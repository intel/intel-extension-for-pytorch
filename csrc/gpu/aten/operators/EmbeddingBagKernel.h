#include <ATen/ATen.h>
#include <ATen/core/Array.h>

#include <runtime/Utils.h>
#include <utils/DPCPP.h>

#include "BatchKernel.h"
#include "comm/AccumulateType.h"
#include "comm/Numerics.h"

namespace xpu {
namespace dpcpp {
namespace detail {

constexpr int MODE_SUM = 0;
constexpr int MODE_MEAN = 1;
constexpr int MODE_MAX = 2;

#define EMBBAG_KERNEL_ACC(                                                   \
    scalar_t,                                                                \
    accscalar_t,                                                             \
    index_t,                                                                 \
    mode,                                                                    \
    vec_size,                                                                \
    output,                                                                  \
    weight,                                                                  \
    input,                                                                   \
    offset,                                                                  \
    offset2bag,                                                              \
    bag_size,                                                                \
    max_indices,                                                             \
    per_sample_weights,                                                      \
    index_len,                                                               \
    bag_num,                                                                 \
    vec_len,                                                                 \
    padding_idx,                                                             \
    ignore_offsets)                                                          \
  embedding_bag_kernel<scalar_t, accscalar_t, index_t, mode, vec_size>(      \
      output.data_ptr<scalar_t>(),                                           \
      weight.data_ptr<scalar_t>(),                                           \
      indices.data_ptr<index_t>(),                                           \
      offsets.data_ptr<index_t>(),                                           \
      offset2bag.data_ptr<index_t>(),                                        \
      bag_size.data_ptr<index_t>(),                                          \
      max_indices.data_ptr<index_t>(),                                       \
      per_sample_weights.defined() ? per_sample_weights.data_ptr<scalar_t>() \
                                   : nullptr,                                \
      index_size,                                                            \
      bag_num,                                                               \
      vec_len,                                                               \
      padding_idx,                                                           \
      ignore_offsets)

#define EMBBAG_KERNEL_NO_ACC(                                                \
    scalar_t,                                                                \
    index_t,                                                                 \
    mode,                                                                    \
    vec_size,                                                                \
    output,                                                                  \
    weight,                                                                  \
    input,                                                                   \
    offset,                                                                  \
    offset2bag,                                                              \
    bag_size,                                                                \
    max_indices,                                                             \
    per_sample_weights,                                                      \
    index_len,                                                               \
    bag_num,                                                                 \
    vec_len,                                                                 \
    padding_idx,                                                             \
    ignore_offsets)                                                          \
  embedding_bag_kernel<scalar_t, scalar_t, index_t, mode, vec_size>(         \
      output.data_ptr<scalar_t>(),                                           \
      weight.data_ptr<scalar_t>(),                                           \
      indices.data_ptr<index_t>(),                                           \
      offsets.data_ptr<index_t>(),                                           \
      offset2bag.data_ptr<index_t>(),                                        \
      bag_size.data_ptr<index_t>(),                                          \
      max_indices.data_ptr<index_t>(),                                       \
      per_sample_weights.defined() ? per_sample_weights.data_ptr<scalar_t>() \
                                   : nullptr,                                \
      index_size,                                                            \
      bag_num,                                                               \
      vec_len,                                                               \
      padding_idx,                                                           \
      ignore_offsets)

template <
    typename scalar_t,
    typename accscalar_t,
    typename index_t,
    int mode,
    int vec_size>
void embedding_bag_kernel(
    scalar_t* const output,
    scalar_t* const weights,
    index_t* const index,
    index_t* const offset,
    index_t* const offset2bag,
    index_t* const bag_size,
    index_t* const max_index,
    scalar_t* const per_sample_weights,
    int64_t index_size,
    int64_t bag_num,
    int64_t vec_len,
    index_t padding_idx,
    bool ignore_offsets) {
  using vec_t = at::detail::Array<scalar_t, vec_size>;
  using vec_acc_t = at::detail::Array<accscalar_t, vec_size>;
  using vec_idx_t = at::detail::Array<index_t, vec_size>;

  vec_t* o_vec = reinterpret_cast<vec_t*>(output);
  vec_t* w_vec = reinterpret_cast<vec_t*>(weights);
  vec_idx_t* max_idx_vec = reinterpret_cast<vec_idx_t*>(max_index);

  vec_len = vec_len / vec_size;
  BatchKernelConfig cfg = {
      bag_num, vec_len, 1, bag_num, true, BatchKernelConfig::Policy::pAdaptive};
  index_t fixing_bag_size = ignore_offsets ? index_size / bag_num : 0;
  auto cgf = DPCPP_Q_CGF(__cgh) {
    auto kfn = DPCPP_Q_KFN(sycl::nd_item<2> item) {
      auto desc = cfg.get_item_desc(item);
      index_t start = 0, end = 0;
      int64_t off_off = -1;

      do {
        if (desc.glb_problem < cfg.problem_ &&
            desc.glb_batch < cfg.problem_batch_) {
          bool walk_on_bag = desc.glb_batch != off_off;
          if (walk_on_bag) {
            off_off = desc.glb_batch;
            bool last_bag = off_off == bag_num - 1;
            if (!ignore_offsets) {
              start = offset[off_off];
              end = last_bag ? index_size : offset[off_off + 1];
            } else {
              start = off_off * fixing_bag_size;
              end = start + fixing_bag_size;
            }
          }

          vec_acc_t value, value_max;
          vec_idx_t index_max;
          index_t padding_cnt = 0;
#pragma unroll
          for (int i = 0; i < vec_size; i++) {
            value[i] = 0;
            value_max[i] = Numerics<accscalar_t>::lower_bound();
            index_max[i] = 0;
          }

          for (index_t off = start; off < end; off++) {
            index_t index_off = off;
            index_t vec_idx = index[index_off];

            if (walk_on_bag && desc.glb_problem == 0) {
              offset2bag[index_off] = off_off;
            }

            if (padding_idx != vec_idx) {
              index_t i_off = vec_idx * vec_len + desc.glb_problem;
              vec_t other = w_vec[i_off];

              if constexpr (mode == MODE_SUM) {
#pragma unroll
                for (int i = 0; i < vec_size; i++) {
                  if (per_sample_weights) {
                    other[i] *= per_sample_weights[index_off];
                  }
                  value[i] += other[i];
                }
              } else if constexpr (mode == MODE_MEAN) {
#pragma unroll
                for (int i = 0; i < vec_size; i++) {
                  value[i] += other[i];
                }
              } else if constexpr (mode == MODE_MAX) {
#pragma unroll
                for (int i = 0; i < vec_size; i++) {
                  if (other[i] > value_max[i]) {
                    value_max[i] = other[i];
                    if (max_index) {
                      index_max[i] = vec_idx;
                    }
                  }
                }
              }
            } else {
              padding_cnt++;
            }
          }

          int64_t bsize = end - start - padding_cnt;
          if (desc.glb_problem == 0) {
            bag_size[off_off] = bsize;
          }

          index_t o_off = off_off * vec_len + desc.glb_problem;
          if constexpr (mode == MODE_SUM) {
            vec_t o;
#pragma unroll
            for (int i = 0; i < vec_size; i++) {
              o[i] = value[i];
            }
            o_vec[o_off] = o;
          } else if constexpr (mode == MODE_MEAN) {
            vec_t o;
            bsize = bsize == 0 ? 1 : bsize;
#pragma unroll
            for (int i = 0; i < vec_size; i++) {
              o[i] = value[i] / bsize;
            }
            o_vec[o_off] = o;
          } else if constexpr (mode == MODE_MAX) {
            vec_t padding;
#pragma unroll
            for (int i = 0; i < vec_size; i++) {
              padding[i] = 0;
            }
            o_vec[o_off] = value_max[0] == Numerics<accscalar_t>::lower_bound()
                ? padding
                : value_max;
            if (max_index) {
              max_idx_vec[o_off] = index_max;
            }
          }
        }
      } while (cfg.next(item, desc));
    };
    __cgh.parallel_for(
        sycl::nd_range<2>(cfg.global_size(), cfg.group_size()), kfn);
  };
  DPCPP_Q_SUBMIT(dpcppGetCurrentQueue(), cgf);
}

void embedding_bag_sum_template(
    const Tensor& indices,
    const Tensor& offsets,
    const Tensor& weights,
    const Tensor& per_sample_weights,
    Tensor& output,
    Tensor& offset2bag,
    Tensor& bag_size,
    Tensor& max_indices,
    int64_t index_size,
    int64_t bag_num,
    int64_t vec_len,
    int64_t padding_idx,
    bool ignore_offsets);

void embedding_bag_mean_template(
    const Tensor& indices,
    const Tensor& offsets,
    const Tensor& weights,
    const Tensor& per_sample_weights,
    Tensor& output,
    Tensor& offset2bag,
    Tensor& bag_size,
    Tensor& max_indices,
    int64_t index_size,
    int64_t bag_num,
    int64_t vec_len,
    int64_t padding_idx,
    bool ignore_offsets);

void embedding_bag_max_template(
    const Tensor& indices,
    const Tensor& offsets,
    const Tensor& weights,
    const Tensor& per_sample_weights,
    Tensor& output,
    Tensor& offset2bag,
    Tensor& bag_size,
    Tensor& max_indices,
    int64_t index_size,
    int64_t bag_num,
    int64_t vec_len,
    int64_t padding_idx,
    bool ignore_offsets);

} // namespace detail
} // namespace dpcpp
} // namespace xpu
