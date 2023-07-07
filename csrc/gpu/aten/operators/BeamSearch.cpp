#include <ATen/ATen.h>
#include <runtime/Utils.h>
#include <utils/DPCPP.h>
#include "comm/ATDispatch.h"
#include "utils/CustomOperatorRegistration.h"
// std::tuple<Tensor, Tensor, Tensor> beam_search_topk(
//   Tensor log_score,
//   int32_t beam_size,
//   int32_t vocab_size

// ) {
//   int32_t tmp_output_len = 2 * beam_size * beam_size;
//   Tensor tmp_log_score = at::empty(logits_score.sizes(),
//   logits_score.options()); Tensor tmp_log_val = at::empty(tmp_output_len,
//   logits_score.options()); Tensor tmp_log_idx = at::empty(tmp_output_len,
//   logits_score.options().dtype(at::kInt));

// }
namespace at {
namespace AtenIpexTypeXPU {

template <typename scalar_t>
void update_beam_indices_kernel(
    scalar_t* src_cache_indices,
    scalar_t* out_cache_indices,
    scalar_t* beam_ids,
    int32_t max_length,
    int32_t step,
    int32_t beam_size,
    int32_t batch_size) {
  int32_t num_step = step + 1;
  auto& dpcpp_queue = dpcppGetCurrentQueue();
  int32_t wg_size = 32;
  int32_t wg_number = (num_step + wg_size - 1) / wg_size;

  auto cgf = DPCPP_Q_CGF(cgh) {
    auto kfn = DPCPP_Q_KFN(sycl::nd_item<2> item) {
      int32_t time_step =
          item.get_group(0) * item.get_local_range(0) + item.get_local_id(0);
      int32_t sentence_id =
          item.get_group(1) * item.get_local_range(1) + item.get_local_id(1);

      int32_t beam_id = sentence_id % beam_size;
      int32_t batch_id = sentence_id / beam_size;

      if (sentence_id < batch_size * beam_size && time_step < num_step) {
        const scalar_t src_beam = beam_ids[sentence_id];
        const int32_t src_offset = batch_size * beam_size * time_step +
            batch_id * beam_size + src_beam;
        const int32_t out_offset =
            batch_size * beam_size * time_step + batch_id * beam_size + beam_id;

        out_cache_indices[out_offset] =
            (time_step == step) ? beam_id : src_cache_indices[src_offset];
      }
    };

    cgh.parallel_for(
        sycl::nd_range<2>(
            sycl::range<2>(wg_number * wg_size, batch_size * beam_size),
            sycl::range<2>(wg_size, 1)),
        kfn);
  };
  DPCPP_Q_SUBMIT(dpcpp_queue, cgf);
}

Tensor& update_beam_indices_for_cache(
    const Tensor& src_cache_indices,
    Tensor& out_cache_indices,
    const Tensor& beam_ids,
    int64_t max_length,
    int64_t step,
    int64_t beam_size,
    int64_t batch_size) {
  IPEX_DISPATCH_INTEGRAL_TYPES(
      src_cache_indices.scalar_type(), "update_beam_indices_for_cache", [&]() {
        update_beam_indices_kernel(
            src_cache_indices.data_ptr<scalar_t>(),
            out_cache_indices.data_ptr<scalar_t>(),
            beam_ids.data_ptr<scalar_t>(),
            max_length,
            step,
            beam_size,
            batch_size);
      });
  return out_cache_indices;
}

namespace {
IPEX_LIBRARY_FRAGMENT() {
  IPEX_OP_REGISTER_DISPATCH(
      "update_beam_indices_for_cache",
      update_beam_indices_for_cache,
      c10::DispatchKey::XPU);
}

} // namespace

} // namespace AtenIpexTypeXPU
} // namespace at