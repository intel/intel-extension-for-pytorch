#include "EmbeddingBagKernel.h"
#include "MemoryAccess.h"
#include "comm/ATDispatch.h"

namespace xpu {
namespace dpcpp {
namespace detail {

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
    bool ignore_offsets) {
#define EXTEND_EMBBAG_KERNEL_VEC(vec_size) \
  EMBBAG_KERNEL_NO_ACC(                    \
      scalar_t,                            \
      index_t,                             \
      MODE_MAX,                            \
      vec_size,                            \
      output,                              \
      weights,                             \
      input,                               \
      offset,                              \
      offset2bag,                          \
      bag_size,                            \
      max_indices,                         \
      per_sample_weights,                  \
      index_len,                           \
      bag_num,                             \
      vec_len,                             \
      padding_idx,                         \
      ignore_offsets)

  IPEX_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      weights.scalar_type(),
      "embedding_bag_max",
      [&] {
        IPEX_DISPATCH_INDEX_TYPES(
            indices.scalar_type(), "embedding_bag_max", [&] {
              using accscalar_t = at::AtenIpexTypeXPU::acc_type<scalar_t>;
              int vec_size =
                  at::native::Memory::can_vectorize_up_to_loop<scalar_t>(
                      dpcppGetDeviceIdOfCurrentQueue(),
                      (char*)weights.data_ptr());
              vec_size = vec_len % vec_size == 0 ? vec_size : 1;
              switch (vec_size) {
                case 8:
                  EXTEND_EMBBAG_KERNEL_VEC(8);
                  break;
                case 4:
                  EXTEND_EMBBAG_KERNEL_VEC(4);
                  break;
                case 2:
                  EXTEND_EMBBAG_KERNEL_VEC(2);
                  break;
                default:
                  EXTEND_EMBBAG_KERNEL_VEC(1);
                  break;
              };
            });
      });
}

} // namespace detail
} // namespace dpcpp
} // namespace xpu
