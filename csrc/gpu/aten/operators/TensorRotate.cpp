#include <ATen/ATen.h>
#include <ATen/core/Array.h>

#include <core/MemoryFormat.h>
#include <core/detail/IndexUtils.h>
#include <runtime/Utils.h>
#include <utils/DPCPP.h>

#include "Reduce.h"
#include "comm/ATDispatch.h"
#include "comm/AccumulateType.h"
#include "comm/RegistrationDeclarations.h"

#include "comm/Numerics.h"
#include "utils/CustomOperatorRegistration.h"

using namespace xpu::dpcpp;

namespace at {
namespace AtenIpexTypeXPU {

template <
    typename scalar_t,
    int N,
    typename index_type,
    bool unsigned_index,
    template <int, typename, bool>
    class OffsetCalculator>
void apply_rotary_embedding_impl(
    scalar_t* tensor,
    scalar_t* sin,
    scalar_t* cos,
    scalar_t* out,
    OffsetCalculator<N, index_type, unsigned_index> offset_calc,
    int64_t problem_size,
    int64_t total_size) {
  auto& dpcpp_queue = dpcppGetCurrentQueue();
  auto dev_id = dpcppGetDeviceIdOfCurrentQueue();
  int64_t max_wg_size = dpcppMaxWorkGroupSize(dev_id);
  int64_t wg_size = std::min(max_wg_size, problem_size);
  int64_t work_group_num = (total_size + problem_size - 1) / problem_size;

  auto cgf = DPCPP_Q_CGF(cgh) {
    auto kfn = DPCPP_Q_KFN(sycl::nd_item<1> item_id) {
      auto item_idx = item_id.get_local_id(0);
      auto item_range = item_id.get_local_range(0);
      auto group_idx = item_id.get_group(0);
      auto sg = item_id.get_sub_group();
      scalar_t tensor_val;
      for (int i = item_idx; i < problem_size; i += item_range) {
        auto global_offset = group_idx * problem_size + i;
        const auto offset = offset_calc.get(global_offset);
        tensor_val = *(tensor + offset[1]);
        scalar_t scale = i % 2 == 0 ? -1 : 1;
        scalar_t shift_val = sg.shuffle_xor(tensor_val, 1) * scale;
        scalar_t sin_val = *(sin + offset[2]);
        scalar_t cos_val = *(cos + offset[3]);
        scalar_t out_val = shift_val * sin_val + tensor_val * cos_val;
        scalar_t* out_ptr = out + offset[0];
        *out_ptr = out_val;
      }
    };

    cgh.parallel_for(
        sycl::nd_range<1>(
            sycl::range<1>(work_group_num * wg_size), sycl::range<1>(wg_size)),
        kfn);
  };
  DPCPP_Q_SUBMIT(dpcpp_queue, cgf);
}

Tensor apply_rotary_embedding(
    const Tensor& tensor,
    const Tensor& sin,
    const Tensor& cos,
    Tensor& out) {
  int64_t tensor_ndim = tensor.ndimension();
  int64_t sin_dim = sin.ndimension();
  int64_t cos_dim = cos.ndimension();
  int64_t out_dim = out.ndimension();
  TORCH_CHECK(
      tensor_ndim == sin_dim && cos_dim == sin_dim && out_dim == sin_dim,
      "The dimensions of all tensor should be equal");
  int64_t numel = tensor.numel();
  int64_t problem_size = tensor.size(tensor_ndim - 1);
  TORCH_CHECK(
      tensor.size(tensor_ndim - 1) == sin.size(tensor_ndim - 1) &&
          sin.size(tensor_ndim - 1) == cos.size(tensor_ndim - 1) &&
          cos.size(tensor_ndim - 1) == out.size(tensor_ndim - 1),
      "The problem size of all tensor should be equal");
  TORCH_CHECK(
      tensor.size(tensor_ndim - 1) % 2 == 0,
      "The problem size should be divisible by 2")
  auto iter = TensorIteratorConfig()
                  .add_output(out)
                  .add_input(tensor)
                  .add_input(sin)
                  .add_input(cos)
                  .build();
  auto offset_calc = make_element_offset_calculator<4>(iter);

  IPEX_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      tensor.scalar_type(),
      "apply_rotary_embedding",
      [&]() {
        apply_rotary_embedding_impl(
            static_cast<scalar_t*>(tensor.data_ptr()),
            static_cast<scalar_t*>(sin.data_ptr()),
            static_cast<scalar_t*>(cos.data_ptr()),
            static_cast<scalar_t*>(out.data_ptr()),
            offset_calc,
            problem_size,
            numel);
      });
  return out;
}

namespace {
IPEX_LIBRARY_FRAGMENT() {
  IPEX_OP_REGISTER_DISPATCH(
      "apply_rotary_embedding", apply_rotary_embedding, c10::DispatchKey::XPU);
}
} // namespace

} // namespace AtenIpexTypeXPU
} // namespace at