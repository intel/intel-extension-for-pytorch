#include <ATen/ATen.h>
#include <ATen/native/TensorIterator.h>
#include <core/detail/OffsetCalculator.h>

#include "Loops.h"
#include "comm/Pointwise.h"
#include "comm/RegistrationDeclarations.h"

using namespace xpu::dpcpp;

namespace at {
namespace AtenIpexTypeXPU {
namespace impl {

void launch_cross_kernel(
    const TensorIteratorBase& iter,
    int64_t ostride,
    int64_t xstride,
    int64_t ystride) {
  const auto N = iter.numel();
  auto offset_calculator = make_element_offset_calculator<3>(iter);
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
      N > 0 && N <= std::numeric_limits<int32_t>::max());
  auto& dpcpp_queue = dpcppGetCurrentQueue();
  auto dev_id = dpcppGetDeviceIdOfCurrentQueue();
  int64_t work_group_size = dpcppMaxWorkGroupSize(dev_id);
  int64_t work_group_num = (N + work_group_size - 1) / work_group_size;

  IPEX_DISPATCH_ALL_TYPES_AND_COMPLEX_AND(
      kHalf, iter.common_dtype(), "cross_xpu", [&]() {
        auto cgf = DPCPP_Q_CGF(cgh) {
          auto out = static_cast<scalar_t*>(iter.data_ptr(0));
          auto x = static_cast<const scalar_t*>(iter.data_ptr(1));
          auto y = static_cast<const scalar_t*>(iter.data_ptr(2));

          auto kfn = DPCPP_Q_KFN(sycl::nd_item<1> item_id) {
            int64_t linear_index = item_id.get_global_id(0);
            for (int64_t i = linear_index; i < N;
                 i += work_group_num * work_group_size) {
              const auto offsets = offset_calculator.get(i);
              auto* out_row = out + offsets[0];
              const auto* x_row = x + offsets[1];
              const auto* y_row = y + offsets[2];

              const scalar_t val0 =
                  (x_row[1 * xstride] * y_row[2 * ystride] -
                   x_row[2 * xstride] * y_row[1 * ystride]);

              const scalar_t val1 =
                  (x_row[2 * xstride] * y_row[0 * ystride] -
                   x_row[0 * xstride] * y_row[2 * ystride]);

              const scalar_t val2 =
                  (x_row[0 * xstride] * y_row[1 * ystride] -
                   x_row[1 * xstride] * y_row[0 * ystride]);

              out_row[0 * ostride] = val0;
              out_row[1 * ostride] = val1;
              out_row[2 * ostride] = val2;
            }
          };
          cgh.parallel_for(
              sycl::nd_range<1>(
                  sycl::range<1>(work_group_num * work_group_size),
                  sycl::range<1>(work_group_size)),
              kfn);
        };

        DPCPP_Q_SUBMIT(dpcpp_queue, cgf);
      });
}

void cross(Tensor& result, const Tensor& x, const Tensor& y, int dimension) {
  const int64_t ostride = result.stride(dimension);
  const int64_t xstride = x.stride(dimension);
  const int64_t ystride = y.stride(dimension);

  auto iter =
      TensorIteratorConfig()
          .add_output(result)
          .add_input(x)
          .add_input(y)
          .resize_outputs(false)
          .declare_static_shape(result.sizes(), /*squash_dims=*/dimension)
          .build();

  if (iter.numel() == 0) {
    return;
  }

  if (iter.can_use_32bit_indexing()) {
    launch_cross_kernel(iter, ostride, xstride, ystride);
  } else {
    for (auto&& sub_iter : iter.with_32bit_indexing()) {
      launch_cross_kernel(sub_iter, ostride, xstride, ystride);
    }
  }
}

} // namespace impl

Tensor& cross_out(
    Tensor& out,
    const Tensor& input,
    const Tensor& other,
    const c10::optional<int64_t> dimension) {
  TORCH_CHECK(
      input.dim() == other.dim(),
      "inconsistent tensors dimensions input: ",
      input.dim(),
      " other: ",
      other.dim());
  TORCH_CHECK(
      input.sizes() == other.sizes(),
      "inconsistent tensors sizes input: ",
      input.sizes(),
      " other: ",
      other.sizes());

  int64_t dim = -1;
  if (!dimension.has_value()) {
    for (int64_t i = 0; i < input.dim(); i++) {
      if (input.size(i) == 3) {
        dim = i;
        break;
      }
    }
    TORCH_CHECK(dim >= 0, "no dimension of size 3 in input");
  } else {
    dim = maybe_wrap_dim(dimension.value(), input.dim());
    TORCH_CHECK(
        input.size(dim) == 3,
        "dimension ",
        dimension.value(),
        " does not have size 3");
  }

  if (out.sizes() != input.sizes()) {
    out.resize_as_(input);
  }

  IPEX_DISPATCH_ALL_TYPES_AND_COMPLEX_AND(
      at::ScalarType::Half, input.scalar_type(), "cross", [&]() {
        impl::cross(out, input, other, dim);
      });

  return out;
}

Tensor cross(
    const Tensor& input,
    const Tensor& other,
    const c10::optional<int64_t> dimension) {
  Tensor out = at::empty_like(input);
  at::AtenIpexTypeXPU::cross_out(out, input, other, dimension);
  return out;
}

} // namespace AtenIpexTypeXPU
} // namespace at
