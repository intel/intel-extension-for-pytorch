#include <math.h>

#include <ATen/ATen.h>
#include <ATen/TensorUtils.h>

namespace at {
namespace dpcpp {

template <typename scalar_t>
DPCPP_DEVICE inline scalar_t min(scalar_t a, scalar_t b) {
  return a < b ? a : b;
}

DPCPP_DEF_K1(nearest_neighbor_4d_dpcpp_kernel);
DPCPP_DEF_K1(nearest_neighbor_4d_bwd_dpcpp_kernel);

static inline void
upsample_2d_shape_check(const Tensor &input, const Tensor &grad_output,
                        int64_t nbatch, int64_t nchannels, int64_t input_height,
                        int64_t input_width, int64_t output_height,
                        int64_t output_width) {
  TORCH_CHECK(input_height > 0 && input_width > 0 && output_height > 0 &&
                  output_width > 0,
              "Input and output sizes should be greater than 0,"
              " but got input (H: ",
              input_height, ", W: ", input_width, ") output (H: ",
              output_height, ", W: ", output_width, ")");

  if (input.defined()) {
    TORCH_CHECK(
        input.numel() != 0 && input.dim() == 4,
        "Non-empty 4D data tensor expected but got a tensor with sizes ",
        input.sizes());
  } else if (grad_output.defined()) {
    check_dim_size(grad_output, 4, 0, nbatch);
    check_dim_size(grad_output, 4, 1, nchannels);
    check_dim_size(grad_output, 4, 2, output_height);
    check_dim_size(grad_output, 4, 3, output_width);
  }
}

DPCPP_DEVICE static int nearest_neighbor_compute_source_index(const float scale,
                                                              int dst_index,
                                                              int input_size) {
  const int src_index =
      min(static_cast<int>(DPCPP::floor(dst_index * scale)), input_size - 1);
  return src_index;
}

} // namespace native
} // namespace at
