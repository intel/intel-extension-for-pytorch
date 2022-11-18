#include <ATen/ATen.h>
#include <ATen/native/TensorIterator.h>
#include <runtime/Utils.h>

#include <core/detail/OffsetCalculator.h>
#include <utils/DPCPP.h>
#include "Loops.h"
#include "comm/ATDispatch.h"
#include "comm/RegistrationDeclarations.h"

constexpr int n_elems_per_work_item = UNROLLED_ELEM_PER_WORK_ITEM;

using namespace xpu::dpcpp;

namespace at {
namespace AtenIpexTypeXPU {

template <int n_elems_per_work_item, typename func_t>
void _elemwise_kernel(int total_n_elems, func_t f) {
  int total_work_items =
      (total_n_elems + n_elems_per_work_item - 1) / n_elems_per_work_item;
  auto& dpcpp_queue = dpcppGetCurrentQueue();
  auto cgf = DPCPP_Q_CGF(cgh) {
    cgh.parallel_for(
        sycl::range<1>(total_work_items), [=](sycl::item<1> itemId) {
          int idx = itemId.get_linear_id();
#pragma unroll
          for (int i = 0; i < n_elems_per_work_item; ++i) {
            if (idx < total_n_elems) {
              f(idx);
              idx += total_work_items;
            }
          }
        });
  };

  DPCPP_Q_SUBMIT(dpcpp_queue, cgf);
}

template <int n_elems_per_work_item, typename func_t>
void _lauch_kernel(int total_n_elems, const func_t& f) {
  TORCH_INTERNAL_ASSERT(
      total_n_elems >= 0 &&
      total_n_elems <= std::numeric_limits<int32_t>::max());

  _elemwise_kernel<n_elems_per_work_item, func_t>(total_n_elems, f);
}

template <typename scalar_t>
void _compute_linear_combination_internal_kernel(
    TensorIterator& iter,
    int32_t in_stride,
    int32_t coeff_stride,
    int32_t num_summations) {
  if (iter.numel() == 0) {
    return;
  }

  if (!iter.can_use_32bit_indexing()) {
    for (auto& sub_iter : iter.with_32bit_indexing()) {
      _compute_linear_combination_internal_kernel<scalar_t>(
          sub_iter, in_stride, coeff_stride, num_summations);
    }
    return;
  }

  auto offset_calc = make_offset_calculator<3>(iter);
  char* __restrict__ out_ptr = reinterpret_cast<char*>(iter.data_ptr(0));
  char* __restrict__ in_ptr = reinterpret_cast<char*>(iter.data_ptr(1));
  char* __restrict__ coeff_ptr = reinterpret_cast<char*>(iter.data_ptr(2));

  auto loop = [=](int idx) {
    auto offsets = offset_calc.get(idx);

    auto* __restrict__ out_data =
        reinterpret_cast<scalar_t*>(out_ptr + offsets[0]);
    auto* __restrict__ in_data =
        reinterpret_cast<scalar_t*>(in_ptr + offsets[1]);
    using primitive_t = typename scalar_value_type<scalar_t>::type;
    auto* __restrict__ coeff_data =
        reinterpret_cast<primitive_t*>(coeff_ptr + offsets[2]);

    // perform summation
    for (int32_t i = 0; i < num_summations; ++i) {
      *out_data += in_data[i * in_stride] * coeff_data[i * coeff_stride];
    }
  };

  _lauch_kernel<n_elems_per_work_item>(iter.numel(), loop);
}

void _compute_linear_combination_kernel_dpcpp(
    TensorIterator& iter,
    int64_t in_stride,
    int64_t coeff_stride,
    int64_t num_summations) {
  IPEX_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(
      at::ScalarType::Half,
      at::ScalarType::Bool,
      at::ScalarType::BFloat16,
      iter.dtype(),
      "_compute_linear_combination",
      [&]() {
        _compute_linear_combination_internal_kernel<scalar_t>(
            iter, in_stride, coeff_stride, num_summations);
      });
}

// If `coefficients` is a [m, n] Tensor and
// `input` is a [n, ...] Tensor, then the output
// `output` is going to be a [m, ...] Tensor such that
// for i in range(m):
//    for j in range(n):
//        output[i, ...] += coefficients[i, j] * input[j, ...]
//
// Note: if input.dtype == scalar_t<T>, then coefficients.dtype == T.
// This is relevant when scalar_t<T> == complex<T>.
Tensor _compute_linear_combination(
    const Tensor& input,
    const Tensor& coefficients) {
  auto output_first_dim_size = coefficients.size(0);

  auto output_sizes = input.sizes().vec();
  output_sizes[0] = output_first_dim_size;
  auto output = at::zeros(
      output_sizes,
      input.options().memory_format(at::MemoryFormat::Contiguous));

  at::AtenIpexTypeXPU::_compute_linear_combination_out(
      input, coefficients, output);

  return output;
}

at::Tensor& _compute_linear_combination_out(
    const at::Tensor& input,
    const at::Tensor& coefficients,
    at::Tensor& out);
// Note: the function is implemented using the __restrict__ memory modifier,
// which means that if `output` actually is aliased by `input`, the result
// produced is undefined.
Tensor& _compute_linear_combination_out(
    const Tensor& input,
    const Tensor& coefficients,
    Tensor& output) {
  auto output_first_dim_size = coefficients.size(0);
  auto input_first_dim_size = coefficients.size(1);

  // Recall that `coefficients` is a [m, n] Tensor,
  // `input` is a [n, ...] Tensor, `output` is a [m, ...] Tensor.
  // We restride Tensors to the common dim == input.dim() + 1, so that
  // coefficients.sizes() = [m, 1 (instead of n), 1 repeated (input.dim() - 1)
  // times], input.sizes() = [1, 1 (instead of n), ...], output.sizes() = [m, 1
  // (instead of n), ...]. The second dimension in newly restrided Tensors is
  // traversed inside the kernels. This is done to avoid synchronizations/atomic
  // operations in the kernels and also quarantees determinism, required by the
  // autograd.

  // restride output
  auto output_to_broadcasted_dim = output.unsqueeze(1);
  auto output_restrided_sizes = output_to_broadcasted_dim.sizes().vec();
  auto output_restrided_strides = output_to_broadcasted_dim.strides().vec();
  output_restrided_sizes[1] = 1;
  output_restrided_strides[1] = 0;
  auto output_restrided =
      output.as_strided(output_restrided_sizes, output_restrided_strides);

  // restride input
  auto input_to_broadcasted_dim = input.unsqueeze(0);
  auto input_restrided_sizes = input_to_broadcasted_dim.sizes().vec();
  auto input_restrided_strides = input_to_broadcasted_dim.strides().vec();
  input_restrided_sizes[1] = 1;
  input_restrided_strides[1] = 0;
  auto input_restrided =
      input.as_strided(input_restrided_sizes, input_restrided_strides);

  // restride coefficients
  auto coefficients_restrided_sizes = std::vector<int64_t>(input.dim() + 1, 1);
  coefficients_restrided_sizes[0] = output_first_dim_size;
  coefficients_restrided_sizes[1] = 1;
  auto coefficients_restrided_strides =
      std::vector<int64_t>(input.dim() + 1, 0);
  coefficients_restrided_strides[0] = coefficients.stride(0);
  coefficients_restrided_strides[1] = 0;
  auto coefficients_restrided = coefficients.as_strided(
      coefficients_restrided_sizes, coefficients_restrided_strides);
  auto iter = TensorIteratorConfig()
                  .set_check_mem_overlap(
                      false) // Output is intentionally 0 strided above
                  .check_all_same_dtype(false)
                  .resize_outputs(false)
                  .add_output(output_restrided)
                  .add_input(input_restrided)
                  .add_input(coefficients_restrided)
                  .build();

  // The dimension of size n is traversed inside the kernels,
  // it is the first dimension of `input` and the second of `coefficients`
  auto input_stride = input.stride(0);
  auto coeff_stride = coefficients.stride(1);
  _compute_linear_combination_kernel_dpcpp(
      iter, input_stride, coeff_stride, input_first_dim_size);
  return output;
}

} // namespace AtenIpexTypeXPU
} // namespace at