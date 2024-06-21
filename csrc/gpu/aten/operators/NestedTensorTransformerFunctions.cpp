#include <ATen/ATen.h>
#include <ATen/NestedTensorImpl.h>
#include <ATen/native/nested/NestedTensorUtils.h>

#include "comm/ATDispatch.h"
#include "runtime/Utils.h"
#include "utils/CustomOperatorRegistration.h"
#include "utils/DPCPP.h"

// keep align with cuda, global range0 is set to output_batch_size, global_range
// for dim1 is set to 16,
#define GRID_DIM_Y 16
#define BLOCK_DIM 1024

using namespace torch_ipex::xpu::dpcpp;

namespace at {
namespace AtenIpexTypeNestedTensorXPU {

namespace impl {

template <typename T>
struct add_padding_1_functor {
  void operator()(sycl::nd_item<2> item) const {
    const int batch_id = item.get_group()[0];
    const int grid_id = item.get_group()[1];
    const int local_range = item.get_local_range()[0];
    const int tid = item.get_local_id()[0] + grid_id * local_range;
    const int grainsize = GRID_DIM_Y * local_range;

    const int offset = offsets[batch_id];
    const int* sizes_i = input_sizes + batch_id * input_dim;
    const int batch_output_offset = batch_id * output_sizes_1;

    for (int ii = 0; ii < (output_sizes_1 / grainsize); ii++) {
      const int i = ii * grainsize + tid;
      const int output_offset = batch_output_offset + i;
      if (batch_id < batch_size && i < sizes_i[0]) {
        const int batch_input_offset = offsets[batch_id];
        output[output_offset] = input[batch_input_offset + i];
      } else {
        output[output_offset] = padding_value;
      }
    }

    const int i = (output_sizes_1 / grainsize) * grainsize + tid;
    if (i < output_sizes_1) {
      const int output_offset = batch_output_offset + i;
      if (batch_id < batch_size && (i < sizes_i[0])) {
        const int batch_input_offset = offsets[batch_id];
        output[output_offset] = input[batch_input_offset + i];
      } else {
        output[output_offset] = padding_value;
      }
    }
  }
  add_padding_1_functor(
      T* input_,
      T* output_,
      T padding_value_,
      const int* offsets_,
      const int* input_sizes_,
      int input_dim_,
      const int64_t output_sizes_1_,
      const int batch_size_,
      const int output_batch_size_)
      : input(input_),
        output(output_),
        padding_value(padding_value_),
        offsets(offsets_),
        input_sizes(input_sizes_),
        input_dim(input_dim_),
        output_sizes_1(output_sizes_1_),
        batch_size(batch_size_),
        output_batch_size(output_batch_size_) {}

 private:
  T* input;
  T* output;
  T padding_value;
  const int* offsets;
  const int* input_sizes;
  int input_dim;
  const int64_t output_sizes_1;
  const int batch_size;
  const int output_batch_size;
};

template <typename T>
struct add_padding_2_functor {
  void operator()(sycl::nd_item<2> item) const {
    const int batch_id = item.get_group()[0];
    const int grid_id = item.get_group()[1];
    const int local_range = item.get_local_range()[0];
    const int tid = item.get_local_id()[0] + grid_id * local_range;
    const int grainsize = GRID_DIM_Y * local_range;
    const int* sizes_i = input_sizes + batch_id * input_dim;
    const int output_offset = batch_id * output_sizes_1 * output_sizes_2;
    const int output_numel = output_sizes_1 * output_sizes_2;
    for (int ii = 0; ii < (output_numel / grainsize); ii++) {
      const int i = ii * grainsize + tid;
      const int i0 = i / (output_sizes_2);
      const int i1 = i - i0 * output_sizes_2;
      if (batch_id < batch_size && i0 < sizes_i[0] && i1 < sizes_i[1]) {
        const int offset = offsets[batch_id];
        const int input_offset = offset + i0 * sizes_i[1] + i1;
        output[output_offset + i] = input[input_offset];
      } else {
        output[output_offset + i] = padding_value;
      }
    }
    const int i = (output_numel / grainsize) * grainsize + tid;
    if (i < output_numel) {
      const int i0 = i / (output_sizes_2);
      const int i1 = i - i0 * output_sizes_2;
      if (batch_id < batch_size && i0 < sizes_i[0] && i1 < sizes_i[1]) {
        const int offset = offsets[batch_id];
        const int input_offset = offset + i0 * sizes_i[1] + i1;
        output[output_offset + i] = input[input_offset];
      } else {
        output[output_offset + i] = padding_value;
      }
    }
  }

  add_padding_2_functor(
      const T* input_,
      T* output_,
      T padding_value_,
      const int* offsets_,
      const int* input_sizes_,
      int input_dim_,
      int output_sizes_1_,
      int output_sizes_2_,
      const int batch_size_,
      const int output_batch_size_)
      : input(input_),
        output(output_),
        padding_value(padding_value_),
        offsets(offsets_),
        input_sizes(input_sizes_),
        input_dim(input_dim_),
        output_sizes_1(output_sizes_1_),
        output_sizes_2(output_sizes_2_),
        batch_size(batch_size_),
        output_batch_size(output_batch_size_) {}

 private:
  const T* input;
  T* output;
  T padding_value;
  const int* offsets;
  const int* input_sizes;
  int input_dim;
  int output_sizes_1;
  int output_sizes_2;
  const int batch_size;
  const int output_batch_size;
};

template <typename T>
struct add_padding_3_functor {
  void operator()(sycl::nd_item<2> item) const {
    const int batch_id = item.get_group()[0];
    const int grid_id = item.get_group()[1];
    const int local_range = item.get_local_range()[0];
    const int tid = item.get_local_id()[0] + grid_id * local_range;
    const int grainsize = GRID_DIM_Y * local_range;
    const int* sizes_i = input_sizes + batch_id * input_dim;
    const int output_offset =
        batch_id * output_sizes_1 * output_sizes_2 * output_sizes_3;
    const int output_numel = output_sizes_1 * output_sizes_2 * output_sizes_3;
    for (int ii = 0; ii < (output_numel / grainsize); ii++) {
      const int i = ii * grainsize + tid;
      const int i0 = i / (output_sizes_2 * output_sizes_3);
      const int i1 = (i % (output_sizes_2 * output_sizes_3)) / output_sizes_3;
      const int i2 = i % output_sizes_3;
      if (batch_id < batch_size && i0 < sizes_i[0] && i1 < sizes_i[1] &&
          i2 < sizes_i[2]) {
        const int offset = offsets[batch_id];
        const int input_offset =
            offset + i0 * (sizes_i[1] * sizes_i[2]) + i1 * sizes_i[2] + i2;
        output[output_offset + i] = input[input_offset];
      } else {
        output[output_offset + i] = padding_value;
      }
    }
    const int i = (output_numel / grainsize) * grainsize + tid;
    if (i < output_numel) {
      const int i0 = i / (output_sizes_2 * output_sizes_3);
      const int i1 = (i % (output_sizes_2 * output_sizes_3)) / output_sizes_3;
      const int i2 = i % output_sizes_3;
      if (batch_id < batch_size && i0 < sizes_i[0] && i1 < sizes_i[1] &&
          i2 < sizes_i[2]) {
        const int offset = offsets[batch_id];
        const int input_offset =
            offset + i0 * (sizes_i[1] * sizes_i[2]) + i1 * sizes_i[2] + i2;
        output[output_offset + i] = input[input_offset];
      } else {
        output[output_offset + i] = padding_value;
      }
    }
  }

  add_padding_3_functor(
      const T* input_,
      T* output_,
      T padding_value_,
      const int* offsets_,
      const int* input_sizes_,
      int input_dim_,
      int output_sizes_1_,
      int output_sizes_2_,
      int output_sizes_3_,
      const int batch_size_,
      const int output_batch_size_)
      : input(input_),
        output(output_),
        padding_value(padding_value_),
        offsets(offsets_),
        input_sizes(input_sizes_),
        input_dim(input_dim_),
        output_sizes_1(output_sizes_1_),
        output_sizes_2(output_sizes_2_),
        output_sizes_3(output_sizes_3_),
        batch_size(batch_size_),
        output_batch_size(output_batch_size_) {}

 private:
  const T* input;
  T* output;
  T padding_value;
  const int* offsets;
  const int* input_sizes;
  int input_dim;
  int output_sizes_1;
  int output_sizes_2;
  int output_sizes_3;
  const int batch_size;
  const int output_batch_size;
};

template <typename T>
void add_padding_kernelLauncher(
    T* input, // [batch_size x None]
    T* output, // [batch_size x max(input.nested_size(1)) x inner_size]
    T padding_value,
    const int* offsets,
    const int* input_sizes,
    int input_dim,
    const std::vector<int64_t>& output_sizes,
    const int batch_size,
    const int output_batch_size) {
  auto& dpcpp_queue = dpcppGetCurrentQueue();
  auto max_wg_size = dpcppMaxWorkGroupSize();
  if (input_dim == 1) {
    auto cgf = DPCPP_Q_CGF(cgh) {
      add_padding_1_functor<T> kfn(
          input,
          output,
          padding_value,
          offsets,
          input_sizes,
          input_dim,
          output_sizes[1],
          batch_size,
          output_batch_size);

      cgh.parallel_for<decltype(kfn)>(
          sycl::nd_range<2>(
              sycl::range<2>(output_batch_size * max_wg_size, GRID_DIM_Y),
              sycl::range<2>(max_wg_size, 1)),
          kfn);
    };
    DPCPP_Q_SUBMIT(dpcpp_queue, cgf);
  }
  if (input_dim == 2) {
    auto cgf = DPCPP_Q_CGF(cgh) {
      add_padding_2_functor<T> kfn(
          input,
          output,
          padding_value,
          offsets,
          input_sizes,
          input_dim,
          output_sizes[1],
          output_sizes[2],
          batch_size,
          output_batch_size);

      cgh.parallel_for<decltype(kfn)>(
          sycl::nd_range<2>(
              sycl::range<2>(output_batch_size * max_wg_size, GRID_DIM_Y),
              sycl::range<2>(max_wg_size, 1)),
          kfn);
    };
    DPCPP_Q_SUBMIT(dpcpp_queue, cgf);
  }
  if (input_dim == 3) {
    auto cgf = DPCPP_Q_CGF(cgh) {
      add_padding_3_functor<T> kfn(
          input,
          output,
          padding_value,
          offsets,
          input_sizes,
          input_dim,
          output_sizes[1],
          output_sizes[2],
          output_sizes[3],
          batch_size,
          output_batch_size);

      cgh.parallel_for<decltype(kfn)>(
          sycl::nd_range<2>(
              sycl::range<2>(output_batch_size * max_wg_size, GRID_DIM_Y),
              sycl::range<2>(max_wg_size, 1)),
          kfn);
    };
    DPCPP_Q_SUBMIT(dpcpp_queue, cgf);
  }
}

Tensor batch_offsets_from_efficient_size(const Tensor& ef_sizes) {
  int64_t* nt_sizes_ptr = ef_sizes.data_ptr<int64_t>();
  int64_t ef_sizes_size_0 = ef_sizes.sizes()[0];
  Tensor offsets = at::empty({1 + ef_sizes_size_0}, at::kLong);
  int64_t* offsets_ptr = offsets.mutable_data_ptr<int64_t>();
  offsets_ptr[0] = 0;
  int64_t ef_sizes_size_1 = ef_sizes.sizes()[1];
  for (const auto i : c10::irange(ef_sizes_size_0)) {
    int64_t prod = 1;
    for (const auto j : c10::irange(ef_sizes_size_1)) {
      prod = prod * nt_sizes_ptr[i * ef_sizes_size_1 + j];
    }
    offsets_ptr[i + 1] = offsets_ptr[i] + prod;
  }
  return offsets;
}

// =================nested from padded related======================
template <typename T>
struct remove_padding_functor {
  void operator()(sycl::nd_item<2> item) const {
    const int batch_id = item.get_group(0);
    const int grid_id = item.get_group(1);
    const int tid = item.get_local_id()[0] + grid_id * BLOCK_DIM;
    const int grainsize = GRID_DIM_Y * BLOCK_DIM;
    const int offset = offsets[batch_id];

    const int* sizes_i = output_sizes + batch_id * output_dim;
    const int numel_i = sizes_i[0] * sizes_i[1] * sizes_i[2];
    int input_offset =
        batch_id * input_sizes[1] * input_sizes[2] * input_sizes[3];
    for (int ii = 0; ii < (numel_i / grainsize); ii++) {
      const int i = ii * grainsize + tid;
      const int i0 = i / (sizes_i[1] * sizes_i[2]);
      const int i1 = (i % (sizes_i[1] * sizes_i[2])) / sizes_i[2];
      const int i2 = i % sizes_i[2];
      const int i0_offset = i0 * input_sizes[2] * input_sizes[3];
      const int i1_offset = i1 * input_sizes[3];
      output[offset + i] = input[input_offset + i0_offset + i1_offset + i2];
    }
    const int i = (numel_i / grainsize) * grainsize + tid;
    if (i < numel_i) {
      const int i0 = i / (sizes_i[1] * sizes_i[2]);
      const int i1 = (i % (sizes_i[1] * sizes_i[2])) / sizes_i[2];
      const int i2 = i % sizes_i[2];
      const int i0_offset = i0 * input_sizes[2] * input_sizes[3];
      const int i1_offset = i1 * input_sizes[3];
      output[offset + i] = input[input_offset + i0_offset + i1_offset + i2];
    }
  }

  remove_padding_functor(
      const T* input_,
      T* output_,
      const int* offsets_,
      const int* input_sizes_,
      const int* output_sizes_,
      int output_dim_,
      const int batch_size_)
      : input(input_),
        output(output_),
        offsets(offsets_),
        input_sizes(input_sizes_),
        output_sizes(output_sizes_),
        output_dim(output_dim_),
        batch_size(batch_size_) {}

 private:
  const T* input;
  T* output;
  const int* offsets;
  const int* input_sizes;
  const int* output_sizes;
  int output_dim;
  const int batch_size;
};

template <typename T>
struct remove_padding_2_functor {
  void operator()(sycl::nd_item<2> item) const {
    const int batch_id = item.get_group(0);
    const int grid_id = item.get_group(1);
    const int tid = item.get_local_id()[0] + grid_id * BLOCK_DIM;
    const int grainsize = GRID_DIM_Y * BLOCK_DIM;
    const int offset = offsets[batch_id];
    const int* sizes_i = output_sizes + batch_id * output_dim;
    const int numel_i = sizes_i[0] * sizes_i[1];
    int input_offset = batch_id * input_sizes[1] * input_sizes[2];
    for (int ii = 0; ii < (numel_i / grainsize); ii++) {
      const int i = ii * grainsize + tid;
      const int i0 = i / sizes_i[1];
      const int i1 = i % sizes_i[1];
      const int i0_offset = i0 * input_sizes[2];
      output[offset + i] = input[input_offset + i0_offset + i1];
    }
    const int i = (numel_i / grainsize) * grainsize + tid;
    if (i < numel_i) {
      const int i0 = i / sizes_i[1];
      const int i1 = i % sizes_i[1];
      const int i0_offset = i0 * input_sizes[2];
      output[offset + i] = input[input_offset + i0_offset + i1];
    }
  }

  remove_padding_2_functor(
      const T* input_,
      T* output_,
      const int* offsets_,
      const int* input_sizes_,
      const int* output_sizes_,
      int output_dim_,
      const int batch_size_)
      : input(input_),
        output(output_),
        offsets(offsets_),
        input_sizes(input_sizes_),
        output_sizes(output_sizes_),
        output_dim(output_dim_),
        batch_size(batch_size_) {}

  const T* input;
  T* output;
  const int* offsets;
  const int* input_sizes;
  const int* output_sizes;
  int output_dim;
  const int batch_size;
};

template <typename T>
struct remove_padding_transform0213_functor {
  void operator()(sycl::nd_item<2> item) const {
    const int batch_id = item.get_group(0);
    const int grid_id = item.get_group(1);
    const int tid = item.get_local_id()[0] + grid_id * BLOCK_DIM;
    const int grainsize = GRID_DIM_Y * BLOCK_DIM;
    const int offset = offsets[batch_id];
    const int* sizes_i = output_sizes + batch_id * output_dim;
    const int numel_i = sizes_i[0] * sizes_i[1];
    int input_offset =
        batch_id * input_sizes[1] * input_sizes[2] * input_sizes[3];
    for (int ii = 0; ii < (numel_i / grainsize); ii++) {
      const int i = ii * grainsize + tid;
      const int i2 = i / sizes_i[1];
      const int i13 = i % sizes_i[1];
      const int i1 = i13 / (sizes_i[1] / input_sizes[1]);
      const int i3 = i13 % (sizes_i[1] / input_sizes[1]);

      output[offset + i] = input
          [input_offset + i1 * input_sizes[2] * input_sizes[3] +
           i2 * input_sizes[3] + i3];
    }
    const int i = (numel_i / grainsize) * grainsize + tid;
    if (i < numel_i) {
      const int i2 = i / sizes_i[1];
      const int i13 = i % sizes_i[1];
      const int i1 = i13 / (sizes_i[1] / input_sizes[1]);
      const int i3 = i13 % (sizes_i[1] / input_sizes[1]);
      output[offset + i] = input
          [input_offset + i1 * input_sizes[2] * input_sizes[3] +
           i2 * input_sizes[3] + i3];
    }
  }

  remove_padding_transform0213_functor(
      const T* input_,
      T* output_,
      const int* offsets_,
      const int* input_sizes_,
      const int* output_sizes_,
      int output_dim_,
      const int batch_size_)
      : input(input_),
        output(output_),
        offsets(offsets_),
        input_sizes(input_sizes_),
        output_sizes(output_sizes_),
        output_dim(output_dim_),
        batch_size(batch_size_) {}

  const T* input;
  T* output;
  const int* offsets;
  const int* input_sizes;
  const int* output_sizes;
  int output_dim;
  const int batch_size;
};

} // namespace impl

template <typename T>
void remove_padding_transform0213_kernelLauncher(
    const T* input,
    T* output,
    const int* offsets,
    const int* input_sizes,
    const int* output_sizes,
    int output_dim,
    const int batch_size) {
  TORCH_CHECK(
      output_dim == 2,
      "remove padding transform0213 only support output dim == 2");

  auto& dpcpp_queue = dpcppGetCurrentQueue();
  auto max_wg_size = dpcppMaxWorkGroupSize();

  auto cgf = DPCPP_Q_CGF(cgh) {
    impl::remove_padding_transform0213_functor<T> kfn(
        input,
        output,
        offsets,
        input_sizes,
        output_sizes,
        output_dim,
        batch_size);

    cgh.parallel_for<decltype(kfn)>(
        sycl::nd_range<2>(
            sycl::range<2>(batch_size * max_wg_size, GRID_DIM_Y),
            sycl::range<2>(max_wg_size, 1)),
        kfn);
  };
  DPCPP_Q_SUBMIT(dpcpp_queue, cgf);
}

template <typename T>
void remove_padding_kernelLauncher(
    const T* input,
    T* output,
    const int* offsets,
    const int* input_sizes,
    const int* output_sizes,
    int output_dim,
    const int batch_size) {
  auto& dpcpp_queue = dpcppGetCurrentQueue();
  auto max_wg_size = dpcppMaxWorkGroupSize();
  if (output_dim == 2) {
    auto cgf = DPCPP_Q_CGF(cgh) {
      impl::remove_padding_2_functor<T> kfn(
          input,
          output,
          offsets,
          input_sizes,
          output_sizes,
          output_dim,
          batch_size);

      cgh.parallel_for<decltype(kfn)>(
          sycl::nd_range<2>(
              sycl::range<2>(batch_size * max_wg_size, GRID_DIM_Y),
              sycl::range<2>(max_wg_size, 1)),
          kfn);
    };
    DPCPP_Q_SUBMIT(dpcpp_queue, cgf);
  } else {
    auto cgf = DPCPP_Q_CGF(cgh) {
      impl::remove_padding_functor<T> kfn(
          input,
          output,
          offsets,
          input_sizes,
          output_sizes,
          output_dim,
          batch_size);
      cgh.parallel_for<decltype(kfn)>(
          sycl::nd_range<2>(
              sycl::range<2>(batch_size * max_wg_size, GRID_DIM_Y),
              sycl::range<2>(max_wg_size, 1)),
          kfn);
    };

    DPCPP_Q_SUBMIT(dpcpp_queue, cgf);
  }
}

Tensor NestedTensor_batch_offsets_from_size_tensor(
    const Tensor& sizes,
    int64_t extra_elements) {
  int64_t* const sizes_ptr = sizes.data_ptr<int64_t>();
  Tensor offsets = at::empty({1 + sizes.size(0) + extra_elements}, at::kInt);
  int32_t* const offsets_ptr = offsets.mutable_data_ptr<int32_t>();
  offsets_ptr[0] = 0;
  const auto sizes_size_1 = sizes.size(1);
  const auto sizes_size_0 = sizes.size(0);
  for (const auto i : c10::irange(sizes_size_0)) {
    int64_t prod = 1;
    for (const auto j : c10::irange(sizes_size_1)) {
      prod *= sizes_ptr[i * sizes_size_1 + j];
    }
    offsets_ptr[i + 1] = offsets_ptr[i] + prod;
  }
  return offsets;
}

int64_t padded_tensor_numel(const Tensor& sizes) {
  const auto sizes_num_rows = sizes.sizes()[0];
  const auto sizes_row_length = sizes.sizes()[1];
  const auto* sizes_data = sizes.data_ptr<int64_t>();
  int64_t numel = 0;
  for (const auto row_num : c10::irange(sizes_num_rows)) {
    const auto* row_ptr = sizes_data + row_num * sizes_row_length;
    int64_t prod = 1;
    for (const auto idx : c10::irange(sizes_row_length)) {
      prod *= row_ptr[idx];
    }
    numel += prod;
  }
  return numel;
}

Tensor to_padded_tensor(
    const Tensor& t,
    double padding,
    c10::OptionalIntArrayRef output_size) {
  int64_t t_dim = t.dim();

  // here we align with cuda, only support dim within [2, 3, 4]
  if (t_dim >= 2 && t_dim <= 4 &&
      (t.dtype() == at::kFloat || t.dtype() == at::kDouble ||
       t.dtype() == at::kHalf)) {
    auto* nt_input = at::native::get_nested_tensor_impl(t);
    TORCH_CHECK(
        nested_tensor_impl_is_contiguous(nt_input),
        "for now to_padded_tensor only supports contiguous nested tensor");
    const auto& nt_buffer = nt_input->get_buffer();

    // if dim equals to 3, and final dim is regular, can be converted to 2 dim
    // nested tensor
    if (t_dim == 3 && nt_input->opt_size(2) && (*nt_input->opt_size(2) > 0) &&
        !(output_size.has_value())) {
      Tensor nt_sizes = nt_input->get_nested_sizes();
      Tensor sizes_dim1 = at::native::narrow_symint(nt_sizes, 1, 0, 1);
      Tensor sizes_dim2 = at::native::narrow_symint(nt_sizes, 1, 1, 1);
      Tensor result = at::detail::make_tensor<native::NestedTensorImpl>(
          nt_input->get_buffer(), sizes_dim1 * sizes_dim2[0]);
      TORCH_INTERNAL_ASSERT_DEBUG_ONLY(result.dim() == 2);
      result = to_padded_tensor(result, padding, output_size);
      return result.reshape({result.sizes()[0], -1, *nt_input->opt_size(2)});
    }

    Tensor nt_sizes = nt_input->get_nested_sizes();
    // get stride between each tensor, if tensor 1 size is 2,3,4
    // tensor 2 size is 2,3,5, will return 0, 24, 54
    Tensor offsets = impl::batch_offsets_from_efficient_size(nt_sizes);
    // get max for each dim of nested tensor
    auto new_size = NestedTensor_get_max_size(*nt_input);
    new_size.insert(new_size.begin(), nt_sizes.sizes()[0]);
    // Pad output tensor to output_size if provided
    if (output_size.has_value()) {
      auto output_size_ = output_size.value();
      TORCH_CHECK(
          output_size_.size() == new_size.size(),
          "Length of output_size does not match NestedTensor dims. Broadcasting is not supported.");
      for (uint64_t i = 0; i < new_size.size(); i++) {
        TORCH_CHECK(
            output_size_[i] >= new_size[i],
            "Value in output_size is less than NestedTensor padded size. Truncation is not supported.");
        new_size[i] = output_size_[i];
      }
    }

    Tensor output = at::empty(IntArrayRef(new_size), nt_buffer.options());

    int64_t batch_size = nt_sizes.sizes()[0];
    int64_t input_dim = nt_sizes.sizes()[1];
    int64_t output_batch_size = new_size[0];
    nt_sizes = nt_sizes.reshape(-1);

    offsets = offsets.to(at::Device(kXPU), at::kInt);
    nt_sizes = nt_sizes.to(at::Device(kXPU), at::kInt);

    IPEX_DISPATCH_FLOATING_TYPES_AND_HALF(
        nt_buffer.scalar_type(), "NestedTensor_to_padded_tensor_xpu", [&]() {
          impl::add_padding_kernelLauncher(
              nt_buffer.data_ptr<scalar_t>(),
              output.data_ptr<scalar_t>(),
              (scalar_t)(padding),
              offsets.data_ptr<int>(),
              nt_sizes.data_ptr<int>(),
              input_dim,
              new_size,
              batch_size,
              output_batch_size);
        });
    return output;
  }

  return at::native::NestedTensor_to_padded_tensor_generic(
      t, padding, output_size);
}

Tensor NestedTensor_to_mask(
    const Tensor& nt,
    c10::optional<int64_t> mask_dim,
    c10::optional<int64_t> mask_dim_length) {
  auto* nt_impl = native::get_nested_tensor_impl(nt);
  TORCH_CHECK(
      nested_tensor_impl_is_contiguous(nt_impl),
      "to_mask only works on contiguous NestedTensors.");
  TORCH_CHECK(
      !mask_dim || *mask_dim < nt.dim(),
      "Requested mask dimension ",
      *mask_dim,
      " is bigger than dimension ",
      nt.dim(),
      " of given NestedTensor.");

  // TODO: port optimization for 1x1 tensors from
  // pytorch/nestedtensor's version.

  TORCH_CHECK(
      mask_dim && *mask_dim == 2 && nt.dim() == 3,
      "Only the special case of mask_dim == 2 on a 3-D NestedTensor is supported right now.")
  const auto& sizes = nt_impl->get_nested_sizes();
  // Shape: # of tensors in our NestedTensor by max size along first dim
  // TODO: calculate this without allocating a std::vector.
  const auto result_size_1 = mask_dim_length
      ? *mask_dim_length
      : native::NestedTensor_get_max_size(*nt_impl)[0];
  auto result = at::ones({sizes.sizes()[0], result_size_1}, at::kBool);
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(sizes.dim() == 2);
  auto* result_data = result.data_ptr<bool>();
  auto* sizes_ptr = sizes.data_ptr<int64_t>();
  const auto sizes_size_1 = sizes.sizes()[1];
  for (const auto ii : c10::irange(sizes.sizes()[0])) {
    auto length = sizes_ptr[ii * sizes_size_1];
    for (const auto jj : c10::irange(length)) {
      result_data[ii * result_size_1 + jj] = false;
    }
  }
  return result;
}

Tensor _nested_tensor_softmax_with_shape(
    const Tensor& self,
    const Tensor& query) {
  c10::optional<Tensor> attn_mask;

  attn_mask = NestedTensor_to_mask(query, 2, self.size(2));
  attn_mask = attn_mask->to(query.device(), /*non-blocking=*/true);
  return _masked_softmax(self, *attn_mask, self.dim() - 1, /*mask type */ 1);
}

} // namespace AtenIpexTypeNestedTensorXPU
} // namespace at

namespace at {
namespace AtenIpexTypeXPU {

Tensor _nested_from_padded(
    const Tensor& padded,
    const Tensor& sizes,
    bool do_transform_0213) {
  if (padded.dim() > 1 && padded.dim() < 5) {
    // Instead of erroring, call the generic version
    if (!(padded.dim() == 4 && do_transform_0213) &&
        !(padded.dim() == 3 && !do_transform_0213)) {
      return at::native::nested_from_padded_generic(
          padded, sizes, do_transform_0213);
    }
    if (padded.dtype() != at::kFloat && padded.dtype() != kHalf) {
      TORCH_WARN_ONCE(
          "nested_from_padded CUDA kernels only support fp32/fp16; falling "
          "back to slower generic kernel");
      return at::native::nested_from_padded_generic(
          padded, sizes, do_transform_0213);
    }

    Tensor target_offsets = AtenIpexTypeNestedTensorXPU::
        NestedTensor_batch_offsets_from_size_tensor(sizes, 0);
    Tensor padded_sizes_tensor = at::tensor(padded.sizes());
    Tensor output = at::empty(
        {AtenIpexTypeNestedTensorXPU::padded_tensor_numel(sizes)},
        padded.options());
    Tensor target_size_sizes = sizes.reshape(-1);

    target_offsets = target_offsets.to(at::Device(kXPU), at::kInt);
    padded_sizes_tensor = padded_sizes_tensor.to(at::Device(kXPU), at::kInt);
    target_size_sizes = target_size_sizes.to(at::Device(kXPU), at::kInt);

    auto output_size_ptr = target_size_sizes.data_ptr<int>();
    auto input_size_ptr = padded_sizes_tensor.data_ptr<int>();
    auto offsets_ptr = target_offsets.data_ptr<int>();

    Tensor padded_contiguous = padded.contiguous();

    if (padded.dtype() == at::kFloat) {
      if (do_transform_0213) {
        AtenIpexTypeNestedTensorXPU::
            remove_padding_transform0213_kernelLauncher(
                padded_contiguous.data_ptr<float>(),
                output.data_ptr<float>(),
                offsets_ptr,
                input_size_ptr,
                output_size_ptr,
                padded_contiguous.dim() - 2,
                padded_contiguous.sizes()[0]);
      } else {
        AtenIpexTypeNestedTensorXPU::remove_padding_kernelLauncher(
            padded_contiguous.data_ptr<float>(),
            output.data_ptr<float>(),
            offsets_ptr,
            input_size_ptr,
            output_size_ptr,
            padded_contiguous.dim() - 1,
            padded_contiguous.sizes()[0]);
      }
    } else if (padded.dtype() == at::kHalf) {
      if (do_transform_0213) {
        AtenIpexTypeNestedTensorXPU::
            remove_padding_transform0213_kernelLauncher(
                padded_contiguous.data_ptr<c10::Half>(),
                output.data_ptr<c10::Half>(),
                offsets_ptr,
                input_size_ptr,
                output_size_ptr,
                padded_contiguous.dim() - 2,
                padded_contiguous.sizes()[0]);
      } else {
        AtenIpexTypeNestedTensorXPU::remove_padding_kernelLauncher(
            padded_contiguous.data_ptr<c10::Half>(),
            output.data_ptr<c10::Half>(),
            offsets_ptr,
            input_size_ptr,
            output_size_ptr,
            padded_contiguous.dim() - 1,
            padded_contiguous.sizes()[0]);
      }
    } else {
      AT_ERROR("Only support fp32/fp16 for padded input");
    }
    return at::detail::make_tensor<at::native::NestedTensorImpl>(
        std::move(output), sizes);
  } else {
    return at::native::nested_from_padded_generic(padded, sizes);
  }
}

} // namespace AtenIpexTypeXPU
} // namespace at
