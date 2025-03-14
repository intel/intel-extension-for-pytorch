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

} // namespace AtenIpexTypeNestedTensorXPU
} // namespace at

namespace at {
namespace native {

Tensor _nested_from_padded_xpu(
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

} // namespace native
} // namespace at
