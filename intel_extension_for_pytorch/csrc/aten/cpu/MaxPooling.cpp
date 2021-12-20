#include "MaxPooling.h"
#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/NamedTensorUtils.h>
#include <ATen/Parallel.h>
#include <ATen/core/grad_mode.h>
#include <ATen/cpu/vec/vec.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/Pool.h>
#include <ATen/native/cpu/utils.h>
#include <ATen/record_function.h>

#include "csrc/utils/library.h"

namespace torch_ipex {
namespace cpu {

template <typename scalar_t, typename accscalar_t>
void cpu_max_pool(
    const at::Tensor& output_,
    const at::Tensor& indices_,
    const at::Tensor& input_,
    int kW,
    int kH,
    int dW,
    int dH,
    int padW,
    int padH,
    int dilationW,
    int dilationH) {
  auto input = input_.contiguous();
  auto output = output_.contiguous();
  auto indices = indices_.contiguous();

  auto input_data = input.data_ptr<scalar_t>();
  auto output_data = output.data_ptr<scalar_t>();
  auto indices_data = indices.data_ptr<int64_t>();

  int64_t numel = output.numel();
  int64_t ndim = input.ndimension();
  // treat batch size and channels as one dimension
  int64_t channels = ndim == 3 ? input.size(0) : input.size(0) * input.size(1);
  int64_t input_height = input.size(-2);
  int64_t input_width = input.size(-1);
  int64_t output_height = output.size(-2);
  int64_t output_width = output.size(-1);

  // parallel on dim N, C, H, W
  at::parallel_for(0, numel, 0, [&](int64_t begin, int64_t end) {
    int64_t c = 0;
    int64_t oh = 0;
    int64_t ow = 0;
    at::native::data_index_init(
        begin, c, channels, oh, output_height, ow, output_width);

    for (int64_t i = begin; i < end; i++) {
      int64_t ih0 = oh * dH - padH;
      int64_t iw0 = ow * dW - padW;
      int64_t ih1 = std::min(ih0 + (kH - 1) * dilationH + 1, input_height);
      int64_t iw1 = std::min(iw0 + (kW - 1) * dilationW + 1, input_width);
      while (ih0 < 0) {
        ih0 += dilationH;
      }
      while (iw0 < 0) {
        iw0 += dilationW;
      }

      // local pointers
      scalar_t* input_ptr = input_data + c * input_height * input_width;

      // compute local max
      int64_t maxindex = ih0 * input_width + iw0;
      accscalar_t maxval = -std::numeric_limits<accscalar_t>::infinity();
      for (int64_t ih = ih0; ih < ih1; ih += dilationH) {
        for (int64_t iw = iw0; iw < iw1; iw += dilationW) {
          int64_t index = ih * input_width + iw;
          accscalar_t val = accscalar_t(input_ptr[index]);
          if ((val > maxval) || std::isnan(val)) {
            maxval = val;
            maxindex = index;
          }
        }
      }

      // set output to local max and store location of max
      output_data[i] = scalar_t(maxval);
      indices_data[i] = maxindex;

      // move on to next output index
      at::native::data_index_step(
          c, channels, oh, output_height, ow, output_width);
    }
  });

  if (!output_.is_contiguous()) {
    output_.copy_(output);
  }
  if (!indices_.is_contiguous()) {
    indices_.copy_(indices);
  }
}

template <typename scalar_t>
void cpu_max_pool_channels_last(
    const at::Tensor& output_,
    const at::Tensor& indices_,
    const at::Tensor& input_,
    int kW,
    int kH,
    int dW,
    int dH,
    int padW,
    int padH,
    int dilationW,
    int dilationH) {
  TORCH_CHECK(
      input_.ndimension() == 4,
      "max pooling with channels last format supports tensors with 4 dims");
  auto memory_format = at::MemoryFormat::ChannelsLast;
  auto input = input_.contiguous(memory_format);
  auto output = output_.contiguous(memory_format);
  auto indices = indices_.contiguous(memory_format);

  auto input_data = input.data_ptr<scalar_t>();
  auto output_data = output.data_ptr<scalar_t>();
  auto indices_data = indices.data_ptr<int64_t>();

  int64_t nbatch = input.size(0);
  int64_t channels = input.size(1);
  int64_t input_height = input.size(2);
  int64_t input_width = input.size(3);
  int64_t output_height = output.size(2);
  int64_t output_width = output.size(3);

  using Vec = at::vec::Vectorized<scalar_t>;
  using integer_t = at::vec::int_same_size_t<scalar_t>;
  using iVec = at::vec::Vectorized<integer_t>;
  // for the convience of vectorization, use integer of the same size of
  // scalar_t,
  //   e.g. int32_t for float, int64_t for double
  // need to make sure doesn't overflow
  TORCH_CHECK(
      input_height * input_width <= std::numeric_limits<integer_t>::max());

  // parallel on dim N, H, W
  at::parallel_for(
      0,
      nbatch * output_height * output_width,
      0,
      [&](int64_t begin, int64_t end) {
        int64_t n = 0;
        int64_t oh = 0;
        int64_t ow = 0;
        at::native::data_index_init(
            begin, n, nbatch, oh, output_height, ow, output_width);

        int64_t size = channels;
        int64_t len = size - (size % Vec::size());
        // temp buffer holding index with integer_t
        // NOLINTNEXTLINE(modernize-avoid-c-arrays,cppcoreguidelines-avoid-c-arrays)
        std::unique_ptr<integer_t[]> index_buffer(new integer_t[len]);

        for (int64_t i = begin; i < end; i++) {
          int64_t ih0 = oh * dH - padH;
          int64_t iw0 = ow * dW - padW;
          int64_t ih1 = std::min(ih0 + (kH - 1) * dilationH + 1, input_height);
          int64_t iw1 = std::min(iw0 + (kW - 1) * dilationW + 1, input_width);
          while (ih0 < 0) {
            ih0 += dilationH;
          }
          while (iw0 < 0) {
            iw0 += dilationW;
          }

          scalar_t* out = output_data + i * channels;
          int64_t* ind = indices_data + i * channels;

          // Pass I: init out lane
          iVec index0_vec = iVec(ih0 * input_width + iw0);
          Vec out_vec = Vec(-std::numeric_limits<scalar_t>::infinity());
          int64_t d1 = 0;
          for (; d1 < len; d1 += Vec::size()) {
            index0_vec.store(index_buffer.get() + d1);
            out_vec.store(out + d1);
          }
          for (; d1 < size; d1++) {
            ind[d1] = ih0 * input_width + iw0;
            out[d1] = -std::numeric_limits<scalar_t>::infinity();
          }
          // Pass II: compute local max
          for (int64_t ih = ih0; ih < ih1; ih += dilationH) {
            for (int64_t iw = iw0; iw < iw1; iw += dilationW) {
              scalar_t* in = input_data +
                  n * input_height * input_width * channels +
                  ih * input_width * channels + iw * channels;

              int64_t d2 = 0;
              for (; d2 < len; d2 += Vec::size()) {
                iVec index_vec = iVec(ih * input_width + iw);
                Vec val_vec = Vec::loadu(in + d2);
                iVec maxindex_vec = iVec::loadu(index_buffer.get() + d2);
                Vec maxval_vec = Vec::loadu(out + d2);

                // true = all ones, false = all zeros
                Vec mask = (val_vec > maxval_vec) | val_vec.isnan();
                iVec imask = at::vec::cast<integer_t>(mask);
                Vec out_vec = Vec::blendv(maxval_vec, val_vec, mask);
                iVec ind_vec = iVec::blendv(maxindex_vec, index_vec, imask);

                out_vec.store(out + d2);
                ind_vec.store(index_buffer.get() + d2);
              }
              for (; d2 < size; d2++) {
                int64_t index = ih * input_width + iw;
                scalar_t val = in[d2];
                int64_t maxindex = ind[d2];
                scalar_t maxval = out[d2];

                bool mask = (val > maxval) || std::isnan(val);
                out[d2] = mask ? val : maxval;
                ind[d2] = mask ? index : maxindex;
              }
            }
          }
          // convert indice data type
          at::vec::convert<integer_t, int64_t>(index_buffer.get(), ind, len);

          // move on to next output index
          at::native::data_index_step(
              n, nbatch, oh, output_height, ow, output_width);
        }
      });

  if (!output_.is_contiguous(memory_format)) {
    output_.copy_(output);
  }
  if (!indices_.is_contiguous(memory_format)) {
    indices_.copy_(indices);
  }
}

template <>
void cpu_max_pool_channels_last<at::BFloat16>(
    const at::Tensor& output_,
    const at::Tensor& indices_,
    const at::Tensor& input_,
    int kW,
    int kH,
    int dW,
    int dH,
    int padW,
    int padH,
    int dilationW,
    int dilationH) {
  TORCH_CHECK(
      input_.ndimension() == 4,
      "max pooling with channels last format supports tensors with 4 dims");
  auto memory_format = at::MemoryFormat::ChannelsLast;
  auto input = input_.contiguous(memory_format);
  auto output = output_.contiguous(memory_format);
  auto indices = indices_.contiguous(memory_format);

  auto input_data = input.data_ptr<at::BFloat16>();
  auto output_data = output.data_ptr<at::BFloat16>();
  auto indices_data = indices.data_ptr<int64_t>();

  int64_t nbatch = input.size(0);
  int64_t channels = input.size(1);
  int64_t input_height = input.size(2);
  int64_t input_width = input.size(3);
  int64_t output_height = output.size(2);
  int64_t output_width = output.size(3);

  using bVec = at::vec::Vectorized<at::BFloat16>;
  using fVec = at::vec::Vectorized<float>;
  using iVec = at::vec::Vectorized<int32_t>;
  // for the convience of vectorization, use int32_t instead of int64_t
  TORCH_CHECK(
      input_height * input_width <= std::numeric_limits<int32_t>::max());

  // parallel on dim N, H, W
  at::parallel_for(
      0,
      nbatch * output_height * output_width,
      0,
      [&](int64_t begin, int64_t end) {
        int64_t n = 0;
        int64_t oh = 0;
        int64_t ow = 0;
        at::native::data_index_init(
            begin, n, nbatch, oh, output_height, ow, output_width);

        int64_t size = channels;
        int64_t len = size - (size % bVec::size());
        // temp buffer holding index with integer_t
        std::unique_ptr<int32_t[]> index_buffer(new int32_t[len]);
        // temp buffer holding max value with float
        std::unique_ptr<float[]> max_arr(new float[size]);
        float* max = max_arr.get();

        for (int64_t i = begin; i < end; i++) {
          int64_t ih0 = oh * dH - padH;
          int64_t iw0 = ow * dW - padW;
          int64_t ih1 = std::min(ih0 + (kH - 1) * dilationH + 1, input_height);
          int64_t iw1 = std::min(iw0 + (kW - 1) * dilationW + 1, input_width);
          while (ih0 < 0) {
            ih0 += dilationH;
          }
          while (iw0 < 0) {
            iw0 += dilationW;
          }

          at::BFloat16* out = output_data + i * channels;
          int64_t* ind = indices_data + i * channels;

          // Pass I: init out lane
          iVec index0_ivec = iVec(ih0 * input_width + iw0);
          fVec max_fvec = fVec(-std::numeric_limits<float>::infinity());
          int64_t d1 = 0;
          for (; d1 < len; d1 += fVec::size()) {
            index0_ivec.store(index_buffer.get() + d1);
            max_fvec.store(max + d1);
          }
          for (; d1 < size; d1++) {
            ind[d1] = ih0 * input_width + iw0;
            max[d1] = -std::numeric_limits<float>::infinity();
          }
          // Pass II: compute local max
          for (int64_t ih = ih0; ih < ih1; ih += dilationH) {
            for (int64_t iw = iw0; iw < iw1; iw += dilationW) {
              at::BFloat16* in = input_data +
                  n * input_height * input_width * channels +
                  ih * input_width * channels + iw * channels;

              int64_t d2 = 0;
              for (; d2 < len; d2 += bVec::size()) {
                iVec index_ivec = iVec(ih * input_width + iw);
                bVec val_bvec = bVec::loadu(in + d2);
                fVec val_fvec0, val_fvec1;
                std::tie(val_fvec0, val_fvec1) =
                    convert_bfloat16_float(val_bvec);

                iVec maxindex_ivec0 = iVec::loadu(index_buffer.get() + d2);
                iVec maxindex_ivec1 =
                    iVec::loadu(index_buffer.get() + d2 + iVec::size());
                fVec maxval_fvec0 = fVec::loadu(max + d2);
                fVec maxval_fvec1 = fVec::loadu(max + d2 + fVec::size());

                // true = all ones, false = all zeros
                fVec mask0 = (val_fvec0 > maxval_fvec0) | val_fvec0.isnan();
                fVec mask1 = (val_fvec1 > maxval_fvec1) | val_fvec1.isnan();
                iVec imask0 = at::vec::cast<int32_t>(mask0);
                iVec imask1 = at::vec::cast<int32_t>(mask1);

                fVec max_fvec0 = fVec::blendv(maxval_fvec0, val_fvec0, mask0);
                fVec max_fvec1 = fVec::blendv(maxval_fvec1, val_fvec1, mask1);
                iVec ind_ivec0 =
                    iVec::blendv(maxindex_ivec0, index_ivec, imask0);
                iVec ind_ivec1 =
                    iVec::blendv(maxindex_ivec1, index_ivec, imask1);

                max_fvec0.store(max + d2);
                max_fvec1.store(max + d2 + fVec::size());
                ind_ivec0.store(index_buffer.get() + d2);
                ind_ivec1.store(index_buffer.get() + d2 + iVec::size());
              }
              for (; d2 < size; d2++) {
                int64_t index = ih * input_width + iw;
                float val = float(in[d2]);
                int64_t maxindex = ind[d2];
                float maxval = max[d2];

                bool mask = (val > maxval) || std::isnan(val);
                max[d2] = mask ? val : maxval;
                ind[d2] = mask ? index : maxindex;
              }
            }
          }
          // Pass III: convert max values from float to bfloat16
          int64_t d3 = 0;
          for (; d3 < len; d3 += bVec::size()) {
            fVec max_fvec0 = fVec::loadu(max + d3);
            fVec max_fvec1 = fVec::loadu(max + d3 + fVec::size());
            bVec max_bvec = convert_float_bfloat16(max_fvec0, max_fvec1);
            max_bvec.store(out + d3);
          }
          for (; d3 < size; d3++) {
            out[d3] = at::BFloat16(max[d3]);
          }
          // convert indice data type
          at::vec::convert<int32_t, int64_t>(index_buffer.get(), ind, len);

          // move on to next output index
          at::native::data_index_step(
              n, nbatch, oh, output_height, ow, output_width);
        }
      });

  if (!output_.is_contiguous(memory_format)) {
    output_.copy_(output);
  }
  if (!indices_.is_contiguous(memory_format)) {
    indices_.copy_(indices);
  }
}

template <typename scalar_t>
void cpu_max_pool_backward(
    const at::Tensor& grad_input_,
    const at::Tensor& grad_output_,
    const at::Tensor& indices_) {
  auto grad_output = grad_output_.contiguous();
  auto indices = indices_.contiguous();
  auto grad_input = grad_input_.contiguous();

  auto grad_output_data = grad_output.data_ptr<scalar_t>();
  auto indices_data = indices.data_ptr<int64_t>();
  auto grad_input_data = grad_input.data_ptr<scalar_t>();

  int64_t ndim = grad_output.ndimension();
  // treat batch size and channels as one dimension
  int64_t channels = ndim == 3 ? grad_output.size(0)
                               : grad_output.size(0) * grad_output.size(1);
  int64_t input_height = grad_input.size(-2);
  int64_t input_width = grad_input.size(-1);
  int64_t output_height = grad_output.size(-2);
  int64_t output_width = grad_output.size(-1);

  // parallel on dim of N, C
  at::parallel_for(0, channels, 0, [&](int64_t begin, int64_t end) {
    for (int64_t c = begin; c < end; c++) {
      scalar_t* grad_input_ptr =
          grad_input_data + c * input_height * input_width;
      scalar_t* grad_output_ptr =
          grad_output_data + c * output_height * output_width;
      int64_t* indices_ptr = indices_data + c * output_height * output_width;

      for (int64_t oh = 0; oh < output_height; oh++) {
        for (int64_t ow = 0; ow < output_width; ow++) {
          // retrieve position of max
          int64_t index = oh * output_width + ow;
          int64_t maxindex = indices_ptr[index];
          if (maxindex != -1) {
            // update gradient
            grad_input_ptr[maxindex] += grad_output_ptr[index];
          }
        }
      }
    }
  });

  if (!grad_input_.is_contiguous()) {
    grad_input_.copy_(grad_input);
  }
}

template <typename scalar_t>
void cpu_max_pool_backward_channels_last(
    const at::Tensor& grad_input_,
    const at::Tensor& grad_output_,
    const at::Tensor& indices_) {
  TORCH_CHECK(
      grad_output_.ndimension() == 4,
      "max pooling backward with "
      "channels last format supports "
      "tensors with 4 dims.");
  auto memory_format = at::MemoryFormat::ChannelsLast;
  auto grad_input = grad_input_.contiguous(memory_format);
  auto grad_output = grad_output_.contiguous(memory_format);
  auto indices = indices_.contiguous(memory_format);

  auto grad_input_data = grad_input.data_ptr<scalar_t>();
  auto grad_output_data = grad_output.data_ptr<scalar_t>();
  auto indices_data = indices.data_ptr<int64_t>();

  int64_t nbatch = grad_input.size(0);
  int64_t channels = grad_input.size(1);
  int64_t input_height = grad_input.size(2);
  int64_t input_width = grad_input.size(3);
  int64_t output_height = grad_output.size(2);
  int64_t output_width = grad_output.size(3);

  // parallel on dim N
  at::parallel_for(0, nbatch, 0, [&](int64_t begin, int64_t end) {
    for (int64_t n = begin; n < end; n++) {
      scalar_t* grad_input_ptr =
          grad_input_data + n * input_height * input_width * channels;
      scalar_t* grad_output_ptr =
          grad_output_data + n * output_height * output_width * channels;
      int64_t* indices_ptr =
          indices_data + n * output_height * output_width * channels;

      for (int64_t oh = 0; oh < output_height; oh++) {
        for (int64_t ow = 0; ow < output_width; ow++) {
          scalar_t* gout =
              grad_output_ptr + oh * output_width * channels + ow * channels;
          int64_t* ind =
              indices_ptr + oh * output_width * channels + ow * channels;
          // TODO: gcc vectorization
          for (int64_t c = 0; c < channels; c++) {
            int64_t maxindex = ind[c];
            if (maxindex != -1) {
              grad_input_ptr[maxindex * channels + c] += gout[c];
            }
          }
        }
      }
    }
  });

  if (!grad_input_.is_contiguous(memory_format)) {
    grad_input_.copy_(grad_input);
  }
}

void max_pool2d_kernel_impl(
    const at::Tensor& output,
    const at::Tensor& indices,
    const at::Tensor& input,
    int kW,
    int kH,
    int dW,
    int dH,
    int padW,
    int padH,
    int dilationW,
    int dilationH) {
  switch (input.suggest_memory_format()) {
    case at::MemoryFormat::Contiguous: {
      AT_DISPATCH_FLOATING_TYPES_AND(
          at::ScalarType::BFloat16, input.scalar_type(), "max_pool2d", [&] {
            if (input.scalar_type() == at::ScalarType::BFloat16) {
              cpu_max_pool<at::BFloat16, /*accscalar_t*/ float>(
                  output,
                  indices,
                  input,
                  kW,
                  kH,
                  dW,
                  dH,
                  padW,
                  padH,
                  dilationW,
                  dilationH);
            } else {
              cpu_max_pool<scalar_t, scalar_t>(
                  output,
                  indices,
                  input,
                  kW,
                  kH,
                  dW,
                  dH,
                  padW,
                  padH,
                  dilationW,
                  dilationH);
            }
          });
      break;
    }
    case at::MemoryFormat::ChannelsLast: {
      AT_DISPATCH_FLOATING_TYPES_AND(
          at::ScalarType::BFloat16,
          input.scalar_type(),
          "max_pool2d_channels_last",
          [&] {
            cpu_max_pool_channels_last<scalar_t>(
                output,
                indices,
                input,
                kW,
                kH,
                dW,
                dH,
                padW,
                padH,
                dilationW,
                dilationH);
          });
      break;
    }
    default:
      TORCH_CHECK(
          false,
          "Unsupported memory format. Supports only ChannelsLast, Contiguous");
  }
}

void max_pool2d_backward_kernel_impl(
    const at::Tensor& grad_input,
    const at::Tensor& grad_output,
    const at::Tensor& indices) {
  switch (grad_output.suggest_memory_format()) {
    case at::MemoryFormat::Contiguous: {
      AT_DISPATCH_FLOATING_TYPES_AND(
          at::ScalarType::BFloat16,
          grad_output.scalar_type(),
          "max_pool2d_backward",
          [&] {
            cpu_max_pool_backward<scalar_t>(grad_input, grad_output, indices);
          });
      break;
    }
    case at::MemoryFormat::ChannelsLast: {
      AT_DISPATCH_FLOATING_TYPES_AND(
          at::ScalarType::BFloat16,
          grad_output.scalar_type(),
          "max_pool2d_backward_channels_last",
          [&] {
            cpu_max_pool_backward_channels_last<scalar_t>(
                grad_input, grad_output, indices);
          });
      break;
    }
    default:
      TORCH_CHECK(
          false,
          "Unsupported memory format. Supports only ChannelsLast, Contiguous");
  }
}

std::tuple<at::Tensor, at::Tensor> max_pool2d_with_indices_out_cpu(
    const at::Tensor& input,
    at::IntArrayRef kernel_size,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::IntArrayRef dilation,
    bool ceil_mode) {
#if defined(IPEX_DISP_OP)
  printf("torch_ipex::max_pool2d_with_indices_out_cpu\n");
#endif
#if defined(IPEX_PROFILE_OP)
  RECORD_FUNCTION(
      "torch_ipex::max_pool2d_with_indices_out_cpu",
      std::vector<c10::IValue>({}));
#endif

  // #20866, #22032: Guarantee this for the official C++ API?
  TORCH_CHECK(
      kernel_size.size() == 1 || kernel_size.size() == 2,
      "max_pool2d: kernel_size must either be a single int, or a tuple "
      "of two ints")
  const int kH = safe_downcast<int, int64_t>(kernel_size[0]);
  const int kW = kernel_size.size() == 1
      ? kH
      : safe_downcast<int, int64_t>(kernel_size[1]);

  // NB: stride default is not expressible as an integer constant, so we accept
  // empty stride for this case
  TORCH_CHECK(
      stride.size() == 0 || stride.size() == 1 || stride.size() == 2,
      "max_pool2d: stride must either be omitted, a single int, or a "
      "tuple of two ints")
  const int dH = stride.empty() ? kH : safe_downcast<int, int64_t>(stride[0]);
  const int dW = stride.empty() ? kW
      : stride.size() == 1      ? dH
                                : safe_downcast<int, int64_t>(stride[1]);

  TORCH_CHECK(
      padding.size() == 1 || padding.size() == 2,
      "max_pool2d: padding must be either be a single int, or a tuple "
      "of two ints");
  const int padH = safe_downcast<int, int64_t>(padding[0]);
  const int padW =
      padding.size() == 1 ? padH : safe_downcast<int, int64_t>(padding[1]);

  TORCH_CHECK(
      dilation.size() == 1 || dilation.size() == 2,
      "max_pool2d: dilation must be either a single int, or a tuple of "
      "two ints");
  const int dilationH = safe_downcast<int, int64_t>(dilation[0]);
  const int dilationW = dilation.size() == 1
      ? dilationH
      : safe_downcast<int, int64_t>(dilation[1]);

  const auto memory_format = input.suggest_memory_format();
  if (memory_format == at::MemoryFormat::ChannelsLast) {
    TORCH_CHECK(
        input.ndimension() == 4,
        "non-empty 4D (batch mode) tensor "
        "expected for input with "
        "channels_last layout");
  } else if (memory_format == at::MemoryFormat::Contiguous) {
    TORCH_CHECK(
        (input.ndimension() == 3 || input.ndimension() == 4),
        "non-empty 3D or 4D (batch mode) tensor expected for input");
  } else {
    TORCH_CHECK(
        false,
        "Unsupport memory format. Supports only ChannelsLast, Contiguous");
  }

  /* sizes */
  const int64_t nbatch = input.ndimension() == 4 ? input.size(-4) : 1;
  const int64_t nInputPlane = input.size(-3);
  const int64_t inputHeight = input.size(-2);
  const int64_t inputWidth = input.size(-1);

  const int64_t outputHeight = at::native::pooling_output_shape<int64_t>(
      inputHeight, kH, padH, dH, dilationH, ceil_mode);
  const int64_t outputWidth = at::native::pooling_output_shape<int64_t>(
      inputWidth, kW, padW, dW, dilationW, ceil_mode);

  at::native::pool2d_shape_check(
      input,
      kH,
      kW,
      dH,
      dW,
      padH,
      padW,
      dilationH,
      dilationW,
      nInputPlane,
      inputHeight,
      inputWidth,
      outputHeight,
      outputWidth,
      memory_format);

  /* resize output and indices */
  at::Tensor output, indices;
  if (input.ndimension() == 3) {
    output = at::empty(
        {nInputPlane, outputHeight, outputWidth},
        input.options().memory_format(memory_format));
    /* indices will contain the locations for each output point */
    indices = at::empty(
        {nInputPlane, outputHeight, outputWidth},
        input.options().memory_format(memory_format).dtype(at::kLong));
  } else {
    output = at::empty(
        {nbatch, nInputPlane, outputHeight, outputWidth},
        input.options().memory_format(memory_format));
    /* indices will contain the locations for each output point */
    indices = at::empty(
        {nbatch, nInputPlane, outputHeight, outputWidth},
        input.options().memory_format(memory_format).dtype(at::kLong));
  }

  max_pool2d_kernel_impl(
      output, indices, input, kW, kH, dW, dH, padW, padH, dilationW, dilationH);

  return std::make_tuple(output, indices);
}

at::Tensor max_pool2d_with_indices_backward_out_cpu(
    const at::Tensor& gradOutput,
    const at::Tensor& input,
    at::IntArrayRef kernel_size,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::IntArrayRef dilation,
    bool ceil_mode,
    const at::Tensor& indices) {
#if defined(IPEX_DISP_OP)
  printf("torch_ipex::max_pool2d_with_indices_backward_out_cpu\n");
#endif
#if defined(IPEX_PROFILE_OP)
  RECORD_FUNCTION(
      "torch_ipex::max_pool2d_with_indices_backward_out_cpu",
      std::vector<c10::IValue>({}));
#endif

  // #20866, #22032: Guarantee this for the official C++ API?
  TORCH_CHECK(
      kernel_size.size() == 1 || kernel_size.size() == 2,
      "max_pool2d: kernel_size must either be a single int, or a tuple "
      "of two ints")
  const int kH = safe_downcast<int, int64_t>(kernel_size[0]);
  const int kW = kernel_size.size() == 1
      ? kH
      : safe_downcast<int, int64_t>(kernel_size[1]);

  // NB: stride default is not expressible as an integer constant, so we accept
  // empty stride for this case
  TORCH_CHECK(
      stride.size() == 0 || stride.size() == 1 || stride.size() == 2,
      "max_pool2d: stride must either be omitted, a single int, or a "
      "tuple of two ints")
  const int dH = stride.empty() ? kH : safe_downcast<int, int64_t>(stride[0]);
  const int dW = stride.empty() ? kW
      : stride.size() == 1      ? dH
                                : safe_downcast<int, int64_t>(stride[1]);

  TORCH_CHECK(
      padding.size() == 1 || padding.size() == 2,
      "max_pool2d: padding must be either be a single int, or a tuple "
      "of two ints");
  const int padH = safe_downcast<int, int64_t>(padding[0]);
  const int padW =
      padding.size() == 1 ? padH : safe_downcast<int, int64_t>(padding[1]);

  TORCH_CHECK(
      dilation.size() == 1 || dilation.size() == 2,
      "max_pool2d: dilation must be either a single int, or a tuple of "
      "two ints");
  const int dilationH = safe_downcast<int, int64_t>(dilation[0]);
  const int dilationW = dilation.size() == 1
      ? dilationH
      : safe_downcast<int, int64_t>(dilation[1]);

  TORCH_CHECK(
      input.dtype() == gradOutput.dtype(),
      "expected dtype ",
      input.dtype(),
      " for `gradOutput` but got dtype ",
      gradOutput.dtype());

  const auto memory_format = input.suggest_memory_format();
  if (memory_format == at::MemoryFormat::ChannelsLast) {
    TORCH_CHECK(
        input.ndimension() == 4,
        "non-empty 4D (batch mode) tensor "
        "expected for input with "
        "channels_last layout");
  } else if (memory_format == at::MemoryFormat::Contiguous) {
    TORCH_CHECK(
        (input.ndimension() == 3 || input.ndimension() == 4),
        "non-empty 3D or 4D (batch mode) tensor expected for input");
  } else {
    TORCH_CHECK(
        false,
        "Unsupport memory format. Supports only ChannelsLast, Contiguous");
  }

  /* sizes */
  const int64_t nbatch = input.ndimension() == 4 ? input.size(-4) : 1;
  const int64_t nInputPlane = input.size(-3);
  const int64_t inputHeight = input.size(-2);
  const int64_t inputWidth = input.size(-1);

  /* XXX preserve the existing shape check behavior */
  const int64_t outputHeight_for_shape_check =
      at::native::pooling_output_shape<int64_t>(
          inputHeight, kH, padH, dH, dilationH, ceil_mode);
  const int64_t outputWidth_for_shape_check =
      at::native::pooling_output_shape<int64_t>(
          inputWidth, kW, padW, dW, dilationW, ceil_mode);

  at::native::max_pool2d_backward_shape_check(
      input,
      gradOutput,
      indices,
      nbatch,
      kH,
      kW,
      dH,
      dW,
      padH,
      padW,
      dilationH,
      dilationW,
      nInputPlane,
      inputHeight,
      inputWidth,
      outputHeight_for_shape_check,
      outputWidth_for_shape_check,
      memory_format);

  // TODO: This is a workaround for the bug that 'at::zeros' does not recognize
  // the memory format tag.
  at::Tensor gradInput =
      at::empty(input.sizes(), input.options().memory_format(memory_format))
          .zero_();

  max_pool2d_backward_kernel_impl(gradInput, gradOutput, indices);

  return gradInput;
}

IPEX_TORCH_LIBRARY_IMPL(aten, CPU, m) {
  m.impl(
      TORCH_SELECTIVE_NAME("aten::max_pool2d_with_indices"),
      TORCH_FN((&torch_ipex::cpu::max_pool2d_with_indices_out_cpu)));
  m.impl(
      TORCH_SELECTIVE_NAME("aten::max_pool2d_with_indices_backward"),
      TORCH_FN((&torch_ipex::cpu::max_pool2d_with_indices_backward_out_cpu)));
}

} // namespace cpu
} // namespace torch_ipex
