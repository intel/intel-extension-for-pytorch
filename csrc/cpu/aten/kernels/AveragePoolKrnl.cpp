#include <ATen/ATen.h>

#include <ATen/Dispatch.h>
#include <ATen/NativeFunctions.h>
#include <ATen/OpMathType.h>
#include <ATen/Parallel.h>
#include <ATen/cpu/vec/functional.h>
#include <ATen/cpu/vec/vec.h>
#include <ATen/native/Pool.h>
#include <ATen/native/cpu/utils.h>
#include <ATen/record_function.h>
#include <c10/util/irange.h>

#include "utils/library.h"

#include <aten/AveragePool.h>

namespace torch_ipex {
namespace cpu {

namespace {

template <typename scalar_t, typename accscalar_t, bool is_3d>
void cpu_avg_pool(
    const at::Tensor& output_,
    const at::Tensor& input_,
    int64_t kW,
    int64_t kH,
    int64_t kD,
    int64_t dW,
    int64_t dH,
    int64_t dD,
    int64_t padW,
    int64_t padH,
    int64_t padD,
    bool count_include_pad,
    c10::optional<int64_t> divisor_override) {
  auto input = input_.contiguous();
  auto output = output_.contiguous();

  auto input_data = input.data_ptr<scalar_t>();
  auto output_data = output.data_ptr<scalar_t>();

  // treat batch size and channels as one dimension
  //
  // AvgPool2d:
  //   ndim == 3: CHW
  //   ndim == 4: NCHW
  //
  // AvgPool3d:
  //   ndim == 4: CDHW
  //   ndim == 5: NCDHW

  int64_t numel = output.numel();
  int64_t ndim = input.ndimension();
  int64_t channels;
  if (is_3d) {
    channels = ndim == 4 ? input.size(0) : input.size(0) * input.size(1);
  } else {
    channels = ndim == 3 ? input.size(0) : input.size(0) * input.size(1);
  }
  int64_t input_depth = is_3d ? input.size(-3) : 1;
  int64_t input_height = input.size(-2);
  int64_t input_width = input.size(-1);
  int64_t output_depth = is_3d ? output.size(-3) : 1;
  int64_t output_height = output.size(-2);
  int64_t output_width = output.size(-1);

  // parallel on dim N, C
  at::parallel_for(0, channels, 0, [&](int64_t begin, int64_t end) {
    for (int64_t c = begin; c < end; c++) {
      scalar_t* input_ptr =
          input_data + c * input_depth * input_height * input_width;
      scalar_t* output_ptr =
          output_data + c * output_depth * output_height * output_width;
      for (int64_t od = 0; od < output_depth; od++) {
        int64_t id0 = od * dD - padD;
        int64_t id1 = std::min(id0 + kD, input_depth + padD);
        int64_t _id0 = std::max(id0, (int64_t)0);
        int64_t _id1 = std::min(id1, input_depth);
        for (int64_t oh = 0; oh < output_height; oh++) {
          int64_t ih0 = oh * dH - padH;
          int64_t ih1 = std::min(ih0 + kH, input_height + padH);
          int64_t _ih0 = std::max(ih0, (int64_t)0);
          int64_t _ih1 = std::min(ih1, input_height);
          for (int64_t ow = 0; ow < output_width; ow++) {
            int64_t iw0 = ow * dW - padW;
            int64_t iw1 = std::min(iw0 + kW, input_width + padW);
            int64_t _iw0 = std::max(iw0, (int64_t)0);
            int64_t _iw1 = std::min(iw1, input_width);

            int64_t index =
                od * output_height * output_width + oh * output_width + ow;
            output_ptr[index] = static_cast<scalar_t>(0);

            if (_id0 >= _id1 || _ih0 >= _ih1 || _iw0 >= _iw1) {
              continue;
            }

            int64_t divide_factor;
            if (divisor_override.has_value()) {
              divide_factor = divisor_override.value();
            } else {
              if (count_include_pad) {
                divide_factor = (id1 - id0) * (ih1 - ih0) * (iw1 - iw0);
              } else {
                divide_factor = (_id1 - _id0) * (_ih1 - _ih0) * (_iw1 - _iw0);
              }
            }

            accscalar_t sum = 0;
            for (int64_t id = _id0; id < _id1; id++) {
              for (int64_t ih = _ih0; ih < _ih1; ih++) {
                for (int64_t iw = _iw0; iw < _iw1; iw++) {
                  sum += input_ptr
                      [id * input_height * input_width + ih * input_width + iw];
                }
              }
            }
            output_ptr[index] = scalar_t(sum / divide_factor);
          }
        }
      }
    }
  });

  if (!output_.is_contiguous()) {
    output_.copy_(output);
  }
}

template <typename scalar_t, bool is_3d>
typename std::enable_if<
    std::is_same<scalar_t, at::opmath_type<scalar_t>>::value,
    void>::type
cpu_avg_pool_channels_last(
    const at::Tensor& output_,
    const at::Tensor& input_,
    int64_t kW,
    int64_t kH,
    int64_t kD,
    int64_t dW,
    int64_t dH,
    int64_t dD,
    int64_t padW,
    int64_t padH,
    int64_t padD,
    bool count_include_pad,
    c10::optional<int64_t> divisor_override) {
  int64_t ndim = input_.ndimension();
  if (is_3d) {
    TORCH_CHECK(
        ndim == 5,
        "AvgPool3d with channels last format supports tensors with 5 dims");
  } else {
    TORCH_CHECK(
        ndim == 4,
        "AvgPool2d with channels last format supports tensors with 4 dims");
  }
  auto memory_format =
      is_3d ? at::MemoryFormat::ChannelsLast3d : at::MemoryFormat::ChannelsLast;
  auto input = input_.contiguous(memory_format);
  auto output = output_.contiguous(memory_format);

  auto input_data = input.data_ptr<scalar_t>();
  auto output_data = output.data_ptr<scalar_t>();

  // MaxPool2d: NHWC
  // MaxPool3d: NDHWC
  int64_t nbatch = input.size(0);
  int64_t channels = input.size(1);
  int64_t input_depth = is_3d ? input.size(2) : 1;
  int64_t input_height = input.size(-2);
  int64_t input_width = input.size(-1);
  int64_t output_depth = is_3d ? output.size(2) : 1;
  int64_t output_height = output.size(-2);
  int64_t output_width = output.size(-1);

  using Vec = at::vec::Vectorized<scalar_t>;
  // parallel on dim N, {D}, H, W
  at::parallel_for(
      0,
      nbatch * output_depth * output_height * output_width,
      0,
      [&](int64_t begin, int64_t end) {
        int64_t n = 0;
        int64_t od = 0;
        int64_t oh = 0;
        int64_t ow = 0;
        at::native::data_index_init(
            begin,
            n,
            nbatch,
            od,
            output_depth,
            oh,
            output_height,
            ow,
            output_width);

        int64_t size = channels;
        int64_t len = size - (size % Vec::size());
        for (const auto i : c10::irange(begin, end)) {
          // compute the mean of the input image...
          int64_t id0 = od * dD - padD;
          int64_t ih0 = oh * dH - padH;
          int64_t iw0 = ow * dW - padW;
          int64_t id1 = std::min(id0 + kD, input_depth + padD);
          int64_t ih1 = std::min(ih0 + kH, input_height + padH);
          int64_t iw1 = std::min(iw0 + kW, input_width + padW);
          int64_t pool_size = (id1 - id0) * (ih1 - ih0) * (iw1 - iw0);
          id0 = std::max(id0, (int64_t)0);
          ih0 = std::max(ih0, (int64_t)0);
          iw0 = std::max(iw0, (int64_t)0);
          id1 = std::min(id1, input_depth);
          ih1 = std::min(ih1, input_height);
          iw1 = std::min(iw1, input_width);

          int64_t divide_factor;
          if (divisor_override.has_value()) {
            divide_factor = divisor_override.value();
          } else {
            if (count_include_pad) {
              divide_factor = pool_size;
            } else {
              divide_factor = (id1 - id0) * (ih1 - ih0) * (iw1 - iw0);
            }
          }

          scalar_t* out = output_data + i * channels;

          // Pass I: zero the out lane
          int64_t d1 = 0;
          for (; d1 < len; d1 += Vec::size()) {
            Vec out_vec = Vec(scalar_t(0));
            out_vec.store(out + d1);
          }
          for (; d1 < size; d1++) {
            out[d1] = scalar_t(0);
          }

          if (id0 >= id1 || ih0 >= ih1 || iw0 >= iw1) {
            // move on to next output index
            at::native::data_index_step(
                n,
                nbatch,
                od,
                output_depth,
                oh,
                output_height,
                ow,
                output_width);
            continue;
          }

          // Pass II: compute local sum
          for (int64_t id = id0; id < id1; id++) {
            for (int64_t ih = ih0; ih < ih1; ih++) {
              for (int64_t iw = iw0; iw < iw1; iw++) {
                scalar_t* in = input_data +
                    (n * input_depth * input_height * input_width +
                     id * input_height * input_width + ih * input_width + iw) *
                        channels;

                int64_t d2 = 0;
                for (; d2 < len; d2 += Vec::size()) {
                  Vec out_vec = Vec::loadu(out + d2) + Vec::loadu(in + d2);
                  out_vec.store(out + d2);
                }
                for (; d2 < size; d2++) {
                  out[d2] += in[d2];
                }
              }
            }
          }

          // Pass III: compute local average
          int64_t d3 = 0;
          for (; d3 < len; d3 += Vec::size()) {
            Vec out_vec = Vec::loadu(out + d3) / Vec(scalar_t(divide_factor));
            out_vec.store(out + d3);
          }
          for (; d3 < size; d3++) {
            out[d3] = out[d3] / divide_factor;
          }

          // move on to next output index
          at::native::data_index_step(
              n, nbatch, od, output_depth, oh, output_height, ow, output_width);
        }
      });

  if (!output_.is_contiguous(memory_format)) {
    output_.copy_(output);
  }
}

template <typename scalar_t, bool is_3d>
typename std::enable_if<
    !std::is_same<scalar_t, at::opmath_type<scalar_t>>::value,
    void>::type
cpu_avg_pool_channels_last(
    const at::Tensor& output_,
    const at::Tensor& input_,
    int64_t kW,
    int64_t kH,
    int64_t kD,
    int64_t dW,
    int64_t dH,
    int64_t dD,
    int64_t padW,
    int64_t padH,
    int64_t padD,
    bool count_include_pad,
    c10::optional<int64_t> divisor_override) {
  TORCH_CHECK(
      input_.ndimension() == 4,
      "average pooling with channels last format supports tensors with 4 dims");
  auto memory_format = at::MemoryFormat::ChannelsLast;
  auto input = input_.contiguous(memory_format);
  auto output = output_.contiguous(memory_format);

  auto input_data = input.data_ptr<scalar_t>();
  auto output_data = output.data_ptr<scalar_t>();

  int64_t nbatch = input.size(0);
  int64_t channels = input.size(1);
  int64_t input_height = input.size(2);
  int64_t input_width = input.size(3);
  int64_t output_height = output.size(2);
  int64_t output_width = output.size(3);

  using bVec = at::vec::Vectorized<scalar_t>;
  using fVec = at::vec::Vectorized<float>;
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

        // temp buffer for sum, use float as accumulation type
        // can't reuse output buffer to store sum since it is BFloat16/Half
        std::unique_ptr<float[]> sum_arr(new float[channels]);
        float* sum = sum_arr.get();

        int64_t size = channels;
        for (const auto i : c10::irange(begin, end)) {
          // compute the mean of the input image...
          int64_t ih0 = oh * dH - padH;
          int64_t iw0 = ow * dW - padW;
          int64_t ih1 = std::min(ih0 + kH, input_height + padH);
          int64_t iw1 = std::min(iw0 + kW, input_width + padW);
          int64_t pool_size = (ih1 - ih0) * (iw1 - iw0);
          ih0 = std::max(ih0, (int64_t)0);
          iw0 = std::max(iw0, (int64_t)0);
          ih1 = std::min(ih1, input_height);
          iw1 = std::min(iw1, input_width);

          int64_t divide_factor;
          if (divisor_override.has_value()) {
            divide_factor = divisor_override.value();
          } else {
            if (count_include_pad) {
              divide_factor = pool_size;
            } else {
              divide_factor = (ih1 - ih0) * (iw1 - iw0);
            }
          }

          scalar_t* out = output_data + i * channels;

          // Pass I: zero the out lane
          int64_t d1 = 0;
          for (; d1 < size - (size % fVec::size()); d1 += fVec::size()) {
            fVec sum_fvec = fVec(float(0));
            sum_fvec.store(sum + d1);
          }
          for (; d1 < size; d1++) {
            sum[d1] = float(0);
          }

          if (ih0 >= ih1 || iw0 >= iw1) {
            // since we are not directly using output as the accumulation
            // buffer, in case the kernel window is out of range, need to zero
            // the output buffer here.
            for (int64_t k = 0; k < size; k++) {
              out[k] = 0;
            }
            // move on to next output index
            at::native::data_index_step(
                n, nbatch, oh, output_height, ow, output_width);
            continue;
          }

          // Pass II: compute local sum
          for (const auto ih : c10::irange(ih0, ih1)) {
            for (const auto iw : c10::irange(iw0, iw1)) {
              scalar_t* in = input_data +
                  n * input_height * input_width * channels +
                  ih * input_width * channels + iw * channels;

              int64_t d2 = 0;
              for (; d2 < size - (size % bVec::size()); d2 += bVec::size()) {
                bVec data_bvec = bVec::loadu(in + d2);
                fVec data_fvec0, data_fvec1;
                std::tie(data_fvec0, data_fvec1) =
                    at::vec::convert_to_float<scalar_t>(data_bvec);

                fVec sum_fvec0 = fVec::loadu(sum + d2) + data_fvec0;
                fVec sum_fvec1 =
                    fVec::loadu(sum + d2 + fVec::size()) + data_fvec1;
                sum_fvec0.store(sum + d2);
                sum_fvec1.store(sum + d2 + fVec::size());
              }
              for (; d2 < size; d2++) {
                sum[d2] += float(in[d2]);
              }
            }
          }

          // Pass III: compute local average
          int64_t d3 = 0;
          for (; d3 < size - (size % bVec::size()); d3 += bVec::size()) {
            fVec out_fvec0 = fVec::loadu(sum + d3) / fVec(float(divide_factor));
            fVec out_fvec1 = fVec::loadu(sum + d3 + fVec::size()) /
                fVec(float(divide_factor));

            bVec out_bvec =
                at::vec::convert_from_float<scalar_t>(out_fvec0, out_fvec1);
            out_bvec.store(out + d3);
          }
          for (; d3 < size; d3++) {
            out[d3] = scalar_t(sum[d3] / divide_factor);
          }

          // move on to next output index
          at::native::data_index_step(
              n, nbatch, oh, output_height, ow, output_width);
        }
      });

  if (!output_.is_contiguous(memory_format)) {
    output_.copy_(output);
  }
}

template <typename scalar_t, bool is_3d>
void cpu_avg_pool_backward(
    const at::Tensor& grad_input_,
    const at::Tensor& grad_output_,
    int kW,
    int kH,
    int kD,
    int dW,
    int dH,
    int dD,
    int padW,
    int padH,
    int padD,
    bool count_include_pad,
    c10::optional<int64_t> divisor_override) {
  auto grad_output = grad_output_.contiguous();
  auto grad_input = grad_input_.contiguous();

  auto grad_output_data = grad_output.data_ptr<scalar_t>();
  auto grad_input_data = grad_input.data_ptr<scalar_t>();

  // treat batch size and channels as one dimension
  //
  // MaxPool2d:
  //   ndim == 3: CHW
  //   ndim == 4: NCHW
  //
  // MaxPool3d:
  //   ndim == 4: CDHW
  //   ndim == 5: NCDHW
  //
  int64_t ndim = grad_output.ndimension();
  int64_t channels;
  if (is_3d) {
    channels = ndim == 4 ? grad_output.size(0)
                         : grad_output.size(0) * grad_output.size(1);
  } else {
    channels = ndim == 3 ? grad_output.size(0)
                         : grad_output.size(0) * grad_output.size(1);
  }
  int64_t input_depth = is_3d ? grad_input.size(-3) : 1;
  int64_t input_height = grad_input.size(-2);
  int64_t input_width = grad_input.size(-1);
  int64_t output_depth = is_3d ? grad_output.size(-3) : 1;
  int64_t output_height = grad_output.size(-2);
  int64_t output_width = grad_output.size(-1);

  // parallel on dim of N, C
  at::parallel_for(0, channels, 0, [&](int64_t begin, int64_t end) {
    for (const auto c : c10::irange(begin, end)) {
      scalar_t* grad_input_ptr =
          grad_input_data + c * input_depth * input_height * input_width;
      scalar_t* grad_output_ptr =
          grad_output_data + c * output_depth * output_height * output_width;

      for (int64_t od = 0; od < output_depth; od++) {
        int64_t id0 = od * dD - padD;
        int64_t id1 = std::min(id0 + kD, input_depth + padD);
        int64_t _id0 = std::max(id0, (int64_t)0);
        int64_t _id1 = std::min(id1, input_depth);

        for (int64_t oh = 0; oh < output_height; oh++) {
          int64_t ih0 = oh * dH - padH;
          int64_t ih1 = std::min(ih0 + kH, input_height + padH);
          int64_t _ih0 = std::max(ih0, (int64_t)0);
          int64_t _ih1 = std::min(ih1, input_height);

          for (int64_t ow = 0; ow < output_width; ow++) {
            int64_t iw0 = ow * dW - padW;
            int64_t iw1 = std::min(iw0 + kW, input_width + padW);
            int64_t _iw0 = std::max(iw0, (int64_t)0);
            int64_t _iw1 = std::min(iw1, input_width);

            int64_t divide_factor;
            if (divisor_override.has_value()) {
              divide_factor = divisor_override.value();
            } else {
              if (count_include_pad) {
                divide_factor = (id1 - id0) * (ih1 - ih0) * (iw1 - iw0);
              } else {
                divide_factor = (_id1 - _id0) * (_ih1 - _ih0) * (_iw1 - _iw0);
              }
            }
            int64_t output_index =
                od * output_height * output_width + oh * output_width + ow;
            scalar_t grad_delta = grad_output_ptr[output_index] / divide_factor;
            for (int64_t id = _id0; id < _id1; id++) {
              for (int64_t ih = _ih0; ih < _ih1; ih++) {
                for (int64_t iw = _iw0; iw < _iw1; iw++) {
                  int64_t input_index =
                      id * input_height * input_width + ih * input_width + iw;
                  grad_input_ptr[input_index] += grad_delta;
                }
              }
            }
          }
        }
      }
    }
  });

  if (!grad_input_.is_contiguous()) {
    grad_input_.copy_(grad_input);
  }
}

template <typename scalar_t, bool is_3d>
void cpu_avg_pool_backward_channels_last(
    const at::Tensor& grad_input_,
    const at::Tensor& grad_output_,
    int kW,
    int kH,
    int kD,
    int dW,
    int dH,
    int dD,
    int padW,
    int padH,
    int padD,
    bool count_include_pad,
    c10::optional<int64_t> divisor_override) {
  auto memory_format =
      is_3d ? at::MemoryFormat::ChannelsLast3d : at::MemoryFormat::ChannelsLast;
  auto grad_input = grad_input_.contiguous(memory_format);
  auto grad_output = grad_output_.contiguous(memory_format);

  auto grad_input_data = grad_input.data_ptr<scalar_t>();
  auto grad_output_data = grad_output.data_ptr<scalar_t>();

  // MaxPool2d: NHWC
  // MaxPool3d: NDHWC
  int64_t nbatch = grad_input.size(0);
  int64_t channels = grad_input.size(1);
  int64_t input_depth = is_3d ? grad_input.size(2) : 1;
  int64_t input_height = grad_input.size(-2);
  int64_t input_width = grad_input.size(-1);
  int64_t output_depth = is_3d ? grad_output.size(2) : 1;
  int64_t output_height = grad_output.size(-2);
  int64_t output_width = grad_output.size(-1);

  using Vec = at::vec::Vectorized<scalar_t>;
  // parallel on dim N
  at::parallel_for(0, nbatch, 0, [&](int64_t begin, int64_t end) {
    for (const auto n : c10::irange(begin, end)) {
      scalar_t* grad_input_ptr = grad_input_data +
          n * input_depth * input_height * input_width * channels;
      scalar_t* grad_output_ptr = grad_output_data +
          n * output_depth * output_height * output_width * channels;

      for (int64_t od = 0; od < output_depth; od++) {
        for (int64_t oh = 0; oh < output_height; oh++) {
          for (int64_t ow = 0; ow < output_width; ow++) {
            int64_t id0 = od * dD - padD;
            int64_t ih0 = oh * dH - padH;
            int64_t iw0 = ow * dW - padW;
            int64_t id1 = std::min(id0 + kD, input_depth + padD);
            int64_t ih1 = std::min(ih0 + kH, input_height + padH);
            int64_t iw1 = std::min(iw0 + kW, input_width + padW);
            int64_t pool_size = (id1 - id0) * (ih1 - ih0) * (iw1 - iw0);
            id0 = std::max(id0, (int64_t)0);
            ih0 = std::max(ih0, (int64_t)0);
            iw0 = std::max(iw0, (int64_t)0);
            id1 = std::min(id1, input_depth);
            ih1 = std::min(ih1, input_height);
            iw1 = std::min(iw1, input_width);

            int64_t divide_factor;
            if (divisor_override.has_value()) {
              divide_factor = divisor_override.value();
            } else {
              if (count_include_pad) {
                divide_factor = pool_size;
              } else {
                divide_factor = (id1 - id0) * (ih1 - ih0) * (iw1 - iw0);
              }
            }
            scalar_t* gout = grad_output_ptr +
                (od * output_height * output_width + oh * output_width + ow) *
                    channels;
            int64_t size = channels;
            int64_t len = size - (size % Vec::size());
            for (int64_t id = id0; id < id1; id++) {
              for (int64_t ih = ih0; ih < ih1; ih++) {
                for (int64_t iw = iw0; iw < iw1; iw++) {
                  scalar_t* gin = grad_input_ptr +
                      (id * input_height * input_width + ih * input_width +
                       iw) *
                          channels;

                  int64_t d = 0;
                  for (; d < len; d += Vec::size()) {
                    Vec gin_vec = Vec::loadu(gin + d) +
                        Vec::loadu(gout + d) / Vec(scalar_t(divide_factor));
                    gin_vec.store(gin + d);
                  }
                  for (; d < size; d++) {
                    gin[d] += gout[d] / divide_factor;
                  }
                }
              }
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

void avg_pool2d_kernel_impl(
    const at::Tensor& output,
    const at::Tensor& input,
    int64_t kW,
    int64_t kH,
    int64_t dW,
    int64_t dH,
    int64_t padW,
    int64_t padH,
    bool count_include_pad,
    c10::optional<int64_t> divisor_override) {
  switch (input.suggest_memory_format()) {
    case at::MemoryFormat::Contiguous: {
      AT_DISPATCH_FLOATING_TYPES_AND3(
          at::ScalarType::Long,
          at::ScalarType::BFloat16,
          at::ScalarType::Half,
          input.scalar_type(),
          "avg_pool2d",
          [&] {
            if (at::isReducedFloatingType(input.scalar_type())) {
              cpu_avg_pool<
                  scalar_t,
                  /*accscalar_t*/ float,
                  /* is_3d */ false>(
                  output,
                  input,
                  kW,
                  kH,
                  /* kD */ 1,
                  dW,
                  dH,
                  /* dD */ 1,
                  padW,
                  padH,
                  /* padD */ 0,
                  count_include_pad,
                  divisor_override);
            } else if (input.scalar_type() == at::ScalarType::Long) {
              cpu_avg_pool<
                  long,
                  /*accscalar_t*/ long,
                  /* is_3d */ false>(
                  output,
                  input,
                  kW,
                  kH,
                  /* kD */ 1,
                  dW,
                  dH,
                  /* dD */ 1,
                  padW,
                  padH,
                  /* padD */ 0,
                  count_include_pad,
                  divisor_override);
            } else if (input.scalar_type() == at::ScalarType::Double) {
              cpu_avg_pool<
                  double,
                  /*accscalar_t*/ double,
                  /* is_3d */ false>(
                  output,
                  input,
                  kW,
                  kH,
                  /* kD */ 1,
                  dW,
                  dH,
                  /* dD */ 1,
                  padW,
                  padH,
                  /* padD */ 0,
                  count_include_pad,
                  divisor_override);
            } else if (input.scalar_type() == at::ScalarType::Float) {
              cpu_avg_pool<float, float, /* is_3d */ false>(
                  output,
                  input,
                  kW,
                  kH,
                  /* kD */ 1,
                  dW,
                  dH,
                  /* dD */ 1,
                  padW,
                  padH,
                  /* padD */ 0,
                  count_include_pad,
                  divisor_override);
            }
          });
      break;
    }
    case at::MemoryFormat::ChannelsLast: {
      AT_DISPATCH_FLOATING_TYPES_AND3(
          at::ScalarType::Long,
          at::ScalarType::BFloat16,
          at::ScalarType::Half,
          input.scalar_type(),
          "avg_pool2d_channels_last",
          [&] {
            cpu_avg_pool_channels_last<scalar_t, /* is_3d */ false>(
                output,
                input,
                kW,
                kH,
                /* kD */ 1,
                dW,
                dH,
                /* dD */ 1,
                padW,
                padH,
                /* padD */ 0,
                count_include_pad,
                divisor_override);
          });
      break;
    }
    default:
      TORCH_CHECK(
          false,
          "Unsupported memory format. Supports only ChannelsLast, Contiguous");
  }
}

void avg_pool2d_backward_kernel_impl(
    const at::Tensor& grad_input,
    const at::Tensor& grad_output,
    int kW,
    int kH,
    int dW,
    int dH,
    int padW,
    int padH,
    bool count_include_pad,
    c10::optional<int64_t> divisor_override) {
  switch (grad_output.suggest_memory_format()) {
    case at::MemoryFormat::Contiguous: {
      AT_DISPATCH_FLOATING_TYPES_AND2(
          at::ScalarType::BFloat16,
          at::ScalarType::Half,
          grad_output.scalar_type(),
          "avg_pool2d_backward",
          [&] {
            cpu_avg_pool_backward<scalar_t, /* is_3d */ false>(
                grad_input,
                grad_output,
                kW,
                kH,
                /* kD */ 1,
                dW,
                dH,
                /* dD */ 1,
                padW,
                padH,
                /* padD */ 0,
                count_include_pad,
                divisor_override);
          });
      break;
    }
    case at::MemoryFormat::ChannelsLast: {
      AT_DISPATCH_FLOATING_TYPES_AND2(
          at::ScalarType::BFloat16,
          at::ScalarType::Half,
          grad_output.scalar_type(),
          "avg_pool2d_backward_channels_last",
          [&] {
            cpu_avg_pool_backward_channels_last<scalar_t, /* is_3d */ false>(
                grad_input,
                grad_output,
                kW,
                kH,
                /* kD */ 1,
                dW,
                dH,
                /* dD */ 1,
                padW,
                padH,
                /* padD */ 0,
                count_include_pad,
                divisor_override);
          });
      break;
    }
    default:
      TORCH_CHECK(
          false,
          "Unsupported memory format. Supports only ChannelsLast, Contiguous");
  }
}

void avg_pool3d_kernel_impl(
    const at::Tensor& output,
    const at::Tensor& input,
    int kW,
    int kH,
    int kD,
    int dW,
    int dH,
    int dD,
    int padW,
    int padH,
    int padD,
    bool count_include_pad,
    c10::optional<int64_t> divisor_override) {
  switch (input.suggest_memory_format()) {
    case at::MemoryFormat::Contiguous: {
      AT_DISPATCH_FLOATING_TYPES_AND(
          at::ScalarType::Long, input.scalar_type(), "avg_pool3d", [&] {
            if (input.scalar_type() == at::ScalarType::Float) {
              cpu_avg_pool<float, /*accscalar_t*/ float, /* is_3d */ true>(
                  output,
                  input,
                  kW,
                  kH,
                  kD,
                  dW,
                  dH,
                  dD,
                  padW,
                  padH,
                  padD,
                  count_include_pad,
                  divisor_override);
            } else if (input.scalar_type() == at::ScalarType::Double) {
              cpu_avg_pool<double, /*accscalar_t*/ double, /* is_3d */ true>(
                  output,
                  input,
                  kW,
                  kH,
                  kD,
                  dW,
                  dH,
                  dD,
                  padW,
                  padH,
                  padD,
                  count_include_pad,
                  divisor_override);
            } else if (input.scalar_type() == at::ScalarType::Long) {
              cpu_avg_pool<long, /*accscalar_t*/ long, /* is_3d */ true>(
                  output,
                  input,
                  kW,
                  kH,
                  kD,
                  dW,
                  dH,
                  dD,
                  padW,
                  padH,
                  padD,
                  count_include_pad,
                  divisor_override);
            }
          });
      break;
    }
    case at::MemoryFormat::ChannelsLast3d: {
      AT_DISPATCH_FLOATING_TYPES_AND(
          at::ScalarType::Long,
          input.scalar_type(),
          "avg_pool3d_channels_last",
          [&] {
            cpu_avg_pool_channels_last<scalar_t, /* is_3d */ true>(
                output,
                input,
                kW,
                kH,
                kD,
                dW,
                dH,
                dD,
                padW,
                padH,
                padD,
                count_include_pad,
                divisor_override);
          });
      break;
    }
    default:
      TORCH_CHECK(
          false,
          "Unsupported memory format. Supports only ChannelsLast3d, Contiguous");
  }
}

void avg_pool3d_backward_kernel_impl(
    const at::Tensor& grad_input,
    const at::Tensor& grad_output,
    int kW,
    int kH,
    int kD,
    int dW,
    int dH,
    int dD,
    int padW,
    int padH,
    int padD,
    bool count_include_pad,
    c10::optional<int64_t> divisor_override) {
  switch (grad_output.suggest_memory_format()) {
    case at::MemoryFormat::Contiguous: {
      AT_DISPATCH_FLOATING_TYPES(
          grad_output.scalar_type(), "avg_pool3d_backward", [&] {
            cpu_avg_pool_backward<scalar_t, /* is_3d */ true>(
                grad_input,
                grad_output,
                kW,
                kH,
                kD,
                dW,
                dH,
                dD,
                padW,
                padH,
                padD,
                count_include_pad,
                divisor_override);
          });
      break;
    }
    case at::MemoryFormat::ChannelsLast3d: {
      AT_DISPATCH_FLOATING_TYPES(
          grad_output.scalar_type(), "avg_pool3d_backward_channels_last", [&] {
            cpu_avg_pool_backward_channels_last<scalar_t, /* is_3d */ true>(
                grad_input,
                grad_output,
                kW,
                kH,
                kD,
                dW,
                dH,
                dD,
                padW,
                padH,
                padD,
                count_include_pad,
                divisor_override);
          });
      break;
    }
    default:
      TORCH_CHECK(
          false,
          "Unsupported memory format. Supports only ChannelsLast3d, Contiguous");
  }
}

} // anonymous namespace

IPEX_REGISTER_DISPATCH(avg_pool2d_kernel_stub, &avg_pool2d_kernel_impl);
IPEX_REGISTER_DISPATCH(
    avg_pool2d_backward_kernel_stub,
    &avg_pool2d_backward_kernel_impl);

IPEX_REGISTER_DISPATCH(avg_pool3d_kernel_stub, &avg_pool3d_kernel_impl);
IPEX_REGISTER_DISPATCH(
    avg_pool3d_backward_kernel_stub,
    &avg_pool3d_backward_kernel_impl);

} // namespace cpu
} // namespace torch_ipex