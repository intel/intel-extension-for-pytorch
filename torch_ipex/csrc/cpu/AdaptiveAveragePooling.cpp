#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/NativeFunctions.h>
#include <ATen/Parallel.h>
#include <ATen/cpu/vec/vec.h>
#include <ATen/native/cpu/utils.h>
#include <ATen/record_function.h>

#include "AdaptiveAveragePooling.h"

#include "torch_ipex/csrc/library.h"

namespace torch_ipex {
namespace cpu {

template <typename scalar_t, typename accscalar_t>
void cpu_adaptive_avg_pool(
    at::Tensor& output_,
    const at::Tensor& input_,
    at::IntArrayRef output_size) {
  auto input = input_.contiguous();
  auto output = output_.contiguous();

  auto input_data = input.data_ptr<scalar_t>();
  auto output_data = output.data_ptr<scalar_t>();

  int64_t ndim = input.ndimension();
  // treat batch size and channels as one dimension
  int64_t channels = ndim == 3 ? input.size(0) : input.size(0) * input.size(1);
  int64_t input_height = input.size(-2);
  int64_t input_width = input.size(-1);
  int64_t output_height = output_size[0];
  int64_t output_width = output_size[1];

  // parallel on dim of N, C
  at::parallel_for(0, channels, 0, [&](int64_t begin, int64_t end) {
    for (int64_t c = begin; c < end; c++) {
      scalar_t* input_ptr = input_data + c * input_height * input_width;
      scalar_t* output_ptr = output_data + c * output_height * output_width;

      for (int64_t oh = 0; oh < output_height; oh++) {
        int64_t ih0 = start_index(oh, output_height, input_height);
        int64_t ih1 = end_index(oh, output_height, input_height);
        int64_t kh = ih1 - ih0;

        for (int64_t ow = 0; ow < output_width; ow++) {
          int64_t iw0 = start_index(ow, output_width, input_width);
          int64_t iw1 = end_index(ow, output_width, input_width);
          int64_t kw = iw1 - iw0;

          // compute local average
          accscalar_t sum = 0;
          for (int64_t ih = ih0; ih < ih1; ih++) {
            for (int64_t iw = iw0; iw < iw1; iw++) {
              sum += accscalar_t(input_ptr[ih * input_width + iw]);
            }
          }
          output_ptr[oh * output_width + ow] = scalar_t(sum / kh / kw);
        }
      }
    }
  });

  if (!output_.is_contiguous()) {
    output_.copy_(output);
  }
}

template <typename scalar_t>
void cpu_adaptive_avg_pool_channels_last(
    at::Tensor& output_,
    const at::Tensor& input_,
    at::IntArrayRef output_size) {
  auto memory_format = at::MemoryFormat::ChannelsLast;
  auto input = input_.contiguous(memory_format);
  auto output = output_.contiguous(memory_format);

  auto input_data = input.data_ptr<scalar_t>();
  auto output_data = output.data_ptr<scalar_t>();

  int64_t nbatch = input.size(0);
  int64_t channels = input.size(1);
  int64_t input_height = input.size(2);
  int64_t input_width = input.size(3);
  int64_t output_height = output_size[0];
  int64_t output_width = output_size[1];

  using Vec = at::vec::Vectorized<scalar_t>;
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

        for (int64_t i = begin; i < end; i++) {
          int64_t ih0 = start_index(oh, output_height, input_height);
          int64_t ih1 = end_index(oh, output_height, input_height);
          int64_t kh = ih1 - ih0;

          int64_t iw0 = start_index(ow, output_width, input_width);
          int64_t iw1 = end_index(ow, output_width, input_width);
          int64_t kw = iw1 - iw0;

          scalar_t* out = output_data + i * channels;
          int64_t size = channels;

          // Note: For oridinary usage scenario, each out lane should
          //   fit in L1 cache; otherwise consider block dim C.
          // Pass I: zero the out lane
          int64_t d1 = 0;
          for (; d1 < size - (size % Vec::size()); d1 += Vec::size()) {
            Vec out_vec = Vec(scalar_t(0));
            out_vec.store(out + d1);
          }
          for (; d1 < size; d1++) {
            out[d1] = scalar_t(0);
          }
          // Pass II: compute local sum
          for (int64_t ih = ih0; ih < ih1; ih++) {
            for (int64_t iw = iw0; iw < iw1; iw++) {
              scalar_t* in = input_data +
                  n * input_height * input_width * channels +
                  ih * input_width * channels + iw * channels;

              int64_t d2 = 0;
              for (; d2 < size - (size % Vec::size()); d2 += Vec::size()) {
                Vec out_vec = Vec::loadu(out + d2) + Vec::loadu(in + d2);
                out_vec.store(out + d2);
              }
              for (; d2 < size; d2++) {
                out[d2] += in[d2];
              }
            }
          }
          // Pass III: compute local average
          int64_t d3 = 0;
          for (; d3 < size - (size % Vec::size()); d3 += Vec::size()) {
            Vec out_vec = Vec::loadu(out + d3) / Vec(scalar_t(kh * kw));
            out_vec.store(out + d3);
          }
          for (; d3 < size; d3++) {
            out[d3] = out[d3] / kh / kw;
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

template <>
void cpu_adaptive_avg_pool_channels_last<at::BFloat16>(
    at::Tensor& output_,
    const at::Tensor& input_,
    at::IntArrayRef output_size) {
  auto memory_format = at::MemoryFormat::ChannelsLast;
  auto input = input_.contiguous(memory_format);
  auto output = output_.contiguous(memory_format);

  auto input_data = input.data_ptr<at::BFloat16>();
  auto output_data = output.data_ptr<at::BFloat16>();

  int64_t nbatch = input.size(0);
  int64_t channels = input.size(1);
  int64_t input_height = input.size(2);
  int64_t input_width = input.size(3);
  int64_t output_height = output_size[0];
  int64_t output_width = output_size[1];

  using bVec = at::vec::Vectorized<at::BFloat16>;
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
        // can't reuse output buffer to store sum since it is BFloat16
        std::unique_ptr<float[]> sum_arr(new float[channels]);
        float* sum = sum_arr.get();

        for (int64_t i = begin; i < end; i++) {
          int64_t ih0 = start_index(oh, output_height, input_height);
          int64_t ih1 = end_index(oh, output_height, input_height);
          int64_t kh = ih1 - ih0;

          int64_t iw0 = start_index(ow, output_width, input_width);
          int64_t iw1 = end_index(ow, output_width, input_width);
          int64_t kw = iw1 - iw0;

          at::BFloat16* out = output_data + i * channels;
          int64_t size = channels;

          // Pass I: zero the out lane
          int64_t d1 = 0;
          for (; d1 < size - (size % fVec::size()); d1 += fVec::size()) {
            fVec sum_fvec = fVec(float(0));
            sum_fvec.store(sum + d1);
          }
          for (; d1 < size; d1++) {
            sum[d1] = float(0);
          }
          // Pass II: compute local sum
          for (int64_t ih = ih0; ih < ih1; ih++) {
            for (int64_t iw = iw0; iw < iw1; iw++) {
              at::BFloat16* in = input_data +
                  n * input_height * input_width * channels +
                  ih * input_width * channels + iw * channels;

              int64_t d2 = 0;
              for (; d2 < size - (size % bVec::size()); d2 += bVec::size()) {
                bVec data_bvec = bVec::loadu(in + d2);
                fVec data_fvec0, data_fvec1;
                std::tie(data_fvec0, data_fvec1) =
                    convert_bfloat16_float(data_bvec);

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
            fVec out_fvec0 = fVec::loadu(sum + d3) / fVec(float(kh * kw));
            fVec out_fvec1 =
                fVec::loadu(sum + d3 + fVec::size()) / fVec(float(kh * kw));

            bVec out_bvec = convert_float_bfloat16(out_fvec0, out_fvec1);
            out_bvec.store(out + d3);
          }
          for (; d3 < size; d3++) {
            out[d3] = at::BFloat16(sum[d3] / kh / kw);
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

template <typename scalar_t>
void cpu_adaptive_avg_pool_backward(
    at::Tensor& grad_input_,
    const at::Tensor& grad_output_) {
  auto grad_output = grad_output_.contiguous();
  auto grad_input = grad_input_.contiguous();

  auto grad_output_data = grad_output.data_ptr<scalar_t>();
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

      for (int64_t oh = 0; oh < output_height; oh++) {
        int64_t ih0 = start_index(oh, output_height, input_height);
        int64_t ih1 = end_index(oh, output_height, input_height);
        int64_t kh = ih1 - ih0;

        for (int64_t ow = 0; ow < output_width; ow++) {
          int64_t iw0 = start_index(ow, output_width, input_width);
          int64_t iw1 = end_index(ow, output_width, input_width);
          int64_t kw = iw1 - iw0;

          scalar_t grad_delta =
              grad_output_ptr[oh * output_width + ow] / kh / kw;
          for (int64_t ih = ih0; ih < ih1; ih++) {
            for (int64_t iw = iw0; iw < iw1; iw++) {
              grad_input_ptr[ih * input_width + iw] += grad_delta;
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

template <typename scalar_t>
void cpu_adaptive_avg_pool_backward_channels_last(
    at::Tensor& grad_input_,
    const at::Tensor& grad_output_) {
  auto memory_format = at::MemoryFormat::ChannelsLast;
  auto grad_input = grad_input_.contiguous(memory_format);
  auto grad_output = grad_output_.contiguous(memory_format);

  auto grad_input_data = grad_input.data_ptr<scalar_t>();
  auto grad_output_data = grad_output.data_ptr<scalar_t>();

  int64_t nbatch = grad_input.size(0);
  int64_t channels = grad_input.size(1);
  int64_t input_height = grad_input.size(2);
  int64_t input_width = grad_input.size(3);
  int64_t output_height = grad_output.size(2);
  int64_t output_width = grad_output.size(3);

  using Vec = at::vec::Vectorized<scalar_t>;
  // parallel on dim N
  at::parallel_for(0, nbatch, 0, [&](int64_t begin, int64_t end) {
    for (int64_t n = begin; n < end; n++) {
      scalar_t* grad_input_ptr =
          grad_input_data + n * input_height * input_width * channels;
      scalar_t* grad_output_ptr =
          grad_output_data + n * output_height * output_width * channels;

      for (int64_t oh = 0; oh < output_height; oh++) {
        int64_t ih0 = start_index(oh, output_height, input_height);
        int64_t ih1 = end_index(oh, output_height, input_height);
        int64_t kh = ih1 - ih0;

        for (int64_t ow = 0; ow < output_width; ow++) {
          int64_t iw0 = start_index(ow, output_width, input_width);
          int64_t iw1 = end_index(ow, output_width, input_width);
          int64_t kw = iw1 - iw0;

          scalar_t* gout =
              grad_output_ptr + oh * output_width * channels + ow * channels;
          int64_t size = channels;
          for (int64_t ih = ih0; ih < ih1; ih++) {
            for (int64_t iw = iw0; iw < iw1; iw++) {
              scalar_t* gin =
                  grad_input_ptr + ih * input_width * channels + iw * channels;

              int64_t d = 0;
              for (; d < size - (size % Vec::size()); d += Vec::size()) {
                Vec gin_vec = Vec::loadu(gin + d) +
                    Vec::loadu(gout + d) / Vec(scalar_t(kh * kw));
                gin_vec.store(gin + d);
              }
              for (; d < size; d++) {
                gin[d] += gout[d] / kh / kw;
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

void adaptive_avg_pool2d_kernel_impl(
    at::Tensor& output,
    const at::Tensor& input,
    at::IntArrayRef output_size) {
  switch (input.suggest_memory_format()) {
    case at::MemoryFormat::Contiguous: {
      AT_DISPATCH_FLOATING_TYPES_AND(
          at::ScalarType::BFloat16,
          input.scalar_type(),
          "adaptive_avg_pool2d",
          [&] {
            if (input.scalar_type() == at::ScalarType::BFloat16) {
              cpu_adaptive_avg_pool<at::BFloat16, /*accscalar_t*/ float>(
                  output, input, output_size);
            } else {
              cpu_adaptive_avg_pool<scalar_t, scalar_t>(
                  output, input, output_size);
            }
          });
      break;
    }
    case at::MemoryFormat::ChannelsLast: {
      AT_DISPATCH_FLOATING_TYPES_AND(
          at::ScalarType::BFloat16,
          input.scalar_type(),
          "adaptive_avg_pool2d_channels_last",
          [&] {
            cpu_adaptive_avg_pool_channels_last<scalar_t>(
                output, input, output_size);
          });
      break;
    }
    default:
      TORCH_CHECK(
          false,
          "Unsupported memory format. Supports only ChannelsLast, Contiguous");
  }
}

void adaptive_avg_pool2d_backward_kernel_impl(
    at::Tensor& grad_input,
    const at::Tensor& grad_output) {
  switch (grad_output.suggest_memory_format()) {
    case at::MemoryFormat::Contiguous: {
      AT_DISPATCH_FLOATING_TYPES_AND(
          at::ScalarType::BFloat16,
          grad_output.scalar_type(),
          "adaptive_avg_pool2d_backward",
          [&] {
            cpu_adaptive_avg_pool_backward<scalar_t>(grad_input, grad_output);
          });
      break;
    }
    case at::MemoryFormat::ChannelsLast: {
      AT_DISPATCH_FLOATING_TYPES_AND(
          at::ScalarType::BFloat16,
          grad_output.scalar_type(),
          "adaptive_avg_pool2d_backward_channels_last",
          [&] {
            cpu_adaptive_avg_pool_backward_channels_last<scalar_t>(
                grad_input, grad_output);
          });
      break;
    }
    default:
      TORCH_CHECK(
          false,
          "Unsupported memory format. Supports only ChannelsLast, Contiguous");
  }
}

void adaptive_avg_pool2d_out_cpu_template(
    at::Tensor& output,
    at::Tensor const& input,
    at::IntArrayRef output_size) {
  TORCH_CHECK(
      output_size.size() == 2, "adaptive_avg_pool2d: output_size must be 2");
  int64_t ndim = input.ndimension();
  for (int64_t i = 1; i < ndim; i++) {
    TORCH_CHECK(
        input.size(i) > 0,
        "adaptive_avg_pool2d(): Expected input to have non-zero size for non-batch dimensions, "
        "but input has sizes ",
        input.sizes(),
        " with dimension ",
        i,
        " being "
        "empty");
  }

  TORCH_CHECK(
      (ndim == 3 || ndim == 4),
      "adaptive_avg_pool2d(): Expected 3D or 4D tensor, but got ",
      input.sizes());
  TORCH_CHECK(
      input.dtype() == output.dtype(),
      "expected dtype ",
      input.dtype(),
      " for `output` but got dtype ",
      output.dtype());

  int64_t channels = input.size(-3);
  int64_t output_height = output_size[0];
  int64_t output_width = output_size[1];

  if (ndim == 3) {
    output.resize_({channels, output_height, output_width});
  } else {
    int64_t nbatch = input.size(0);
    output.resize_(
        {nbatch, channels, output_height, output_width},
        input.suggest_memory_format());
  }

  if (output.numel() == 0) {
    return;
  }

  adaptive_avg_pool2d_kernel_impl(output, input, output_size);
}

at::Tensor& adaptive_avg_pool2d_backward_out_cpu_template(
    at::Tensor& grad_input,
    const at::Tensor& grad_output,
    const at::Tensor& input) {
  int64_t ndim = grad_output.ndimension();
  for (int64_t i = 1; i < ndim; i++) {
    TORCH_CHECK(
        grad_output.size(i) > 0,
        "adaptive_avg_pool2d_backward(): Expected grad_output to have non-zero size for non-batch dimensions, "
        "but grad_output has sizes ",
        grad_output.sizes(),
        " with dimension ",
        i,
        " being "
        "empty");
  }

  TORCH_CHECK(
      (ndim == 3 || ndim == 4),
      "adaptive_avg_pool2d_backward(): Expected 3D or 4D tensor, but got ",
      input.sizes());
  TORCH_CHECK(
      input.dtype() == grad_output.dtype(),
      "expected dtype ",
      input.dtype(),
      " for `grad_output` but got dtype ",
      grad_output.dtype());
  TORCH_CHECK(
      input.dtype() == grad_input.dtype(),
      "expected dtype ",
      input.dtype(),
      " for `grad_input` but got dtype ",
      grad_input.dtype());

  grad_input.resize_(input.sizes(), input.suggest_memory_format());
  grad_input.zero_();

  adaptive_avg_pool2d_backward_kernel_impl(grad_input, grad_output);
  return grad_input;
}

at::Tensor& adaptive_avg_pool2d_out_cpu(
    const at::Tensor& input,
    at::IntArrayRef output_size,
    at::Tensor& output) {
#if defined(IPEX_DISP_OP)
  printf("torch_ipex::adaptive_avg_pool2d_out_cpu\n");
#endif
#if defined(IPEX_PROFILE_OP)
  RECORD_FUNCTION(
      "torch_ipex::adaptive_avg_pool2d_out_cpu", std::vector<c10::IValue>({}));
#endif
  adaptive_avg_pool2d_out_cpu_template(output, input, output_size);
  return output;
}

at::Tensor adaptive_avg_pool2d_cpu(
    at::Tensor const& input,
    at::IntArrayRef output_size) {
#if defined(IPEX_DISP_OP)
  printf("torch_ipex::adaptive_avg_pool2d_cpu\n");
#endif
#if defined(IPEX_PROFILE_OP)
  RECORD_FUNCTION(
      "torch_ipex::adaptive_avg_pool2d_cpu", std::vector<c10::IValue>({}));
#endif
  auto output = at::empty({0}, input.options());
  adaptive_avg_pool2d_out_cpu_template(output, input, output_size);
  return output;
}

at::Tensor adaptive_avg_pool2d(
    at::Tensor const& input,
    at::IntArrayRef output_size) {
  TORCH_CHECK(
      output_size.size() == 2, "adaptive_avg_pool2d: output_size must be 2");

  if (input.is_mkldnn()) {
    return at::mkldnn_adaptive_avg_pool2d(input, output_size);
  }

  if (!input.is_quantized() && output_size[0] == 1 && output_size[1] == 1) {
// in this case, adaptive pooling is just computing mean over hw
// dimensions, which can be done more efficiently
#if defined(C10_MOBILE) && defined(USE_XNNPACK)
    if (xnnpack::use_global_average_pool(input)) {
      return xnnpack::global_average_pool(input);
    }
#endif

    at::Tensor out = input.mean({-1, -2}, /* keepdim = */ true);
    if (input.suggest_memory_format() == at::MemoryFormat::ChannelsLast) {
      // assert ndim == 4, since ndim = 3 doesn't give channels_last
      const int n = input.size(0);
      const int c = input.size(1);
      out.as_strided_({n, c, 1, 1}, {c, 1, c, c});
    }
    return out;
  } else {
    return _adaptive_avg_pool2d(input, output_size);
  }
}

at::Tensor& adaptive_avg_pool2d_backward_out_cpu(
    at::Tensor& grad_input,
    const at::Tensor& grad_output,
    const at::Tensor& input) {
#if defined(IPEX_DISP_OP)
  printf("torch_ipex::adaptive_avg_pool2d_backward_out_cpu\n");
#endif
#if defined(IPEX_PROFILE_OP)
  RECORD_FUNCTION(
      "torch_ipex::adaptive_avg_pool2d_backward_out_cpu",
      std::vector<c10::IValue>({}));
#endif
  adaptive_avg_pool2d_backward_out_cpu_template(grad_input, grad_output, input);
  return grad_input;
}

at::Tensor adaptive_avg_pool2d_backward_cpu(
    const at::Tensor& grad_output,
    const at::Tensor& input) {
#if defined(IPEX_DISP_OP)
  printf("torch_ipex::adaptive_avg_pool2d_backward_cpu\n");
#endif
#if defined(IPEX_PROFILE_OP)
  RECORD_FUNCTION(
      "torch_ipex::adaptive_avg_pool2d_backward_cpu",
      std::vector<c10::IValue>({}));
#endif
  auto grad_input = at::empty({0}, input.options());
  adaptive_avg_pool2d_backward_out_cpu_template(grad_input, grad_output, input);
  return grad_input;
}

IPEX_TORCH_LIBRARY_IMPL(aten, CPU, m) {
  m.impl(
      TORCH_SELECTIVE_NAME("aten::_adaptive_avg_pool2d"),
      TORCH_FN((&torch_ipex::cpu::adaptive_avg_pool2d_cpu)));
  m.impl(
      TORCH_SELECTIVE_NAME("aten::_adaptive_avg_pool2d_backward"),
      TORCH_FN((&torch_ipex::cpu::adaptive_avg_pool2d_backward_cpu)));
}

} // namespace cpu
} // namespace torch_ipex
