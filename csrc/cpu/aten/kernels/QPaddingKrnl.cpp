// This file can be removed after
// https://github.com/pytorch/pytorch/pull/100789 landed.
#include <ATen/core/Tensor.h>

#include <ATen/Dispatch.h>
#include <ATen/Parallel.h>
#include <ATen/cpu/vec/vec.h>
#include <ATen/native/cpu/utils.h>
#include <c10/util/irange.h>

#include "utils/library.h"

#include <aten/QPadding.h>

namespace torch_ipex {
namespace cpu {

namespace {

using namespace at::vec;

struct PaddingParams {
  int ndim;
  int64_t nbatch;
  int64_t channels;

  // use vectorized logic on width when output index is in [pad, input_witdh +
  // pad), applies only to Channels First format when pad_l and pad_r are both
  // positive.
  bool is_padding_positive_width;

  c10::SmallVector<int64_t, 3u> ishape;
  c10::SmallVector<int64_t, 3u> oshape;
  c10::SmallVector<int64_t, 3u> pads;
  c10::SmallVector<int64_t, 3u> offsets;

  PaddingParams(
      const at::Tensor& input,
      const at::Tensor& output,
      c10::IntArrayRef padding) {
    ndim = padding.size() / 2;

    bool is_batch = input.dim() == ndim + 2;
    nbatch = is_batch ? input.size(0) : 1;
    channels = is_batch ? input.size(1) : input.size(0);

    is_padding_positive_width = padding[0] >= 0 && padding[1] >= 0;

    // handle sizes with batch-mode and non-batch-mode
    int ind = is_batch ? 2 : 1;
    for (const auto d : c10::irange(ndim)) {
      ishape.emplace_back(input.size(ind + d));
      oshape.emplace_back(output.size(ind + d));
    }

    // padding is organized in order of:
    //   { left, right, top, bottom, front, back }
    //
    // re-organize into order of:
    //   { depth, height, width}
    //
    if (ndim == 1) {
      pads.emplace_back(padding[0]);
    } else if (ndim == 2) {
      pads.emplace_back(padding[2]);
      pads.emplace_back(padding[0]);
    } else {
      pads.emplace_back(padding[4]);
      pads.emplace_back(padding[2]);
      pads.emplace_back(padding[0]);
    }
    for (const auto d : c10::irange(ndim)) {
      int64_t pad = pads[d];
      auto i_start = std::max(int64_t(0), -pad);
      auto o_start = std::max(int64_t(0), pad);
      offsets.emplace_back(i_start - o_start);
    }
  };
};

struct ReplicationPad {
  static int64_t index(int64_t j, int64_t size, int64_t pad, int64_t offset) {
    int64_t i;
    if (j < pad) {
      i = pad;
    } else if (j >= pad && j < size + pad) {
      i = j;
    } else {
      i = size + pad - 1;
    }
    return i + offset;
  }
};

struct ReflectionPad {
  static int64_t index(int64_t j, int64_t size, int64_t pad, int64_t offset) {
    int64_t i;
    if (j < pad) {
      i = pad * 2 - j;
    } else if (j >= pad && j < size + pad) {
      i = j;
    } else {
      i = (size + pad - 1) * 2 - j;
    }
    return i + offset;
  }
};

template <typename scalar_t>
static inline void copy_stub(scalar_t* out, const scalar_t* in, int64_t size) {
  using Vec = Vectorized<scalar_t>;
  int64_t d = 0;
  for (; d < size - (size % Vec::size()); d += Vec::size()) {
    Vec in_vec = Vec::loadu(in + d);
    in_vec.store(out + d);
  }
#if !defined(_MSC_VER) && !defined(COMPILING_FOR_MIN_SIZE)
#pragma unroll
#endif
  for (; d < size; d++) {
    out[d] = in[d];
  }
}

template <typename scalar_t, typename PaddingType>
void cpu_padding(
    const at::Tensor& output_,
    const at::Tensor& input_,
    PaddingParams& p) {
  auto input = input_.contiguous();
  auto output = output_.contiguous();

  auto input_data = input.data_ptr<scalar_t>();
  auto output_data = output.data_ptr<scalar_t>();

  // fold nbatch and channels into single dimension for channels first.
  int64_t channels = p.nbatch * p.channels;

  int ndim = p.ndim;
  int64_t input_depth = ndim == 3 ? p.ishape[ndim - 3] : 1;
  int64_t input_height = ndim >= 2 ? p.ishape[ndim - 2] : 1;
  int64_t input_width = p.ishape[ndim - 1];
  int64_t output_depth = ndim == 3 ? p.oshape[ndim - 3] : 1;
  int64_t output_height = ndim >= 2 ? p.oshape[ndim - 2] : 1;
  int64_t output_width = p.oshape[ndim - 1];
  int64_t pad_d = ndim == 3 ? p.pads[ndim - 3] : 0;
  int64_t pad_h = ndim >= 2 ? p.pads[ndim - 2] : 0;
  int64_t pad_w = p.pads[ndim - 1];
  int64_t offset_d = ndim == 3 ? p.offsets[ndim - 3] : 0;
  int64_t offset_h = ndim >= 2 ? p.offsets[ndim - 2] : 0;
  int64_t offset_w = p.offsets[ndim - 1];

  // do vectorized copy whe output is overlapped with input on W,
  // only applies to positive padding
  auto loop = [=](scalar_t* out, scalar_t* in, bool positive_padding) {
    if (positive_padding) {
      for (const auto ow : c10::irange(pad_w)) {
        int64_t iw = PaddingType::index(ow, input_width, pad_w, offset_w);
        out[ow] = in[iw];
      }
      copy_stub(out + pad_w, in, input_width);
      for (const auto ow : c10::irange(input_width + pad_w, output_width)) {
        int64_t iw = PaddingType::index(ow, input_width, pad_w, offset_w);
        out[ow] = in[iw];
      }
    } else {
      for (const auto ow : c10::irange(output_width)) {
        int64_t iw = PaddingType::index(ow, input_width, pad_w, offset_w);
        out[ow] = in[iw];
      }
    }
  };

  if (ndim == 1) {
    // parallel on N,C,W
    at::parallel_for(
        0, channels * output_width, 1, [&](int64_t begin, int64_t end) {
          int64_t c{0}, ow{0};
          at::native::data_index_init(begin, c, channels, ow, output_width);

          for (const auto i : c10::irange(begin, end)) {
            int64_t iw = PaddingType::index(ow, input_width, pad_w, offset_w);
            output_data[i] = input_data[c * input_width + iw];
            at::native::data_index_step(c, channels, ow, output_width);
          }
        });
  } else if (ndim == 2) {
    // parallel on N,C,H, vectorize on W
    at::parallel_for(
        0, channels * output_height, 1, [&](int64_t begin, int64_t end) {
          int64_t c{0}, oh{0};
          at::native::data_index_init(begin, c, channels, oh, output_height);

          for (const auto i : c10::irange(begin, end)) {
            int64_t ih = PaddingType::index(oh, input_height, pad_h, offset_h);
            scalar_t* output_ptr = output_data + i * output_width;
            scalar_t* input_ptr =
                input_data + c * input_height * input_width + ih * input_width;

            loop(output_ptr, input_ptr, p.is_padding_positive_width);
            at::native::data_index_step(c, channels, oh, output_height);
          }
        });
  } else if (ndim == 3) {
    // parallel on N,C,D,H, vectorize on W
    at::parallel_for(
        0,
        channels * output_depth * output_height,
        1,
        [&](int64_t begin, int64_t end) {
          int64_t c{0}, od{0}, oh{0};
          at::native::data_index_init(
              begin, c, channels, od, output_depth, oh, output_height);

          for (const auto i : c10::irange(begin, end)) {
            int64_t id = PaddingType::index(od, input_depth, pad_d, offset_d);
            int64_t ih = PaddingType::index(oh, input_height, pad_h, offset_h);
            scalar_t* output_ptr = output_data + i * output_width;
            scalar_t* input_ptr = input_data +
                c * input_depth * input_height * input_width +
                id * input_height * input_width + ih * input_width;

            loop(output_ptr, input_ptr, p.is_padding_positive_width);
            at::native::data_index_step(
                c, channels, od, output_depth, oh, output_height);
          }
        });
  } else {
    TORCH_INTERNAL_ASSERT(false, "expect input dim to be 1d, 2d or 3d.");
  }

  if (!output_.is_contiguous()) {
    output_.copy_(output);
  }
}

template <typename scalar_t, typename PaddingType>
void cpu_padding_channels_last(
    const at::Tensor& output_,
    const at::Tensor& input_,
    PaddingParams& p) {
  auto memory_format = p.ndim == 2 ? at::MemoryFormat::ChannelsLast
                                   : at::MemoryFormat::ChannelsLast3d;

  auto input = input_.contiguous(memory_format);
  auto output = output_.contiguous(memory_format);

  auto input_data = input.data_ptr<scalar_t>();
  auto output_data = output.data_ptr<scalar_t>();

  int64_t nbatch = p.nbatch;
  int64_t channels = p.channels;

  int ndim = p.ndim;
  int64_t input_depth = ndim == 3 ? p.ishape[ndim - 3] : 1;
  int64_t input_height = ndim >= 2 ? p.ishape[ndim - 2] : 1;
  int64_t input_width = p.ishape[ndim - 1];
  int64_t output_depth = ndim == 3 ? p.oshape[ndim - 3] : 1;
  int64_t output_height = ndim >= 2 ? p.oshape[ndim - 2] : 1;
  int64_t output_width = p.oshape[ndim - 1];
  int64_t pad_d = ndim == 3 ? p.pads[ndim - 3] : 0;
  int64_t pad_h = ndim >= 2 ? p.pads[ndim - 2] : 0;
  int64_t pad_w = p.pads[ndim - 1];
  int64_t offset_d = ndim == 3 ? p.offsets[ndim - 3] : 0;
  int64_t offset_h = ndim >= 2 ? p.offsets[ndim - 2] : 0;
  int64_t offset_w = p.offsets[ndim - 1];

  if (ndim == 2) {
    // parallel on N,H,W, vectorize on C
    at::parallel_for(
        0,
        nbatch * output_height * output_width,
        1,
        [&](int64_t begin, int64_t end) {
          int64_t n{0}, oh{0}, ow{0};
          at::native::data_index_init(
              begin, n, nbatch, oh, output_height, ow, output_width);

          for (const auto i : c10::irange(begin, end)) {
            int64_t ih = PaddingType::index(oh, input_height, pad_h, offset_h);
            int64_t iw = PaddingType::index(ow, input_width, pad_w, offset_w);

            scalar_t* output_ptr = output_data + i * channels;
            scalar_t* input_ptr = input_data +
                (n * input_height * input_width + ih * input_width + iw) *
                    channels;
            copy_stub(output_ptr, input_ptr, channels);

            at::native::data_index_step(
                n, nbatch, oh, output_height, ow, output_width);
          }
        });
  } else if (ndim == 3) {
    // parallel on N,D,H,W, vectorize on C
    at::parallel_for(
        0,
        nbatch * output_depth * output_height * output_width,
        1,
        [&](int64_t begin, int64_t end) {
          int64_t n{0}, od{0}, oh{0}, ow{0};
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

          for (const auto i : c10::irange(begin, end)) {
            int64_t id = PaddingType::index(od, input_depth, pad_d, offset_d);
            int64_t ih = PaddingType::index(oh, input_height, pad_h, offset_h);
            int64_t iw = PaddingType::index(ow, input_width, pad_w, offset_w);

            scalar_t* output_ptr = output_data + i * channels;
            scalar_t* input_ptr = input_data +
                (n * input_depth * input_height * input_width +
                 id * input_height * input_width + ih * input_width + iw) *
                    channels;
            copy_stub(output_ptr, input_ptr, channels);

            at::native::data_index_step(
                n,
                nbatch,
                od,
                output_depth,
                oh,
                output_height,
                ow,
                output_width);
          }
        });
  } else {
    TORCH_INTERNAL_ASSERT(false, "expect input dim to be 2d or 3d.");
  }

  if (!output_.is_contiguous(memory_format)) {
    output_.copy_(output);
  }
}

// non-batch mode 4d input will be considered as Contiguous in format of CDHW
at::MemoryFormat padding_memory_format_3d(const at::Tensor& input) {
  return input.dim() == 4 ? at::MemoryFormat::Contiguous
                          : input.suggest_memory_format();
}

void replication_pad2d_kernel_impl(
    const at::Tensor& output,
    const at::Tensor& input,
    c10::IntArrayRef padding) {
  PaddingParams param{input, output, padding};
  switch (input.suggest_memory_format()) {
    case at::MemoryFormat::Contiguous: {
      AT_DISPATCH_QINT_TYPES(input.scalar_type(), "qreplication_pad2d", [&] {
        cpu_padding<scalar_t, ReplicationPad>(output, input, param);
      });
      break;
    }
    case at::MemoryFormat::ChannelsLast: {
      AT_DISPATCH_QINT_TYPES(
          input.scalar_type(), "qreplication_pad2d_channels_last", [&] {
            cpu_padding_channels_last<scalar_t, ReplicationPad>(
                output, input, param);
          });
      break;
    }
    default:
      TORCH_CHECK(
          false,
          "Unsupported memory format. Supports only ChannelsLast, Contiguous");
  }
}

void replication_pad3d_kernel_impl(
    const at::Tensor& output,
    const at::Tensor& input,
    at::IntArrayRef padding) {
  PaddingParams param{input, output, padding};
  switch (padding_memory_format_3d(input)) {
    case at::MemoryFormat::Contiguous: {
      AT_DISPATCH_QINT_TYPES(input.scalar_type(), "qreplication_pad3d", [&] {
        cpu_padding<scalar_t, ReplicationPad>(output, input, param);
      });
      break;
    }
    case at::MemoryFormat::ChannelsLast3d: {
      AT_DISPATCH_QINT_TYPES(
          input.scalar_type(), "qreplication_pad3d_channels_last", [&] {
            cpu_padding_channels_last<scalar_t, ReplicationPad>(
                output, input, param);
          });
      break;
    }
    default:
      TORCH_CHECK(
          false,
          "Unsupported memory format. Supports only ChannelsLast3d, Contiguous");
  }
}

void reflection_pad2d_kernel_impl(
    const at::Tensor& output,
    const at::Tensor& input,
    c10::IntArrayRef padding) {
  PaddingParams param{input, output, padding};
  switch (input.suggest_memory_format()) {
    case at::MemoryFormat::Contiguous: {
      AT_DISPATCH_QINT_TYPES(input.scalar_type(), "qreflection_pad2d", [&] {
        cpu_padding<scalar_t, ReflectionPad>(output, input, param);
      });
      break;
    }
    case at::MemoryFormat::ChannelsLast: {
      AT_DISPATCH_QINT_TYPES(
          input.scalar_type(), "qreflection_pad2d_channels_last", [&] {
            cpu_padding_channels_last<scalar_t, ReflectionPad>(
                output, input, param);
          });
      break;
    }
    default:
      TORCH_CHECK(
          false,
          "Unsupported memory format. Supports only ChannelsLast2d, Contiguous");
  }
}

void reflection_pad3d_kernel_impl(
    const at::Tensor& output,
    const at::Tensor& input,
    c10::IntArrayRef padding) {
  PaddingParams param{input, output, padding};
  switch (padding_memory_format_3d(input)) {
    case at::MemoryFormat::Contiguous: {
      AT_DISPATCH_QINT_TYPES(input.scalar_type(), "qreflection_pad3d", [&] {
        cpu_padding<scalar_t, ReflectionPad>(output, input, param);
      });
      break;
    }
    case at::MemoryFormat::ChannelsLast3d: {
      AT_DISPATCH_QINT_TYPES(
          input.scalar_type(), "qreflection_pad3d_channels_last", [&] {
            cpu_padding_channels_last<scalar_t, ReflectionPad>(
                output, input, param);
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

// replication padding
IPEX_REGISTER_DISPATCH(
    replication_pad2d_kernel_stub,
    &replication_pad2d_kernel_impl);
IPEX_REGISTER_DISPATCH(
    replication_pad3d_kernel_stub,
    &replication_pad3d_kernel_impl);
IPEX_REGISTER_DISPATCH(
    reflection_pad2d_kernel_stub,
    &reflection_pad2d_kernel_impl);
IPEX_REGISTER_DISPATCH(
    reflection_pad3d_kernel_stub,
    &reflection_pad3d_kernel_impl);

} // namespace cpu
} // namespace torch_ipex
