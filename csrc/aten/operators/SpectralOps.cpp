#include <ATen/ATen.h>
#include <ATen/native/SpectralOpsUtils.h>
#include <core/detail/TensorInfo.h>
#include <runtime/Utils.h>
#include "Utils.h"
#include "comm/ATDispatch.h"
#include "comm/Numerics.h"

#ifdef USE_ONEMKL
#include <mkl.h>
#include <oneapi/mkl.hpp>
#include <oneapi/mkl/dfti.hpp>
#include <utils/oneMKLUtils.h>
#endif

using namespace xpu::dpcpp::detail;
using namespace xpu::dpcpp;

namespace at {
namespace AtenIpexTypeXPU {
namespace impl {

template <typename scalar_t>
static inline void _fft_fill_with_conjugate_symmetry_slice(
    Tensor& output,
    int64_t signal_ndim,
    int64_t size_last_dim,
    int64_t start_last_dim_idx,
    int64_t numel) {
  TensorInfo<scalar_t, int64_t> output_info =
      getTensorInfo<scalar_t, int64_t>(output);
  auto& dpcpp_queue = dpcppGetCurrentQueue();

  int64_t last_dim_to_fill_size =
      size_last_dim - start_last_dim_idx; // (last - 1) dim to be filled size
  auto cgf = DPCPP_Q_CGF(cgh) {
    auto kfn = DPCPP_Q_KFN(DPCPP::item<1> id) {
      auto out_ptr = output_info.data;
      size_t idx = id.get_id(0);

      // work item index => write index
      // [b, x, y, z, imag] => [b, x, y, z + start_last_dim_idx, imag]
      int64_t imag = idx % 2;
      int64_t offset = idx / 2;
      int64_t num_dim = offset / last_dim_to_fill_size;
      int64_t slice_idx = offset % last_dim_to_fill_size;
      int64_t write_index =
          (num_dim * size_last_dim + start_last_dim_idx + slice_idx) * 2 + imag;

      // write index => read index
      // [b, x, y, z, imag] => [b, size[1]-x, size[2]-y, size[3]-z, imag]
      int64_t read_idx = 0;
      int64_t remainder = write_index - imag;
      int64_t dim_idx, dim_stride;
      for (int i = 0; i < output_info.dims - 1; i++) {
        dim_stride = output_info.strides[i];
        dim_idx = remainder / dim_stride;
        if (i == 0) {
          read_idx += dim_idx * dim_stride;
        } else if (dim_idx != 0) {
          read_idx += (output_info.sizes[i] - dim_idx) * dim_stride;
        }
        remainder = remainder % dim_stride;
      }

      if (imag) {
        out_ptr[write_index] = -out_ptr[read_idx + 1];
      } else {
        out_ptr[write_index] = out_ptr[read_idx];
      }
    };

    cgh.parallel_for(DPCPP::range<1>(numel), kfn);
  };

  DPCPP_Q_SUBMIT(dpcpp_queue, cgf);
}

static inline void _fft_fill_with_conjugate_symmetry_(
    Tensor& input,
    int64_t signal_ndim,
    int64_t size_last_dim,
    int64_t last_dim_start_slice) {
  if (last_dim_start_slice >= size_last_dim) {
    return;
  }

  // elements to be filled
  int64_t numel =
      input.numel() / size_last_dim * (size_last_dim - last_dim_start_slice);

  IPEX_DISPATCH_FLOATING_TYPES(
      input.scalar_type(), "_fft_fill_with_conjugate_symmetry_", [&] {
        _fft_fill_with_conjugate_symmetry_slice<scalar_t>(
            input, signal_ndim, size_last_dim, last_dim_start_slice, numel);
      });
}

#ifdef USE_ONEMKL
template <
    oneapi::mkl::dft::precision prec,
    oneapi::mkl::dft::domain signal_type,
    typename scalar_t>
void _mkl_dft(
    Tensor input,
    Tensor output,
    int64_t signal_ndim,
    bool complex_input,
    bool complex_output,
    bool inverse,
    IntArrayRef checked_signal_sizes,
    int64_t normalization,
    bool onesided,
    int64_t batch) {
  auto& dpcpp_queue = dpcppGetCurrentQueue();
  std::vector<int64_t> mkl_signal_sizes(
      checked_signal_sizes.begin(), checked_signal_sizes.end());
  oneapi::mkl::dft::descriptor<prec, signal_type> desc(mkl_signal_sizes);
  desc.set_value(oneapi::mkl::dft::config_param::PLACEMENT, DFTI_NOT_INPLACE);
  desc.set_value(oneapi::mkl::dft::config_param::NUMBER_OF_TRANSFORMS, batch);

  auto istrides = input.strides();
  auto ostrides = output.strides();
  int64_t idist = complex_input ? istrides[0] >> 1 : istrides[0];
  int64_t odist = complex_output ? ostrides[0] >> 1 : ostrides[0];
  desc.set_value(oneapi::mkl::dft::config_param::FWD_DISTANCE, idist);
  desc.set_value(oneapi::mkl::dft::config_param::BWD_DISTANCE, odist);
  std::vector<int64_t> mkl_istrides(1 + signal_ndim, 0),
      mkl_ostrides(1 + signal_ndim, 0);
  for (int64_t i = 1; i <= signal_ndim; i++) {
    mkl_istrides[i] = complex_input ? istrides[i] >> 1 : istrides[i];
    mkl_ostrides[i] = complex_output ? ostrides[i] >> 1 : ostrides[i];
  }
  desc.set_value(
      oneapi::mkl::dft::config_param::INPUT_STRIDES, mkl_istrides.data());
  desc.set_value(
      oneapi::mkl::dft::config_param::OUTPUT_STRIDES, mkl_ostrides.data());
  if (!complex_input || !complex_output) {
    desc.set_value(
        oneapi::mkl::dft::config_param::CONJUGATE_EVEN_STORAGE,
        DFTI_COMPLEX_COMPLEX);
  }

  // rescale if requested
  const auto norm = static_cast<at::native::fft_norm_mode>(normalization);
  if (norm != at::native::fft_norm_mode::none) {
    auto signal_numel = at::prod_intlist(checked_signal_sizes);
    double double_scale;
    if (norm == at::native::fft_norm_mode::by_root_n) {
      double_scale = 1.0 / Numerics<double>::sqrt(signal_numel);
    } else {
      double_scale = 1.0 / static_cast<double>(signal_numel);
    }
    desc.set_value(
        inverse ? oneapi::mkl::dft::config_param::BACKWARD_SCALE
                : oneapi::mkl::dft::config_param::FORWARD_SCALE,
        prec == oneapi::mkl::dft::precision::DOUBLE
            ? double_scale
            : static_cast<float>(double_scale));
  }

  desc.commit(dpcpp_queue);

  auto in_data = (scalar_t*)input.data_ptr();
  auto out_data = (scalar_t*)output.data_ptr();

  if (!inverse) {
    DPCPP_ONEMKL_SUBMIT(
        dpcpp_queue,
        oneapi::mkl::dft::compute_forward,
        desc,
        in_data,
        out_data);
  } else {
    DPCPP_ONEMKL_SUBMIT(
        dpcpp_queue,
        oneapi::mkl::dft::compute_backward,
        desc,
        in_data,
        out_data);
  }

  if (!complex_input && complex_output && !onesided) {
    auto size_last_signal_dim = checked_signal_sizes[signal_ndim - 1];
    auto start_slice = at::native::infer_ft_real_to_complex_onesided_size(
        size_last_signal_dim);
    _fft_fill_with_conjugate_symmetry_(
        output, signal_ndim, size_last_signal_dim, start_slice);
  }

  // wait for the queue.  For MKL this is a must, as it can have allocated some
  // USM internally, and that will be freed when the destructor is called.  So,
  // before that is legal, anything submitted to the queue that is using this
  // USM must have finished.
  dpcpp_queue.wait();
}
#endif

} // namespace impl

Tensor _fft_with_size(
    const Tensor& self,
    int64_t signal_ndim,
    bool complex_input,
    bool complex_output,
    bool inverse,
    IntArrayRef checked_signal_sizes,
    int64_t normalization,
    bool onesided,
    IntArrayRef output_sizes) {
#ifdef USE_ONEMKL
  int64_t batch = self.size(0);
  Tensor input_ = self;
  // real/imag dimension must aligned when viewed as of complex type

  if (complex_input) {
    bool need_contiguous = input_.stride(-1) != 1;
    for (int64_t i = 0; !need_contiguous && i <= signal_ndim; i++) {
      need_contiguous |= input_.stride(i) % 2 != 0;
    }
    if (need_contiguous) {
      input_ = input_.contiguous();
    }
  }

  Tensor output = at::empty(output_sizes, input_.options());

  bool complex_type;
  if (!inverse) {
    complex_type = complex_input ? true : false;
  } else {
    complex_type = complex_output ? true : false;
  }

  Tensor input;
  if (input_.scalar_type() == ScalarType::BFloat16) {
    input = at::empty(input_.numel(), input_.options().dtype(at::kFloat));
    dtype_convert_by_scalar(
        input.data_ptr<float>(),
        input_.data_ptr<at::BFloat16>(),
        input_.numel());
    auto output_ =
        at::empty(output.numel(), output.options().dtype(at::kFloat));
    if (complex_type) {
      impl::_mkl_dft<
          oneapi::mkl::dft::precision::SINGLE,
          oneapi::mkl::dft::domain::COMPLEX,
          float>(
          input,
          output_,
          signal_ndim,
          complex_input,
          complex_output,
          inverse,
          checked_signal_sizes,
          normalization,
          onesided,
          batch);
    } else {
      impl::_mkl_dft<
          oneapi::mkl::dft::precision::SINGLE,
          oneapi::mkl::dft::domain::REAL,
          float>(
          input,
          output_,
          signal_ndim,
          complex_input,
          complex_output,
          inverse,
          checked_signal_sizes,
          normalization,
          onesided,
          batch);
    }
    dtype_convert_by_scalar(
        output.data_ptr<at::BFloat16>(),
        output_.data_ptr<float>(),
        output.numel());

  } else {
    input = self;

    if (input.scalar_type() == ScalarType::Float) {
      if (complex_type) {
        impl::_mkl_dft<
            oneapi::mkl::dft::precision::SINGLE,
            oneapi::mkl::dft::domain::COMPLEX,
            float>(
            input,
            output,
            signal_ndim,
            complex_input,
            complex_output,
            inverse,
            checked_signal_sizes,
            normalization,
            onesided,
            batch);
      } else {
        impl::_mkl_dft<
            oneapi::mkl::dft::precision::SINGLE,
            oneapi::mkl::dft::domain::REAL,
            float>(
            input,
            output,
            signal_ndim,
            complex_input,
            complex_output,
            inverse,
            checked_signal_sizes,
            normalization,
            onesided,
            batch);
      }
    } else if (input.scalar_type() == ScalarType::Double) {
      if (complex_type) {
        impl::_mkl_dft<
            oneapi::mkl::dft::precision::DOUBLE,
            oneapi::mkl::dft::domain::COMPLEX,
            double>(
            input,
            output,
            signal_ndim,
            complex_input,
            complex_output,
            inverse,
            checked_signal_sizes,
            normalization,
            onesided,
            batch);
      } else {
        impl::_mkl_dft<
            oneapi::mkl::dft::precision::DOUBLE,
            oneapi::mkl::dft::domain::REAL,
            double>(
            input,
            output,
            signal_ndim,
            complex_input,
            complex_output,
            inverse,
            checked_signal_sizes,
            normalization,
            onesided,
            batch);
      }
    } else {
      AT_ERROR("MKL FFT doesn't support tensor of type");
    }
  }

  return output;

#else
  AT_ERROR("fft: IPEX not compiled with MKL support");
#endif
}

} // namespace AtenIpexTypeXPU
} // namespace at
