#include <ATen/ATen.h>
#include <ATen/core/DimVector.h>
#include <ATen/native/Resize.h>
#include <ATen/native/SpectralOpsUtils.h>
#include <core/detail/ListUtils.h>
#include <core/detail/OffsetCalculator.h>
#include <core/detail/TensorInfo.h>
#include <runtime/Utils.h>
#include <utils/LRUCache.h>
#include "Utils.h"
#include "comm/ATDispatch.h"
#include "comm/Numerics.h"
#include "comm/RegistrationDeclarations.h"

using namespace xpu::dpcpp;
using namespace xpu::dpcpp::detail;

#ifdef USE_ONEMKL
using namespace oneapi::mkl::dft;
#endif

namespace at {
namespace AtenIpexTypeXPU {
namespace impl {
template <typename index_t>
struct HermitianSymmetryOffsetCalculator {
  using offset_type = xpu::dpcpp::Array<index_t, 1>;
  using dim_type = std::remove_cv_t<decltype(MAX_TENSORINFO_DIMS)>;
  dim_type dims;
  IntDivider<index_t> sizes_[MAX_TENSORINFO_DIMS];
  index_t strides_[MAX_TENSORINFO_DIMS];
  uint32_t mirror_dim_; // bit mask
  static_assert(MAX_TENSORINFO_DIMS < 32, "Need a bigger mask type");

  HermitianSymmetryOffsetCalculator(
      IntArrayRef sizes,
      IntArrayRef strides,
      IntArrayRef dim,
      const int64_t element_size) {
    TORCH_INTERNAL_ASSERT(sizes.size() == strides.size());
    TORCH_INTERNAL_ASSERT(sizes.size() <= MAX_TENSORINFO_DIMS);
    dims = sizes.size();

    for (dim_type i = 0; i < MAX_TENSORINFO_DIMS; ++i) {
      if (i < dims) {
        sizes_[i] = IntDivider<index_t>(sizes[i]);
        strides_[i] = strides[i] / element_size;
      } else {
        sizes_[i] = IntDivider<index_t>(1);
        strides_[i] = 0;
      }
    }

    mirror_dim_ = 0;
    for (int64_t i = 0; i < dim.size(); ++i) {
      mirror_dim_ |= (uint32_t{1} << dim[i]);
    }
  }

  offset_type get(index_t linear_idx) const {
    index_t offset = 0;

    for (dim_type dim = 0; dim < dims; ++dim) {
      auto divmod = sizes_[dim].divmod(linear_idx);
      linear_idx = divmod.div;

      if ((mirror_dim_ & (uint32_t{1} << dim)) == 0) {
        offset += divmod.mod * strides_[dim];
      } else if (divmod.mod != 0) {
        offset += (sizes_[dim].divisor - divmod.mod) * strides_[dim];
      }
    }
    offset_type offsets;
    offsets[0] = offset;
    return offsets;
  }
};

template <typename scalar_t, typename inp_calc_t, typename out_calc_t>
void _fft_conjugate_copy_kernel(
    int64_t numel,
    scalar_t* out_data,
    const scalar_t* in_data,
    inp_calc_t ic,
    out_calc_t oc) {
  auto& dpcpp_queue = dpcppGetCurrentQueue();
  int thread_num = numel;

  auto cgf = DPCPP_Q_CGF(cgh) {
    auto kfn = DPCPP_Q_KFN(DPCPP::item<1> item_id) {
      auto in_offset = ic.get(item_id)[0];
      auto out_offset = oc.get(item_id)[0];
      out_data[out_offset] = std::conj(in_data[in_offset]);
    };

    cgh.parallel_for(DPCPP::range</*dim=*/1>(thread_num), kfn);
  };

  DPCPP_Q_SUBMIT(dpcpp_queue, cgf);
}

void _fft_fill_with_conjugate_symmetry_xpu(
    ScalarType dtype,
    IntArrayRef mirror_dims,
    IntArrayRef signal_half_sizes,
    IntArrayRef in_strides,
    const void* in_data,
    IntArrayRef out_strides,
    void* out_data) {
  // Do the actual conjugate mirroring.
  auto* in_strides_ptr = in_strides.data();
  const int ndim = in_strides.size();
  const int64_t element_size = scalarTypeToTypeMeta(dtype).itemsize();
  OffsetCalculator<1, int64_t> input_offset_calculator(
      ndim, signal_half_sizes.data(), &in_strides_ptr, &element_size);
  HermitianSymmetryOffsetCalculator<int64_t> output_offset_calculator(
      signal_half_sizes, out_strides, mirror_dims, element_size);

  const auto numel = c10::multiply_integers(signal_half_sizes);
  IPEX_DISPATCH_COMPLEX_TYPES(dtype, "_fft_fill_with_conjugate_symmetry_", [&] {
    _fft_conjugate_copy_kernel(
        numel,
        static_cast<scalar_t*>(out_data),
        static_cast<const scalar_t*>(in_data),
        input_offset_calculator,
        output_offset_calculator);
  });
}

void _fft_fill_with_conjugate_symmetry_(const Tensor& input, IntArrayRef dim_) {
  const auto input_sizes = input.sizes();
  const auto input_strides = input.strides();
  TORCH_CHECK(dim_.size() > 0);
  DimVector dim(dim_.begin(), dim_.end());
  at::maybe_wrap_dims(dim, input_strides.size());

  if (input.numel() == 0 || input_sizes[dim.back()] <= 2) {
    return; // No elements need writing
  }

  // Small dimensions may be treated as batch dims since they don't get mirrored
  dim.erase(
      std::remove_if(
          dim.begin(),
          dim.end(),
          [&](int64_t dim) { return (input_sizes[dim] <= 2); }),
      dim.end());

  // Use TensorIterator to coalesce batch dimensions
  // NOTE: Can't use TensorIterator loops because we need negative strides
  auto iter = TensorIteratorConfig()
                  .add_output(input)
                  .add_input(input)
                  .resize_outputs(false)
                  .declare_static_shape(input_sizes, dim)
                  .build();

  const auto iter_strides = iter.strides(0);
  const auto iter_sizes = iter.shape();
  const auto ndim = static_cast<int64_t>(iter_strides.size() + dim.size());
  DimVector in_strides(ndim), signal_half_sizes(ndim);
  // Take coalesced batch dimensions from TensorIterator
  std::copy(iter_strides.begin(), iter_strides.end(), in_strides.begin());
  std::copy(iter_sizes.begin(), iter_sizes.end(), signal_half_sizes.begin());

  // Take transformed dimensions directly from the input
  const auto element_size = iter.element_size(0);
  for (const auto i : c10::irange(dim.size())) {
    // Convert to byte strides to match TensorIterator
    in_strides[iter_strides.size() + i] = input_strides[dim[i]] * element_size;
    signal_half_sizes[iter_strides.size() + i] = input_sizes[dim[i]];
  }

  // For the last dimension, use negative strides to perform the mirroring
  signal_half_sizes.back() = (input_sizes[dim.back()] - 1) / 2;
  auto out_strides = in_strides;
  out_strides.back() *= -1;

  auto* data_ptr = static_cast<char*>(input.data_ptr());
  const auto* in_data = data_ptr + input_strides[dim.back()] * element_size;
  auto* out_data = data_ptr +
      (input_strides[dim.back()] * (input_sizes[dim.back()] - 1) *
       element_size);

  // Reorder dimensions by stride to maximize data locality
  DimVector dim_permute(ndim);
  std::iota(dim_permute.begin(), dim_permute.end(), 0);
  std::sort(dim_permute.begin(), dim_permute.end(), [&](auto dim1, auto dim2) {
    return in_strides[dim1] < in_strides[dim2];
  });
  DimVector temp(ndim);
  auto apply_permutation = [&](DimVector& vec) {
    // Do permuted index copy into a temporary, then copy back
    for (const auto i : c10::irange(ndim)) {
      temp[i] = vec[dim_permute[i]];
    }
    vec = temp;
  };
  apply_permutation(in_strides);
  apply_permutation(out_strides);
  apply_permutation(signal_half_sizes);

  // Find dims.slice(dims.size() - 1) in the new permuted order.
  // These are the dimensions that need explicit Hermitian mirroring
  DimVector mirror_dims;
  mirror_dims.reserve(dim.size() - 1);
  for (const auto i : c10::irange(ndim)) {
    if (dim_permute[i] >= static_cast<int64_t>(
                              iter_strides.size()) && // Not a batch dimension
        dim_permute[i] != ndim - 1) { // Not the last dim, which is mirrored
                                      // separately with negative strides
      mirror_dims.push_back(i);
    }
  }
  TORCH_INTERNAL_ASSERT(mirror_dims.size() == dim.size() - 1);

  _fft_fill_with_conjugate_symmetry_xpu(
      input.scalar_type(),
      mirror_dims,
      signal_half_sizes,
      in_strides,
      in_data,
      out_strides,
      out_data);
}

// Sort transform dimensions by input layout, for best performance
// exclude_last is for onesided transforms where the last dimension cannot be
// reordered
static DimVector _sort_dims(
    const Tensor& self,
    IntArrayRef dim,
    bool exclude_last = false) {
  DimVector sorted_dims(dim.begin(), dim.end());
  auto self_strides = self.strides();
  std::sort(
      sorted_dims.begin(),
      sorted_dims.end() - exclude_last,
      [&](int64_t a, int64_t b) { return self_strides[a] > self_strides[b]; });
  return sorted_dims;
}

#ifdef USE_ONEMKL
class dft_config_t {
 public:
  using config_int64_t = std::unordered_map<config_param, int64_t>;
  using config_float_t = std::unordered_map<config_param, float>;
  using config_double_t = std::unordered_map<config_param, double>;

  dft_config_t() {
    val_int64_.clear();
    val_float_.clear();
    val_double_.clear();
    mkl_istrides_.clear();
    mkl_ostrides_.clear();
  }

  void set_strides(
      std::vector<int64_t>& istrides,
      std::vector<int64_t>& ostrides) {
    mkl_istrides_ = istrides;
    mkl_ostrides_ = ostrides;
  }

  template <typename T>
  void set_value(config_param key, T value) {
    if (std::is_same<DFTI_CONFIG_VALUE, T>::value ||
        std::is_same<int64_t, T>::value) {
      val_int64_.insert({key, value});
    } else if (std::is_same<T, float>::value) {
      val_float_.insert({key, value});
    } else if (std::is_same<T, double>::value) {
      val_double_.insert({key, value});
    } else {
      TORCH_CHECK(0, "Unsupported value type in FFT config!");
    }
  }

  template <precision prec, domain dom>
  void commit_values(descriptor<prec, dom>& desc) {
#define COMMIT_VAL(val_map)                    \
  for (auto& value : (val_map)) {              \
    desc.set_value(value.first, value.second); \
  }

    COMMIT_VAL(val_int64_);
    COMMIT_VAL(val_float_);
    COMMIT_VAL(val_double_);

    if (!mkl_istrides_.empty()) {
      desc.set_value(config_param::INPUT_STRIDES, mkl_istrides_.data());
    }
    if (!mkl_ostrides_.empty()) {
      desc.set_value(config_param::OUTPUT_STRIDES, mkl_ostrides_.data());
    }
  }

  void to_bytes(bytestring& bytes) {
#define MAP_TO_BYTES(val_map)                  \
  for (auto& value : (val_map)) {              \
    xpu::dpcpp::to_bytes(bytes, value.first);  \
    xpu::dpcpp::to_bytes(bytes, value.second); \
  }

    MAP_TO_BYTES(val_int64_);
    MAP_TO_BYTES(val_float_);
    MAP_TO_BYTES(val_double_);

    xpu::dpcpp::to_bytes(bytes, mkl_istrides_);
    xpu::dpcpp::to_bytes(bytes, mkl_ostrides_);
  }

 private:
  config_int64_t val_int64_;
  config_float_t val_float_;
  config_double_t val_double_;
  std::vector<int64_t> mkl_istrides_;
  std::vector<int64_t> mkl_ostrides_;
};

template <precision prec, domain dom>
class dft_desc_t {
 public:
  using mkl_desc_t = descriptor<prec, dom>;

  dft_desc_t(
      DPCPP::queue& q,
      std::vector<std::int64_t>& dimensions,
      std::shared_ptr<dft_config_t> configs)
      : desc_(dimensions), configs_(configs) {
    configs_->commit_values(desc_);
    desc_.commit(q);
  }

  mkl_desc_t& raw() {
    return desc_;
  }

  static DPCPP_STATUS dft_desc_destroy(dft_desc_t* dft_desc) {
    if (dft_desc)
      delete dft_desc;
    return DPCPP_SUCCESS;
  }

 private:
  mkl_desc_t desc_;
  std::shared_ptr<dft_config_t> configs_;
};
#endif

} // namespace impl
} // namespace AtenIpexTypeXPU
} // namespace at

#ifdef USE_ONEMKL
namespace xpu {
namespace dpcpp {

template <precision prec, domain dom>
struct lru_traits<at::AtenIpexTypeXPU::impl::dft_desc_t<prec, dom>*> {
  static constexpr auto destructor =
      &at::AtenIpexTypeXPU::impl::dft_desc_t<prec, dom>::dft_desc_destroy;
};

template <precision prec, domain dom>
class dft_desc_handle
    : public lru_handle<at::AtenIpexTypeXPU::impl::dft_desc_t<prec, dom>*> {
 public:
  dft_desc_handle(
      DPCPP::queue& q,
      std::vector<std::int64_t>& dimensions,
      std::shared_ptr<at::AtenIpexTypeXPU::impl::dft_config_t> configs) {
    at::AtenIpexTypeXPU::impl::dft_desc_t<prec, dom>* dft_desc =
        new at::AtenIpexTypeXPU::impl::dft_desc_t<prec, dom>(
            q, dimensions, configs);
    lru_handle<at::AtenIpexTypeXPU::impl::dft_desc_t<prec, dom>*>::reset(
        dft_desc);
  }
};

} // namespace dpcpp
} // namespace xpu
#endif

namespace at {
namespace AtenIpexTypeXPU {
namespace impl {

#ifdef USE_ONEMKL
template <precision prec, domain signal_type, typename scalar_t>
void _mkl_dft(
    const Tensor& input,
    Tensor& output,
    int64_t signal_ndim,
    bool complex_input,
    bool complex_output,
    bool inverse,
    IntArrayRef checked_signal_sizes,
    int64_t normalization,
    bool onesided,
    int64_t batch) {
  auto dev_id = dpcppGetDeviceIdOfCurrentQueue();
  auto queue_id = dpcppGetCurrentQueueId();
  auto& dpcpp_queue = dpcppGetCurrentQueue();
  std::vector<int64_t> mkl_signal_sizes(
      checked_signal_sizes.begin() + 1, checked_signal_sizes.end());

  std::shared_ptr<dft_config_t> desc_config(new dft_config_t);
  desc_config->set_value(config_param::PLACEMENT, DFTI_NOT_INPLACE);
  desc_config->set_value(config_param::NUMBER_OF_TRANSFORMS, batch);

  auto istrides = input.strides();
  auto ostrides = output.strides();
  int64_t idist = istrides[0];
  int64_t odist = ostrides[0];
  desc_config->set_value(config_param::FWD_DISTANCE, idist);
  desc_config->set_value(config_param::BWD_DISTANCE, odist);
  std::vector<int64_t> mkl_istrides(1 + signal_ndim, 0),
      mkl_ostrides(1 + signal_ndim, 0);
  for (int64_t i = 1; i <= signal_ndim; i++) {
    mkl_istrides[i] = istrides[i];
    mkl_ostrides[i] = ostrides[i];
  }
  desc_config->set_strides(mkl_istrides, mkl_ostrides);
  if (!complex_input || !complex_output) {
    desc_config->set_value(
        config_param::CONJUGATE_EVEN_STORAGE, DFTI_COMPLEX_COMPLEX);
  }

  // rescale if requested
  const auto norm = static_cast<native::fft_norm_mode>(normalization);
  if (norm != native::fft_norm_mode::none) {
    int64_t signal_numel = c10::multiply_integers(
        IntArrayRef(checked_signal_sizes.data() + 1, signal_ndim));
    double double_scale;
    if (norm == native::fft_norm_mode::by_root_n) {
      double_scale = 1.0 / Numerics<double>::sqrt(signal_numel);
    } else {
      double_scale = 1.0 / static_cast<double>(signal_numel);
    }

    auto config_inverse =
        inverse ? config_param::BACKWARD_SCALE : config_param::FORWARD_SCALE;
    if (prec == precision::DOUBLE) {
      desc_config->set_value(config_inverse, double_scale);
    } else {
      desc_config->set_value(config_inverse, static_cast<float>(double_scale));
    }
  }

  lru_key_t key;
  // FIXME: Add queue_id and dev_id in search key to avoid conflict
  // usage due to dft_desc owned workspace buffer inside so far.
  create_key(key, queue_id, dev_id, prec, signal_type, *desc_config);
  auto desc = fetch_or_create_m<dft_desc_handle<prec, signal_type>>(
      key, dpcpp_queue, mkl_signal_sizes, desc_config);

  auto in_data = (scalar_t*)input.data_ptr();
  auto out_data = (scalar_t*)output.data_ptr();
  if (!inverse) {
    DPCPP_ONEMKL_SUBMIT(
        dpcpp_queue, compute_forward, desc->raw(), in_data, out_data);
  } else {
    DPCPP_ONEMKL_SUBMIT(
        dpcpp_queue, compute_backward, desc->raw(), in_data, out_data);
  }
}
#endif

void _fft_with_size(
    Tensor& output,
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

  bool complex_type;
  if (!inverse) {
    complex_type = complex_input ? true : false;
  } else {
    complex_type = complex_output ? true : false;
  }

  Tensor input = self;
  if (input.scalar_type() == ScalarType::Float ||
      input.scalar_type() == ScalarType::ComplexFloat) {
    if (complex_type) {
      _mkl_dft<precision::SINGLE, domain::COMPLEX, float>(
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
      _mkl_dft<precision::SINGLE, domain::REAL, float>(
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
  } else if (
      input.scalar_type() == ScalarType::Double ||
      input.scalar_type() == ScalarType::ComplexDouble) {
    if (complex_type) {
      _mkl_dft<precision::DOUBLE, domain::COMPLEX, double>(
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
      _mkl_dft<precision::DOUBLE, domain::REAL, double>(
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

#else
  AT_ERROR("fft: IPEX not compiled with MKL support");
#endif
}

// Execute a general fft operation (can be c2c, onesided r2c or onesided c2r)
static Tensor& _exec_fft(
    Tensor& out,
    Tensor self,
    IntArrayRef out_sizes,
    IntArrayRef dim,
    int64_t normalization,
    bool onesided,
    bool forward) {
  const auto ndim = self.dim();
  const int64_t signal_ndim = dim.size();
  const auto batch_dims = ndim - signal_ndim;
  // Permute dimensions so batch dimensions come first, and in stride order
  // This maximizes data locality when collapsing to a single batch dimension
  DimVector dim_permute(ndim);
  std::iota(dim_permute.begin(), dim_permute.end(), int64_t{0});

  c10::SmallVector<bool, kDimVectorStaticSize> is_transformed_dim(ndim);
  for (const auto& d : dim) {
    is_transformed_dim[d] = true;
  }
  auto batch_end =
      std::partition(dim_permute.begin(), dim_permute.end(), [&](int64_t d) {
        return !is_transformed_dim[d];
      });
  auto self_strides = self.strides();
  std::sort(dim_permute.begin(), batch_end, [&](int64_t a, int64_t b) {
    return self_strides[a] > self_strides[b];
  });
  std::copy(dim.cbegin(), dim.cend(), batch_end);
  auto input = self.permute(dim_permute);

  // Collapse batch dimensions into a single dimension
  DimVector batched_sizes(signal_ndim + 1);
  batched_sizes[0] = -1;
  std::copy(
      input.sizes().cbegin() + batch_dims,
      input.sizes().cend(),
      batched_sizes.begin() + 1);
  input = input.reshape(batched_sizes);

  const auto batch_size = input.sizes()[0];
  DimVector signal_size(signal_ndim + 1);
  signal_size[0] = batch_size;
  for (int64_t i = 0; i < signal_ndim; ++i) {
    auto in_size = input.sizes()[i + 1];
    auto out_size = out_sizes[dim[i]];
    signal_size[i + 1] = std::max(in_size, out_size);
    TORCH_INTERNAL_ASSERT(
        in_size == signal_size[i + 1] ||
        in_size == (signal_size[i + 1] / 2) + 1);
    TORCH_INTERNAL_ASSERT(
        out_size == signal_size[i + 1] ||
        out_size == (signal_size[i + 1] / 2) + 1);
  }

  batched_sizes[0] = batch_size;
  DimVector batched_out_sizes(batched_sizes.begin(), batched_sizes.end());
  for (size_t i = 0; i < dim.size(); ++i) {
    batched_out_sizes[i + 1] = out_sizes[dim[i]];
  }

  const auto value_type = c10::toValueType(input.scalar_type());
  out.resize_(batched_out_sizes, MemoryFormat::Contiguous);
  // run the FFT
  impl::_fft_with_size(
      out,
      input,
      signal_ndim,
      input.is_complex(),
      out.is_complex(),
      !forward,
      signal_size,
      normalization,
      onesided,
      out_sizes);
  // Inplace reshaping to original batch shape and inverting the dimension
  // permutation
  DimVector out_strides(ndim);
  int64_t batch_numel = 1;
  for (int64_t i = batch_dims - 1; i >= 0; --i) {
    out_strides[dim_permute[i]] = batch_numel * out.strides()[0];
    batch_numel *= out_sizes[dim_permute[i]];
  }
  for (int64_t i = batch_dims; i < ndim; ++i) {
    out_strides[dim_permute[i]] = out.strides()[1 + (i - batch_dims)];
  }
  out.as_strided_(out_sizes, out_strides, out.storage_offset());
  return out;
}

} // namespace impl

Tensor& _fft_r2c_out(
    const Tensor& self,
    IntArrayRef dim,
    int64_t normalization,
    bool onesided,
    Tensor& out) {
  auto result = at::AtenIpexTypeXPU::_fft_r2c(
      self, dim, normalization, /*onesided=*/true);
  if (onesided) {
    native::resize_output(out, result.sizes());
    return out.copy_(result);
  }

  native::resize_output(out, self.sizes());

  auto last_dim = dim.back();
  auto last_dim_halfsize = result.sizes()[last_dim];
  auto out_slice = out.slice(last_dim, 0, last_dim_halfsize);
  out_slice.copy_(result);
  at::native::_fft_fill_with_conjugate_symmetry_(out, dim);
  return out;
}

Tensor& _fft_c2r_out(
    const Tensor& self,
    IntArrayRef dim,
    int64_t normalization,
    int64_t last_dim_size,
    Tensor& out) {
  auto result =
      at::AtenIpexTypeXPU::_fft_c2r(self, dim, normalization, last_dim_size);
  native::resize_output(out, result.sizes());
  return out.copy_(result);
}

Tensor& _fft_c2c_out(
    const Tensor& self,
    IntArrayRef dim,
    int64_t normalization,
    bool forward,
    Tensor& out) {
  auto result =
      at::AtenIpexTypeXPU::_fft_c2c(self, dim, normalization, forward);
  native::resize_output(out, result.sizes());
  return out.copy_(result);
}

Tensor _fft_c2r(
    const Tensor& self,
    IntArrayRef dim,
    int64_t normalization,
    int64_t last_dim_size) {
  TORCH_CHECK(self.is_complex());
  auto input = self;
  if (dim.size() > 1) {
    auto c2c_dims = dim.slice(0, dim.size() - 1);
    input = at::AtenIpexTypeXPU::_fft_c2c(
        self, c2c_dims, normalization, /*forward=*/false);
    dim = dim.slice(dim.size() - 1);
  }

  auto in_sizes = input.sizes();
  DimVector out_sizes(in_sizes.begin(), in_sizes.end());
  out_sizes[dim.back()] = last_dim_size;
  auto out = at::empty(
      out_sizes, self.options().dtype(c10::toValueType(self.scalar_type())));
  return impl::_exec_fft(
      out,
      input,
      out_sizes,
      dim,
      normalization,
      /*onesided=*/true,
      /*forward=*/false);
}

Tensor _fft_r2c(
    const Tensor& self,
    IntArrayRef dim,
    int64_t normalization,
    bool onesided) {
  TORCH_CHECK(self.is_floating_point());
  auto input_sizes = self.sizes();
  DimVector out_sizes(input_sizes.begin(), input_sizes.end());
  auto last_dim = dim.back();
  auto last_dim_halfsize = (input_sizes[last_dim]) / 2 + 1;
  if (onesided) {
    out_sizes[last_dim] = last_dim_halfsize;
  }

  auto sorted_dims = impl::_sort_dims(self, dim, /*exclude_last=*/true);
  auto out = at::empty(
      out_sizes, self.options().dtype(c10::toComplexType(self.scalar_type())));
  return impl::_exec_fft(
      out,
      self,
      out_sizes,
      sorted_dims,
      normalization,
      onesided,
      /*forward=*/true);

  // Only need to normalize the onesided slice since data in the other half is
  // overwritten
  auto out_slice = out.slice(last_dim, 0, last_dim_halfsize);
  auto working_tensor = self;
  if (!onesided) {
    if (out.sizes()[last_dim] != out_sizes[last_dim]) {
      working_tensor.resize_(out_sizes, MemoryFormat::Contiguous);
      working_tensor.slice(last_dim, 0, last_dim_halfsize).copy_(out);
      out = std::move(working_tensor);
    }
    impl::_fft_fill_with_conjugate_symmetry_(out, dim);
  }
  return out;
}
Tensor _fft_c2c(
    const Tensor& self,
    IntArrayRef dim,
    int64_t normalization,
    bool forward) {
  TORCH_CHECK(self.is_complex());
  const auto sorted_dims = impl::_sort_dims(self, dim);
  auto out = at::empty(self.sizes(), self.options());
  return impl::_exec_fft(
      out,
      self,
      self.sizes(),
      sorted_dims,
      normalization,
      /*onesided=*/true,
      forward);
}

} // namespace AtenIpexTypeXPU
} // namespace at
