#include "IDeepConversions.h"
#include <ATen/OpaqueTensorImpl.h>
#include <c10/core/Allocator.h>

namespace torch_ipex {

namespace cpu {

/**
 * `IntrusivePtrTargetWrapper` wraps a custom storage handle  of a tensor
 *  (as template param) and inherits `c10::intrusive_ptr_target` so that it
 *  can be used with `c10::intrusive_ptr`.
 *
 * It currently only supports wrapping the custom handle by:
 * - Constructing with an existing custom handle by copy/move constructor.
 *
 * See `OpaqueTensorImpl::opaque_handle_`.
 *
 * NOTE: if this is generally useful we may want to move this to its own header.
 */
template <typename T>
struct TORCH_API IntrusivePtrTargetWrapper : c10::intrusive_ptr_target {
 private:
  T target_;

 public:
  IntrusivePtrTargetWrapper() = delete;
  IntrusivePtrTargetWrapper(const T& target) : target_(target) {}
  IntrusivePtrTargetWrapper(T&& target) : target_(std::move(target)) {}

  T& get_target() {
    return target_;
  }
};

using IDeepTensorWrapper = IntrusivePtrTargetWrapper<ideep::tensor>;
using IDeepTensorWrapperPtr = c10::intrusive_ptr<IDeepTensorWrapper>;
using MKLDNNTensorImpl = at::OpaqueTensorImpl<IDeepTensorWrapperPtr>;
using MKLDNNTensor = at::Tensor;

ideep::tensor::data_type get_mkldnn_dtype(at::ScalarType type) {
  switch (type) {
    case at::ScalarType::Float:
      return ideep::tensor::data_type::f32;
    case at::ScalarType::QInt32:
      return ideep::tensor::data_type::s32;
    case at::ScalarType::QInt8:
      return ideep::tensor::data_type::s8;
    case at::ScalarType::QUInt8:
    case at::ScalarType::Byte:
      return ideep::tensor::data_type::u8;
    case at::ScalarType::BFloat16:
      return ideep::tensor::data_type::bf16;
    default:
      TORCH_CHECK(false, "get_mkldnn_dtype: unsupported data type");
  }
}

ideep::tensor& itensor_from_mkldnn(const MKLDNNTensor& mkldnn_tensor) {
  TORCH_CHECK(
      mkldnn_tensor.is_mkldnn(),
      "itensor_from_mkldnn expects MKL-DNN tensor input");
  MKLDNNTensorImpl* mklimpl =
      static_cast<MKLDNNTensorImpl*>(mkldnn_tensor.unsafeGetTensorImpl());
  return mklimpl->unsafe_opaque_handle()->get_target();
}

ideep::tensor itensor_view_from_dense(const at::Tensor& tensor) {
  TORCH_CHECK(
      tensor.device().is_cpu(),
      "itensor_view_from_dense expects CPU tensor input");
  TORCH_CHECK(
      tensor.layout() == c10::Layout::Strided,
      "itensor_view_from_dense expects dense tensor input");
  TORCH_CHECK(
      tensor.scalar_type() == at::ScalarType::Float ||
          tensor.scalar_type() == at::ScalarType::BFloat16,
      "itensor_view_from_dense expects float tensor input");
  return {
      {tensor.sizes().vec(),
       get_mkldnn_dtype(tensor.scalar_type()),
       tensor.strides().vec()},
      tensor.data_ptr()};
}

ideep::tensor itensor_view_from_dense(
    const at::Tensor& tensor,
    const ideep::tensor::desc& desc) {
  TORCH_CHECK(
      tensor.device().is_cpu(),
      "itensor_view_from_dense expects CPU tensor input");
  TORCH_CHECK(
      tensor.layout() == at::Layout::Strided,
      "itensor_view_from_dense expects dense tensor input");
  TORCH_CHECK(
      tensor.scalar_type() == at::ScalarType::Float ||
          tensor.scalar_type() == at::ScalarType::BFloat16,
      "itensor_view_from_dense expects float or bfloat16 tensor input");
  return {desc, tensor.data_ptr()};
}

at::Tensor new_with_itensor_mkldnn(
    ideep::tensor&& it,
    c10::optional<at::ScalarType> dtype,
    c10::optional<c10::Device> device) {
  // NOTE: int32_t dims from ideep::tensor but sizes needs int64_t
  // TODO: support int64_t dims in ideep::tensor to avoid extra conversion
  auto dims = it.get_dims();
  IDeepTensorWrapperPtr handle =
      c10::make_intrusive<IDeepTensorWrapper>(std::move(it));
  caffe2::TypeMeta dtype_ = scalarTypeToTypeMeta(dtype_or_default(dtype));
  c10::Device device_ = device_or_default(device);
  return at::detail::make_tensor<MKLDNNTensorImpl>(
      c10::DispatchKeySet(c10::DispatchKey::MkldnnCPU),
      dtype_,
      device_,
      handle,
      std::vector<int64_t>(dims.begin(), dims.end()));
}

at::Tensor mkldnn_to_dense(
    const at::Tensor& mkldnn_tensor,
    c10::optional<at::ScalarType> dtype) {
  TORCH_CHECK(
      mkldnn_tensor.scalar_type() == at::ScalarType::Float ||
          mkldnn_tensor.scalar_type() == at::ScalarType::BFloat16,
      "mkldnn_to_dense expects float or bfloat16 tensor input");
  ideep::tensor& stensor = itensor_from_mkldnn(mkldnn_tensor);
  auto dims = stensor.get_dims();
  auto data_type =
      dtype.has_value() ? dtype.value() : mkldnn_tensor.scalar_type();
  TORCH_CHECK(
      data_type == at::ScalarType::Float ||
          data_type == at::ScalarType::BFloat16,
      "mkldnn tensor only can be converted to be a float or bfloat16 "
      "cpu tensor")
  // NOTE: int32_t dims from ideep::tensor but sizes needs int64_t
  at::Tensor cpu_tensor = at::empty(
      std::vector<int64_t>(dims.begin(), dims.end()),
      mkldnn_tensor.options().layout(c10::kStrided).dtype(data_type));
  if (stensor.is_empty())
    return cpu_tensor;
  auto pub_tensor = data_type == at::ScalarType::Float
      ? stensor.to_public(
            cpu_tensor.template data_ptr<float>(),
            ideep::tensor::data_type::f32)
      : stensor.to_public(
            cpu_tensor.template data_ptr<c10::BFloat16>(),
            ideep::tensor::data_type::bf16);
  cpu_tensor.as_strided_(dims, pub_tensor.get_strides());
  return cpu_tensor;
}

// Helper function for getting an ideep tensor out of an aten Tensor.
// Note in case the aten Tensor is a dense tensor, the returned ideep
// tensor is just a view of the storage of the aten dense tensor, so
// caller needs to make sure the aten dense tensor's lifetime is
// longer than the ideep tensor.
ideep::tensor itensor_from_tensor(const at::Tensor& tensor) {
  if (tensor.is_mkldnn()) {
    return itensor_from_mkldnn(tensor);
  } else {
    return itensor_view_from_dense(tensor);
  }
}

// Init a aten tensor according to ideep tensor's desc.
at::Tensor empty_aten_tensor_from_desc(
    const ideep::tensor::desc& desc,
    const at::TensorOptions& options) {
  auto ndims = desc.data.ndims;
  auto nblks = desc.blocking_desc().inner_nblks;
  std::vector<int64_t> at_sizes(ndims + nblks);
  auto padded_dims = desc.padded_dims();
  auto blk_sizes = desc.blocking_desc().inner_blks;
  auto blk_idxs = desc.blocking_desc().inner_idxs;
  std::vector<int64_t> blk_size_per_dim(ndims, 1);
  for (auto i = 0; i < nblks; i++) {
    at_sizes[i + ndims] = blk_sizes[i];
    blk_size_per_dim[blk_idxs[i]] *= blk_sizes[i];
  }
  for (auto i = 0; i < ndims; i++) {
    at_sizes[i] = padded_dims[i] / blk_size_per_dim[i];
  }
  return at::empty(at_sizes, options);
}

} // namespace cpu
} // namespace torch_ipex
