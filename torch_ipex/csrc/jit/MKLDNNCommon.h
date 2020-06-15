#pragma once

#include <ATen/OpaqueTensorImpl.h>
#include <c10/core/Allocator.h>
#include <ATen/core/Tensor.h>
#include <ideep.hpp>
#include <ATen/ATen.h>

namespace at { namespace native {
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
struct CAFFE2_API IntrusivePtrTargetWrapper : c10::intrusive_ptr_target {
private:
  T target_;

public:
  IntrusivePtrTargetWrapper() = delete;
  IntrusivePtrTargetWrapper(const T& target): target_(target) {}
  IntrusivePtrTargetWrapper(T&& target): target_(std::move(target)) {}

  T& get_target() {
    return target_;
  }
};

using IDeepTensorWrapper = IntrusivePtrTargetWrapper<ideep::tensor>;
using IDeepTensorWrapperPtr = c10::intrusive_ptr<IDeepTensorWrapper>;
using MKLDNNTensorImpl = at::OpaqueTensorImpl<IDeepTensorWrapperPtr>;

at::Tensor new_with_itensor_mkldnn(ideep::tensor&& it, const c10::TensorOptions& options) {
  // NOTE: int32_t dims from ideep::tensor but sizes needs int64_t
  // TODO: support int64_t dims in ideep::tensor to avoid extra conversion
  auto dims = it.get_dims();
  IDeepTensorWrapperPtr handle = c10::make_intrusive<IDeepTensorWrapper>(std::move(it));
  return at::detail::make_tensor<MKLDNNTensorImpl>(
    c10::DispatchKeySet(c10::DispatchKey::MkldnnCPUTensorId), options.dtype(), options.device(), handle,
    std::vector<int64_t>(dims.begin(), dims.end()));
}

ideep::tensor& itensor_from_mkldnn(const at::Tensor& mkldnn_tensor) {
  AT_ASSERTM(mkldnn_tensor.is_mkldnn(), "mkldnn_to_dense expects MKL-DNN tensor input");
  // XXX: Because constant propagation runs reorder and resulted a constant 'variable'
  // So here we have a problem:
  // AT_ASSERTM(!mkldnn_tensor.is_variable(), "_internal_get_MKLDNNImpl: should not be a variable");
  MKLDNNTensorImpl *mklimpl = static_cast<MKLDNNTensorImpl *>(mkldnn_tensor.unsafeGetTensorImpl());
  return mklimpl->unsafe_opaque_handle()->get_target();
}

ideep::tensor itensor_view_from_dense(const at::Tensor& tensor) {
  // Change it after review tensor id set
  // AT_ASSERTM(
  //     tensor.type_id() == c10::TensorTypeId::CPUTensorId,
  //     "itensor_view_from_dense expects dense CPU tensor input");
  AT_ASSERTM(tensor.scalar_type() == c10::ScalarType::Float,
             "itensor_view_from_dense expects float tensor input"); 
  // Do we need this check, the tensor we get always a variable?
  /*
  AT_ASSERTM(
      !tensor.is_variable(),
      "itensor_view_from_dense: should not be a variable");
  */
  return {{{tensor.sizes().cbegin(), tensor.sizes().cend()},
           ideep::tensor::data_type::f32},
           tensor.template data_ptr<float>()};
}

at::Tensor dense_view_from_itensor(const ideep::tensor& it) {
  auto dims = it.get_dims();
  return at::from_blob(it.get_data_handle(),
      std::vector<int64_t>(dims.begin(), dims.end()),
      at::device(at::kCPU).dtype(at::kFloat));
}

// Note in case the aten Tensor is a dense tensor, the retured ideep
// tensor is just a view of the storage of the aten dense tensor, so
// caller needs to make sure the aten dense tensor's lifetime is
// longer than the ideep tensor.
ideep::tensor get_mkldnn_tensor(const at::Tensor& tensor) {
  if (tensor.is_mkldnn()) {
    return itensor_from_mkldnn(tensor);
  } else {
    return itensor_view_from_dense(tensor);
  }
}

// temporary
struct alloc {
  static char* malloc(size_t size) {
    // Assume allocator is a singleton
    auto allocator = at::getCPUAllocator();
    return reinterpret_cast<char *>(allocator->raw_allocate(size));
  }

  static void free(void *p) {
    auto allocator = at::getCPUAllocator();
    allocator->raw_deallocate(p);
  }
};

at::Tensor _empty_like(const at::Tensor& input) {
  if (input.is_mkldnn()) {
    /* We need get extra mem-space for quirk formats */
    auto it = itensor_from_mkldnn(input);

    /* pre-alloc memory */
    ideep::tensor o;
    o.init<alloc>(it.get_descriptor());
    return new_with_itensor_mkldnn(std::move(o), input.options());
  }

  return at::empty_like(input);
}

at::Tensor empty_mkldnn(
    c10::IntArrayRef size, const c10::TensorOptions& options,
    ideep::format format, int64_t groups) {
  ideep::tensor o;

  std::vector<int64_t> dst_sizes {size.begin(), size.end()};
  if (groups > 1) {
    dst_sizes.insert(dst_sizes.begin(), groups);
    dst_sizes[1] /= groups;
  }
  o.init<alloc>({{dst_sizes.begin(), dst_sizes.end()},
      ideep::tensor::data_type::f32, format});
  return new_with_itensor_mkldnn(std::move(o), options);
}

}} // namespace torch::jit
