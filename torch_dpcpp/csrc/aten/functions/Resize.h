#pragma once

#include <ATen/ATen.h>
#include <THDP/THSYCLTensor.hpp>

#include <c10/dpcpp/SYCLGuard.h>

namespace at { namespace native {

// These functions are called by native::resize_ as well as (legacy) THSYCL resize.
// They are not in THDP/THSYCLTensor.cpp because the at namespace is easier
// to benchmark than THSYCL; I can't get gbenchmark to call fns from THTensor.cpp

static inline void maybe_resize_storage_sycl(TensorImpl* self, int64_t new_size) {
  if (new_size + self->storage_offset() > 0) {
    if (!THTensor_getStoragePtr(self)) {
      AT_ERROR("Tensor: invalid null storage");
    }
    if (new_size + self->storage_offset() > self->storage().numel()) {
      THSYCLStorage_resize(
        globalContext().getTHSYCLState(),
        THTensor_getStoragePtr(self),
        new_size + self->storage_offset());
    }
  }
}

inline TensorImpl* resize_impl_sycl_(
    TensorImpl* self,
    IntArrayRef size,
    c10::optional<IntArrayRef> stride,
    bool device_guard = true) {
  if (self->sizes() == size && (!stride || self->strides() == stride)) {
    return self;
  }

  // NB: We don't need to hold device guard when calling from TH
  c10::sycl::OptionalSYCLGuard guard;
  if (device_guard) {
    guard.set_index(self->storage().device().index());
  }

  int64_t storage_size = 1;
  if (stride) {
    self->set_sizes_and_strides(size, *stride);
    // NB: storage size can be different from numel
    for (size_t dim = 0; dim < size.size(); ++dim) {
      // FIXME:  Don't rely on storage_size being negative because thi
      // may not be true for some edge cases.
      if (size[dim] == 0) {
        storage_size = 0;
        break;
      }
      storage_size += (size[dim] -1) * stride.value()[dim];
    }
  } else {
    self->set_sizes_contiguous(size);
    storage_size = self->numel();
  }
  maybe_resize_storage_sycl(self, storage_size);
  return self;
}

Tensor& resize_sycl_(Tensor& self, IntArrayRef size);

}} // namepsace at::native
