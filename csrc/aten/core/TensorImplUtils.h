#pragma once

#include <ATen/TensorUtils.h>
#include <core/Device.h>
#include <c10/core/StorageImpl.h>
#include <c10/core/TensorImpl.h>
#include <c10/core/DeviceType.h>

using namespace at;

namespace xpu {
namespace dpcpp {

#define DPCPP_DESC_BUFF_LEN 64

using DPCPPDescBuff = struct DPCPPDescBuff {char str[DPCPP_DESC_BUFF_LEN];};

TensorImpl* TensorImpl_new(bool is_quantized);
at::Tensor TensorImpl_wrap(TensorImpl* tensor);
at::TensorImpl* TensorImpl_Unwrap(const at::Tensor& tensor);
int TensorImpl_nDimension(const at::TensorImpl* self);
int TensorImpl_nDimensionLegacyNoScalars(const at::TensorImpl* self);
int TensorImpl_nDimensionLegacyAll(const at::TensorImpl* self);
const int64_t* TensorImpl_getSizePtr(at::TensorImpl* tensor);
int64_t TensorImpl_size(const at::TensorImpl* self, int dim);
int64_t TensorImpl_sizeLegacyNoScalars(const at::TensorImpl* self, int dim);
std::vector<int64_t> TensorImpl_sizesLegacyNoScalars(
    const at::TensorImpl* self);
const int64_t* TensorImpl_getStridePtr(at::TensorImpl* tensor);
int64_t TensorImpl_stride(const at::TensorImpl* self, int dim);
int64_t TensorImpl_strideLegacyNoScalars(const at::TensorImpl* self, int dim);
TensorImpl* TensorImpl_resizeImpl(
    at::TensorImpl* self,
    at::IntArrayRef size,
    c10::optional<IntArrayRef> stride,
    bool device_guard = true);
void TensorImpl_resize(
    at::TensorImpl* self,
    at::IntArrayRef size,
    at::IntArrayRef stride);
void TensorImpl_resizeAs(at::TensorImpl* self, TensorImpl* src);
void TensorImpl_resizeNd(
    at::TensorImpl* self,
    int nDimension,
    const int64_t* size,
    const int64_t* stride);
at::StorageImpl* TensorImpl_getStoragePtr(const at::TensorImpl* tensor);
void TensorImpl_stealAndSetStoragePtr(
    at::TensorImpl* tensor,
    at::StorageImpl* storage);
void TensorImpl_set(at::TensorImpl* self, at::TensorImpl* src);
void TensorImpl_setStorage(
    at::TensorImpl* self,
    at::StorageImpl* storage_,
    ptrdiff_t storageOffset_,
    at::IntArrayRef size_,
    at::IntArrayRef stride_);
void TensorImpl_setStorageNd(
    at::TensorImpl* self,
    at::StorageImpl* storage,
    ptrdiff_t storageOffset,
    int nDimension,
    const int64_t* size,
    const int64_t* stride);
bool TensorImpl_isSetTo(const at::TensorImpl* self, const at::TensorImpl* src);
void TensorImpl_squeeze1d(
    at::TensorImpl* self,
    at::TensorImpl* src,
    int dimension);
void TensorImpl_unsqueeze1d(
    at::TensorImpl* self,
    at::TensorImpl* src,
    int dimension);
bool TensorImpl_allContiguous(at::TensorImpl** inputs, int numInputs);
int64_t TensorImpl_nElement(const at::TensorImpl* self);
void TensorImpl_retain(at::TensorImpl* self);
void TensorImpl_free(at::TensorImpl* self);
int TensorImpl_getDevice(const at::TensorImpl* tensor);
bool TensorImpl_allSameDevice(at::TensorImpl** inputs, int numInputs);
bool TensorImpl_canUse32BitIndexMath(
    const at::TensorImpl* t,
    ptrdiff_t max_elem = INT32_MAX);
bool TensorImpl_all32BitIndexable(at::TensorImpl** inputs, int numInputs);
void TensorImpl_preserveReduceDimSemantics(
    TensorImpl* tensor,
    int in_dims,
    int64_t dimension,
    int keepdim);
bool TensorImpl_maybeOverlappingIndices(const at::TensorImpl* t);

DPCPPDescBuff TensorImpl_sizeDesc(const at::TensorImpl* tensor);

static inline void IsOnSameDevice(
    at::CheckedFrom c,
    const at::TensorArg& t1,
    const at::TensorArg& t2) {
  if ((t1->device().type() != at::kXPU) ||
      (t2->device().type() != at::kXPU)) {
    std::ostringstream oss;
    if (t1->device().type() != at::kXPU) {
      oss << "Tensor for " << t1 << " is not on DPCPP, ";
    }
    if (t2->device().type() != at::kXPU) {
      oss << "Tensor for " << t2 << " is not on DPCPP, ";
    }
    oss << "but expected "
        << ((!(t1->device().type() == at::kXPU ||
               t2->device().type() == at::kXPU))
                ? "them"
                : "it")
        << " to be on DPCPP (while checking arguments for " << c << ")";
    TORCH_CHECK(0, oss.str());
  }
  TORCH_CHECK(
      t1->get_device() == t2->get_device(),
      "Expected tensor for ",
      t1,
      " to have the same device as tensor for ",
      t2,
      "; but device ",
      t1->get_device(),
      " does not equal ",
      t2->get_device(),
      " (while checking arguments for ",
      c,
      ")");
}

static inline bool IsOnSameDevice(const at::TensorList& tensor_list) {
  if (tensor_list.empty()) {
    return true;
  }
  Device curDevice = Device(kXPU, current_device());
  for (const Tensor& t : tensor_list) {
    if (t.device() != curDevice) return false;
  }
  return true;
}

} // namespace dpcpp
} // namespace xpu
