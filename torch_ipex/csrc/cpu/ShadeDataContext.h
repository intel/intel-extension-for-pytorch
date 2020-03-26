#pragma once

#include <ATen/Tensor.h>
#include <c10/util/Exception.h>

#include "dil/dil.hpp"

namespace torch_ipex {
namespace cpu {

enum SHADE_DATA_TYPE {CPU_RAW, DIL};

struct ShadeDataContext {
  void              *cpu_raw_data;
  c10::DeleterFnPtr  cpu_del_run;
  dil::tensor        dil_tensor;

  SHADE_DATA_TYPE    data_type;

  ShadeDataContext() : dil_tensor(),
                       cpu_raw_data(nullptr),
                       cpu_del_run(nullptr),
                       data_type(SHADE_DATA_TYPE::CPU_RAW) {}

  ~ShadeDataContext() {
    if (this->data_type == SHADE_DATA_TYPE::DIL) { // DIL Tensor
      if (this->dil_tensor.is_public_format()) {
        // If the dis tensor is plain format, then it means that its buffer is cpu buffer and should
        // be as same as cpu_raw_data
        TORCH_INTERNAL_ASSERT(this->dil_tensor.get() == cpu_raw_data);
        TORCH_INTERNAL_ASSERT(this->cpu_raw_data != nullptr);
        TORCH_INTERNAL_ASSERT(this->cpu_del_run != nullptr);
        this->cpu_del_run(this->cpu_raw_data);
        this->cpu_raw_data = nullptr;
      } else {
        // If dil tensor is block format, the cpu raw data means nothing here.
        TORCH_INTERNAL_ASSERT(this->cpu_raw_data == nullptr);
        TORCH_INTERNAL_ASSERT(this->cpu_del_run == nullptr);
      }
    } else { // CPU Tensor here
      TORCH_INTERNAL_ASSERT(this->cpu_del_run != nullptr);
      this->cpu_del_run(this->cpu_raw_data);
      this->cpu_raw_data = nullptr;
    }
  }

  static void freeShadeDataContext(void *raw_data) {
    TORCH_INTERNAL_ASSERT(raw_data != nullptr);
    ShadeDataContext *shade_data_ctx = (ShadeDataContext*)raw_data;
    auto data_type = shade_data_ctx->data_type;
    TORCH_INTERNAL_ASSERT((data_type == SHADE_DATA_TYPE::CPU_RAW) || (data_type == SHADE_DATA_TYPE::DIL));
    delete shade_data_ctx;
  }

  static ShadeDataContext *allocShadeDataContext() {
    ShadeDataContext *shade_data_context = new ShadeDataContext();
    return shade_data_context;
  }

  static inline bool isDilTensor(void *raw_context) {
    TORCH_INTERNAL_ASSERT(raw_context != nullptr);
    ShadeDataContext *shade_data_context = (ShadeDataContext*)raw_context;
    auto data_type = shade_data_context->data_type;
    TORCH_INTERNAL_ASSERT((data_type == SHADE_DATA_TYPE::CPU_RAW) || (data_type == SHADE_DATA_TYPE::DIL));
    return data_type == SHADE_DATA_TYPE::DIL;
  }

  static inline bool isDilTensor(const at::Tensor &tensor) {
    TORCH_INTERNAL_ASSERT(tensor.has_storage());
    // Make sure simple case
    //TORCH_INTERNAL_ASSERT(tensor.unsafeGetTensorImpl()->version_counter().current_version() <= 1);
    void *storage_context = tensor.storage().data_ptr().get_context();
    return isDilTensor(storage_context);
  }

  static inline bool isCpuTensor(const at::Tensor &tensor) {
    return !isDilTensor(tensor);
  }

  static inline dil::tensor getDilTensor(void *raw_context) {
    TORCH_INTERNAL_ASSERT(raw_context != nullptr);
    if (isDilTensor(raw_context)) {
      ShadeDataContext *shade_data_context = (ShadeDataContext*)raw_context;
      return shade_data_context->dil_tensor;
    } else {
      TORCH_INTERNAL_ASSERT(false);
      return dil::tensor();
    }
  }

  static inline dil::tensor getDilTensor(const at::Tensor &tensor) {
    TORCH_INTERNAL_ASSERT(tensor.has_storage());
    void *raw_context = tensor.storage().data_ptr().get_context();
    TORCH_INTERNAL_ASSERT(raw_context != nullptr);
    return getDilTensor(raw_context);
  }

  static inline void * getCpuRawData(const at::Tensor &tensor) {
    TORCH_INTERNAL_ASSERT(tensor.has_storage());
    TORCH_INTERNAL_ASSERT(tensor.unsafeGetTensorImpl()->unique_version());
    if (isCpuTensor(tensor)) {
      auto& data_ptr = tensor.storage().data_ptr();
      ShadeDataContext *shade_data_context = (ShadeDataContext*)(data_ptr.get_context());
      TORCH_INTERNAL_ASSERT(shade_data_context != nullptr);
      return shade_data_context->cpu_raw_data;
    } else {
      TORCH_INTERNAL_ASSERT(false);
      return nullptr;
    }
  }
};

}  // namespace cpu
}  // namespace torch_ipex
