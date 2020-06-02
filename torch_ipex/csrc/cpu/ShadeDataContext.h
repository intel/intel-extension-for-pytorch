#pragma once

#include <ATen/Tensor.h>
#include <c10/util/Exception.h>
#include <c10/util/Optional.h>

#include "dil/dil.hpp"

#include "torch_ipex/csrc/utils.h"

namespace torch_ipex {
namespace cpu {

enum SHADE_DATA_TYPE {CPU_RAW, DIL};

struct ShadeDataContext {
  c10::optional<dil::tensor> dil_tensor; ///< DNNL memory buffer for lazy reorder
  void              *cpu_raw_data; ///< The raw memory buffer of storage
  c10::DeleterFnPtr  cpu_del_fun;  ///< Delete function to release cpu_raw_data

  SHADE_DATA_TYPE    data_type;    ///< Memory buffer type

  ShadeDataContext() : dil_tensor(),
                       cpu_raw_data(nullptr),
                       cpu_del_fun(nullptr),
                       data_type(SHADE_DATA_TYPE::CPU_RAW) {}

  ~ShadeDataContext() {
    if (this->data_type == SHADE_DATA_TYPE::DIL) { // DIL Tensor
      TORCH_INTERNAL_ASSERT_DEBUG_ONLY(this->dil_tensor.has_value());
      if (this->dil_tensor->is_public_format()) {
        TORCH_INTERNAL_ASSERT_DEBUG_ONLY(this->cpu_raw_data != nullptr);
        TORCH_INTERNAL_ASSERT_DEBUG_ONLY(this->dil_tensor->get_data_handle() == this->cpu_raw_data);
        TORCH_INTERNAL_ASSERT_DEBUG_ONLY(this->cpu_del_fun == &(c10::detail::deleteNothing));
      } else {
        // If dil tensor is block format, the cpu raw data means nothing here.
        TORCH_INTERNAL_ASSERT_DEBUG_ONLY(this->cpu_raw_data == nullptr);
        TORCH_INTERNAL_ASSERT_DEBUG_ONLY(this->cpu_del_fun == nullptr);
      }
    } else { // CPU Tensor here
      TORCH_INTERNAL_ASSERT_DEBUG_ONLY(this->cpu_del_fun != nullptr);
      TORCH_INTERNAL_ASSERT_DEBUG_ONLY(this->cpu_del_fun != &(c10::detail::deleteNothing));
      this->cpu_del_fun(this->cpu_raw_data);
      this->cpu_raw_data = nullptr;
    }
  }

  /**
   * The deleter function to release @class ShadeDataContext
   * 
   * @param raw_data Raw pointer of @class ShadeDataContext
   */
  static void freeShadeDataContext(void *raw_data) {
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(raw_data != nullptr);
    ShadeDataContext *shade_data_ctx = (ShadeDataContext*)raw_data;
    auto data_type = shade_data_ctx->data_type;
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY((data_type == SHADE_DATA_TYPE::CPU_RAW) || (data_type == SHADE_DATA_TYPE::DIL));
    delete shade_data_ctx;
  }

  /**
   * Create new @class ShadeDataContext
   */
  static ShadeDataContext *allocShadeDataContext() {
    ShadeDataContext *shade_data_context = new ShadeDataContext();
    return shade_data_context;
  }

  /**
   * Check the buffer of aten tensor is DNNL buffer or raw CPU buffer
   * 
   * @param tensor input aten tensor
   * 
   * @note If the storage contains both DNNL buffer and CPU buffer, and the DNNL buffer shares
   * all data with CPU buffer, then the tensor is DNNL tensor. Besides that if current storage
   * only contains DNNL buffer, it obiviouly is DNNL tensor
   */
  static inline bool isDilTensor(const at::Tensor &tensor) {
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(tensor.has_storage());
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(tensor.layout() == c10::Layout::Strided);

    if (tensor.device().type() != c10::DeviceType::DPCPP) {
      TORCH_INTERNAL_ASSERT_DEBUG_ONLY(tensor.device().type() == c10::DeviceType::CPU);
      return false;
    }

    // Make sure simple case
    //TORCH_INTERNAL_ASSERT(tensor.unsafeGetTensorImpl()->version_counter().current_version() <= 1);
    void *storage_context = tensor.storage().data_ptr().get_context();
    ShadeDataContext *shade_data_context = (ShadeDataContext*)storage_context;
    auto data_type = shade_data_context->data_type;
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY((data_type == SHADE_DATA_TYPE::CPU_RAW) || (data_type == SHADE_DATA_TYPE::DIL));

    if (data_type == SHADE_DATA_TYPE::DIL) {
      TORCH_INTERNAL_ASSERT_DEBUG_ONLY(shade_data_context->dil_tensor.has_value());
      auto raw_cpu_data = tensor.storage().data_ptr().get();
      if (raw_cpu_data == nullptr) {
        // the dnnl tensor does not share data with raw tensor data.
        TORCH_INTERNAL_ASSERT_DEBUG_ONLY(! (shade_data_context->dil_tensor->is_empty()));
        TORCH_INTERNAL_ASSERT_DEBUG_ONLY(! (shade_data_context->dil_tensor->is_public_format()));
        TORCH_INTERNAL_ASSERT_DEBUG_ONLY(check_tensor_own_whole_storage(tensor));
        return true;
      } else {
        // The dnnl tensor shares some data with raw tensor.
        TORCH_INTERNAL_ASSERT_DEBUG_ONLY(shade_data_context->dil_tensor->is_public_format());

        // For the case:
        //   1. There is a tensor named A
        //   2. There is a tensor named B that shares some data with A
        //   3. There is a tensor named C that shares some data with A
        //   4. There is a tensor named C that shares some data with B
        // example:
        //   A = torch.rand((10, 10))
        //   B = A[2:5, :]
        //   C = A[4:7, :]
        // All these tensors share same buffer of Tensor A with different storge offsets and elements.
        // So the context modification will impact all these tensors.
        if (check_tensor_own_whole_storage(tensor)) {
          TORCH_INTERNAL_ASSERT_DEBUG_ONLY(shade_data_context->dil_tensor->get_size() == tensor.storage().capacity());
          return true;
        }
      }
    }

    return false;
  }

  /**
   * Check if the input tensor only contains CPU buffer.
   * 
   * @param tensor input aten tensor
   */
  static inline bool isCpuTensor(const at::Tensor &tensor) {
    return !isDilTensor(tensor);
  }

  /**
   * Unpack DNNL buffer from the input tensor
   * 
   * @param tensor input aten tensor
   * 
   * @return If the input tensor does not contain DNNL buffer, the function will return
   * an empty DNNL buffer. The caller should check the return buffer is empty or not.
   */
  static inline dil::tensor& getDilTensor(const at::Tensor &tensor) {
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(tensor.has_storage());
    void *raw_context = tensor.storage().data_ptr().get_context();
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(raw_context != nullptr);
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(isDilTensor(tensor));
    ShadeDataContext *shade_data_context = (ShadeDataContext*)raw_context;
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(shade_data_context->dil_tensor.has_value());
    return *(shade_data_context->dil_tensor);
  }

  /**
   * Unpack raw CPU buffer from the input tensor
   * 
   * @param tensor input aten tensor
   * 
   * @return If the input tensor contains CPU buffer, the buffer will be unpacked from @class ShadeDataContext
   * and return it to the caller. Otherwise, the function will return nullptr
   */
  static inline void * getCpuRawData(const at::Tensor &tensor) {
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(tensor.has_storage());
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(tensor.unsafeGetTensorImpl()->unique_version());
    if (isCpuTensor(tensor)) {
      auto& data_ptr = tensor.storage().data_ptr();
      ShadeDataContext *shade_data_context = (ShadeDataContext*)(data_ptr.get_context());
      TORCH_INTERNAL_ASSERT_DEBUG_ONLY(shade_data_context != nullptr);
      return shade_data_context->cpu_raw_data;
    } else {
      TORCH_INTERNAL_ASSERT_DEBUG_ONLY(false);
      return nullptr;
    }
  }
};

}  // namespace cpu
}  // namespace torch_ipex
