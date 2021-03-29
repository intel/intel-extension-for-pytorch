#pragma once

#include <ATen/ATen.h>
#include <c10/util/Exception.h>
#include <c10/util/Optional.h>

#include "dil/dil.hpp"

#include "torch_ipex/csrc/utils.h"
#include <mutex>

namespace torch_ipex {
namespace cpu {

enum SHADE_DATA_TYPE {CPU_RAW, DIL};

enum MIX_PREC_TYPE {NONE, MIX_BF16_FP32, MIX_INT8_FP32};

enum SHADE_TENSOR_TAG{PARAM, OTHER};

#define SANITY_CHECK_SHADE_DATA_CONTEXT(THIS) \
  { \
    if (THIS->data_type == SHADE_DATA_TYPE::DIL) { \
      TORCH_INTERNAL_ASSERT_DEBUG_ONLY(THIS->dil_tensor.has_value()); \
      if (THIS->dil_tensor->is_public_format()) { \
        if (THIS->mix_prec_type == MIX_PREC_TYPE::MIX_BF16_FP32) { \
          TORCH_INTERNAL_ASSERT_DEBUG_ONLY(THIS->cpu_raw_data == nullptr); \
          TORCH_INTERNAL_ASSERT_DEBUG_ONLY(THIS->dil_tensor->get_data_handle() != THIS->cpu_raw_data); \
          TORCH_INTERNAL_ASSERT_DEBUG_ONLY(THIS->cpu_del_fun == nullptr); \
        } else if (THIS->mix_prec_type == MIX_PREC_TYPE::MIX_INT8_FP32) { \
          TORCH_INTERNAL_ASSERT_DEBUG_ONLY(THIS->cpu_raw_data == nullptr); \
          TORCH_INTERNAL_ASSERT_DEBUG_ONLY(THIS->dil_tensor->get_data_handle() != THIS->cpu_raw_data); \
          TORCH_INTERNAL_ASSERT_DEBUG_ONLY(THIS->cpu_del_fun == nullptr); \
        } else { \
          TORCH_INTERNAL_ASSERT_DEBUG_ONLY(THIS->cpu_raw_data != nullptr); \
          TORCH_INTERNAL_ASSERT_DEBUG_ONLY(THIS->dil_tensor->get_data_handle() == THIS->cpu_raw_data); \
          TORCH_INTERNAL_ASSERT_DEBUG_ONLY(THIS->cpu_del_fun == &(c10::detail::deleteNothing)); \
        } \
      } else { \
        TORCH_INTERNAL_ASSERT_DEBUG_ONLY(THIS->cpu_raw_data == nullptr); \
        TORCH_INTERNAL_ASSERT_DEBUG_ONLY(THIS->cpu_del_fun == nullptr); \
      } \
    } else { \
      TORCH_INTERNAL_ASSERT_DEBUG_ONLY(THIS->cpu_del_fun != nullptr); \
      TORCH_INTERNAL_ASSERT_DEBUG_ONLY(THIS->cpu_del_fun != &(c10::detail::deleteNothing)); \
    } \
  }

struct ShadeDataContext {
  c10::optional<dil::tensor> dil_tensor; ///< DNNL memory buffer for lazy reorder
  void              *cpu_raw_data; ///< The raw memory buffer of storage
  c10::DeleterFnPtr  cpu_del_fun;  ///< Delete function to release cpu_raw_data
  std::mutex mutex;

  SHADE_DATA_TYPE    data_type;    ///< Memory buffer type
  MIX_PREC_TYPE      mix_prec_type; ///< Record if the aten tensor is mix-precision
  SHADE_TENSOR_TAG   shade_tensor_tag; ///< Record if the tensor is a PARAMETER (in mix-precision, never reorder a PARAMETER to bf16)

  // Generally, to get bert performance oneDNN kernel,  we need pack tensor to specific format (like ab -> ba)
  // For weight tensor, since each iter the packed format is the same, we should only pack once via using this flag
  // Note: while a tensor has been inplace reordered, this tensor will be attached a new ShadeDataContext
  // and the flag for "packed" will be auto automaticly set to "false". This is  exactly what we want now because
  // only reorder for auto mix precision is inplace now, and new dtype may need new format for oneDNN kernel
  bool packed;

  ShadeDataContext() : dil_tensor(),
                       cpu_raw_data(nullptr),
                       cpu_del_fun(nullptr),
                       packed(false),
                       data_type(SHADE_DATA_TYPE::CPU_RAW),
                       mix_prec_type(MIX_PREC_TYPE::NONE),
                       shade_tensor_tag(SHADE_TENSOR_TAG::OTHER) {}

  ~ShadeDataContext() {
    SANITY_CHECK_SHADE_DATA_CONTEXT(this);
    if (this->data_type == SHADE_DATA_TYPE::CPU_RAW) { // CPU Tensor here
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

    if (tensor.device().type() != c10::DeviceType::XPU) {
      TORCH_INTERNAL_ASSERT_DEBUG_ONLY(tensor.device().type() == c10::DeviceType::CPU);
      return false;
    }

    // Make sure simple case
    //TORCH_INTERNAL_ASSERT(tensor.unsafeGetTensorImpl()->version_counter().current_version() <= 1);
    void *storage_context = tensor.storage().data_ptr().get_context();
    ShadeDataContext *shade_data_context = (ShadeDataContext*)storage_context;
    auto data_type = shade_data_context->data_type;
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY((data_type == SHADE_DATA_TYPE::CPU_RAW) || (data_type == SHADE_DATA_TYPE::DIL));

    SANITY_CHECK_SHADE_DATA_CONTEXT(shade_data_context);

    return data_type == SHADE_DATA_TYPE::DIL;
  }

  /**
   * Check if the input tensor only contains CPU buffer.
   *
   * @param tensor input aten tensor
   */
  static inline bool isCpuTensor(const at::Tensor &tensor) {
    return !isDilOwnTheTensor(tensor);
  }

  /**
   * Unpack DNNL buffer from the input tensor
   *
   * @param tensor input aten tensor
   *
   * @return If the input tensor does not contain DNNL buffer, the function will return
   * an empty DNNL buffer. The caller should check the return buffer is empty or not.
   */
  static inline dil::tensor& getDilStorage(const at::Tensor &tensor) {
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(tensor.has_storage());
    void *raw_context = tensor.storage().data_ptr().get_context();
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(raw_context != nullptr);
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(isDilTensor(tensor));
    ShadeDataContext *shade_data_context = (ShadeDataContext*)raw_context;
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(shade_data_context->dil_tensor.has_value());
    return *(shade_data_context->dil_tensor);
  }

  static inline std::mutex& getMutex(const at::Tensor &tensor) {
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(tensor.has_storage());
    void *storage_context = tensor.storage().data_ptr().get_context();
    ShadeDataContext *shade_data_context = (ShadeDataContext*)storage_context;
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(shade_data_context != nullptr);
    return shade_data_context->mutex;
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

  /**
   * Check if the buffer of input tensor is owned by DNNL.
   *
   * @param tensor input aten tensor
   */
  static inline bool isDilOwnTheTensor(const at::Tensor &tensor) {
    void *storage_context = tensor.storage().data_ptr().get_context();
    ShadeDataContext *shade_data_context = (ShadeDataContext*)storage_context;
    auto data_type = shade_data_context->data_type;
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY((data_type == SHADE_DATA_TYPE::CPU_RAW) || (data_type == SHADE_DATA_TYPE::DIL));
    return data_type == SHADE_DATA_TYPE::DIL;
  }


  /**
   * Check if the data type of dnnl buffer is as same as the data type of aten tensor.
   *
   * @param tensor input aten tensor
   */
  static inline bool isTensorMixPrecision(const at::Tensor &tensor) {
    auto dil_tensor_type = getDilStorage(tensor).get_data_type();
    auto aten_tensor_type = tensor.scalar_type();
    if (aten_tensor_type != at::kFloat) {
      return false;
    }
    auto res = (dil_tensor_type == dil::data_type::bf16
        || dil_tensor_type == dil::data_type::u8 || dil_tensor_type == dil::data_type::s8);

    // Check mix_precision
    void *raw_context = tensor.storage().data_ptr().get_context();
    ShadeDataContext *shade_data_context = (ShadeDataContext*)raw_context;
    if (shade_data_context->mix_prec_type == MIX_PREC_TYPE::MIX_BF16_FP32 || shade_data_context->mix_prec_type == MIX_PREC_TYPE::MIX_INT8_FP32) {
      TORCH_INTERNAL_ASSERT_DEBUG_ONLY(res);
    } else {
      TORCH_INTERNAL_ASSERT_DEBUG_ONLY(!res);
    }

    return res;
  }

  /**
   * Check if the input aten tensor is a parameter.
   *
   * @param tensor input aten tensor
   */
  static inline bool isParameterTensor(const at::Tensor &tensor) {
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(tensor.has_storage());

    if (tensor.device().type() != c10::DeviceType::XPU) {
      TORCH_INTERNAL_ASSERT_DEBUG_ONLY(tensor.device().type() == c10::DeviceType::CPU);
      return false;
    }

    void *storage_context = tensor.storage().data_ptr().get_context();
    ShadeDataContext *shade_data_context = (ShadeDataContext*)storage_context;
    auto shade_tensor_tag = shade_data_context->shade_tensor_tag;
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY((shade_tensor_tag == SHADE_TENSOR_TAG::OTHER) || (shade_tensor_tag == SHADE_TENSOR_TAG::PARAM));

    SANITY_CHECK_SHADE_DATA_CONTEXT(shade_data_context);

    return shade_tensor_tag == SHADE_TENSOR_TAG::PARAM;
  }

  /**
   * Set the shade_tensor_tag of the input aten tensor to PARAM.
   *
   * @param tensor input aten tensor
   */
  static inline void setParameterTensor(const at::Tensor &tensor) {
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(tensor.has_storage());

    // TORCH_INTERNAL_ASSERT_DEBUG_ONLY(tensor.device().type() == c10::DeviceType::XPU);
    // TODO: if device is cpu, this function should not be called
    if (tensor.device().type() != c10::DeviceType::XPU) {
      TORCH_INTERNAL_ASSERT_DEBUG_ONLY(tensor.device().type() == c10::DeviceType::CPU);
      return;
    }

    void *storage_context = tensor.storage().data_ptr().get_context();
    ShadeDataContext *shade_data_context = (ShadeDataContext*)storage_context;
    shade_data_context->shade_tensor_tag = SHADE_TENSOR_TAG::PARAM;
  }

  /**
   * Check if the input aten tensor is already packed.
   *
   * @param tensor input aten tensor
   */
  static inline bool isPackedTensor(const at::Tensor &tensor) {
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(tensor.has_storage());
    if (!check_tensor_own_shade_context(tensor)) return false;
    void *storage_context = tensor.storage().data_ptr().get_context();
    ShadeDataContext *shade_data_context = (ShadeDataContext*)storage_context;
    return shade_data_context->packed;
  }

  /**
   * Set the packing status of the input aten tensor.
   *
   * @param tensor input aten tensor
   */
  static inline void setPackedTensor(const at::Tensor &tensor, bool value) {
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(tensor.has_storage());
    void *storage_context = tensor.storage().data_ptr().get_context();
    ShadeDataContext *shade_data_context = (ShadeDataContext*)storage_context;
    shade_data_context->packed = value;
  }
};

}  // namespace cpu
}  // namespace torch_ipex
