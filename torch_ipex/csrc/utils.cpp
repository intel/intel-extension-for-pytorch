#include "utils.h"

#include <ATen/Tensor.h>
#include <c10/util/Exception.h>

#include "auto_opt_config.h"

namespace torch_ipex {

thread_local IPEXFuncStatus g_current_ipex_func_stat = IPEXFuncStatus::IPEX_SUCCESS;

bool check_device_by_tensor(const at::Tensor& tensor, DPCPPSubDev sub_dev) {
  auto dev_idx = tensor.get_device();
  if (sub_dev == DPCPPSubDev::CPU) {
    return dev_idx <= 0 ? true : false;
  }
  AT_ASSERT(false);
  return false;
}

bool check_device_by_tensor_list(const at::TensorList& tensor_list, DPCPPSubDev sub_dev) {
  for (const auto& tensor : tensor_list) {
    if (tensor.defined()) {
        return check_device_by_tensor(tensor, sub_dev);
    }
  }
  AT_ASSERT(false);
  return false;
}

bool check_device_by_options(const at::TensorOptions& tensor_options, DPCPPSubDev sub_dev) {
  TORCH_CHECK(tensor_options.has_device());
  TORCH_CHECK(sub_dev == DPCPPSubDev::CPU);
  auto dev = tensor_options.device();
  int64_t dev_idx = 0;
  if (dev.has_index())
    dev_idx = dev.index();
  if (sub_dev == DPCPPSubDev::CPU) {
    return dev_idx <= 0 ? true : false;
  }
  AT_ASSERT(false);
  return false;
}

bool check_layout_by_options(const at::TensorOptions& tensor_options, c10::Layout layout) {
  TORCH_CHECK(tensor_options.has_layout());
  TORCH_CHECK(((layout == c10::kStrided) || (layout == c10::kSparse)));
  return tensor_options.layout() == layout;
}

bool check_device_by_device(const at::Device& device, DPCPPSubDev sub_dev) {
  int64_t dev_idx = 0;
  if (device.has_index())
    dev_idx = device.index();
  if (sub_dev == DPCPPSubDev::CPU) {
    return dev_idx <= 0 ? true : false;
  }
  AT_ASSERT(false);
  return false;
}

bool get_device_count(c10::Device device, c10::DeviceIndex *count) {
  TORCH_CHECK(device.type() == at::DeviceType::DPCPP);
  // TORCH_WARN(device.has_index());
  if (device.index() <= 0) {
    // Always set cpu count to 1
    *count = 1;
    return true;
  } else {
    AT_ASSERT(false);
    return false;
  }
}

dil::data_type get_dil_data_type(at::ScalarType at_dt) {
  if (at_dt == at::ScalarType::BFloat16) {
    return dil::data_type::bf16;
  } else if (at_dt == at::ScalarType::Float) {
    return dil::data_type::f32;
  } else if (at_dt == at::kInt) {
    return dil::data_type::s32;
  }  else if (at_dt == at::ScalarType::QInt8) {
    return dil::data_type::s8;
  }  else if (at_dt == at::ScalarType::QUInt8) {
    return dil::data_type::u8;
  } else {
#if defined(_DEBUG)
    TORCH_WARN("DNNL does not support current data type.");
#endif
    return dil::data_type::undef;
  }
}

at::ScalarType get_at_data_type(dil::data_type dil_dt) {
  if (dil_dt == dil::data_type::bf16) {
    return at::ScalarType::BFloat16;
  } else if (dil_dt == dil::data_type::f32) {
    return at::ScalarType::Float;
  } else if (dil_dt == dil::data_type::s32) {
    return at::kInt;
  }  else if (dil_dt == dil::data_type::s8) {
    return at::ScalarType::QInt8;
  }  else if (dil_dt == dil::data_type::u8) {
    return at::ScalarType::QUInt8;
  } else {
    AT_ASSERT(false);
    return at::ScalarType::Undefined;
  }
}

bool check_auto_dnnl() {
  return AutoOptConfig::singleton().get_auto_dnnl();
}

bool check_auto_mix_bf16_fp32() {
  return AutoOptConfig::singleton().get_mix_bf16_fp32();
}

bool check_train() {
  return AutoOptConfig::singleton().get_train();
}

bool check_auto_mix_int8_fp32() {
  return AutoOptConfig::singleton().get_mix_int8_fp32();
}

bool check_int8_calibration() {
  return AutoOptConfig::singleton().get_int8_calibration();
}

void insert_or_updata_observer(const at::Tensor& self) {
  std::vector<int64_t> input_sizes = self.sizes().vec();
  int64_t channel_axis = 0; // not used now
  // now only support min_max observer for activation
  std::vector<float> mins = {self.min().item<float>()};
  // only need max value for dnnl
  std::vector<float> maxs = {self.abs().max().item<float>()};
  AutoOptConfig::singleton().insert_or_updata_observer(input_sizes, channel_axis, mins, maxs);
}

std::tuple<std::vector<float>, std::vector<float>> get_indictor_scales(bool uint8_used) {
  return AutoOptConfig::singleton().get_indictor_scales(uint8_used);
}

bool check_tensor_own_whole_storage(const at::Tensor& tensor) {
  if (!(tensor.defined()))
    return false;

  return (tensor.storage_offset() == 0) &&
         (tensor.numel() == tensor.storage().numel()) &&
         (tensor.itemsize() == tensor.storage().itemsize());
}

bool check_tensor_own_shade_context(const at::Tensor& tensor) {
  if (!(tensor.defined()))
    return false;

  // [NOTE]: We assume the real data of storage should be as same as its context.
  //         Then we use the assumption to check if current tensor has contained
  //         shade data context.
  void *data_ptr = tensor.unsafeGetTensorImpl()->storage().data_ptr().get();
  void *data_ctx = tensor.unsafeGetTensorImpl()->storage().data_ptr().get_context();
  return (data_ptr != data_ctx) && (data_ctx != nullptr);
}

bool check_aten_dil_shape_info(const at::Tensor& ipex_tensor, const dil::tensor &dil_tensor) {
  if (dil_tensor.is_public_format()) {
    return ipex_tensor.sizes().vec() == dil_tensor.get_dims() &&
          ipex_tensor.strides().vec() == dil_tensor.get_strides();
  } else {
    return ipex_tensor.sizes().vec() == dil_tensor.get_dims();
  }
}

IPEXFuncStatus get_ipex_func_status() {
  return g_current_ipex_func_stat;
}

void set_ipex_func_status(IPEXFuncStatus ipex_fun_stat) {
  g_current_ipex_func_stat = ipex_fun_stat;
}

void reset_ipex_func_status() {
  set_ipex_func_status(IPEXFuncStatus::IPEX_SUCCESS);
}

bool is_ipex_func_success() {
  return g_current_ipex_func_stat == IPEXFuncStatus::IPEX_SUCCESS;
}

}  // namespace torch_ipex
