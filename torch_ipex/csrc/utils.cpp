#include "utils.h"

#include <ATen/ATen.h>
#include <c10/util/Exception.h>

#include "auto_opt_config.h"
#include "cpu/int8/Config.h"

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
  TORCH_CHECK(device.type() == at::DeviceType::XPU);
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

void insert_or_updata_observer(const at::TensorList &inputs,
                               const at::TensorList &outputs,
                               std::string op_name, int64_t ops_id) {
  std::vector<std::vector<float>> inputs_min_max_values, outputs_min_max_values;
  for (auto i = 0; i < inputs.size(); i++) {
    inputs_min_max_values.push_back({inputs[i].abs().min().item<float>(), inputs[i].abs().max().item<float>()});
  }
  for (auto j = 0; j < outputs.size(); j++) {
    outputs_min_max_values.push_back({outputs[j].abs().min().item<float>(), outputs[j].abs().max().item<float>()});
  }
  Int8OptConfig::get_config().insert_or_updata_observer(
      op_name, inputs_min_max_values, outputs_min_max_values, ops_id);
}

std::vector<std::vector<float>>
get_indicator_scales(std::vector<bool> i_uint8_used,
                     std::vector<bool> o_uint8_used, const int64_t ops_id) {
  return Int8OptConfig::get_config().get_indicator_scales(i_uint8_used,
                                                          o_uint8_used, ops_id);
}

bool get_indicator_quantized_status(const int64_t ops_id) {
  return Int8OptConfig::get_config().get_indicator_quantized_status(ops_id);
}

bool check_tensor_own_whole_storage(const at::Tensor& tensor) {
  if (!(tensor.defined()))
    return false;

  return (tensor.storage_offset() == 0) &&
         (tensor.numel() * tensor.itemsize() == tensor.storage().nbytes());
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

bool is_scalar_tensor(const at::Tensor& tensor) {
  auto strides = tensor.strides();
  for (int i = 0; i < strides.size(); i++) {
    if (strides[i] != 0) return false;
  }

  return tensor.numel() == 1;
}

}  // namespace torch_ipex
