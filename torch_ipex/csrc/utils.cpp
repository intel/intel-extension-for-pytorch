#include "utils.h"

#include <ATen/Tensor.h>
#include <c10/util/Exception.h>

#include "auto_opt_config.h"

namespace torch_ipex {

bool check_device_by_tensor(const at::Tensor& tensor, DPCPPSubDev sub_dev) {
  auto dev_idx = tensor.get_device();
  //TORCH_WARN(dev_idx >= 0);
  if (sub_dev == DPCPPSubDev::CPU) {
    return dev_idx <= 0 ? true : false;
  }
  if (sub_dev == DPCPPSubDev::GPU) {
    return dev_idx > 0 ? true : false;
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
  TORCH_CHECK(((sub_dev == DPCPPSubDev::CPU) || (sub_dev == DPCPPSubDev::GPU)));
  auto dev = tensor_options.device();
  int64_t dev_idx = 0;
  if (dev.has_index())
    dev_idx = dev.index();
  if (sub_dev == DPCPPSubDev::CPU) {
    return dev_idx <= 0 ? true : false;
  }
  if (sub_dev == DPCPPSubDev::GPU) {
    return dev_idx > 0 ? true : false;
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
  if (sub_dev == DPCPPSubDev::GPU) {
    return dev_idx > 0 ? true : false;
  }
  AT_ASSERT(false);
  return false;
}

bool get_device_count(c10::Device device, c10::DeviceIndex *count) {
  TORCH_CHECK(device.type() == at::DeviceType::DPCPP);
  TORCH_WARN(device.has_index());
  if (device.index() <= 0) {
    // Always set cpu count to 1
    *count = 1;
    return true;
  } else if (device.index() > 1) {
    // TODO: get GPU device count here
    AT_ASSERT(false);
    return true;
  } else {
    return false;
  }
}

dil::data_type get_dil_data_type(at::ScalarType at_dt) {
  if (at_dt == at::ScalarType::BFloat16) {
    return dil::data_type::bf16;
  } else if (at_dt == at::ScalarType::Float) {
    return dil::data_type::f32;
  } else if (at_dt == at::ScalarType::Half) {
    return dil::data_type::f16;
  } else if (at_dt == at::ScalarType::Int) {
    return dil::data_type::s32;
  }  else if (at_dt == at::ScalarType::QInt8) {
    return dil::data_type::s8;
  }  else if (at_dt == at::ScalarType::QUInt8) {
    return dil::data_type::u8;
  } else {
    AT_ASSERT(false);
    return dil::data_type::undef;
  }
}

at::ScalarType get_at_data_type(dil::data_type dil_dt) {
  if (dil_dt == dil::data_type::bf16) {
    return at::ScalarType::BFloat16;
  } else if (dil_dt == dil::data_type::f32) {
    return at::ScalarType::Float;
  } else if (dil_dt == dil::data_type::f16) {
    return at::ScalarType::Half;
  } else if (dil_dt == dil::data_type::s32) {
    return at::ScalarType::Int;
  }  else if (dil_dt == dil::data_type::s8) {
    return at::ScalarType::QInt8;
  }  else if (dil_dt == dil::data_type::u8) {
    return at::ScalarType::QUInt8;
  } else {
    AT_ASSERT(false);
    return at::ScalarType::Undefined;
  }
}

bool check_force_dnnl_env() {
  return AutoOptConfig::singleton().get_auto_dnnl();
}

} // namespace torch_ipex
