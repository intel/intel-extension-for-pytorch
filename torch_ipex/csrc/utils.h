#pragma once

#include <ATen/Tensor.h>
#include "cpu/dil/dil.hpp"

namespace torch_ipex {

enum DPCPPSubDev {
  CPU,
};

enum IPEXFuncStatus {
  IPEX_SUCCESS,
  IPEX_UNIMPLEMENTED,
  IPEX_FALLBACK
};

bool check_device_by_tensor(const at::Tensor& tensor, DPCPPSubDev sub_dev);
bool check_device_by_tensor_list(const at::TensorList& tensor_list, DPCPPSubDev sub_dev);
bool check_device_by_options(const at::TensorOptions& tensor_options, DPCPPSubDev sub_dev);
bool check_device_by_device(const at::Device& device, DPCPPSubDev sub_dev);
bool check_layout_by_options(const at::TensorOptions& tensor_options, c10::Layout layout);
bool get_device_count(c10::Device dev_type, c10::DeviceIndex *count);
bool check_auto_dnnl();
bool check_train();
bool check_auto_mix_bf16_fp32();
bool check_auto_mix_int8_fp32();
bool check_int8_calibration();
void insert_or_updata_observer(const at::Tensor& tensor);
std::tuple<std::vector<float>, std::vector<float>> get_indictor_scales(bool uint8_used);
bool check_tensor_own_whole_storage(const at::Tensor& tensor);
bool check_tensor_own_shade_context(const at::Tensor& tensor);
bool check_aten_dil_shape_info(const at::Tensor& ipex_tensor, const dil::tensor &dil_tensor);

dil::data_type get_dil_data_type(at::ScalarType);
at::ScalarType get_at_data_type(dil::data_type);

IPEXFuncStatus get_ipex_func_status();
bool is_ipex_func_success();
void reset_ipex_func_status();
void set_ipex_func_status(IPEXFuncStatus ipex_fun_status);

// A light-weight TORCH_CHECK that does not collect any backtrace info
#if defined(_DEBUG)
#define IPEX_CHECK(cond, ...)                                                  \
  if (!(cond)) {                                                               \
    throw std::runtime_error(                                                  \
      c10::detail::if_empty_then(                                              \
        c10::str(__VA_ARGS__),                                                 \
        "Expected " #cond " to be true, but got false."));                     \
  }
#else
// quick path of IPEX_CHECK without reporting message
#define IPEX_CHECK(cond, ...)                                                  \
  if (!(cond)) { throw std::exception(); }
#endif
} // namespace torch_ipex
