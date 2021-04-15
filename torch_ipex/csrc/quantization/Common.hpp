#pragma once

#include <ATen/Tensor.h>
#include <ATen/native/quantized/cpu/quant_utils.h>

namespace torch_ipex {

bool check_int8_calibration();
void insert_or_updata_observer(const at::TensorList &inputs,
                               const at::TensorList &ouputs,
                               std::string op_name,
                               int64_t ops_id,
                               std::vector<std::string> inputs_flow,
                               std::vector<std::string> outputs_flow);

void insert_or_updata_observer(const at::TensorList &inputs,
                               const at::TensorList &ouputs,
                               const at::Tensor& weight,
                               std::string op_name, int64_t ops_id,
                               std::vector<std::string> inputs_flow,
                               std::vector<std::string> outputs_flow);

std::vector<std::vector<quant_utils::TensorQuantizationParams>> get_int8_scales(const int64_t ops_id);

std::string get_int8_weight_granularity(const int64_t ops_id);

float get_int8_weight_scale(const int64_t ops_id);

at::Tensor& get_int8_weight_tensor_scale(const int64_t ops_id);

bool get_int8_quantized_status(const int64_t ops_id);

std::tuple<std::vector<bool>, std::vector<bool>> get_int8_insert_quantized_status(const int64_t ops_id);

std::tuple<std::vector<at::ScalarType>, std::vector<at::ScalarType>> get_int8_quantized_dtypes(const int64_t ops_id);

}
