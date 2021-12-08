#include <ATen/NativeFunctions.h>
#include <ATen/native/quantized/cpu/quant_utils.h>
#include <torch/csrc/autograd/function.h>

#include "auto_opt_config.hpp"
#include "csrc/utils/utils.h"

#include "Config.hpp"

namespace torch_ipex {
using namespace int8;

std::vector<quant_utils::TensorQuantizationParams> ComputeQuantizationParams(
    const std::vector<std::vector<float>>& min_max_values,
    const std::vector<std::string>& quantized_types,
    bool preserve_sparsity,
    int precision) {
  std::vector<quant_utils::TensorQuantizationParams> QParams;
  for (auto j = 0; j < min_max_values.size(); j++) {
    bool is_signed = quantized_types[j] == "int8" ? true : false;
    quant_utils::TensorQuantizationParams qparams{};
    if (preserve_sparsity) {
      auto max_value =
          std::max(std::abs(min_max_values[j][0]), min_max_values[j][1]);
      qparams = quant_utils::ChooseQuantizationParams(
          /*min*/ -max_value,
          /*max*/ max_value,
          /*q_min*/ is_signed ? -(1 << (precision - 1)) : 0,
          /*q_max*/
          is_signed ? ((1 << (precision - 1)) - 1) : (1 << precision) - 1,
          /*preserve_sparsity=*/true);
    } else {
      qparams = quant_utils::ChooseQuantizationParams(
          /*min*/ min_max_values[j][0],
          /*max*/ min_max_values[j][1],
          /*q_min*/ is_signed ? -(1 << (precision - 1)) : 0,
          /*q_max*/
          is_signed ? ((1 << (precision - 1)) - 1) : (1 << precision) - 1,
          /*preserve_sparsity=*/false);
    }
    QParams.push_back(qparams);
  }
  return QParams;
}

void Int8OptConfig::insert_or_updata_observer(
    std::string op_name,
    std::vector<std::vector<float>> i_min_max_values,
    std::vector<std::vector<std::vector<float>>> w_min_max_values,
    std::vector<std::vector<float>> o_min_max_values,
    int64_t ops_id,
    std::vector<std::string> inputs_flow,
    std::vector<std::string> outputs_flow) {
  if (observers_.size() <= ops_id) {
    // this path is that to set int8 op's configure, using default configures if
    // user not set it. Note: weight's value only set onece.
    std::string observer_algorithm = "min_max";
    float averaging_constant =
        0.01; // will be enabled for moving_averager_min_max
    std::string weight_granularity = "per_channel";
    if (op_name == "embedding_bag") {
      weight_granularity = "per_tensor";
    }
    const int nums_input = i_min_max_values.size();
    const int nums_output = o_min_max_values.size();
    std::vector<std::string> input_quantized_dtypes(nums_input, "uint8");
    std::vector<std::string> output_quantized_dtypes(nums_output, "uint8");

    auto qscheme = AutoOptConfig::singleton().get_int8_qscheme();
    TORCH_CHECK(
        qscheme == at::QScheme::PER_TENSOR_AFFINE ||
            qscheme == at::QScheme::PER_TENSOR_SYMMETRIC,
        "Activation is only support per-tensor quantization");
    if (qscheme == at::QScheme::PER_TENSOR_SYMMETRIC) {
      // for symmetrice quantization, quantized's dtype is always int8.
      std::fill(
          input_quantized_dtypes.begin(), input_quantized_dtypes.end(), "int8");
      std::fill(
          output_quantized_dtypes.begin(),
          output_quantized_dtypes.end(),
          "int8");
    }
    const auto num_inputs = i_min_max_values.size();
    std::vector<bool> inputs_quantized(num_inputs, true);
    const auto num_outputs = o_min_max_values.size();
    std::vector<bool> outputs_quantized(num_outputs, true);
    if (op_name == "relu_" || op_name == "add_") {
      std::fill(inputs_quantized.begin(), inputs_quantized.end(), false);
      std::fill(outputs_quantized.begin(), outputs_quantized.end(), false);
    }
    if (!indicators_.empty()) {
      observer_algorithm = indicators_[ops_id].get_indicator_algorithm();
      weight_granularity =
          indicators_[ops_id].get_indicator_weight_granularity();
      std::tie(input_quantized_dtypes, output_quantized_dtypes) =
          indicators_[ops_id].get_indicator_quantized_dtypes();
      std::tie(inputs_quantized, outputs_quantized) =
          indicators_[ops_id].get_indicator_insert_quantized_status();
    }
    Observer new_observer = {
        ops_id,
        op_name,
        i_min_max_values,
        w_min_max_values,
        o_min_max_values,
        observer_algorithm,
        averaging_constant,
        weight_granularity,
        input_quantized_dtypes,
        output_quantized_dtypes,
        inputs_quantized,
        outputs_quantized,
        inputs_flow,
        outputs_flow};
    observers_.push_back(new_observer);
  } else {
    // user has set configure or have run one interation
    auto inputs_pre = observers_[ops_id].inputs_min_max_values;
    auto outputs_pre = observers_[ops_id].outputs_min_max_values;
    if (observers_[ops_id].algorithm == "min_max") {
      for (auto i = 0; i < i_min_max_values.size(); i++) {
        observers_[ops_id].inputs_min_max_values[i][0] =
            std::min(inputs_pre[i][0], i_min_max_values[i][0]);
        observers_[ops_id].inputs_min_max_values[i][1] =
            std::max(inputs_pre[i][1], i_min_max_values[i][1]);
      }
      for (auto j = 0; j < o_min_max_values.size(); j++) {
        observers_[ops_id].outputs_min_max_values[j][0] =
            std::min(outputs_pre[j][0], o_min_max_values[j][0]);
        observers_[ops_id].outputs_min_max_values[j][1] =
            std::max(outputs_pre[j][1], o_min_max_values[j][1]);
      }
    } else if (observers_[ops_id].algorithm == "moving_averager_min_max") {
      auto c = observers_[ops_id].averaging_constant;
      for (auto i = 0; i < i_min_max_values.size(); i++) {
        observers_[ops_id].inputs_min_max_values[i][0] =
            (1 - c) * inputs_pre[i][0] + c * i_min_max_values[i][0];
        observers_[ops_id].inputs_min_max_values[i][1] =
            (1 - c) * inputs_pre[i][1] + c * i_min_max_values[i][1];
      }
      for (auto j = 0; j < o_min_max_values.size(); j++) {
        observers_[ops_id].outputs_min_max_values[j][0] =
            (1 - c) * outputs_pre[j][0] + c * o_min_max_values[j][0];
        observers_[ops_id].outputs_min_max_values[j][1] =
            (1 - c) * outputs_pre[j][1] + c * o_min_max_values[j][1];
      }
    }
  }
}

void Int8OptConfig::clear_indicators() {
  indicators_.clear();
  weights_scales_.clear();
}

void Int8OptConfig::add_indicators() {
  indicators_.clear();
  // default used is u8
  const int precision = 8;
  for (auto i = 0; i < observers_.size(); i++) {
    std::vector<quant_utils::TensorQuantizationParams> input_params,
        output_params;
    std::vector<std::vector<float>> weights_scales;

    auto input_values = observers_[i].inputs_min_max_values;
    auto output_values = observers_[i].outputs_min_max_values;
    auto weights_values = observers_[i].weights_min_max_values;
    auto x_quantized_types = observers_[i].input_quantized_dtypes;
    auto y_quantized_types = observers_[i].output_quantized_dtypes;
    // for symmetric: s = 2max(|x_min|, x_max) / (Q_max - Q_min),
    // z = 0 for qint8 and z = 128 for quint8;
    // otherwise: s = (x_max - x_min) / (Q_max - Q_min),
    // z = Q_min - round(x_min / s).
    auto qscheme = AutoOptConfig::singleton().get_int8_qscheme();
    TORCH_CHECK(
        qscheme == at::QScheme::PER_TENSOR_AFFINE ||
            qscheme == at::QScheme::PER_TENSOR_SYMMETRIC,
        "Activation is only support per-tensor quantization");
    // Note: preserve_sparsity here means symmetric quantization.
    bool preserve_sparsity = false;
    if (qscheme == at::QScheme::PER_TENSOR_SYMMETRIC) {
      preserve_sparsity = true;
    }
    input_params = ComputeQuantizationParams(
        input_values, x_quantized_types, preserve_sparsity, precision);
    output_params = ComputeQuantizationParams(
        output_values, y_quantized_types, preserve_sparsity, precision);
    // for weight, always using symetric quantization, quantized to int8 dtype.
    // is_signed = true;
    for (auto m = 0; m < weights_values.size(); m++) {
      auto w = weights_values[m];
      std::vector<float> w_scales;
      for (auto n = 0; n < w.size(); n++) {
        auto w_max_value = std::max(
            std::abs(weights_values[m][n][0]), weights_values[m][n][1]);
        auto qparams = quant_utils::ChooseQuantizationParams(
            /*min*/ -w_max_value,
            /*max*/ w_max_value,
            /*q_min*/ -(1 << (precision - 1)),
            /*q_max*/ ((1 << (precision - 1)) - 1),
            /*preserve_sparsity=*/true);
        w_scales.push_back(qparams.scale);
      }
      weights_scales.push_back(w_scales);
    }
    Indicator new_indicator(
        observers_[i].id,
        observers_[i].name,
        observers_[i].algorithm,
        observers_[i].weight_granularity,
        input_params,
        weights_scales,
        output_params,
        observers_[i].input_quantized_dtypes,
        observers_[i].output_quantized_dtypes,
        observers_[i].inputs_quantized,
        observers_[i].outputs_quantized,
        observers_[i].inputs_flow,
        observers_[i].outputs_flow);
    indicators_.push_back(new_indicator);
  }
  observers_.clear();
}

std::vector<std::vector<quant_utils::TensorQuantizationParams>> Int8OptConfig::
    get_indicator_scales(const int64_t ops_id) {
  std::vector<quant_utils::TensorQuantizationParams> x_params, y_params;
  std::tie(x_params, y_params) = indicators_[ops_id].get_indicator_scales();
  return {x_params, y_params};
}

std::string Int8OptConfig::get_indicator_weight_granularity(
    const int64_t ops_id) {
  std::string weight_granularity = "per_channel";
  // user not set weight granularity, using default granularity
  if (indicators_.empty()) {
    return weight_granularity;
  }

  weight_granularity = indicators_[ops_id].get_indicator_weight_granularity();
  return weight_granularity;
}

// per tensor quantization for weight
std::vector<float> Int8OptConfig::get_indicator_weight_scale(
    const int64_t ops_id) {
  std::vector<float> w_scales;
  auto weights_scales = indicators_[ops_id].get_indicator_weight_scales();
  TORCH_CHECK(
      weights_scales.size() > 0,
      "weights_scales should be greater than zero when get weight scale");
  for (auto i = 0; i < weights_scales.size(); i++) {
    w_scales.push_back(weights_scales[i][0]);
  }
  return w_scales;
}

// per channel quantization for weight
std::vector<at::Tensor>& Int8OptConfig::get_indicator_weight_tensor_scale(
    const int64_t ops_id) {
  TORCH_CHECK(
      weights_scales_[ops_id].size() > 0,
      "weights_scales_ should be "
      "greater than zero when get "
      "the weight scale tensors");
  return weights_scales_[ops_id];
}

std::tuple<std::vector<bool>, std::vector<bool>> Int8OptConfig::
    get_indicator_insert_quantized_status(const int64_t ops_id) {
  return indicators_[ops_id].get_indicator_insert_quantized_status();
}

std::tuple<std::vector<std::string>, std::vector<std::string>> Int8OptConfig::
    get_indicator_quantized_dtypes(const int64_t ops_id) {
  return indicators_[ops_id].get_indicator_quantized_dtypes();
}

void Int8OptConfig::set_indicators(std::vector<Indicator> indicators) {
  // avoid to use copy assignment since the copy assignment for indicator with
  // rw_mutex have not been handdled properly
  indicators_.reserve(indicators.size());
  for (auto i : indicators) {
    // if weight_granularity is per_channle, first cache the scales tensor for
    // trace.
    if (i.get_indicator_weight_granularity() == "per_channel") {
      auto id = i.get_indicator_id();
      auto w_scales = i.get_indicator_weight_scales();
      std::vector<at::Tensor> casted_scales;
      for (auto i = 0; i < w_scales.size(); i++) {
        casted_scales.emplace_back(
            at::tensor(w_scales[i], at::device(at::kCPU).dtype(at::kDouble)));
      }
      weights_scales_.emplace(id, casted_scales);
    }
    indicators_.emplace_back(i);
  }
}

std::vector<Indicator> Int8OptConfig::get_indicators() {
  return indicators_;
}

int64_t Int8OptConfig::get_indicators_size() {
  return indicators_.size();
}

void Int8OptConfig::calibration_reset() {
  current_ops_id = 0;
}

int64_t Int8OptConfig::fetch_and_add_ops_id() {
  int64_t ops_id = current_ops_id++;
  int64_t indicator_size = Int8OptConfig::get_config().get_indicators_size();
  if (current_ops_id == indicator_size)
    current_ops_id = 0;
  return ops_id;
}

thread_local int64_t Int8OptConfig::current_ops_id = 0;

} // namespace torch_ipex
