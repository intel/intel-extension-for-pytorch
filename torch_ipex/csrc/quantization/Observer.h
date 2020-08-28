#pragma once

namespace torch_ipex {
namespace cpu {
namespace lp {
namespace int8 {

struct Observer {
  int64_t id;
  std::string name;
  std::vector<std::vector<float>> inputs_min_max_values;
  std::vector<std::vector<float>> outputs_min_max_values;
  // default uising min/max to compute the quantization parameters,
  // only support min_max, MovingAverageMinMax and other none per_channel merthod
  std::string algorithm = "min_max";
  float averaging_constant = 0.01;  // for MovingAverage method
  // only useful for conv, onednn only support per_channel foo conv's weight,
  // default is per_tensor
  std::string weight_granularity = "per_tensor";
  // ture means input will be quantized to int8, otherwise quantized to uint8.
  std::vector<bool> inputs_dtype_uint8 = {false};
  std::vector<bool> outputs_dtype_uint8 = {false};
  bool quantized = true;
};

class Indicator {
  public:
    Indicator(int64_t i = 0, std::string n = "", std::string alg = "min_max",
      std::string granu = "per_tensor", std::vector<float> i_scale = {1},
      std::vector<float> o_scale = {1}, std::vector<bool> i_uint8_used = {false},
      std::vector<bool> o_uint8_used = {false}, bool quant = true):
      id(i), name(n), algorithm(alg), weight_granularity(granu), inputs_scale(i_scale),
      outputs_scale(o_scale), inputs_uint8_used(i_uint8_used), outputs_uint8_used(o_uint8_used),
      quantized(quant) {}

  int64_t get_indicator_id() {
    return id;
  }

  std::string get_indicator_name() {
    return name;
  }

  std::string get_indicator_algorithm() {
    return algorithm;
  }

  std::string get_indicator_weight_granularity() {
    return weight_granularity;
  }

  std::tuple<std::vector<float>, std::vector<float>> get_indicator_scales() {
    return std::make_tuple(inputs_scale, outputs_scale);
  }

  std::tuple<std::vector<bool>, std::vector<bool>> get_indicator_uint8_status() {
    return std::make_tuple(inputs_uint8_used, outputs_uint8_used);
  }

  bool get_indicator_quantized_status() {
    return quantized;
  }

  void set_indicator_scales(std::vector<float> new_inputs_scale, std::vector<float> new_outputs_scale) {
    inputs_scale = new_inputs_scale;
    outputs_scale = new_outputs_scale;
  }

  void set_indicator_uint8_status(std::vector<bool> new_inputs_uint8_used, std::vector<bool> new_outputs_uint8_used) {
    inputs_uint8_used = new_inputs_uint8_used;
    outputs_uint8_used = new_outputs_uint8_used;
  }

  void set_indicator_quantized_status(bool new_quantized) {
    quantized = new_quantized;
  }

  private:
    int64_t id;
    std::string name;
    std::string algorithm;
    std::string weight_granularity;
    std::vector<float> inputs_scale;
    std::vector<float> outputs_scale;
    std::vector<bool> inputs_uint8_used;
    std::vector<bool> outputs_uint8_used;
    bool quantized;
};

}  // namespace int8
}  // namespace lp
}  // namespace cpu
}  // namespace torch_ipex
