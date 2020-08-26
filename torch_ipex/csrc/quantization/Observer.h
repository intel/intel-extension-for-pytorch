#pragma once

namespace torch_ipex {
namespace cpu {
namespace lp {
namespace int8 {

struct Observer {
  int64_t Id;
  std::string Name;
  std::vector<float> Input_min_max_values;
  std::vector<float> Output_min_max_values;
  // default uising min/max to compute the quantization parameters,
  // only support min_max, MovingAverageMinMax and other none per_channel merthod
  std::string Algorithm = "min_max";
  float Averaging_constant = 0.01;  // for MovingAverage method
  // only useful for conv, onednn only support per_channel foo conv's weight,
  // default is per_tensor
  std::string Weight_granularity = "per_tensor";
  // ture means input will be quantized to int8, otherwise quantized to uint8.
  bool Input_dtype_uint8 = false;
  bool Output_dtype_uint8 = false;
  bool Quantized = true;
};

class Indicator {
  public:
    Indicator(int64_t id = 0, std::string name = "", std::string algorithm = "min_max",
      std::string weight_granularity = "per_tensor", std::vector<float> scales = std::vector<float>(2, 1),
      std::vector<bool> uint8_used = std::vector<bool>(2, false),bool quantized = true):
      Id(id), Name(name), Algorithm(algorithm), Weight_granularity(weight_granularity),
      Scales(scales), Uint8_used(uint8_used), Quantized(quantized) {}

  int64_t get_indicator_id() {
    return Id;
  }

  std::string get_indicator_name() {
    return Name;
  }

  std::string get_indicator_algorithm() {
    return Algorithm;
  }

  std::string get_indicator_weight_granularity() {
    return Weight_granularity;
  }

  std::vector<float> get_indicator_scales() {
    return Scales;
  }

  std::vector<bool> get_indicator_uint8_status() {
    return Uint8_used;
  }

  bool get_indicator_quantized_status() {
    return Quantized;
  }

  void set_indicator_scales(std::vector<float> new_scales) {
    Scales = new_scales;
  }

  void set_indicator_uint8_status(std::vector<bool> new_uint8_used) {
    Uint8_used = new_uint8_used;
  }

  void set_indicator_quantized_status(bool new_quantized) {
    Quantized = new_quantized;
  }

  private:
    int64_t Id;
    std::string Name;
    std::string Algorithm;
    std::string Weight_granularity;
    std::vector<float> Scales;
    std::vector<bool> Uint8_used;
    bool Quantized;
};

}  // namespace int8
}  // namespace lp
}  // namespace cpu
}  // namespace torch_ipex
