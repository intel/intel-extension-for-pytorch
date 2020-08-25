#pragma once

namespace torch_ipex {
namespace cpu {
namespace lp {
namespace int8 {

struct Observer {
  int64_t Id;
  std::string Name;
  // the max_values of input and output for one op
  std::vector<float> max_values;
};

class Indicator {
  public:
    Indicator(int64_t id = 0, std::string name = "", std::vector<float> scales = std::vector<float>(2, 1),
        std::vector<bool> uint8_used = std::vector<bool>(2, false) , bool quantized = true):
      Id(id), Name(name), Scales(scales), Uint8_used(uint8_used), Quantized(quantized) {}

  int64_t get_indicator_id() {
    return Id;
  }

  std::string get_indicator_name() {
    return Name;
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
    std::vector<float> Scales;
    std::vector<bool> Uint8_used;
    bool Quantized;
};

}  // namespace int8
}  // namespace lp
}  // namespace cpu
}  // namespace torch_ipex
