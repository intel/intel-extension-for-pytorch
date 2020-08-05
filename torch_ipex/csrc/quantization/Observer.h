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

  std::vector<float> get_indicator_scales() {
    return Scales;
  }

  void set_indicator_scales(std::vector<float> new_scales) {
    Scales = new_scales;
  }

  std::vector<bool> get_indicator_uint8_status() {
    return Uint8_used;
  }

  void set_indicator_uint8_status(std::vector<bool> new_uint8_used) {
    Uint8_used = new_uint8_used;
  }

  bool get_indicator_quantized_status() {
    return Quantized;
  }

  void set_indicator_quantized_status(bool new_quantized) {
    Quantized = new_quantized;
  }

  friend std::ostream & operator << (std::ostream &out, const Indicator& obj) {
    out << obj.Id <<"\n"<<obj.Name<< "\n";
    out << obj.Scales[0] << "\n" << obj.Scales[1] << "\n";
    out << obj.Uint8_used[0] << "\n" << obj.Uint8_used[1] << "\n";
    out << obj.Quantized << "\n" << std::endl;
    return out;
  }

  friend std::istream & operator >> (std::istream &in, Indicator &obj) {
    in >> obj.Id;
    in >> obj.Name;
    in >> obj.Scales[0];
    in >> obj.Scales[1];
    bool temp;
    in >> temp;
    obj.Uint8_used[0] = temp;
    in >> temp;
    obj.Uint8_used[1] = temp;
    in >> obj.Quantized;
    return in;
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
