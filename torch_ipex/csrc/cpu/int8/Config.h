#pragma once
#include "cpu/int8/quantization/Observer.h"


namespace torch_ipex {

using namespace torch_ipex::cpu::lp::int8;

class Int8OptConfig {
public:
  static Int8OptConfig &get_config() {
    static Int8OptConfig int8_opt_conf;
    return int8_opt_conf;
  }

public:
  void insert_or_updata_observer(
      std::string op_name, std::vector<std::vector<float>> i_min_max_values,
      std::vector<std::vector<float>> o_min_max_values, int64_t ops_id);

  void clear_indicators();

  void add_indicators();

  std::vector<std::vector<float>>
  get_indicator_scales(std::vector<bool> i_uint8_used,
                       std::vector<bool> o_uint8_used, int64_t ops_id);

  bool get_indicator_quantized_status(int64_t ops_id);

  void set_indicators(std::vector<Indicator> indicators);

  std::vector<Indicator> get_indicators();

  void calibration_reset();

  int64_t fetch_and_add_ops_id();

private:
  Int8OptConfig() : observers_{}, indicators_{} {}
  ~Int8OptConfig() = default;
  Int8OptConfig(const Int8OptConfig &) = default;
  Int8OptConfig &operator=(const Int8OptConfig &) = default;

private:
  std::vector<Observer> observers_;
  std::vector<Indicator> indicators_;

public:
  thread_local static int64_t current_ops_id;
};

} // namespace torch_ipex
