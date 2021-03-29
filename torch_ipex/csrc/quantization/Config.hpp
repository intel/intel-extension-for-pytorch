#pragma once

#include <ATen/ATen.h>

#include <ATen/Tensor.h>
#include <torch/torch.h>

#include "Observer.hpp"

namespace torch_ipex {

using namespace int8;

class Int8OptConfig {
public:
  static Int8OptConfig &get_config() {
    static Int8OptConfig int8_opt_conf;
    return int8_opt_conf;
  }

public:
  void insert_or_updata_observer(std::string op_name,
                                 std::vector<std::vector<float>> i_min_max_values,
                                 std::vector<std::vector<float>> w_min_max_values,
                                 std::vector<std::vector<float>> o_min_max_values,
                                 int64_t ops_id,
                                 std::vector<std::string> inputs_flow,
                                 std::vector<std::string> output_flow);

  void clear_indicators();

  void add_indicators();

  std::vector<std::vector<float>>
  get_indicator_scales(std::vector<bool> i_uint8_used,
                       std::vector<bool> o_uint8_used, int64_t ops_id);

  std::string get_indicator_weight_granularity(const int64_t ops_id);

  float get_indicator_weight_scale(const int64_t ops_id);

  at::Tensor& get_indicator_weight_tensor_scale(const int64_t ops_id);

  bool get_indicator_quantized_status(const int64_t ops_id);

  std::tuple<bool, bool> get_indicator_insert_quantized_status(const int64_t ops_id);

  void set_indicators(std::vector<Indicator> indicators);

  std::vector<Indicator> get_indicators();

  int64_t get_indicators_size();

  static void calibration_reset();

  static int64_t fetch_and_add_ops_id();

private:
  Int8OptConfig() : observers_{}, indicators_{} {}
  ~Int8OptConfig() = default;
  Int8OptConfig(const Int8OptConfig &) = default;
  Int8OptConfig &operator=(const Int8OptConfig &) = default;

private:
  std::vector<Observer> observers_;
  std::vector<Indicator> indicators_;
  std::unordered_map<int64_t, at::Tensor> weights_scales_;
  thread_local static int64_t current_ops_id;
};

} // namespace torch_ipex
