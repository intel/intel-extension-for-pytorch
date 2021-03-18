#pragma once

#include "torch_ipex/csrc/rw_lock.h"

namespace torch_ipex {
namespace int8 {

struct Observer {
  int64_t id;
  std::string name;
  std::vector<std::vector<float>> inputs_min_max_values;
  std::vector<std::vector<float>> weight_min_max_values; // per_channel or per_tensor
  std::vector<std::vector<float>> outputs_min_max_values;
  // default uising min/max to compute the quantization parameters,
  // only support min_max, MovingAverageMinMax and other none per_channel
  // merthod
  std::string algorithm = "min_max";
  float averaging_constant = 0.01; // for MovingAverage method
  // only useful for conv, onednn only support per_channel foo conv's weight,
  // default is per_channel
  std::string weight_granularity = "per_channel";
  // ture means input will be quantized to int8, otherwise quantized to uint8.
  std::vector<bool> inputs_dtype_uint8 = {false};
  std::vector<bool> outputs_dtype_uint8 = {false};
  bool quantized = true;
  bool pre_quantized = true;
  bool post_quantized = true;
};

class Indicator {
public:
  Indicator(int64_t i = 0, std::string n = "", std::string alg = "min_max",
            std::string granu = "per_tensor", std::vector<float> i_scale = {1},
            std::vector<float> w_scales = {},
            std::vector<float> o_scale = {1},
            std::vector<bool> i_uint8_used = {false},
            std::vector<bool> o_uint8_used = {false}, bool quant = true,
            bool pre_quant = true, bool post_quant = true)
      : id(i), name(n), algorithm(alg), weight_granularity(granu),
        inputs_scale(i_scale), weight_scales(std::move(w_scales)), outputs_scale(o_scale),
        inputs_uint8_used(i_uint8_used), outputs_uint8_used(o_uint8_used),
        quantized(quant), pre_quantized(pre_quant), post_quantized(post_quant) {}
    Indicator(const Indicator& other){
      UniqueReadLock<ReadWriteMutex> lock(rwmutex);
      id = other.id;
      name = other.name;
      algorithm = other.algorithm;
      weight_granularity = other.weight_granularity;
      inputs_scale = other.inputs_scale;
      weight_scales = other.weight_scales;
      outputs_scale = other.outputs_scale;
      inputs_uint8_used = other.inputs_uint8_used;
      outputs_uint8_used = other.outputs_uint8_used;
      quantized = other.quantized;
      pre_quantized = other.pre_quantized;
      post_quantized = other.post_quantized;
    }

  int64_t get_indicator_id() { return id; }

  std::string get_indicator_name() { return name; }

  std::string get_indicator_algorithm() { return algorithm; }

  std::string get_indicator_weight_granularity() { return weight_granularity; }

  std::tuple<std::vector<float>, std::vector<float>> get_indicator_scales() {
    UniqueReadLock<ReadWriteMutex> lock(rwmutex);
    return std::make_tuple(inputs_scale, outputs_scale);
  }

  std::vector<float> get_indicator_weight_scales() {
    return weight_scales;
  }

  std::tuple<std::vector<bool>, std::vector<bool>>
  get_indicator_uint8_status() {
    UniqueReadLock<ReadWriteMutex> lock(rwmutex);
    return std::make_tuple(inputs_uint8_used, outputs_uint8_used);
  }

  bool get_indicator_quantized_status() { return quantized; }

  std::tuple<bool, bool> get_indicator_insert_quantized_status() {
    return std::make_tuple(pre_quantized, post_quantized);
  }

  void set_indicator_scales(std::vector<float> new_inputs_scale,
                            std::vector<float> new_outputs_scale) {
    UniqueWriteLock<ReadWriteMutex> lock(rwmutex);
    inputs_scale = new_inputs_scale;
    outputs_scale = new_outputs_scale;
  }

  void set_indicator_uint8_status(std::vector<bool> new_inputs_uint8_used,
                                  std::vector<bool> new_outputs_uint8_used) {
    UniqueWriteLock<ReadWriteMutex> lock(rwmutex);
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
  std::vector<float> weight_scales;
  std::vector<float> outputs_scale;
  std::vector<bool> inputs_uint8_used;
  std::vector<bool> outputs_uint8_used;
  bool quantized;
  bool pre_quantized;
  bool post_quantized;
  mutable ReadWriteMutex rwmutex;
};

} // namespace int8
} // namespace torch_ipex
