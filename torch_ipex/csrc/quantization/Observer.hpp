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
  // "int8" means input will be quantized to int8, otherwise quantized to "uint8".
  std::vector<std::string> input_quantized_dtypes;
  std::vector<std::string> output_quantized_dtypes;
  std::vector<bool> inputs_quantized;
  std::vector<bool> outputs_quantized;
  std::vector<std::string> inputs_flow;
  std::vector<std::string> outputs_flow;
};

class Indicator {
public:
  Indicator(int64_t i, std::string n,
            std::string alg,
            std::string granu,
            std::vector<float> i_scale,
            std::vector<float> w_scales,
            std::vector<float> o_scale,
            std::vector<std::string> i_quantized_dtype,
            std::vector<std::string> o_quantized_dtype,
            std::vector<bool> inputs_quant,
            std::vector<bool> outputs_quant,
            std::vector<std::string> i_flow,
            std::vector<std::string> o_flow )
      : id(i), name(n), algorithm(alg), weight_granularity(granu),
        inputs_scale(i_scale), weight_scales(std::move(w_scales)), outputs_scale(o_scale),
        input_quantized_dtypes(i_quantized_dtype), output_quantized_dtypes(o_quantized_dtype),
        inputs_quantized(inputs_quant), outputs_quantized(outputs_quant),
        inputs_flow(i_flow), outputs_flow(o_flow) {}

    Indicator(const Indicator& other){
      UniqueReadLock<ReadWriteMutex> lock(rwmutex);
      id = other.id;
      name = other.name;
      algorithm = other.algorithm;
      weight_granularity = other.weight_granularity;
      inputs_scale = other.inputs_scale;
      weight_scales = other.weight_scales;
      outputs_scale = other.outputs_scale;
      input_quantized_dtypes = other.input_quantized_dtypes;
      output_quantized_dtypes = other.output_quantized_dtypes;
      inputs_quantized = other.inputs_quantized;
      outputs_quantized = other.outputs_quantized;
      inputs_flow = other.inputs_flow;
      outputs_flow = other.outputs_flow;
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

  std::tuple<std::vector<std::string>, std::vector<std::string>>
  get_indicator_quantized_dtypes() {
    UniqueReadLock<ReadWriteMutex> lock(rwmutex);
    return std::make_tuple(input_quantized_dtypes, output_quantized_dtypes);
  }

  std::tuple<std::vector<bool>, std::vector<bool>> get_indicator_insert_quantized_status() {
    return std::make_tuple(inputs_quantized, outputs_quantized);
  }

  void set_indicator_scales(std::vector<float> new_inputs_scale,
                            std::vector<float> new_outputs_scale) {
    UniqueWriteLock<ReadWriteMutex> lock(rwmutex);
    inputs_scale = new_inputs_scale;
    outputs_scale = new_outputs_scale;
  }

  void set_indicator_quantized_dtypes(std::vector<std::string> new_input_quantized_dtypes,
                                  std::vector<std::string> new_output_quantized_dtypes) {
    UniqueWriteLock<ReadWriteMutex> lock(rwmutex);
    input_quantized_dtypes = new_input_quantized_dtypes;
    output_quantized_dtypes = new_output_quantized_dtypes;
  }

  std::tuple<std::vector<std::string>, std::vector<std::string>> get_indicator_quantized_flow() {
    return std::make_tuple(inputs_flow, outputs_flow);
  }

private:
  int64_t id;
  std::string name;
  std::string algorithm;
  std::string weight_granularity;
  std::vector<float> inputs_scale;
  std::vector<float> weight_scales;
  std::vector<float> outputs_scale;
  std::vector<std::string> input_quantized_dtypes;
  std::vector<std::string> output_quantized_dtypes;
  std::vector<bool> inputs_quantized;
  std::vector<bool> outputs_quantized;
  std::vector<std::string> inputs_flow;
  std::vector<std::string> outputs_flow;
  mutable ReadWriteMutex rwmutex;
};

} // namespace int8
} // namespace torch_ipex
