#pragma once
#include <ATen/native/quantized/cpu/quant_utils.h>
#include "torch_ipex/csrc/rw_lock.h"

namespace torch_ipex {
namespace int8 {

using namespace quant_utils;

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
            std::vector<TensorQuantizationParams> i_params,
            std::vector<float> w_scales,
            std::vector<TensorQuantizationParams> o_params,
            std::vector<std::string> i_quantized_dtypes,
            std::vector<std::string> o_quantized_dtypes,
            std::vector<bool> inputs_quant,
            std::vector<bool> outputs_quant,
            std::vector<std::string> i_flow,
            std::vector<std::string> o_flow )
      : id(i), name(n), algorithm(alg), weight_granularity(granu),
        input_params(i_params), weight_scales(std::move(w_scales)), output_params(o_params),
        input_quantized_dtypes(i_quantized_dtypes), output_quantized_dtypes(o_quantized_dtypes),
        inputs_quantized(inputs_quant), outputs_quantized(outputs_quant),
        inputs_flow(i_flow), outputs_flow(o_flow) {}

    Indicator(const Indicator& other){
      UniqueReadLock<ReadWriteMutex> lock(rwmutex);
      id = other.id;
      name = other.name;
      algorithm = other.algorithm;
      weight_granularity = other.weight_granularity;
      input_params = other.input_params;
      weight_scales = other.weight_scales;
      output_params = other.output_params;
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

  std::tuple<std::vector<TensorQuantizationParams>, std::vector<TensorQuantizationParams>> get_indicator_scales() {
    UniqueReadLock<ReadWriteMutex> lock(rwmutex);
    return std::make_tuple(input_params, output_params);
  }

  std::vector<float> get_indicator_weight_scales() {
    return weight_scales;
  }

  std::tuple<std::vector<std::string>, std::vector<std::string>> get_indicator_quantized_dtypes() {
    UniqueReadLock<ReadWriteMutex> lock(rwmutex);
    return std::make_tuple(input_quantized_dtypes, output_quantized_dtypes);
  }

  std::tuple<std::vector<bool>, std::vector<bool>> get_indicator_insert_quantized_status() {
    return std::make_tuple(inputs_quantized, outputs_quantized);
  }

  void set_indicator_scales(std::vector<TensorQuantizationParams> new_input_params,
                            std::vector<TensorQuantizationParams> new_output_params) {
    UniqueWriteLock<ReadWriteMutex> lock(rwmutex);
    input_params = new_input_params;
    output_params = new_output_params;
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
  std::vector<TensorQuantizationParams> input_params;
  std::vector<TensorQuantizationParams> output_params; 
  std::vector<float> weight_scales;
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
