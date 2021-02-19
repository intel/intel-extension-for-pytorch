#pragma once

#include "cpu/int8/quantization/Observer.h"
#include "utils.h"
namespace torch_ipex {

using namespace torch_ipex::cpu::lp::int8;

enum class XPUMode  {CPU, GPU, FPGA};

inline XPUMode stringToXPUMode(std::string mode){
   if(mode == "CPU"){
     return XPUMode::CPU;
   }else{
     throw std::runtime_error("Unknown xpu mode");     
   }
}

class AutoOptConfig {
public:
  static AutoOptConfig& singleton() {
    static AutoOptConfig auto_opt_conf;
    return auto_opt_conf;
  }

public:
  inline void set_auto_dnnl(bool auto_dnnl) {
    auto_dnnl_ = auto_dnnl;
  }
  inline bool get_auto_dnnl() {
    return auto_dnnl_;
  }

  inline void set_jit_fuse(bool jit_fuse) {
    jit_fuse_ = jit_fuse;
  }

  inline bool get_jit_fuse() {
    return jit_fuse_;
  }

  // bf16
  inline void set_mix_bf16_fp32(bool value) {
    mix_bf16_fp32_ = value;
  }
  inline bool get_mix_bf16_fp32() {
    return mix_bf16_fp32_;
  }

  inline void set_train(bool value) {
    train_ = value;
  }
  inline bool get_train() {
    return train_;
  }

  // int8
  inline void set_mix_int8_fp32(bool value) {
    mix_int8_fp32_ = value;
  }
  inline bool get_mix_int8_fp32() {
    return mix_int8_fp32_;
  }

  inline void set_int8_calibration(bool value) {
    calibration_step_ = value;
  }
  inline bool get_int8_calibration() {
    return calibration_step_;
  }
  inline void set_xpu_mode(XPUMode xpu_mode){
    xpu_mode_ = xpu_mode;
  }
 
  inline XPUMode get_xpu_mode(){
    return xpu_mode_;
  }

private:
  AutoOptConfig() : auto_dnnl_(true), mix_bf16_fp32_(false), mix_int8_fp32_(false),
                    jit_fuse_(true), train_(false), calibration_step_(false), xpu_mode_(XPUMode::CPU) {}

  ~AutoOptConfig() = default;
  AutoOptConfig(const AutoOptConfig&) = default;
  AutoOptConfig& operator=(const AutoOptConfig&) = default;

private:
  bool auto_dnnl_;
  bool jit_fuse_;
  bool mix_bf16_fp32_;
  bool train_;
  // int8
  bool mix_int8_fp32_;
  // the flag for one iteration of calibration step whether end or not
  bool calibration_step_;
  XPUMode xpu_mode_;
};

} // namespace torch_ipex
