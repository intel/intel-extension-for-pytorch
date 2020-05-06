#pragma once

namespace torch_ipex {

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

  inline void set_mix_bf16_fp32(bool value) {
    mix_bf16_fp32_ = value;
  }
  inline bool get_mix_bf16_fp32() {
    return mix_bf16_fp32_;
  }

  inline void set_pure_bf16(bool value) {
    pure_bf16_ = value;
  }
  inline bool get_pure_bf16() {
    return pure_bf16_;
  }

private:
  AutoOptConfig() : auto_dnnl_(false), mix_bf16_fp32_(false), pure_bf16_(false) {}
  ~AutoOptConfig() = default;
  AutoOptConfig(const AutoOptConfig&) = default;
  AutoOptConfig& operator=(const AutoOptConfig&) = default;

private:
  bool auto_dnnl_;
  bool mix_bf16_fp32_;
  bool pure_bf16_;
};

} // namespace torch_ipex
