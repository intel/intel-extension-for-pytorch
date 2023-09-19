#pragma once
#include <ATen/ATen.h>
#include <Macros.h>

namespace torch_ipex {

class IPEX_API AutoOptConfig {
 public:
  static AutoOptConfig& singleton() {
    static AutoOptConfig auto_opt_conf;
    return auto_opt_conf;
  }
  inline void set_jit_fuse(bool jit_fuse) {
    jit_fuse_ = jit_fuse;
  }

  inline bool get_jit_fuse() {
    return jit_fuse_;
  }

  inline void set_jit_repack_for_linear(bool jit_repack_for_linear) {
    jit_repack_for_linear_ = jit_repack_for_linear;
  }

  inline bool get_jit_repack_for_linear() {
    return jit_repack_for_linear_;
  }

 private:
  AutoOptConfig()
      : jit_fuse_(true),
        // jit repack  (ipex linear -> aten linear -> ipex linear) will use
        // extra memory since the orinal graph will be always hold by design
        // https://github.com/pytorch/pytorch/blob/8e2a86c2a54719fd66a3e612fe8b433fbb1d4522/torch/csrc/jit/runtime/profiling_graph_executor_impl.cpp#L668
        // We use this flag to let custom disable repack to same meory
        // This is default False for 2 reasons:
        //    (1) JIT repack stage can get a real input, so the block format
        //    will be the best format. (2) Linear + binary cannot be folded if
        //    we do not do repack, since it is implemented on aten:linear
        jit_repack_for_linear_(true),
        calibration_step_(false),
        qscheme_(at::QScheme::PER_TENSOR_AFFINE) {}

  ~AutoOptConfig() = default;
  AutoOptConfig(const AutoOptConfig&) = default;
  AutoOptConfig& operator=(const AutoOptConfig&) = default;

  bool jit_fuse_;
  bool jit_repack_for_linear_;
  // the flag for one iteration of calibration step whether end or not.
  bool calibration_step_;
  at::QScheme qscheme_;
};

} // namespace torch_ipex
