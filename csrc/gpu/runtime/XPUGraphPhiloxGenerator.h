#pragma once

#include <ATen/xpu/XPUGeneratorImpl.h>

namespace torch_ipex::xpu {

namespace dpcpp {

// Please distinguish this PhiloxXPUState against another
// PhiloxState defined in aten/operators/DistributionTemplates.h
// although both of the two structs are designed to store the state
// of using philox seed and offset.
//
// As PhiloxState who is used by operators and PhiloxXPUState who is
// used for graphs are very similar in code design and calls, we can try
// to reuse code into one single struct. But please remember in mind that
// it should be used in two different senerios.
struct PhiloxXPUState {
  PhiloxXPUState() = default;
  // Called if graph recording is not underway
  PhiloxXPUState(uint64_t seed, uint64_t offset) {
    seed_.val = seed;
    offset_.val = offset;
  }
  // Called if graph recording is underway
  PhiloxXPUState(
      int64_t* seed,
      int64_t* offset_extragraph,
      uint32_t offset_intragraph) {
    seed_.ptr = seed;
    offset_.ptr = offset_extragraph;
    offset_intragraph_ = offset_intragraph;
    recorded_ = true;
  }

  // If upstream this part into PyTorch's ATen as how CUDA does,
  // please also remember keep this union as public without getters/setters
  union Payload {
    uint64_t val;
    int64_t* ptr;
  };

  Payload seed_;
  Payload offset_;
  uint32_t offset_intragraph_ = 0;
  bool recorded_ = false;
};

class XPUGraphPhiloxGenerator : public at::XPUGeneratorImpl {
 public:
  XPUGraphPhiloxGenerator() = delete;
  XPUGraphPhiloxGenerator(at::XPUGeneratorImpl* gen);

  void graph_recording_prologue(
      int64_t* seed_extragraph,
      int64_t* offset_extragraph);
  uint64_t graph_recording_epilogue();
  PhiloxXPUState philox_xpu_state(uint64_t increment);
  at::XPUGeneratorImpl* get_generator() {
    return correlated_gen_;
  }

 private:
  int64_t* seed_extragraph_{};
  int64_t* offset_extragraph_{};
  uint32_t offset_intragraph_ = 0;
  bool graph_expects_this_gen_ = false;
  // FIXME:
  // Because the upstream at::XPUGeneratorImpl has not contained parts
  // which should be prepared for graph mode, that is why I made this
  // inherit class as an external extension and relink this extension
  // to the correct Generator used for the graph. Please relocate this
  // class and all methods implemented inside it as members into
  // at::XPUGeneratorImpl when upstream to PyTorch.
  at::XPUGeneratorImpl* correlated_gen_;
};

} // namespace dpcpp
} // namespace torch_ipex::xpu
