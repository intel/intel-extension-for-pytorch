#include "initialization.h"
#include <torch/csrc/jit/passes/pass_manager.h>
#include "intel_extension_for_pytorch/csrc/jit/fusion_pass.h"
#include "intel_extension_for_pytorch/csrc/quantization/auto_opt_config.hpp"

namespace torch_ipex {

void init_jit_fusion_pass() {
  // jit fusion pass
  torch::jit::registerPrePass([](std::shared_ptr<torch::jit::Graph>& g) {
    if (AutoOptConfig::singleton().get_jit_fuse()) {
      torch::jit::FusionPass(g);
    }
  });
}

InitIPEX::InitIPEX() = default;
InitIPEX::~InitIPEX() = default;
InitIPEX::InitIPEX(InitIPEX&&) noexcept = default;
InitIPEX& InitIPEX::operator=(InitIPEX&&) noexcept = default;

static auto init = InitIPEX().init(&init_jit_fusion_pass);

} // namespace torch_ipex
