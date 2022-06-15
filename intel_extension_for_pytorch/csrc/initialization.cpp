#include "initialization.h"
#include <stdio.h>
#include <torch/csrc/jit/passes/autocast.h>
#include <torch/csrc/jit/passes/pass_manager.h>
#include <torch/torch.h>
#include <regex>
#include <string>
#include "intel_extension_for_pytorch/csrc/jit/auto_opt_config.h"
#include "intel_extension_for_pytorch/csrc/jit/fusion_pass.h"
#include "intel_extension_for_pytorch/csrc/version.h"

namespace torch_ipex {

void init_jit_fusion_pass() {
  // jit fusion pass
  torch::jit::registerPrePass([](std::shared_ptr<torch::jit::Graph>& g) {
    if (AutoOptConfig::singleton().get_jit_fuse()) {
      torch::jit::FusionPass(g);
    }
  });
}

void disable_autocast_for_jit_script() {
  // We need disable autocast pass by default after
  // https://github.com/pytorch/pytorch/pull/74178. Will remove this after we
  // can extend the cast policy for this autocast pass.
  torch::jit::setAutocastMode(false);
}

InitIPEX::InitIPEX() = default;
InitIPEX::~InitIPEX() = default;
InitIPEX::InitIPEX(InitIPEX&&) noexcept = default;
InitIPEX& InitIPEX::operator=(InitIPEX&&) noexcept = default;

static auto init = InitIPEX()
                       .init(&init_jit_fusion_pass)
                       .init(&disable_autocast_for_jit_script);

void InitIPEX::check_pytorch_version() {
  int IPEX_VERSION_MAJOR = 0;
  int IPEX_VERSION_MINOR = 0;
  const std::regex regex("(\\d+)\\.(\\d+).*");
  const std::string ipex_version = torch_ipex::__version__();
  std::smatch match;
  if (std::regex_match(ipex_version, match, regex)) {
    if (match.size() == 3) {
      IPEX_VERSION_MAJOR = std::stoi(match[1]);
      IPEX_VERSION_MINOR = std::stoi(match[2]);
    }
  }
  if (IPEX_VERSION_MAJOR != TORCH_VERSION_MAJOR ||
      IPEX_VERSION_MINOR != TORCH_VERSION_MINOR) {
    printf(
        "ERROR! Intel® Extension for PyTorch* needs to work with PyTorch/libtorch %d.%d.*, but PyTorch/libtorch %d.%d.%d is found. Please switch to the matching version and run again.\n",
        IPEX_VERSION_MAJOR,
        IPEX_VERSION_MINOR,
        TORCH_VERSION_MAJOR,
        TORCH_VERSION_MINOR,
        TORCH_VERSION_PATCH);
    exit(127);
  }
}

} // namespace torch_ipex
