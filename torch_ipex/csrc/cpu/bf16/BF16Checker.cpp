#include "BF16Checker.h"

#include "torch_ipex/csrc/utils.h"
#include "torch_ipex/csrc/auto_opt_config.h"

namespace torch_ipex {
namespace cpu {
namespace bf16 {
namespace chk {

bool bf16_support_the_tensors(const std::vector<at::Tensor> &tensor_vec) {
  for (auto it = tensor_vec.begin(); it != tensor_vec.end(); ++it) {
    if (!check_tensor_own_whole_storage(*it)) {
      return false;
    }
  }

  return true;
}

}  // namespace chk
}  // namespace bf16
}  // namespace cpu
}  // namespace torch_ipex
