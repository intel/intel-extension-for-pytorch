#include "onednn_utils.h"

#include "ideep/ideep.hpp"

namespace torch_ipex {
namespace utils {

int onednn_set_verbose(int level) {
  return ideep::utils::set_verbose(level);
}

bool onednn_has_bf16_type_support() {
  return ideep::has_bf16_type_support();
}

} // namespace utils
} // namespace torch_ipex
