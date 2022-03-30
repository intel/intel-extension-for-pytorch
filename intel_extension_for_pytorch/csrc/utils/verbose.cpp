#include "verbose.hpp"

#include "csrc/cpu/ideep/ideep.hpp"

namespace torch_ipex {
namespace verbose {

int _mkldnn_set_verbose(int level) {
  return ideep::utils::set_verbose(level);
}

} // namespace verbose
} // namespace torch_ipex
