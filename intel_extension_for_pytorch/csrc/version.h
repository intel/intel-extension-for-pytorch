#pragma once

#include <string>

namespace torch_ipex {

const std::string __version__();
const std::string __gitrev__();
const std::string __torch_gitrev__();
const std::string __mode__();

} // namespace torch_ipex
