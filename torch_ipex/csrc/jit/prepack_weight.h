#pragma once

#include <torch/csrc/jit/api/module.h>

namespace torch{
namespace jit{

Module prepack_conv_weight(const Module& module);

}

}
