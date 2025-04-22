#pragma once

#include <ATen/ATen.h>
#include <ATen/native/ReduceOpsUtils.h>
#include <core/Memory.h>
#include <runtime/Utils.h>
#include <utils/DPCPP.h>
#include <cmath>

using namespace at::native;
using namespace torch_ipex::xpu::dpcpp;

namespace at {
namespace AtenIpexTypeXPU {} // namespace AtenIpexTypeXPU
} // namespace at
