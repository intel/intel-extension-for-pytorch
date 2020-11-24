#pragma once

#include <ATen/ATen.h>
#include "Sum.h"
#include "Binary.h"


namespace at {
namespace dpcpp {
namespace oneDNN {

enum post_attr {
  with_relu = 0b01,
  with_sum = 0b10,
  with_sigmoid = 0b100,
};

}}}
