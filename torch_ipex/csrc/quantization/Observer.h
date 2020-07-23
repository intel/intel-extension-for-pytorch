#pragma once

namespace torch_ipex {
namespace cpu {
namespace lp {
namespace int8 {

struct Observer {
  int64_t id;
  float max_value;
};

struct Indicator {
  int64_t id;
  float scale;
  float zero_point;
  bool uint8_used;
};

}  // namespace int8
}  // namespace lp
}  // namespace cpu
}  // namespace torch_ipex
