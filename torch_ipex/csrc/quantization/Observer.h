#pragma once

namespace torch_ipex {
namespace cpu {
namespace lp {
namespace int8 {


struct Operator {
  uint64_t id;
  uint64_t op_id;
};

struct InputMeta {
  uint64_t dims;
  uint64_t *sizes;
};

struct Observer {
  //Operator op;
  // int input_tensors_num;
  // InputMeta *input_tensors;
  //InputMeta input_meta;
  int64_t id;
  std::vector<int64_t> input_sizes;
  int64_t channel_axis;
  std::vector<float> mins;
  std::vector<float> maxs;
  //float *mins;
  //float *maxs;
};

struct Indicator {
  // Operator op;
  // int input_tensors_num;
  // InputMeta *input_tensors;
  //InputMeta input_meta;
  int64_t id;
  std::vector<int64_t> input_sizes;
  int64_t channel_axis;
  std::vector<float> scales;
  std::vector<float> zero_points;
  bool uint8_used;
  //float *zero_points_;
  //float *scales_;
};

}  // namespace int8
}  // namespace lp
}  // namespace cpu
}  // namespace torch_ipex
