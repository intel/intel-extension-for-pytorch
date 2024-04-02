#include <ATen/Tensor.h>
#include <c10/util/Float8_e4m3fn.h>
#include <c10/util/Float8_e5m2.h>

namespace torch_ipex {
namespace cpu {

using namespace at;

using fp8e5m2 = at::Float8_e5m2;
using fp8e4m3 = at::Float8_e4m3fn;

enum Float8Format {
  NOT_VALID = 0,
  kFloat8_E5M2 = 1,
  kFloat8_E4M3 = 2,
};

// Each tensor here is shape (N, ) holding all scaling
// data for a single FP8 block, e.g. LayerNormLinear
class FP8TensorMeta {
 public:
  at::Tensor scale;
  at::Tensor scale_inv;
  at::Tensor amax_history;
};

// Used as named indices on the `scale`, `scale_inv`,
// and `amax` tensors in the `FP8TensorMeta` class.
enum FP8FwdTensors {
  GEMM1_INPUT = 0,
  GEMM1_WEIGHT = 1,
  GEMM1_OUTPUT = 2,
  GEMM2_INPUT = 3,
  GEMM2_WEIGHT = 4,
  GEMM2_OUTPUT = 5
};

// Used as named indices on the `scale`, `scale_inv`,
// and `amax` tensors in the `FP8TensorMeta` class.
enum FP8BwdTensors {
  GRAD_OUTPUT1 = 0,
  GRAD_INPUT1 = 1,
  GRAD_OUTPUT2 = 2,
  GRAD_INPUT2 = 3
};

template <typename T>
struct is_fp8 : std::false_type {};

template <>
struct is_fp8<fp8e4m3> : std::true_type {};

template <>
struct is_fp8<fp8e5m2> : std::true_type {};

#define FP8_CHECK(stat, msg) stat ? Status::OK() : errors::InvalidArgument(msg)

#define IPEX_TYPE_SWITCH_FP8ONLY(dtype, type, ...) \
  switch (dtype) {                                 \
    case Float8Format::kFloat8_E5M2: {             \
      using type = fp8e5m2;                        \
      { __VA_ARGS__ }                              \
    } break;                                       \
    case Float8Format::kFloat8_E4M3: {             \
      using type = fp8e4m3;                        \
      { __VA_ARGS__ }                              \
    } break;                                       \
    default:                                       \
      TORCH_CHECK(false, "invalid type!!\n");      \
      break;                                       \
  }

#define FP8_PTR(PTR, TYPE) reinterpret_cast<TYPE*>(PTR)

} // namespace cpu
} // namespace torch_ipex