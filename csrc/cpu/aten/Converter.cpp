#include "Converter.h"
#include <ATen/Parallel.h>
#include <torch/all.h>
#include "csrc/utils/CustomOperatorRegistration.h"

namespace torch_ipex {
namespace cpu {

IPEX_DEFINE_DISPATCH(cat_bfloat16_float_kernel_stub);
IPEX_DEFINE_DISPATCH(split_float_bfloat16_kernel_stub);

} // namespace cpu
} // namespace torch_ipex

namespace torch_ipex {
namespace cpu {
namespace bf16 {
namespace converter {

using namespace torch_ipex::cpu;

at::Tensor cat_bfloat16_float(
    const at::Tensor top_half_,
    const at::Tensor bottom_half_) {
  TORCH_CHECK(
      top_half_.scalar_type() == at::kBFloat16 &&
          bottom_half_.scalar_type() == at::kBFloat16,
      "pack_bfloat16_float: expect both args to be at::BFloat16");

  // pointer to cat_bfloat16_float_kernel_impl(top_half_, bottom_half_);
  return cat_bfloat16_float_kernel_stub(kCPU, top_half_, bottom_half_);
}

std::tuple<at::Tensor, at::Tensor> split_float_bfloat16(
    const at::Tensor tensor_) {
  TORCH_CHECK(
      tensor_.scalar_type() == at::kFloat,
      "pack_bfloat16_float: expect both tensor to be at::kFloat");

  // pointer to split_float_bfloat16_kernel_impl(tensor_);
  return split_float_bfloat16_kernel_stub(kCPU, tensor_);
}

} // namespace converter
} // namespace bf16
} // namespace cpu
} // namespace torch_ipex

namespace {

IPEX_LIBRARY_FRAGMENT() {
  IPEX_OP_IPEX_REGISTER_DISPATCH(
      "split_float_bfloat16",
      torch_ipex::cpu::bf16::converter::split_float_bfloat16,
      c10::DispatchKey::CPU);
  IPEX_OP_IPEX_REGISTER_DISPATCH(
      "cat_bfloat16_float",
      torch_ipex::cpu::bf16::converter::cat_bfloat16_float,
      c10::DispatchKey::CPU);
}

} // namespace
