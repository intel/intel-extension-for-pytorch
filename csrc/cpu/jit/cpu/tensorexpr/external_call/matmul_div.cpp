#include "matmul_div.h"

#include <ATen/ATen.h>
#include <torch/csrc/jit/tensorexpr/fwd_decls.h>
#include <torch/csrc/jit/tensorexpr/lowerings.h>
#include <torch/library.h>

#include "csrc/cpu/jit/cpu/kernels/Matmul.h"
#include "csrc/cpu/jit/cpu/tensorexpr/nnc_lowering_register.h"
#include "csrc/cpu/jit/cpu/tensorexpr/utils.h"

#include "csrc/cpu/utils/library.h"

namespace torch_ipex {
namespace jit {
namespace cpu {
namespace tensorexpr {

torch::jit::tensorexpr::Tensor computeMatmulDiv(
    const std::vector<torch::jit::tensorexpr::ArgValue>& inputs,
    const std::vector<torch::jit::tensorexpr::ExprHandle>& output_shape,
    const std::vector<torch::jit::tensorexpr::ExprHandle>& output_strides,
    const c10::optional<torch::jit::tensorexpr::ScalarType>& output_type,
    at::Device device) {
  using PytTeBufHandle = torch::jit::tensorexpr::BufHandle;
  using PytTeStmtPtr = torch::jit::tensorexpr::StmtPtr;
  using PytTeExternalCall = torch::jit::tensorexpr::ExternalCall;
  using PytTeTensor = torch::jit::tensorexpr::Tensor;

  auto te_dtype = torch::jit::tensorexpr::Dtype(*output_type);
  PytTeBufHandle result_buf(
      "nnc_ipex_matmul_div", output_shape, output_strides, te_dtype);
  const PytTeBufHandle& mm_a = std::get<PytTeBufHandle>(inputs[0]);
  const PytTeBufHandle& mm_b = std::get<PytTeBufHandle>(inputs[1]);
  const PytTeBufHandle& div_a = std::get<PytTeBufHandle>(inputs[2]);
  PytTeStmtPtr s = PytTeExternalCall::make(
      result_buf, "nnc_ipex_matmul_div", {mm_a, mm_b, div_a}, {});
  return PytTeTensor(result_buf.node(), s);
}

void nncMatmulDiv(
    int64_t bufs_num,
    void** buf_data,
    int64_t* buf_ranks,
    int64_t* buf_dims,
    int64_t* buf_strides,
    int8_t* buf_dtypes,
    int64_t args_num,
    int64_t* extra_args) {
  std::vector<at::Tensor> tensors = constructTensors(
      bufs_num, buf_data, buf_ranks, buf_dims, buf_strides, buf_dtypes);
  at::Tensor& r = tensors[0];
  const at::Tensor& mm_a = tensors[1];
  const at::Tensor& mm_b = tensors[2];
  const at::Tensor& div_a = tensors[3];
  torch_ipex::cpu::dil_matmul_div(mm_a, mm_b, r, div_a);
}

} // namespace tensorexpr
} // namespace cpu
} // namespace jit
} // namespace torch_ipex

namespace {
using namespace torch_ipex::jit::cpu::tensorexpr;
static NNCOperatorRegister _nnc_ipex_matmul_div(
    "ipex::matmul_div(Tensor left, Tensor right,  Tensor div_input) -> Tensor",
    "nnc_ipex_matmul_div",
    computeMatmulDiv,
    nncMatmulDiv);
} // namespace
