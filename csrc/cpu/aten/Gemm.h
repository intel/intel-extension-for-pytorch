#pragma once

#include <ATen/ATen.h>
#include <ATen/native/CPUBlas.h>
#include <dyndisp/DispatchStub.h>

namespace torch_ipex {
namespace cpu {

namespace {
// amx-bf16
#define TILE_M 16
#define TILE_N 16
#define TILE_K 32
#define VNNI_BLK 2
#define TILE_SIZE 512
// work around compiler internal error
#define BLOCK_K 128 // 4 * TILE_K

// block size for AMX gemm
constexpr int block_size_m() {
  return 1 * TILE_M;
}
int block_size_n() {
  return 4 * TILE_N;
}
at::Tensor bmm_forward_cpu(
    at::Tensor& out,
    at::Tensor& mat1,
    at::Tensor& mat2,
    bool is_vnni,
    const c10::optional<at::Tensor>& scale);

at::Tensor convert_weight_packed(
    at::Tensor& weight,
    bool use_tuned_block_n = false);
} // namespace

using bmm_kernel_fn = at::Tensor (*)(
    at::Tensor& out,
    at::Tensor& mat1,
    at::Tensor& mat2,
    bool is_vnni,
    const c10::optional<at::Tensor>& scale,
    const bool enforce_brgemm,
    const bool enforce_not_fp8,
    int block_n);
using convert_weight_packed_kernel_fn =
    at::Tensor (*)(at::Tensor& weight, bool use_tuned_block_n);
IPEX_DECLARE_DISPATCH(bmm_kernel_fn, bmm_kernel_stub);
IPEX_DECLARE_DISPATCH(
    convert_weight_packed_kernel_fn,
    convert_weight_packed_kernel_stub);

} // namespace cpu
} // namespace torch_ipex