#include "DSMoE.h"
#include <aten/utils/amx.h>
#include <aten/utils/common.h>
#include <torch/all.h>
#include <torch/csrc/autograd/function.h>

#include <ATen/native/CPUBlas.h>
namespace torch_ipex {
namespace cpu {

IPEX_DEFINE_DISPATCH(fused_experts_impl_stub);

at::Tensor fused_experts(
    at::Tensor& hidden_states,
    at::Tensor& w1,
    at::Tensor& w2,
    at::Tensor& topk_weights,
    at::Tensor& topk_ids,
    bool inplace,
    bool is_vnni,
    bool is_distributed,
    bool is_woq,
    at::Tensor w1_scale,
    at::Tensor w1_zp,
    at::Tensor w2_scale,
    at::Tensor w2_zp) {
  RECORD_FUNCTION("ipex::fused_experts", c10::ArrayRef<c10::IValue>({}));

  return fused_experts_impl_stub(
      kCPU,
      hidden_states,
      w1,
      w2,
      topk_weights,
      topk_ids,
      inplace,
      is_vnni,
      is_distributed,
      is_woq,
      w1_scale,
      w1_zp,
      w2_scale,
      w2_zp);
}
constexpr int block_size_m() {
  return 1 * TILE_M;
}
constexpr int block_size_n() {
  return 8 * TILE_N;
}
// convert to vnni format
// from [N, K] to [K/2, N, 2] for bfloat16 and float16
//
// [N, K/2, 2] to [K/2, N, 2]
template <typename scalar_t>
inline void pack_vnni(
    scalar_t* __restrict__ packed,
    const scalar_t* __restrict__ weight,
    int N,
    int K) {
  for (int n = 0; n < N; ++n) {
    for (int k = 0; k < K / VNNI_BLK; ++k) {
      for (int d = 0; d < VNNI_BLK; ++d) {
        packed[k * N * VNNI_BLK + n * VNNI_BLK + d] =
            weight[n * K + k * VNNI_BLK + d];
      }
    }
  }
}


at::Tensor convert_weight_packed_bf16(at::Tensor& weight) {
  // weight : [E, OC, IC]
  //     w1 : [E, 2N,  K]
  //     w2 : [E,  K,  N]
  CHECK_DIM(3, weight);
  const auto st = weight.scalar_type();
  const int E = weight.size(0);
  const int OC = weight.size(1);
  const int IC = weight.size(2);
  // we handle 2 TILE_N at a time.
  TORCH_CHECK(OC % TILE_N == 0, "invalid weight out features ", OC);
  TORCH_CHECK(IC % TILE_K == 0, "invalid weight input features ", IC);
  constexpr int BLOCK_N = block_size_n();
  // use phony sizes here [E, OC, IC], for each [E], [OC, IC] -> [IC / 2, OC, 2]
  auto packed_weight = at::empty({E, OC, IC}, weight.options());
  const int stride = OC * IC;
  // TODO: add float8 support
  TORCH_CHECK(
      st == at::kBFloat16 || st == at::kHalf,
      "expect weight to be bfloat16 or float16.");
  AT_DISPATCH_REDUCED_FLOATING_TYPES(st, "conver_weight_packed_impl", [&] {
    const scalar_t* w_data = weight.data_ptr<scalar_t>();
    scalar_t* packed_data = packed_weight.data_ptr<scalar_t>();
    // parallel on {E}
    at::parallel_for(0, E, 0, [&](int begin, int end) {
      for (int e = begin; e < end; ++e) {
        for (int n = 0; n < OC; n += BLOCK_N) {
          int n_size = std::min(BLOCK_N, OC - n);
          pack_vnni<scalar_t>(
              packed_data + e * stride + n * IC,
              w_data + e * stride + n * IC,
              n_size,
              IC);
        }
      }
    });
  });

  return packed_weight;
}
} // namespace cpu
} // namespace torch_ipex

namespace {

TORCH_LIBRARY_FRAGMENT(torch_ipex, m) {
  m.def(
      "fused_experts(Tensor hidden_states, Tensor w1, Tensor w2, Tensor topk_weights, \
      Tensor topk_ids, bool inplace, bool is_vnni, \
       bool is_distributed, bool is_woq, Tensor w1_scale, Tensor w1_zp, Tensor w2_scale, Tensor w2_zp) -> Tensor");
  m.impl(
      "fused_experts", c10::DispatchKey::CPU, torch_ipex::cpu::fused_experts);
  m.def("convert_weight_packed_bf16(Tensor weight) -> Tensor");
  m.impl(
      "convert_weight_packed_bf16",
      c10::DispatchKey::CPU,
      torch_ipex::cpu::convert_weight_packed_bf16);
}
} // namespace