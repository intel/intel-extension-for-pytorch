
#include <ATen/ATen.h>
#include <ATen/CPUApplyUtils.h>
#include <ATen/record_function.h>
#include <CL/sycl.hpp>
#include <core/detail/IndexUtils.h>
#include <core/detail/TensorInfo.h>
#include <runtime/Utils.h>
#include <stdlib.h>
#include <utils/oneMKLUtils.h>
#include "../comm/ATDispatch.h"
#include "../comm/AccumulateType.h"
#include "../esimd/api.hpp"
#include "utils/CustomOperatorRegistration.h"

using namespace torch_ipex::xpu::dpcpp::detail;
using namespace torch_ipex::xpu::dpcpp;
using namespace at::native;

namespace at {
namespace AtenIpexTypeXPU {

static Tensor fmha_esimd(
    const Tensor& query,
    const Tensor& key,
    const Tensor& value,
    const Tensor& mask,
    bool is_head_first) {
  TORCH_CHECK(
      query.scalar_type() == at::kHalf, "IPEX SDP only supports half datatype");
  TORCH_CHECK(
      key.scalar_type() == at::kHalf, "IPEX SDP only supports half datatype");
  TORCH_CHECK(
      value.scalar_type() == at::kHalf, "IPEX SDP only supports half datatype");

  uint32_t num_batches = query.size(0);
  uint32_t num_heads_q = query.size(1);
  uint32_t num_heads_k = key.size(1);
  uint32_t qo_len = query.size(-2);
  uint32_t kv_len = key.size(-2);
  uint32_t head_size = key.size(-1);

  auto output = at::empty_like(query);
  auto dpcpp_queue = dpcppGetCurrentQueue();

  // TODO(zw): support other datatype

  auto cgf = esimd::launch_fused_mha(
      reinterpret_cast<sycl::half*>(query.data_ptr()),
      reinterpret_cast<sycl::half*>(key.data_ptr()),
      reinterpret_cast<sycl::half*>(value.data_ptr()),
      reinterpret_cast<sycl::half*>(output.data_ptr()),
      reinterpret_cast<uint8_t*>(mask.data_ptr()),
      num_batches,
      num_heads_q,
      num_heads_k,
      head_size,
      qo_len,
      kv_len,
      is_head_first);
  DPCPP_Q_SUBMIT(dpcpp_queue, cgf);

  return output;
}

} // namespace AtenIpexTypeXPU
} // namespace at

namespace {
IPEX_LIBRARY_FRAGMENT() {
  IPEX_OP_REGISTER_DISPATCH(
      "fmha_esimd.xpu", at::AtenIpexTypeXPU::fmha_esimd, c10::DispatchKey::XPU);
}
} // namespace