
#include <ATen/ATen.h>
#include <ATen/CPUApplyUtils.h>
#include <ATen/record_function.h>
#include <core/detail/IndexUtils.h>
#include <core/detail/TensorInfo.h>
// #include <oneDNN/oneDNN.h>
#include <runtime/Utils.h>
#include <stdlib.h>
#include <utils/oneMKLUtils.h>
#include "comm/ATDispatch.h"
// #include "comm/Numerics.h"
#include "comm/RegistrationDeclarations.h"
// #include "utils/ComputeEngine.h"
#include "utils/CustomOperatorRegistration.h"

#include <sycl/ext/intel/esimd.hpp>
#include <sycl/sycl.hpp>
using namespace sycl::ext::intel::esimd;
using namespace sycl::ext::intel::esimd::xmx;
using fp16 = sycl::half;
using namespace sycl;

#include "esimd/qkvFusion128To2048KvlengthLoopFp32QFp16KvXveSimd16Slm.h"

inline double report_time(const std::string& msg, event e0, event en) {
  uint64_t time_start =
      e0.get_profiling_info<info::event_profiling::command_start>();
  uint64_t time_end =
      en.get_profiling_info<info::event_profiling::command_end>();
  double elapsed = (time_end - time_start) / 1e6;
  // cerr << msg << elapsed << " msecs" << std::endl;
  std::cout << msg << elapsed << " msecs" << std::endl;
  return elapsed;
}

using namespace xpu::dpcpp::detail;
using namespace xpu::dpcpp;
using namespace at::native;

namespace at {
namespace AtenIpexTypeXPU {

namespace impl {

static void dump_element(const Tensor src, int nele, std::string str) {
  std::cout << str;
  for (int i = 0; i < nele; i++) {
    std::cout << " " << src[0][i];
  }
  std::cout << std::endl;
}

// forward dpcpp implementation
template <typename scalar_t>
static inline void sdp_esimd_kernel(
    int heads_kv,
    scalar_t* query,
    scalar_t* key,
    scalar_t* value,
    void* alibi,
    void* attn_mask,
    void* head_mask,
    scalar_t* output,
    float alpha,
    float beta,
    float dropout_prob,
    bool is_causal,
    bool seq_last,
    int64_t kv_len) {
  // YC std::cout << "alpha: " << alpha << std::endl;
  // YC std::cout << "beta: " << beta << std::endl;
  // YC std::cout << "dropout_prob: " << dropout_prob << std::endl;
  auto& dpcpp_queue = dpcppGetCurrentQueue();

  const int batch_size = 1;
  sycl::range<2> GlobalRange(
      32 * 32, batch_size); // num_head x kv_len, batch size
  sycl::range<2> LocalRange(32, 1); // kv_len, x
  sycl::nd_range<2> Range(GlobalRange, LocalRange);
  sycl::event e;
  int vCacheStride = 0; // not used.
  if (heads_kv == 32) {
    e = dpcpp_queue.submit([&](handler& cgh) {
      cgh.parallel_for(Range, [=](nd_item<2> ndi) SYCL_ESIMD_KERNEL {
        qkvFusion128To2048KvlengthLoopFp32QFp16KvXveSimd16Slm_ipex<32>(
            (uint8_t*)query,
            (uint8_t*)key,
            (uint8_t*)value,
            (uint8_t*)output,
            kv_len,
            vCacheStride,
            ndi);
      });
    });
  } else if (heads_kv == 8) {
    e = dpcpp_queue.submit([&](handler& cgh) {
      cgh.parallel_for(Range, [=](nd_item<2> ndi) SYCL_ESIMD_KERNEL {
        qkvFusion128To2048KvlengthLoopFp32QFp16KvXveSimd16Slm_ipex<8>(
            (uint8_t*)query,
            (uint8_t*)key,
            (uint8_t*)value,
            (uint8_t*)output,
            kv_len,
            vCacheStride,
            ndi);
      });
    });
  } else {
    AT_ERROR("ESIMD SDP: invalid HEADS_KV!");
  }

  // YC e.wait();
  // double etime = report_time("SDP fused kernel time", e, e);

  return;
}

} // namespace impl

static Tensor sdp_esimd(
    const Tensor& query,
    const Tensor& key,
    const Tensor& value,
    const c10::optional<Tensor>& alibi,
    const c10::optional<Tensor>& attn_mask,
    const c10::optional<Tensor>& head_mask,
    const double alpha,
    const double beta,
    const double dropout_p,
    bool is_causal,
    bool seq_last) {
  // YC std::cout << "start sdp_esimd" << std::endl;
  TORCH_CHECK(
      query.scalar_type() == at::kHalf, "IPEX SDP only supports half datatype");
  TORCH_CHECK(
      key.scalar_type() == at::kHalf, "IPEX SDP only supports half datatype");
  TORCH_CHECK(
      value.scalar_type() == at::kHalf, "IPEX SDP only supports half datatype");
  // xpu::COMPUTE_ENG real_eng =
  //     choose_compute_eng(xpu::COMPUTE_ENG::ESIMD, query, key);
  // bool compute_eng_valid = (real_eng == xpu::COMPUTE_ENG::ESIMD);
  bool compute_eng_valid = true;

  int64_t num_heads_q = query.size(1);
  int64_t num_heads_k = key.size(1);
  int64_t M = query.size(-2);
  int64_t N = key.size(-2);
  auto kv_len = N;
  TORCH_CHECK(num_heads_q == 32, "ESIMD SDP only supports num_heads == 32");
  TORCH_CHECK(query.size(-1) == 128, "ESIMD SDP only supports head_dim == 32");

  auto output = at::empty_like(query);
  auto dpcpp_queue = dpcppGetCurrentQueue();
  char str__[100];
  sprintf(
      str__,
      "sdp_esimd(Nq=%ld, Nkv=%ld, M=%ld, N=%ld)",
      num_heads_q,
      num_heads_k,
      M,
      N);
  RECORD_FUNCTION(str__, {});

  if (compute_eng_valid) {
    IPEX_DISPATCH_FLOATING_TYPES_AND_HALF(
        query.scalar_type(), "sdp_esimd", [&] {
          impl::sdp_esimd_kernel<scalar_t>(
              num_heads_k,
              query.data_ptr<scalar_t>(),
              key.data_ptr<scalar_t>(),
              value.data_ptr<scalar_t>(),
              alibi.has_value() ? alibi.value().data_ptr() : (void*)nullptr,
              attn_mask.has_value() ? attn_mask.value().data_ptr()
                                    : (void*)nullptr,
              head_mask.has_value() ? head_mask.value().data_ptr()
                                    : (void*)nullptr,
              output.data_ptr<scalar_t>(),
              alpha,
              beta,
              dropout_p,
              is_causal,
              seq_last,
              kv_len);
        });
  } else {
    AT_ERROR("ESIMD SDP: invalid COMPUTE_ENG!");
  }

  return output;
}

} // namespace AtenIpexTypeXPU
} // namespace at

namespace {
IPEX_LIBRARY_FRAGMENT() {
  IPEX_OP_REGISTER_DISPATCH(
      "sdp_esimd.xpu", at::AtenIpexTypeXPU::sdp_esimd, c10::DispatchKey::XPU);
}
} // namespace
