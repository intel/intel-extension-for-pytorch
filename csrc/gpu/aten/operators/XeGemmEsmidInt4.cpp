
#include <ATen/ATen.h>
#include <ATen/CPUApplyUtils.h>
#include <ATen/record_function.h>
#include <core/detail/IndexUtils.h>
#include <core/detail/TensorInfo.h>
#include <oneDNN/oneDNN.h>
#include <runtime/Utils.h>
#include <stdlib.h>
#include <utils/oneMKLUtils.h>
#include "comm/ATDispatch.h"
#include "comm/Numerics.h"
#include "comm/RegistrationDeclarations.h"
#include "utils/ComputeEngine.h"
#include "utils/CustomOperatorRegistration.h"

using namespace xpu::dpcpp::detail;
using namespace xpu::dpcpp;
using namespace at::native;

static constexpr int64_t GRU_WORKSPACE_MULTIPLIER = 5;

namespace at {
namespace AtenIpexTypeXPU {

namespace impl {

template <typename scalar_t>
struct XeGemmInt4EsimdKernelFunctor {
  void operator()(sycl::nd_item<1> itemId) const {
    // todo : kernel calculation
  }
  XeGemmInt4EsimdKernelFunctor(
      scalar_t* input_,
      uint8_t* weight_,
      scalar_t* output_,
      scalar_t* weight_scl_,
      uint8_t* weight_zp_,
      int64_t calib_gz_,
      uint32_t k_)
      : input(input_),
        weight(weight_),
        output(output_),
        weight_scl(weight_scl_),
        weight_zp(weight_zp_),
        calib_gz(calib_gz_),
        k(k_) {}

 private:
  scalar_t* input;
  uint8_t* weight;
  scalar_t* output;
  scalar_t* weight_scl;
  uint8_t* weight_zp;
  int64_t calib_gz;
  uint32_t k;
};

// forward dpcpp implementation
template <typename scalar_t, typename uint8_t>
static inline void xegemm_int4_esimd_kernel(
    scalar_t* input,
    uint8_t* weight,
    scalar_t* output,
    scalar_t* weight_scl,
    uint8_t* weight_zp,
    int64_t calib_gz,
    uint32_t k) {
  auto& dpcpp_queue = dpcppGetCurrentQueue();

  auto dev_id = dpcppGetDeviceIdOfCurrentQueue();
  int64_t max_group_size = dpcppMaxWorkGroupSize(dev_id);

  // TODO: add global_size local_size
  const int64_t COL = calib_gz;
  const int64_t ROW = calib_gz;
  int64_t GROUP_SIZE = std::min(calib_gz, max_group_size);

  const sycl::range<1> global_size{ROW * GROUP_SIZE};
  const sycl::range<1> local_size{GROUP_SIZE};

  auto cgf = DPCPP_Q_CGF(cgh) {
    XeGemmInt4EsimdKernelFunctor<scalar_t> kfn(
        input, weight, output, weight_scl, weight_zp, calib_gz, k);
    cgh.parallel_for<decltype(kfn)>(
        sycl::nd_range<1>(global_size, local_size), kfn);
  };
  DPCPP_Q_SUBMIT(dpcpp_queue, cgf);
}

} // namespace impl

static Tensor mm_esimd_int4(
    const Tensor& input,
    const Tensor& weight,
    const Tensor& weight_scl,
    const Tensor& weight_zp,
    int64_t calib_gz) {
  std::cout << "start mm_esimd_int4" << std::endl;
  TORCH_CHECK(input.scalar_type() == ScalarType::Half);
  auto input_flat = input.flatten(0, -2); // 1, 1, 4096 -> 1, 4096
  auto weight_flat = weight.flatten(0, -2); // 4096, 5504, 1 -> 4096, 5504

  xpu::COMPUTE_ENG real_eng =
      choose_compute_eng(xpu::COMPUTE_ENG::ESIMD, input, weight);
  bool compute_eng_valid = (real_eng == xpu::COMPUTE_ENG::ESIMD);

  int m = input_flat.sizes()[0]; // 1
  int k = input_flat.sizes()[1]; // 4096
  int n = weight.sizes()[1] * 2; // 11008
  auto output = at::empty({m, n}, input.options()); // 11008, 11008

  TORCH_CHECK(input_flat.dim() == 2 && weight_flat.dim() == 2);

  if (compute_eng_valid) {
    std::cout << "get in esimd int4 gemm" << std::endl;
    uint32_t k = weight_flat.size(1); // k 4096
    IPEX_DISPATCH_FLOATING_TYPES_AND_HALF(
        input_flat.scalar_type(), "XeGemm_int4_esimd", [&] {
          impl::xegemm_int4_esimd_kernel<scalar_t, uint8_t>(
              input_flat.data_ptr<scalar_t>(),
              weight_flat.data_ptr<uint8_t>(),
              output.data_ptr<scalar_t>(),
              weight_scl.data_ptr<scalar_t>(),
              weight_zp.data_ptr<uint8_t>(),
              calib_gz,
              k);
        });
  } else {
    AT_ERROR("GEMM INT4: invalid COMPUTE_ENG!");
  }
}

} // namespace AtenIpexTypeXPU
} // namespace at

namespace {
IPEX_LIBRARY_FRAGMENT() {
  IPEX_OP_REGISTER("mm_esimd_int4.xpu", at::AtenIpexTypeXPU::mm_esimd_int4);
}
} // namespace