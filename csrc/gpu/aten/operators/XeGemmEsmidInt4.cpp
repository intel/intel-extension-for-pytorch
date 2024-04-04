
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
      const scalar_t* input_,
      const uint8_t* weight_,
      scalar_t* output_,
      const scalar_t* weight_scl_,
      const uint8_t* weight_zp_,
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
  const scalar_t* input;
  const uint8_t* weight;
  scalar_t* output;
  const scalar_t* weight_scl;
  const uint8_t* weight_zp;
  int64_t calib_gz;
  uint32_t k;
};

// forward dpcpp implementation
template <typename scalar_t, typename uint8_t>
static inline void xegemm_int4_esimd_kernel(
    const scalar_t* input,
    const uint8_t* weight,
    scalar_t* output,
    const scalar_t* weight_scl,
    const uint8_t* weight_zp,
    int64_t calib_gz,
    uint32_t m,
    uint32_t n,
    uint32_t k) {
  std::cout << "m: " << m << std::endl;
  std::cout << "n: " << n << std::endl;
  std::cout << "k: " << k << std::endl;
  auto& dpcpp_queue = dpcppGetCurrentQueue();

  uint32_t pixelPerGroupCommonDim4096 = 16;
  if (n != 4096 && k != 1) {
    std::cout << "n should be 4096 and k should be 1" << std::endl;
    return;
  }

  // if (dispatchPattern[1] > 3) {
  //   std::cout << "matrixMulCommonDim11008Int4NoReshape kernel require
  //   dispatchPattern[1] to be 0 ~ 2" << std::endl; return false;
  // }

  int groupsV2 =
      (m + pixelPerGroupCommonDim4096 - 1) / pixelPerGroupCommonDim4096;
  int localThread[2];
  localThread[0] = 16;

  sycl::range<1> GlobalRangeCommonDim4096V2(groupsV2 * localThread[0]);
  sycl::range<1> LocalRangeCommonDim4096V2(localThread[0]);
  sycl::nd_range<1> RangeCommonDim4096V2(
      GlobalRangeCommonDim4096V2, LocalRangeCommonDim4096V2);

  auto cgf = DPCPP_Q_CGF(cgh) {
    XeGemmInt4EsimdKernelFunctor<scalar_t> kfn(
        input, weight, output, weight_scl, weight_zp, calib_gz, k);
    cgh.parallel_for<decltype(kfn)>(
        sycl::nd_range<1>(
            GlobalRangeCommonDim4096V2, LocalRangeCommonDim4096V2),
        kfn);
  };
  DPCPP_Q_SUBMIT(dpcpp_queue, cgf);

  // try {
  //   if (n == 4096) {
  //       switch (pixelPerGroupCommonDim4096) {
  //       case 16:
  //         e = q.submit([&](handler& cgh) {
  //           cgh.parallel_for(RangeCommonDim4096V2, [=](nd_item<1> ndi)
  //           SYCL_ESIMD_KERNEL{
  //             matrixMulCommonDim4096Int4NoReshapeNx16V3_ipex<4>((uint8_t*)a,
  //             (uint8_t*)b, (uint8_t*)c, (uint8_t*)d, ndi);
  //             });
  //           });
  //         break;
  //       default:
  //         break;
  //       }
  //   }
  // } catch (sycl::exception const& e) {
  //   std::cout << "SYCL exception caught: " << e.what() << '\n';
  //   return false;
  // }

  // bool success = true;
  // return success;
}

} // namespace impl

inline Tensor resize_as_mat2(const Tensor& mat1, const Tensor& output) {
  auto output_ = output.flatten(0, -2);
  int n = output_.sizes()[1];
  auto sizes = mat1.sym_sizes().vec();
  sizes[sizes.size() - 1] = n;
  return output.view_symint(sizes);
}

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

  uint32_t m = input_flat.sizes()[0]; // 1
  uint32_t k = input_flat.sizes()[1]; // 4096
  uint32_t n = weight.sizes()[1] * 2; // 11008
  auto output = at::empty({m, n}, input.options()); // 11008, 11008

  TORCH_CHECK(input_flat.dim() == 2 && weight_flat.dim() == 2);

  if (compute_eng_valid) {
    std::cout << "get in esimd int4 gemm" << std::endl;
    IPEX_DISPATCH_FLOATING_TYPES_AND_HALF(
        input_flat.scalar_type(), "XeGemm_int4_esimd", [&] {
          impl::xegemm_int4_esimd_kernel<scalar_t, uint8_t>(
              input_flat.data_ptr<scalar_t>(),
              weight_flat.data_ptr<uint8_t>(),
              output.data_ptr<scalar_t>(),
              weight_scl.data_ptr<scalar_t>(),
              weight_zp.data_ptr<uint8_t>(),
              calib_gz,
              m,
              n,
              k);
        });
  } else {
    AT_ERROR("GEMM INT4: invalid COMPUTE_ENG!");
  }
  return resize_as_mat2(input, output);
}

} // namespace AtenIpexTypeXPU
} // namespace at

namespace {
IPEX_LIBRARY_FRAGMENT() {
  IPEX_OP_REGISTER("mm_esimd_int4.xpu", at::AtenIpexTypeXPU::mm_esimd_int4);
}
} // namespace