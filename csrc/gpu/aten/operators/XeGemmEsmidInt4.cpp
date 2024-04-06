
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
// #include "comm/Numerics.h"
#include "comm/RegistrationDeclarations.h"
#include "utils/ComputeEngine.h"
#include "utils/CustomOperatorRegistration.h"

#include <sycl/ext/intel/esimd.hpp>
#include <sycl/sycl.hpp>
using namespace sycl::ext::intel::esimd;
using namespace sycl::ext::intel::esimd::xmx;
using fp16 = sycl::half;
using namespace sycl;

#include "esimd/matrixMulCommonDim4096Int4NoReshapeNx16V3.h"

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
      uint8_t* weight_,
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
  uint8_t* weight;
  scalar_t* output;
  const scalar_t* weight_scl;
  const uint8_t* weight_zp;
  int64_t calib_gz;
  uint32_t k;
};

static void dump_element(const Tensor src, int nele, std::string str) {
  std::cout << str;
  for (int i = 0; i < nele; i++) {
    std::cout << " " << src[0][i];
  }
  std::cout << std::endl;
}

// forward dpcpp implementation
template <typename scalar_t, typename uint8_t>
static inline void xegemm_int4_esimd_kernel(
    const scalar_t* input,
    uint8_t* weight,
    scalar_t* output,
    const scalar_t* weight_scl,
    const uint8_t* weight_zp,
    uint8_t* reorder_buffer,
    int64_t calib_gz,
    uint32_t m,
    uint32_t n,
    uint32_t k) {
  std::cout << "m: " << m << std::endl;
  std::cout << "n: " << n << std::endl;
  std::cout << "k: " << k << std::endl;
  auto& dpcpp_queue = dpcppGetCurrentQueue();

  uint32_t pixelPerGroupCommonDim4096 = 16;
  if (k != 4096 && m != 1) {
    std::cout << "n should be 4096 and k should be 1" << std::endl;
    return;
  }

  // reorder for the weights and scaling
  // Assume group size 32.   4096 / 32 = 128
  {
    uint8_t* weight_reorder = (uint8_t*)weight;
    uint8_t* reorderTmp = reorder_buffer;
    dpcpp_queue.submit([&](handler& cgh) {
      cgh.parallel_for(sycl::range<2>(4096, n), [=](sycl::id<2> idx) {
        int i = idx[0];
        int j = idx[1];

        int32_t origIdx = i * n + j;
        int32_t afterIdx = j * 4096 + i;

        int8_t tmp = weight_reorder[origIdx / 2];
        if (origIdx % 2 == 0) {
          tmp = tmp & 0xf;
        } else {
          tmp = tmp >> 4;
        }
        reorderTmp[afterIdx] = tmp;
      });
    });
    dpcpp_queue.submit([&](handler& cgh) {
      cgh.parallel_for(sycl::range<2>(2048, n), [=](sycl::id<2> idx) {
        int i = idx[0];
        int j = idx[1];

        int32_t afterIdxInt4 = j * 2048 + i;

        int8_t tmpLow = reorderTmp[afterIdxInt4 * 2];
        int8_t tmpHigh = reorderTmp[afterIdxInt4 * 2 + 1];

        int8_t tmp = (tmpLow & 0xf) | (tmpHigh << 4);

        weight_reorder[afterIdxInt4] = tmp;
      });
    });
    dpcpp_queue.wait();

    fp16* reorderTmpScal = (fp16*)reorder_buffer;
    fp16* weight_scl_reorder = (fp16*)weight_scl;
    dpcpp_queue.submit([&](handler& cgh) {
      cgh.parallel_for(sycl::range<2>(128, n), [=](sycl::id<2> idx) {
        int i = idx[0];
        int j = idx[1];

        int32_t origIdx = i * n + j;
        int32_t afterIdx = j * 128 + i;

        reorderTmpScal[afterIdx] = weight_scl_reorder[origIdx];
      });
    });
    dpcpp_queue.submit([&](handler& cgh) {
      cgh.parallel_for(sycl::range<2>(128, n), [=](sycl::id<2> idx) {
        int i = idx[0];
        int j = idx[1];

        int32_t afterIdx = j * 128 + i;

        weight_scl_reorder[afterIdx] = reorderTmpScal[afterIdx];
      });
    });
    dpcpp_queue.wait();
  }

  int groupsV2 =
      (n + pixelPerGroupCommonDim4096 - 1) / pixelPerGroupCommonDim4096;
  int localThread[2];
  localThread[0] = 16;

  sycl::range<1> GlobalRangeCommonDim4096V2(groupsV2 * localThread[0]);
  sycl::range<1> LocalRangeCommonDim4096V2(localThread[0]);
  sycl::nd_range<1> RangeCommonDim4096V2(
      GlobalRangeCommonDim4096V2, LocalRangeCommonDim4096V2);

  sycl::event e;
  // Launches the task on the GPU.
  if (k == 4096) {
    e = dpcpp_queue.submit([&](handler& cgh) {
      cgh.parallel_for(
          RangeCommonDim4096V2, [=](nd_item<1> ndi) SYCL_ESIMD_KERNEL {
            matrixMulCommonDim4096Int4NoReshapeNx16V3_ipex2<4>(
                (uint8_t*)weight,
                (uint8_t*)input,
                (uint8_t*)output,
                (uint8_t*)weight_scl,
                ndi);
          });
    });
  }
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
    Tensor& weight,
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
  auto reorder_buffer = at::empty({k, n}, weight.options());
  auto output = at::empty({m, n}, input.options()); // 11008, 11008

  TORCH_CHECK(input_flat.dim() == 2 && weight_flat.dim() == 2);
  impl::dump_element(weight_flat, 10, "weight first 10 elem: ");
  //impl::dump_element(reorder_buffer, 10, "reorder_buffer first 10 elem: ");
  impl::dump_element(weight_scl, 10, "scal first 10 elem: ");

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
              reorder_buffer.data_ptr<uint8_t>(),
              calib_gz,
              m,
              n,
              k);
        });
  } else {
    AT_ERROR("GEMM INT4: invalid COMPUTE_ENG!");
  }
  impl::dump_element(weight_flat, 10, "weight before output first 10 elem: ");
  //impl::dump_element(
  //    reorder_buffer, 10, "reorder_buffer  before output first 10 elem: ");
  impl::dump_element(output, 10, "output first 10 elem: ");
  return resize_as_mat2(input, output);
}

} // namespace AtenIpexTypeXPU
} // namespace at

namespace {
IPEX_LIBRARY_FRAGMENT() {
  IPEX_OP_REGISTER("mm_esimd_int4.xpu", at::AtenIpexTypeXPU::mm_esimd_int4);
}
} // namespace
