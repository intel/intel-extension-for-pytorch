
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

#include "esimd/gemmReduce2048WeightsQ40InputFp32.h"
#include "esimd/matrixMulCommonDim4096Int4NoReshapeNx16V3.h"

using namespace xpu::dpcpp::detail;
using namespace xpu::dpcpp;
using namespace at::native;

static constexpr int64_t GRU_WORKSPACE_MULTIPLIER = 5;

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
template <typename scalar_t, typename uint8_t>
static inline void gemm_int4_esimd_kernel(
    const scalar_t* input,
    uint8_t* weight,
    scalar_t* output,
    const scalar_t* weight_scl,
    const uint8_t* weight_zp,
    uint8_t* reorder_buffer,
    int64_t calib_gz,
    uint32_t m,
    uint32_t n,
    uint32_t k,
    bool need_reorder) {
  // YC std::cout << "m: " << m << std::endl;
  // YC std::cout << "n: " << n << std::endl;
  // YC std::cout << "k: " << k << std::endl;
  auto& dpcpp_queue = dpcppGetCurrentQueue();

  uint32_t pixelPerGroupCommonDim4096 = 16;
  uint32_t pixelPerGroupCommonDim11008 = 64;
  if (!(k == 4096 || k == 11008 || k == 14336)) {
    std::cout << "k should be 4096 or 11008 or 14336" << std::endl;
    return;
  }

  // reorder for the weights and scaling
  // Assume group size 32.   4096 / 32 = 128
  if (need_reorder) {
    sycl::event e0;
    uint8_t* weight_reorder = (uint8_t*)weight;
    uint8_t* reorderTmp = reorder_buffer;
    e0 = dpcpp_queue.submit([&](handler& cgh) {
      cgh.parallel_for(sycl::range<2>(k, n), [=](sycl::id<2> idx) {
        int i = idx[0];
        int j = idx[1];

        int32_t origIdx = i * n + j;
        int32_t afterIdx = j * k + i;

        int8_t tmp = weight_reorder[origIdx / 2];
        if (origIdx % 2 == 0) {
          tmp = tmp & 0xf;
        } else {
          tmp = tmp >> 4;
        }
        reorderTmp[afterIdx] = tmp;
      });
    });
    e0.wait();
    report_time("reshape1 kernel time", e0, e0);
    e0 = dpcpp_queue.submit([&](handler& cgh) {
      cgh.parallel_for(sycl::range<2>(k / 2, n), [=](sycl::id<2> idx) {
        int i = idx[0];
        int j = idx[1];

        int32_t afterIdxInt4 = j * k / 2 + i;

        int8_t tmpLow = reorderTmp[afterIdxInt4 * 2];
        int8_t tmpHigh = reorderTmp[afterIdxInt4 * 2 + 1];

        int8_t tmp = (tmpLow & 0xf) | (tmpHigh << 4);

        weight_reorder[afterIdxInt4] = tmp;
      });
    });
    e0.wait();
    report_time("reshape2 kernel time", e0, e0);

    fp16* reorderTmpScal = (fp16*)reorder_buffer;
    fp16* weight_scl_reorder = (fp16*)weight_scl;
    e0 = dpcpp_queue.submit([&](handler& cgh) {
      cgh.parallel_for(sycl::range<2>(k / 32, n), [=](sycl::id<2> idx) {
        int i = idx[0];
        int j = idx[1];

        int32_t origIdx = i * n + j;
        int32_t afterIdx = j * k / 32 + i;

        reorderTmpScal[afterIdx] = weight_scl_reorder[origIdx];
      });
    });
    e0.wait();
    report_time("reshape3 kernel time", e0, e0);
    e0 = dpcpp_queue.submit([&](handler& cgh) {
      cgh.parallel_for(sycl::range<2>(k / 32, n), [=](sycl::id<2> idx) {
        int i = idx[0];
        int j = idx[1];

        int32_t afterIdx = j * k / 32 + i;

        weight_scl_reorder[afterIdx] = reorderTmpScal[afterIdx];
      });
    });
    e0.wait();
    report_time("reshape4 kernel time", e0, e0);
    dpcpp_queue.wait();

    return;
  }

  if (m == 1) // GEMV
  {
    int groupsV2 =
        (n + pixelPerGroupCommonDim4096 - 1) / pixelPerGroupCommonDim4096;
    int localThread[2];
    localThread[0] = 16;
    localThread[1] = 16;

    sycl::range<1> GlobalRangeCommonDim4096V2(groupsV2 * localThread[0]);
    sycl::range<1> LocalRangeCommonDim4096V2(localThread[0]);
    sycl::nd_range<1> RangeCommonDim4096V2(
        GlobalRangeCommonDim4096V2, LocalRangeCommonDim4096V2);

    // Opt 14336
    if (k == 14336)
    {
      pixelPerGroupCommonDim11008 = 16;
    }

    groupsV2 =
        (n + pixelPerGroupCommonDim11008 - 1) / pixelPerGroupCommonDim11008;
    sycl::range<1> GlobalRangeCommonDim11008V2(groupsV2 * localThread[1]);
    sycl::range<1> LocalRangeCommonDim11008V2(localThread[1]);
    sycl::nd_range<1> RangeCommonDim11008V2(
        GlobalRangeCommonDim11008V2, LocalRangeCommonDim11008V2);

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
      // YC e.wait();
      // double etime = report_time("GEMV kernel time", e, e);
    } else if (k == 11008) {
      e = dpcpp_queue.submit([&](handler& cgh) {
        cgh.parallel_for(
            RangeCommonDim11008V2, [=](nd_item<1> ndi) SYCL_ESIMD_KERNEL {
              matrixMulCommonDim11008Int4NoReshapeNx16V2_ipex<6>(
                  (uint8_t*)weight,
                  (uint8_t*)input,
                  (uint8_t*)output,
                  (uint8_t*)weight_scl,
                  ndi);
            });
      });
      // YC e.wait();
      // double etime = report_time("GEMV kernel time", e, e);
    } else if (k == 14336) {
      e = dpcpp_queue.submit([&](handler& cgh) {
        cgh.parallel_for(
            RangeCommonDim11008V2, [=](nd_item<1> ndi) SYCL_ESIMD_KERNEL {
              matrixMulCommonDim14336Int4NoReshapeNx16V2_ipex<4>( // PPG comes to 16
                  (uint8_t*)weight,
                  (uint8_t*)input,
                  (uint8_t*)output,
                  (uint8_t*)weight_scl,
                  ndi);
            });
      });
      // YC e.wait();
      // double etime = report_time("GEMV kernel time", e, e);
    }
  } else // GEMM
  {
    int groupReduce2048H = (n + 15) / 16;
    int groupReduce2048V = 1;
    int localReduce2048H = 64; // internalPrecision == 0  (fp32), not 32
    int localReduce2048V = 1;
    sycl::range<2> GlobalRangeReduce2048(
        groupReduce2048H * localReduce2048H,
        groupReduce2048V * localReduce2048V);
    sycl::range<2> LocalRangeReduce2048(localReduce2048H, localReduce2048V);
    sycl::nd_range<2> RangeReduce2048(
        GlobalRangeReduce2048, LocalRangeReduce2048);

    int groupReduce768H = (n + 15) / 16;
    int groupReduce768V = 1;
    int localReduce768H = 12;
    // if (internalPrecision == 0) {
    localReduce768H = 24;
    // }
    int localReduce768V = 1;
    sycl::range<2> GlobalRangeReduce768(
        groupReduce768H * localReduce768H, groupReduce768V * localReduce768V);
    sycl::range<2> LocalRangeReduce768(localReduce768H, localReduce768V);
    sycl::nd_range<2> RangeReduce768(GlobalRangeReduce768, LocalRangeReduce768);

    sycl::event e;
    int lastReduce = 0;
    if (k == 4096) {
      for (int ii = 0; ii < 2; ii++) {
        if (ii == 2 - 1) {
          lastReduce = 1;
        } else {
          lastReduce = 0;
        }
        e = dpcpp_queue.submit([&](handler& cgh) {
          cgh.parallel_for(
              RangeReduce2048, [=](nd_item<2> ndi) SYCL_ESIMD_KERNEL {
                gemmReduce2048WeightsQ40InputFp32_ipex(
                    (uint8_t*)weight,
                    (uint8_t*)input,
                    (uint8_t*)output,
                    (uint8_t*)weight_scl,
                    k,
                    m /*tokenLen*/,
                    ii,
                    lastReduce,
                    ndi);
              });
        });
        // YC e.wait();
        // double etime = report_time("GEMV kernel time", e, e);
      }
    } else if (k == 14336) {
      for (int ii = 0; ii < 7; ii++) {
        if (ii == 7 - 1) {
          lastReduce = 1;
        } else {
          lastReduce = 0;
        }
        e = dpcpp_queue.submit([&](handler& cgh) {
          cgh.parallel_for(
              RangeReduce2048, [=](nd_item<2> ndi) SYCL_ESIMD_KERNEL {
                gemmReduce2048WeightsQ40InputFp32_ipex(
                    (uint8_t*)weight,
                    (uint8_t*)input,
                    (uint8_t*)output,
                    (uint8_t*)weight_scl,
                    k,
                    m /*tokenLen*/,
                    ii,
                    lastReduce,
                    ndi);
              });
        });
        // YC e.wait();
        // double etime = report_time("GEMV kernel time", e, e);
      }
    } else if (k == 11008) {
      for (int ii = 0; ii < 5; ii++) {
        lastReduce = 0;
        e = dpcpp_queue.submit([&](handler& cgh) {
          cgh.parallel_for(
              RangeReduce2048, [=](nd_item<2> ndi) SYCL_ESIMD_KERNEL {
                gemmReduce2048WeightsQ40InputFp32_ipex(
                    (uint8_t*)weight,
                    (uint8_t*)input,
                    (uint8_t*)output,
                    (uint8_t*)weight_scl,
                    k,
                    m /*tokenLen*/,
                    ii,
                    lastReduce,
                    ndi);
              });
        });
        // YC e.wait();
        // double etime = report_time("GEMV kernel time", e, e);
      }

      int ii = 5;
      {
        lastReduce = 1;
        e = dpcpp_queue.submit([&](handler& cgh) {
          cgh.parallel_for(
              RangeReduce768, [=](nd_item<2> ndi) SYCL_ESIMD_KERNEL {
                gemmReduce768WeightsQ40InputFp32_ipex(
                    (uint8_t*)weight,
                    (uint8_t*)input,
                    (uint8_t*)output,
                    (uint8_t*)weight_scl,
                    k,
                    m /*tokenLen*/,
                    ii,
                    lastReduce,
                    ndi);
              });
        });
        // YC e.wait();
        // double etime = report_time("GEMV kernel time", e, e);
      }
    }
  }
}

// forward dpcpp implementation
template <typename scalar_t, typename uint8_t>
static inline void qkv_gemm_int4_esimd_kernel(
    const scalar_t* input,
    uint8_t* weight,
    const scalar_t* weight_scl,
    const uint8_t* weight_zp,
    void* bias,
    uint8_t* reorder_buffer,
    int64_t calib_gz,
    uint32_t m,
    uint32_t n,
    uint32_t k,
    scalar_t* out0, // Q
    scalar_t* out1, // K
    scalar_t* out2, // V
    bool need_reorder) {
  // YC std::cout << "qkv_gemm_int4_esimd_kernel m: " << m << std::endl;
  // YC std::cout << "qkv_gemm_int4_esimd_kernel n: " << n << std::endl;
  // YC std::cout << "qkv_gemm_int4_esimd_kernel k: " << k << std::endl;
  auto& dpcpp_queue = dpcppGetCurrentQueue();

  if (!(k == 4096 && n == 4096)) {
    std::cout << "k should be 4096 and n should be 4096" << std::endl;
    // m is input vector num
    return;
  }

  uint32_t pixelPerGroupCommonDim4096 = 16;

  // reorder for the weights and scaling
  // Assume group size 32.   4096 / 32 = 128
  if (need_reorder) {
    for (int index = 0; index < 3; index++) {
      sycl::event e0;
      uint8_t* weight_reorder = (uint8_t*)weight + 2048 * 4096 * index;
      uint8_t* reorderTmp = reorder_buffer;
      e0 = dpcpp_queue.submit([&](handler& cgh) {
        cgh.parallel_for(sycl::range<2>(k, n), [=](sycl::id<2> idx) {
          int i = idx[0];
          int j = idx[1];

          int32_t origIdx = i * n + j;
          int32_t afterIdx = j * k + i;

          int8_t tmp = weight_reorder[origIdx / 2];
          if (origIdx % 2 == 0) {
            tmp = tmp & 0xf;
          } else {
            tmp = tmp >> 4;
          }
          reorderTmp[afterIdx] = tmp;
        });
      });
      e0.wait();
      report_time("reshape1 kernel time", e0, e0);
      e0 = dpcpp_queue.submit([&](handler& cgh) {
        cgh.parallel_for(sycl::range<2>(k / 2, n), [=](sycl::id<2> idx) {
          int i = idx[0];
          int j = idx[1];

          int32_t afterIdxInt4 = j * k / 2 + i;

          int8_t tmpLow = reorderTmp[afterIdxInt4 * 2];
          int8_t tmpHigh = reorderTmp[afterIdxInt4 * 2 + 1];

          int8_t tmp = (tmpLow & 0xf) | (tmpHigh << 4);

          weight_reorder[afterIdxInt4] = tmp;
        });
      });
      e0.wait();
      report_time("reshape2 kernel time", e0, e0);

      fp16* reorderTmpScal = (fp16*)reorder_buffer;
      fp16* weight_scl_reorder = (fp16*)weight_scl;
      weight_scl_reorder = weight_scl_reorder + 4096 * 128 * index;
      e0 = dpcpp_queue.submit([&](handler& cgh) {
        cgh.parallel_for(sycl::range<2>(k / 32, n), [=](sycl::id<2> idx) {
          int i = idx[0];
          int j = idx[1];

          int32_t origIdx = i * n + j;
          int32_t afterIdx = j * k / 32 + i;

          reorderTmpScal[afterIdx] = weight_scl_reorder[origIdx];
        });
      });
      e0.wait();
      report_time("reshape3 kernel time", e0, e0);
      e0 = dpcpp_queue.submit([&](handler& cgh) {
        cgh.parallel_for(sycl::range<2>(k / 32, n), [=](sycl::id<2> idx) {
          int i = idx[0];
          int j = idx[1];

          int32_t afterIdx = j * k / 32 + i;

          weight_scl_reorder[afterIdx] = reorderTmpScal[afterIdx];
        });
      });
      e0.wait();
      report_time("reshape4 kernel time", e0, e0);
      dpcpp_queue.wait();
    }
    return;
  }

  if (m == 1) // GEMV
  {
    int groupsV2 =
        (n + pixelPerGroupCommonDim4096 - 1) / pixelPerGroupCommonDim4096;
    int localThread[2];
    localThread[0] = 16;
    localThread[1] = 16;

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
                  (uint8_t*)out0,
                  (uint8_t*)weight_scl,
                  ndi);
            });
      });
      // YC e.wait();
      // double etime = report_time("GEMV kernel time", e, e);
    }
    if (k == 4096) {
      e = dpcpp_queue.submit([&](handler& cgh) {
        cgh.parallel_for(
            RangeCommonDim4096V2, [=](nd_item<1> ndi) SYCL_ESIMD_KERNEL {
              matrixMulCommonDim4096Int4NoReshapeNx16V3_ipex2<4>(
                  (uint8_t*)weight + 2048 * 4096 * 1,
                  (uint8_t*)input,
                  (uint8_t*)out1,
                  (uint8_t*)(weight_scl + 128 * 4096 * 1),
                  ndi);
            });
      });
      // YC e.wait();
      // double etime = report_time("GEMV kernel time", e, e);
    }
    if (k == 4096) {
      e = dpcpp_queue.submit([&](handler& cgh) {
        cgh.parallel_for(
            RangeCommonDim4096V2, [=](nd_item<1> ndi) SYCL_ESIMD_KERNEL {
              matrixMulCommonDim4096Int4NoReshapeNx16V3_ipex2<4>(
                  (uint8_t*)weight + 2048 * 4096 * 2,
                  (uint8_t*)input,
                  (uint8_t*)out2,
                  (uint8_t*)(weight_scl + 128 * 4096 * 2),
                  ndi);
            });
      });
      // YC e.wait();
      // double etime = report_time("GEMV kernel time", e, e);
    }
  } else // GEMM
  {
    int groupReduce2048H = (n + 15) / 16;
    int groupReduce2048V = 1;
    int localReduce2048H = 64; // internalPrecision == 0  (fp32), not 32
    int localReduce2048V = 1;
    sycl::range<2> GlobalRangeReduce2048(
        groupReduce2048H * localReduce2048H,
        groupReduce2048V * localReduce2048V);
    sycl::range<2> LocalRangeReduce2048(localReduce2048H, localReduce2048V);
    sycl::nd_range<2> RangeReduce2048(
        GlobalRangeReduce2048, LocalRangeReduce2048);

    sycl::event e;
    int lastReduce = 0;
    if (k == 4096) {
      for (int ii = 0; ii < 2; ii++) {
        if (ii == 2 - 1) {
          lastReduce = 1;
        } else {
          lastReduce = 0;
        }
        e = dpcpp_queue.submit([&](handler& cgh) {
          cgh.parallel_for(
              RangeReduce2048, [=](nd_item<2> ndi) SYCL_ESIMD_KERNEL {
                gemmReduce2048WeightsQ40InputFp32_ipex(
                    (uint8_t*)weight,
                    (uint8_t*)input,
                    (uint8_t*)out0,
                    (uint8_t*)weight_scl,
                    k,
                    m /*tokenLen*/,
                    ii,
                    lastReduce,
                    ndi);
              });
        });
        // YC e.wait();
        // double etime = report_time("GEMV kernel time", e, e);
      }
    }
    if (k == 4096) {
      for (int ii = 0; ii < 2; ii++) {
        if (ii == 2 - 1) {
          lastReduce = 1;
        } else {
          lastReduce = 0;
        }
        e = dpcpp_queue.submit([&](handler& cgh) {
          cgh.parallel_for(
              RangeReduce2048, [=](nd_item<2> ndi) SYCL_ESIMD_KERNEL {
                gemmReduce2048WeightsQ40InputFp32_ipex(
                    (uint8_t*)(weight + 2048 * 4096 * 1),
                    (uint8_t*)input,
                    (uint8_t*)out1,
                    (uint8_t*)(weight_scl + 128 * 4096 * 1),
                    k,
                    m /*tokenLen*/,
                    ii,
                    lastReduce,
                    ndi);
              });
        });
        // YC e.wait();
        // double etime = report_time("GEMV kernel time", e, e);
      }
    }
    if (k == 4096) {
      for (int ii = 0; ii < 2; ii++) {
        if (ii == 2 - 1) {
          lastReduce = 1;
        } else {
          lastReduce = 0;
        }
        e = dpcpp_queue.submit([&](handler& cgh) {
          cgh.parallel_for(
              RangeReduce2048, [=](nd_item<2> ndi) SYCL_ESIMD_KERNEL {
                gemmReduce2048WeightsQ40InputFp32_ipex(
                    (uint8_t*)(weight + 2048 * 4096 * 2),
                    (uint8_t*)input,
                    (uint8_t*)out2,
                    (uint8_t*)(weight_scl + 128 * 4096 * 2),
                    k,
                    m /*tokenLen*/,
                    ii,
                    lastReduce,
                    ndi);
              });
        });
        // YC e.wait();
        // double etime = report_time("GEMV kernel time", e, e);
      }
    }
  }

  return;
}

// forward dpcpp implementation
template <typename scalar_t, typename uint8_t>
static inline void qkv_gemm_int4_esimd_kernel_fused(
    const scalar_t* input,
    uint8_t* weight,
    const scalar_t* weight_scl,
    const uint8_t* weight_zp,
    void* bias,
    uint8_t* reorder_buffer,
    int64_t calib_gz,
    uint32_t m,
    uint32_t n,
    uint32_t k,
    scalar_t* out0, // Q
    scalar_t* out1, // K
    scalar_t* out2, // V
    bool need_reorder) {
  auto& dpcpp_queue = dpcppGetCurrentQueue();

  if (!(k == 4096 && (n == 4096 || n == 6144))) {
    std::cout << "k should be 4096 and n should be 4096" << std::endl;
    // m is input vector num
    return;
  }

  uint32_t pixelPerGroupCommonDim4096 = 16;

  // reorder for the weights and scaling
  // Assume group size 32.   4096 / 32 = 128
  if (need_reorder) {
    sycl::event e0;
    uint8_t* weight_reorder = (uint8_t*)weight;
    uint8_t* reorderTmp = reorder_buffer;
    // reorder weight
    e0 = dpcpp_queue.submit([&](handler& cgh) {
      cgh.parallel_for(sycl::range<2>(k, n), [=](sycl::id<2> idx) {
        int i = idx[0];
        int j = idx[1];

        int32_t origIdx = i * n + j;
        int32_t afterIdx = j * k + i;

        int8_t tmp = weight_reorder[origIdx / 2];
        if (origIdx % 2 == 0) {
          tmp = tmp & 0xf;
        } else {
          tmp = tmp >> 4;
        }
        reorderTmp[afterIdx] = tmp;
      });
    });
    e0.wait();
    report_time("reshape1 kernel time", e0, e0);
    e0 = dpcpp_queue.submit([&](handler& cgh) {
      cgh.parallel_for(sycl::range<2>(k / 2, n), [=](sycl::id<2> idx) {
        int i = idx[0];
        int j = idx[1];

        int32_t afterIdxInt4 = j * k / 2 + i;

        int8_t tmpLow = reorderTmp[afterIdxInt4 * 2];
        int8_t tmpHigh = reorderTmp[afterIdxInt4 * 2 + 1];

        int8_t tmp = (tmpLow & 0xf) | (tmpHigh << 4);

        weight_reorder[afterIdxInt4] = tmp;
      });
    });
    e0.wait();
    report_time("reshape2 kernel time", e0, e0);

    // reorder scale
    fp16* reorderTmpScal = (fp16*)reorder_buffer;
    fp16* weight_scl_reorder = (fp16*)weight_scl;
    e0 = dpcpp_queue.submit([&](handler& cgh) {
      cgh.parallel_for(sycl::range<2>(k / 32, n), [=](sycl::id<2> idx) {
        int i = idx[0];
        int j = idx[1];

        int32_t origIdx = i * n + j;
        int32_t afterIdx = j * k / 32 + i;

        reorderTmpScal[afterIdx] = weight_scl_reorder[origIdx];
      });
    });
    e0.wait();
    report_time("reshape3 kernel time", e0, e0);
    e0 = dpcpp_queue.submit([&](handler& cgh) {
      cgh.parallel_for(sycl::range<2>(k / 32, n), [=](sycl::id<2> idx) {
        int i = idx[0];
        int j = idx[1];

        int32_t afterIdx = j * k / 32 + i;

        weight_scl_reorder[afterIdx] = reorderTmpScal[afterIdx];
      });
    });
    e0.wait();
    report_time("reshape4 kernel time", e0, e0);
    dpcpp_queue.wait();
    return;
  }

  if (m == 1) // GEMV
  {
    // ************ fused kqv gemv
    int groupsV2 =
        (n + pixelPerGroupCommonDim4096 - 1) / pixelPerGroupCommonDim4096;
    int localThread[2];
    localThread[0] = 16;
    localThread[1] = 16;

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
              matrixMulCommonDim4096Int4NoReshapeNx16V3_ipex2_fused<4>(
                  (uint8_t*)weight,
                  (uint8_t*)input,
                  (uint8_t*)out0,
                  (uint8_t*)out1,
                  (uint8_t*)out2,
                  (uint8_t*)weight_scl,
                  ndi);
            });
      });
      e.wait();
      double etime = report_time("GEMV kernel qkv fused time", e, e);
    }
    // ************

  } else // GEMM
  {
    int groupReduce2048H = (4096 + 15) / 16;
    int groupReduce2048V = 1;
    int localReduce2048H = 64; // internalPrecision == 0  (fp32), not 32
    int localReduce2048V = 1;
    sycl::range<2> GlobalRangeReduce2048(
        groupReduce2048H * localReduce2048H,
        groupReduce2048V * localReduce2048V);
    sycl::range<2> LocalRangeReduce2048(localReduce2048H, localReduce2048V);
    sycl::nd_range<2> RangeReduce2048(
        GlobalRangeReduce2048, LocalRangeReduce2048);

    sycl::event e;
    int lastReduce = 0;
    if (k == 4096) {
      for (int ii = 0; ii < 2; ii++) {
        if (ii == 2 - 1) {
          lastReduce = 1;
        } else {
          lastReduce = 0;
        }
        e = dpcpp_queue.submit([&](handler& cgh) {
          cgh.parallel_for(
              RangeReduce2048, [=](nd_item<2> ndi) SYCL_ESIMD_KERNEL {
                gemmReduce2048WeightsQ40InputFp32_ipex(
                    (uint8_t*)weight,
                    (uint8_t*)input,
                    (uint8_t*)out0,
                    (uint8_t*)weight_scl,
                    k,
                    m /*tokenLen*/,
                    ii,
                    lastReduce,
                    ndi);
              });
        });
        // YC e.wait();
        // double etime = report_time("GEMV kernel time", e, e);
      }
    }
    int groupReduce2048H_kv = (1024 + 15) / 16;
    int groupReduce2048V_kv = 1;
    int localReduce2048H_kv = 64; // internalPrecision == 0  (fp32), not 32
    int localReduce2048V_kv = 1;
    sycl::range<2> GlobalRangeReduce2048_kv(
        groupReduce2048H_kv * localReduce2048H_kv,
        groupReduce2048V_kv * localReduce2048V_kv);
    sycl::range<2> LocalRangeReduce2048_kv(
        localReduce2048H_kv, localReduce2048V_kv);
    sycl::nd_range<2> RangeReduce2048_kv(
        GlobalRangeReduce2048_kv, LocalRangeReduce2048_kv);
    if (k == 4096) {
      for (int ii = 0; ii < 2; ii++) {
        if (ii == 2 - 1) {
          lastReduce = 1;
        } else {
          lastReduce = 0;
        }
        e = dpcpp_queue.submit([&](handler& cgh) {
          cgh.parallel_for(
              RangeReduce2048_kv, [=](nd_item<2> ndi) SYCL_ESIMD_KERNEL {
                gemmReduce2048WeightsQ40InputFp32_ipex(
                    (uint8_t*)(weight + 2048 * 4096 * 1),
                    (uint8_t*)input,
                    (uint8_t*)out1,
                    (uint8_t*)(weight_scl + 128 * 4096 * 1),
                    k,
                    m /*tokenLen*/,
                    ii,
                    lastReduce,
                    ndi);
              });
        });
        // YC e.wait();
        // double etime = report_time("GEMV kernel time", e, e);
      }
    }
    if (k == 4096) {
      for (int ii = 0; ii < 2; ii++) {
        if (ii == 2 - 1) {
          lastReduce = 1;
        } else {
          lastReduce = 0;
        }
        e = dpcpp_queue.submit([&](handler& cgh) {
          cgh.parallel_for(
              RangeReduce2048_kv, [=](nd_item<2> ndi) SYCL_ESIMD_KERNEL {
                gemmReduce2048WeightsQ40InputFp32_ipex(
                    (uint8_t*)(weight + 2048 * 4096 + 512 * 4096),
                    (uint8_t*)input,
                    (uint8_t*)out2,
                    (uint8_t*)(weight_scl + 128 * 4096 + 128 * 1024),
                    k,
                    m /*tokenLen*/,
                    ii,
                    lastReduce,
                    ndi);
              });
        });
        // YC e.wait();
        // double etime = report_time("GEMV kernel time", e, e);
      }
    }
  }

  return;
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
    int64_t calib_gz,
    bool need_reorder) {
  // YC std::cout << "start mm_esimd_int4: " << need_reorder << std::endl;
  TORCH_CHECK(input.scalar_type() == ScalarType::Half);
  auto input_flat = input.flatten(0, -2); // 1, 1, 4096 -> 1, 4096
  auto weight_flat = weight.flatten(0, -2); // 4096, 5504, 1 -> 4096, 5504

  // xpu::COMPUTE_ENG real_eng =
  //     choose_compute_eng(xpu::COMPUTE_ENG::ESIMD, input, weight);
  // bool compute_eng_valid = (real_eng == xpu::COMPUTE_ENG::ESIMD);
  bool compute_eng_valid = true;

  uint32_t m = input_flat.sizes()[0]; // 1
  uint32_t k = input_flat.sizes()[1]; // 4096
  uint32_t n = weight.sizes()[1] * 2; // 11008
  Tensor reorder_buffer;
  if (need_reorder)
    reorder_buffer = at::empty({k, n}, weight.options());
  auto output = at::empty({m, n}, input.options()); // 11008, 11008

  TORCH_CHECK(input_flat.dim() == 2 && weight_flat.dim() == 2);
  // impl::dump_element(weight_flat, 10, "weight first 10 elem: ");
  // impl::dump_element(reorder_buffer, 10, "reorder_buffer first 10 elem: ");
  // impl::dump_element(weight_scl, 10, "scal first 10 elem: ");

  if (compute_eng_valid) {
    // YC std::cout << "get in esimd int4 gemm" << std::endl;
    IPEX_DISPATCH_FLOATING_TYPES_AND_HALF(
        input_flat.scalar_type(), "XeGemm_int4_esimd", [&] {
          impl::gemm_int4_esimd_kernel<scalar_t, uint8_t>(
              input_flat.data_ptr<scalar_t>(),
              weight_flat.data_ptr<uint8_t>(),
              output.data_ptr<scalar_t>(),
              weight_scl.data_ptr<scalar_t>(),
              weight_zp.data_ptr<uint8_t>(),
              need_reorder ? reorder_buffer.data_ptr<uint8_t>() : nullptr,
              calib_gz,
              m,
              n,
              k,
              need_reorder);
        });
  } else {
    AT_ERROR("GEMM INT4: invalid COMPUTE_ENG!");
  }
  // impl::dump_element(weight_flat, 10, "weight before output first 10 elem:
  // "); impl::dump_element(
  //    reorder_buffer, 10, "reorder_buffer  before output first 10 elem: ");
  // impl::dump_element(output, 10, "output first 10 elem: ");
  return resize_as_mat2(input, output);
}

static void qkv_mm_esimd_int4(
    const Tensor& input_,
    const Tensor& weight,
    const Tensor& weight_scl,
    const Tensor& weight_zp,
    const optional<Tensor>& bias_,
    const Tensor& out0_,
    const Tensor& out1_,
    const Tensor& out2_,
    int64_t calib_gz,
    bool need_reorder) {
  // YC std::cout << "start qkv_mm_esimd_int4" << std::endl;

  // xpu::COMPUTE_ENG real_eng =
  //     choose_compute_eng(xpu::COMPUTE_ENG::ESIMD, input_, weight);
  // bool compute_eng_valid = (real_eng == xpu::COMPUTE_ENG::ESIMD);
  bool compute_eng_valid = true;

  auto input = input_.flatten(0, -2);
  if (input.scalar_type() == ScalarType::Float)
    input = input.to(at::kHalf);
  auto out0 = out0_.flatten(0, -2);
  auto out1 = out1_.flatten(0, -2);
  auto out2 = out2_.flatten(0, -2);
  // input: m,k; weight: 3,k,n, bias(opt): 3,n
  TORCH_CHECK(input.dim() == 2 && weight.dim() == 3);
  TORCH_CHECK(out0.dim() == 2 && out1.dim() == 2 && out2.dim() == 2);
  int m = input.sizes()[0]; // m is input vector num
  int k = input.sizes()[1];
  int n = weight.sizes()[2] * 2;

  bool has_bias = bias_.has_value();
  if (has_bias) {
    auto bias = bias_.value();
    TORCH_CHECK(
        bias.dim() == 2 && bias.sizes()[0] == 3 && bias.sizes()[1] == n);
  }
  // TORCH_CHECK(
  //     out0.sizes()[0] == m && out1.sizes()[0] == m && out2.sizes()[0] == m);
  // TORCH_CHECK(
  //     out0.sizes()[1] == n && out1.sizes()[1] == n && out2.sizes()[1] == n);

  TORCH_CHECK(
      input.scalar_type() == kHalf &&
      (weight.scalar_type() == kQUInt8 || weight.scalar_type() == kByte ||
       weight.scalar_type() == kChar));
  Tensor reorder_buffer;
  if (need_reorder)
    reorder_buffer = at::empty({k, n}, weight.options());
  // impl::dump_element(weight, 10, "weight first 10 elem: ");

  if (compute_eng_valid) {
    // YC std::cout << "get in esimd int4 gemm" << std::endl;
    IPEX_DISPATCH_FLOATING_TYPES_AND_HALF(
        input.scalar_type(), "XeGemm_int4_esimd", [&] {
          impl::qkv_gemm_int4_esimd_kernel<scalar_t, uint8_t>(
              input.data_ptr<scalar_t>(),
              weight.data_ptr<uint8_t>(),
              weight_scl.data_ptr<scalar_t>(),
              weight_zp.data_ptr<uint8_t>(),
              bias_.has_value() ? bias_.value().data_ptr() : (void*)nullptr,
              need_reorder ? reorder_buffer.data_ptr<uint8_t>() : nullptr,
              calib_gz,
              m,
              n,
              k,
              out0.data_ptr<scalar_t>(),
              out1.data_ptr<scalar_t>(),
              out2.data_ptr<scalar_t>(),
              need_reorder);
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
  IPEX_OP_REGISTER(
      "qkv_mm_esimd_int4.xpu", at::AtenIpexTypeXPU::qkv_mm_esimd_int4);
  // IPEX_OP_REGISTER("mm_esimd_int4.xpu",
  // at::AtenIpexTypeXPU::mm_weight_reorder_esimd_int4); IPEX_OP_REGISTER(
  //     "qkv_mm_esimd_int4.xpu",
  //     at::AtenIpexTypeXPU::qkv_weight_reorde_mm_esimd_int4);
}
} // namespace
