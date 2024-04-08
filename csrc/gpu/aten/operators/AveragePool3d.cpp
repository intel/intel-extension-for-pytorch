#include <ATen/AccumulateType.h>
#include <ATen/Config.h>
#include <ATen/NativeFunctions.h>
#include <ATen/ceil_div.h>
#include <ATen/native/Pool.h>

#include <core/detail/IndexUtils.h>
#include <oneDNN/oneDNN.h>
#include <vector>
#include "comm/ATDispatch.h"
#include "comm/AccumulateType.h"
#include "comm/Atomics.h"
#include "comm/Numerics.h"
#include "comm/RegistrationDeclarations.h"
#include "utils/ComputeEngine.h"

using namespace dnnl;
using namespace at::native;
using namespace xpu::dpcpp;
using namespace xpu::oneDNN;

namespace at {
namespace AtenIpexTypeXPU {
namespace impl {

template <typename scalar_t, typename accscalar_t, typename index_t>
struct AvgPool3dBackwardOutFrameAtomicKernelFunctor {
  void operator()(sycl::nd_item<3> item) const {
    index_t oCol = item.get_global_id()[2];
    index_t oRow = item.get_global_id()[1];
    index_t oFrame = (item.get_group(0) + offsetZ) % osize1;
    index_t slice = (item.get_group(0) + offsetZ) / osize1;

    if (oRow < osize2 && oCol < osize3) {
      index_t tstart = oFrame * dT - padT;
      index_t hstart = oRow * dH - padH;
      index_t wstart = oCol * dW - padW;
      index_t tend = Numerics<index_t>::min(tstart + kT, isize1 + padT);
      index_t hend = Numerics<index_t>::min(hstart + kH, isize2 + padH);
      index_t wend = Numerics<index_t>::min(wstart + kW, isize3 + padW);
      index_t pool_size = (tend - tstart) * (hend - hstart) * (wend - wstart);
      tstart = Numerics<index_t>::max(tstart, 0);
      hstart = Numerics<index_t>::max(hstart, 0);
      wstart = Numerics<index_t>::max(wstart, 0);
      tend = Numerics<index_t>::min(tend, isize1);
      hend = Numerics<index_t>::min(hend, isize2);
      wend = Numerics<index_t>::min(wend, isize3);

      accscalar_t divide_factor;
      if (divisor_override) {
        divide_factor = static_cast<accscalar_t>(divisor_override);
      } else {
        if (count_include_pad) {
          divide_factor = static_cast<accscalar_t>(pool_size);
        } else {
          divide_factor = static_cast<accscalar_t>(
              (tend - tstart) * (hend - hstart) * (wend - wstart));
        }
      }

      scalar_t val = static_cast<scalar_t>(
          static_cast<accscalar_t>(grad_output_ptr
                                       [slice * ostride0 + oFrame * ostride1 +
                                        oRow * ostride2 + oCol * ostride3]) /
          divide_factor);

      for (index_t iFrame = tstart; iFrame < tend; ++iFrame) {
        for (index_t iRow = hstart; iRow < hend; ++iRow) {
          for (index_t iCol = wstart; iCol < wend; ++iCol) {
            const index_t index = slice * istride0 + iFrame * istride1 +
                iRow * istride2 + iCol * istride3;
            atomicAdd(
                (dpcpp_global_ptr_pt<scalar_t>)&grad_input_ptr[index], val);
          }
        }
      }
    }
  }
  AvgPool3dBackwardOutFrameAtomicKernelFunctor(
      int kT_,
      int kH_,
      int kW_,
      int dT_,
      int dH_,
      int dW_,
      int padT_,
      int padH_,
      int padW_,
      bool count_include_pad_,
      int offsetZ_,
      int totalZ_,
      int divisor_override_,
      index_t ostride0_,
      index_t ostride1_,
      index_t ostride2_,
      index_t ostride3_,
      index_t istride0_,
      index_t istride1_,
      index_t istride2_,
      index_t istride3_,
      index_t width_group_size_,
      index_t height_group_size_,
      index_t width_group_range_,
      index_t height_group_range_,
      index_t z_group_range_,
      scalar_t* grad_output_ptr_,
      scalar_t* grad_input_ptr_,
      index_t osize1_,
      index_t osize2_,
      index_t osize3_,
      index_t isize1_,
      index_t isize2_,
      index_t isize3_)
      : kT(kT_),
        kH(kH_),
        kW(kW_),
        dT(dT_),
        dH(dH_),
        dW(dW_),
        padT(padT_),
        padH(padH_),
        padW(padW_),
        count_include_pad(count_include_pad_),
        offsetZ(offsetZ_),
        totalZ(totalZ_),
        divisor_override(divisor_override_),
        ostride0(ostride0_),
        ostride1(ostride1_),
        ostride2(ostride2_),
        ostride3(ostride3_),
        istride0(istride0_),
        istride1(istride1_),
        istride2(istride2_),
        istride3(istride3_),
        width_group_size(width_group_size_),
        height_group_size(height_group_size_),
        width_group_range(width_group_range_),
        height_group_range(height_group_range_),
        z_group_range(z_group_range_),
        grad_output_ptr(grad_output_ptr_),
        grad_input_ptr(grad_input_ptr_),
        osize1(osize1_),
        osize2(osize2_),
        osize3(osize3_),
        isize1(isize1_),
        isize2(isize2_),
        isize3(isize3_) {}

 private:
  int kT;
  int kH;
  int kW;
  int dT;
  int dH;
  int dW;
  int padT;
  int padH;
  int padW;
  bool count_include_pad;
  int offsetZ;
  int totalZ;
  int divisor_override;
  index_t ostride0;
  index_t ostride1;
  index_t ostride2;
  index_t ostride3;
  index_t istride0;
  index_t istride1;
  index_t istride2;
  index_t istride3;
  index_t width_group_size;
  index_t height_group_size;
  index_t width_group_range;
  index_t height_group_range;
  index_t z_group_range;
  scalar_t* grad_output_ptr;
  scalar_t* grad_input_ptr;
  index_t osize1;
  index_t osize2;
  index_t osize3;
  index_t isize1;
  index_t isize2;
  index_t isize3;
};

template <typename scalar_t, typename accscalar_t, typename index_t>
void avg_pool3d_backward_out_frame_atomic(
    const Tensor& grad_output,
    Tensor& grad_input,
    int kT,
    int kH,
    int kW,
    int dT,
    int dH,
    int dW,
    int padT,
    int padH,
    int padW,
    bool count_include_pad,
    int offsetZ,
    int totalZ,
    int divisor_override) {
  index_t ostride0 = grad_output.stride(0);
  index_t ostride1 = grad_output.stride(1);
  index_t ostride2 = grad_output.stride(2);
  index_t ostride3 = grad_output.stride(3);
  index_t istride0 = grad_input.stride(0);
  index_t istride1 = grad_input.stride(1);
  index_t istride2 = grad_input.stride(2);
  index_t istride3 = grad_input.stride(3);

  // width size is fixed size = 32, height dim equals = dpcppMaxWorkGroupSize /
  // width_size
  index_t width_group_size = 32;
  index_t height_group_size = dpcppMaxWorkGroupSize() / width_group_size;
  index_t width_group_range =
      ceil_div<index_t>(grad_output.size(-1), width_group_size);
  index_t height_group_range =
      ceil_div<index_t>(grad_output.size(-2), height_group_size);

  index_t z_group_range = totalZ > 65535 ? 65535 : totalZ;

  auto grad_output_ptr = grad_output.data_ptr<scalar_t>();
  auto grad_input_ptr = grad_input.data_ptr<scalar_t>();

  index_t osize1 = grad_output.size(1);
  index_t osize2 = grad_output.size(2);
  index_t osize3 = grad_output.size(3);

  index_t isize1 = grad_input.size(1);
  index_t isize2 = grad_input.size(2);
  index_t isize3 = grad_input.size(3);

  auto cgf = DPCPP_Q_CGF(cgh) {
    AvgPool3dBackwardOutFrameAtomicKernelFunctor<scalar_t, accscalar_t, index_t>
        kfn(kT,
            kH,
            kW,
            dT,
            dH,
            dW,
            padT,
            padH,
            padW,
            count_include_pad,
            offsetZ,
            totalZ,
            divisor_override,
            ostride0,
            ostride1,
            ostride2,
            ostride3,
            istride0,
            istride1,
            istride2,
            istride3,
            width_group_size,
            height_group_size,
            width_group_range,
            height_group_range,
            z_group_range,
            grad_output_ptr,
            grad_input_ptr,
            osize1,
            osize2,
            osize3,
            isize1,
            isize2,
            isize3);

    cgh.parallel_for<decltype(kfn)>(
        sycl::nd_range<3>(
            sycl::range<3>{
                z_group_range,
                height_group_range * height_group_size,
                width_group_range * width_group_size,
            },
            sycl::range<3>{1, height_group_size, width_group_size}),
        kfn);
  };

  DPCPP_Q_SUBMIT(dpcppGetCurrentQueue(), cgf);
}

template <typename scalar_t, typename accscalar_t, typename index_t>
struct AvgPool3dBackwardOutFrameStride1KernelFunctor {
  void operator()(sycl::nd_item<3> item) const {
    index_t iCol = item.get_global_id()[2];
    index_t iRow = item.get_global_id()[1];
    index_t iFrame = (item.get_group(0) + offsetZ) % isize1;
    index_t slice = (item.get_group(0) + offsetZ) / isize1;

    if (iRow < isize2 && iCol < isize3) {
      accscalar_t sum = 0.0;
      scalar_t* gOut =
          &grad_output_ptr
              [slice * ostride0 +
               Numerics<index_t>::max(0, iFrame - kT + 1) * ostride1 +
               Numerics<index_t>::max(0, iRow - kH + 1) * ostride2 +
               Numerics<index_t>::max(0, iCol - kW + 1) * ostride3];
      index_t frameOffset = 0;
      for (index_t oFrame = Numerics<index_t>::max(0, iFrame - kT + 1);
           oFrame < Numerics<index_t>::min(iFrame + 1, osize1);
           ++oFrame) {
        index_t rowOffset = frameOffset;
        for (index_t oRow = Numerics<index_t>::max(0, iRow - kH + 1);
             oRow < Numerics<index_t>::min(iRow + 1, osize2);
             ++oRow) {
          index_t colOffset = rowOffset;
          for (index_t oCol = Numerics<index_t>::max(0, iCol - kW + 1);
               oCol < Numerics<index_t>::min(iCol + 1, osize3);
               ++oCol) {
            sum += gOut[colOffset];
            ++colOffset;
          }
          rowOffset += osize3;
        }
        frameOffset += osize2 * osize3;
      }
      grad_input_ptr
          [slice * istride0 + iFrame * istride1 + iRow * istride2 +
           iCol * istride3] = static_cast<scalar_t>(sum * normFactor);
    }
  }
  AvgPool3dBackwardOutFrameStride1KernelFunctor(
      int kT_,
      int kH_,
      int kW_,
      accscalar_t normFactor_,
      int offsetZ_,
      int totalZ_,
      index_t ostride0_,
      index_t ostride1_,
      index_t ostride2_,
      index_t ostride3_,
      index_t istride0_,
      index_t istride1_,
      index_t istride2_,
      index_t istride3_,
      index_t width_group_size_,
      index_t height_group_size_,
      index_t width_group_range_,
      index_t height_group_range_,
      index_t z_group_range_,
      scalar_t* grad_output_ptr_,
      scalar_t* grad_input_ptr_,
      index_t osize1_,
      index_t osize2_,
      index_t osize3_,
      index_t isize1_,
      index_t isize2_,
      index_t isize3_)
      : kT(kT_),
        kH(kH_),
        kW(kW_),
        normFactor(normFactor_),
        offsetZ(offsetZ_),
        totalZ(totalZ_),
        ostride0(ostride0_),
        ostride1(ostride1_),
        ostride2(ostride2_),
        ostride3(ostride3_),
        istride0(istride0_),
        istride1(istride1_),
        istride2(istride2_),
        istride3(istride3_),
        width_group_size(width_group_size_),
        height_group_size(height_group_size_),
        width_group_range(width_group_range_),
        height_group_range(height_group_range_),
        z_group_range(z_group_range_),
        grad_output_ptr(grad_output_ptr_),
        grad_input_ptr(grad_input_ptr_),
        osize1(osize1_),
        osize2(osize2_),
        osize3(osize3_),
        isize1(isize1_),
        isize2(isize2_),
        isize3(isize3_) {}

 private:
  int kT;
  int kH;
  int kW;
  accscalar_t normFactor;
  int offsetZ;
  int totalZ;
  index_t ostride0;
  index_t ostride1;
  index_t ostride2;
  index_t ostride3;
  index_t istride0;
  index_t istride1;
  index_t istride2;
  index_t istride3;
  index_t width_group_size;
  index_t height_group_size;
  index_t width_group_range;
  index_t height_group_range;
  index_t z_group_range;
  scalar_t* grad_output_ptr;
  scalar_t* grad_input_ptr;
  index_t osize1;
  index_t osize2;
  index_t osize3;
  index_t isize1;
  index_t isize2;
  index_t isize3;
};

template <typename scalar_t, typename accscalar_t, typename index_t>
void avg_pool3d_backward_out_frame_stride1(
    const Tensor& grad_output,
    Tensor& grad_input,
    int kT,
    int kH,
    int kW,
    accscalar_t normFactor,
    int offsetZ,
    int totalZ) {
  index_t ostride0 = grad_output.stride(0);
  index_t ostride1 = grad_output.stride(1);
  index_t ostride2 = grad_output.stride(2);
  index_t ostride3 = grad_output.stride(3);
  index_t istride0 = grad_input.stride(0);
  index_t istride1 = grad_input.stride(1);
  index_t istride2 = grad_input.stride(2);
  index_t istride3 = grad_input.stride(3);

  // width size is fixed size = 32, height dim equals = dpcppMaxWorkGroupSize /
  // width_size
  index_t width_group_size = 32;
  index_t height_group_size = dpcppMaxWorkGroupSize() / width_group_size;
  index_t width_group_range =
      ceil_div<index_t>(grad_output.size(-1), width_group_size);
  index_t height_group_range =
      ceil_div<index_t>(grad_output.size(-2), height_group_size);

  index_t z_group_range = totalZ > 65535 ? 65535 : totalZ;

  auto grad_output_ptr = grad_output.data_ptr<scalar_t>();
  auto grad_input_ptr = grad_input.data_ptr<scalar_t>();

  index_t osize1 = grad_output.size(1);
  index_t osize2 = grad_output.size(2);
  index_t osize3 = grad_output.size(3);

  index_t isize1 = grad_input.size(1);
  index_t isize2 = grad_input.size(2);
  index_t isize3 = grad_input.size(3);
  auto cgf = DPCPP_Q_CGF(cgh) {
    AvgPool3dBackwardOutFrameStride1KernelFunctor<
        scalar_t,
        accscalar_t,
        index_t>
        kfn(kT,
            kH,
            kW,
            normFactor,
            offsetZ,
            totalZ,
            ostride0,
            ostride1,
            ostride2,
            ostride3,
            istride0,
            istride1,
            istride2,
            istride3,
            width_group_size,
            height_group_size,
            width_group_range,
            height_group_range,
            z_group_range,
            grad_output_ptr,
            grad_input_ptr,
            osize1,
            osize2,
            osize3,
            isize1,
            isize2,
            isize3);

    cgh.parallel_for<decltype(kfn)>(
        sycl::nd_range<3>(
            sycl::range<3>{
                z_group_range,
                height_group_range * height_group_size,
                width_group_range * width_group_size,
            },
            sycl::range<3>{1, height_group_size, width_group_size}),
        kfn);
  };

  DPCPP_Q_SUBMIT(dpcppGetCurrentQueue(), cgf);
}

template <typename scalar_t, typename accscalar_t, typename index_t>
struct AvgPool3dBackwardOutFrameKernelFunctor {
  void operator()(sycl::nd_item<3> item) const {
    index_t oCol = item.get_global_id()[2];
    index_t oRow = item.get_global_id()[1];
    index_t oFrame = (item.get_group(0) + offsetZ) % osize1;
    index_t slice = (item.get_group(0) + offsetZ) / osize1;

    if (oRow < osize2 && oCol < osize3) {
      index_t tstart = oFrame * dT - padT;
      index_t hstart = oRow * dH - padH;
      index_t wstart = oCol * dW - padW;
      index_t tend = Numerics<index_t>::min(tstart + kT, isize1 + padT);
      index_t hend = Numerics<index_t>::min(hstart + kH, isize2 + padH);
      index_t wend = Numerics<index_t>::min(wstart + kW, isize3 + padW);
      index_t pool_size = (tend - tstart) * (hend - hstart) * (wend - wstart);
      tstart = Numerics<index_t>::max(tstart, 0);
      hstart = Numerics<index_t>::max(hstart, 0);
      wstart = Numerics<index_t>::max(wstart, 0);
      tend = Numerics<index_t>::min(tend, isize1);
      hend = Numerics<index_t>::min(hend, isize2);
      wend = Numerics<index_t>::min(wend, isize3);

      accscalar_t divide_factor;
      if (divisor_override) {
        divide_factor = static_cast<accscalar_t>(divisor_override);
      } else {
        if (count_include_pad) {
          divide_factor = static_cast<accscalar_t>(pool_size);
        } else {
          divide_factor = static_cast<accscalar_t>(
              (tend - tstart) * (hend - hstart) * (wend - wstart));
        }
      }

      scalar_t val = static_cast<scalar_t>(
          static_cast<accscalar_t>(grad_output_ptr
                                       [slice * ostride0 + oFrame * ostride1 +
                                        oRow * ostride2 + oCol * ostride3]) /
          divide_factor);
      for (index_t iFrame = tstart; iFrame < tend; ++iFrame) {
        for (index_t iRow = hstart; iRow < hend; ++iRow) {
          for (index_t iCol = wstart; iCol < wend; ++iCol) {
            grad_input_ptr
                [slice * istride0 + iFrame * istride1 + iRow * istride2 +
                 iCol * istride3] = val;
          }
        }
      }
    }
  }
  AvgPool3dBackwardOutFrameKernelFunctor(
      int kT_,
      int kH_,
      int kW_,
      int dT_,
      int dH_,
      int dW_,
      int padT_,
      int padH_,
      int padW_,
      bool count_include_pad_,
      int offsetZ_,
      int totalZ_,
      int divisor_override_,
      index_t ostride0_,
      index_t ostride1_,
      index_t ostride2_,
      index_t ostride3_,
      index_t istride0_,
      index_t istride1_,
      index_t istride2_,
      index_t istride3_,
      index_t width_group_size_,
      index_t height_group_size_,
      index_t width_group_range_,
      index_t height_group_range_,
      index_t z_group_range_,
      scalar_t* grad_output_ptr_,
      scalar_t* grad_input_ptr_,
      index_t osize1_,
      index_t osize2_,
      index_t osize3_,
      index_t isize1_,
      index_t isize2_,
      index_t isize3_)
      : kT(kT_),
        kH(kH_),
        kW(kW_),
        dT(dT_),
        dH(dH_),
        dW(dW_),
        padT(padT_),
        padH(padH_),
        padW(padW_),
        count_include_pad(count_include_pad_),
        offsetZ(offsetZ_),
        totalZ(totalZ_),
        divisor_override(divisor_override_),
        ostride0(ostride0_),
        ostride1(ostride1_),
        ostride2(ostride2_),
        ostride3(ostride3_),
        istride0(istride0_),
        istride1(istride1_),
        istride2(istride2_),
        istride3(istride3_),
        width_group_size(width_group_size_),
        height_group_size(height_group_size_),
        width_group_range(width_group_range_),
        height_group_range(height_group_range_),
        z_group_range(z_group_range_),
        grad_output_ptr(grad_output_ptr_),
        grad_input_ptr(grad_input_ptr_),
        osize1(osize1_),
        osize2(osize2_),
        osize3(osize3_),
        isize1(isize1_),
        isize2(isize2_),
        isize3(isize3_) {}

 private:
  int kT;
  int kH;
  int kW;
  int dT;
  int dH;
  int dW;
  int padT;
  int padH;
  int padW;
  bool count_include_pad;
  int offsetZ;
  int totalZ;
  int divisor_override;
  index_t ostride0;
  index_t ostride1;
  index_t ostride2;
  index_t ostride3;
  index_t istride0;
  index_t istride1;
  index_t istride2;
  index_t istride3;
  index_t width_group_size;
  index_t height_group_size;
  index_t width_group_range;
  index_t height_group_range;
  index_t z_group_range;
  scalar_t* grad_output_ptr;
  scalar_t* grad_input_ptr;
  index_t osize1;
  index_t osize2;
  index_t osize3;
  index_t isize1;
  index_t isize2;
  index_t isize3;
};

template <typename scalar_t, typename accscalar_t, typename index_t>
void avg_pool3d_backward_out_frame(
    const Tensor& grad_output,
    Tensor& grad_input,
    int kT,
    int kH,
    int kW,
    int dT,
    int dH,
    int dW,
    int padT,
    int padH,
    int padW,
    bool count_include_pad,
    int offsetZ,
    int totalZ,
    int divisor_override) {
  index_t ostride0 = grad_output.stride(0);
  index_t ostride1 = grad_output.stride(1);
  index_t ostride2 = grad_output.stride(2);
  index_t ostride3 = grad_output.stride(3);
  index_t istride0 = grad_input.stride(0);
  index_t istride1 = grad_input.stride(1);
  index_t istride2 = grad_input.stride(2);
  index_t istride3 = grad_input.stride(3);

  // width size is fixed size = 32, height dim equals = dpcppMaxWorkGroupSize /
  // width_size
  index_t width_group_size = 32;
  index_t height_group_size = dpcppMaxWorkGroupSize() / width_group_size;
  index_t width_group_range =
      ceil_div<index_t>(grad_output.size(-1), width_group_size);
  index_t height_group_range =
      ceil_div<index_t>(grad_output.size(-2), height_group_size);

  index_t z_group_range = totalZ > 65535 ? 65535 : totalZ;

  auto grad_output_ptr = grad_output.data_ptr<scalar_t>();
  auto grad_input_ptr = grad_input.data_ptr<scalar_t>();

  index_t osize1 = grad_output.size(1);
  index_t osize2 = grad_output.size(2);
  index_t osize3 = grad_output.size(3);

  index_t isize1 = grad_input.size(1);
  index_t isize2 = grad_input.size(2);
  index_t isize3 = grad_input.size(3);
  auto cgf = DPCPP_Q_CGF(cgh) {
    AvgPool3dBackwardOutFrameKernelFunctor<scalar_t, accscalar_t, index_t> kfn(
        kT,
        kH,
        kW,
        dT,
        dH,
        dW,
        padT,
        padH,
        padW,
        count_include_pad,
        offsetZ,
        totalZ,
        divisor_override,
        ostride0,
        ostride1,
        ostride2,
        ostride3,
        istride0,
        istride1,
        istride2,
        istride3,
        width_group_size,
        height_group_size,
        width_group_range,
        height_group_range,
        z_group_range,
        grad_output_ptr,
        grad_input_ptr,
        osize1,
        osize2,
        osize3,
        isize1,
        isize2,
        isize3);

    cgh.parallel_for<decltype(kfn)>(
        sycl::nd_range<3>(
            sycl::range<3>{
                z_group_range,
                height_group_range * height_group_size,
                width_group_range * width_group_size,
            },
            sycl::range<3>{1, height_group_size, width_group_size}),
        kfn);
  };

  DPCPP_Q_SUBMIT(dpcppGetCurrentQueue(), cgf);
}

template <typename scalar_t, typename accscalar_t, typename index_t>
struct AvgPool3dOutFrameKernelFunctor {
  void operator()(sycl::nd_item<3> item) const {
    index_t oCol = item.get_global_id()[2];
    index_t oRow = item.get_global_id()[1];
    index_t oFrame = (item.get_group(0) + offsetZ) % oDepth;
    index_t slice = (item.get_group(0) + offsetZ) / oDepth;
    auto out_data = output_acc;

    if (oRow < oHeight && oCol < oWidth) {
      accscalar_t sum = 0.0f;

      index_t tstart = oFrame * dT - padT;
      index_t hstart = oRow * dH - padH;
      index_t wstart = oCol * dW - padW;
      index_t tend = Numerics<index_t>::min(tstart + kT, iDepth + padT);
      index_t hend = Numerics<index_t>::min(hstart + kH, iHeight + padH);
      index_t wend = Numerics<index_t>::min(wstart + kW, iWidth + padW);
      index_t pool_size = (tend - tstart) * (hend - hstart) * (wend - wstart);

      tstart = Numerics<index_t>::max(tstart, 0);
      hstart = Numerics<index_t>::max(hstart, 0);
      wstart = Numerics<index_t>::max(wstart, 0);
      tend = Numerics<index_t>::min(tend, iDepth);
      hend = Numerics<index_t>::min(hend, iHeight);
      wend = Numerics<index_t>::min(wend, iWidth);

      if (tstart >= tend || hstart >= hend || wstart >= wend) {
        out_data[slice][oFrame][oRow][oCol] = static_cast<scalar_t>(0.0f);
        return;
      }

      accscalar_t divide_factor;
      if (divisor_override) {
        divide_factor = static_cast<accscalar_t>(divisor_override);
      } else {
        if (count_include_pad) {
          divide_factor = static_cast<accscalar_t>(pool_size);
        } else {
          divide_factor = static_cast<accscalar_t>(
              (tend - tstart) * (hend - hstart) * (wend - wstart));
        }
      }

      index_t ti, hi, wi;
      for (ti = tstart; ti < tend; ++ti) {
        for (hi = hstart; hi < hend; ++hi) {
          for (wi = wstart; wi < wend; ++wi) {
            scalar_t val = input_acc[slice][ti][hi][wi];
            sum += val;
          }
        }
      }
      out_data[slice][oFrame][oRow][oCol] =
          static_cast<scalar_t>(sum / divide_factor);
    }
  }
  AvgPool3dOutFrameKernelFunctor(
      int kT_,
      int kH_,
      int kW_,
      int dT_,
      int dH_,
      int dW_,
      int padT_,
      int padH_,
      int padW_,
      bool count_include_pad_,
      int offsetZ_,
      int totalZ_,
      int divisor_override_,
      index_t oWidth_,
      index_t oHeight_,
      index_t oDepth_,
      index_t iWidth_,
      index_t iHeight_,
      index_t iDepth_,
      index_t ostride0_,
      index_t ostride1_,
      index_t ostride2_,
      index_t ostride3_,
      index_t istride0_,
      index_t istride1_,
      index_t istride2_,
      index_t istride3_,
      index_t width_group_size_,
      index_t height_group_size_,
      index_t width_group_range_,
      index_t height_group_range_,
      index_t z_group_range_,
      PackedTensorAccessor64<scalar_t, 4> input_acc_,
      PackedTensorAccessor64<scalar_t, 4> output_acc_)
      : kT(kT_),
        kH(kH_),
        kW(kW_),
        dT(dT_),
        dH(dH_),
        dW(dW_),
        padT(padT_),
        padH(padH_),
        padW(padW_),
        count_include_pad(count_include_pad_),
        offsetZ(offsetZ_),
        totalZ(totalZ_),
        divisor_override(divisor_override_),
        oWidth(oWidth_),
        oHeight(oHeight_),
        oDepth(oDepth_),
        iWidth(iWidth_),
        iHeight(iHeight_),
        iDepth(iDepth_),
        ostride0(ostride0_),
        ostride1(ostride1_),
        ostride2(ostride2_),
        ostride3(ostride3_),
        istride0(istride0_),
        istride1(istride1_),
        istride2(istride2_),
        istride3(istride3_),
        width_group_size(width_group_size_),
        height_group_size(height_group_size_),
        width_group_range(width_group_range_),
        height_group_range(height_group_range_),
        z_group_range(z_group_range_),
        input_acc(input_acc_),
        output_acc(output_acc_) {}

 private:
  int kT;
  int kH;
  int kW;
  int dT;
  int dH;
  int dW;
  int padT;
  int padH;
  int padW;
  bool count_include_pad;
  int offsetZ;
  int totalZ;
  int divisor_override;
  index_t oWidth;
  index_t oHeight;
  index_t oDepth;
  index_t iWidth;
  index_t iHeight;
  index_t iDepth;
  index_t ostride0;
  index_t ostride1;
  index_t ostride2;
  index_t ostride3;
  index_t istride0;
  index_t istride1;
  index_t istride2;
  index_t istride3;
  index_t width_group_size;
  index_t height_group_size;
  index_t width_group_range;
  index_t height_group_range;
  index_t z_group_range;
  PackedTensorAccessor64<scalar_t, 4> input_acc;
  PackedTensorAccessor64<scalar_t, 4> output_acc;
};

template <typename scalar_t, typename accscalar_t, typename index_t>
void avg_pool3d_out_frame(
    Tensor& work_input,
    Tensor& work_output,
    const int kT,
    const int kH,
    const int kW,
    const int dT,
    const int dH,
    const int dW,
    const int padT,
    const int padH,
    const int padW,
    const bool count_include_pad,
    const int offsetZ,
    const int totalZ,
    const int divisor_override) {
  index_t oWidth = work_output.size(-1);
  index_t oHeight = work_output.size(-2);
  index_t oDepth = work_output.size(-3);
  index_t iWidth = work_input.size(-1);
  index_t iHeight = work_input.size(-2);
  index_t iDepth = work_input.size(-3);

  index_t ostride0 = work_output.stride(0);
  index_t ostride1 = work_output.stride(1);
  index_t ostride2 = work_output.stride(2);
  index_t ostride3 = work_output.stride(3);

  index_t istride0 = work_input.stride(0);
  index_t istride1 = work_input.stride(1);
  index_t istride2 = work_input.stride(2);
  index_t istride3 = work_input.stride(3);

  // width size is fixed size = 32, height dim equals = dpcppMaxWorkGroupSize /
  // width_size
  index_t width_group_size = 32;
  index_t height_group_size = dpcppMaxWorkGroupSize() / width_group_size;
  index_t width_group_range = ceil_div<index_t>(oHeight, width_group_size);
  index_t height_group_range = ceil_div<index_t>(oHeight, height_group_size);

  index_t z_group_range = totalZ > 65535 ? 65535 : totalZ;

  auto input_acc = work_input.packed_accessor64<scalar_t, 4>();
  auto output_acc = work_output.packed_accessor64<scalar_t, 4>();
  auto cgf = DPCPP_Q_CGF(cgh) {
    AvgPool3dOutFrameKernelFunctor<scalar_t, accscalar_t, index_t> kfn(
        kT,
        kH,
        kW,
        dT,
        dH,
        dW,
        padT,
        padH,
        padW,
        count_include_pad,
        offsetZ,
        totalZ,
        divisor_override,
        oWidth,
        oHeight,
        oDepth,
        iWidth,
        iHeight,
        iDepth,
        ostride0,
        ostride1,
        ostride2,
        ostride3,
        istride0,
        istride1,
        istride2,
        istride3,
        width_group_size,
        height_group_size,
        width_group_range,
        height_group_range,
        z_group_range,
        input_acc,
        output_acc);

    cgh.parallel_for<decltype(kfn)>(
        sycl::nd_range<3>(
            sycl::range<3>{
                z_group_range,
                height_group_range * height_group_size,
                width_group_range * width_group_size,
            },
            sycl::range<3>{1, height_group_size, width_group_size}),
        kfn);
  };

  DPCPP_Q_SUBMIT(dpcppGetCurrentQueue(), cgf);
}

void avg_pool3d_out_template(
    Tensor& output,
    const Tensor& input,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    bool ceil_mode,
    bool count_include_pad,
    c10::optional<int64_t> divisor_override) {
  TORCH_CHECK(
      kernel_size.size() == 1 || kernel_size.size() == 3,
      "avg_pool3d: kernel_size must either be a single int, or a tuple of "
      "three ints");
  const int kD = safe_downcast<int, int64_t>(kernel_size[0]);
  const int kH = kernel_size.size() == 1
      ? kD
      : safe_downcast<int, int64_t>(kernel_size[1]);
  const int kW = kernel_size.size() == 1
      ? kD
      : safe_downcast<int, int64_t>(kernel_size[2]);

  TORCH_CHECK(
      stride.empty() || stride.size() == 1 || stride.size() == 3,
      "avg_pool3d: stride must either be omitted, a single int, or a tuple of "
      "three ints");
  const int dD = stride.empty() ? kD : safe_downcast<int, int64_t>(stride[0]);
  const int dH = stride.empty() ? kH
      : stride.size() == 1      ? dD
                                : safe_downcast<int, int64_t>(stride[1]);
  const int dW = stride.empty() ? kW
      : stride.size() == 1      ? dD
                                : safe_downcast<int, int64_t>(stride[2]);

  TORCH_CHECK(
      padding.size() == 1 || padding.size() == 3,
      "avg_pool3d: padding must either be a single int, or a tuple of three "
      "ints");
  const int padD = safe_downcast<int, int64_t>(padding[0]);
  const int padH =
      padding.size() == 1 ? padD : safe_downcast<int, int64_t>(padding[1]);
  const int padW =
      padding.size() == 1 ? padD : safe_downcast<int, int64_t>(padding[2]);

  /* Applies a 3D average pooling over an input signal composed of
     several input planes. This op only support 4D and 5D input. 4D: Input (C,
     D, H, W),  Output (C, D0, H0, W0) 5D: Input (N, C, D, H, W),  Output (N,
     C, D0, H0, W0)
  */
  TORCH_CHECK(
      (input.ndimension() == 4 || input.ndimension() == 5),
      "non-empty 4D or 5D (batch mode) tensor expected for input");

  /* sizes */
  const int64_t nbatch = input.ndimension() == 5 ? input.size(-5) : 1;
  const int64_t nblock = input.size(-4);
  const int64_t idepth = input.size(-3);
  const int64_t iheight = input.size(-2);
  const int64_t iwidth = input.size(-1);

  const int64_t outputDepth =
      pooling_output_shape<int64_t>(idepth, kD, padD, dD, 1, ceil_mode);
  const int64_t outputHeight =
      pooling_output_shape<int64_t>(iheight, kH, padH, dH, 1, ceil_mode);
  const int64_t outputWidth =
      pooling_output_shape<int64_t>(iwidth, kW, padW, dW, 1, ceil_mode);

  // if divisor==0 then we will ignore it
  int64_t divisor = 0;
  if (divisor_override.has_value()) {
    divisor = divisor_override.value();
  }

  pool3d_shape_check(
      input,
      nblock,
      kD,
      kH,
      kW,
      dD,
      dH,
      dW,
      padD,
      padH,
      padW,
      1,
      1,
      1,
      idepth,
      iheight,
      iwidth,
      outputDepth,
      outputHeight,
      outputWidth,
      "avg_pool3d_out_template()",
      /*check_input_size=*/true);

  xpu::COMPUTE_ENG real_eng =
      choose_compute_eng(xpu::COMPUTE_ENG::BASIC, input);

  // for onednn block format
  if (xpu::COMPUTE_ENG::ONEDNN == real_eng) {
    Tensor input_;
    if (input.ndimension() == 4) {
      // 4D: Input (C, D, H, W),  Output (C, D0, H0, W0)
      // cannot give channels last for 4D tensor from frontend user
      // perspective the 2nd dim is outputDepth, not channel dim
      input_ = input.contiguous();
      output.resize_({nblock, outputDepth, outputHeight, outputWidth});
    } else {
      // 5D: Input (N, C, D, H, W),  Output (N, C, D0, H0, W0)
      // smf supports ChannelsLast3D and Contiguous cases.
      auto smf = input.suggest_memory_format();
      input_ = contiguous_if_needed(input, smf);
      output.resize_(
          {nbatch, nblock, outputDepth, outputHeight, outputWidth}, smf);
    }
    std::vector<int64_t> kernel_size_vec = {kD, kH, kW};
    std::vector<int64_t> stride_vec = {dD, dH, dW};
    std::vector<int64_t> padding_vec = {padD, padH, padW};
    // per oneDNN definition, no dilation means dilation ratio is 0
    std::vector<int64_t> dilation_vec = {0, 0, 0};
    if (count_include_pad) {
      ::xpu::oneDNN::pooling<::xpu::oneDNN::alg::pooling_avg_include_padding>(
          output,
          input_,
          nbatch,
          nblock,
          idepth,
          iheight,
          iwidth,
          outputDepth,
          outputHeight,
          outputWidth,
          stride_vec,
          kernel_size_vec,
          dilation_vec,
          padding_vec,
          padding_vec);
    } else {
      ::xpu::oneDNN::pooling<::xpu::oneDNN::alg::pooling_avg_exclude_padding>(
          output,
          input_,
          nbatch,
          nblock,
          idepth,
          iheight,
          iwidth,
          outputDepth,
          outputHeight,
          outputWidth,
          stride_vec,
          kernel_size_vec,
          dilation_vec,
          padding_vec,
          padding_vec);
    }
    return;
  } else {
    // for plain format
    Tensor work_input = input.contiguous();
    Tensor work_output = output;
    if (input.ndimension() == 5) {
      work_input =
          work_input.reshape({nbatch * nblock, idepth, iheight, iwidth});
      work_output = at::zeros_like(output);
      work_output = work_output.reshape(
          {nbatch * nblock, outputDepth, outputHeight, outputWidth});
    }

    IPEX_DISPATCH_FLOATING_TYPES_AND2(
        kHalf, kBFloat16, input.scalar_type(), "avg_pool3d_out_template", [&] {
          using accscalar_t = acc_type<scalar_t>;
          int64_t totalZ = outputDepth * nblock * nbatch;
          int64_t offsetZ = 0;

          while (totalZ > 0) {
            if (xpu::dpcpp::detail::canUse32BitIndexMath(input)) {
              avg_pool3d_out_frame<scalar_t, accscalar_t, int32_t>(
                  work_input,
                  work_output,
                  kD,
                  kH,
                  kW,
                  dD,
                  dH,
                  dW,
                  padD,
                  padH,
                  padW,
                  count_include_pad,
                  offsetZ,
                  totalZ,
                  divisor);
            } else {
              avg_pool3d_out_frame<scalar_t, accscalar_t, int64_t>(
                  work_input,
                  work_output,
                  kD,
                  kH,
                  kW,
                  dD,
                  dH,
                  dW,
                  padD,
                  padH,
                  padW,
                  count_include_pad,
                  offsetZ,
                  totalZ,
                  divisor);
            }
            totalZ -= 65535;
            offsetZ += 65535;
          }
        });
    auto work_output1 = work_output.resize_as_(output);
    output.copy_(work_output1);
  }
}

Tensor& avg_pool3d_backward_out_template(
    Tensor& gradInput,
    const Tensor& gradOutput,
    const Tensor& input,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    bool ceil_mode,
    bool count_include_pad,
    c10::optional<int64_t> divisor_override) {
  TORCH_CHECK(
      kernel_size.size() == 1 || kernel_size.size() == 3,
      "avg_pool3d: kernel_size must either be a single int, or a tuple of "
      "three ints");
  const int kD = safe_downcast<int, int64_t>(kernel_size[0]);
  const int kH = kernel_size.size() == 1
      ? kD
      : safe_downcast<int, int64_t>(kernel_size[1]);
  const int kW = kernel_size.size() == 1
      ? kD
      : safe_downcast<int, int64_t>(kernel_size[2]);
  std::vector<int64_t> kernel_vec = {kD, kH, kW};

  TORCH_CHECK(
      stride.empty() || stride.size() == 1 || stride.size() == 3,
      "avg_pool3d: stride must either be omitted, a single int, or a tuple of "
      "three ints");
  const int dD = stride.empty() ? kD : safe_downcast<int, int64_t>(stride[0]);
  const int dH = stride.empty() ? kH
      : stride.size() == 1      ? dD
                                : safe_downcast<int, int64_t>(stride[1]);
  const int dW = stride.empty() ? kW
      : stride.size() == 1      ? dD
                                : safe_downcast<int, int64_t>(stride[2]);
  std::vector<int64_t> stride_vec = {dD, dH, dW};

  TORCH_CHECK(
      padding.size() == 1 || padding.size() == 3,
      "avg_pool3d: padding must either be a single int, or a tuple of three "
      "ints");
  const int padD = safe_downcast<int, int64_t>(padding[0]);
  const int padH =
      padding.size() == 1 ? padD : safe_downcast<int, int64_t>(padding[1]);
  const int padW =
      padding.size() == 1 ? padD : safe_downcast<int, int64_t>(padding[2]);
  std::vector<int64_t> padding_vec = {padD, padH, padW};

  TORCH_CHECK(
      (input.ndimension() == 4 || input.ndimension() == 5),
      "non-empty 4D or 5D (batch mode) tensor expected for input");

  TORCH_CHECK(
      (gradOutput.ndimension() == 4 || gradOutput.ndimension() == 5),
      "non-empty 4D or 5D (batch mode) tensor expected for gradOutput");

  /* sizes */
  const int64_t nbatch = input.ndimension() == 5 ? input.size(-5) : 1;
  const int64_t nblock = input.size(-4);
  const int64_t idepth = input.size(-3);
  const int64_t iheight = input.size(-2);
  const int64_t iwidth = input.size(-1);

  const int64_t odepth = gradOutput.size(-3);
  const int64_t oheight = gradOutput.size(-2);
  const int64_t owidth = gradOutput.size(-1);

  const int64_t odepth_for_shape_check =
      pooling_output_shape<int64_t>(idepth, kD, padD, dD, 1, ceil_mode);
  const int64_t oheight_for_shape_check =
      pooling_output_shape<int64_t>(iheight, kH, padH, dH, 1, ceil_mode);
  const int64_t owidth_for_chape_check =
      pooling_output_shape<int64_t>(iwidth, kW, padW, dW, 1, ceil_mode);

  avg_pool3d_backward_shape_check(
      input,
      gradOutput,
      nblock,
      kD,
      kH,
      kW,
      dD,
      dH,
      dW,
      padD,
      padH,
      padW,
      idepth,
      iheight,
      iwidth,
      odepth,
      oheight,
      owidth,
      "avg_pool3d_backward_out_template()");

  xpu::COMPUTE_ENG real_eng =
      choose_compute_eng(xpu::COMPUTE_ENG::BASIC, gradInput);

  if (xpu::COMPUTE_ENG::ONEDNN == real_eng) {
    // per oneDNN definition, no dilation means dilation ratio is 0
    std::vector<int64_t> dilation_vec = {0, 0, 0};
    if (count_include_pad) {
      ::xpu::oneDNN::pooling_backward<
          ::xpu::oneDNN::alg::pooling_avg_include_padding>(
          gradInput,
          gradOutput,
          input,
          nbatch,
          nblock,
          idepth,
          iheight,
          iwidth,
          odepth,
          oheight,
          owidth,
          stride_vec,
          kernel_vec,
          dilation_vec,
          padding_vec,
          padding_vec);
    } else {
      ::xpu::oneDNN::pooling_backward<
          ::xpu::oneDNN::alg::pooling_avg_exclude_padding>(
          gradInput,
          gradOutput,
          input,
          nbatch,
          nblock,
          idepth,
          iheight,
          iwidth,
          odepth,
          oheight,
          owidth,
          stride_vec,
          kernel_vec,
          dilation_vec,
          padding_vec,
          padding_vec);
    }
  } else {
    // if divisor==0 then we will ignore it
    int64_t divisor = 0;
    if (divisor_override.has_value()) {
      divisor = divisor_override.value();
    }

    Tensor workOutput = const_cast<Tensor&>(gradOutput);
    Tensor workInput = zeros_like(gradInput);
    if (gradOutput.ndimension() == 5) {
      // Collapse batch and feature dimensions.
      workInput = workInput.reshape({nbatch * nblock, idepth, iheight, iwidth});
      workOutput =
          workOutput.reshape({nbatch * nblock, odepth, oheight, owidth});
    }
    if (xpu::dpcpp::detail::canUse32BitIndexMath(workInput)) {
      if (dD == 1 && dH == 1 && dW == 1 && padD == 0 && padH == 0 &&
          padW == 0) {
        IPEX_DISPATCH_FLOATING_TYPES_AND2(
            kHalf,
            kBFloat16,
            input.scalar_type(),
            "avg_pool3d_backward_out_frame_stride_1",
            [&] {
              using accscalar_t = acc_type<scalar_t>;
              int64_t totalZ = idepth * nblock * nbatch;
              int64_t offsetZ = 0;

              accscalar_t divide_factor;
              if (divisor) {
                divide_factor = static_cast<accscalar_t>(divisor);
              } else {
                divide_factor = static_cast<accscalar_t>(kD * kH * kW);
              }

              while (totalZ > 0) {
                avg_pool3d_backward_out_frame_stride1<
                    scalar_t,
                    accscalar_t,
                    int32_t>(
                    workOutput,
                    workInput,
                    kD,
                    kH,
                    kW,
                    1.0f / divide_factor,
                    offsetZ,
                    totalZ);
                totalZ -= 65535;
                offsetZ += 65535;
              }
            });
      } else {
        // for contiguous format and channels_last3d format
        const bool kernelsOverlap = (dD < kD) || (dH < kH) || (dW < kW);

        IPEX_DISPATCH_FLOATING_TYPES_AND2(
            kHalf,
            kBFloat16,
            input.scalar_type(),
            "avg_pool3d_backward_out_frame",
            [&] {
              using accscalar_t = acc_type<scalar_t>;
              int64_t totalZ = odepth * nblock * nbatch;
              int64_t offsetZ = 0;

              while (totalZ > 0) {
                if (kernelsOverlap) {
                  avg_pool3d_backward_out_frame_atomic<
                      scalar_t,
                      accscalar_t,
                      int32_t>(
                      workOutput,
                      workInput,
                      kD,
                      kH,
                      kW,
                      dD,
                      dH,
                      dW,
                      padD,
                      padH,
                      padW,
                      count_include_pad,
                      offsetZ,
                      totalZ,
                      divisor);
                } else {
                  avg_pool3d_backward_out_frame<scalar_t, accscalar_t, int32_t>(
                      workOutput,
                      workInput,
                      kD,
                      kH,
                      kW,
                      dD,
                      dH,
                      dW,
                      padD,
                      padH,
                      padW,
                      count_include_pad,
                      offsetZ,
                      totalZ,
                      divisor);
                }

                totalZ -= 65535;
                offsetZ += 65535;
              }
            });
      }
    } else {
      if (dD == 1 && dH == 1 && dW == 1 && padD == 0 && padH == 0 &&
          padW == 0) {
        IPEX_DISPATCH_FLOATING_TYPES_AND2(
            kHalf,
            kBFloat16,
            input.scalar_type(),
            "avg_pool3d_backward_out_frame_stride_1",
            [&] {
              using accscalar_t = acc_type<scalar_t>;
              int64_t totalZ = idepth * nblock * nbatch;
              int64_t offsetZ = 0;

              accscalar_t divide_factor;
              if (divisor) {
                divide_factor = static_cast<accscalar_t>(divisor);
              } else {
                divide_factor = static_cast<accscalar_t>(kD * kH * kW);
              }

              while (totalZ > 0) {
                avg_pool3d_backward_out_frame_stride1<
                    scalar_t,
                    accscalar_t,
                    int64_t>(
                    workOutput,
                    workInput,
                    kD,
                    kH,
                    kW,
                    1.0f / divide_factor,
                    offsetZ,
                    totalZ);
                totalZ -= 65535;
                offsetZ += 65535;
              }
            });
      } else {
        // for contiguous format and channels_last3d format
        const bool kernelsOverlap = (dD < kD) || (dH < kH) || (dW < kW);

        IPEX_DISPATCH_FLOATING_TYPES_AND2(
            kHalf,
            kBFloat16,
            input.scalar_type(),
            "avg_pool3d_backward_out_frame",
            [&] {
              using accscalar_t = acc_type<scalar_t>;
              int64_t totalZ = odepth * nblock * nbatch;
              int64_t offsetZ = 0;

              while (totalZ > 0) {
                if (kernelsOverlap) {
                  avg_pool3d_backward_out_frame_atomic<
                      scalar_t,
                      accscalar_t,
                      int64_t>(
                      workOutput,
                      workInput,
                      kD,
                      kH,
                      kW,
                      dD,
                      dH,
                      dW,
                      padD,
                      padH,
                      padW,
                      count_include_pad,
                      offsetZ,
                      totalZ,
                      divisor);
                } else {
                  avg_pool3d_backward_out_frame<scalar_t, accscalar_t, int64_t>(
                      workOutput,
                      workInput,
                      kD,
                      kH,
                      kW,
                      dD,
                      dH,
                      dW,
                      padD,
                      padH,
                      padW,
                      count_include_pad,
                      offsetZ,
                      totalZ,
                      divisor);
                }

                totalZ -= 65535;
                offsetZ += 65535;
              }
            });
      }
    }

    gradInput = workInput.resize_as_(gradInput);
  }
  return gradInput;
}
} // namespace impl

Tensor& avg_pool3d_out(
    const Tensor& input,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    bool ceil_mode,
    bool count_include_pad,
    c10::optional<int64_t> divisor_override,
    Tensor& output) {
  impl::avg_pool3d_out_template(
      output,
      input,
      kernel_size,
      stride,
      padding,
      ceil_mode,
      count_include_pad,
      divisor_override);

  return output;
}

Tensor& avg_pool3d_backward_out(
    const Tensor& grad_output_,
    const Tensor& self_,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    bool ceil_mode,
    bool count_include_pad,
    c10::optional<int64_t> divisor_override,
    Tensor& grad_input) {
  Tensor self, grad_output;
  if (self_.ndimension() == 4) {
    // 4D: Input (C, D, H, W),  Output (C, D0, H0, W0)
    // cannot give channels last for 4D tensor from frontend user perspective
    // the 2nd dim is outputDepth, not channel dim
    self = self_.contiguous();
    grad_output = grad_output_.contiguous();
    grad_input.resize_as_(self);
  } else {
    // 5D: Input (N, C, D, H, W),  Output (N, C, D0, H0, W0)
    // smf supports ChannelsLast3D and Contiguous cases.
    auto smf = self_.suggest_memory_format();
    self = self_.contiguous(smf);
    grad_output = grad_output_.contiguous(smf);
    grad_input.resize_as_(self_, smf);
  }

  impl::avg_pool3d_backward_out_template(
      grad_input,
      grad_output,
      self,
      kernel_size,
      stride,
      padding,
      ceil_mode,
      count_include_pad,
      divisor_override);
  return grad_input;
}
} // namespace AtenIpexTypeXPU
} // namespace at