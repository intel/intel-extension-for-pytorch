#include <ATen/ATen.h>
#include <ATen/Config.h>
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#include <ATen/core/op_registration/op_registration.h>
#include <c10/core/CompileTimeFunctionPointer.h>
#include <c10/core/DeviceType.h>
#include <c10/util/Optional.h>
#include <torch/library.h>

#include <oneDNN/oneDNN.h>
#include "comm/Numerics.h"

#include <core/detail/ListUtils.h>
#include <quantized/QUtils.h>
#include <runtime/Utils.h>
#include <tensor/OpaqueTensorFactories.h>
#include <functional>
#include "Loops.h"
#include "comm/ATDispatch.h"
#include "comm/RegistrationDeclarations.h"

using namespace torch_ipex::xpu::dpcpp;
using namespace at::native;

namespace at {
namespace AtenIpexTypeQuantizedXPU {
namespace impl {

template <typename scalar_t>
struct QReLUXpuKernelFunctor {
  void operator()(sycl::nd_item<1> item) const {
    auto id = item.get_global_linear_id();
    if (id < num_ele) {
      out_ptr[id] = in_ptr[id] > zero_point ? in_ptr[id] : zero_point;
    }
  }
  QReLUXpuKernelFunctor(
      scalar_t* in_ptr_,
      scalar_t* out_ptr_,
      int64_t num_ele_,
      int64_t zero_point_)
      : in_ptr(in_ptr_),
        out_ptr(out_ptr_),
        num_ele(num_ele_),
        zero_point(zero_point_) {}

 private:
  scalar_t* in_ptr;
  scalar_t* out_ptr;
  int64_t num_ele;
  int64_t zero_point;
};

template <typename scalar_t>
void q_relu_xpu_kernel(
    scalar_t* in_ptr,
    scalar_t* out_ptr,
    int64_t num_ele,
    int64_t zero_point) {
  auto& dpcpp_queue = dpcppGetCurrentQueue();
  int64_t group_size = dpcppMaxWorkGroupSize(dpcppGetDeviceIdOfCurrentQueue());
  int64_t num_groups = CeilDiv(num_ele, group_size);

  auto cgf = DPCPP_Q_CGF(cgh) {
    QReLUXpuKernelFunctor<scalar_t> kfn(in_ptr, out_ptr, num_ele, zero_point);
    cgh.parallel_for<decltype(kfn)>(
        sycl::nd_range<1>(
            sycl::range<1>(num_groups * group_size),
            sycl::range<1>(group_size)),
        kfn);
  };
  DPCPP_Q_SUBMIT(dpcpp_queue, cgf);
}

template <typename scalar_t>
struct QLeakyReLUXpuKernelFunctor {
  void operator()(sycl::nd_item<1> item) const {
    auto id = item.get_global_linear_id();
    if (id < num_ele) {
      float dequant_val = (in_ptr[id] - in_zp) * in_sc;
      dequant_val = dequant_val > 0.f ? dequant_val : dequant_val * neg_val;
      out_ptr[id] = quantize_val<scalar_t>(out_sc, out_zp, dequant_val);
    }
  }
  QLeakyReLUXpuKernelFunctor(
      scalar_t* in_ptr_,
      scalar_t* out_ptr_,
      int64_t num_ele_,
      float in_sc_,
      int32_t in_zp_,
      float out_sc_,
      int32_t out_zp_,
      float neg_val_)
      : in_ptr(in_ptr_),
        out_ptr(out_ptr_),
        num_ele(num_ele_),
        in_sc(in_sc_),
        in_zp(in_zp_),
        out_sc(out_sc_),
        out_zp(out_zp_),
        neg_val(neg_val_) {}

 private:
  scalar_t* in_ptr;
  scalar_t* out_ptr;
  int64_t num_ele;
  float in_sc;
  int32_t in_zp;
  float out_sc;
  int32_t out_zp;
  float neg_val;
};

template <typename scalar_t>
void q_leaky_relu_xpu_kernel(
    scalar_t* in_ptr,
    scalar_t* out_ptr,
    int64_t num_ele,
    float in_sc,
    int32_t in_zp,
    float out_sc,
    int32_t out_zp,
    float neg_val) {
  auto& dpcpp_queue = dpcppGetCurrentQueue();
  int64_t group_size = dpcppMaxWorkGroupSize(dpcppGetDeviceIdOfCurrentQueue());
  int64_t num_groups = CeilDiv(num_ele, group_size);

  auto cgf = DPCPP_Q_CGF(cgh) {
    QLeakyReLUXpuKernelFunctor<scalar_t> kfn(
        in_ptr, out_ptr, num_ele, in_sc, in_zp, out_sc, out_zp, neg_val);
    cgh.parallel_for<decltype(kfn)>(
        sycl::nd_range<1>(
            sycl::range<1>(num_groups * group_size),
            sycl::range<1>(group_size)),
        kfn);
  };
  DPCPP_Q_SUBMIT(dpcpp_queue, cgf);
}

} // namespace impl

void q_relu_xpu(const Tensor& qx, Tensor& qy) {
  IPEX_DISPATCH_QINT_TYPES(qx.scalar_type(), "qrelu", [&]() {
    qy = at::_empty_affine_quantized(
        qx.sizes(),
        qx.options().dtype(toQIntType(qx.scalar_type())),
        qx.q_scale(),
        qx.q_zero_point());
    auto qx_plain = AtenIpexTypeXPU::to_plain_if_needed(qx);
    int64_t num_ele = torch_ipex::xpu::dpcpp::detail::prod_intlist(qx.sizes());
    int64_t zp = static_cast<int64_t>(qx.q_zero_point());

    impl::q_relu_xpu_kernel(
        reinterpret_cast<underlying_t*>(qx_plain.data_ptr()),
        reinterpret_cast<underlying_t*>(qy.data_ptr()),
        num_ele,
        zp);
  });
}

Tensor relu(const Tensor& qx) {
  Tensor qy;
  q_relu_xpu(qx, qy);
  return qy;
}

Tensor& q_leaky_relu(Tensor& out, const Tensor& self, Scalar negative_slope) {
  if (self.q_zero_point() == 0) {
    float alpha = negative_slope.to<float>();
    torch_ipex::xpu::oneDNN::eltwise<dnnl::algorithm::eltwise_relu>(
        out, self, alpha, 0.0f);
    return out;
  } else {
    IPEX_DISPATCH_QINT_TYPES(self.scalar_type(), "q_leakyrelu", [&]() {
      int64_t num_ele =
          torch_ipex::xpu::dpcpp::detail::prod_intlist(self.sizes());
      impl::q_leaky_relu_xpu_kernel<underlying_t>(
          reinterpret_cast<underlying_t*>(self.data_ptr()),
          reinterpret_cast<underlying_t*>(out.data_ptr()),
          num_ele,
          self.q_scale(),
          self.q_zero_point(),
          out.q_scale(),
          out.q_zero_point(),
          negative_slope.to<float>());
    });
  }
  return out;
}

Tensor& leaky_relu_(Tensor& self, const Scalar& negative_slope) {
  return q_leaky_relu(self, self, negative_slope);
}

Tensor leaky_relu(const Tensor& self, const Scalar& negative_slope) {
  Tensor out = at::_empty_affine_quantized(
      self.sizes(),
      self.options().dtype(toQIntType(self.scalar_type())),
      self.q_scale(),
      self.q_zero_point());
  auto result = q_leaky_relu(out, self, negative_slope);
  return result;
}
} // namespace AtenIpexTypeQuantizedXPU
} // namespace at
