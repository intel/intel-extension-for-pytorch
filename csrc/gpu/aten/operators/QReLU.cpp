#include <ATen/ATen.h>
#include <ATen/Config.h>
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#include <ATen/core/op_registration/op_registration.h>
#include <c10/core/CompileTimeFunctionPointer.h>
#include <c10/core/DeviceType.h>
#include <c10/util/Optional.h>
#include <torch/library.h>

#include "comm/Numerics.h"

#include <core/detail/ListUtils.h>
#include <runtime/Utils.h>
#include <functional>
#include "Loops.h"
#include "comm/ATDispatch.h"
#include "comm/RegistrationDeclarations.h"

using namespace xpu::dpcpp;
using namespace at::native;

namespace at {
namespace AtenIpexTypeQuantizedXPU {
namespace impl {

template <typename scalar_t_in, typename scalar_t_out>
void q_relu_xpu_kernel(
    scalar_t_in* in_ptr,
    scalar_t_out* out_ptr,
    int64_t num_ele,
    int64_t zero_point,
    bool s8tou8) {
  auto& dpcpp_queue = dpcppGetCurrentQueue();
  int64_t group_size = dpcppMaxWorkGroupSize(dpcppGetDeviceIdOfCurrentQueue());
  int64_t num_groups = CeilDiv(num_ele, group_size);
  float dnn_factor = s8tou8 ? 2.f : 1.f;

  auto cgf = DPCPP_Q_CGF(cgh) {
    cgh.parallel_for(
        sycl::nd_range<1>(
            sycl::range<1>(num_groups * group_size),
            sycl::range<1>(group_size)),
        [=](sycl::nd_item<1> item) {
          auto id = item.get_global_linear_id();
          if (id < num_ele) {
            out_ptr[id] = in_ptr[id] > zero_point
                ? static_cast<scalar_t_out>(
                      static_cast<float>(in_ptr[id]) * dnn_factor)
                : static_cast<scalar_t_out>(zero_point);
          }
        });
  };
  DPCPP_Q_SUBMIT(dpcpp_queue, cgf);
}
} // namespace impl

void q_relu_xpu(const Tensor& qx, Tensor& qy) {
  IPEX_DISPATCH_QINT_TYPES(qx.scalar_type(), "qrelu", [&]() {
    int64_t torch_zp = 128;
    qy = at::_empty_affine_quantized(
        qx.sizes(),
        qx.options().device(kXPU).dtype(kQUInt8),
        qx.q_scale(),
        128);
    bool s8tou8 = (qx.scalar_type() == kQInt8) ? true : false;
    int64_t num_ele = xpu::dpcpp::detail::prod_intlist(qx.sizes());
    int64_t dnn_zp = static_cast<int64_t>(0);
    // This is a workaroud for oneDNN symmetric INT8, will remove it after
    // oneDNN Asymmetric INT8 is ready.

    impl::q_relu_xpu_kernel(
        reinterpret_cast<underlying_t*>(qx.data_ptr<scalar_t>()),
        reinterpret_cast<uint8_t*>(qy.data_ptr<c10::quint8>()),
        num_ele,
        dnn_zp,
        s8tou8);
  });
}

Tensor relu(const Tensor& qx) {
  Tensor qy;
  q_relu_xpu(qx, qy);
  return qy;
}

} // namespace AtenIpexTypeQuantizedXPU
} // namespace at
