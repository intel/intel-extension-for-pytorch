#include <ATen/ATen.h>
#include <ATen/Config.h>
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#include <ATen/core/op_registration/op_registration.h>
#include <c10/core/CompileTimeFunctionPointer.h>
#include <c10/core/DeviceType.h>
#include <c10/util/Optional.h>
#include <torch/library.h>

#include <runtime/Utils.h>
#include <functional>
#include "Loops.h"
#include "comm/ATDispatch.h"
#include "comm/RegistrationDeclarations.h"

using namespace xpu::dpcpp;
using namespace at::native;

namespace at {
namespace AtenIpexTypeQuantizedXPU {

void qsigmoid_kernel(const Tensor& qx, Tensor& qy) {
  IPEX_DISPATCH_QINT_TYPES(qx.scalar_type(), "qsigmoid_xpu", [&]() {
    float output_scale = 0.00390625; // 1.0 / 2^8
    // Algin with PyTorch symmetric quantization. The real dtype is a QInt8
    // See Note: [Opaque u8 tensor]
    int64_t output_zero_point = 0;
    // with zp = 1/(2^8-1)  at the end of this operator to maximize the
    // accuracy, which is diffenent from pytorch cpu implement(zp=1/2^8).

    if (SCALAR_TYPE == at::kQInt32) {
      output_scale = 2.3283064365386963e-10; // 1.0 / 2^32
    } else if (SCALAR_TYPE == at::kQInt8) {
      output_zero_point = -128;
    }
    auto x = at::dequantize(qx);
    auto y = at::sigmoid(x);
    qy = at::quantize_per_tensor(
        y, output_scale, output_zero_point, SCALAR_TYPE);
  });
};

Tensor sigmoid(const Tensor& qx) {
  Tensor qy;
  qsigmoid_kernel(qx, qy);
  return qy;
}

Tensor& sigmoid_(Tensor& qx) {
  qsigmoid_kernel(qx, qx);
  return qx;
}

Tensor& sigmoid_out(const Tensor& qx, Tensor& qy) {
  qsigmoid_kernel(qx, qy);
  return qy;
}

} // namespace AtenIpexTypeQuantizedXPU
} // namespace at
