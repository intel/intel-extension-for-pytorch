#include <ATen/ATen.h>
#include <ATen/Config.h>
#include <ATen/NativeFunctions.h>
#include <ATen/core/op_registration/adaption.h>
#include <ATen/core/op_registration/op_registration.h>
#include <torch/custom_class.h>
#include "utils/CustomOperatorRegistration.h"

#include <oneDNN/oneDNN.h>
#include <quantized/QUtils.h>
#include <runtime/Utils.h>
#include "comm/ParamUtils.h"

using namespace torch_ipex::xpu::dpcpp;
using namespace at::native;

namespace at {
namespace AtenIpexTypeQuantizedXPU {

template <typename T, typename FuncType>
struct QAddImplKernelFunctor {
  void operator()(sycl::item<1> item) const {
    auto i = item.get_linear_id();
    o_ptr[i] = func(qa_ptr[i], qb_ptr[i]);
  }
  QAddImplKernelFunctor(T* qa_ptr_, T* qb_ptr_, T* o_ptr_, FuncType func_)
      : qa_ptr(qa_ptr_), qb_ptr(qb_ptr_), o_ptr(o_ptr_), func(func_) {}

 private:
  T* qa_ptr;
  T* qb_ptr;
  T* o_ptr;
  FuncType func;
};

template <typename T>
struct q_add_impl_func {
  T operator()(T a, T b) const {
    float fa = (a - azp) * ascale;
    float fb = (b - bzp) * bscale;
    float fo = fa + fb;
    if (with_relu)
      fo = fo >= 0.f ? fo : 0.f;
    return quantize_val<T>((float)(oscale), zero_point, fo);
  }

  q_add_impl_func(
      float oscale_,
      float ascale_,
      int32_t azp_,
      float bscale_,
      int32_t bzp_,
      bool with_relu_,
      int64_t zero_point_)
      : oscale(oscale_),
        ascale(ascale_),
        azp(azp_),
        bscale(bscale_),
        bzp(bzp_),
        with_relu(with_relu_),
        zero_point(zero_point_) {}

 private:
  float oscale;
  float ascale;
  int32_t azp;
  float bscale;
  int32_t bzp;
  bool with_relu;
  int64_t zero_point;
};

template <typename T>
Tensor q_add_impl(
    Tensor qa,
    Tensor qb,
    double scale,
    int64_t zero_point,
    bool with_relu) {
  auto qa_ = to_plain_if_needed(qa);
  auto qb_ = to_plain_if_needed(qb);

  auto out = at::_empty_affine_quantized(
      qa_.sizes(),
      qa_.scalar_type(),
      c10::nullopt,
      qa_.device(),
      c10::nullopt,
      scale,
      zero_point,
      qa_.suggest_memory_format());

  float oscale = scale;
  float ascale = qa_.q_scale();
  int32_t azp = qa_.q_zero_point();
  float bscale = qb_.q_scale();
  int32_t bzp = qb_.q_zero_point();

  q_add_impl_func<T> func(
      oscale, ascale, azp, bscale, bzp, with_relu, zero_point);

  auto& dpcpp_queue = torch_ipex::xpu::dpcpp::dpcppGetCurrentQueue();
  auto cgf = DPCPP_Q_CGF(cgh) {
    T* qa_ptr = (T*)qa_.data_ptr();
    T* qb_ptr = (T*)qb_.data_ptr();
    T* o_ptr = (T*)out.data_ptr();
    QAddImplKernelFunctor<T, decltype(func)> kfn(qa_ptr, qb_ptr, o_ptr, func);
    cgh.parallel_for<decltype(kfn)>(sycl::range<1>(qa_.numel()), kfn);
  };
  DPCPP_Q_SUBMIT(dpcpp_queue, cgf);

  return out;
}

Tensor q_add(Tensor qa, Tensor qb, double scale, int64_t zero_point) {
  Tensor c;
  if (qa.scalar_type() == kQUInt8)
    c = q_add_impl<uint8_t>(qa, qb, scale, zero_point, /*with_relu=*/false);
  else
    c = q_add_impl<int8_t>(qa, qb, scale, zero_point, /*with_relu=*/false);
  return c;
}

Tensor q_add_relu(Tensor qa, Tensor qb, double scale, int64_t zero_point) {
  Tensor c;
  if (qa.scalar_type() == kQUInt8)
    c = q_add_impl<uint8_t>(qa, qb, scale, zero_point, /*with_relu=*/true);
  else
    c = q_add_impl<int8_t>(qa, qb, scale, zero_point, /*with_relu=*/true);
  return c;
}

TORCH_LIBRARY_IMPL(quantized, QuantizedXPU, m) {
  IPEX_QOP_REGISTER("add", q_add);
  IPEX_QOP_REGISTER("add_relu", q_add_relu);
}

} // namespace AtenIpexTypeQuantizedXPU
} // namespace at
