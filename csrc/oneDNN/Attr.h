#pragma once

#include <ATen/ATen.h>
#include <core/MemoryFormat.h>
#include <core/detail/TensorInfo.h>
#include <oneapi/dnnl/dnnl.hpp>
#include <oneapi/dnnl/dnnl_types.h>
#include <tensor/Context.h>
#include <utils/Macros.h>
#include "Utils.h"

using namespace dnnl;
using namespace xpu::dpcpp;

namespace xpu {
namespace oneDNN {
/* oneDNN quantization usage:
   https://oneapi-src.github.io/oneDNN/dev_guide_attributes_quantization.html#

   src_fp32 = scale_src * (src_int8 - zero_point)
   wei_fp32 = scale_wei * (wei_int8 - zero_point)
   dst_fp32 = scale_dst * (dst_int8 - zero_point)
   fp32 Convolution: dst_fp32 = src_fp32 * wei_fp32
   Int8 Convolution: dst_fp32 = (src_int8 * wei_int8) * (scale_src * scale_wei)
   Int8 Convolution: dst_int8 = 1 / scale_dst * dst_fp32;

   considering bias:
   fp32 Convolution: dst_fp32 = src_fp32 * wei_fp32 + bias
   Int8 Convolution: dst_fp32 = (src_int8 * wei_int8) * (scale_src * scale_wei)
   + bias Int8 Convolution: dst_fp32 = (src_int8 * wei_int8 + bias/(scale_src *
   scale_wei)) * (scale_src * scale_wei) Int8 Convolution: dst_int8 = 1 /
   scale_dst * dst_fp32;
*/

/*
   oneDNN postops usage:
   Currently, oneDNN supports 5 kinds of post ops. More details can be refered
to oneDNN doc.
   https://oneapi-src.github.io/oneDNN/dev_guide_attributes_post_ops.html#doxid-dev-guide-attributes-post-ops-1dev-guide-attributes-post-ops-eltwise

0. without post ops
   dst = Conv(src, wei) + bias;
   dst_int8 = 1/q_scale * dst; q_scale is the op output quantization scale
   fp32 API: Attr attr;
   int8 API: Attr attr(q_scale);

1. append eltwise post op
   dst = elt_scale * Eltwise{conv_scale * [Conv(src, wei) + bias], alpha, beta}
   dst_int8 = 1/q_scale * dst;
   fp32 API:
   Attr attr;
   attr.append_post_eltwise(1.f, conv_scale, 0.f, kind_with_linear)
   attr.append_post_eltwise(elt_scale, alpha, beta, eltwise_algorithm)
   int8 API:
   Attr attr(q_scale);
   attr.append_post_eltwise(1.f, conv_scale, 0.f, kind_with_linear)
   attr.append_post_eltwise(elt_scale, alpha, beta, eltwise_algorithm)

2. append sum post op
   dst = conv_scale * Conv(src, wei) + sum_scale * (dst - zp)
   dst_int8 = 1/q_scale * dst;
   fp32 API:
   Attr attr;
   attr.append_post_eltwise(1.f, conv_scale, 0.f, kind_with_linear)
   attr.append_post_sum(sum_scale)
   int8 API:
   Attr attr(q_scale);
   attr.append_post_eltwise(1.f, conv_scale, 0.f, kind_with_linear)
   attr.append_post_sum(sum_scale)

3. append binary post op
   // TODO:
   dst = Binary[Conv(src, wei)]

4. append prelu post op
   // TODO:
   fusion_dst = prelu(Conv(src, wei), weights[:])

5. append depthwise conv post op
   // TODO:
   fusion_dst = Convdw(Conv1x1(...))
*/

using kind_t = dnnl::primitive::kind;
struct PostOpParam {
  // eltwise post op constructor
  PostOpParam(float scale, float alpha, float beta, algorithm algo, kind_t kind)
      : scale_(scale), alpha_(alpha), beta_(beta), algo_(algo), kind_(kind) {}
  // sum post op constructor
  PostOpParam(float scale, kind_t kind) : scale_(scale), kind_(kind) {}
  // prelu post op constructor
  PostOpParam(int mask, kind_t kind) : mask_(mask), kind_(kind) {}
  // prelu post op constructor

  float scale_;
  float alpha_;
  float beta_;
  algorithm algo_;
  int mask_;
  kind_t kind_;
};

class Attr {
 public:
  Attr() : q_scale_(1.f), q_zero_point_(0) {}
  Attr(float q_scale, int64_t zp = 0) : q_scale_(q_scale), q_zero_point_(zp) {}

  algorithm kind_with_relu = algorithm::eltwise_relu;
  algorithm kind_with_sigmoid = algorithm::eltwise_logistic;
  algorithm kind_with_gelu = algorithm::eltwise_gelu_tanh;
  algorithm kind_with_mish = algorithm::eltwise_mish;
  algorithm kind_with_linear = algorithm::eltwise_linear;
  algorithm kind_with_swish = algorithm::eltwise_swish;

  // append sum post op
  Attr& append_post_sum(
      float sum_scale,
      float sum_q_scale = 1.f,
      int64_t zp = 0) {
    ops_params_.push_back(
        PostOpParam(/*scale_sum*/ sum_scale * sum_q_scale, kind_t::sum));
    return *this;
  }

  // append eltwise post op
  Attr& append_post_eltwise(
      float scale,
      float alpha,
      float beta,
      algorithm algo) {
    ops_params_.push_back(
        PostOpParam(scale, alpha, beta, algo, kind_t::eltwise));
    return *this;
  }

  // append prelu post op
  Attr& append_post_prelu(int mask) {
    ops_params_.push_back(PostOpParam(mask, kind_t::prelu));
    return *this;
  }

  // This function only work for int8
  ScalarType get_dst_dtype() {
    // this function is used to check whether the last post op is relu or not
    // if with relu and alpha<=0 in leakyrelu, the dst should U8 type
    // otherwise, the output should in S8 type
    auto dtype = at::kQInt8;
    auto last_op = ops_params_[-1];
    if (last_op.algo_ == kind_with_relu && last_op.alpha_ <= 0.0) {
      dtype = at::kQUInt8;
    }
    return dtype;
  }

  void extract_post_ops(post_ops& dnnl_post_ops, const Tensor& dst) {
    // this function is used to extract post ops params from the ops_params_
    // and put them into onednn post ops
    for (int i = 0; i < ops_params_.size(); ++i) {
      float scale = ops_params_[i].scale_;
      float alpha = ops_params_[i].alpha_;
      float beta = ops_params_[i].beta_;
      algorithm algo = ops_params_[i].algo_;
      kind_t kind = ops_params_[i].kind_;

      switch (kind) {
        case kind_t::eltwise:
          dnnl_post_ops.append_eltwise(scale, algo, alpha, beta);
          break;
        case kind_t::sum:
          dnnl_post_ops.append_sum(scale);
          break;
        default:
          break;
      }
    }

    // if output is quantized, then append the eltwise linear to adjust the
    // output scale/zero_point
    if (dst.is_quantized()) {
      /*The /2 here is for output_scale collected by observer is different from
quantize requirement from oneDNN. For Observer, the conv_scale (activation scale
in other case) is computed through 2max_v/(qmax - qmin). The max_v is collected
from the tensor to be observerd.
(https://pytorch.org/docs/stable/generated/torch.quantization.observer.MinMaxObserver.html#torch.quantization.observer.MinMaxObserver)
On the other hand, for u8 in oneDNN, the scale for quantization is defined as
max_v/(qmax-qmin). Hence, we need to divide by 2 here.
(https://oneapi-src.github.io/oneDNN/dev_guide_inference_int8.html)*/
      q_scale_ =
          (get_onednn_dtype_include_double(dst) == memory::data_type::u8 &&
           dst.q_zero_point() == 128)
          ? q_scale_ / 2
          : q_scale_;
      dnnl_post_ops.append_eltwise(
          1.0, kind_with_linear, 1.f / q_scale_, q_zero_point_);
    }
  }

  bool with_sum() {
    for (int i = 0; i < ops_params_.size(); ++i) {
      if (ops_params_[i].kind_ == kind_t::sum) {
        return true;
      }
    }
    return false;
  }

#ifdef USE_PRIMITIVE_CACHE
  void to_bytes(bytestring& bytes) {
    xpu::dpcpp::to_bytes(bytes, q_scale_);
    for (int i = 0; i < ops_params_.size(); ++i) {
      xpu::dpcpp::to_bytes(bytes, ops_params_[i].scale_);
      xpu::dpcpp::to_bytes(bytes, ops_params_[i].alpha_);
      xpu::dpcpp::to_bytes(bytes, ops_params_[i].beta_);
      xpu::dpcpp::to_bytes(bytes, ops_params_[i].algo_);
      xpu::dpcpp::to_bytes(bytes, ops_params_[i].kind_);
    }
  }
#endif

  float q_scale_; // the scale used to quantize the fused result from fp32 to
                  // int8, only works for int8 case
  int64_t q_zero_point_;
  std::vector<PostOpParam> ops_params_; // series of post ops
};

} // namespace oneDNN
} // namespace xpu
