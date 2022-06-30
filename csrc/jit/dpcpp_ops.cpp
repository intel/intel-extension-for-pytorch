#include <ATen/ATen.h>
#include <ATen/CPUApplyUtils.h>
#include <ATen/NativeFunctions.h>
#include <ATen/record_function.h>
#include <intrinsic/intrinsic.h>
#include <oneapi/dnnl/dnnl.hpp>

#include <oneDNN/oneDNN.h>

using namespace xpu::oneDNN;

namespace torch {
namespace jit {
namespace xpu {

typedef enum {
  undef = 0,
  with_sum = MatmulAttr::kind_with_sum,
  with_relu = MatmulAttr::kind_with_relu,
  with_gelu = MatmulAttr::kind_with_gelu,
  with_sigmoid = MatmulAttr::kind_with_sigmoid,
  with_mul = MatmulAttr::kind_with_bin_mul,
  with_add = MatmulAttr::kind_with_bin_add,
  with_sub = MatmulAttr::kind_with_bin_sub,
} FusionType;

at::Tensor& conv2d_sum(
    at::Tensor& accumu,
    const at::Tensor& input,
    const at::Tensor& weight,
    const at::Tensor& bias,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::IntArrayRef dilation,
    int64_t groups,
    at::Scalar alpha) {
  RECORD_FUNCTION(
      "conv2d_sum", std::vector<c10::IValue>({input, weight, bias, accumu}));
  at::AtenIpexTypeXPU::convolution_sum(
      input,
      weight,
      bias,
      stride,
      padding,
      dilation,
      false,
      {{0, 0}},
      groups,
      accumu,
      alpha);
  return accumu;
}

at::Tensor& conv2d_sum_relu(
    at::Tensor& accumu,
    const at::Tensor& input,
    const at::Tensor& weight,
    const at::Tensor& bias,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::IntArrayRef dilation,
    int64_t groups,
    at::Scalar alpha) {
  RECORD_FUNCTION(
      "conv2d_sum_relu",
      std::vector<c10::IValue>({input, weight, bias, accumu}));
  at::AtenIpexTypeXPU::convolution_sum_relu(
      input,
      weight,
      bias,
      stride,
      padding,
      dilation,
      false,
      {{0, 0}},
      groups,
      accumu,
      alpha);
  return accumu;
}

at::Tensor q_conv2d_dequantize(
    const at::Tensor& input,
    const c10::intrusive_ptr<ConvPackedParamsBase<2>>& packed_weight,
    double output_scale,
    int64_t output_zero_point) {
  RECORD_FUNCTION(
      "q_conv2d_dequantize", std::vector<c10::IValue>({input, packed_weight}));
  return at::AtenIpexTypeXPU::q_conv2d_dequantize(
      input, packed_weight, output_scale, output_zero_point);
}

at::Tensor softplus_tanh(
    const Tensor& self,
    const Scalar& beta,
    const Scalar& threshold) {
  return at::AtenIpexTypeXPU::softplus_tanh(self, beta, threshold);
}

at::Tensor softplus_tanh_mul(
    const Tensor& self,
    const Scalar& beta,
    const Scalar& threshold,
    const Tensor& mul_input) {
  return at::AtenIpexTypeXPU::softplus_tanh_mul(
      self, beta, threshold, mul_input);
}

at::Tensor q_conv2d_dequantize_softplus_tanh_mul(
    const at::Tensor& input,
    const c10::intrusive_ptr<ConvPackedParamsBase<2>>& packed_weight,
    double output_scale,
    int64_t output_zero_point,
    const Scalar& beta,
    const Scalar& threshold) {
  return at::AtenIpexTypeXPU::q_conv2d_dequantize_softplus_tanh_mul(
      input, packed_weight, output_scale, output_zero_point, beta, threshold);
}

at::Tensor q_conv2d_dequantize_softplus_tanh_mul_quantize(
    const at::Tensor& input,
    const c10::intrusive_ptr<ConvPackedParamsBase<2>>& packed_weight,
    double output_scale,
    int64_t output_zero_point,
    const Scalar& beta,
    const Scalar& threshold,
    double q_scale,
    int64_t q_zpoint,
    at::ScalarType dtype) {
  RECORD_FUNCTION(
      "q_conv2d_dequantize_softplus_tanh_mul_quantize",
      std::vector<c10::IValue>({input}));
  return at::AtenIpexTypeXPU::q_conv2d_dequantize_softplus_tanh_mul_quantize(
      input,
      packed_weight,
      output_scale,
      output_zero_point,
      beta,
      threshold,
      q_scale,
      q_zpoint,
      dtype);
}

at::Tensor q_conv2d_dequantize_softplus_tanh_mul_quantize_add(
    const at::Tensor& input,
    const c10::intrusive_ptr<ConvPackedParamsBase<2>>& packed_weight,
    double output_scale,
    int64_t output_zero_point,
    const Scalar& beta,
    const Scalar& threshold,
    double q_scale,
    int64_t q_zpoint,
    at::ScalarType dtype,
    Tensor qb,
    double add_scale,
    int64_t add_zero_point) {
  RECORD_FUNCTION(
      "q_conv2d_dequantize_softplus_tanh_mul_quantize_add",
      std::vector<c10::IValue>({input}));
  return at::AtenIpexTypeXPU::
      q_conv2d_dequantize_softplus_tanh_mul_quantize_add(
          input,
          packed_weight,
          output_scale,
          output_zero_point,
          beta,
          threshold,
          q_scale,
          q_zpoint,
          dtype,
          qb,
          add_scale,
          add_zero_point);
}

at::Tensor conv2d_relu(
    const at::Tensor& input,
    const at::Tensor& weight,
    const at::Tensor& bias,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::IntArrayRef dilation,
    int64_t groups) {
  RECORD_FUNCTION(
      "conv2d_relu", std::vector<c10::IValue>({input, weight, bias}));
  return at::AtenIpexTypeXPU::convolution_relu(
      input, weight, bias, stride, padding, dilation, false, {{0, 0}}, groups);
}

at::Tensor pad_conv2d(
    const at::Tensor& input,
    at::IntArrayRef pad_nd,
    Scalar value,
    const at::Tensor& weight,
    const at::Tensor& bias,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::IntArrayRef dilation,
    int64_t groups) {
  RECORD_FUNCTION(
      "pad_conv2d", std::vector<c10::IValue>({input, weight, bias}));
  return at::AtenIpexTypeXPU::pad_convolution(
      input,
      pad_nd,
      value,
      weight,
      bias,
      stride,
      padding,
      dilation,
      false,
      {{0, 0}},
      groups);
}

at::Tensor conv2d_sigmoid(
    const at::Tensor& input,
    const at::Tensor& weight,
    const at::Tensor& bias,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::IntArrayRef dilation,
    int64_t groups) {
  RECORD_FUNCTION(
      "conv2d_sigmoid", std::vector<c10::IValue>({input, weight, bias}));
  return at::AtenIpexTypeXPU::convolution_sigmoid(
      input, weight, bias, stride, padding, dilation, false, {{0, 0}}, groups);
}

// r = alpha x m1 x m2 + beta1 x accumul1 + beta2 x accumul2
// support 0/1/2 added tensor after matrix mul
// current we only support 1 x (m1 x m2 + accumu*) + beta* x accumul* case for 2
// added tensor
at::Tensor matmul_fusion_variants(
    at::Tensor& accumul1,
    at::Tensor& accumul2,
    const at::Tensor& tensor1,
    const at::Tensor& tensor2,
    float alpha,
    float beta1,
    float beta2,
    bool trans,
    int fusion_type = 0) {
  RECORD_FUNCTION(
      "matmul_fusion_variants", std::vector<c10::IValue>({tensor1, tensor2}));
  auto dim_tensor1 = tensor1.dim();
  auto dim_tensor2 = tensor2.dim();

  // TODO: matmul case is complicated
  // supported fusion cases,
  // 1. 2D x 2D
  // 2. 3D x 3D
  at::Tensor result;
  result = at::empty({0}, tensor1.options());

  if (dim_tensor1 == 2 && dim_tensor2 == 2) {
    // no bias no post sum
    // alpha x m1 x m2
    if (beta1 == 0.f && beta2 == 0.f) {
      TORCH_CHECK(
          !accumul1.defined(),
          "we cannot support add tensor with scalar multiplier for tensor is 0");
      TORCH_CHECK(
          !accumul2.defined(),
          "we cannot support add tensor with scalar multiplier for tensor is 0");
      at::AtenIpexTypeXPU::matmul(
          result,
          tensor1,
          tensor2,
          at::Tensor(),
          at::Tensor(),
          0.f,
          alpha,
          trans,
          (int)(fusion_type));
      // both bias and post sum
      // m1 x m2 + bias + beta x accumul
    } else if ((beta1 != 0.f) && (beta2 != 0.f)) {
      TORCH_CHECK(
          alpha == 1.f,
          "alpha must be 1 if both bias add and post sum supported");
      TORCH_CHECK(
          beta1 == 1.f || beta2 == 1.f,
          "at least one scalar multiplier for tensor is 1");
      TORCH_CHECK(
          accumul1.defined() && accumul2.defined(),
          "accumulate tensor must be defined");
      if (beta1 == 1.f) {
        at::AtenIpexTypeXPU::matmul(
            result,
            tensor1,
            tensor2,
            accumul1,
            accumul2,
            beta2,
            alpha,
            trans,
            (int)(fusion_type | with_sum));
      } else {
        at::AtenIpexTypeXPU::matmul(
            result,
            tensor1,
            tensor2,
            accumul2,
            accumul1,
            beta1,
            alpha,
            trans,
            (int)(fusion_type | with_sum));
      }
      // only have one tensor for adding, we decide bias or post sum
      // m1 x m2 + bias or alpha x m1 x m2 + beta x accumul
    } else {
      if (beta1 == 0.f) {
        TORCH_CHECK(
            !accumul1.defined(),
            "we cannot support add tensor with scalar multiplier for tensor is 0");
        TORCH_CHECK(accumul2.defined(), "accumulate tensor must be defined");
        if (alpha == 1.f && beta2 == 1.f) {
          at::AtenIpexTypeXPU::matmul(
              result,
              tensor1,
              tensor2,
              accumul2,
              at::Tensor(),
              0.f,
              alpha,
              trans,
              (int)(fusion_type));
        } else {
          at::AtenIpexTypeXPU::matmul(
              result,
              tensor1,
              tensor2,
              at::Tensor(),
              accumul2,
              beta2,
              alpha,
              trans,
              (int)(fusion_type | with_sum));
        }
      } else if (beta2 == 0.f) {
        TORCH_CHECK(
            !accumul2.defined(),
            "we cannot support add tensor with scalar multiplier for tensor is 0");
        TORCH_CHECK(accumul1.defined(), "accumulate tensor must be defined");
        if (alpha == 1.f && beta1 == 1.f) {
          at::AtenIpexTypeXPU::matmul(
              result,
              tensor1,
              tensor2,
              accumul1,
              at::Tensor(),
              0.f,
              alpha,
              trans,
              (int)(fusion_type));
        } else {
          at::AtenIpexTypeXPU::matmul(
              result,
              tensor1,
              tensor2,
              at::Tensor(),
              accumul1,
              beta1,
              alpha,
              trans,
              (int)(fusion_type | with_sum));
        }
      } else {
        AT_ERROR("at least one scala multiplier is not 0!");
      }
    }
    return result;
  } else if (dim_tensor1 >= 3 && (dim_tensor2 == 1 || dim_tensor2 == 2)) {
    // collaps a,b,c to axb,c for m1
    // FIXME: no m2 to collaps so far
    std::vector<int64_t> tensor1_shape, r_shape;

    for (int i = 0; i < tensor1.sizes().size() - 1; i++) {
      tensor1_shape.push_back(tensor1.sizes()[i]);
      r_shape.push_back(tensor1.sizes()[i]);
    }
    tensor1_shape.push_back(tensor1.sizes()[tensor1.sizes().size() - 1]);
    r_shape.push_back(trans ? tensor2.sizes()[1] : tensor2.sizes()[0]);

    std::vector<int64_t> sizes = tensor1.sizes().vec();
    std::vector<int64_t> strides = tensor1.strides().vec();
    at::collapse_dims(
        sizes.data(), strides.data(), tensor1.dim(), tensor1.dim() - 1);
    tensor1.resize_({sizes.data()[0], sizes.data()[1]});

    // no bias no post sum
    // alpha x m1 x m2
    if (beta1 == 0.f && beta2 == 0.f) {
      TORCH_CHECK(
          !accumul1.defined(),
          "we cannot support add tensor with scalar multiplier for tensor is 0");
      TORCH_CHECK(
          !accumul2.defined(),
          "we cannot support add tensor with scalar multiplier for tensor is 0");
      at::AtenIpexTypeXPU::matmul(
          result,
          tensor1,
          tensor2,
          at::Tensor(),
          at::Tensor(),
          0.f,
          alpha,
          trans,
          (int)(fusion_type));
      // both bias and post sum
      // m1 x m2 + bias + beta x accumul
    } else if ((beta1 != 0.f) && (beta2 != 0.f)) {
      TORCH_CHECK(
          alpha == 1.f,
          "alpha must be 1 if both bias add and post sum supported");
      TORCH_CHECK(
          beta1 == 1.f || beta2 == 1.f,
          "at least one scalar multiplier for tensor is 1");
      TORCH_CHECK(
          accumul1.defined() && accumul2.defined(),
          "accumulate tensor must be defined");
      if (beta1 == 1.f) {
        if (accumul2.dim() == 3) {
          TORCH_CHECK(
              accumul2.sizes() == r_shape, "wrong shape for accumulate");
          accumul2.resize_({tensor1.size(0), r_shape[2]});
          at::AtenIpexTypeXPU::matmul(
              result,
              tensor1,
              tensor2,
              accumul1,
              accumul2,
              beta2,
              alpha,
              trans,
              (int)(fusion_type | with_sum));
        } else {
          at::AtenIpexTypeXPU::matmul(
              result,
              tensor1,
              tensor2,
              accumul1,
              at::Tensor(),
              0.f,
              alpha,
              trans,
              (int)(fusion_type));
          result.resize_(r_shape).add_(accumul2, beta2);
        }
      } else {
        if (accumul1.dim() == 3) {
          TORCH_CHECK(
              accumul1.sizes() == r_shape, "wrong shape for accumulate");
          accumul1.resize_({tensor1.size(0), r_shape[2]});
          at::AtenIpexTypeXPU::matmul(
              result,
              tensor1,
              tensor2,
              accumul2,
              accumul1,
              beta1,
              alpha,
              trans,
              (int)(fusion_type | with_sum));
        } else {
          at::AtenIpexTypeXPU::matmul(
              result,
              tensor1,
              tensor2,
              accumul2,
              at::Tensor(),
              0.f,
              alpha,
              trans,
              (int)(fusion_type));
          result.resize_(r_shape).add_(accumul1, beta1);
        }
      }
      // only have one tensor for adding, we decide bias or post sum
      // m1 x m2 + bias or alpha x m1 x m2 + beta x accumul
    } else {
      if (beta1 == 0.f) {
        TORCH_CHECK(
            !accumul1.defined(),
            "we cannot support add tensor with scalar multiplier for tensor is 0");
        TORCH_CHECK(accumul2.defined(), "accumulate tensor must be defined");
        if (alpha == 1.f && beta2 == 1.f) {
          at::AtenIpexTypeXPU::matmul(
              result,
              tensor1,
              tensor2,
              accumul2,
              at::Tensor(),
              0.f,
              alpha,
              trans,
              (int)(fusion_type));
        } else {
          at::AtenIpexTypeXPU::matmul(
              result,
              tensor1,
              tensor2,
              at::Tensor(),
              accumul2,
              beta2,
              alpha,
              trans,
              (int)(fusion_type | with_sum));
        }
      } else if (beta2 == 0.f) {
        TORCH_CHECK(
            !accumul2.defined(),
            "we cannot support add tensor with scalar multiplier for tensor is 0");
        TORCH_CHECK(accumul1.defined(), "accumulate tensor must be defined");
        if (alpha == 1.f && beta1 == 1.f) {
          at::AtenIpexTypeXPU::matmul(
              result,
              tensor1,
              tensor2,
              accumul1,
              at::Tensor(),
              0.f,
              alpha,
              trans,
              (int)(fusion_type));
        } else {
          at::AtenIpexTypeXPU::matmul(
              result,
              tensor1,
              tensor2,
              at::Tensor(),
              accumul1,
              beta1,
              alpha,
              trans,
              (int)(fusion_type | with_sum));
        }
      } else {
        AT_ERROR("at least one scala multiplier is not 0!");
      }
    }

    if (r_shape.size()) {
      tensor1.resize_(tensor1_shape);
      result.resize_(r_shape);
    }
    return result;

  } else if (
      (dim_tensor1 >= 1 && dim_tensor2 >= 1) &&
      (dim_tensor1 >= 3 || dim_tensor2 >= 3)) {
    // We are multiplying b1 x n x m1 by x2 x m2 x p (where b1 can be a list);
    // we track m1 vs m2 separately even though they must match for nicer error
    // messages
    TORCH_CHECK(
        !((beta1 != 0.f) && (beta2 != 0.f)),
        "for 3D matmul, we only support one accumulate tensor");
    int64_t n = dim_tensor1 > 1 ? tensor1.size(-2) : 1;
    int64_t m1 = tensor1.size(-1);
    at::IntArrayRef batch_tensor1(
        tensor1.sizes().data(), std::max<int64_t>(dim_tensor1 - 2, 0));

    // inverse dims in non-transpose case
    int64_t m2 = dim_tensor2 > 1 ? tensor2.size(-1) : 1;
    int64_t p = tensor2.size(-2);

    at::IntArrayRef batch_tensor2(
        tensor2.sizes().data(), std::max<int64_t>(dim_tensor2 - 2, 0));

    // expand the batch portion (i.e. cut off matrix dimensions and expand rest)
    std::vector<int64_t> expand_batch_portion =
        at::infer_size(batch_tensor1, batch_tensor2);

    std::vector<int64_t> tensor1_expand_size(expand_batch_portion);
    tensor1_expand_size.insert(tensor1_expand_size.end(), {n, m1});

    std::vector<int64_t> tensor2_expand_size(expand_batch_portion);
    if (!trans)
      tensor2_expand_size.insert(tensor2_expand_size.end(), {p, m2});
    else
      tensor2_expand_size.insert(tensor2_expand_size.end(), {m2, p});
    int expand_batch_product = std::accumulate(
        expand_batch_portion.begin(),
        expand_batch_portion.end(),
        1,
        std::multiplies<int64_t>());

    std::vector<int64_t> tensor1_bmm_view({expand_batch_product});
    tensor1_bmm_view.insert(tensor1_bmm_view.end(), {n, m1});

    std::vector<int64_t> tensor2_bmm_view({expand_batch_product});
    if (!trans)
      tensor2_bmm_view.insert(tensor2_bmm_view.end(), {p, m2});
    else
      tensor2_bmm_view.insert(tensor2_bmm_view.end(), {m2, p});
    // flatten expanded batches
    at::Tensor tensor1_expanded =
        tensor1.expand(tensor1_expand_size).contiguous().view(tensor1_bmm_view);
    at::Tensor tensor2_expanded =
        tensor2.expand(tensor2_expand_size).contiguous().view(tensor2_bmm_view);

    // reshape batches back into result
    std::vector<int64_t> output_shape(expand_batch_portion);
    if (dim_tensor1 > 1) {
      output_shape.push_back(n);
    }
    if (dim_tensor2 > 1) {
      output_shape.push_back(p);
    }

    at::Tensor accumu, output, self;
    float beta;
    if (beta1 != 0.f) {
      accumu = accumul1;
      beta = beta1;
    } else {
      // 1, beta1 == 0, beta2 != 0
      // 2, beta1 == 0, beta2 ==0
      accumu = accumul2;
      beta = beta2;
    }

    TORCH_CHECK(tensor1_expanded.dim() == 3, "expected 3D tensor");
    TORCH_CHECK(tensor2_expanded.dim() == 3, "expected 3D tensor");

    if (accumu.defined() && accumu.sizes().vec() == output_shape) {
      if (alpha != 1.f || beta != 1.f) {
        at::AtenIpexTypeXPU::matmul(
            accumu,
            tensor1_expanded,
            tensor2_expanded,
            at::Tensor(),
            accumu,
            beta,
            alpha,
            trans,
            (int)(fusion_type | with_sum));
      } else {
        at::AtenIpexTypeXPU::matmul(
            accumu,
            tensor1_expanded,
            tensor2_expanded,
            accumu,
            at::Tensor(),
            beta,
            alpha,
            trans,
            (int)(fusion_type));
      }
      output = at::_unsafe_view(accumu, output_shape);
    } else {
      self = at::empty({0}, tensor1_expanded.options());
      at::AtenIpexTypeXPU::matmul(
          self,
          tensor1_expanded,
          tensor2_expanded,
          at::Tensor(),
          at::Tensor(),
          0.f,
          alpha,
          trans,
          (int)(fusion_type));
      output = at::_unsafe_view(self, output_shape);
      if (accumu.defined() && beta != 0.f)
        output = at::add(output, accumu, beta);
    }
    return output;
  } else {
    // fallback
    at::Tensor r1;
    if (trans)
      r1 = at::mul(at::native::matmul(tensor1, tensor2), alpha);
    else
      r1 = at::mul(
          at::native::matmul(tensor1, tensor2.transpose(-1, -2)), alpha);
    auto r2 = at::mul(accumul1, beta1);
    auto r3 = at::mul(accumul2, beta2);
    return r1 + r2 + r3;
  }
}

at::Tensor matmul_fusion_variants_gelu(
    at::Tensor& accumul1,
    at::Tensor& accumul2,
    const at::Tensor& tensor1,
    const at::Tensor& tensor2,
    float alpha,
    float beta1,
    float beta2,
    bool trans) {
  return matmul_fusion_variants(
      accumul1,
      accumul2,
      tensor1,
      tensor2,
      alpha,
      beta1,
      beta2,
      trans,
      int(FusionType::with_gelu));
}

at::Tensor matmul_fusion_variants_dropout(
    at::Tensor& accumul1,
    at::Tensor& accumul2,
    const at::Tensor& tensor1,
    const at::Tensor& tensor2,
    float alpha,
    float beta1,
    float beta2,
    bool trans,
    double p,
    bool train,
    bool inplace) {
  if (!train) {
    return matmul_fusion_variants(
        accumul1, accumul2, tensor1, tensor2, alpha, beta1, beta2, trans);
  } else {
    auto res = matmul_fusion_variants(
        accumul1, accumul2, tensor1, tensor2, alpha, beta1, beta2, trans);
    if (inplace)
      return at::dropout_(res, p, train);
    else
      return at::dropout(res, p, train);
  }
}

at::Tensor mul_add(
    const at::Tensor& self,
    const at::Tensor& other,
    const at::Tensor& accumu,
    at::Scalar alpha) {
  RECORD_FUNCTION("mul_add", std::vector<c10::IValue>({self, other, accumu}));
  return at::AtenIpexTypeXPU::mul_add(self, other, accumu, alpha);
}

at::Tensor dequant_pixelshuffle(
    const at::Tensor& self,
    int64_t upscale_factor) {
  return at::empty_like(self);
}

at::Tensor dequant_pixelshuffle_quant(
    const at::Tensor& self,
    int64_t upscale_factor,
    double scale,
    int64_t zero_pad,
    at::ScalarType dtype) {
  return at::pixel_shuffle(self, upscale_factor);
}

at::Tensor q_conv2d_sum_relu(
    at::Tensor& accumu,
    const at::Tensor& input,
    const c10::intrusive_ptr<ConvPackedParamsBase<2>>& packed_weight,
    double conv_scale,
    int64_t conv_zpoint,
    double sum_scale,
    int64_t sum_zpoint) {
  RECORD_FUNCTION(
      "q_conv2d_sum_relu", std::vector<c10::IValue>({input, packed_weight}));
  return at::AtenIpexTypeXPU::q_conv2d_sum_relu(
      accumu,
      input,
      packed_weight,
      conv_scale,
      conv_zpoint,
      sum_scale,
      sum_zpoint);
}

at::Tensor q_conv2d_sigmoid(
    const at::Tensor& input,
    const c10::intrusive_ptr<ConvPackedParamsBase<2>>& packed_weight,
    double output_scale,
    int64_t output_zpoint) {
  RECORD_FUNCTION(
      "q_conv2d_sigmoid", std::vector<c10::IValue>({input, packed_weight}));
  return at::AtenIpexTypeXPU::q_conv2d_sigmoid(
      input, packed_weight, output_scale, output_zpoint);
}

at::Tensor q_conv2d_leaky_relu(
    const Tensor& input,
    const c10::intrusive_ptr<ConvPackedParamsBase<2>>& packed_weight,
    double output_scale,
    int64_t output_zpoint,
    Scalar negative_slope) {
  RECORD_FUNCTION(
      "q_conv2d_leaky_relu", std::vector<c10::IValue>({input, packed_weight}));
  return at::AtenIpexTypeXPU::q_conv2d_leaky_relu(
      input, packed_weight, output_scale, output_zpoint, negative_slope);
}

at::Tensor batch_norm(
    const at::Tensor& input,
    const at::Tensor& weight,
    const at::Tensor& bias,
    const at::Tensor& running_mean,
    const at::Tensor& running_var,
    bool train,
    double momentum,
    double eps,
    bool use_dnn) {
  return at::empty_like(input);
}

at::Tensor fold_weight(
    const at::Tensor& weight,
    const at::Tensor& bn_weight,
    const at::Tensor& running_var,
    float eps) {
  return at::empty_like(weight);
}

at::Tensor fold_bias(
    const at::Tensor& weight,
    const at::Tensor& bias,
    const at::Tensor& bn_weight,
    const at::Tensor& bn_bias,
    const at::Tensor& running_mean,
    const at::Tensor& running_var,
    float eps) {
  return at::empty_like(bias);
}

at::Tensor reorder(
    const at::Tensor& input,
    dnnl::memory::format_tag from,
    dnnl::memory::format_tag to,
    int64_t groups) {
  return at::empty_like(input);
}

at::Tensor trans_addmm(
    const at::Tensor& weight,
    const at::Tensor& bias,
    const at::Tensor& input,
    at::Scalar beta,
    at::Scalar alpha) {
  return at::AtenIpexTypeXPU::trans_addmm(bias, input, weight, beta, alpha);
}

at::Tensor trans_addmm_relu(
    const at::Tensor& weight,
    const at::Tensor& bias,
    const at::Tensor& input,
    at::Scalar beta,
    at::Scalar alpha) {
  return at::AtenIpexTypeXPU::trans_addmm_relu(
      input, weight, bias, beta, alpha);
}

at::Tensor trans_addmm_sigmoid(
    const at::Tensor& weight,
    const at::Tensor& bias,
    const at::Tensor& input,
    at::Scalar beta,
    at::Scalar alpha) {
  return at::AtenIpexTypeXPU::trans_addmm_sigmoid(
      input, weight, bias, beta, alpha);
}

at::Tensor trans_addmm_dropout(
    const at::Tensor& weight,
    const at::Tensor& bias,
    const at::Tensor& input,
    at::Scalar beta,
    at::Scalar alpha,
    double p,
    bool train,
    bool inplace) {
  if (!train) {
    return at::AtenIpexTypeXPU::trans_addmm(bias, input, weight, beta, alpha);
  } else {
    auto res =
        at::AtenIpexTypeXPU::trans_addmm(bias, input, weight, beta, alpha);
    if (inplace)
      return at::dropout_(res, p, train);
    else
      return at::dropout(res, p, train);
  }
}

at::Tensor fusion_amdd(
    at::Tensor& p,
    at::Tensor& d_p,
    at::Tensor& buf,
    float weight_decay,
    float momentum,
    float dampening,
    float lr) {
  return at::AtenIpexTypeXPU::fusion_amdd(
      p, d_p, buf, weight_decay, momentum, dampening, lr);
}

} // namespace xpu
} // namespace jit
} // namespace torch
