#include <ATen/native/quantized/PackedParams.h>
#include <torch/all.h>

#include "Eltwise.h"
#include "Linear.h"
#include "WeightPack.h"
#include "autocast/autocast_mode.h"
#include "ideep/IDeepConversions.h"

namespace torch_ipex {
namespace cpu {

enum EltwiseType { NotFused = 0, ReLU = 1, Sigmoid = 2 };
/**
 * Linear inplace version with oneDNN kernel.
 * Inplace version will be used when user provides output tensor. eg: Linear+Add
 *fusion.
 *
 *
 *@param self Activatin input for Linear
 *@param weight Weight for Linear
 *@param bias Bias for Linear
 *@param output Output tensor provided by user
 *@param attr Attribute for oneDNN primitive.
 */
void linear_kernel_output(
    const at::Tensor& self,
    const ideep::tensor& mkldnn_weight,
    const at::Tensor& bias,
    at::Tensor& output,
    const ideep::attr_t& attr,
    const std::vector<ideep::tensor>& post_op_src) {
  auto self_ = self.is_contiguous() ? self : self.contiguous();
  const int64_t dim = self.dim();
  // [Note: onednn inner product with Pytorc Linear]
  // We use onednn inner_product primitive to support Pytorch linear
  // Since the semantic of onednn inner_product is different with
  // Pytorch linear while input's dimension > 2
  // https://oneapi-src.github.io/oneDNN/dev_guide_inner_product.html#forward
  // https://pytorch.org/docs/stable/generated/torch.nn.functional.linear.html?highlight=linear#torch.nn.functional.linear
  // We need to reshape input to 2d to make them semantic aligned
  auto self_reshaped =
      dim == 2 ? self_ : self_.reshape({-1, self.size(self.dim() - 1)});
  const ideep::tensor mkldnn_input = itensor_view_from_dense(self_reshaped);
  // output.sizes() will return a reference for output's size which will not
  // hold the underlaying storage. It will be released if output are dead
  // (output = output.reshape(output_size_reshaped)) output.sizes().vec() will
  // trigger a copy and can hold the sizes vector.
  auto output_size = output.sizes().vec();
  bool out_is_contiguous = output.is_contiguous();
  auto output_ = out_is_contiguous ? output : output.contiguous();
  if (dim != 2) {
    std::vector<int64_t> output_size_reshaped = {
        self_reshaped.size(0), mkldnn_weight.get_dim(0)};
    output_ = output_.reshape(output_size_reshaped);
  }
  ideep::tensor mkldnn_output = itensor_view_from_dense(output_);

  if (bias.defined()) {
    auto bias_ = self.is_contiguous() ? bias : bias.contiguous();
    const ideep::tensor mkldnn_bias = itensor_view_from_dense(bias_);
    if (post_op_src.empty()) {
      ideep::inner_product_forward::
          compute</*reorder_src=*/false, /*reorder_weight=*/false>(
              mkldnn_input, mkldnn_weight, mkldnn_bias, mkldnn_output, attr);
    } else {
      ideep::inner_product_forward::
          compute_binary</*reorder_src=*/false, /*reorder_weight=*/false>(
              mkldnn_input,
              post_op_src,
              mkldnn_weight,
              mkldnn_bias,
              mkldnn_output,
              attr);
    }
  } else {
    if (post_op_src.empty()) {
      ideep::inner_product_forward::
          compute</*reorder_src=*/false, /*reorder_weight=*/false>(
              mkldnn_input, mkldnn_weight, mkldnn_output, attr);
    } else {
      ideep::inner_product_forward::
          compute_binary</*reorder_src=*/false, /*reorder_weight=*/false>(
              mkldnn_input, post_op_src, mkldnn_weight, mkldnn_output, attr);
    }
  }
  if (self.dim() != 2) {
    output_ = output_.reshape(output_size);
  }
  if (!out_is_contiguous || output.data_ptr() != output_.data_ptr()) {
    output.copy_(output_);
  }
}

at::Tensor linear_kernel(
    const at::Tensor& self,
    const ideep::tensor& mkldnn_weight,
    const at::Tensor& bias,
    const ideep::attr_t& attr,
    const std::vector<ideep::tensor>& post_op_src) {
  auto input_size = self.sizes();
  std::vector<int64_t> output_size(input_size.begin(), input_size.end() - 1);
  output_size.push_back(mkldnn_weight.get_dim(0));
  auto output = at::empty(output_size, self.options());
  linear_kernel_output(self, mkldnn_weight, bias, output, attr, post_op_src);
  return output;
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> linear_backward_kernel(
    const at::Tensor& input,
    const at::Tensor& grad_output,
    const at::Tensor& weight,
    std::array<bool, 3> output_mask,
    ideep::tensor packed_weight,
    const c10::optional<at::Tensor>& bias) {
  at::Tensor grad_input, grad_weight, grad_bias;
  // weight's desc is needed for both bw_d and bw_w
  // for IP, currently both stag=ab and dtag=ab are only supported by onednn, we
  // need first make both src and diff_dst contiguous if the input or
  // grad_output is not expected
  auto input_contiguous = input.is_contiguous() ? input : input.contiguous();
  auto input_reshaped = input_contiguous.dim() > 2
      ? input_contiguous.reshape(
            {-1, input_contiguous.size(input_contiguous.dim() - 1)})
      : input_contiguous;
  auto grad_output_contiguous =
      grad_output.is_contiguous() ? grad_output : grad_output.contiguous();
  auto grad_output_reshaped = grad_output_contiguous.dim() > 2
      ? grad_output_contiguous.reshape(
            {-1, grad_output_contiguous.size(grad_output_contiguous.dim() - 1)})
      : grad_output_contiguous;
  const ideep::tensor grady = itensor_view_from_dense(grad_output_reshaped);
  if (output_mask[0]) {
    at::Tensor grad_input_reshaped = at::empty_like(input_reshaped);
    ideep::tensor gradx = itensor_view_from_dense(grad_input_reshaped);

    // bw_d
    ideep::inner_product_backward_data::compute(
        grady,
        packed_weight,
        input_reshaped.sizes().vec(),
        gradx,
        ideep::attr_t(torch_ipex::fpmath_mode));
    grad_input = input_contiguous.dim() > 2
        ? grad_input_reshaped.reshape(input_contiguous.sizes().vec())
        : grad_input_reshaped;
  }
  if (output_mask[1] || output_mask[2]) {
    // bw_w
    grad_weight = at::empty_like(weight);
    const ideep::tensor x = itensor_view_from_dense(input_reshaped);
    auto diff_weight_type = packed_weight.get_data_type();
    ideep::tensor gradw(packed_weight.get_desc(), grad_weight.data_ptr());
    if (output_mask[2]) {
      grad_bias = at::empty({packed_weight.get_dim(0)}, weight.options());
      ideep::tensor gradb = itensor_view_from_dense(grad_bias);
      ideep::inner_product_backward_weights::compute(
          x,
          grady,
          gradw,
          gradb,
          diff_weight_type,
          ideep::attr_t(torch_ipex::fpmath_mode));
    } else {
      ideep::inner_product_backward_weights::compute(
          x,
          grady,
          gradw,
          diff_weight_type,
          ideep::attr_t(torch_ipex::fpmath_mode));
    }
  }
  return std::make_tuple(grad_input, grad_weight, grad_bias);
}

at::Tensor linear_forward(
    const at::Tensor& input,
    const at::Tensor& weight,
    const c10::optional<at::Tensor>& bias,
    const at::Tensor& op_context,
    const c10::optional<int64_t> out_features) {
  return reinterpret_cast<IpexLinearOpContext*>(
             op_context.data_ptr<int64_t>()[0])
      ->run(input, ideep::attr_t(torch_ipex::fpmath_mode));
}

at::Tensor linear_eltwise_forward(
    const at::Tensor& input,
    const at::Tensor& weight,
    const c10::optional<at::Tensor>& bias,
    const int64_t eltwise,
    const at::Tensor& op_context,
    const c10::optional<int64_t> out_features) {
  auto attr = ideep::attr_t();
  if (eltwise == ReLU)
    attr = ideep::attr_t::fuse_relu();
  else
    attr = ideep::attr_t::fuse_sigmoid();
  return reinterpret_cast<IpexLinearOpContext*>(
             op_context.data_ptr<int64_t>()[0])
      ->run(input, attr.set_fpmath_mode(torch_ipex::fpmath_mode));
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> linear_backward(
    const at::Tensor& input,
    const at::Tensor& weight,
    const c10::optional<at::Tensor>& bias,
    const at::Tensor& grad_output,
    std::array<bool, 3> output_mask,
    const at::Tensor& op_context) {
  return reinterpret_cast<IpexLinearOpContext*>(
             op_context.data_ptr<int64_t>()[0])
      ->run_backward(input, grad_output, output_mask);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> linear_eltwise_backward(
    const at::Tensor& input,
    const at::Tensor& weight,
    const c10::optional<at::Tensor>& bias,
    const at::Tensor& output,
    const int64_t eltwise,
    const at::Tensor& grad_output,
    std::array<bool, 3> output_mask,
    const at::Tensor& op_context) {
  at::Tensor grad_output_new = eltwise == ReLU
      ? relu_use_dst_for_bwd(grad_output, output)
      : sigmoid_use_dst_for_bwd(grad_output, output);
  return reinterpret_cast<IpexLinearOpContext*>(
             op_context.data_ptr<int64_t>()[0])
      ->run_backward(input, grad_output_new, output_mask);
}

at::Tensor IPEXLinearOp::_forward(
    const at::Tensor& input,
    const at::Tensor& weight,
    const c10::optional<at::Tensor>& bias,
    const int64_t eltwise,
    const at::Tensor& op_context,
    const c10::optional<int64_t> out_features) {
  at::AutoDispatchBelowADInplaceOrView g;
  RECORD_FUNCTION("IPEXLinearOp::_forward", c10::ArrayRef<c10::IValue>({}));

  if (eltwise == NotFused) {
    static auto op = torch::Dispatcher::singleton()
                         .findSchemaOrThrow("torch_ipex::ipex_linear", "")
                         .typed<decltype(ipex_linear)>();
    return op.call(input, weight, bias, op_context, out_features);
  } else {
    static auto op =
        torch::Dispatcher::singleton()
            .findSchemaOrThrow("torch_ipex::ipex_linear_eltwise", "")
            .typed<decltype(ipex_linear_eltwise)>();
    return op.call(input, weight, bias, eltwise, op_context, out_features);
  }
}

at::Tensor IPEXLinearOp::forward(
    torch::autograd::AutogradContext* ctx,
    const at::Tensor& input,
    const at::Tensor& weight,
    const c10::optional<at::Tensor>& bias,
    const int64_t eltwise,
    const at::Tensor& op_context,
    const c10::optional<int64_t> out_features) {
  RECORD_FUNCTION("IPEXLinearOp::forward", c10::ArrayRef<c10::IValue>({}));

  at::AutoDispatchBelowADInplaceOrView g;
  ctx->saved_data["op_context"] = op_context;
  ctx->saved_data["input_requires_grad"] = input.requires_grad();
  ctx->saved_data["weight_requires_grad"] = weight.requires_grad();
  ctx->saved_data["bias_requires_grad"] =
      bias.has_value() && bias.value().requires_grad() ? true : false;
  ctx->saved_data["eltwise"] = eltwise;
  ctx->saved_data["bias"] = bias;
  auto output =
      _forward(input, weight, bias, eltwise, op_context, out_features);
  if (eltwise == NotFused)
    ctx->save_for_backward({input, weight});
  else
    ctx->save_for_backward({input, weight, output});
  return output;
}

torch::autograd::tensor_list IPEXLinearOp::backward(
    torch::autograd::AutogradContext* ctx,
    torch::autograd::tensor_list grad_outputs) {
  RECORD_FUNCTION("IPEXLinearOp::backward", c10::ArrayRef<c10::IValue>({}));

  auto saved = ctx->get_saved_variables();
  at::Tensor input = saved[0];
  at::Tensor weight = saved[1];
  auto op_context = ctx->saved_data["op_context"].toTensor();
  std::array<bool, 3> output_mask;
  output_mask[0] = ctx->saved_data["input_requires_grad"].toBool();
  output_mask[1] = ctx->saved_data["weight_requires_grad"].toBool();
  output_mask[2] = ctx->saved_data["bias_requires_grad"].toBool();
  int64_t eltwise = ctx->saved_data["eltwise"].toInt();
  auto batch_size = ctx->saved_data["batch_size"].toOptional<int64_t>();
  auto bias = ctx->saved_data["bias"].toOptional<at::Tensor>();
  at::Tensor grad_output;

  at::Tensor grad_input, grad_weight, grad_bias;
  if (eltwise == NotFused) {
    static auto op = torch::Dispatcher::singleton()
                         .findSchemaOrThrow("torch_ipex::linear_backward", "")
                         .typed<decltype(linear_backward)>();
    std::tie(grad_input, grad_weight, grad_bias) =
        op.call(input, weight, bias, grad_outputs[0], output_mask, op_context);
  } else {
    at::Tensor output = saved[2];
    static auto op =
        torch::Dispatcher::singleton()
            .findSchemaOrThrow("torch_ipex::linear_eltwise_backward", "")
            .typed<decltype(linear_eltwise_backward)>();
    std::tie(grad_input, grad_weight, grad_bias) = op.call(
        input,
        weight,
        bias,
        output,
        eltwise,
        grad_outputs[0],
        output_mask,
        op_context);
  }

  return {
      grad_input,
      grad_weight,
      grad_bias,
      at::Tensor(),
      at::Tensor(),
      at::Tensor()};
}

at::Tensor ipex_linear(
    const at::Tensor& input,
    const at::Tensor& weight,
    const c10::optional<at::Tensor>& bias,
    const at::Tensor& op_context,
    const c10::optional<int64_t> out_features) {
  if (at::GradMode::is_enabled())
    return IPEXLinearOp::apply(
        input, weight, bias, NotFused, op_context, out_features);
  return IPEXLinearOp::_forward(
      input, weight, bias, NotFused, op_context, out_features);
}

at::Tensor ipex_linear_eltwise(
    const at::Tensor& input,
    const at::Tensor& weight,
    const c10::optional<at::Tensor>& bias,
    const int64_t eltwise,
    const at::Tensor& op_context,
    const c10::optional<int64_t> out_features) {
  return IPEXLinearOp::apply(
      input, weight, bias, eltwise, op_context, out_features);
}

#ifdef USE_LIBXSMM
IPEX_DEFINE_DISPATCH(woq_tpp_gemm_packB_stub);
at::Tensor woq_linear_pack_weight(
    const at::Tensor& weight,
    int64_t weight_dtype,
    std::vector<int64_t>& weight_shape,
    int64_t group_size,
    int64_t lowp_mode) {
  // TPP kernel does not support edge cases
  // It generates packed weight in 4d (Nc, Kc, block_k, block_n)
  auto N = weight_shape[0], K = weight_shape[1];
  // For TPP kernel, we only consider even K
  if (K % 2 == 0) {
    size_t block_n = 32;
    size_t block_k = group_size > 0 ? std::min(group_size, (int64_t)64) : 64;
    while (K % block_k != 0) {
      block_k /= 2;
    }
    assert(block_k > 0);
    if (weight_dtype == WOQ_DTYPE_INT4 || weight_dtype == WOQ_DTYPE_NF4) {
      if (block_k % 4 && lowp_mode == 3) {
        // This case is not supported by kernel
        return weight;
      }
      // Create a new non-quantized tensor in data type uint8 (Byte)
      // One uint8 holds two int4 values. Compressed along K.
      // N is padded to the nearest multiple of block_n.
      // Note that weight is already compressed
      int64_t K_int4_compressed = K / 2;
      int64_t N_int4 = N % block_n ? N / block_n * block_n + block_n : N;
      at::Tensor weight_int4 = at::empty(
          {N_int4, K_int4_compressed}, device(c10::kCPU).dtype(c10::kByte));
      int64_t weight_size_bytes = weight.numel();
      int64_t weight_int4_size_bytes = weight_int4.numel();
      int64_t pad_size_bytes = weight_int4_size_bytes - weight_size_bytes;
      std::memcpy(weight_int4.data_ptr(), weight.data_ptr(), weight_size_bytes);
      std::memset(
          (uint8_t*)weight_int4.data_ptr() + weight_size_bytes,
          0,
          pad_size_bytes);
      return woq_tpp_gemm_packB_stub(
          kCPU, weight_int4, weight_dtype, block_n, block_k, lowp_mode);
    }
    if (N % block_n) {
      return weight;
    } else {
      return woq_tpp_gemm_packB_stub(
          kCPU, weight, weight_dtype, block_n, block_k, lowp_mode);
    }
  }
  return weight;
}

IPEX_DEFINE_DISPATCH(woq_tpp_gemm_unpackB_stub);
at::Tensor woq_linear_unpack_weight(
    const at::Tensor& weight,
    int64_t weight_dtype,
    int64_t lowp_mode) {
  return woq_tpp_gemm_unpackB_stub(kCPU, weight, weight_dtype, lowp_mode);
}

IPEX_DEFINE_DISPATCH(woq_tpp_gemm_kernel_stub);
at::Tensor woq_linear_kernel(
    const at::Tensor& self,
    const at::Tensor& weight,
    int64_t weight_dtype,
    const std::vector<at::Tensor>& scales_list,
    const std::vector<at::Tensor>& zps_list,
    const std::vector<at::Tensor>& bias_list,
    int64_t group_size,
    int64_t lowp_mode,
    int64_t act_quant_mode) {
  int64_t quant_w_mode = group_size > 0 ? 1 : 0;
  return woq_tpp_gemm_kernel_stub(
      kCPU,
      self,
      weight,
      scales_list,
      zps_list,
      bias_list,
      weight_dtype,
      lowp_mode,
      WOQ_FUSE_NONE, // no post op fusion
      std::vector<at::Tensor>(),
      act_quant_mode,
      quant_w_mode,
      group_size);
}

at::Tensor woq_linear_forward(
    const at::Tensor& input,
    const at::Tensor& op_context) {
  RECORD_FUNCTION(
      "torch_ipex::ipex_woq_linear", c10::ArrayRef<c10::IValue>({}));
  return reinterpret_cast<IpexWoqLinearOpContext*>(
             op_context.data_ptr<int64_t>()[0])
      ->run(input);
}

at::Tensor woq_linear_eltwise_kernel(
    const at::Tensor& self,
    const at::Tensor& weight,
    int64_t weight_dtype,
    const std::vector<at::Tensor>& scales_list,
    const std::vector<at::Tensor>& zps_list,
    const std::vector<at::Tensor>& bias_list,
    const c10::string_view& post_op,
    const torch::List<c10::optional<at::Scalar>>& scalars,
    const c10::optional<c10::string_view>& algorithm,
    int64_t group_size,
    int64_t lowp_mode,
    int64_t act_quant_mode) {
  int64_t post_op_fusion_type = WOQ_FUSE_NONE;
  if (post_op == "gelu") {
    if (algorithm == "none") {
      post_op_fusion_type = WOQ_FUSE_GELU;
    } else if (algorithm == "tanh") {
      post_op_fusion_type = WOQ_FUSE_NEW_GELU;
    }
  }
  int64_t quant_w_mode = group_size > 0 ? 1 : 0;
  return woq_tpp_gemm_kernel_stub(
      kCPU,
      self,
      weight,
      scales_list,
      zps_list,
      bias_list,
      weight_dtype,
      lowp_mode,
      post_op_fusion_type,
      std::vector<at::Tensor>(),
      act_quant_mode,
      quant_w_mode,
      group_size);
}

at::Tensor woq_linear_gelu_forward(
    const at::Tensor& input,
    const at::Tensor& op_context) {
  RECORD_FUNCTION(
      "torch_ipex::woq_linear_gelu", c10::ArrayRef<c10::IValue>({}));
  return reinterpret_cast<IpexWoqLinearOpContext*>(
             op_context.data_ptr<int64_t>()[0])
      ->run_eltwise(
          input, "gelu", torch::List<c10::optional<at::Scalar>>(), "none");
}

at::Tensor woq_linear_new_gelu_forward(
    const at::Tensor& input,
    const at::Tensor& op_context) {
  RECORD_FUNCTION(
      "torch_ipex::woq_linear_new_gelu", c10::ArrayRef<c10::IValue>({}));
  return reinterpret_cast<IpexWoqLinearOpContext*>(
             op_context.data_ptr<int64_t>()[0])
      ->run_eltwise(
          input, "gelu", torch::List<c10::optional<at::Scalar>>(), "tanh");
}

at::Tensor woq_linear_add_kernel(
    const at::Tensor& self,
    const at::Tensor& weight,
    int64_t weight_dtype,
    const std::vector<at::Tensor>& scales_list,
    const std::vector<at::Tensor>& zps_list,
    const std::vector<at::Tensor>& bias_list,
    int64_t group_size,
    int64_t lowp_mode,
    const std::vector<at::Tensor>& others,
    int64_t act_quant_mode) {
  int64_t quant_w_mode = group_size > 0 ? 1 : 0;
  return woq_tpp_gemm_kernel_stub(
      kCPU,
      self,
      weight,
      scales_list,
      zps_list,
      bias_list,
      weight_dtype,
      lowp_mode,
      WOQ_FUSE_ADD, // post op add
      others,
      act_quant_mode,
      quant_w_mode,
      group_size);
}

at::Tensor woq_linear_add_add_kernel(
    const at::Tensor& self,
    const at::Tensor& weight,
    int64_t weight_dtype,
    const std::vector<at::Tensor>& scales_list,
    const std::vector<at::Tensor>& zps_list,
    const std::vector<at::Tensor>& bias_list,
    int64_t group_size,
    int64_t lowp_mode,
    const std::vector<at::Tensor>& others,
    int64_t act_quant_mode) {
  int64_t quant_w_mode = group_size > 0 ? 1 : 0;
  return woq_tpp_gemm_kernel_stub(
      kCPU,
      self,
      weight,
      scales_list,
      zps_list,
      bias_list,
      weight_dtype,
      lowp_mode,
      WOQ_FUSE_ADD_ADD, // post op add-add
      others,
      act_quant_mode,
      quant_w_mode,
      group_size);
}

at::Tensor woq_linear_add_forward(
    const at::Tensor& input,
    const at::Tensor& op_context,
    const std::vector<at::Tensor>& others) {
  RECORD_FUNCTION("torch_ipex::woq_linear_add", c10::ArrayRef<c10::IValue>({}));
  return reinterpret_cast<IpexWoqLinearOpContext*>(
             op_context.data_ptr<int64_t>()[0])
      ->run_add(input, others);
}

at::Tensor woq_linear_add_add_forward(
    const at::Tensor& input,
    const at::Tensor& op_context,
    const std::vector<at::Tensor>& others) {
  RECORD_FUNCTION(
      "torch_ipex::woq_linear_add_add", c10::ArrayRef<c10::IValue>({}));
  return reinterpret_cast<IpexWoqLinearOpContext*>(
             op_context.data_ptr<int64_t>()[0])
      ->run_add_add(input, others);
}
#endif

at::Tensor matmul_i8i8i32(const at::Tensor& input, const at::Tensor& weight) {
  // x:s8 * w:s8 -> y:s32
  // No bias
  TORCH_CHECK(
      input.scalar_type() == c10::kChar,
      "matmul_i8i8i32: input dtype should be signed int8 but found ",
      input.scalar_type());
  TORCH_CHECK(
      weight.scalar_type() == c10::kChar,
      "matmul_i8i8i32: weight dtype should be signed int8 but found ",
      weight.scalar_type());
  TORCH_CHECK(
      input.dim() == 2 && weight.dim() == 2,
      "matmul_i8i8i32: Expect Input and weight are 2d but got ",
      input.dim(),
      " and ",
      weight.dim());
  TORCH_CHECK(
      input.size(1) == weight.size(1),
      "matmul_i8i8i32: Input shape and weight shape do not match, got ",
      input.sizes(),
      " and ",
      weight.sizes());
  auto output_shape = input.sizes().vec();
  output_shape.back() = weight.size(0);
  auto output = at::empty(output_shape, input.options().dtype(c10::kInt));
  auto input_contig = input.contiguous();
  auto weight_contig = weight.t().contiguous();
  // Create ideep tensors for oneDNN computation
  auto src = ideep::tensor(
      {input_contig.sizes().vec(),
       ideep::tensor::data_type::s8,
       input_contig.strides().vec()},
      input_contig.data_ptr());
  auto wei = ideep::tensor(
      {weight_contig.sizes().vec(),
       ideep::tensor::data_type::s8,
       weight_contig.strides().vec()},
      weight_contig.data_ptr());
  auto dst = ideep::tensor(
      {output.sizes().vec(),
       ideep::tensor::data_type::s32,
       output.strides().vec()},
      output.data_ptr());
  // Create primitive desc
  auto engine = ideep::engine::cpu_engine();
  ideep::attr_t op_attr;
  op_attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);
  auto src_desc = src.get_desc();
  auto wei_desc = wei.get_desc();
  auto dst_desc = dst.get_desc();
  auto prim_desc = dnnl::matmul::primitive_desc(
      engine, src_desc, wei_desc, dst_desc, op_attr);
  // Reorder weight
  auto expected_weight = wei.reorder_if_differ_in(prim_desc.weights_desc());
  // Prepare args for primitive
  ideep::tensor scratchpad(prim_desc.scratchpad_desc());
  ideep::exec_args args;
  args.insert({DNNL_ARG_SRC, src});
  args.insert({DNNL_ARG_WEIGHTS, expected_weight});
  args.insert({DNNL_ARG_DST, dst});
  args.insert({DNNL_ARG_SCRATCHPAD, scratchpad});
  // Create primitve and execute
  auto primitive = dnnl::matmul(prim_desc);
  primitive.execute(ideep::stream::default_stream(), args);
  return output;
}

} // namespace cpu
} // namespace torch_ipex

namespace torch_ipex {
namespace autocast {

at::Tensor ipex_linear(
    const at::Tensor& input,
    const at::Tensor& weight,
    const c10::optional<at::Tensor>& bias,
    const at::Tensor& op_context,
    const c10::optional<int64_t> out_features) {
  c10::impl::ExcludeDispatchKeyGuard no_autocastCPU(DispatchKey::AutocastCPU);
  static auto op = torch::Dispatcher::singleton()
                       .findSchemaOrThrow("torch_ipex::ipex_linear", "")
                       .typed<decltype(ipex_linear)>();
  auto target_type = get_autocast_dtype();
  TORCH_CHECK(
      weight.scalar_type() == at::kBFloat16 ||
          weight.scalar_type() == at::kHalf ||
          weight.scalar_type() == at::kFloat,
      "ipex_linear only support bfloat16, float16 and float autocast dtype");
  // should not autocast weight/bias here since we are using it from op_context,
  // The cast for weight/bias should be only handled in ipex.optimize
  return op.call(
      cpu_cached_cast(target_type, input),
      weight,
      bias,
      op_context,
      out_features);
}

at::Tensor ipex_linear_eltwise(
    const at::Tensor& input,
    const at::Tensor& weight,
    const c10::optional<at::Tensor>& bias,
    const int64_t eltwise,
    const at::Tensor& op_context,
    const c10::optional<int64_t> out_features) {
  c10::impl::ExcludeDispatchKeyGuard no_autocastCPU(DispatchKey::AutocastCPU);
  static auto op = torch::Dispatcher::singleton()
                       .findSchemaOrThrow("torch_ipex::ipex_linear_eltwise", "")
                       .typed<decltype(ipex_linear_eltwise)>();
  auto target_type = get_autocast_dtype();
  TORCH_CHECK(
      weight.scalar_type() == at::kBFloat16 ||
          weight.scalar_type() == at::kFloat,
      "ipex_linear_eltwise only support bfloat16 and float autocast dtype");
  // should not autocast weight/bias here since we are using it from op_context,
  // The cast for weight/bias should be only handled in ipex.optimize
  return op.call(
      cpu_cached_cast(target_type, input),
      weight,
      bias,
      eltwise,
      op_context,
      out_features);
}

#ifdef USE_LIBXSMM
at::Tensor woq_linear_forward(
    const at::Tensor& input,
    const at::Tensor& op_context) {
  c10::impl::ExcludeDispatchKeyGuard no_autocastCPU(DispatchKey::AutocastCPU);
  static auto op = torch::Dispatcher::singleton()
                       .findSchemaOrThrow("torch_ipex::ipex_woq_linear", "")
                       .typed<decltype(woq_linear_forward)>();
  auto target_type = get_autocast_dtype();
  return op.call(cpu_cached_cast(target_type, input), op_context);
}

at::Tensor woq_linear_gelu_forward(
    const at::Tensor& input,
    const at::Tensor& op_context) {
  c10::impl::ExcludeDispatchKeyGuard no_autocastCPU(DispatchKey::AutocastCPU);
  static auto op = torch::Dispatcher::singleton()
                       .findSchemaOrThrow("torch_ipex::woq_linear_gelu", "")
                       .typed<decltype(woq_linear_gelu_forward)>();
  auto target_type = get_autocast_dtype();
  return op.call(cpu_cached_cast(target_type, input), op_context);
}

at::Tensor woq_linear_new_gelu_forward(
    const at::Tensor& input,
    const at::Tensor& op_context) {
  c10::impl::ExcludeDispatchKeyGuard no_autocastCPU(DispatchKey::AutocastCPU);
  static auto op = torch::Dispatcher::singleton()
                       .findSchemaOrThrow("torch_ipex::woq_linear_new_gelu", "")
                       .typed<decltype(woq_linear_new_gelu_forward)>();
  auto target_type = get_autocast_dtype();
  return op.call(cpu_cached_cast(target_type, input), op_context);
}

at::Tensor woq_linear_add_forward(
    const at::Tensor& input,
    const at::Tensor& op_context,
    const std::vector<at::Tensor>& others) {
  c10::impl::ExcludeDispatchKeyGuard no_autocastCPU(DispatchKey::AutocastCPU);
  static auto op = torch::Dispatcher::singleton()
                       .findSchemaOrThrow("torch_ipex::woq_linear_add", "")
                       .typed<decltype(woq_linear_add_forward)>();
  auto target_type = get_autocast_dtype();
  return op.call(
      cpu_cached_cast(target_type, input),
      op_context,
      cpu_cached_cast(target_type, others));
}

at::Tensor woq_linear_add_add_forward(
    const at::Tensor& input,
    const at::Tensor& op_context,
    const std::vector<at::Tensor>& others) {
  c10::impl::ExcludeDispatchKeyGuard no_autocastCPU(DispatchKey::AutocastCPU);
  static auto op = torch::Dispatcher::singleton()
                       .findSchemaOrThrow("torch_ipex::woq_linear_add_add", "")
                       .typed<decltype(woq_linear_add_add_forward)>();
  auto target_type = get_autocast_dtype();
  return op.call(
      cpu_cached_cast(target_type, input),
      op_context,
      cpu_cached_cast(target_type, others));
}
#endif

at::Tensor matmul_i8i8i32(const at::Tensor& input, const at::Tensor& weight) {
  c10::impl::ExcludeDispatchKeyGuard no_autocastCPU(DispatchKey::AutocastCPU);
  static auto op = torch::Dispatcher::singleton()
                       .findSchemaOrThrow("torch_ipex::matmul_i8i8i32", "")
                       .typed<decltype(matmul_i8i8i32)>();
  // input is int8. Don't cast to autocast dtype
  return op.call(input, weight);
}

} // namespace autocast
} // namespace torch_ipex

namespace {

TORCH_LIBRARY_FRAGMENT(torch_ipex, m) {
  m.def(
      "ipex_linear(Tensor input, Tensor weight, Tensor? bias, "
      "Tensor W_prepack, int? out_features) -> Tensor");
  m.impl(
      "ipex_linear", c10::DispatchKey::Autograd, torch_ipex::cpu::ipex_linear);
  m.impl(
      "ipex_linear",
      c10::DispatchKey::AutocastCPU,
      torch_ipex::autocast::ipex_linear);
  m.impl("ipex_linear", c10::DispatchKey::CPU, torch_ipex::cpu::linear_forward);
#ifdef USE_LIBXSMM
  m.def("ipex_woq_linear(Tensor input, Tensor W_prepack) -> Tensor");
  m.impl(
      "ipex_woq_linear",
      c10::DispatchKey::CPU,
      torch_ipex::cpu::woq_linear_forward);
  m.impl(
      "ipex_woq_linear",
      c10::DispatchKey::AutocastCPU,
      torch_ipex::autocast::woq_linear_forward);
  m.def("woq_linear_gelu(Tensor input, Tensor W_prepack) -> Tensor");
  m.impl(
      "woq_linear_gelu",
      c10::DispatchKey::CPU,
      torch_ipex::cpu::woq_linear_gelu_forward);
  m.impl(
      "woq_linear_gelu",
      c10::DispatchKey::AutocastCPU,
      torch_ipex::autocast::woq_linear_gelu_forward);
  m.def("woq_linear_new_gelu(Tensor input, Tensor W_prepack) -> Tensor");
  m.impl(
      "woq_linear_new_gelu",
      c10::DispatchKey::CPU,
      torch_ipex::cpu::woq_linear_new_gelu_forward);
  m.impl(
      "woq_linear_new_gelu",
      c10::DispatchKey::AutocastCPU,
      torch_ipex::autocast::woq_linear_new_gelu_forward);
  m.def(
      "woq_linear_add(Tensor input, Tensor W_prepack, Tensor[] others) -> Tensor");
  m.impl(
      "woq_linear_add",
      c10::DispatchKey::CPU,
      torch_ipex::cpu::woq_linear_add_forward);
  m.impl(
      "woq_linear_add",
      c10::DispatchKey::AutocastCPU,
      torch_ipex::autocast::woq_linear_add_forward);
  m.def(
      "woq_linear_add_add(Tensor input, Tensor W_prepack, Tensor[] others) -> Tensor");
  m.impl(
      "woq_linear_add_add",
      c10::DispatchKey::CPU,
      torch_ipex::cpu::woq_linear_add_add_forward);
  m.impl(
      "woq_linear_add_add",
      c10::DispatchKey::AutocastCPU,
      torch_ipex::autocast::woq_linear_add_add_forward);
#endif
  // fuse eltwise
  m.def(
      "ipex_linear_eltwise(Tensor input, Tensor weight, Tensor? bias, int eltwise, "
      "Tensor W_prepack, int? out_features) -> Tensor");
  m.impl(
      "ipex_linear_eltwise",
      c10::DispatchKey::Autograd,
      torch_ipex::cpu::ipex_linear_eltwise);
  m.impl(
      "ipex_linear_eltwise",
      c10::DispatchKey::AutocastCPU,
      torch_ipex::autocast::ipex_linear_eltwise);
  m.impl(
      "ipex_linear_eltwise",
      c10::DispatchKey::CPU,
      torch_ipex::cpu::linear_eltwise_forward);
  // bw
  m.def(
      "linear_backward(Tensor input, Tensor weight, Tensor? bias, Tensor grad_output, bool[3] out_mask, "
      "Tensor W_prepack) -> (Tensor, Tensor, Tensor)");
  m.impl(
      "linear_backward",
      c10::DispatchKey::CPU,
      torch_ipex::cpu::linear_backward);
  m.def(
      "linear_eltwise_backward(Tensor input, Tensor weight, Tensor? bias, Tensor output, int eltwise, Tensor grad_output, bool[3] out_mask, "
      "Tensor W_prepack) -> (Tensor, Tensor, Tensor)");
  m.impl(
      "linear_eltwise_backward",
      c10::DispatchKey::CPU,
      torch_ipex::cpu::linear_eltwise_backward);
  // bnb
  m.def("matmul_i8i8i32(Tensor input, Tensor weight) -> Tensor");
  m.impl(
      "matmul_i8i8i32", c10::DispatchKey::CPU, torch_ipex::cpu::matmul_i8i8i32);
  m.impl(
      "matmul_i8i8i32",
      c10::DispatchKey::AutocastCPU,
      torch_ipex::autocast::matmul_i8i8i32);
}

} // namespace
