#include "csrc/utils/CustomOperatorRegistration.h"
#include "fp8_utils.h"
#include "ideep/IDeepConversions.h"

namespace torch_ipex {
namespace cpu {

using namespace torch_ipex::cpu;
at::Tensor fp8_linear_impl(
    at::Tensor inp_fp8,
    at::Tensor scale_invA,
    int64_t idxA,
    at::Tensor weight_fp8,
    at::Tensor bias,
    at::Tensor scale_invB,
    int64_t idxB,
    at::Tensor& out) {
  RECORD_FUNCTION("fp8_linear_impl", c10::ArrayRef<c10::IValue>({}));

  const int64_t dim = inp_fp8.dim();
  // reshape first if input dim != 2 and the reshape will cost a memory copy.
  auto inp_fp8_reshaped = dim == 2
      ? inp_fp8
      : inp_fp8.reshape({-1, inp_fp8.size(inp_fp8.dim() - 1)});
  at::Tensor out_reshaped;
  std::vector<int64_t> out_sizes;
  if (out.defined()) {
    out_reshaped = out;
    out_sizes = out.sizes().vec();
  } else {
    out_sizes = inp_fp8.sizes().vec();
    out_sizes[1] = weight_fp8.size(0);
    out_reshaped =
        at::empty(out_sizes, device(c10::kCPU).dtype(c10::ScalarType::Float));
  }

  float input_scale = scale_invA[idxA].item<float>();
  float weight_scale = scale_invB[idxB].item<float>();
  auto weight_fp8_t = weight_fp8.transpose(0, 1).contiguous();
  auto src = torch_ipex::cpu::itensor_view_from_dense(inp_fp8_reshaped);
  auto weight_t = torch_ipex::cpu::itensor_view_from_dense(weight_fp8_t);
  bool with_bias = bias.defined();
  int64_t K = inp_fp8_reshaped.size(-1), M = inp_fp8_reshaped.numel() / K,
          N = weight_fp8_t.size(1);

  std::vector<int64_t> src_dims = {M, K};
  std::vector<int64_t> weight_dims = {K, N};
  std::vector<int64_t> dst_dims = {M, N};
  assert(out_reshaped.size(0) == M);
  assert(out_reshaped.size(1) == N);

  ideep::tensor dst = torch_ipex::cpu::itensor_view_from_dense(out_reshaped);
  auto src_desc = ideep::tensor::desc(
      src_dims,
      get_mkldnn_dtype(inp_fp8.scalar_type()),
      ideep::format_tag::any);
  auto weights_desc = ideep::tensor::desc(
      weight_dims,
      get_mkldnn_dtype(weight_fp8.scalar_type()),
      ideep::format_tag::any);
  auto dst_desc = ideep::tensor::desc(
      dst_dims,
      get_mkldnn_dtype(out_reshaped.scalar_type()),
      ideep::format_tag::any);
  ideep::tensor onednn_bias;
  at::Tensor bias_val_float;
  if (with_bias) {
    if (bias.dim() == 1) {
      auto b_reshape = bias.reshape({1, bias.size(0)});
      onednn_bias = torch_ipex::cpu::itensor_view_from_dense(b_reshape);
    } else {
      onednn_bias = torch_ipex::cpu::itensor_view_from_dense(bias);
    }
  }
  auto bias_desc = with_bias ? ideep::tensor::desc(
                                   onednn_bias.get_dims(),
                                   get_mkldnn_dtype(bias.scalar_type()),
                                   ideep::format_tag::any)
                             : ideep::tensor::desc();
  auto op_attr = ideep::attr_t();
  if (input_scale != 1.0f) {
    op_attr.set_scales_mask(DNNL_ARG_SRC, 0);
  }
  if (input_scale != 1.0f) {
    op_attr.set_scales_mask(DNNL_ARG_WEIGHTS, 0);
  }

  op_attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);
  auto engine = ideep::engine::cpu_engine();
  // TODO: Remove this try/catch when oneDNN provides API to notify
  // framework whether current platform can run FP8 primitives.
  dnnl::matmul::primitive_desc primitive_desc;
  try {
    primitive_desc = with_bias
        ? dnnl::matmul::primitive_desc(
              engine, src_desc, weights_desc, bias_desc, dst_desc, op_attr)
        : dnnl::matmul::primitive_desc(
              engine, src_desc, weights_desc, dst_desc, op_attr);
  } catch (dnnl::error& e) {
    if (e.status == dnnl_unimplemented)
      throw std::runtime_error("Running FP8 on not supported platform.");
    // on any other error just re-throw
    throw;
  }
  auto primitive = dnnl::matmul(primitive_desc);

  // Prepare args and execute primitive
  ideep::tensor scratchpad(primitive_desc.scratchpad_desc());
  ideep::exec_args args;
  args.insert({DNNL_ARG_SRC, src});
  args.insert({DNNL_ARG_WEIGHTS, weight_t});
  args.insert({DNNL_ARG_DST, dst});
  args.insert({DNNL_ARG_SCRATCHPAD, scratchpad});
  if (with_bias) {
    args.insert({DNNL_ARG_BIAS, onednn_bias});
  }
  ideep::tensor src_scales_t = ideep::tensor(ideep::scale_t(1, input_scale));
  ideep::tensor wei_scales_t = ideep::tensor(ideep::scale_t(1, weight_scale));

  if (input_scale != 1.0f) {
    args.insert({DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC, src_scales_t});
  }
  args.insert({DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS, wei_scales_t});

  primitive.execute(ideep::stream::default_stream(), args);

  if (dim != 2) {
    out_reshaped.reshape(out_sizes);
  }
  return out_reshaped;
}

at::Tensor fp8_linear(
    at::Tensor inp_fp8,
    at::Tensor scale_invA,
    int64_t idxA,
    int64_t Atype,
    at::Tensor weight_fp8,
    at::Tensor scale_invB,
    int64_t idxB,
    int64_t Btype,
    at::Tensor bias,
    at::Tensor& out) {
  at::Tensor res = fp8_linear_impl(
      inp_fp8, scale_invA, idxA, weight_fp8, bias, scale_invB, idxB, out);
  return res;
}

} // namespace cpu
} // namespace torch_ipex

namespace {

IPEX_LIBRARY_FRAGMENT() {
  IPEX_OP_IPEX_REGISTER_DISPATCH(
      "fp8_linear", torch_ipex::cpu::fp8_linear, c10::DispatchKey::CPU);
}

} // namespace
