#include <ATen/ATen.h>
#include <ATen/record_function.h>
#include <dnnl.hpp>

#include <ATen/AtenIpexTypeXPU.h>
#include <ATen/ipex_type_dpcpp_customized.h>


namespace torch {
namespace jit {
namespace dpcpp {

at::Tensor& conv2d_sum(at::Tensor& accumu,
    const at::Tensor& input, const at::Tensor& weight, const at::Tensor& bias,
    at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation,
    int64_t groups, at::Scalar alpha) {
  RECORD_FUNCTION("conv2d_sum",
                  std::vector<c10::IValue>({input, weight, bias, accumu}));
  at::AtenIpexTypeXPU::convolution_sum(input, weight, bias,
      stride, padding, dilation, false, {{0, 0}}, groups, accumu, alpha);
  return accumu;
}

at::Tensor& conv2d_sum_relu(at::Tensor& accumu,
    const at::Tensor& input, const at::Tensor& weight, const at::Tensor& bias,
    at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation,
    int64_t groups, at::Scalar alpha) {
  RECORD_FUNCTION("conv2d_sum_relu",
                  std::vector<c10::IValue>({input, weight, bias, accumu}));
  at::AtenIpexTypeXPU::convolution_sum_relu(input, weight, bias,
      stride, padding, dilation, false, {{0, 0}}, groups, accumu, alpha);
  return accumu;
}

at::Tensor conv2d_relu(
    const at::Tensor& input, const at::Tensor& weight, const at::Tensor& bias,
    at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation,
    int64_t groups) {
  RECORD_FUNCTION("conv2d_relu",
                  std::vector<c10::IValue>({input, weight, bias}));
  return at::AtenIpexTypeXPU::convolution_relu(
      input, weight, bias, stride, padding, dilation, false, {{0, 0}}, groups);
}

at::Tensor conv2d_sigmoid(
    const at::Tensor& input, const at::Tensor& weight, const at::Tensor& bias,
    at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation,
    int64_t groups) {
  RECORD_FUNCTION("conv2d_sigmoid",
                  std::vector<c10::IValue>({input, weight, bias}));
  return at::AtenIpexTypeXPU::convolution_sigmoid(
      input, weight, bias, stride, padding, dilation, false, {{0, 0}}, groups);
}

at::Tensor matmul_sum(at::Tensor& accumu,
    const at::Tensor& m1, const at::Tensor& m2, at::Scalar alpha) {
  RECORD_FUNCTION("matmul_sum",
                  std::vector<c10::IValue>({m1, m2, accumu}));
  return at::AtenIpexTypeXPU::matmul_sum(accumu, m1, m2, alpha);
}

at::Tensor trans_matmul_scale_sum(at::Tensor& accumu, const at::Tensor& tensor1,
    const at::Tensor& tensor2, at::Scalar oscale, at::Scalar alpha) {
  RECORD_FUNCTION("trans_matmul_scale_sum",
                  std::vector<c10::IValue>({tensor1, tensor2}));

  auto dim_tensor1 = tensor1.dim();
  auto dim_tensor2 = tensor2.dim();

  // TODO: matmul case is complicated
  // temporarily we only support div fusion for bmm case
  if ((dim_tensor1 >= 1 && dim_tensor2 >= 1) && (dim_tensor1 >= 3 || dim_tensor2 >= 3)) {
    // We are multiplying b1 x n x m1 by x2 x m2 x p (where b1 can be a list);
    // we track m1 vs m2 separately even though they must match for nicer error messages
    int64_t n = dim_tensor1 > 1 ? tensor1.size(-2) : 1;
    int64_t m1 = tensor1.size(-1);
    at::IntArrayRef batch_tensor1(tensor1.sizes().data(), std::max<int64_t>(dim_tensor1 - 2, 0));

    // inverse dims in non-transpose case
    int64_t m2 = dim_tensor2 > 1 ? tensor2.size(-1) : 1;
    int64_t p = tensor2.size(-2);

    at::IntArrayRef batch_tensor2(tensor2.sizes().data(), std::max<int64_t>(dim_tensor2 - 2, 0));

    // expand the batch portion (i.e. cut off matrix dimensions and expand rest)
    std::vector<int64_t> expand_batch_portion = at::infer_size(batch_tensor1, batch_tensor2);

    std::vector<int64_t> tensor1_expand_size(expand_batch_portion);
    tensor1_expand_size.insert(tensor1_expand_size.end(), {n, m1});

    std::vector<int64_t> tensor2_expand_size(expand_batch_portion);
    tensor2_expand_size.insert(tensor2_expand_size.end(), {p, m2});

    int expand_batch_product = std::accumulate(expand_batch_portion.begin(), expand_batch_portion.end(),
                                               1, std::multiplies<int64_t>());

    std::vector<int64_t> tensor1_bmm_view({expand_batch_product});
    tensor1_bmm_view.insert(tensor1_bmm_view.end(), {n, m1});

    std::vector<int64_t> tensor2_bmm_view({expand_batch_product});
    tensor2_bmm_view.insert(tensor2_bmm_view.end(), {p, m2});

    // flatten expanded batches
    at::Tensor tensor1_expanded = tensor1.expand(tensor1_expand_size).contiguous().view(tensor1_bmm_view);
    at::Tensor tensor2_expanded = tensor2.expand(tensor2_expand_size).contiguous().view(tensor2_bmm_view);

    // reshape batches back into result
    std::vector<int64_t> output_shape(expand_batch_portion);
    if (dim_tensor1 > 1) {
      output_shape.push_back(n);
    }
    if (dim_tensor2 > 1) {
      output_shape.push_back(p);
    }

    at::Tensor output, self, input;
    if (accumu.sizes().vec() == output_shape) {
      output = at::_unsafe_view(
          at::AtenIpexTypeXPU::trans_baddbmm_out(accumu,
                                                   accumu,
                                                   tensor1_expanded,
                                                   tensor2_expanded,
                                                   alpha, 1 / oscale.to<float>()),
          output_shape
      );
    } else {
      self = at::empty({0}, tensor1_expanded.options());
      output = at::_unsafe_view(
          at::AtenIpexTypeXPU::trans_baddbmm_out(self,
                                                   input,
                                                   tensor1_expanded,
                                                   tensor2_expanded,
                                                   0.f, 1 / oscale.to<float>()),
          output_shape
      );
      output = at::AtenIpexTypeXPU::add(output, accumu, alpha);
    }

    return output;
  } else {
    return at::AtenIpexTypeXPU::add(
               at::AtenIpexTypeXPU::mul(
                   at::native::matmul(tensor1, tensor2.transpose(-1, -2)),
                   1 / oscale.to<float>()
               ), accumu, alpha
           );
  }
}

at::Tensor mul_add(const at::Tensor& self,
    const at::Tensor& other, const at::Tensor& accumu, at::Scalar alpha) {
  RECORD_FUNCTION("mul_add",
                  std::vector<c10::IValue>({self, other, accumu}));
  return at::AtenIpexTypeXPU::mul_add(self, other, accumu, alpha);
}

at::Tensor q_conv2d_sum_relu(at::Tensor& accumu,
    const at::Tensor& input, const c10::intrusive_ptr<ConvPackedParamsBase<2>>& packed_weight,
    double conv_scale, int64_t conv_zpoint, double sum_scale,
    int64_t sum_zpoint) {
  RECORD_FUNCTION("q_conv2d_sum_relu",
                  std::vector<c10::IValue>({input, packed_weight}));
  return at::AtenIpexTypeXPU::q_conv2d_sum_relu(accumu,
      input, packed_weight, conv_scale,
      conv_zpoint, sum_scale, sum_zpoint);
}

at::Tensor batch_norm(
    const at::Tensor& input, const at::Tensor& weight, const at::Tensor& bias, const at::Tensor& running_mean,
    const at::Tensor& running_var, bool train, double momentum, double eps, bool use_dnn) {
  return at::empty_like(input);
}

at::Tensor fold_weight(
    const at::Tensor& weight, const at::Tensor& bn_weight, const at::Tensor& running_var, float eps) {
  return at::empty_like(weight);
}

at::Tensor fold_bias(
    const at::Tensor& weight, const at::Tensor& bias, const at::Tensor& bn_weight,
    const at::Tensor& bn_bias, const at::Tensor& running_mean, const at::Tensor& running_var, float eps) {
  return at::empty_like(bias);
}

at::Tensor reorder(
    const at::Tensor& input,
    dnnl::memory::format_tag from, dnnl::memory::format_tag to, int64_t groups) {
  return at::empty_like(input);
}

} // dpcpp
} // jit
} // torch
