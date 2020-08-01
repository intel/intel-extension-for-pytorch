#include <ATen/ATen.h>
#include <ATen/ExpandUtils.h>
#include <torch/csrc/autograd/record_function.h>

#include <core/TensorImplUtils.h>

#include <utils/ATDispatch.h>

#include <core/Runtime.h>
#include <dnnl.hpp>

using namespace dnnl;
using namespace at::dpcpp;

namespace at {
namespace AtenIpexTypeDPCPP {
namespace impl {


#if defined(USE_COMPUTECPP)
Tensor computecpp_workaround(const Tensor& t) {
  if (dpcppGetBufferMap().get_offset(t.data_ptr()) != 0) {
    return t.clone();
  }
  return t;
}

template <typename scalar_t>
void mkldnnGemmImpl(
    Tensor& result_,
    scalar_t beta,
    scalar_t alpha,
    const Tensor& m1_,
    const Tensor& m2_,
    bool with_relu) {
  Tensor result = computecpp_workaround(result_);
  Tensor m1 = computecpp_workaround(m1_);
  Tensor m2 = computecpp_workaround(m2_);
#else
template <typename scalar_t>
void mkldnnGemmImpl(
    Tensor& result,
    scalar_t beta,
    scalar_t alpha,
    const Tensor& m1,
    const Tensor& m2,
    bool with_relu) {
#endif
  size_t dims = result.dim();
  TORCH_CHECK(dims == 2 || dims == 3, "mkldnn matmul only works with 2D or 3D, got ", dims);
  TORCH_CHECK(dims == m1.dim() && dims == m2.dim(), "mkldnn input matrixes must have the same ranks");

  int64_t m = result.size(-2);
  int64_t n = result.size(-1);
  int64_t k = m1.size(-1);
  int64_t mb = 1;
  if (dims == 3) {
    mb = result.size(0);
    TORCH_CHECK(mb == m1.size(0) && mb == m2.size(0), "batch size mismatch, result mb: ",\
        mb, "m1 mb", m1.size(0), " m2 mb: ", m2.size(0));
  }
  TORCH_CHECK(k == m2.size(-2), "size mismatch, m1: ", m1.sizes(), " m2: ", m2.sizes());

  memory::data_type data_t;
  if (std::is_same<scalar_t, at::Half>::value) {
    data_t = memory::data_type::f16;
  } else if (std::is_same<scalar_t, at::BFloat16>::value) {
    data_t = memory::data_type::bf16;
  } else {
    data_t = memory::data_type::f32;
  }
  memory::desc m1_md;
  memory::desc m2_md;
  memory::desc r_md;
  if (dims == 2) {
    m1_md = memory::desc({m, k}, data_t, {m1.stride(0), m1.stride(1)});
    m2_md = memory::desc({k, n}, data_t, {m2.stride(0), m2.stride(1)});
    r_md = memory::desc({m, n}, data_t, {result.stride(0), result.stride(1)});
  } else {
    m1_md = memory::desc({mb, m, k}, data_t,
      {m1.stride(0), m1.stride(1), m1.stride(2)});
    m2_md = memory::desc({mb, k, n}, data_t,
      {m2.stride(0), m2.stride(1), m2.stride(2)});
    r_md = memory::desc({mb, m, n}, data_t,
      {result.stride(0), result.stride(1), result.stride(2)});
  }

  primitive_attr attr;
  post_ops po;
  if (alpha != 1.f)
    attr.set_output_scales(/* mask */ 0, {(float)alpha});
  if (beta != 0.f)
    po.append_sum(beta);
  if (with_relu)
    po.append_eltwise(1.f, dnnl::algorithm::eltwise_relu, 0.f, 0.f);;
  attr.set_post_ops(po);

  at::Device curDevice = at::Device(at::kDPCPP, current_device());
  auto engine = GpuEngineManager::Instance().get_engine(curDevice);
  auto strm = GpuStreamManager::Instance().get_stream();

  std::shared_ptr<dnnl::matmul::desc> matmul_desc;
  matmul_desc.reset(new dnnl::matmul::desc(m1_md, m2_md, r_md));
  std::shared_ptr<dnnl::matmul::primitive_desc> matmul_pd;
  matmul_pd.reset(new dnnl::matmul::primitive_desc(*matmul_desc, attr, engine));
  std::shared_ptr<dnnl::matmul> matmul_p;
  matmul_p.reset(new dnnl::matmul(*matmul_pd));

  auto m1_memory = dpcpp_onednn_memory(m1_md, engine, m1.data_ptr());

  auto m2_memory = dpcpp_onednn_memory(m2_md, engine, m2.data_ptr());

  auto r_memory = dpcpp_onednn_memory(r_md, engine, result.data_ptr());


  DPCPP_ONEDNN_EXEC(*matmul_p, strm,
    {{DNNL_ARG_SRC, m1_memory}, {DNNL_ARG_WEIGHTS, m2_memory},
      {DNNL_ARG_DST, r_memory}});

#if defined(USE_COMPUTECPP)
  if (!result_.is_same(result))
    result_.copy_(result);
#endif
}

bool check_broadcast(const Tensor& src, const IntArrayRef& shape){
  auto src_dim = src.dim();
  auto tgt_dim = shape.size();
  if (src_dim == 0 || src_dim > tgt_dim)
    return false;
  do {
    src_dim--;
    tgt_dim--;
    auto size = src.size(src_dim);
    if (size != 1 && size != shape[tgt_dim])
      return false;
  } while(src_dim);
  return true;
}

template <typename scalar_t>
void gemm_broadcast(Tensor& result,
                    const Tensor& m1,
                    const Tensor& m2,
                    scalar_t beta,
                    scalar_t alpha,
                    const Tensor bias = at::Tensor(),
                    bool with_relu = false) {
  std::vector<int64_t> result_shape;
  auto dim = m1.dim();
  if (dim == 2) {
    result_shape = std::vector<int64_t>{m1.size(0), m2.size(1)};
  } else {
    result_shape = std::vector<int64_t>{m1.size(0), m1.size(1), m2.size(2)};
  }
  if (bias.defined() && beta) {
    TORCH_CHECK(check_broadcast(bias, result_shape),
                "bias ", bias.sizes(), " cannot broadcast to ", result_shape);
    Tensor bc_bias;
    std::tie(bc_bias) = expand_size(bias, result_shape, "gemm_broadcast");
    if (!result.is_same(bc_bias))
      result.resize_(bc_bias.sizes()).copy_(bc_bias);
  } else {
    result.resize_(result_shape);
  }

  mkldnnGemmImpl<scalar_t>(result, beta, alpha, m1, m2, with_relu);
}
} // namespace impl

Tensor& addmm_(
    Tensor& self,
    const Tensor& m1,
    const Tensor& m2,
    Scalar beta,
    Scalar alpha) {
  checkBackend("addmm_", {self, m1, m2}, Backend::DPCPP);
  TORCH_CHECK(self.dim() == 2, "expected 2D tensor");
  TORCH_CHECK(m1.dim() == 2, "expected 2D tensor");
  TORCH_CHECK(m2.dim() == 2, "expected 2D tensor");
  TORCH_CHECK(self.size(0) ==  m1.size(0) && self.size(1) == m2.size(1),
              "size mismatch input ", self.sizes(), " m1 ", m1.sizes(), " m2 ", m2.sizes());

  IPEX_DISPATCH_ALL_TYPES_AND2(
    at::ScalarType::Half,
    at::ScalarType::BFloat16,
    self.scalar_type(),
    "addmm",
    [&]() {
      impl::gemm_broadcast<scalar_t>(self, m1, m2, beta.to<scalar_t>(), alpha.to<scalar_t>(), self);
    });

  return self;
}

Tensor addmm(
    const Tensor& input,
    const Tensor& m1,
    const Tensor& m2,
    Scalar beta,
    Scalar alpha) {
  checkBackend("addmm", {input, m1, m2}, Backend::DPCPP);
  TORCH_CHECK(m1.dim() == 2, "expected 2D tensor");
  TORCH_CHECK(m2.dim() == 2, "expected 2D tensor");

  auto result = at::empty({0}, input.options());
  IPEX_DISPATCH_ALL_TYPES_AND2(
    at::ScalarType::Half,
    at::ScalarType::BFloat16,
    result.scalar_type(),
    "addmm",
    [&]() {
      impl::gemm_broadcast<scalar_t>(result, m1, m2, beta.to<scalar_t>(), alpha.to<scalar_t>(), input);
    });

  return result;
}

Tensor& mm_out(Tensor& result, const Tensor& self, const Tensor& mat2) {
  checkBackend("mm_out", {result, self, mat2}, Backend::DPCPP);
  TORCH_CHECK(self.dim() == 2, "expected 2D tensor");
  TORCH_CHECK(mat2.dim() == 2, "expected 2D tensor");

  IPEX_DISPATCH_FLOATING_TYPES_AND2(
    at::ScalarType::Half,
    at::ScalarType::BFloat16,
    self.scalar_type(),
    "mm_out",
    [&]() {
      impl::gemm_broadcast<scalar_t>(result, self, mat2, 0, 1);
    });

  return result;
}

Tensor mm(const Tensor& self, const Tensor& mat2) {
  auto result = at::empty({0}, self.options());
  at::AtenIpexTypeDPCPP::mm_out(result, self, mat2);
  return result;
}

Tensor& baddbmm_(
  Tensor& self,
  const Tensor& batch1,
  const Tensor& batch2,
  Scalar beta,
  Scalar alpha) {
  checkBackend("baddbmm_", {self, batch1, batch2}, Backend::DPCPP);
  TORCH_CHECK(self.dim() == 3, "expected 3D tensor");
  TORCH_CHECK(batch1.dim() == 3, "expected 3D tensor");
  TORCH_CHECK(batch2.dim() == 3, "expected 3D tensor");
  TORCH_CHECK(self.size(0) == batch1.size(0) && \
              self.size(1) == batch1.size(1) && \
              self.size(2) == batch2.size(2),
              "size mismatch input ", self.sizes(),
              " batch1 ", batch1.sizes(), " batch2 ", batch2.sizes());

  IPEX_DISPATCH_FLOATING_TYPES_AND2(
    at::ScalarType::Half,
    at::ScalarType::BFloat16,
    self.scalar_type(),
    "baddbmm_",
    [&]() {
      impl::gemm_broadcast<scalar_t>(self, batch1, batch2, beta.to<scalar_t>(), alpha.to<scalar_t>(), self);
    });

  return self;
}

Tensor& baddbmm_out(
    Tensor& result,
    const Tensor& input,
    const Tensor& batch1,
    const Tensor& batch2,
    Scalar beta,
    Scalar alpha) {
  checkBackend("baddbmm_out", {input, batch1, batch2}, Backend::DPCPP);
  TORCH_CHECK(batch1.dim() == 3, "expected 3D tensor");
  TORCH_CHECK(batch2.dim() == 3, "expected 3D tensor");

  IPEX_DISPATCH_FLOATING_TYPES_AND2(
    at::ScalarType::Half,
    at::ScalarType::BFloat16,
    input.scalar_type(),
    "baddbmm_out",
    [&]() {
      impl::gemm_broadcast<scalar_t>(result, batch1, batch2, beta.to<scalar_t>(), alpha.to<scalar_t>(), input);
    });

  return result;
}

Tensor baddbmm(
    const Tensor& input,
    const Tensor& batch1,
    const Tensor& batch2,
    Scalar beta,
    Scalar alpha) {
  Tensor r = at::empty({0}, input.options());
  at::AtenIpexTypeDPCPP::baddbmm_out(r, input, batch1, batch2, beta, alpha);
  return r;
}

Tensor& bmm_out(Tensor& result, const Tensor& self, const Tensor& batch2) {
  checkBackend("bmm_out", {result, self, batch2}, Backend::DPCPP);
  TORCH_CHECK(self.dim() == 3, "expected 3D tensor");
  TORCH_CHECK(batch2.dim() == 3, "expected 3D tensor");

  IPEX_DISPATCH_FLOATING_TYPES_AND2(
    at::ScalarType::Half,
    at::ScalarType::BFloat16,
    self.scalar_type(),
    "bmm_out",
    [&]() {
      impl::gemm_broadcast<scalar_t>(result, self, batch2, 0, 1);
    });
  return result;
}

Tensor bmm(const Tensor& self, const Tensor& batch2) {
  auto result = at::empty({0}, self.options());
  at::AtenIpexTypeDPCPP::bmm_out(result, self, batch2);
  return result;
}

// FIXME: should not be here
Tensor linear_relu(const Tensor & input, const Tensor & weight, const Tensor & bias) {
  RECORD_FUNCTION("linear_relu",
                  std::vector<c10::IValue>({input, weight, bias}));
  if (input.dim() == 2 && bias.defined()) {
    // Fused op is marginally faster.
    checkBackend("linear_relu", {input, weight, bias}, Backend::DPCPP);
    TORCH_CHECK(input.dim() == 2, "expected 2D tensor");
    TORCH_CHECK(weight.dim() == 2, "expected 2D tensor");

    auto result = at::empty({0}, input.options());
    IPEX_DISPATCH_ALL_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      result.scalar_type(),
      "linear_relu",
      [&]() {
        impl::gemm_broadcast<scalar_t>(result, input, weight.t(), 1.f, 1.f, bias, true);
      });

    return result;
  }

  auto output = at::matmul(input, weight.t());
  if (bias.defined()) {
    output.add_(bias);
  }
  return at::relu(output);
}

} // namespace AtenIpexTypeDPCPP
} // namespace at
