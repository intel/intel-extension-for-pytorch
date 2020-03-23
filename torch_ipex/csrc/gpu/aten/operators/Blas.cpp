#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/ExpandUtils.h>

#include <core/TensorImplUtils.h>
#include <dnnl/InnerProduct.hpp>

#define ERROR_ONLY_FP_TYPES(func)                                  \
  AT_ERROR(                                                        \
      #func,                                                       \
      "for DPCPP tensors only supports floating-point types. Try " \
      "converting the tensors with .float()");

using namespace dnnl;
using namespace at::dpcpp;

namespace at {
namespace AtenIpexTypeDPCPP {
namespace impl {

template <typename scalar_t>
void mkldnnGemmImpl(
    Tensor& r_,
    scalar_t beta,
    scalar_t alpha,
    const Tensor& _m1,
    const Tensor& _m2) {
  char transpose_r, transpose_m1, transpose_m2;

#define TRANSPOSE_TRUE 't'
#define TRANSPOSE_FALSE 'n'
// n == 1 || ldc >= max(1, m)
#define Max(X, Y) ((X) > (Y) ? (X) : (Y))
#define LDC_COND(M, N, LDC) ((N) == 1 || (LDC) >= Max(1, M))

  Tensor m1 = _m1, m2 = _m2;

  /* r_ */
  if (r_.stride(0) == 1 && LDC_COND(r_.size(0), r_.size(1), r_.stride(1))) {
    // if column major, no swap, no transpose
    m1 = _m2;
    m2 = _m1;
    // Tensor swap = _m2;
    // m2 = _m1;
    // m1 = swap;
    transpose_r = TRANSPOSE_TRUE;
  } else if (
      r_.stride(1) == 1 && LDC_COND(r_.size(1), r_.size(0), r_.stride(0))) {
    // if row majoar
    transpose_r = TRANSPOSE_FALSE;
  } else {
    // make r_ FORTRAN contiguous
    AT_ERROR("THDPCPP addmm r unsupported transpose");
  }

#undef LDC_COND

  int64_t transpose_size0 = (transpose_r == TRANSPOSE_FALSE ? 0 : 1);
  int64_t transpose_size1 = (transpose_r == TRANSPOSE_FALSE ? 1 : 0);
  int64_t m = r_.size(transpose_size0);
  int64_t n = r_.size(transpose_size1);
  int64_t k = m1.size(transpose_size1);
  int64_t ldr = r_.size(transpose_size1);

  /* m1 */
  /* Need ldm1_ >= max(1, (transpose_m1 == 'n' ? m : k)) */
  if (m1.stride(transpose_size0) == 1 &&
      m1.stride(transpose_size1) >= Max(1, m)) {
    // column major
    transpose_m1 = TRANSPOSE_TRUE;
  } else if (
      m1.stride(transpose_size1) == 1 &&
      m1.stride(transpose_size0) >= Max(1, k)) {
    // row major
    transpose_m1 = TRANSPOSE_FALSE;
  } else {
    AT_ERROR("THDPCPP addmm m1 unsupported transpose");
  }

  /* m2 */
  /* Need ldm2_ >= max(1, (transpose_m2 == 'n' ? k : n)) */
  if (m2.stride(transpose_size0) == 1 &&
      m2.stride(transpose_size1) >= Max(1, k)) {
    // column major
    transpose_m2 = TRANSPOSE_TRUE;
  } else if (
      m2.stride(transpose_size1) == 1 &&
      m2.stride(transpose_size0) >= Max(1, n)) {
    // row major
    transpose_m2 = TRANSPOSE_FALSE;
  } else {
    AT_ERROR("THDPCPP addmm m2 unsupported transpose");
  }

  int64_t ldm1 =
      (transpose_m1 == TRANSPOSE_TRUE ? m1.size(transpose_size0)
                                      : m1.size(transpose_size1));
  int64_t ldm2 =
      (transpose_m2 == TRANSPOSE_TRUE ? m2.size(transpose_size0)
                                      : m2.size(transpose_size1));

  // assume dnnl_notrans = 0 & dnnl_trans = 1
  auto transpose_m1_ = transpose_m1 == TRANSPOSE_FALSE ? 'N' : 'T';
  auto transpose_m2_ = transpose_m2 == TRANSPOSE_FALSE ? 'N' : 'T';

  // Reference from THBlas_(gemm)
  // Fix mkl-dnn generic_gemm type check failure
  if (n == 1)
    ldr = m;

  at::Device curDevice = at::Device(at::kDPCPP, current_device());
  auto engine = GpuEngineManager::Instance().get_engine(curDevice);
  auto strm = GpuStreamManager::Instance().get_stream();

  memory::dims m1_strides =
      transpose_m1_ == 'N' ? memory::dims{ldm1, 1} : memory::dims{1, ldm1};
  memory::dims m2_strides =
      transpose_m2_ == 'N' ? memory::dims{ldm2, 1} : memory::dims{1, ldm2};

  memory::data_type data_t;
  if (std::is_same<scalar_t, at::Half>::value) {
    data_t = memory::data_type::f16;
  } else if (std::is_same<scalar_t, at::BFloat16>::value) {
    data_t = memory::data_type::bf16;
  } else {
    data_t = memory::data_type::f32;
  }
  memory::desc m1_md({m, k}, data_t, m1_strides);
  memory::desc m2_md({k, n}, data_t, m2_strides);
  memory::desc r_md({m, n}, data_t, {ldr, 1});

  primitive_attr attr;
  if (alpha != 1.f)
    attr.set_output_scales(/* mask */ 0, {(float)alpha});
  if (beta != 0.f) {
    post_ops po;
    po.append_sum(beta);
    attr.set_post_ops(po);
  }

  std::shared_ptr<dnnl::matmul::desc> matmul_desc;
  matmul_desc.reset(new dnnl::matmul::desc(m1_md, m2_md, r_md));
  std::shared_ptr<dnnl::matmul::primitive_desc> matmul_pd;
  matmul_pd.reset(new dnnl::matmul::primitive_desc(*matmul_desc, attr, engine));
  std::shared_ptr<dnnl::matmul> matmul_p;
  matmul_p.reset(new dnnl::matmul(*matmul_pd));

  auto m1_memory = memory({m1_md, engine});
  dpcpp_set_mkldnn_buffer(m1.data_ptr(), m1_memory);

  auto m2_memory = memory({m2_md, engine});
  dpcpp_set_mkldnn_buffer(m2.data_ptr(), m2_memory);

  auto r_memory = memory({r_md, engine});
  dpcpp_set_mkldnn_buffer(r_.data_ptr(), r_memory);

  matmul_p->execute(
      strm,
      {{DNNL_ARG_SRC, m1_memory},
       {DNNL_ARG_WEIGHTS, m2_memory},
       {DNNL_ARG_DST, r_memory}});
  strm.wait();

#undef TRANSPOSE_TRUE
#undef TRANSPOSE_FALSE
}

template <typename scalar_t>
void addmm(
    Tensor& r_,
    scalar_t beta,
    Tensor& t,
    scalar_t alpha,
    const Tensor& m1,
    const Tensor& m2) {
  if ((m1.dim() != 2) || (m2.dim() != 2))
    AT_ERROR(
        "2D tensors expected, got ", m1.dim(), "D, ", m2.dim(), "D tensors");

  if (t.dim() != 2)
    AT_ERROR("2D tensor expected, got ", t.dim(), "D tensor for t");

  if (m1.size(1) != m2.size(0)) {
    DPCPPDescBuff bm1 = TensorImpl_sizeDesc(m1.unsafeGetTensorImpl());
    DPCPPDescBuff bm2 = TensorImpl_sizeDesc(m2.unsafeGetTensorImpl());
    AT_ERROR("size mismatch, m1: ", bm1.str, " m2: ", bm2.str);
  }

  if ((t.size(0) != m1.size(0)) || (t.size(1) != m2.size(1))) {
    DPCPPDescBuff bt = TensorImpl_sizeDesc(t.unsafeGetTensorImpl());
    DPCPPDescBuff bm1 = TensorImpl_sizeDesc(m1.unsafeGetTensorImpl());
    DPCPPDescBuff bm2 = TensorImpl_sizeDesc(m2.unsafeGetTensorImpl());
    AT_ERROR("size mismatch, t:", bt.str, " m1: ", bm1.str, " m2: ", bm2.str);
  }

  if (TensorImpl_Unwrap(t) != TensorImpl_Unwrap(r_)) {
    r_.resize_as_(t);
    if (beta != 0.0) {
      r_.copy_(t);
    }
  }

  mkldnnGemmImpl(r_, beta, alpha, m1, m2);
}

static std::vector<at::Tensor> initTensorArray(
    const std::vector<at::Tensor>& tensors) {
  int numt = tensors.size();
  std::vector<at::Tensor> _tensors;

  for (int i = 0; i < numt; i++) {
    Tensor tmp = at::empty({0}, tensors[i].options());
    tmp.resize_as_(tensors[i]);
    tmp.copy_(tensors[i]);
    _tensors.push_back(tmp);
  }

  return _tensors;
}

static std::vector<at::Tensor> squeeze1dTensorArray(
    std::vector<at::Tensor>& tensors) {
  std::vector<at::Tensor> squeezed;
  for (int i = 0; i < tensors.size(); i++)
    squeezed.push_back(at::squeeze(tensors[i], 0));
  return squeezed;
}

static std::vector<at::Tensor> unsqueeze1dTensorArray(
    std::vector<at::Tensor>& tensors) {
  std::vector<at::Tensor> unsqueezed;
  for (int i = 0; i < tensors.size(); i++)
    unsqueezed.push_back(at::unsqueeze(tensors[i], 0));
  return unsqueezed;
}

template <typename scalar_t>
void baddbmm(
    Tensor& result,
    scalar_t beta,
    const Tensor& t,
    scalar_t alpha,
    const Tensor& batch1,
    const Tensor& batch2) {
  checkBackend("baddbmm", {result, t, batch1, batch2}, Backend::DPCPP);
  TORCH_CHECK(t.dim() == 3, "expected 3D tensor");
  TORCH_CHECK(batch1.dim() == 3, "expected 3D tensor");
  TORCH_CHECK(batch2.dim() == 3, "expected 3D tensor");
  TORCH_CHECK(t.size(0) == batch1.size(0), "equal number of batches expected");
  TORCH_CHECK(t.size(0) == batch2.size(0), "equal number of batches expected");
  TORCH_CHECK(t.size(1) == batch1.size(1), "wrong matrix size");
  TORCH_CHECK(t.size(2) == batch2.size(2), "wrong matrix size");
  TORCH_CHECK(batch1.size(2) == batch2.size(1), "wrong matrix size");

  if (TensorImpl_Unwrap(t) != TensorImpl_Unwrap(result)) {
    result.resize_as_(t);
    if (beta != 0.0) {
      result.copy_(t);
    }
  }

  // TODO: This is the work-around implementation for BatchGemm. We should
  // replace it when Blas library is available.
  auto num_batches = result.size(0);

  // First split t, batch1, batch2 into chunks along 0 dim
  auto ts = at::chunk(t, num_batches, 0);
  auto b1s = at::chunk(batch1, num_batches, 0);
  auto b2s = at::chunk(batch2, num_batches, 0);

  // Initiliaze Tensor array and init value from tensor vector
  auto _ts = initTensorArray(ts);
  auto _b1s = initTensorArray(b1s);
  auto _b2s = initTensorArray(b2s);

  // Squeeze tensor in tensor array along 0 dim
  auto _ts_squeezed = squeeze1dTensorArray(_ts);
  auto _b1s_squeezed = squeeze1dTensorArray(_b1s);
  auto _b2s_squeezed = squeeze1dTensorArray(_b2s);

  // Use GEMM to do th computation
  for (int i = 0; i < num_batches; i++)
    mkldnnGemmImpl(
        _ts_squeezed[i], beta, alpha, _b1s_squeezed[i], _b2s_squeezed[i]);

  // Unsqueeze t tensor array and concat to result array along 0 dim
  auto _ts_unsqueezed = unsqueeze1dTensorArray(_ts_squeezed);
  at::cat_out(result, _ts_unsqueezed, 0);
}

} // namespace impl

Tensor addmm(
    const Tensor& self,
    const Tensor& mat1,
    const Tensor& mat2,
    Scalar beta,
    Scalar alpha) {
  Tensor b_self;
  std::tie(b_self) =
      expand_size(self, {mat1.size(0), mat2.size(1)}, "addmm_out");
  Tensor r = at::empty({0}, self.options());

  AT_DISPATCH_ALL_TYPES(self.scalar_type(), "addmm_out", [&]() {
    impl::addmm<scalar_t>(
        r, beta.to<float>(), b_self, alpha.to<float>(), mat1, mat2);
  });

  return r;
}

Tensor& mm_out(Tensor& result, const Tensor& self, const Tensor& mat2) {
  AT_DISPATCH_ALL_TYPES(self.scalar_type(), "mm_out", [&]() {
    impl::addmm<scalar_t>(result, scalar_t(0), result, scalar_t(1), self, mat2);
  });

  return result;
}

Tensor mm(const Tensor& self, const Tensor& mat2) {
  auto result = at::empty({0}, self.options());
  result.resize_({self.size(0), mat2.size(1)});

  return at::AtenIpexTypeDPCPP::mm_out(result, self, mat2);
}

Tensor& baddbmm_out(
    Tensor& result,
    const Tensor& self,
    const Tensor& batch1,
    const Tensor& batch2,
    Scalar beta,
    Scalar alpha) {
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(self.scalar_type(), "baddbmm", [&]() {
    impl::baddbmm<scalar_t>(
        result,
        beta.to<scalar_t>(),
        self,
        alpha.to<scalar_t>(),
        batch1,
        batch2);
  });
  return result;
}

Tensor baddbmm(
    const Tensor& self,
    const Tensor& batch1,
    const Tensor& batch2,
    Scalar beta,
    Scalar alpha) {
  auto result = at::empty({0}, self.options());
  return at::AtenIpexTypeDPCPP::baddbmm_out(
      result, self, batch1, batch2, beta, alpha);
}

Tensor& baddbmm_(
    Tensor& self,
    const Tensor& batch1,
    const Tensor& batch2,
    Scalar beta,
    Scalar alpha) {
  return at::AtenIpexTypeDPCPP::baddbmm_out(
      self, self, batch1, batch2, beta, alpha);
}

Tensor bmm(const Tensor& self, const Tensor& batch2) {
  auto result =
      at::empty({self.size(0), self.size(1), batch2.size(2)}, self.options());
  return at::AtenIpexTypeDPCPP::baddbmm_out(result, result, self, batch2, 0, 1);
}

Tensor& bmm_out(Tensor& result, const Tensor& self, const Tensor& batch2) {
  result.resize_({self.size(0), self.size(1), batch2.size(2)});
  return at::AtenIpexTypeDPCPP::baddbmm_out(result, result, self, batch2, 0, 1);
}

} // namespace AtenIpexTypeDPCPP
} // namespace at
