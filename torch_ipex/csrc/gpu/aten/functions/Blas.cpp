#include <ATen/ExpandUtils.h>
#include <ATen/Dispatch.h>

#include <core/TensorImplUtils.h>
#include <functions/Resize.h>
#include <dnnl/InnerProduct.hpp>

#define ERROR_ONLY_FP_TYPES(func) \
  AT_ERROR(#func, "for SYCL tensors only supports floating-point types. Try converting the tensors with .float()");

namespace at {
namespace native {

template <typename scalar_t>
void mkldnnGemmImpl(Tensor& r_, scalar_t beta,
    scalar_t alpha, const Tensor& _m1, const Tensor& _m2) {
  char transpose_r, transpose_m1, transpose_m2;

  #define TRANSPOSE_TRUE    't'
  #define TRANSPOSE_FALSE   'n'
  // n == 1 || ldc >= max(1, m)
  #define Max(X, Y)  ((X) > (Y) ? (X) : (Y))
  #define LDC_COND(M, N, LDC) ((N) == 1 || (LDC) >= Max(1, M))

  Tensor m1 = _m1, m2 = _m2;

  printf("[1] result stride(0) %d stirde(1) %d shape(0) %d shape (1) %d\n",
      r_.stride(0), r_.stride(1), r_.size(0), r_.size(0));

  /* r_ */
  if(r_.stride(0) == 1 &&
     LDC_COND(r_.size(0), r_.size(1), r_.stride(1)))
  {
    // if column major, no swap, no transpose
    m1 = _m2;
    m2 = _m1;
    // Tensor swap = _m2;
    // m2 = _m1;
    // m1 = swap;
    transpose_r = TRANSPOSE_TRUE;
  }
  else if(r_.stride(1) == 1 &&
          LDC_COND(r_.size(1), r_.size(0), r_.stride(0)))
  {
    // if row majoar
    transpose_r = TRANSPOSE_FALSE;
  }
  else
  {
    // make r_ FORTRAN contiguous
    AT_ERROR("THSYCL addmm r unsupported transpose");
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
  if(m1.stride(transpose_size0) == 1 &&
     m1.stride(transpose_size1) >= Max(1, m))
  {
    // column major
    transpose_m1 = TRANSPOSE_TRUE;
  }
  else if(m1.stride(transpose_size1) == 1 &&
          m1.stride(transpose_size0) >= Max(1, k))
  {
    // row major
    transpose_m1 = TRANSPOSE_FALSE;
  }
  else
  {
    AT_ERROR("THSYCL addmm m1 unsupported transpose");
  }

  /* m2 */
  /* Need ldm2_ >= max(1, (transpose_m2 == 'n' ? k : n)) */
  if(m2.stride(transpose_size0) == 1 &&
     m2.stride(transpose_size1) >= Max(1, k))
  {
    // column major
    transpose_m2 = TRANSPOSE_TRUE;
  }
  else if(m2.stride(transpose_size1) == 1 &&
          m2.stride(transpose_size0) >= Max(1, n))
  {
    // row major
    transpose_m2 = TRANSPOSE_FALSE;
  }
  else
  {
    AT_ERROR("THSYCL addmm m2 unsupported transpose");
  }

  int64_t ldm1 = (transpose_m1 == TRANSPOSE_TRUE ? m1.size(transpose_size0) :
                                                    m1.size(transpose_size1));
  int64_t ldm2 = (transpose_m2 == TRANSPOSE_TRUE ? m2.size(transpose_size0) :
                                                    m2.size(transpose_size1));

  auto& sycl_queue = c10::sycl::getCurrentSYCLStream().sycl_queue();
#if defined(THSYCL_REAL_IS_HALF)
  auto m1_sb = c10::sycl::syclGetBufferMap().template get_buffer<cl::sycl::half>(
      m1.data<scalar_t>());
  auto m2_sb = c10::sycl::syclGetBufferMap().template get_buffer<cl::sycl::half>(
      m2.data<scalar_t>());
  auto r_sb = c10::sycl::syclGetBufferMap().template get_buffer<cl::sycl::half>(
      r_.data<scalar_t>());
#else
  auto m1_sb = c10::sycl::syclGetBufferMap().template get_buffer<float>(
      m1.data<scalar_t>());
  auto m2_sb = c10::sycl::syclGetBufferMap().template get_buffer<float>(
      m2.data<scalar_t>());
  auto r_sb = c10::sycl::syclGetBufferMap().template get_buffer<float>(
      r_.data<scalar_t>());
#endif
  // assume dnnl_notrans = 0 & dnnl_trans = 1
  auto transpose_m1_ = transpose_m1 == TRANSPOSE_FALSE ? 'N' : 'T';
  auto transpose_m2_ = transpose_m2 == TRANSPOSE_FALSE ? 'N' : 'T';

  // Reference from THBlas_(gemm)
  // Fix mkl-dnn generic_gemm type check failure
  if (n == 1)
    ldr = m;

  // This is a work-around synchronization because we suspect that mkldnn gemm didn't
  // handle Dependency very well.
  sycl_queue.wait();

  mkldnn::gemm(sycl_queue, transpose_m1_,transpose_m2_, m, n, k, alpha,
               m1_sb, 0, ldm1, m2_sb, 0, ldm2, beta, r_sb, 0, ldr);
  #undef TRANSPOSE_TRUE
  #undef TRANSPOSE_FALSE
  #undef Max
}

template <typename scalar_t>
void addmm(Tensor& r_, scalar_t beta,
    Tensor& t, scalar_t alpha, const Tensor& m1, const Tensor& m2) {
  if ( (m1.dim() != 2) || (m2.dim() != 2) )
    AT_ERROR("2D tensors expected, got ", m1.dim(), "D, ", m2.dim(), "D tensors");

  if (t.dim() != 2)
    AT_ERROR("2D tensor expected, got ", t.dim(), "D tensor for t");

  if (m1.size(1) != m2.size(0)) {
    SYCLDescBuff bm1 = TensorImpl_sizeDesc(m1.unsafeGetTensorImpl());
    SYCLDescBuff bm2 = TensorImpl_sizeDesc(m2.unsafeGetTensorImpl());
    AT_ERROR("size mismatch, m1: ", bm1.str, " m2: ", bm2.str);
  }

  if( (t.size(0) != m1.size(0)) || (t.size(1) != m2.size(1)) ) {
    SYCLDescBuff bt  = TensorImpl_sizeDesc(t.unsafeGetTensorImpl());
    SYCLDescBuff bm1 = TensorImpl_sizeDesc(m1.unsafeGetTensorImpl());
    SYCLDescBuff bm2 = TensorImpl_sizeDesc(m2.unsafeGetTensorImpl());
    AT_ERROR("size mismatch, t:", bt.str, " m1: ", bm1.str, " m2: ", bm2.str);
  }

  if (&t != &r_) {
    at::native::resize_as_sycl_(r_, t);
    if (beta != 0.0) {
      r_.copy_(t);
    }
  }

  mkldnnGemmImpl(r_, beta, alpha, m1, m2);
}

#ifdef UTILS_PORTED
static std::vector<at::Tensor>
initTensorArray(const std::vector<at::Tensor> &tensors) {
  int numt = tensors.size();
  std::vector<at::Tensor> _tensors;

  for (int i = 0; i < numt; i++) {
    auto tmp = at::AtenIpexTypeDPCPP::empty({0}, tensors[i].options());
    at::native::resize_as_sycl_(tmp, tensors[i]);
    tmp.copy_(tensor[i]);
    _tensors.push_back(tmp);
  }

  return _tensors;
}

static std::vector<at::Tensor>
squeeze1dTensorArray(std::vector<at::Tensor> &tensors) {
  std::vector<at::Tensor> squeezed;
  for (int i = 0; i < tensors.size(); i++)
    squeezed.push_back(at::squeeze_(tensors[i], 0));
  return squeezed;
}

static std::vector<at::Tensor>
unsqueeze1dTensorArray(std::vector<at::Tensor> &tensors, int numel) {
  std::vector<at::Tensor> unsqueezed;
  for (int i = 0; i < numel; i++)
    unsqueezed.push_back(at::unsqueeze1d(tensors[i], tensors[i], 0));
  return unsqueezed;
}

void baddbmm(Tensor *result, scalar_t beta,
    Tensor *t, scalar_t alpha, Tensor *batch1, Tensor *batch2)
{
#if defined(THSYCL_REAL_IS_FLOAT)
  THSYCLAssertSameGPU(THSYCLTensor_(checkGPU)(state, 4, result, t, batch1, batch2));
  THArgCheck(THSYCLTensor_(nDimensionLegacyNoScalars)(state, t) == 3, 4, "expected 3D tensor");
  THArgCheck(THSYCLTensor_(nDimensionLegacyNoScalars)(state, batch1) == 3, 6, "expected 3D tensor");
  THArgCheck(THSYCLTensor_(nDimensionLegacyNoScalars)(state, batch2) == 3, 7, "expected 3D tensor");
  THArgCheck(THSYCLTensor_(size)(state, t, 0) == THSYCLTensor_(size)(state, batch1, 0), 6,
             "equal number of batches expected");
  THArgCheck(THSYCLTensor_(size)(state, t, 0) == THSYCLTensor_(size)(state, batch2, 0), 7,
             "equal number of batches expected");
  THArgCheck(THSYCLTensor_(size)(state, t, 1) == THSYCLTensor_(size)(state, batch1, 1), 6,
             "wrong matrix size");
  THArgCheck(THSYCLTensor_(size)(state, t, 2) == THSYCLTensor_(size)(state, batch2, 2), 7,
             "wrong matrix size");
  THArgCheck(THSYCLTensor_(size)(state, batch1, 2) == THSYCLTensor_(size)(state, batch2, 1), 6,
             "wrong matrix size");

  if (t != result) {
    at::native::resize_as_sycl_(result, t);
    if (beta != 0.0) {
      result.copy(t);
    }
  }

  // TODO: This is the work-around implementation for BatchGemm. We should
  // replace it when Blas library is available.
  auto num_batches = result->size(0);

  // First split t, batch1, batch2 into chunks along 0 dim
  auto ts = at::chunk(t, num_batches, 0);
  auto b1s = at::chunk(batch1, num_batches, 0);
  auto b2s = at::chunk(batch2, num_batches, 0);

  // Initiliaze Tensor array and init value from tensor vector
  auto _ts = initTensorArray(ts);
  auto _b1s = initTensorArray(b1s);
  auto _b2s = initTensorArray(b2s);

  // Squeeze tensor in tensor array along 0 dim
  auto _ts_squeezed = squeeze1dTensorArray(_ts, num_batches);
  auto _b1s_squeezed = squeeze1dTensorArray(_b1s, num_batches);
  auto _b2s_squeezed = squeeze1dTensorArray(_b2s, num_batches);

  // Use GEMM to do th computation
  for (int i = 0; i < num_batches; i++)
    mkldnnGemmImpl(_ts_squeezed[i], beta, alpha, _b1s_squeezed[i], _b2s_squeezed[i]);

  // Unsqueeze t tensor array and concat to result array along 0 dim
  auto _ts_unsqueezed = unsqueeze1dTensorArray(_ts_squeezed, num_batches);
  at::catArray(result, _ts_unsqueezed, num_batches, 0);
#else
  ERROR_ONLY_FP_TYPES("baddbmm");
#endif
}
#endif

} // namespace native

namespace AtenIpexTypeDPCPP {

Tensor addmm(const Tensor & self, const Tensor & mat1,
    const Tensor & mat2, Scalar beta, Scalar alpha) {
  Tensor b_self;
  std::tie(b_self) = expand_size(self, {mat1.size(0), mat2.size(1)}, "addmm_out");
  Tensor r = at::empty({0}, self.options());

  AT_DISPATCH_ALL_TYPES(self.scalar_type(), "addmm_out",
      [&]() {
        at::native::addmm<scalar_t>(
            r, beta.to<float>(), b_self, alpha.to<float>(), mat1, mat2);
      }
  );

  return r;
}
Tensor baddbmm(const Tensor & self, const Tensor & batch1, const Tensor & batch2, Scalar beta, Scalar alpha) {
  Tensor b_self;
  std::tie(b_self) = expand_size(self, {batch1.size(0), batch2.size(1)}, "baddbmm_out");
  Tensor r = at::empty({0}, self.options());

  AT_DISPATCH_ALL_TYPES(self.scalar_type(), "baddbmm_out",
      [&]() {
        at::native::baddbmm<scalar_t>(
            r, beta.to<float>(), b_self, alpha.to<float>(), batch1, batch2);
      }
  );

  return r;
}
}

} // namespace at
