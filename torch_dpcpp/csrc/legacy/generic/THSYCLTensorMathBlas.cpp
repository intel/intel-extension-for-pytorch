#ifndef THSYCL_GENERIC_FILE
#define THSYCL_GENERIC_FILE "THDP/generic/THSYCLTensorMathBlas.cpp"
#else

#include <ATen/native/mkldnn/InnerProduct.hpp>

#define ERROR_ONLY_FP_TYPES(func) \
  THError("%s for SYCL tensors only supports floating-point types. Try converting the tensors with .float()", func);

#define ERROR_ONLY_LIENAR_ALG(func) \
  THError("%s for SYCL tensors only supports alpha = 1.0, beta = 0.0.", func);

void THSYCLTensor_(addmv)(THSYCLState* state, THSYCLTensor* r_, scalar_t beta, THSYCLTensor* t, scalar_t alpha, THSYCLTensor* mat, THSYCLTensor* vec) {
  AT_ERROR("not implemented THSYCLTensor_addmv\n");
}

void THSYCLTensor_(addr)(THSYCLState* state, THSYCLTensor* r_, scalar_t beta, THSYCLTensor* t, scalar_t alpha, THSYCLTensor* vec1, THSYCLTensor* vec2) {
  AT_ERROR("not implemented THSYCLTensor_addr\n");
}

#if defined(THSYCL_REAL_IS_FLOAT) || defined(THSYCL_REAL_IS_HALF)
void THSYCLTensor_(mkldnnGemmImpl)(THSYCLTensor *r_, scalar_t beta, scalar_t alpha, THSYCLTensor *m1, THSYCLTensor *m2) {
  char transpose_r, transpose_m1, transpose_m2;

  #define TRANSPOSE_TRUE    't'
  #define TRANSPOSE_FALSE   'n'
  // n == 1 || ldc >= max(1, m)
  #define LDC_COND(M, N, LDC) ((N) == 1 || (LDC) >= THMax(1, M))

  /* r_ */
  if(r_->stride(0) == 1 &&
     LDC_COND(r_->size(0), r_->size(1), r_->stride(1)))
  {
    // if column major, no swap, no transpose
    THSYCLTensor *swap = m2;
    m2 = m1;
    m1 = swap;
    transpose_r = TRANSPOSE_TRUE;
  }
  else if(r_->stride(1) == 1 &&
          LDC_COND(r_->size(1), r_->size(0), r_->stride(0)))
  {
    // if row majoar
    transpose_r = TRANSPOSE_FALSE;
  }
  else
  {
    // make r_ FORTRAN contiguous
    THError("THSYCL addmm r unsupported transpose");
  }

  #undef LDC_COND

  int64_t transpose_size0 = (transpose_r == TRANSPOSE_FALSE ? 0 : 1);
  int64_t transpose_size1 = (transpose_r == TRANSPOSE_FALSE ? 1 : 0);
  int64_t m = r_->size(transpose_size0);
  int64_t n = r_->size(transpose_size1);
  int64_t k = m1->size(transpose_size1);
  int64_t ldr = r_->size(transpose_size1);

  /* m1 */
  /* Need ldm1_ >= max(1, (transpose_m1 == 'n' ? m : k)) */
  if(m1->stride(transpose_size0) == 1 &&
     m1->stride(transpose_size1) >= THMax(1, m))
  {
    // column major
    transpose_m1 = TRANSPOSE_TRUE;
  }
  else if(m1->stride(transpose_size1) == 1 &&
          m1->stride(transpose_size0) >= THMax(1, k))
  {
    // row major
    transpose_m1 = TRANSPOSE_FALSE;
  }
  else
  {
    THError("THSYCL addmm m1 unsupported transpose");
  }

  /* m2 */
  /* Need ldm2_ >= max(1, (transpose_m2 == 'n' ? k : n)) */
  if(m2->stride(transpose_size0) == 1 &&
     m2->stride(transpose_size1) >= THMax(1, k))
  {
    // column major
    transpose_m2 = TRANSPOSE_TRUE;
  }
  else if(m2->stride(transpose_size1) == 1 &&
          m2->stride(transpose_size0) >= THMax(1, n))
  {
    // row major
    transpose_m2 = TRANSPOSE_FALSE;
  }
  else
  {
    THError("THSYCL addmm m2 unsupported transpose");
  }

  int64_t ldm1 = (transpose_m1 == TRANSPOSE_TRUE ? m1->size(transpose_size0) : 
                                                    m1->size(transpose_size1));
  int64_t ldm2 = (transpose_m2 == TRANSPOSE_TRUE ? m2->size(transpose_size0) : 
                                                    m2->size(transpose_size1));

  auto& sycl_queue = c10::sycl::getCurrentSYCLStream().sycl_queue();
#if defined(THSYCL_REAL_IS_HALF)
  auto m1_sb = c10::sycl::syclGetBufferMap().template get_buffer<cl::sycl::half>(
      m1->data<scalar_t>());
  auto m2_sb = c10::sycl::syclGetBufferMap().template get_buffer<cl::sycl::half>(
      m2->data<scalar_t>());
  auto r_sb = c10::sycl::syclGetBufferMap().template get_buffer<cl::sycl::half>(
      r_->data<scalar_t>());
#else
  auto m1_sb = c10::sycl::syclGetBufferMap().template get_buffer<float>(
      m1->data<scalar_t>());
  auto m2_sb = c10::sycl::syclGetBufferMap().template get_buffer<float>(
      m2->data<scalar_t>());
  auto r_sb = c10::sycl::syclGetBufferMap().template get_buffer<float>(
      r_->data<scalar_t>());
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
}
#endif

void THSYCLTensor_(addmm)(THSYCLState *state, THSYCLTensor *r_, scalar_t beta, THSYCLTensor *t, scalar_t alpha, THSYCLTensor *m1, THSYCLTensor *m2) {
#if defined(THSYCL_REAL_IS_FLOAT) || defined(THSYCL_REAL_IS_HALF)
  THSYCLTensor *r__;

  if ( (m1->dim() != 2) || (m2->dim() != 2) )
    THError("2D tensors expected, got %dD, %dD tensors", m1->dim(), m2->dim());

  if (t->dim() != 2)
    THError("2D tensor expected, got %dD tensor for t", t->dim());

  if (m1->size(1) != m2->size(0)) {
    THSYCLDescBuff bm1 = THSYCLTensor_(sizeDesc)(state, m1);
    THSYCLDescBuff bm2 = THSYCLTensor_(sizeDesc)(state, m2);
    THError("size mismatch, m1: %s, m2: %s", bm1.str, bm2.str);
  }

  if( (t->size(0) != m1->size(0)) || (t->size(1) != m2->size(1)) ) {
    THSYCLDescBuff bt  = THSYCLTensor_(sizeDesc)(state, t);
    THSYCLDescBuff bm1 = THSYCLTensor_(sizeDesc)(state, m1);
    THSYCLDescBuff bm2 = THSYCLTensor_(sizeDesc)(state, m2);
    THError("size mismatch, t: %s, m1: %s, m2: %s", bt.str, bm1.str, bm2.str);
  }

  if (t != r_)
  {
    THSYCLTensor_(resizeAs)(state, r_, t);
    THSYCLTensor_(copy)(state, r_, t);
  }

  r__ = r_;

  THSYCLTensor_(mkldnnGemmImpl)(r_, beta, alpha, m1, m2);

  if (r__ != r_) {
    // could not be here
    THSYCLTensor_(freeCopyTo)(state, r__, r_);
  }

#else
  ERROR_ONLY_FP_TYPES("addmm");
#endif
}

void THSYCLTensor_(addbmm)(THSYCLState* state, THSYCLTensor* result, scalar_t beta,
    THSYCLTensor* t, scalar_t alpha, THSYCLTensor* batch1, THSYCLTensor* batch2) {
  AT_ERROR("not implemented THSYCLTensor_addbmm\n");
}

#if defined(THSYCL_REAL_IS_FLOAT)
static THSYCLTensor ** THSYCLTensor_(initTensorArray)(THSYCLState *state, const std::vector<at::Tensor> &t_vec) {
  int numel = t_vec.size();
  THSYCLTensor **t_arr = (THSYCLTensor **)malloc(numel * sizeof(THSYCLTensor *));
  for (int i = 0; i < numel; i++) {
    t_arr[i] = THSYCLTensor_(new)(state);
    THSYCLTensor_(resizeAs)(state, t_arr[i], t_vec[i].unsafeGetTensorImpl());
    THSYCLTensor_(copy)(state, t_arr[i], t_vec[i].unsafeGetTensorImpl());
  }
  return t_arr; 
}

static void THSYCLTensor_(finiTensorArray)(THSYCLState *state, THSYCLTensor **t_arr, int numel) {
  for (int i = 0; i < numel; i++) {
    THSYCLTensor_(free)(state, t_arr[i]);
  }
  free(t_arr);
}

static void THSYCLTensor_(squeeze1dTensorArray)(THSYCLState *state, THSYCLTensor **t_arr, int numel) {
  for (int i = 0; i < numel; i++) {
    THSYCLTensor_(squeeze1d)(state, t_arr[i], t_arr[i], 0);
  }
}

static void THSYCLTensor_(unsqueeze1dTensorArray)(THSYCLState *state, THSYCLTensor **t_arr, int numel) {
  for (int i = 0; i < numel; i++) {
    THSYCLTensor_(unsqueeze1d)(state, t_arr[i], t_arr[i], 0);
  }
}
#endif

THSYCL_API void THSYCLTensor_(baddbmm)(THSYCLState *state, THSYCLTensor *result, scalar_t beta, THSYCLTensor *t, scalar_t alpha, THSYCLTensor *batch1, THSYCLTensor *batch2) 
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
    THSYCLTensor_(resizeAs)(state, result, t);
    if (ScalarConvert<scalar_t, double>::to(beta) != 0.0) {
      THSYCLTensor_(copy)(state, result, t);
    }
  }

  // TODO: This is the work-around implementation for BatchGemm. We should 
  // replace it when Blas library is available.
  auto num_batches = result->size(0);
  // First split t, batch1, batch2 into chunks along 0 dim
  auto t_tensor_vec = at::chunk(THTensor_wrap(t), num_batches, 0);
  auto b1_tensor_vec = at::chunk(THTensor_wrap(batch1), num_batches, 0);
  auto b2_tensor_vec = at::chunk(THTensor_wrap(batch2), num_batches, 0);
  // Initiliaze THSYCL Tensor array and init value from tensor vector
  auto t_tensor_arr =  THSYCLTensor_(initTensorArray)(state, t_tensor_vec);
  auto b1_tensor_arr =  THSYCLTensor_(initTensorArray)(state, b1_tensor_vec);
  auto b2_tensor_arr =  THSYCLTensor_(initTensorArray)(state, b2_tensor_vec);
  // Squeeze tensor in tensor array along 0 dim
  THSYCLTensor_(squeeze1dTensorArray)(state, t_tensor_arr, num_batches);
  THSYCLTensor_(squeeze1dTensorArray)(state, b1_tensor_arr, num_batches);
  THSYCLTensor_(squeeze1dTensorArray)(state, b2_tensor_arr, num_batches);
  // Use GEMM to do the computation
  for (int i = 0; i < num_batches; i++) {
    THSYCLTensor_(mkldnnGemmImpl)(t_tensor_arr[i], beta, alpha, b1_tensor_arr[i], b2_tensor_arr[i]);
  }
  // Unsqueeze t tensor array and concat to result array along 0 dim
  THSYCLTensor_(unsqueeze1dTensorArray)(state, t_tensor_arr, num_batches);
  THSYCLTensor_(catArray)(state, result, t_tensor_arr, num_batches, 0);
  // clear tensor array 
  THSYCLTensor_(finiTensorArray)(state, t_tensor_arr, num_batches);
  THSYCLTensor_(finiTensorArray)(state, b1_tensor_arr, num_batches);
  THSYCLTensor_(finiTensorArray)(state, b2_tensor_arr, num_batches);
#else
  ERROR_ONLY_FP_TYPES("baddbmm");
#endif
}

#endif
