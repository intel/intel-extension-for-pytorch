#include "csrc/aten/cpu/LSTM.h"

#include <c10/core/CPUAllocator.h>
#include <torch/extension.h>
#include <dnnl.h>
#include <omp.h>
#include <immintrin.h>

#include <cassert>
#include <mutex>


namespace torch_ipex {
namespace cpu {

namespace {

namespace math {

#define ALIGN_SIZE 16

template <typename T>
class Matrix {
private:
  int rows_;
  int cols_;
  int stride_;
  T* data_;
  // How many elements was allocated
  int alloc_size_;
  Allocator* allocator;

  Matrix(const Matrix &m);
  Matrix& operator=(const Matrix &m);

public:
  Matrix() {
    this->rows_ = 0;
    this->cols_ = 0;
    this->stride_ = 0;
    this->data_ = NULL;
    this->alloc_size_ = 0;
    allocator = c10::GetAllocator(c10::DeviceType::CPU);
  }
  ~Matrix() {
    this->Release();
  }

  void ShallowCopy(int rows, int cols, int stride, T* data) {
    this->rows_ = rows;
    this->cols_ = cols;
    this->stride_ = stride;
    this->data_ = data;
    this->alloc_size_ = 0;
  }
  void Resize(int rows, int cols) {
    if (rows == rows_ && cols == cols_) {
      return;
    }
    if (rows < 0 && cols < 0) {
      return;
    }
    if (rows == 0 || cols == 0) {
      this->Release();
      return;
    }
    if (cols > 1){
      stride_ = cols % ALIGN_SIZE == 0 ? cols : (cols / ALIGN_SIZE + 1) * ALIGN_SIZE;
    }
    else { //col == 1
      stride_ = 1;
    }
    rows_ = rows;
    cols_ = cols;
    if (alloc_size_ >= stride_ * rows) {
      return;
    } else {
      if (data_) {
        allocator->raw_deallocate(data_);
      }
      alloc_size_ = stride_ * rows_;
      data_ = (T*)allocator->raw_allocate(sizeof(T) * alloc_size_);
      if (data_ == NULL) {
        throw std::bad_alloc();
      }
      memset(data_, 0, sizeof(T) * alloc_size_);
    }
  }
  T* Data() {
    return data_;
  }
  T* Data(const int row) {
    return data_ + stride_ * row;
  }
  T* Data(const int row, const int col) {
    return data_ + stride_ * row + col;
  }
  void Release() {
    if (data_ && alloc_size_ > 0 && std::is_const<T>::value) {
      typedef typename std::remove_const<T>::type TT;
      allocator->raw_deallocate((TT*)data_);
      data_ = NULL;
    }
    rows_ = 0;
    cols_ = 0;
    stride_ = 0;
    alloc_size_ = 0;
  }
  int Rows() {
    return rows_;
  }
  int Cols() {
    return cols_;
  }
  int Stride() {
    return stride_;
  }
  T* Row(const int idx) {
    //assert(idx < rows_ && idx >= 0);
    return data_ + stride_ * idx;
  }
  T& operator()(int r, int c) {
    //assert(r >= 0 && r < rows_ && c >= 0 && c < cols_);
    return *(data_ + r * stride_ + c);
  }
};

template <typename T>
class Vector {
private:
  T* data_;
  int size_;
  int alloc_size_;
  Allocator* allocator;

public:
  Vector() {
    data_ = NULL;
    size_ = 0;
    alloc_size_ = 0;
    allocator = c10::GetAllocator(c10::DeviceType::CPU);
  }
  ~Vector() {
    this->Release();
  }

  void ShallowCopy(int size, T* data) {
    this->size_ = size;
    this->data_ = data;
    this->alloc_size_ = 0;
  }
  void Resize(int size) {
    if (size <= 0){
      this->Release();
      return;
    }
    int skip = (16 - size % 16) % 16;
    if (alloc_size_ >= size + skip) { // space is enough
      size_ = size;
      return;
    }

    alloc_size_ = size + skip;
    size_ = size;
    if (data_) {
      allocator->raw_deallocate(data_);
    }
    data_ = (T*)allocator->raw_allocate(sizeof(T) * alloc_size_);
    if (data_ == NULL) {
      throw std::bad_alloc();
    }

    SetZero();
  }

  void SetZero() {
    memset(data_, 0, sizeof(T) * size_);
  }
  T* Data() {
    return data_;
  }
  T* Data(const int i) {
    return data_ + i;
  }
  void Release() {
    if (data_ && alloc_size_ > 0 && std::is_const<T>::value) {
      typedef typename std::remove_const<T>::type TT;
      allocator->raw_deallocate((TT*)data_);
      data_ = NULL;
    }
    size_ = 0;
    alloc_size_ = 0;
  }
  int Size() {
    return size_;
  }
  T& operator()(const int i) {
    //assert(r >= 0 && r < rows_ && c >= 0 && c < cols_);
    return *(data_ + i);
  }
};

} // end namespace math

template <typename T>
class LstmBase {
public:
  LstmBase() {}
  ~LstmBase() {}
private:
  LstmBase(const LstmBase &b);
  LstmBase& operator=(const LstmBase &b);

public:
  virtual void forward(T*, T *, T*, T*, long int*) = 0;
  virtual void forward_reverse(T*, T *, T*, T*, long int*) = 0;

  // Initialization
  void init(size_t input_size, size_t rnn_size, size_t proj_size, bool has_bias) {
    this->input_size = input_size;
    this->rnn_size = rnn_size;
    this->proj_size = proj_size;
    this->weight_set = false;
    this->batch_first = true;
    this->batch_size = 0;
    this->seq_length = 0;
    this->has_bias = has_bias;
    bias.Resize(4 * rnn_size);
  }

  void set_kernel(const T *forward_weight_ih,
                  const T *forward_bias_ih,
                  const T *forward_weight_hh,
                  const T *forward_bias_hh,
                  const T *projection_weight) {
    if (this->has_bias) {
      #pragma omp parallel for
      for (int i = 0; i < bias.Size(); ++i) {
        bias(i) = forward_bias_ih[i] + forward_bias_hh[i];
      }
    }
    else
      memset(bias.Data(), 0, sizeof(T) * bias.Size());

    w_ifgo.ShallowCopy(4 * rnn_size, input_size, input_size, forward_weight_ih);

    if (proj_size > 0) {
      u_ifgo.ShallowCopy(4 * rnn_size, proj_size, proj_size, forward_weight_hh);
      w_projection.ShallowCopy(proj_size, rnn_size, rnn_size, projection_weight);
    }
    else {
      u_ifgo.ShallowCopy(4 * rnn_size, rnn_size, rnn_size, forward_weight_hh);
    }

    this->weight_set = true;
  }

  bool weight_setted() {
    return this->weight_set;
  }

  void set_input(int seq_length, int batch_size, bool batch_first) {
    if (batch_size != this->batch_size) {
      hu.Resize(batch_size, 4 * rnn_size);
      ct.Resize(batch_size, rnn_size);
      if (proj_size > 0) {
        htp.Resize(batch_size, rnn_size);
        ht.Resize(batch_size, proj_size);
      }
      else {
        ht.Resize(batch_size, rnn_size);
      }
    }
    if (batch_size != this->batch_size || seq_length != this->seq_length) {
      xw.Resize(batch_size * seq_length, 4 * rnn_size);
    }
    this->batch_size = batch_size;
    this->seq_length = seq_length;
    this->batch_first = batch_first;
  }

  void set_initial_state(T* p_h0, T* p_c0) {
    for (int r = 0; r < ct.Rows(); ++r) {
      memcpy(ct.Row(r), p_c0, ct.Cols() * sizeof(T));
      p_c0 += ct.Cols();
    }
    for (int r = 0; r < ht.Rows(); ++r) {
      memcpy(ht.Row(r), p_h0, ht.Cols() * sizeof(T));
      p_h0 += ht.Cols();
    }
  }

protected:
  inline T ref_exp(T x) {
    if (x < -88.0f) { // avoid inf
      x = -88.0f;
    }
    T ret = exp(x);
    return ret;
  }

#if defined(CPU_CAPABILITY_AVX512)
  inline __m512 avx3_exp(const __m512& _x) {
    __m512 p16f_1 = _mm512_set1_ps(1.0f);
    __m512 p16f_half = _mm512_set1_ps(0.5f);
    __m512 p16f_127 = _mm512_set1_ps(127.f);
    __m512 p16f_exp_hi = _mm512_set1_ps(88.3762626647950f);
    __m512 p16f_exp_lo = _mm512_set1_ps(-88.3762626647949f);

    __m512 p16f_cephes_LOG2EF = _mm512_set1_ps(1.44269504088896341f);

    __m512 p16f_cephes_exp_p0 = _mm512_set1_ps(1.9875691500E-4f);
    __m512 p16f_cephes_exp_p1 = _mm512_set1_ps(1.3981999507E-3f);
    __m512 p16f_cephes_exp_p2 = _mm512_set1_ps(8.3334519073E-3f);
    __m512 p16f_cephes_exp_p3 = _mm512_set1_ps(4.1665795894E-2f);
    __m512 p16f_cephes_exp_p4 = _mm512_set1_ps(1.6666665459E-1f);
    __m512 p16f_cephes_exp_p5 = _mm512_set1_ps(5.0000001201E-1f);

    // Clamp x.
    __m512 x = _mm512_max_ps(_mm512_min_ps(_x, p16f_exp_hi), p16f_exp_lo);

    // Express exp(x) as exp(m*ln(2) + r), start by extracting
    // m = floor(x/ln(2) + 0.5).
    __m512 m = _mm512_floor_ps(_mm512_fmadd_ps(x,
                                p16f_cephes_LOG2EF, p16f_half));

    // Get r = x - m*ln(2). If no FMA instructions are available, m*ln(2) is
    // subtracted out in two parts, m*C1+m*C2 = m*ln(2),
    // to avoid accumulating truncation errors.
    // Note that we don't use the "pmadd" function here to
    // ensure that a precision-preserving FMA instruction is used.
    __m512 p16f_nln2 = _mm512_set1_ps(-0.6931471805599453f);
    __m512 r = _mm512_fmadd_ps(m, p16f_nln2, x);

    __m512 r2 = _mm512_mul_ps(r, r);

    // TODO(gonnet): Split into odd/even polynomials and try to exploit
    //               instruction-level parallelism.
    __m512 y = p16f_cephes_exp_p0;
    y = _mm512_fmadd_ps(y, r, p16f_cephes_exp_p1);
    y = _mm512_fmadd_ps(y, r, p16f_cephes_exp_p2);
    y = _mm512_fmadd_ps(y, r, p16f_cephes_exp_p3);
    y = _mm512_fmadd_ps(y, r, p16f_cephes_exp_p4);
    y = _mm512_fmadd_ps(y, r, p16f_cephes_exp_p5);
    y = _mm512_fmadd_ps(y, r2, r);
    y = _mm512_add_ps(y, p16f_1);

    // Build emm0 = 2^m.
    __m512i emm0 = _mm512_cvttps_epi32(_mm512_add_ps(m, p16f_127));
    emm0 = _mm512_slli_epi32(emm0, 23);

    // Return 2^m * exp(r).
    return _mm512_max_ps(_mm512_mul_ps(y, _mm512_castsi512_ps(emm0)), _x);
  }
#endif

  inline T ref_tanh(T x) {
    T ret;
    if (x > 10.0) { // avoid inf -nan
      ret = 1.0;
    } else if (x < -10.0) {
      ret = -1.0;
    } else {
      T tmp = exp(2 * x);
      ret = (tmp - 1) / (tmp + 1);
    }
    return ret;
  }

#if defined(CPU_CAPABILITY_AVX512)
  // tanh(x)= sinh(x)/cosh(x) = (e^2x − 1) / (e^2x + 1)
  inline __m512 avx3_tanh(__m512 x) {
    x = _mm512_min_ps(x, _mm512_set1_ps(10.0f));
    x = _mm512_max_ps(x, _mm512_set1_ps(-10.0f));
    x = _mm512_mul_ps(x, _mm512_set1_ps(2.0f));
    x = avx3_exp(x);
    x = _mm512_div_ps(_mm512_sub_ps(x, _mm512_set1_ps(1.0f)),
                      _mm512_add_ps(x, _mm512_set1_ps(1.0f)));
    return x;
  }
#endif

public:
  std::once_flag flag_set_kernel;

protected:
  // Number of sequences
  int seq_length;
  int batch_size;
  int input_size;
  int rnn_size;
  int proj_size;
  bool weight_set;
  bool batch_first; // for inputs
  bool has_bias;

  // it =    σ(Wi·xt + bii + Ui·ht−1 + bhi)
  // ft =    σ(Wf·xt + bif + Uf·ht−1 + bhf)
  // gt = tanh(Wg·xt + big + Ug·ht−1 + bhg)
  // ot =    σ(Wo·xt + bio + Uo·ht−1 + bho)
  // ct = ft ⊙ ct−1 + it ⊙ gt
  // ht = ot ⊙ tanh(ct)
  //   htp = ot ⊙ tanh(ct)
  //   ht = w_projection · htp
  //     · is gemm
  //     σ is the sigmoid function
  //     ⊙ is the Hadamard product

  // Combined Wi, Wf, Wg, Wo
  math::Matrix<const T> w_ifgo;
  // Combined Ui, Uf, Ug, Uo
  math::Matrix<const T> u_ifgo;
  // Combined bi + bh
  math::Vector<T> bias;
  // Output projection matrix
  math::Matrix<const T> w_projection;

  // Store temp result of x · (Wi, Wf, Wg, Wo) (include all sequences)
  math::Matrix<T> xw;
  // Store temp result of ht-1 · (Ui, Uf, Ug, Uo) (include all sequences)
  math::Matrix<T> hu;
  // Store temp result
  math::Matrix<T> ct;
  math::Matrix<T> ht;
  math::Matrix<T> htp;
};

template <typename T>
class LstmBatch : public LstmBase<T> {
public:
  LstmBatch() {}
  ~LstmBatch() {}
private:
  LstmBatch(const LstmBatch &b);
  LstmBatch& operator=(const LstmBatch &b);

public:
  void forward(T* output, T *p_input_data, T* hy, T* cy, long int*) {
    // xw = x · w_ifgo
    matmul(p_input_data);

    for (int i = 0; i < seq_length; ++i) {
      // hu = ht · u_ifgo
      matmul();
#if defined(CPU_CAPABILITY_AVX512)
      avx3_compute_ct_ht(i);
#else
      ref_compute_ct_ht(i);
#endif
      if (proj_size > 0)
        projection();

      if (batch_first == true) {
        #pragma omp parallel for
        for (int r = 0; r < ht.Rows(); r++) {
          memcpy(&output[(i + r * seq_length) * ht.Cols()], ht.Row(r), sizeof(T)*ht.Cols());
        }
      }
      else {
        #pragma omp parallel for
        for (int r = 0; r < ht.Rows(); r++) {
          memcpy(&output[(i * batch_size + r) * ht.Cols()], ht.Row(r), sizeof(T)*ht.Cols());
        }
      }
    }

    #pragma omp parallel for
    for (int r = 0; r < ht.Rows(); r++) {
      memcpy(hy + r * ht.Cols(), ht.Row(r), sizeof(T)*ht.Cols());
    }

    #pragma omp parallel for
    for (int r = 0; r < ct.Rows(); r++) {
      memcpy(cy + r * ct.Cols(), ct.Row(r), sizeof(T)*ct.Cols());
    }
  }

  void forward_reverse(T* output, T *p_input_data, T* hy, T* cy, long int*) {
    // xw = x · w_ifgo
    matmul(p_input_data);

    for (int i = seq_length - 1; i >= 0; --i) {
      // hu = ht · u_ifgo
      matmul();
#if defined(CPU_CAPABILITY_AVX512)
      avx3_compute_ct_ht(i);
#else
      ref_compute_ct_ht(i);
#endif
      if (proj_size > 0)
        projection();

      if (batch_first == true) {
        #pragma omp parallel for
        for (int r = 0; r < ht.Rows(); r++) {
          memcpy(&output[(i + r * seq_length) * ht.Cols()], ht.Row(r), sizeof(T)*ht.Cols());
        }
      }
      else {
        #pragma omp parallel for
        for (int r = 0; r < ht.Rows(); r++) {
          memcpy(&output[(i * batch_size + r) * ht.Cols()], ht.Row(r), sizeof(T)*ht.Cols());
        }
      }
    }

    #pragma omp parallel for
    for (int r = 0; r < ht.Rows(); r++) {
      memcpy(hy + r * ht.Cols(), ht.Row(r), sizeof(T)*ht.Cols());
    }

    #pragma omp parallel for
    for (int r = 0; r < ct.Rows(); r++) {
      memcpy(cy + r * ct.Cols(), ct.Row(r), sizeof(T)*ct.Cols());
    }
  }

private:
  // xw(batch_size, sequence_length, 4 * rnn_size) = x(batch_size, sequence_length, input_size) · w_ifgo(4 * rnn_size, input_size)
  void matmul(float *input) {
    assert(w_ifgo.Rows() == 4 * rnn_size);
    assert(w_ifgo.Cols() == input_size);
    assert(xw.Rows() == batch_size * seq_length);
    assert(xw.Cols() == 4 * rnn_size);

    // const CBLAS_LAYOUT    layout = CblasRowMajor;
    // const CBLAS_TRANSPOSE transA = CblasNoTrans;
    // const CBLAS_TRANSPOSE transB = CblasTrans;
    const float *pA = input;
    const float *pB = w_ifgo.Row(0);
    float *pC = xw.Row(0);
    int m = xw.Rows();
    int n = xw.Cols();
    int k = w_ifgo.Cols();
    const int lda = input_size;
    const int ldb = w_ifgo.Stride();
    const int ldc = xw.Stride();
    const float alpha = 1.0;
    const float beta = 0.0;

    // printf("MM1: m%d, n%d, k%d, lda%d, ldb%d, ldc%d\n", m, n, k, lda, ldb, ldc);
    // cblas_sgemm(layout, transA, transB, m, n, k, alpha, pA, lda, pB, ldb, beta, pC, ldc);
    dnnl_sgemm('N', 'T', m, n, k, alpha, pA, lda, pB, ldb, beta, pC, ldc);
  }

  // hu(batch_size, 4 * rnn_size) = ht(batch_size, rnn_size) · u_ifgo(4 * rnn_size, rnn_size) if proj_size == 0
  // hu(batch_size, 4 * rnn_size) = ht(batch_size, proj_size) · u_ifgo(4 * rnn_size, proj_size) if proj_size > 0
  void matmul() {
    assert(ht.Rows() == batch_size);
    assert(ht.Cols() == rnn_size || ht.Cols() == proj_size);
    assert(u_ifgo.Rows() == 4 * rnn_size);
    assert(u_ifgo.Cols() == rnn_size || u_ifgo.Cols() == proj_size);
    assert(hu.Rows() == batch_size);
    assert(hu.Cols() == 4 * rnn_size);

    // const CBLAS_LAYOUT    layout = CblasRowMajor;
    // const CBLAS_TRANSPOSE transA = CblasNoTrans;
    // const CBLAS_TRANSPOSE transB = CblasTrans;
    const float *pA = ht.Row(0);
    const float *pB = u_ifgo.Row(0);
    float *pC = hu.Row(0);
    const int m = hu.Rows();
    const int n = hu.Cols();
    const int k = u_ifgo.Cols();
    const int lda = ht.Stride();
    const int ldb = u_ifgo.Stride();
    const int ldc = hu.Stride();
    const float alpha = 1.0;
    const float beta = 0.0;

    // printf("MM2: m%d, n%d, k%d, lda%d, ldb%d, ldc%d\n", m, n, k, lda, ldb, ldc);
    // cblas_sgemm(layout, transA, transB, m, n, k, alpha, pA, lda, pB, ldb, beta, pC, ldc);
    dnnl_sgemm('N', 'T', m, n, k, alpha, pA, lda, pB, ldb, beta, pC, ldc);

    // TODO(batch_size ==1)
    // cblas_sgemv
  }

  // ht(batch_size, proj_size) = htp(batch_size, rnn_size) · w_projection(proj_size, rnn_size)
  void projection() {
    assert(htp.Rows() == batch_size);
    assert(htp.Cols() == rnn_size);
    assert(w_projection.Rows() == proj_size);
    assert(w_projection.Cols() == rnn_size);
    assert(ht.Rows() == batch_size);
    assert(ht.Cols() == proj_size);

    // const CBLAS_LAYOUT    layout = CblasRowMajor;
    // const CBLAS_TRANSPOSE transA = CblasNoTrans;
    // const CBLAS_TRANSPOSE transB = CblasTrans;
    const float *pA = htp.Row(0);
    const float *pB = w_projection.Row(0);
    float *pC = ht.Row(0);
    const int m = ht.Rows();
    const int n = ht.Cols();
    const int k = w_projection.Cols();
    const int lda = htp.Stride();
    const int ldb = w_projection.Stride();
    const int ldc = ht.Stride();
    const float alpha = 1.0;
    const float beta = 0.0;

    // printf("MM2: m%d, n%d, k%d, lda%d, ldb%d, ldc%d\n", m, n, k, lda, ldb, ldc);
    // cblas_sgemm(layout, transA, transB, m, n, k, alpha, pA, lda, pB, ldb, beta, pC, ldc);
    dnnl_sgemm('N', 'T', m, n, k, alpha, pA, lda, pB, ldb, beta, pC, ldc);

    // TODO(batch_size ==1)
    // cblas_sgemv
  }

  // it =    σ(Wi·xt + bii + Ui·ht−1 + bhi)
  // ft =    σ(Wf·xt + bif + Uf·ht−1 + bhf)
  // gt = tanh(Wg·xt + big + Ug·ht−1 + bhg)
  // ot =    σ(Wo·xt + bio + Uo·ht−1 + bho)
  // ct = ft ⊙ ct−1 + it ⊙ gt
  // ht = ot ⊙ tanh(ct)
  //   htp = ot ⊙ tanh(ct)  if proj_size > 0
  //   ht = w_projection · htp
  void ref_compute_ct_ht(int seq_idx) {
    assert(ct.Rows() == ht.Rows());
    assert(ct.Cols() == ht.Cols() || ht.Cols() == proj_size);
    assert(ct.Stride() == ht.Stride());
    const int it_off = 0 * rnn_size;
    const int ft_off = 1 * rnn_size;
    const int gt_off = 2 * rnn_size;
    const int ot_off = 3 * rnn_size;

    const T *it_bias_ptr = bias.Data(it_off);
    const T *ft_bias_ptr = bias.Data(ft_off);
    const T *gt_bias_ptr = bias.Data(gt_off);
    const T *ot_bias_ptr = bias.Data(ot_off);

    #pragma omp parallel for
    for (int r = 0; r < ct.Rows(); ++r) {
      int wx_row = 0;
      if (batch_first == true)
        wx_row = seq_idx + r * seq_length;
      else
        wx_row = seq_idx * batch_size + r;

      const T *it_xw_ptr = xw.Data(wx_row, it_off);
      const T *ft_xw_ptr = xw.Data(wx_row, ft_off);
      const T *gt_xw_ptr = xw.Data(wx_row, gt_off);
      const T *ot_xw_ptr = xw.Data(wx_row, ot_off);
      const T *it_hu_ptr = hu.Data(r, it_off);
      const T *ft_hu_ptr = hu.Data(r, ft_off);
      const T *gt_hu_ptr = hu.Data(r, gt_off);
      const T *ot_hu_ptr = hu.Data(r, ot_off);
      T *ct_ptr = ct.Data(r);
      T *ht_ptr = ht.Data(r);
      if (proj_size > 0)
        ht_ptr = htp.Data(r);

      #pragma omp simd
      for (int c = 0; c < ct.Cols(); ++c) {
        T it_sum = it_xw_ptr[c] + it_hu_ptr[c] + it_bias_ptr[c];
        T ft_sum = ft_xw_ptr[c] + ft_hu_ptr[c] + ft_bias_ptr[c];
        T gt_sum = gt_xw_ptr[c] + gt_hu_ptr[c] + gt_bias_ptr[c];
        T ot_sum = ot_xw_ptr[c] + ot_hu_ptr[c] + ot_bias_ptr[c];
        // if (foget_gate) {
        //     sum += forget_bias;
        // }
        T it = 1.0f / (1.0f + ref_exp(0.0f - it_sum));
        T ft = 1.0f / (1.0f + ref_exp(0.0f - ft_sum));
        T gt = ref_tanh(gt_sum);
        T ot = 1.0f / (1.0f + ref_exp(0.0f - ot_sum));

        ct_ptr[c] = ft * ct_ptr[c] + it * gt;
        ht_ptr[c] = ot * ref_tanh(ct_ptr[c]);
      }
    }
  }

#if defined(CPU_CAPABILITY_AVX512)
  void avx3_compute_ct_ht(int seq_idx) {
    assert(ct.Rows() == ht.Rows());
    assert(ct.Cols() == ht.Cols() || ht.Cols() == proj_size);
    assert(ct.Stride() == ht.Stride());
    const int it_off = 0 * rnn_size;
    const int ft_off = 1 * rnn_size;
    const int gt_off = 2 * rnn_size;
    const int ot_off = 3 * rnn_size;

    const T *it_bias_ptr = bias.Data(it_off);
    const T *ft_bias_ptr = bias.Data(ft_off);
    const T *gt_bias_ptr = bias.Data(gt_off);
    const T *ot_bias_ptr = bias.Data(ot_off);

    const __m512 minimum = _mm512_set1_ps(-88.0f);
    const __m512 ones = _mm512_set1_ps(1.0f);
    const __m512 zeros = _mm512_set1_ps(0.0f);
    const __mmask16 len2mask[17] = { 0x0000, 0x0001, 0x0003, 0x0007, 0x000F,
                                      0x001F, 0x003F, 0x007F, 0x00FF,
                                      0x01FF, 0x03FF, 0x07FF, 0x0FFF,
                                      0x1FFF, 0x3FFF, 0x7FFF, 0xFFFF};
    #define AVX3_BLOCK_SIZE 16
    const int avx3_block_idx = ct.Cols() / 16 * 16;

    auto compute = [&](__m512 &it_avx3, __m512 &it_hu_avx3, __m512 &it_bias_avx3,
                        __m512 &ft_avx3, __m512 &ft_hu_avx3, __m512 &ft_bias_avx3,
                        __m512 &gt_avx3, __m512 &gt_hu_avx3, __m512 &gt_bias_avx3,
                        __m512 &ot_avx3, __m512 &ot_hu_avx3, __m512 &ot_bias_avx3,
                        __m512 &ct_avx3, __m512 &ht_avx3) {
      // it_avx3 = 1.0f / (1.0f + exp(0.f - sum));
      it_avx3 = _mm512_add_ps(_mm512_add_ps(it_avx3, it_hu_avx3), it_bias_avx3);
      it_avx3 = _mm512_max_ps(it_avx3, minimum);
      it_avx3 = _mm512_sub_ps(zeros, it_avx3);
      it_avx3 = avx3_exp(it_avx3);
      it_avx3 = _mm512_add_ps(it_avx3, ones);
      it_avx3 = _mm512_div_ps(ones, it_avx3);

      // ft_avx3 = 1.0f / (1.0f + exp(0.f - sum));
      ft_avx3 = _mm512_add_ps(_mm512_add_ps(ft_avx3, ft_hu_avx3), ft_bias_avx3);
      ft_avx3 = _mm512_max_ps(ft_avx3, minimum);
      ft_avx3 = _mm512_sub_ps(zeros, ft_avx3);
      ft_avx3 = avx3_exp(ft_avx3);
      ft_avx3 = _mm512_add_ps(ft_avx3, ones);
      ft_avx3 = _mm512_div_ps(ones, ft_avx3);

      // gt_avx3 = tanh(sum);
      gt_avx3 = _mm512_add_ps(_mm512_add_ps(gt_avx3, gt_hu_avx3), gt_bias_avx3);
      gt_avx3 = avx3_tanh(gt_avx3);

      // ot_avx3 = 1.0f / (1.0f + exp(0.f - sum));
      ot_avx3 = _mm512_add_ps(_mm512_add_ps(ot_avx3, ot_hu_avx3), ot_bias_avx3);
      ot_avx3 = _mm512_max_ps(ot_avx3, minimum);
      ot_avx3 = _mm512_sub_ps(zeros, ot_avx3);
      ot_avx3 = avx3_exp(ot_avx3);
      ot_avx3 = _mm512_add_ps(ot_avx3, ones);
      ot_avx3 = _mm512_div_ps(ones, ot_avx3);

      // ct(r, c) = ft * ct(r, c) + it * gt;
      ct_avx3 = _mm512_add_ps(_mm512_mul_ps(ft_avx3, ct_avx3), _mm512_mul_ps(it_avx3, gt_avx3));

      // ht(r, c) = ot * ref_tanh(ct(r, c));
      ht_avx3 = _mm512_mul_ps(ot_avx3, avx3_tanh(ct_avx3));
    };

    #pragma omp parallel for
    for (int r = 0; r < ct.Rows(); ++r) {
      int wx_row = 0;
      if (batch_first == true)
        wx_row = seq_idx + r * seq_length;
      else
        wx_row = seq_idx * batch_size + r;

      const T *it_xw_ptr = xw.Data(wx_row, it_off);
      const T *ft_xw_ptr = xw.Data(wx_row, ft_off);
      const T *gt_xw_ptr = xw.Data(wx_row, gt_off);
      const T *ot_xw_ptr = xw.Data(wx_row, ot_off);
      const T *it_hu_ptr = hu.Data(r, it_off);
      const T *ft_hu_ptr = hu.Data(r, ft_off);
      const T *gt_hu_ptr = hu.Data(r, gt_off);
      const T *ot_hu_ptr = hu.Data(r, ot_off);
      T *ct_ptr = ct.Data(r);
      T *ht_ptr = ht.Data(r);
      if (proj_size > 0)
        ht_ptr = htp.Data(r);

      int c = 0;
      for(; c < avx3_block_idx; c += 16) {
        __m512 it_avx3 = _mm512_loadu_ps(&it_xw_ptr[c]);
        __m512 it_hu_avx3 = _mm512_loadu_ps(&it_hu_ptr[c]);
        __m512 it_bias_avx3 = _mm512_loadu_ps(&it_bias_ptr[c]);

        __m512 ft_avx3 = _mm512_loadu_ps(&ft_xw_ptr[c]);
        __m512 ft_hu_avx3 = _mm512_loadu_ps(&ft_hu_ptr[c]);
        __m512 ft_bias_avx3 = _mm512_loadu_ps(&ft_bias_ptr[c]);

        __m512 gt_avx3 = _mm512_loadu_ps(&gt_xw_ptr[c]);
        __m512 gt_hu_avx3 = _mm512_loadu_ps(&gt_hu_ptr[c]);
        __m512 gt_bias_avx3 = _mm512_loadu_ps(&gt_bias_ptr[c]);

        __m512 ot_avx3 = _mm512_loadu_ps(&ot_xw_ptr[c]);
        __m512 ot_hu_avx3 = _mm512_loadu_ps(&ot_hu_ptr[c]);
        __m512 ot_bias_avx3 = _mm512_loadu_ps(&ot_bias_ptr[c]);

        __m512 ct_avx3 = _mm512_loadu_ps(&ct_ptr[c]);
        __m512 ht_avx3;

        compute(it_avx3, it_hu_avx3, it_bias_avx3,
                ft_avx3, ft_hu_avx3, ft_bias_avx3,
                gt_avx3, gt_hu_avx3, gt_bias_avx3,
                ot_avx3, ot_hu_avx3, ot_bias_avx3,
                ct_avx3, ht_avx3);

        _mm512_storeu_ps(&ct_ptr[c], ct_avx3);
        _mm512_storeu_ps(&ht_ptr[c], ht_avx3);
      }

      int remainder = ct.Cols() - avx3_block_idx;
      if (remainder > 0) {
        __mmask16 mask = len2mask[remainder];
        // it_avx3 = 1.0f / (1.0f + exp(0.f - sum));
        __m512 it_avx3 = _mm512_maskz_loadu_ps(mask, &it_xw_ptr[c]);
        __m512 it_hu_avx3 = _mm512_maskz_loadu_ps(mask, &it_hu_ptr[c]);
        __m512 it_bias_avx3 = _mm512_maskz_loadu_ps(mask, &it_bias_ptr[c]);

        __m512 ft_avx3 = _mm512_maskz_loadu_ps(mask, &ft_xw_ptr[c]);
        __m512 ft_hu_avx3 = _mm512_maskz_loadu_ps(mask, &ft_hu_ptr[c]);
        __m512 ft_bias_avx3 = _mm512_maskz_loadu_ps(mask, &ft_bias_ptr[c]);

        __m512 gt_avx3 = _mm512_maskz_loadu_ps(mask, &gt_xw_ptr[c]);
        __m512 gt_hu_avx3 = _mm512_maskz_loadu_ps(mask, &gt_hu_ptr[c]);
        __m512 gt_bias_avx3 = _mm512_maskz_loadu_ps(mask, &gt_bias_ptr[c]);

        __m512 ot_avx3 = _mm512_maskz_loadu_ps(mask, &ot_xw_ptr[c]);
        __m512 ot_hu_avx3 = _mm512_maskz_loadu_ps(mask, &ot_hu_ptr[c]);
        __m512 ot_bias_avx3 = _mm512_maskz_loadu_ps(mask, &ot_bias_ptr[c]);

        __m512 ct_avx3 = _mm512_maskz_loadu_ps(mask, &ct_ptr[c]);
        __m512 ht_avx3;

        compute(it_avx3, it_hu_avx3, it_bias_avx3,
                ft_avx3, ft_hu_avx3, ft_bias_avx3,
                gt_avx3, gt_hu_avx3, gt_bias_avx3,
                ot_avx3, ot_hu_avx3, ot_bias_avx3,
                ct_avx3, ht_avx3);

        _mm512_mask_storeu_ps(&ct_ptr[c], mask, ct_avx3);
        _mm512_mask_storeu_ps(&ht_ptr[c], mask, ht_avx3);
      }
    }
  }
#endif

private:
  using LstmBase<T>::ref_exp;
  using LstmBase<T>::ref_tanh;
#if defined(CPU_CAPABILITY_AVX512)
  using LstmBase<T>::avx3_exp;
  using LstmBase<T>::avx3_tanh;
#endif

  // Inputs
  using LstmBase<T>::seq_length;
  using LstmBase<T>::batch_size;
  using LstmBase<T>::input_size;
  using LstmBase<T>::rnn_size;
  using LstmBase<T>::proj_size;
  using LstmBase<T>::batch_first;

  // Temporary value
  using LstmBase<T>::xw;
  using LstmBase<T>::hu;
  using LstmBase<T>::ct;
  using LstmBase<T>::ht;
  using LstmBase<T>::htp;

  // Weights
  using LstmBase<T>::w_ifgo;
  using LstmBase<T>::u_ifgo;
  using LstmBase<T>::bias;
  using LstmBase<T>::w_projection;
};

template <typename T>
class LstmPacked : public LstmBase<T> {
public:
  LstmPacked() {}
  ~LstmPacked() {}
private:
  LstmPacked(const LstmPacked &b);
  LstmPacked& operator=(const LstmPacked &b);

public:
  void forward(T* output, T *p_input_data, T* hy, T* cy, long int* batch_sizes) {
    // xw = x · w_ifgo
    matmul(p_input_data, batch_sizes);

    for (int i = 0; i < seq_length; ++i) {
      // hu = ht · u_ifgo
      matmul(batch_sizes[i]);
#if defined(CPU_CAPABILITY_AVX512)
      avx3_compute_ct_ht(i, batch_sizes);
#else
      ref_compute_ct_ht(i, batch_sizes);
#endif
      if (proj_size > 0)
        projection(batch_sizes[i]);

      int batch_sizes_sum = std::accumulate(batch_sizes, batch_sizes + i, 0);
      int cur_batch_size = batch_sizes[i];
      #pragma omp parallel for
      for (int r = 0; r < cur_batch_size; r++) {
        memcpy(&output[(batch_sizes_sum + r) * ht.Cols()], ht.Row(r), sizeof(T)*ht.Cols());
      }
    }

    #pragma omp parallel for
    for (int r = 0; r < ht.Rows(); r++) {
      memcpy(hy + r * ht.Cols(), ht.Row(r), sizeof(T)*ht.Cols());
    }

    #pragma omp parallel for
    for (int r = 0; r < ct.Rows(); r++) {
      memcpy(cy + r * ct.Cols(), ct.Row(r), sizeof(T)*ct.Cols());
    }
  }

  void forward_reverse(T* output, T *p_input_data, T* hy, T* cy,  long int* batch_sizes) {
    // xw = x · w_ifgo
    matmul(p_input_data, batch_sizes);

    for (int i = seq_length - 1; i >= 0; --i) {
      // hu = ht · u_ifgo
      matmul(batch_sizes[i]);
#if defined(CPU_CAPABILITY_AVX512)
      avx3_compute_ct_ht(i, batch_sizes);
#else
      ref_compute_ct_ht(i, batch_sizes);
#endif
      if (proj_size > 0)
        projection(batch_sizes[i]);

      int batch_sizes_sum = std::accumulate(batch_sizes, batch_sizes + i, 0);
      int cur_batch_size = batch_sizes[i];
      #pragma omp parallel for
      for (int r = 0; r < cur_batch_size; r++) {
        memcpy(&output[(batch_sizes_sum + r) * ht.Cols()], ht.Row(r), sizeof(T)*ht.Cols());
      }
    }

    #pragma omp parallel for
    for (int r = 0; r < ht.Rows(); r++) {
      memcpy(hy + r * ht.Cols(), ht.Row(r), sizeof(T)*ht.Cols());
    }

    #pragma omp parallel for
    for (int r = 0; r < ct.Rows(); r++) {
      memcpy(cy + r * ct.Cols(), ct.Row(r), sizeof(T)*ct.Cols());
    }
  }

private:
  // batch 1: 147035
  // batch 2: 25814N
  // batch 3: 3692NN
  // xw(batch_sizes * max_sequence_length, 4 * rnn_size) = x(batch_sizes * max_sequence_length, input_size) · w_ifgo(4 * rnn_size, input_size)
  // batch_sizes = [3, 3, 3, 3, 2, 1]
  void matmul(float *input, long int* batch_sizes) {
    assert(w_ifgo.Rows() == 4 * rnn_size);
    assert(w_ifgo.Cols() == input_size);
    assert(xw.Rows() == batch_size * seq_length);
    assert(xw.Cols() == 4 * rnn_size);

    // const CBLAS_LAYOUT    layout = CblasRowMajor;
    // const CBLAS_TRANSPOSE transA = CblasNoTrans;
    // const CBLAS_TRANSPOSE transB = CblasTrans;
    const float *pA = input;
    const float *pB = w_ifgo.Row(0);
    float *pC = xw.Row(0);
    int m = std::accumulate(batch_sizes, batch_sizes + seq_length, 0);
    int n = xw.Cols();
    int k = w_ifgo.Cols();
    const int lda = input_size;
    const int ldb = w_ifgo.Stride();
    const int ldc = xw.Stride();
    const float alpha = 1.0;
    const float beta = 0.0;

    // printf("MM1: m%d, n%d, k%d, lda%d, ldb%d, ldc%d\n", m, n, k, lda, ldb, ldc);
    // cblas_sgemm(layout, transA, transB, m, n, k, alpha, pA, lda, pB, ldb, beta, pC, ldc);
    dnnl_sgemm('N', 'T', m, n, k, alpha, pA, lda, pB, ldb, beta, pC, ldc);
  }

  // hu(cur_batch_size, 4 * rnn_size) = ht(cur_batch_size, rnn_size) · u_ifgo(4 * rnn_size, rnn_size) if proj_size == 0
  // hu(cur_batch_size, 4 * rnn_size) = ht(cur_batch_size, proj_size) · u_ifgo(4 * rnn_size, proj_size) if proj_size > 0
  void matmul(int cur_batch_size) {
    assert(ht.Rows() == batch_size);
    assert(ht.Cols() == rnn_size || ht.Cols() == proj_size);
    assert(u_ifgo.Rows() == 4 * rnn_size);
    assert(u_ifgo.Cols() == rnn_size || u_ifgo.Cols() == proj_size);
    assert(hu.Rows() == batch_size);
    assert(hu.Cols() == 4 * rnn_size);

    // const CBLAS_LAYOUT    layout = CblasRowMajor;
    // const CBLAS_TRANSPOSE transA = CblasNoTrans;
    // const CBLAS_TRANSPOSE transB = CblasTrans;
    const float *pA = ht.Row(0);
    const float *pB = u_ifgo.Row(0);
    float *pC = hu.Row(0);
    const int m = cur_batch_size;
    const int n = hu.Cols();
    const int k = u_ifgo.Cols();
    const int lda = ht.Stride();
    const int ldb = u_ifgo.Stride();
    const int ldc = hu.Stride();
    const float alpha = 1.0;
    const float beta = 0.0;

    // printf("MM2: m%d, n%d, k%d, lda%d, ldb%d, ldc%d\n", m, n, k, lda, ldb, ldc);
    // cblas_sgemm(layout, transA, transB, m, n, k, alpha, pA, lda, pB, ldb, beta, pC, ldc);
    dnnl_sgemm('N', 'T', m, n, k, alpha, pA, lda, pB, ldb, beta, pC, ldc);

    // TODO(cur_batch_size == 1)
    // cblas_sgemv
  }

  // ht(cur_batch_size, proj_size) = htp(cur_batch_size, rnn_size) · w_projection(proj_size, rnn_size)
  void projection(int cur_batch_size) {
    assert(htp.Rows() == batch_size);
    assert(htp.Cols() == rnn_size);
    assert(w_projection.Rows() == proj_size);
    assert(w_projection.Cols() == rnn_size);
    assert(ht.Rows() == batch_size);
    assert(ht.Cols() == proj_size);

    // const CBLAS_LAYOUT    layout = CblasRowMajor;
    // const CBLAS_TRANSPOSE transA = CblasNoTrans;
    // const CBLAS_TRANSPOSE transB = CblasTrans;
    const float *pA = htp.Row(0);
    const float *pB = w_projection.Row(0);
    float *pC = ht.Row(0);
    const int m = cur_batch_size;
    const int n = ht.Cols();
    const int k = w_projection.Cols();
    const int lda = htp.Stride();
    const int ldb = w_projection.Stride();
    const int ldc = ht.Stride();
    const float alpha = 1.0;
    const float beta = 0.0;

    // printf("MM2: m%d, n%d, k%d, lda%d, ldb%d, ldc%d\n", m, n, k, lda, ldb, ldc);
    // cblas_sgemm(layout, transA, transB, m, n, k, alpha, pA, lda, pB, ldb, beta, pC, ldc);
    dnnl_sgemm('N', 'T', m, n, k, alpha, pA, lda, pB, ldb, beta, pC, ldc);

    // TODO(batch_size ==1)
    // cblas_sgemv
  }

  // it =    σ(Wi·xt + bii + Ui·ht−1 + bhi)
  // ft =    σ(Wf·xt + bif + Uf·ht−1 + bhf)
  // gt = tanh(Wg·xt + big + Ug·ht−1 + bhg)
  // ot =    σ(Wo·xt + bio + Uo·ht−1 + bho)
  // ct = ft ⊙ ct−1 + it ⊙ gt
  // ht = ot ⊙ tanh(ct)
  //   htp = ot ⊙ tanh(ct)  if proj_size > 0
  //   ht = w_projection · htp
  void ref_compute_ct_ht(int seq_idx, long int* batch_sizes) {
    assert(ct.Rows() == ht.Rows());
    assert(ct.Cols() == ht.Cols() || ht.Cols() == proj_size);
    assert(ct.Stride() == ht.Stride());
    const int it_off = 0 * rnn_size;
    const int ft_off = 1 * rnn_size;
    const int gt_off = 2 * rnn_size;
    const int ot_off = 3 * rnn_size;

    const T *it_bias_ptr = bias.Data(it_off);
    const T *ft_bias_ptr = bias.Data(ft_off);
    const T *gt_bias_ptr = bias.Data(gt_off);
    const T *ot_bias_ptr = bias.Data(ot_off);

    int cur_batch_size = batch_sizes[seq_idx];
    #pragma omp parallel for
    for (int r = 0; r < cur_batch_size; ++r) {
      int wx_row = std::accumulate(batch_sizes, batch_sizes + seq_idx, r);
      const T *it_xw_ptr = xw.Data(wx_row, it_off);
      const T *ft_xw_ptr = xw.Data(wx_row, ft_off);
      const T *gt_xw_ptr = xw.Data(wx_row, gt_off);
      const T *ot_xw_ptr = xw.Data(wx_row, ot_off);
      const T *it_hu_ptr = hu.Data(r, it_off);
      const T *ft_hu_ptr = hu.Data(r, ft_off);
      const T *gt_hu_ptr = hu.Data(r, gt_off);
      const T *ot_hu_ptr = hu.Data(r, ot_off);
      T *ct_ptr = ct.Data(r);
      T *ht_ptr = ht.Data(r);
      if (proj_size > 0)
        ht_ptr = htp.Data(r);

      #pragma omp simd
      for (int c = 0; c < ct.Cols(); ++c) {
        T it_sum = it_xw_ptr[c] + it_hu_ptr[c] + it_bias_ptr[c];
        T ft_sum = ft_xw_ptr[c] + ft_hu_ptr[c] + ft_bias_ptr[c];
        T gt_sum = gt_xw_ptr[c] + gt_hu_ptr[c] + gt_bias_ptr[c];
        T ot_sum = ot_xw_ptr[c] + ot_hu_ptr[c] + ot_bias_ptr[c];
        // if (foget_gate) {
        //     sum += forget_bias;
        // }
        T it = 1.0f / (1.0f + ref_exp(0.0f - it_sum));
        T ft = 1.0f / (1.0f + ref_exp(0.0f - ft_sum));
        T gt = ref_tanh(gt_sum);
        T ot = 1.0f / (1.0f + ref_exp(0.0f - ot_sum));

        ct_ptr[c] = ft * ct_ptr[c] + it * gt;
        ht_ptr[c] = ot * ref_tanh(ct_ptr[c]);
      }
    }
  }

#if defined(CPU_CAPABILITY_AVX512)
  void avx3_compute_ct_ht(int seq_idx, long int* batch_sizes) {
    assert(ct.Rows() == ht.Rows());
    assert(ct.Cols() == ht.Cols() || ht.Cols() == proj_size);
    assert(ct.Stride() == ht.Stride());
    const int it_off = 0 * rnn_size;
    const int ft_off = 1 * rnn_size;
    const int gt_off = 2 * rnn_size;
    const int ot_off = 3 * rnn_size;

    const T *it_bias_ptr = bias.Data(it_off);
    const T *ft_bias_ptr = bias.Data(ft_off);
    const T *gt_bias_ptr = bias.Data(gt_off);
    const T *ot_bias_ptr = bias.Data(ot_off);

    const __m512 minimum = _mm512_set1_ps(-88.0f);
    const __m512 ones = _mm512_set1_ps(1.0f);
    const __m512 zeros = _mm512_set1_ps(0.0f);
    const __mmask16 len2mask[17] = { 0x0000, 0x0001, 0x0003, 0x0007, 0x000F,
                                      0x001F, 0x003F, 0x007F, 0x00FF,
                                      0x01FF, 0x03FF, 0x07FF, 0x0FFF,
                                      0x1FFF, 0x3FFF, 0x7FFF, 0xFFFF};
    #define AVX3_BLOCK_SIZE 16
    const int avx3_block_idx = ct.Cols() / 16 * 16;

    auto compute = [&](__m512 &it_avx3, __m512 &it_hu_avx3, __m512 &it_bias_avx3,
                        __m512 &ft_avx3, __m512 &ft_hu_avx3, __m512 &ft_bias_avx3,
                        __m512 &gt_avx3, __m512 &gt_hu_avx3, __m512 &gt_bias_avx3,
                        __m512 &ot_avx3, __m512 &ot_hu_avx3, __m512 &ot_bias_avx3,
                        __m512 &ct_avx3, __m512 &ht_avx3) {
      // it_avx3 = 1.0f / (1.0f + exp(0.f - sum));
      it_avx3 = _mm512_add_ps(_mm512_add_ps(it_avx3, it_hu_avx3), it_bias_avx3);
      it_avx3 = _mm512_max_ps(it_avx3, minimum);
      it_avx3 = _mm512_sub_ps(zeros, it_avx3);
      it_avx3 = avx3_exp(it_avx3);
      it_avx3 = _mm512_add_ps(it_avx3, ones);
      it_avx3 = _mm512_div_ps(ones, it_avx3);

      // ft_avx3 = 1.0f / (1.0f + exp(0.f - sum));
      ft_avx3 = _mm512_add_ps(_mm512_add_ps(ft_avx3, ft_hu_avx3), ft_bias_avx3);
      ft_avx3 = _mm512_max_ps(ft_avx3, minimum);
      ft_avx3 = _mm512_sub_ps(zeros, ft_avx3);
      ft_avx3 = avx3_exp(ft_avx3);
      ft_avx3 = _mm512_add_ps(ft_avx3, ones);
      ft_avx3 = _mm512_div_ps(ones, ft_avx3);

      // gt_avx3 = tanh(sum);
      gt_avx3 = _mm512_add_ps(_mm512_add_ps(gt_avx3, gt_hu_avx3), gt_bias_avx3);
      gt_avx3 = avx3_tanh(gt_avx3);

      // ot_avx3 = 1.0f / (1.0f + exp(0.f - sum));
      ot_avx3 = _mm512_add_ps(_mm512_add_ps(ot_avx3, ot_hu_avx3), ot_bias_avx3);
      ot_avx3 = _mm512_max_ps(ot_avx3, minimum);
      ot_avx3 = _mm512_sub_ps(zeros, ot_avx3);
      ot_avx3 = avx3_exp(ot_avx3);
      ot_avx3 = _mm512_add_ps(ot_avx3, ones);
      ot_avx3 = _mm512_div_ps(ones, ot_avx3);

      // ct(r, c) = ft * ct(r, c) + it * gt;
      ct_avx3 = _mm512_add_ps(_mm512_mul_ps(ft_avx3, ct_avx3), _mm512_mul_ps(it_avx3, gt_avx3));

      // ht(r, c) = ot * ref_tanh(ct(r, c));
      ht_avx3 = _mm512_mul_ps(ot_avx3, avx3_tanh(ct_avx3));
    };

    int cur_batch_size = batch_sizes[seq_idx];
    #pragma omp parallel for
    for (int r = 0; r < cur_batch_size; ++r) {
      int wx_row = std::accumulate(batch_sizes, batch_sizes + seq_idx, r);
      const T *it_xw_ptr = xw.Data(wx_row, it_off);
      const T *ft_xw_ptr = xw.Data(wx_row, ft_off);
      const T *gt_xw_ptr = xw.Data(wx_row, gt_off);
      const T *ot_xw_ptr = xw.Data(wx_row, ot_off);
      const T *it_hu_ptr = hu.Data(r, it_off);
      const T *ft_hu_ptr = hu.Data(r, ft_off);
      const T *gt_hu_ptr = hu.Data(r, gt_off);
      const T *ot_hu_ptr = hu.Data(r, ot_off);
      T *ct_ptr = ct.Data(r);
      T *ht_ptr = ht.Data(r);
      if (proj_size > 0)
          ht_ptr = htp.Data(r);

      int c = 0;
      for(; c < avx3_block_idx; c += 16) {
        __m512 it_avx3 = _mm512_loadu_ps(&it_xw_ptr[c]);
        __m512 it_hu_avx3 = _mm512_loadu_ps(&it_hu_ptr[c]);
        __m512 it_bias_avx3 = _mm512_loadu_ps(&it_bias_ptr[c]);

        __m512 ft_avx3 = _mm512_loadu_ps(&ft_xw_ptr[c]);
        __m512 ft_hu_avx3 = _mm512_loadu_ps(&ft_hu_ptr[c]);
        __m512 ft_bias_avx3 = _mm512_loadu_ps(&ft_bias_ptr[c]);

        __m512 gt_avx3 = _mm512_loadu_ps(&gt_xw_ptr[c]);
        __m512 gt_hu_avx3 = _mm512_loadu_ps(&gt_hu_ptr[c]);
        __m512 gt_bias_avx3 = _mm512_loadu_ps(&gt_bias_ptr[c]);

        __m512 ot_avx3 = _mm512_loadu_ps(&ot_xw_ptr[c]);
        __m512 ot_hu_avx3 = _mm512_loadu_ps(&ot_hu_ptr[c]);
        __m512 ot_bias_avx3 = _mm512_loadu_ps(&ot_bias_ptr[c]);

        __m512 ct_avx3 = _mm512_loadu_ps(&ct_ptr[c]);
        __m512 ht_avx3;

        compute(it_avx3, it_hu_avx3, it_bias_avx3,
                ft_avx3, ft_hu_avx3, ft_bias_avx3,
                gt_avx3, gt_hu_avx3, gt_bias_avx3,
                ot_avx3, ot_hu_avx3, ot_bias_avx3,
                ct_avx3, ht_avx3);

        _mm512_storeu_ps(&ct_ptr[c], ct_avx3);
        _mm512_storeu_ps(&ht_ptr[c], ht_avx3);
      }

      int remainder = ct.Cols() - avx3_block_idx;
      if (remainder > 0) {
        __mmask16 mask = len2mask[remainder];
        // it_avx3 = 1.0f / (1.0f + exp(0.f - sum));
        __m512 it_avx3 = _mm512_maskz_loadu_ps(mask, &it_xw_ptr[c]);
        __m512 it_hu_avx3 = _mm512_maskz_loadu_ps(mask, &it_hu_ptr[c]);
        __m512 it_bias_avx3 = _mm512_maskz_loadu_ps(mask, &it_bias_ptr[c]);

        __m512 ft_avx3 = _mm512_maskz_loadu_ps(mask, &ft_xw_ptr[c]);
        __m512 ft_hu_avx3 = _mm512_maskz_loadu_ps(mask, &ft_hu_ptr[c]);
        __m512 ft_bias_avx3 = _mm512_maskz_loadu_ps(mask, &ft_bias_ptr[c]);

        __m512 gt_avx3 = _mm512_maskz_loadu_ps(mask, &gt_xw_ptr[c]);
        __m512 gt_hu_avx3 = _mm512_maskz_loadu_ps(mask, &gt_hu_ptr[c]);
        __m512 gt_bias_avx3 = _mm512_maskz_loadu_ps(mask, &gt_bias_ptr[c]);

        __m512 ot_avx3 = _mm512_maskz_loadu_ps(mask, &ot_xw_ptr[c]);
        __m512 ot_hu_avx3 = _mm512_maskz_loadu_ps(mask, &ot_hu_ptr[c]);
        __m512 ot_bias_avx3 = _mm512_maskz_loadu_ps(mask, &ot_bias_ptr[c]);

        __m512 ct_avx3 = _mm512_maskz_loadu_ps(mask, &ct_ptr[c]);
        __m512 ht_avx3;

        compute(it_avx3, it_hu_avx3, it_bias_avx3,
                ft_avx3, ft_hu_avx3, ft_bias_avx3,
                gt_avx3, gt_hu_avx3, gt_bias_avx3,
                ot_avx3, ot_hu_avx3, ot_bias_avx3,
                ct_avx3, ht_avx3);

        _mm512_mask_storeu_ps(&ct_ptr[c], mask, ct_avx3);
        _mm512_mask_storeu_ps(&ht_ptr[c], mask, ht_avx3);
      }
    }
  }
#endif

private:
  using LstmBase<T>::ref_exp;
  using LstmBase<T>::ref_tanh;
#if defined(CPU_CAPABILITY_AVX512)
  using LstmBase<T>::avx3_exp;
  using LstmBase<T>::avx3_tanh;
#endif

  // Inputs
  using LstmBase<T>::seq_length;
  using LstmBase<T>::batch_size;
  using LstmBase<T>::input_size;
  using LstmBase<T>::rnn_size;
  using LstmBase<T>::proj_size;
  using LstmBase<T>::batch_first;

  // Temporary value
  using LstmBase<T>::xw;
  using LstmBase<T>::hu;
  using LstmBase<T>::ct;
  using LstmBase<T>::ht;
  using LstmBase<T>::htp;

  // Weights
  using LstmBase<T>::w_ifgo;
  using LstmBase<T>::u_ifgo;
  using LstmBase<T>::bias;
  using LstmBase<T>::w_projection;
};

template <typename T>
class OptimzedLSTMOp : public torch::CustomClassHolder {
public:
  OptimzedLSTMOp() {
    layer_ = nullptr;
    layer_reverse_ = nullptr;
  }

  ~OptimzedLSTMOp() {
    if (layer_ != nullptr) {
      delete[] layer_;
      layer_ = nullptr;
      // std::cout << " Delete OptimzedLSTMOp" << std::endl;
    }
    if (layer_reverse_ != nullptr) {
      delete[] layer_reverse_;
      layer_reverse_ = nullptr;
    }
  }

  void init(int64_t input_size, int64_t hidden_size, int64_t num_layers,
            bool batch_first, bool bidirectional, int64_t proj_size) {
    input_size_ = input_size;
    hidden_size_ = hidden_size;
    num_layers_ = num_layers;
    proj_size_ = proj_size;
    batch_first_ = batch_first;
    bidirectional_ = bidirectional;
    kernel_setted_ = false;
    layer_ = nullptr;
    layer_reverse_ = nullptr;
    mode_ = 0; // mode 0: null, 1: lstm_batch, 2: lstm_packed
    num_directions = 1;
    if (true == bidirectional) {
      num_directions = 2;
    }
  }

  /*
    lstm_batch is an optimized LSTM implementation in inference for single batch input
    and multi-batch inputs.
    like: input([[A B C D E F G],
                 [L M N O P Q W],
                 [H I J K R S T]])
          or input([[A B C D E F G]])
          shape(A)=[200]
  */
  std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> lstm_batch(const torch::Tensor &input,
      const torch::TensorList &hx, const torch::TensorList &flat_weights, bool bias,
      int64_t num_layers, bool bidirectional, bool batch_first) {
    assert(num_layers == num_layers_);
    assert(bidirectional == bidirectional_);
    assert(batch_first == batch_first_);
    int64_t input_dimensions = input.dim();
    int64_t batch_size = 0;
    int64_t sequence_length = 0;

    if (input_dimensions == 2) {
      batch_size = 1;
      sequence_length = input.sizes()[0];
      assert(input_size_ == input.sizes()[1]);
    }
    else if (input_dimensions == 3) {
      if (batch_first == true) {
        batch_size = input.sizes()[0];
        sequence_length = input.sizes()[1];
      }
      else {
        sequence_length = input.sizes()[0];
        batch_size = input.sizes()[1];
      }
      assert(input_size_ == input.sizes()[2]);
    }

    assert(3 == hx[0].dim());
    assert(3 == hx[1].dim());
    assert(hidden_size_ == hx[0].sizes()[2] || proj_size_ == hx[0].sizes()[2]);
    assert(hidden_size_ == hx[1].sizes()[2]);

    if (mode_ != 1) {
      if (layer_ != nullptr) {
        delete[] layer_;
        layer_ = nullptr;
        // std::cout << " Delete LstmPacked" << std::endl;
      }
      if (layer_reverse_ != nullptr) {
        delete[] layer_reverse_;
        layer_reverse_ = nullptr;
      }

      layer_ = new LstmBatch<T>[num_layers_];
      if (true == bidirectional) {
        layer_reverse_ = new LstmBatch<T>[num_layers_];
      }

      for (int i = 0; i < num_layers_; ++i) {
        int64_t input_size = 0;
        if (i == 0)
          input_size = input_size_;
        else {
          if (proj_size_ > 0)
            input_size = proj_size_ * num_directions;
          else
            input_size = hidden_size_ * num_directions;
        }

        layer_[i].init(input_size, hidden_size_, proj_size_, bias);
        if (true == bidirectional) {
          layer_reverse_[i].init(input_size, hidden_size_, proj_size_, bias);
        }
      }

      kernel_setted_ = false;
      mode_ = 1;
      // std::cout << "New LstmBatch" << std::endl;
    }

    if (kernel_setted_ == false) {
      int64_t idx = 0;
      for (int i = 0; i < num_layers_; ++i) {
        const torch::Tensor &weight_ih = flat_weights[idx++];
        const torch::Tensor &weight_hh = flat_weights[idx++];
        if (bias) {
          const torch::Tensor &bias_ih = flat_weights[idx++];
          const torch::Tensor &bias_hh = flat_weights[idx++];
          if (proj_size_ > 0) {
            const torch::Tensor &proj = flat_weights[idx++];
            std::call_once(layer_[i].flag_set_kernel, [&]{ layer_[i].set_kernel(
                            weight_ih.data_ptr<T>(),
                            bias_ih.data_ptr<T>(),
                            weight_hh.data_ptr<T>(),
                            bias_hh.data_ptr<T>(),
                            proj.data_ptr<T>()); });
          }
          else {
            std::call_once(layer_[i].flag_set_kernel, [&]{ layer_[i].set_kernel(
                            weight_ih.data_ptr<T>(),
                            bias_ih.data_ptr<T>(),
                            weight_hh.data_ptr<T>(),
                            bias_hh.data_ptr<T>(),
                            nullptr); });
          }
        }
        else {
          if (proj_size_ > 0) {
            const torch::Tensor &proj = flat_weights[idx++];
            std::call_once(layer_[i].flag_set_kernel, [&]{ layer_[i].set_kernel(
                            weight_ih.data_ptr<T>(),
                            nullptr,
                            weight_hh.data_ptr<T>(),
                            nullptr,
                            proj.data_ptr<T>()); });
          }
          else {
            std::call_once(layer_[i].flag_set_kernel, [&]{ layer_[i].set_kernel(
                            weight_ih.data_ptr<T>(),
                            nullptr,
                            weight_hh.data_ptr<T>(),
                            nullptr,
                            nullptr); });
          }
        }

        if (true == bidirectional) {
          const torch::Tensor &weight_ih_reverse = flat_weights[idx++];
          const torch::Tensor &weight_hh_reverse = flat_weights[idx++];
          if (bias) {
            const torch::Tensor &bias_ih_reverse = flat_weights[idx++];
            const torch::Tensor &bias_hh_reverse = flat_weights[idx++];
            if (proj_size_ > 0) {
              const torch::Tensor &proj_reverse = flat_weights[idx++];
              std::call_once(layer_reverse_[i].flag_set_kernel, [&]{ layer_reverse_[i].set_kernel(
                              weight_ih_reverse.data_ptr<T>(),
                              bias_ih_reverse.data_ptr<T>(),
                              weight_hh_reverse.data_ptr<T>(),
                              bias_hh_reverse.data_ptr<T>(),
                              proj_reverse.data_ptr<T>()); });
            }
            else {
              std::call_once(layer_reverse_[i].flag_set_kernel, [&]{ layer_reverse_[i].set_kernel(
                              weight_ih_reverse.data_ptr<T>(),
                              bias_ih_reverse.data_ptr<T>(),
                              weight_hh_reverse.data_ptr<T>(),
                              bias_hh_reverse.data_ptr<T>(),
                              nullptr); });
            }
          }
          else {
            if (proj_size_ > 0) {
              const torch::Tensor &proj_reverse = flat_weights[idx++];
              std::call_once(layer_reverse_[i].flag_set_kernel, [&]{ layer_reverse_[i].set_kernel(
                              weight_ih_reverse.data_ptr<T>(),
                              nullptr,
                              weight_hh_reverse.data_ptr<T>(),
                              nullptr,
                              proj_reverse.data_ptr<T>()); });
            }
            else {
              std::call_once(layer_reverse_[i].flag_set_kernel, [&]{ layer_reverse_[i].set_kernel(
                              weight_ih_reverse.data_ptr<T>(),
                              nullptr,
                              weight_hh_reverse.data_ptr<T>(),
                              nullptr,
                              nullptr); });
            }
          }
        }
      }
      kernel_setted_ = true;
    }

    torch::Tensor result = input;
    int64_t hidden_size = hidden_size_;
    if (proj_size_ > 0)
      hidden_size = proj_size_;
    torch::Tensor hy = torch::zeros({num_layers * num_directions, batch_size, hidden_size}, torch::kFloat32);
    torch::Tensor cy = torch::zeros({num_layers * num_directions, batch_size, hidden_size_}, torch::kFloat32);
    int64_t idx = 0;

    for (int i = 0; i < num_layers_; ++i) {
      layer_[i].set_input(sequence_length, batch_size, batch_first);

      const torch::Tensor &h0 = hx[0][idx];
      const torch::Tensor &c0 = hx[1][idx];
      layer_[i].set_initial_state(h0.data_ptr<T>(), c0.data_ptr<T>());

      torch::Tensor output = torch::zeros({result.sizes()[0], result.sizes()[1], hidden_size}, torch::kFloat32);
      layer_[i].forward(output.data_ptr<T>(),
                        result.data_ptr<T>(),
                        hy[idx].data_ptr<T>(),
                        cy[idx].data_ptr<T>(),
                        nullptr);
      idx++;

      if (true == bidirectional) {
        layer_reverse_[i].set_input(sequence_length, batch_size, batch_first);

        const torch::Tensor &h0_reverse = hx[0][idx];
        const torch::Tensor &c0_reverse = hx[1][idx];
        layer_reverse_[i].set_initial_state(h0_reverse.data_ptr<T>(), c0_reverse.data_ptr<T>());

        torch::Tensor output_reverse = torch::zeros({result.sizes()[0], result.sizes()[1], hidden_size}, torch::kFloat32);
        layer_reverse_[i].forward_reverse(output_reverse.data_ptr<T>(), // output_reverse
                                          result.data_ptr<T>(),
                                          hy[idx].data_ptr<T>(),
                                          cy[idx].data_ptr<T>(),
                                          nullptr);
        idx++;
        result = torch::cat({output, output_reverse}, 2);
      }
      else {
        result = output;
      }
    }

    return std::make_tuple(std::move(result), std::move(hy), std::move(cy));
  }

  /*
    lstm_packed is an optimized LSTM implementation in inference for PackedSequence inputs.
    like: input([[A B C D E F G],
                 [L M N O P Q 0],
                 [H I J K 0 0 0]]),
          batch_sizes([3, 3, 3, 3, 2, 2, 1])
          shape(A)=[200]
  */
  std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> lstm_packed(const torch::Tensor &input,
      const torch::Tensor batch_sizes, const torch::TensorList &hx,
      const torch::TensorList &flat_weights, bool bias, int64_t num_layers, bool bidirectional) {
    assert(2 == input.dim());
    assert(input_size_ == input.sizes()[1]);

    assert(3 == hx[0].dim());
    assert(3 == hx[1].dim());
    assert(hidden_size_ == hx[0].sizes()[2] || proj_size_ == hx[0].sizes()[2]);
    assert(hidden_size_ == hx[1].sizes()[2]);

    assert(1 == batch_sizes.dim());
    // std::cout << "sequence_length: " << batch_sizes.sizes()[0] << std::endl;
    // std::cout << "batch_size: " << batch_sizes.data_ptr<int64_t>()[0] << std::endl;
    int64_t sequence_length = batch_sizes.sizes()[0];
    int64_t batch_size = batch_sizes.data_ptr<int64_t>()[0];

    assert(num_layers == num_layers_);
    if (mode_ != 2) {
      if (layer_ != nullptr) {
        delete[] layer_;
        layer_ = nullptr;
        // std::cout << "Delete LstmBatch" << std::endl;
      }
      if (layer_reverse_ != nullptr) {
        delete[] layer_reverse_;
        layer_reverse_ = nullptr;
      }

      layer_ = new LstmPacked<T>[num_layers_];
      if (true == bidirectional) {
        layer_reverse_ = new LstmPacked<T>[num_layers_];
      }

      for (int i = 0; i < num_layers_; ++i) {
        int64_t input_size = 0;
        if (i == 0)
          input_size = input_size_;
        else {
          if (proj_size_ > 0)
            input_size = proj_size_ * num_directions;
          else
            input_size = hidden_size_ * num_directions;
        }

        layer_[i].init(input_size, hidden_size_, proj_size_, bias);
        if (true == bidirectional) {
          layer_reverse_[i].init(input_size, hidden_size_, proj_size_, bias);
        }
      }

      kernel_setted_ = false;
      mode_ = 2;
      // std::cout << "New LstmPacked" << std::endl;
    }

    if (kernel_setted_ == false) {
      int64_t idx = 0;
      for (int i = 0; i < num_layers_; ++i) {
        const torch::Tensor &weight_ih = flat_weights[idx++];
        const torch::Tensor &weight_hh = flat_weights[idx++];
        if (bias) {
          const torch::Tensor &bias_ih = flat_weights[idx++];
          const torch::Tensor &bias_hh = flat_weights[idx++];
          if (proj_size_ > 0) {
            const torch::Tensor &proj = flat_weights[idx++];
            std::call_once(layer_[i].flag_set_kernel, [&]{ layer_[i].set_kernel(
                            weight_ih.data_ptr<T>(),
                            bias_ih.data_ptr<T>(),
                            weight_hh.data_ptr<T>(),
                            bias_hh.data_ptr<T>(),
                            proj.data_ptr<T>()); });
          }
          else {
            std::call_once(layer_[i].flag_set_kernel, [&]{ layer_[i].set_kernel(
                            weight_ih.data_ptr<T>(),
                            bias_ih.data_ptr<T>(),
                            weight_hh.data_ptr<T>(),
                            bias_hh.data_ptr<T>(),
                            nullptr); });
          }
        }
        else {
          if (proj_size_ > 0) {
            const torch::Tensor &proj = flat_weights[idx++];
            std::call_once(layer_[i].flag_set_kernel, [&]{ layer_[i].set_kernel(
                            weight_ih.data_ptr<T>(),
                            nullptr,
                            weight_hh.data_ptr<T>(),
                            nullptr,
                            proj.data_ptr<T>());  });
          }
          else {
            std::call_once(layer_[i].flag_set_kernel, [&]{ layer_[i].set_kernel(
                            weight_ih.data_ptr<T>(),
                            nullptr,
                            weight_hh.data_ptr<T>(),
                            nullptr,
                            nullptr); });
          }
        }

        if (true == bidirectional) {
          const torch::Tensor &weight_ih_reverse = flat_weights[idx++];
          const torch::Tensor &weight_hh_reverse = flat_weights[idx++];
          if (bias) {
            const torch::Tensor &bias_ih_reverse = flat_weights[idx++];
            const torch::Tensor &bias_hh_reverse = flat_weights[idx++];
            if (proj_size_ > 0) {
              const torch::Tensor &proj_reverse = flat_weights[idx++];
              std::call_once(layer_reverse_[i].flag_set_kernel, [&]{ layer_reverse_[i].set_kernel(
                              weight_ih_reverse.data_ptr<T>(),
                              bias_ih_reverse.data_ptr<T>(),
                              weight_hh_reverse.data_ptr<T>(),
                              bias_hh_reverse.data_ptr<T>(),
                              proj_reverse.data_ptr<T>()); });
            }
            else {
              std::call_once(layer_reverse_[i].flag_set_kernel, [&]{ layer_reverse_[i].set_kernel(
                              weight_ih_reverse.data_ptr<T>(),
                              bias_ih_reverse.data_ptr<T>(),
                              weight_hh_reverse.data_ptr<T>(),
                              bias_hh_reverse.data_ptr<T>(),
                              nullptr); });
            }
          }
          else {
            if (proj_size_ > 0) {
              const torch::Tensor &proj_reverse = flat_weights[idx++];
              std::call_once(layer_reverse_[i].flag_set_kernel, [&]{ layer_reverse_[i].set_kernel(
                              weight_ih_reverse.data_ptr<T>(),
                              nullptr,
                              weight_hh_reverse.data_ptr<T>(),
                              nullptr,
                              proj_reverse.data_ptr<T>()); });
            }
            else {
              std::call_once(layer_reverse_[i].flag_set_kernel, [&]{ layer_reverse_[i].set_kernel(
                              weight_ih_reverse.data_ptr<T>(),
                              nullptr,
                              weight_hh_reverse.data_ptr<T>(),
                              nullptr,
                              nullptr); });
            }
          }
        }
      }
      kernel_setted_ = true;
    }

    torch::Tensor result = input;
    int64_t hidden_size = hidden_size_;
    if (proj_size_ > 0)
      hidden_size = proj_size_;
    torch::Tensor hy = torch::zeros({num_layers * num_directions, batch_size, hidden_size}, torch::kFloat32);
    torch::Tensor cy = torch::zeros({num_layers * num_directions, batch_size, hidden_size_}, torch::kFloat32);
    int64_t idx = 0;

    for (int i = 0; i < num_layers_; ++i) {
      layer_[i].set_input(sequence_length, batch_size, false);

      const torch::Tensor &h0 = hx[0][idx];
      const torch::Tensor &c0 = hx[1][idx];
      layer_[i].set_initial_state(h0.data_ptr<T>(), c0.data_ptr<T>());

      torch::Tensor output = torch::zeros({result.sizes()[0], hidden_size}, torch::kFloat32);
      layer_[i].forward(output.data_ptr<T>(),
                        result.data_ptr<T>(),
                        hy[idx].data_ptr<T>(),
                        cy[idx].data_ptr<T>(),
                        batch_sizes.data_ptr<int64_t>());
      idx++;

      if (true == bidirectional) {
        layer_reverse_[i].set_input(sequence_length, batch_size, false);

        const torch::Tensor &h0_reverse = hx[0][idx];
        const torch::Tensor &c0_reverse = hx[1][idx];
        layer_reverse_[i].set_initial_state(h0_reverse.data_ptr<T>(), c0_reverse.data_ptr<T>());

        torch::Tensor output_reverse = torch::zeros({result.sizes()[0], hidden_size}, torch::kFloat32);
        layer_reverse_[i].forward_reverse(output_reverse.data_ptr<T>(), // output_reverse
                                          result.data_ptr<T>(),
                                          hy[idx].data_ptr<T>(),
                                          cy[idx].data_ptr<T>(),
                                          batch_sizes.data_ptr<int64_t>());
        idx++;
        result = torch::cat({output, output_reverse}, 1);
      }
      else {
        result = output;
      }
    }

    return std::make_tuple(std::move(result), std::move(hy), std::move(cy));
  }

private:
  int64_t input_size_;
  int64_t hidden_size_;
  int64_t num_layers_;
  int64_t proj_size_;
  bool batch_first_;
  bool bidirectional_;
  int num_directions;
  bool kernel_setted_;
  LstmBase<T> *layer_;
  LstmBase<T> *layer_reverse_;
  int64_t mode_;
};

std::tuple<at::Tensor, at::Tensor, at::Tensor> optimized_lstm_batch_kernel_impl(
    const at::Tensor& input,
    const std::vector<at::Tensor>& hx,
    const std::vector<at::Tensor>& params,
    int64_t input_size,
    int64_t hidden_size,
    int64_t proj_size,
    bool has_biases,
    int64_t num_layers,
    bool bidirectional,
    bool batch_first) {
  OptimzedLSTMOp<float> m;
  m.init(input_size, hidden_size, num_layers,
         batch_first, bidirectional, proj_size);
  auto result = m.lstm_batch(input, hx, params, has_biases,
               num_layers, bidirectional, batch_first);
  return result;
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> optimized_lstm_packed_kernel_impl(
    const at::Tensor& input,
    const at::Tensor& batch_sizes,
    const std::vector<at::Tensor>& hx,
    const std::vector<at::Tensor>& params,
    int64_t input_size,
    int64_t hidden_size,
    int64_t proj_size,
    bool has_biases,
    int64_t num_layers,
    bool bidirectional) {
  OptimzedLSTMOp<float> m;
  m.init(input_size, hidden_size, num_layers,
         false, bidirectional, proj_size);
  auto result = m.lstm_packed(input, batch_sizes, hx, params, has_biases,
               num_layers, bidirectional);
  return result;
}

} // anonymous namespace

REGISTER_DISPATCH(optimized_lstm_batch_kernel_stub, &optimized_lstm_batch_kernel_impl);

REGISTER_DISPATCH(optimized_lstm_packed_kernel_stub, &optimized_lstm_packed_kernel_impl);

} // namespace cpu
} // namespace torch_ipex
