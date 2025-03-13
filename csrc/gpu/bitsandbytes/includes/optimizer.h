template <typename T, int BLOCK_SIZE, int NUM_VALS>
class kPercentileClipping {
 private:
  T* g;
  float* gnorm_vec;
  int step;
  const int n;

 public:
  kPercentileClipping(T* g, float* gnorm_vec, int step, const int n)
      : g(g), gnorm_vec(gnorm_vec), step(step), n(n) {}

  SYCL_EXTERNAL void operator()(sycl::nd_item<1> item) const;
};

template <typename T, int OPTIMIZER, int BLOCK_SIZE, int N_PER_TH>
class kOptimizerStatic8bit2StateBlockwise {
 private:
  T* p;
  T* const g;
  unsigned char* state1;
  unsigned char* state2;
  const float beta1;
  const float beta2;
  const float beta3;
  const float alpha;
  const float eps;
  const int step;
  const float lr;
  float* const quantiles1;
  float* const quantiles2;
  float* absmax1;
  float* absmax2;
  float weight_decay;
  const float gnorm_scale;
  const bool skip_zeros;
  const int n;

 public:
  kOptimizerStatic8bit2StateBlockwise(
      T* p,
      T* const g,
      unsigned char* state1,
      unsigned char* state2,
      const float beta1,
      const float beta2,
      const float beta3,
      const float alpha,
      const float eps,
      const int step,
      const float lr,
      float* const quantiles1,
      float* const quantiles2,
      float* absmax1,
      float* absmax2,
      float weight_decay,
      const float gnorm_scale,
      const bool skip_zeros,
      const int n)
      : p(p),
        g(g),
        state1(state1),
        state2(state2),
        beta1(beta1),
        beta2(beta2),
        beta3(beta3),
        alpha(alpha),
        eps(eps),
        step(step),
        lr(lr),
        quantiles1(quantiles1),
        quantiles2(quantiles2),
        absmax1(absmax1),
        absmax2(absmax2),
        weight_decay(weight_decay),
        gnorm_scale(gnorm_scale),
        skip_zeros(skip_zeros),
        n(n) {}

  SYCL_EXTERNAL void operator()(sycl::nd_item<1> item) const;
};

template <typename T, int TILE_SIZE, int THREADS, int NUM_PER_TH, int DATA_TYPE>
class kDequantizeBlockwise {
 private:
  float* code;
  unsigned char* A;
  float* absmax;
  T* out;
  const int blocksize;
  const int n;

 public:
  kDequantizeBlockwise(
      float* code,
      unsigned char* A,
      float* absmax,
      T* out,
      const int blocksize,
      const int n)
      : code(code),
        A(A),
        absmax(absmax),
        out(out),
        blocksize(blocksize),
        n(n) {}

  SYCL_EXTERNAL void operator()(sycl::nd_item<1> item) const;
};