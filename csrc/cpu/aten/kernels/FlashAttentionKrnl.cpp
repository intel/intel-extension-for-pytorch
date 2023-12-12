#include <ATen/Tensor.h>
#include <aten/FlashAttention.h>
#include <torch/all.h>
#include <torch/csrc/autograd/function.h>
#include <limits>
#include "mkl.h"
#include "vec/vec.h"

#include <ATen/Dispatch.h>
#include <ATen/Parallel.h>
#include <ATen/cpu/vec/functional.h>
#include <ATen/cpu/vec/vec.h>
#include <ATen/native/CPUBlas.h>
#include <ATen/native/cpu/utils.h>

inline void _mkl_gemm(
    const CBLAS_LAYOUT layout,
    const CBLAS_TRANSPOSE transa,
    const CBLAS_TRANSPOSE transb,
    const int& m,
    const int& n,
    const int& k,
    const float& alpha,
    const float* a,
    const int& lda,
    const float* b,
    const int& ldb,
    const float& beta,
    float* c,
    const int& ldc) {
  cblas_sgemm(
      layout, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

inline void _mkl_gemm(
    const CBLAS_LAYOUT layout,
    const CBLAS_TRANSPOSE transa,
    const CBLAS_TRANSPOSE transb,
    const int& m,
    const int& n,
    const int& k,
    const double& alpha,
    const double* a,
    const int& lda,
    const double* b,
    const int& ldb,
    const double& beta,
    double* c,
    const int& ldc) {
  cblas_dgemm(
      layout, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

inline void _mkl_gemm(
    const CBLAS_LAYOUT layout,
    const CBLAS_TRANSPOSE transa,
    const CBLAS_TRANSPOSE transb,
    const int& m,
    const int& n,
    const int& k,
    const float& alpha,
    const at::BFloat16* a,
    const int& lda,
    const at::BFloat16* b,
    const int& ldb,
    const float& beta,
    float* c,
    const int& ldc) {
  cblas_gemm_bf16bf16f32(
      layout,
      transa,
      transb,
      m,
      n,
      k,
      alpha,
      (const MKL_BF16*)(a),
      lda,
      (const MKL_BF16*)(b),
      ldb,
      beta,
      c,
      ldc);
}

namespace torch_ipex {
namespace cpu {

namespace {

template <typename scalar_t>
static inline scalar_t* conditional_data_ptr(scalar_t* ptr, scalar_t* ptr2) {
  TORCH_INTERNAL_ASSERT(ptr2 == nullptr);
  return ptr;
}

template <
    typename scalar_t,
    typename std::enable_if_t<
        at::vec::CPU_CAPABILITY::is_reduced_floating_point_v<scalar_t>,
        int> = 0>
static inline scalar_t* conditional_data_ptr(float* ptr, scalar_t* ptr2) {
  return ptr2;
}

inline c10::SymFloat calculate_scale(
    const at::Tensor& query,
    c10::optional<double> scale) {
  const auto softmax_scale = scale.has_value()
      ? scale.value()
      : (c10::SymFloat(1.0) / (c10::SymFloat(query.sym_size(-1)).sqrt()));
  return c10::SymFloat(softmax_scale);
}

/*
 *Caculate the flash attention SDPA.
 *@template scalar_t: q/k/v data type
 *@template q_split_size: q block size
 *@template kv_split_size: kv block size
 *@param output: output result
 *@param logsumexp: logsumexp for backward
 *@param cum_seq_q: to adapt pt kernel; not used
 *@param cum_seq_k: to adapt pt kernel; not used
 *@param max_q: to adapt pt kernel; not used
 *@param max_k: to adapt pt kernel; not used
 *@param philox_seed: to adapt pt kernel; not used
 *@param philox_offset: to adapt pt kernel; not used
 *@param debug_attn_mask: to adapt pt kernel; not used
 *@param q: query
 *@param k: key
 *@param v: value
 *@param dropout_p: dropout probability
 *@param is_causal: assume causal attention masking if true
 *@param return_debug_mask
 *@param attention_mask: attention mask
 *@param scale: scaling factor applied prior to softmax
 */
template <typename scalar_t, int64_t q_split_size, int64_t kv_split_size>
void cpu_flash_attention(
    const at::Tensor& output,
    const at::Tensor& logsumexp,
    const at::Tensor& cum_seq_q,
    const at::Tensor& cum_seq_k,
    int64_t& max_q,
    int64_t& max_k,
    const at::Tensor& philox_seed,
    const at::Tensor& philox_offset,
    const at::Tensor& debug_attn_mask,
    const at::Tensor& q,
    const at::Tensor& k,
    const at::Tensor& v,
    double dropout_p,
    bool is_causal,
    bool return_debug_mask,
    c10::optional<at::Tensor> attention_mask,
    c10::optional<double> scale) {
  // Query (Batch x Num_heads  x Q_seq_len  x Dim_per_head)
  //    -> (Batch x Q_seq_len  x Num_heads  x Dim_per_head)
  // Key   (Batch x Num_heads  x KV_seq_len x Dim_per_head)
  //    -> (Batch x KV_seq_len x Num_heads  x Dim_per_head)
  // Value (Batch x Num_heads  x KV_seq_len x Dim_per_head)
  //    -> (Batch x KV_seq_len x Num_heads  x Dim_per_head)
  at::Tensor query = q.transpose(1, 2);
  at::Tensor key = k.transpose(1, 2);
  at::Tensor value = v.transpose(1, 2);

  constexpr bool is_reduced_type = is_reduced_floating_point_v<scalar_t>;
  bool is_bool_mask = attention_mask.has_value() &&
      attention_mask.value().scalar_type() == ScalarType::Bool;
  using accum_t = at::opmath_type<scalar_t>;
  using Vec = at::vec::Vectorized<accum_t>;
  accum_t scaling_factor = calculate_scale(query, scale).as_float_unchecked();
  if (attention_mask.has_value() && (is_reduced_type || is_bool_mask)) {
    attention_mask.value() = attention_mask.value().to(at::kFloat);
  }

  // Sizes
  TORCH_CHECK(
      (query.size(3) == value.size(3)) && (key.size(3) == value.size(3)),
      "scaled_dot_product_attention_flash_attention: Q/K/V should have the same head size");
  int64_t batchSize = query.size(0);
  int64_t qSize = query.size(1);
  int64_t kvSize = value.size(1);
  int64_t num_head = query.size(2);
  int64_t headSize = query.size(3);

  // Strides
  int64_t qStrideB = query.stride(0);
  int64_t qStrideM = query.stride(1);
  int64_t qStrideH = query.stride(2);
  int64_t kStrideB = key.stride(0);
  int64_t kStrideN = key.stride(1);
  int64_t kStrideH = key.stride(2);
  int64_t vStrideB = value.stride(0);
  int64_t vStrideN = value.stride(1);
  int64_t vStrideH = value.stride(2);
  int64_t oStrideB = output.stride(0);
  int64_t oStrideM = output.stride(1);
  int64_t oStrideH = output.stride(2);
  int64_t mStrideB =
      (attention_mask.has_value() && attention_mask.value().size(0) > 1)
      ? attention_mask.value().stride(0)
      : 0;
  int64_t mStrideH =
      (attention_mask.has_value() && attention_mask.value().size(1) > 1)
      ? attention_mask.value().stride(1)
      : 0;
  int64_t mStrideM =
      attention_mask.has_value() ? attention_mask.value().stride(2) : 0;

  int64_t qSplitSize = q_split_size > qSize ? qSize : q_split_size;
  int64_t kvSplitSize = kv_split_size > kvSize ? kvSize : kv_split_size;
  int64_t qSlice = (qSize - 1) / qSplitSize + 1;
  int64_t num_thread = at::get_num_threads();

  const auto dtype = query.scalar_type();
  const auto accumulate_dtype = at::toOpMathType(dtype);

  // allocate per thread temp buf (accumulate type)
  int64_t size_per_thread =
      /* qk     */ qSplitSize * kvSplitSize +
      /* qk_max */ qSplitSize +
      /* qk_sum */ qSplitSize +
      /* dst    */ qSplitSize * headSize;

  at::Tensor buf = at::empty(
      {num_thread, size_per_thread}, query.options().dtype(accumulate_dtype));
  at::Tensor buf_reduced = at::empty(
      {num_thread, qSplitSize, is_reduced_type ? kvSplitSize : 0},
      query.options());
  // Data ptrs
  scalar_t* q_data = query.data_ptr<scalar_t>();
  scalar_t* k_data = key.data_ptr<scalar_t>();
  scalar_t* v_data = value.data_ptr<scalar_t>();
  accum_t* mask_data = attention_mask.has_value()
      ? attention_mask.value().data_ptr<accum_t>()
      : nullptr;
  scalar_t* out_data = output.data_ptr<scalar_t>();
  accum_t* buf_data = buf.data_ptr<accum_t>();
  scalar_t* buf_reduced_data =
      is_reduced_type ? buf_reduced.data_ptr<scalar_t>() : nullptr;

  at::parallel_for(
      0, batchSize * num_head * qSlice, 1, [&](int64_t begin, int64_t end) {
        int64_t i = 0, j = 0, k = 0;
        at::native::data_index_init(
            begin, i, batchSize, j, num_head, k, qSlice);
        int ompIdx = at::get_thread_num();
        accum_t* buf_ptr = buf_data + ompIdx * size_per_thread;
        accum_t* qk_data = buf_ptr;
        accum_t* qk_max_data = qk_data + qSplitSize * kvSplitSize;
        accum_t* qk_sum_data = qk_max_data + qSplitSize;
        accum_t* dst_data = qk_sum_data + qSplitSize;
        scalar_t* qk_reduced_data = is_reduced_type
            ? buf_reduced_data + ompIdx * qSplitSize * kvSplitSize
            : nullptr;

        for (const auto z : c10::irange(begin, end)) {
          (void)z; // Suppress unused variable
          int64_t m = k * qSplitSize;
          int64_t qBlockSize = std::min(qSplitSize, qSize - m);
          // Initialize max and sum
          torch_ipex::cpu::kernel::fill_stub(
              qk_max_data,
              -std::numeric_limits<accum_t>::infinity(),
              qBlockSize);
          torch_ipex::cpu::kernel::fill_stub(
              qk_sum_data, static_cast<accum_t>(0), qBlockSize);
          int64_t num_keys =
              is_causal ? std::min(m + qBlockSize, kvSize) : kvSize;
          for (int64_t n = 0; n < num_keys; n += kvSplitSize) {
            int64_t kvBlockSize = std::min(kvSplitSize, kvSize - n);
            // Calculate scale * q @ k.T
            _mkl_gemm(
                CblasColMajor,
                CblasTrans,
                CblasNoTrans,
                kvBlockSize,
                qBlockSize,
                headSize,
                scaling_factor,
                k_data + i * kStrideB + j * kStrideH + n * kStrideN,
                kStrideN,
                q_data + i * qStrideB + j * qStrideH + m * qStrideM,
                qStrideM,
                static_cast<accum_t>(0),
                qk_data,
                kvBlockSize);
            // Apply causal mask, fill unused with -inf
            if (is_causal && num_keys - n <= kvSplitSize) {
              for (const auto row : c10::irange(qBlockSize)) {
                int64_t last_col = m + row - n;
                accum_t* row_ptr = qk_data + row * kvBlockSize;
                torch_ipex::cpu::kernel::fill_stub(
                    row_ptr + last_col + 1,
                    -std::numeric_limits<accum_t>::infinity(),
                    kvBlockSize - last_col - 1);
              }
            }
            // Update attention weights with attention mask
            if (attention_mask.has_value()) {
              for (int64_t row = 0; row < qBlockSize; ++row) {
                if (is_bool_mask) {
                  // qk <- attn_mask ? qk : -inf
                  auto neg_inf = -std::numeric_limits<accum_t>::infinity();
                  at::vec::map2<accum_t>(
                      [neg_inf](Vec x, Vec m) {
                        return Vec::blendv(Vec(neg_inf), x, m);
                      },
                      qk_data + row * kvBlockSize,
                      qk_data + row * kvBlockSize,
                      mask_data + i * mStrideB + j * mStrideH +
                          (m + row) * mStrideM + n,
                      kvBlockSize);
                } else {
                  // qk <- qk + attn_mask
                  at::vec::map2<accum_t>(
                      [](Vec x, Vec y) { return x + y; },
                      qk_data + row * kvBlockSize,
                      qk_data + row * kvBlockSize,
                      mask_data + i * mStrideB + j * mStrideH +
                          (m + row) * mStrideM + n,
                      kvBlockSize);
                }
              }
            }
            // Update coefficients with Softmax
            accum_t tmp_max = 0, tmp_sum = 0, sum_old = 0, exp_tmp = 0;
            for (int64_t row = 0; row < qBlockSize; ++row) {
              sum_old = qk_sum_data[row];
              // max per row
              tmp_max = at::vec::reduce_all<accum_t>(
                  [](Vec& x, Vec& y) { return at::vec::maximum(x, y); },
                  qk_data + row * kvBlockSize,
                  kvBlockSize);
              tmp_max = qk_max_data[row] > tmp_max ? qk_max_data[row] : tmp_max;
              // qk <- exp(qk - max)
              at::vec::map<accum_t>(
                  [tmp_max](Vec x) { return (x - Vec(tmp_max)).exp(); },
                  qk_data + row * kvBlockSize,
                  qk_data + row * kvBlockSize,
                  kvBlockSize);
              // sum per row
              tmp_sum = at::vec::reduce_all<accum_t>(
                  [](Vec& x, Vec& y) { return x + y; },
                  qk_data + row * kvBlockSize,
                  kvBlockSize);
              // exp_tmp <- exp(max[row] - max)
              exp_tmp = std::exp(qk_max_data[row] - tmp_max);
              // sum[row] <- sum + exp_tmp * sum[row]
              qk_sum_data[row] = tmp_sum + exp_tmp * qk_sum_data[row];
              // max[row] <- max
              qk_max_data[row] = tmp_max;
              // qk <- qk / sum[row]
              accum_t sum_new = qk_sum_data[row];
              at::vec::map<accum_t>(
                  [sum_new](Vec x) { return x / Vec(sum_new); },
                  qk_data + row * kvBlockSize,
                  qk_data + row * kvBlockSize,
                  kvBlockSize);
              if (is_reduced_type) {
                at::vec::convert<accum_t, scalar_t>(
                    qk_data + row * kvBlockSize,
                    qk_reduced_data + row * kvBlockSize,
                    kvBlockSize);
              }
              // dst <- dst * sum_old / sum_new * exp_tmp
              if (n > 0) {
                accum_t sum_cor = sum_old / sum_new;
                at::vec::map<accum_t>(
                    [sum_cor, exp_tmp](Vec x) {
                      return x * Vec(sum_cor) * Vec(exp_tmp);
                    },
                    dst_data + row * headSize,
                    dst_data + row * headSize,
                    headSize);
              }
            }
            // Calculate Softmax(q @ k.T) @ v
            _mkl_gemm(
                CblasColMajor,
                CblasNoTrans,
                CblasNoTrans,
                headSize,
                qBlockSize,
                kvBlockSize,
                static_cast<accum_t>(1),
                v_data + i * vStrideB + j * vStrideH + n * vStrideN,
                vStrideN,
                conditional_data_ptr(qk_data, qk_reduced_data),
                kvBlockSize,
                n == 0 ? static_cast<accum_t>(0) : static_cast<accum_t>(1),
                dst_data,
                headSize);
          }
          // reorder MHA output with strides
          for (int64_t row = 0; row < qBlockSize; ++row) {
            at::vec::map<scalar_t>(
                [](Vec x) { return x; },
                out_data + i * oStrideB + j * oStrideH + m * oStrideM +
                    row * oStrideM,
                dst_data + row * headSize,
                headSize);
          }
          // Move to the next query
          at::native::data_index_step(i, batchSize, j, num_head, k, qSlice);
        }
      });
}

void flash_attention_kernel_impl(
    const at::Tensor& output,
    const at::Tensor& logsumexp,
    const at::Tensor& cum_seq_q,
    const at::Tensor& cum_seq_k,
    int64_t& max_q,
    int64_t& max_k,
    const at::Tensor& philox_seed,
    const at::Tensor& philox_offset,
    const at::Tensor& debug_attn_mask,
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    double dropout_p,
    bool is_causal,
    bool return_debug_mask,
    c10::optional<at::Tensor> attention_mask,
    c10::optional<double> scale) {
  auto q_seq_len = query.size(2);

  AT_DISPATCH_FLOATING_TYPES_AND(
      kBFloat16, query.scalar_type(), "flash_attention", [&] {
        if (q_seq_len >= 768) {
          cpu_flash_attention<scalar_t, 256, 512>(
              output,
              logsumexp,
              cum_seq_q,
              cum_seq_k,
              max_q,
              max_k,
              philox_seed,
              philox_offset,
              debug_attn_mask,
              query,
              key,
              value,
              dropout_p,
              is_causal,
              return_debug_mask,
              attention_mask,
              scale);
        } else if (q_seq_len >= 192) {
          cpu_flash_attention<scalar_t, 64, 512>(
              output,
              logsumexp,
              cum_seq_q,
              cum_seq_k,
              max_q,
              max_k,
              philox_seed,
              philox_offset,
              debug_attn_mask,
              query,
              key,
              value,
              dropout_p,
              is_causal,
              return_debug_mask,
              attention_mask,
              scale);
        } else {
          cpu_flash_attention<scalar_t, 32, 512>(
              output,
              logsumexp,
              cum_seq_q,
              cum_seq_k,
              max_q,
              max_k,
              philox_seed,
              philox_offset,
              debug_attn_mask,
              query,
              key,
              value,
              dropout_p,
              is_causal,
              return_debug_mask,
              attention_mask,
              scale);
        }
      });
}

std::tuple<
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    c10::SymInt,
    c10::SymInt,
    at::Tensor,
    at::Tensor,
    at::Tensor>
flash_attention_kernel_base(
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    double dropout_p,
    bool is_causal,
    bool return_debug_mask,
    c10::optional<at::Tensor> attention_mask,
    c10::optional<double> scale) {
  const auto dtype = query.scalar_type();
  int64_t batchSize = query.size(0);
  int64_t qSize = query.size(2);
  int64_t num_head = query.size(1);
  int64_t headSize = query.size(3);

  TORCH_CHECK(
      c10::isFloatingType(dtype) && dtype != ScalarType::Half,
      "IPEX flash_attention: Expected data type in FP32, FP64, BF16, but got ",
      dtype,
      " instead.");
  TORCH_CHECK(
      dtype == key.scalar_type() && dtype == value.scalar_type(),
      "IPEX flash_attention: Q/K/V should have the same data type");
  TORCH_CHECK(
      !attention_mask.has_value() ||
          dtype == attention_mask.value().scalar_type() ||
          attention_mask.value().scalar_type() == ScalarType::Bool,
      "IPEX flash_attention: Mask should have the same data type as Q/K/V or Bool");
  TORCH_CHECK(
      query.dim() == 4 && key.dim() == 4 && value.dim() == 4,
      "IPEX flash_attention: Accept only 4 dims inputs shape of {B, H, T, K}");
  TORCH_CHECK(
      dropout_p == 0.0,
      "IPEX flash_attention: Currently do not support dropout > 0");
  TORCH_CHECK(
      (query.size(3) == value.size(3)) && (key.size(3) == value.size(3)),
      "IPEX flash_attention: Q/K/V should have the same head size");
  TORCH_CHECK(
      (query.stride(-1) == 1) && (key.stride(-1) == 1) &&
          (value.stride(-1) == 1) &&
          (!attention_mask.has_value() ||
           attention_mask.value().stride(-1) == 1),
      "IPEX flash_attention: Q/K/V/Mask should be continuous on the last dim");
  TORCH_CHECK(
      return_debug_mask == false,
      "IPEX flash_attention: Currently do not support 'return_debug_mask'");

  at::Tensor output =
      at::empty({batchSize, qSize, num_head, headSize}, query.options());
  const auto accumulate_dtype = at::toOpMathType(dtype);
  at::Tensor logsumexp = at::empty(
      {batchSize, qSize, num_head}, query.options().dtype(accumulate_dtype));
  at::Tensor cum_seq_q = at::empty({}, at::kLong);
  at::Tensor cum_seq_k = at::empty({}, at::kLong);
  int64_t max_q = 0;
  int64_t max_k = 0;
  at::Tensor philox_seed = at::empty({}, at::kLong);
  at::Tensor philox_offset = at::empty({}, at::kLong);
  at::Tensor debug_attn_mask = at::empty({}, query.options());

  flash_attention_kernel_impl(
      output,
      logsumexp,
      cum_seq_q,
      cum_seq_k,
      max_q,
      max_k,
      philox_seed,
      philox_offset,
      debug_attn_mask,
      query,
      key,
      value,
      dropout_p,
      is_causal,
      return_debug_mask,
      attention_mask,
      scale);

  output = output.transpose(1, 2);
  logsumexp = logsumexp.transpose(1, 2);

  return std::make_tuple(
      std::move(output),
      std::move(logsumexp),
      std::move(cum_seq_q),
      std::move(cum_seq_k),
      max_q,
      max_k,
      std::move(philox_seed),
      std::move(philox_offset),
      std::move(debug_attn_mask));
}

std::tuple<
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    c10::SymInt,
    c10::SymInt,
    at::Tensor,
    at::Tensor,
    at::Tensor>
flash_attention_kernel(
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    double dropout_p,
    bool is_causal,
    bool return_debug_mask,
    c10::optional<double> scale) {
  RECORD_FUNCTION(
      "torch_ipex::flash_attention_kernel", c10::ArrayRef<c10::IValue>({}));
  return flash_attention_kernel_base(
      query,
      key,
      value,
      dropout_p,
      is_causal,
      return_debug_mask,
      c10::nullopt,
      scale);
}

std::tuple<
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    c10::SymInt,
    c10::SymInt,
    at::Tensor,
    at::Tensor,
    at::Tensor>
flash_attention_mask_kernel(
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    double dropout_p,
    bool is_causal,
    bool return_debug_mask,
    c10::optional<at::Tensor> attention_mask,
    c10::optional<double> scale) {
  RECORD_FUNCTION(
      "torch_ipex::flash_attention_mask_kernel",
      c10::ArrayRef<c10::IValue>({}));
  return flash_attention_kernel_base(
      query,
      key,
      value,
      dropout_p,
      is_causal,
      return_debug_mask,
      attention_mask,
      scale);
}
} // anonymous namespace

ALSO_REGISTER_AVX512_DISPATCH(
    flash_attention_kernel_stub,
    &flash_attention_kernel);
ALSO_REGISTER_AVX512_DISPATCH(
    flash_attention_mask_kernel_stub,
    &flash_attention_mask_kernel);

} // namespace cpu
} // namespace torch_ipex
