#include <ATen/Tensor.h>
#include <aten/RotaryPositionEmbedding.h>
#include <torch/all.h>
#include <torch/csrc/autograd/function.h>
#include "vec/vec.h"

namespace torch_ipex {
namespace cpu {

namespace {
bool is_fused_qkv(at::Tensor& t_in, int64_t hidden_size) {
  auto in_stride_s = t_in.stride(1);
  if (t_in.stride(0) * t_in.size(0) != t_in.numel()) {
    if (t_in.dim() == 4) {
      in_stride_s = t_in.size(2) * t_in.size(3);
    } else if (t_in.dim() == 3) {
      in_stride_s = t_in.size(2);
    }
  }
  if (in_stride_s > hidden_size) {
    return true;
  }
  return false;
}

/**
 * Applies the Rotary Position Embedding Kernel to the input tensors.
 *
 * @param t_in The input tensor. t_in can be either [B][S][F] for concat_qkv
 * output or [B][S][N][H] for query or key only .
 * @param t_emb_pos The tensor containing the rotary position embeddings.
 * t_emb_pos should be [MP][HR] where MP is the max position and HR is the
 * rotary dimension.
 * @param t_pos The tensor containing the positions. t_pos should be [B][S]
 * where B is the batch size and S is the sequence length. In some cases, there
 * is only one element which the past_kv_length.In this case, position id can
 * construced by past_kv_length + current_position
 * @param N The number of heads.
 * @param H The head size.
 * @param offset The offset value. For GPT-J 6B/ChatGLM, cos/sin is applied to
 * the neighboring 2 elements, so the offset is 1. For lamma, cos/sin is applied
 * to the neighboring rotary_dim elements, so the offset is rotary_dim/2.
 * @param rotary_dim The rotary dimension.
 * @return A tuple containing the query, key, and value tensors.
 */

template <typename scalar_t>
inline void apply_rotary_embedding(
    const scalar_t* __restrict__ arr,
    const float* __restrict__ cos_ptr,
    const float* __restrict__ sin_ptr,
    scalar_t* __restrict__ out,
    int embed_dim) {
  using Vec = Vectorized<scalar_t>;
  const int kVecSize = Vec::size();
  const int len = embed_dim - (embed_dim % kVecSize);

  // GPT-J style rotary embedding.
  // format: {d, 2}, stride-2 access need permute to be vectorized.
  int d = 0;
  for (; d < len; d += kVecSize) {
    Vec x = Vec::loadu(arr + 2 * d + 0 * kVecSize);
    Vec y = Vec::loadu(arr + 2 * d + 1 * kVecSize);
    Vec cos = Vec::loadu(cos_ptr + d);
    Vec sin = Vec::loadu(sin_ptr + d);
    // x: {x0, y0, x1, y1, x2, y2, x3, y3}
    // y: {x4, y4, x5, y5, x6, y6, x7, y7}
    // x1: {x0, x1, x2, x3, x4, x5, x6, x7}
    // y1: {y0, y1, y2, y3, y4, y5, y6, y7}
    auto xy = deinterleave2(x, y);
    Vec x1 = std::get<0>(xy);
    Vec y1 = std::get<1>(xy);
    Vec x2 = x1 * cos - y1 * sin;
    Vec y2 = y1 * cos + x1 * sin;
    // x2: {x0, x1, x2, x3, x4, x5, x6, x7}
    // y2: {y0, y1, y2, y3, y4, y5, y6, y7}
    // x_out: {x0, y0, x1, y1, x2, y2, x3, y3}
    // y_out: {x4, y4, x5, y5, x6, y6, x7, y7}
    xy = interleave2(x2, y2);
    Vec x_out = std::get<0>(xy);
    Vec y_out = std::get<1>(xy);
    x_out.store(out + 2 * d + 0 * kVecSize);
    y_out.store(out + 2 * d + 1 * kVecSize);
  }
  for (; d < embed_dim; d++) {
    scalar_t x = arr[2 * d + 0];
    scalar_t y = arr[2 * d + 1];
    scalar_t x_out = x * cos_ptr[d] - y * sin_ptr[d];
    scalar_t y_out = y * cos_ptr[d] + x * sin_ptr[d];
    out[2 * d + 0] = x_out;
    out[2 * d + 1] = y_out;
  }
}

template <>
inline void apply_rotary_embedding<at::BFloat16>(
    const at::BFloat16* __restrict__ arr,
    const float* __restrict__ cos_ptr,
    const float* __restrict__ sin_ptr,
    at::BFloat16* __restrict__ out,
    int embed_dim) {
  using fVec = Vectorized<float>;
  using bVec = Vectorized<at::BFloat16>;

  const int kVecSize = bVec::size();
  const int len = 2 * embed_dim - (2 * embed_dim % kVecSize);

  // GPT-J style rotary embedding.
  // format: {d, 2}, stride-2 access need permute to be vectorized.
  int d = 0;
  for (; d < len; d += kVecSize) {
    bVec a = bVec::loadu(arr + d);
    fVec x, y;
    std::tie(x, y) = convert_bfloat16_float(a);
    fVec cos = fVec::loadu(cos_ptr + d / 2);
    fVec sin = fVec::loadu(sin_ptr + d / 2);
    // x: {x0, y0, x1, y1, x2, y2, x3, y3}
    // y: {x4, y4, x5, y5, x6, y6, x7, y7}
    // x1: {x0, x1, x2, x3, x4, x5, x6, x7}
    // y1: {y0, y1, y2, y3, y4, y5, y6, y7}
    auto xy = deinterleave2(x, y);
    fVec x1 = std::get<0>(xy);
    fVec y1 = std::get<1>(xy);
    fVec x2 = x1 * cos - y1 * sin;
    fVec y2 = y1 * cos + x1 * sin;
    // x2: {x0, x1, x2, x3, x4, x5, x6, x7}
    // y2: {y0, y1, y2, y3, y4, y5, y6, y7}
    // x_out: {x0, y0, x1, y1, x2, y2, x3, y3}
    // y_out: {x4, y4, x5, y5, x6, y6, x7, y7}
    xy = interleave2(x2, y2);
    fVec x_out = std::get<0>(xy);
    fVec y_out = std::get<1>(xy);
    bVec a_out = convert_float_bfloat16(x_out, y_out);
    a_out.store(out + d);
  }
  for (; d < embed_dim; d++) {
    float x = static_cast<float>(arr[2 * d + 0]);
    float y = static_cast<float>(arr[2 * d + 1]);
    float x_out = x * cos_ptr[d] - y * sin_ptr[d];
    float y_out = y * cos_ptr[d] + x * sin_ptr[d];
    out[2 * d + 0] = static_cast<at::BFloat16>(x_out);
    out[2 * d + 1] = static_cast<at::BFloat16>(y_out);
  }
}

template <typename scalar_t>
inline void RotateEveryTwo(
    const scalar_t* in_query_ptr,
    const scalar_t* in_key_ptr,
    scalar_t* out_query_ptr,
    scalar_t* out_key_ptr,
    const float* sin_start,
    const float* cos_start,
    const int HR,
    const int offset,
    const bool calc_key) {
  // TODO: remove overhead for loading sin and cos
  int embed_dim = HR / 2;
  apply_rotary_embedding<scalar_t>(
      in_query_ptr, cos_start, sin_start, out_query_ptr, embed_dim);

  if (calc_key) {
    apply_rotary_embedding<scalar_t>(
        in_key_ptr, cos_start, sin_start, out_key_ptr, embed_dim);
  }
}

template <>
inline void RotateEveryTwo<at::BFloat16>(
    const at::BFloat16* in_query_ptr,
    const at::BFloat16* in_key_ptr,
    at::BFloat16* out_query_ptr,
    at::BFloat16* out_key_ptr,
    const float* sin_ptr,
    const float* cos_ptr,
    const int HR,
    const int offset,
    const bool calc_key) {
  int embed_dim = HR / 2;

  using fVec = Vectorized<float>;
  using bVec = Vectorized<at::BFloat16>;

  const int kVecSize = bVec::size();
  const int len = HR - (HR % kVecSize);

  // GPT-J style rotary embedding.
  // format: {d, 2}, stride-2 access need permute to be vectorized.
  int d = 0;
  for (; d < len; d += kVecSize) {
    bVec in_query = bVec::loadu(in_query_ptr + d);
    fVec x, y;
    std::tie(x, y) = convert_bfloat16_float(in_query);
    fVec cos = fVec::loadu(cos_ptr + d / 2);
    fVec sin = fVec::loadu(sin_ptr + d / 2);
    // x: {x0, y0, x1, y1, x2, y2, x3, y3}
    // y: {x4, y4, x5, y5, x6, y6, x7, y7}
    // x1: {x0, x1, x2, x3, x4, x5, x6, x7}
    // y1: {y0, y1, y2, y3, y4, y5, y6, y7}
    auto xy = deinterleave2(x, y);
    fVec x1 = std::get<0>(xy);
    fVec y1 = std::get<1>(xy);
    fVec x2 = x1 * cos - y1 * sin;
    fVec y2 = y1 * cos + x1 * sin;
    // x2: {x0, x1, x2, x3, x4, x5, x6, x7}
    // y2: {y0, y1, y2, y3, y4, y5, y6, y7}
    // x_out: {x0, y0, x1, y1, x2, y2, x3, y3}
    // y_out: {x4, y4, x5, y5, x6, y6, x7, y7}
    xy = interleave2(x2, y2);
    fVec x_out = std::get<0>(xy);
    fVec y_out = std::get<1>(xy);
    bVec a_out = convert_float_bfloat16(x_out, y_out);
    a_out.store(out_query_ptr + d);
    if (calc_key) {
      bVec in_key = bVec::loadu(in_key_ptr + d);
      fVec x, y;
      std::tie(x, y) = convert_bfloat16_float(in_key);
      // x: {x0, y0, x1, y1, x2, y2, x3, y3}
      // y: {x4, y4, x5, y5, x6, y6, x7, y7}
      // x1: {x0, x1, x2, x3, x4, x5, x6, x7}
      // y1: {y0, y1, y2, y3, y4, y5, y6, y7}
      auto xy = deinterleave2(x, y);
      fVec x1 = std::get<0>(xy);
      fVec y1 = std::get<1>(xy);
      fVec x2 = x1 * cos - y1 * sin;
      fVec y2 = y1 * cos + x1 * sin;
      // x2: {x0, x1, x2, x3, x4, x5, x6, x7}
      // y2: {y0, y1, y2, y3, y4, y5, y6, y7}
      // x_out: {x0, y0, x1, y1, x2, y2, x3, y3}
      // y_out: {x4, y4, x5, y5, x6, y6, x7, y7}
      xy = interleave2(x2, y2);
      fVec x_out = std::get<0>(xy);
      fVec y_out = std::get<1>(xy);
      bVec a_out = convert_float_bfloat16(x_out, y_out);
      a_out.store(out_key_ptr + d);
    }
  }
  for (; d < embed_dim; d++) {
    float x = static_cast<float>(in_query_ptr[2 * d + 0]);
    float y = static_cast<float>(in_query_ptr[2 * d + 1]);
    float cos = cos_ptr[d];
    float sin = sin_ptr[d];
    float x_out = x * cos - y * sin;
    float y_out = y * cos + x * sin;
    out_query_ptr[2 * d + 0] = static_cast<at::BFloat16>(x_out);
    out_query_ptr[2 * d + 1] = static_cast<at::BFloat16>(y_out);
    if (calc_key) {
      float x = static_cast<float>(in_key_ptr[2 * d + 0]);
      float y = static_cast<float>(in_key_ptr[2 * d + 1]);
      float x_out = x * cos - y * sin;
      float y_out = y * cos + x * sin;
      out_key_ptr[2 * d + 0] = static_cast<at::BFloat16>(x_out);
      out_key_ptr[2 * d + 1] = static_cast<at::BFloat16>(y_out);
    }
  }
}

template <typename scalar_t>
inline void RotateEveryTwoNaive(
    const scalar_t* in_query_ptr,
    const scalar_t* in_key_ptr,
    scalar_t* out_query_ptr,
    scalar_t* out_key_ptr,
    const float* sin_start,
    const float* cos_start,
    const int HR,
    const int offset,
    const bool calc_key) {
  int embed_dim = HR / 2;
  for (int h = 0, h2 = 0; h < HR; h += 2, h2++) {
    float sin = sin_start[h2];
    float cos = cos_start[h2];
    float in0 = in_query_ptr[h];
    float in1 = in_query_ptr[h + offset];
    float out0 = in0 * cos - in1 * sin;
    float out1 = in1 * cos + in0 * sin;
    out_query_ptr[h] = out0;
    out_query_ptr[h + offset] = out1;
    if (calc_key) {
      in0 = in_key_ptr[h];
      in1 = in_key_ptr[h + offset];
      out0 = in0 * cos - in1 * sin;
      out1 = in1 * cos + in0 * sin;
      out_key_ptr[h] = out0;
      out_key_ptr[h + offset] = out1;
    }
  }
}

template <>
inline void RotateEveryTwo<at::Half>(
    const at::Half* in_query_ptr,
    const at::Half* in_key_ptr,
    at::Half* out_query_ptr,
    at::Half* out_key_ptr,
    const float* sin_start,
    const float* cos_start,
    const int HR,
    const int offset,
    const bool calc_key) {
  // TODO: vectorized
  RotateEveryTwoNaive<Half>(
      in_query_ptr,
      in_key_ptr,
      out_query_ptr,
      out_key_ptr,
      sin_start,
      cos_start,
      HR,
      offset,
      calc_key);
}

template <typename T>
std::tuple<at::Tensor, at::Tensor, at::Tensor> ApplyROPEKernel(
    at::Tensor& t_in,
    at::Tensor& t_emb_pos,
    at::Tensor& t_pos,
    int64_t N, // N: number of head, H: head size
    int64_t H,
    int64_t offset,
    int64_t rotary_dim) {
  auto in_sizes = t_in.sizes(); // in[B][S][F] or [B][S][N][H]
  auto MP = t_emb_pos.size(0); // Max Pos
  auto HR = t_emb_pos.size(1); // rotary_dim
  auto B = in_sizes[0];
  auto S = in_sizes[1];
  auto HS = in_sizes[2];
  auto in_stride_b = t_in.stride(0);
  auto in_stride_s = t_in.stride(1);
  auto N_KV = N; // GQA/MQA, N_KV: number of head for key/value
  auto concat_qkv = in_stride_s > N * H;

  if (is_fused_qkv(t_in, N * H)) {
    TORCH_CHECK(
        t_in.dim() == 3,
        "The shape of input tensor of rotary_position_embedding should be in (batch, seq_len, qkv_hidden_size) when using fused qkv)");
    N_KV = (HS - N * H) / (2 * H);
  }

  auto COFF = HR / 2;
  auto in_ptr = t_in.data_ptr<T>();
  // initialize empty q/k/v
  auto query = at::empty({B, S, N, H}, t_in.options());
  auto key =
      concat_qkv ? at::empty({B, S, N_KV, H}, t_in.options()) : at::Tensor();
  auto value =
      concat_qkv ? at::empty({B, S, N_KV, H}, t_in.options()) : at::Tensor();
  auto query_ptr = query.data_ptr<T>();
  auto key_ptr = concat_qkv ? key.data_ptr<T>() : nullptr;
  auto value_ptr = concat_qkv ? value.data_ptr<T>() : nullptr;
  auto out_stride_qb = query.stride(0);
  auto out_stride_qs = query.stride(1);
  auto out_stride_kb = concat_qkv ? key.stride(0) : 0;
  auto out_stride_ks = concat_qkv ? key.stride(1) : 0;
  auto emb_pos_ptr = t_emb_pos.data_ptr<float>(); // [MP][HR]
  auto pos_ptr = t_pos.data_ptr<long>(); // [B][S] or [1][S]
  bool t_pos_no_repeated_for_batch = false;
  if (t_pos.numel() != 1 && t_pos.size(0) == 1 && B > 1) {
    // we do not perform t_pos.repeat here to avoid the overhead of copying
    t_pos_no_repeated_for_batch = true;
  }
  {
#pragma omp parallel for collapse(3)
    for (int b = 0; b < B; b++) {
      for (int s = 0; s < S; s++) {
        for (int n = 0; n < N; n++) {
          auto in_offset_q = b * in_stride_b + s * in_stride_s + n * H;
          auto out_offset_q = b * out_stride_qb + s * out_stride_qs + n * H;
          auto out_offset_k =
              concat_qkv ? b * out_stride_kb + s * out_stride_ks + n * H : 0;
          auto in_offset_k = concat_qkv ? in_offset_q + N * H : 0;
          long p = 0;
          float* sin_start = nullptr;
          float* cos_start = nullptr;
          // step 0) get the rotary position embedding for the current position
          if (t_pos.numel() == 1) { // used by Falcon & ChatGLM,
            p = pos_ptr[0];
            sin_start = emb_pos_ptr + (p + s) * HR;
            cos_start = emb_pos_ptr + (p + s) * HR + COFF;
          } else {
            auto start_idx = t_pos_no_repeated_for_batch ? 0 : b * S;
            p = pos_ptr[start_idx + s];
            sin_start = emb_pos_ptr + p * HR;
            cos_start = emb_pos_ptr + p * HR + COFF;
          }
          // step 1) apply_rotary_pos_emb for the rotary_dim elements in every
          // head of query/key
          if (offset !=
              1) { // use vectorized version if there are more than 16
                   // continuous elements, used by lamma/gpt-neox/falcon
                   // logic is like to the rotate_half in python code
            torch_ipex::cpu::kernel::apply_rope_along_head_kernel<T>(
                in_ptr + in_offset_q,
                query_ptr + out_offset_q,
                cos_start,
                sin_start,
                rotary_dim,
                offset);
            if (concat_qkv && n < N_KV) {
              torch_ipex::cpu::kernel::apply_rope_along_head_kernel<T>(
                  in_ptr + in_offset_k,
                  key_ptr + out_offset_k,
                  cos_start,
                  sin_start,
                  rotary_dim,
                  offset);
            }
          } else { // used by GPT-J 6B & CodeGen & ChatGLM
                   // logic is like to the rotate_every_two in python code
            RotateEveryTwo<T>(
                &in_ptr[in_offset_q],
                &in_ptr[in_offset_k],
                &query_ptr[out_offset_q],
                &key_ptr[out_offset_k],
                sin_start,
                cos_start,
                HR,
                offset,
                (concat_qkv && n < N_KV));
          }
          // step 2) copy the rest of the input tensor to query/key (query_pass
          // & key_pass)
          if (rotary_dim < H) {
            torch_ipex::cpu::kernel::move_ker<T, T>(
                query_ptr + out_offset_q + rotary_dim,
                in_ptr + in_offset_q + rotary_dim,
                H - rotary_dim);
            if (concat_qkv && n < N_KV) {
              torch_ipex::cpu::kernel::move_ker<T, T>(
                  key_ptr + out_offset_k + rotary_dim,
                  in_ptr + in_offset_k + rotary_dim,
                  H - rotary_dim);
            }
          }
          // step 3) copy value from t_in when concat_qkv is true
          if (concat_qkv && n < N_KV) {
            auto in_offset_v = in_offset_k + N_KV * H;
            torch_ipex::cpu::kernel::move_ker<T, T>(
                value_ptr + out_offset_k, in_ptr + in_offset_v, H);
          }
        }
      }
    }
  }
  return std::make_tuple(query, key, value);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor>
rotary_position_embedding_kernel_impl(
    at::Tensor& t_in,
    at::Tensor& t_emb_pos,
    at::Tensor& t_pos,
    int64_t N, // N: number of head, H: head size
    int64_t H,
    int64_t offset,
    int64_t rotary_dim) {
  t_in = t_in.contiguous();
  t_emb_pos = t_emb_pos.contiguous();
  t_pos = t_pos.contiguous();
  if (t_in.scalar_type() == at::kFloat) {
    return ApplyROPEKernel<float>(
        t_in, t_emb_pos, t_pos, N, H, offset, rotary_dim);
  } else if (t_in.scalar_type() == at::kBFloat16) {
    return ApplyROPEKernel<at::BFloat16>(
        t_in, t_emb_pos, t_pos, N, H, offset, rotary_dim);
  } else if (t_in.scalar_type() == at::kHalf) {
    return ApplyROPEKernel<at::Half>(
        t_in, t_emb_pos, t_pos, N, H, offset, rotary_dim);
  } else {
    TORCH_CHECK(
        false,
        "rotary_position_embedding_kernel_impl: unsupported '",
        t_in.scalar_type(),
        "'");
    return std::make_tuple(at::Tensor(), at::Tensor(), at::Tensor());
  }
}

} // anonymous namespace

IPEX_REGISTER_DISPATCH(
    rotary_position_embedding_kernel_stub,
    &rotary_position_embedding_kernel_impl);

} // namespace cpu
} // namespace torch_ipex
