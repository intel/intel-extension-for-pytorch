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
        in_stride_s == HS,
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
  auto pos_ptr = t_pos.data_ptr<long>(); // [MB][S]
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
            p = pos_ptr[b * S + s];
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
            for (int h = 0, h2 = 0; h < HR; h += 2, h2++) {
              float sin = sin_start[h2];
              float cos = cos_start[h2];
              float in0 = in_ptr[in_offset_q + h];
              float in1 = in_ptr[in_offset_q + h + offset];
              float out0 = in0 * cos - in1 * sin;
              float out1 = in1 * cos + in0 * sin;
              query_ptr[out_offset_q + h] = out0;
              query_ptr[out_offset_q + h + offset] = out1;
              if (concat_qkv && n < N_KV) {
                in0 = in_ptr[in_offset_k + h];
                in1 = in_ptr[in_offset_k + h + offset];
                out0 = in0 * cos - in1 * sin;
                out1 = in1 * cos + in0 * sin;
                key_ptr[out_offset_k + h] = out0;
                key_ptr[out_offset_k + h + offset] = out1;
              }
            }
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
