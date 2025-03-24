#include <aten/Conv.h>
#include "mkl.h"
#include "vec/vec.h"

namespace torch_ipex {
namespace cpu {
namespace {

template <typename T>
std::tuple<at::Tensor, at::Tensor> selective_scan_kernel_inner(
    const at::Tensor& u,
    const at::Tensor& delta,
    const at::Tensor& A,
    const at::Tensor& B,
    const at::Tensor& C,
    const c10::optional<at::Tensor>& D,
    const c10::optional<at::Tensor>& z,
    const c10::optional<at::Tensor>& delta_bias,
    bool delta_softplus,
    bool return_last_state) {
  using Vec = at::vec::Vectorized<T>;
  using fVec = at::vec::Vectorized<float>;
  auto vec_size = Vec::size();
  auto fvec_size = fVec::size();
  auto batch = u.size(0);
  bool transposed_A_dt_u =
      (A.size(-1) == u.size(-1) && A.size(-1) == delta.size(-1));
  auto A2 = A;
  auto u2 = u;
  auto delta2 = delta;
  if (!transposed_A_dt_u) {
    A2 = A.transpose(-1, -2).contiguous();
    u2 = u.transpose(-1, -2).contiguous();
    delta2 = delta.transpose(-1, -2).contiguous();
  }
  auto dim = A2.size(1);
  auto len = u2.size(1);
  auto dstate = A2.size(0);
  auto x = at::zeros({batch, dstate, dim}, at::kFloat);
  auto out = at::zeros({batch, len, dim}, u.options());
  auto B_dim = B.dim();
  auto C_dim = C.dim();
  auto delta_ptr = delta2.data_ptr<float>();
  auto u_ptr = u2.data_ptr<T>();
  auto A_ptr = A2.data_ptr<float>();
  auto x_ptr = x.data_ptr<float>();
  auto out_ptr = out.data_ptr<T>();
  bool has_dt_bias = delta_bias.has_value();
  auto delta_bias_ptr =
      has_dt_bias ? delta_bias.value().data_ptr<float>() : nullptr;
  bool has_D = D.has_value();
  auto D_ptr = has_D ? D.value().data_ptr<float>() : nullptr;
  bool has_z = z.has_value();
  auto z_ptr = has_z ? z.value().data_ptr<T>() : nullptr;
  auto z_strideB = has_z ? z.value().stride(0) : 0;
  auto z_strideD = has_z ? z.value().stride(1) : 0;
  auto z_strideL = has_z ? z.value().stride(2) : 0;
  auto delta_strideB = delta2.stride(0);
  auto delta_strideD = delta2.stride(2);
  auto delta_strideL = delta2.stride(1);
  auto u_strideB = u2.stride(0);
  auto u_strideD = u2.stride(2);
  auto u_strideL = u2.stride(1);
  auto A_strideD = A2.stride(1);
  auto A_strideS = A2.stride(0);
  auto x_strideB = x.stride(0);
  auto x_strideD = x.stride(2);
  auto x_strideS = x.stride(1);
  auto out_strideB = out.stride(0);
  auto out_strideD = out.stride(2);
  auto out_strideL = out.stride(1);
  auto B_ptr = B.data_ptr<T>();
  auto C_ptr = C.data_ptr<T>();
  auto B_stride0 = B.stride(0);
  auto B_stride1 = B.stride(1);
  auto B_stride2 = B_dim >= 3 ? B.stride(2) : 0;
  auto B_stride3 = B_dim == 4 ? B.stride(3) : 0;
  auto C_stride0 = C.stride(0);
  auto C_stride1 = C.stride(1);
  auto C_stride2 = C_dim >= 3 ? C.stride(2) : 0;
  auto C_stride3 = C_dim == 4 ? C.stride(3) : 0;
  auto BC_group = B_dim == 4 ? B.size(1) : 1;
  auto one_vec = Vec(1);
  auto one_vec_fp32 = fVec(1);
  auto threshold = 80.0f;
  auto threshold_vec = fVec(threshold);

#pragma omp parallel for collapse(2)
  for (auto bi = 0; bi < batch; bi++) {
    for (auto li = 0; li < len; li++) {
      auto di = 0;
      for (; di < dim - (dim % vec_size); di += vec_size) {
        fVec dt_fvec0 = fVec::loadu(
            delta_ptr + bi * delta_strideB + di * delta_strideD +
            li * delta_strideL);
        fVec dt_fvec1 = fVec::loadu(
            delta_ptr + bi * delta_strideB + di * delta_strideD +
            li * delta_strideL + fvec_size);
        if (has_dt_bias) {
          fVec dt_bias_fvec0 = fVec::loadu(delta_bias_ptr + di);
          fVec dt_bias_fvec1 = fVec::loadu(delta_bias_ptr + di + fvec_size);
          dt_fvec0 += dt_bias_fvec0;
          dt_fvec1 += dt_bias_fvec1;
        }
        if (delta_softplus) {
          dt_fvec0 = fVec::blendv(
              dt_fvec0.exp().log1p(), dt_fvec0, dt_fvec0 > threshold_vec);
          dt_fvec1 = fVec::blendv(
              dt_fvec1.exp().log1p(), dt_fvec1, dt_fvec1 > threshold_vec);
        }
        dt_fvec0.store(
            delta_ptr + bi * delta_strideB + di * delta_strideD +
            li * delta_strideL);
        dt_fvec1.store(
            delta_ptr + bi * delta_strideB + di * delta_strideD +
            li * delta_strideL + fvec_size);
      }
      for (; di < dim; di++) {
        auto dt_idx =
            bi * delta_strideB + di * delta_strideD + li * delta_strideL;
        float delta_val = delta_ptr[dt_idx];
        if (has_dt_bias) {
          delta_val += delta_bias_ptr[di];
        }
        if (delta_softplus) {
          delta_val = delta_val > threshold ? delta_val
                                            : std::log1p(std::exp(delta_val));
        }
        delta_ptr[dt_idx] = delta_val;
      }
    }
  }
#pragma omp parallel for collapse(2)
  for (auto bi = 0; bi < batch; bi++) {
    for (auto di = 0; di < dim; di++) {
      for (auto li = 0; li < len; li++) {
        float delta_val = delta_ptr
            [bi * delta_strideB + di * delta_strideD + li * delta_strideL];
        float u_val = u_ptr[bi * u_strideB + di * u_strideD + li * u_strideL];
        auto out_idx = bi * out_strideB + di * out_strideD + li * out_strideL;
        float dt_u_mul = delta_val * u_val;
        float out_val = out_ptr[out_idx];
        for (auto dsi = 0; dsi < dstate; dsi++) {
          auto x_idx = bi * x_strideB + di * x_strideD + dsi * x_strideS;
          float x_val = x_ptr[x_idx];
          float deltaA_A_mul =
              delta_val * A_ptr[A_strideD * di + dsi * A_strideS];
          float deltaA =
              deltaA_A_mul > threshold ? deltaA_A_mul : std::exp(deltaA_A_mul);
          x_val *= deltaA;
          if (B_dim == 2) { // dim x dstate
            float B_val = B_ptr[di * B_stride0 + dsi * B_stride1];
            x_val += B_val * dt_u_mul;
          } else if (B_dim == 3) { // batch x dstate x len
            float B_val =
                B_ptr[bi * B_stride0 + dsi * B_stride1 + li * B_stride2];
            x_val += B_val * dt_u_mul;
          } else { // batch x BC_group x dstate x len
            float B_val = B_ptr
                [bi * B_stride0 + int(di / BC_group) * B_stride1 +
                 dsi * B_stride2 + li * B_stride3];
            x_val += B_val * dt_u_mul;
          }

          if (C_dim == 2) {
            float C_val = C_ptr[di * C_stride0 + dsi * C_stride1];
            out_val += x_val * C_val;
          } else if (C_dim == 3) {
            float C_val =
                C_ptr[bi * C_stride0 + dsi * C_stride1 + li * C_stride2];
            out_val += x_val * C_val;
          } else {
            float C_val = C_ptr
                [bi * C_stride0 + int(di / BC_group) * C_stride1 +
                 dsi * C_stride2 + li * C_stride3];
            out_val += x_val * C_val;
          }
          x_ptr[x_idx] = x_val;
        }
        if (has_D) {
          out_val += u_val * D_ptr[di];
        }
        if (has_z) {
          float z_val = z_ptr[bi * z_strideB + di * z_strideD + li * z_strideL];
          out_val *= z_val / (1 + expf(-z_val));
        }
        out_ptr[out_idx] = out_val;
      }
    }
  }
  return std::make_tuple(
      std::move(out.transpose_(1, 2)), std::move(x.transpose_(1, 2)));
}

template <>
std::tuple<at::Tensor, at::Tensor> selective_scan_kernel_inner<float>(
    const at::Tensor& u,
    const at::Tensor& delta,
    const at::Tensor& A,
    const at::Tensor& B,
    const at::Tensor& C,
    const c10::optional<at::Tensor>& D,
    const c10::optional<at::Tensor>& z,
    const c10::optional<at::Tensor>& delta_bias,
    bool delta_softplus,
    bool return_last_state) {
  auto vec_size = at::vec::Vectorized<float>::size();
  auto batch = u.size(0);
  bool transposed_A_dt_u =
      (A.size(-1) == u.size(-1) && A.size(-1) == delta.size(-1));
  auto A2 = A;
  auto u2 = u;
  auto delta2 = delta;
  if (!transposed_A_dt_u) {
    A2 = A.transpose(-1, -2).contiguous();
    u2 = u.transpose(-1, -2).contiguous();
    delta2 = delta.transpose(-1, -2).contiguous();
  }
  auto dim = A2.size(1);
  auto len = u2.size(1);
  auto dstate = A2.size(0);
  auto x = at::zeros({batch, dstate, dim}, u.options());
  auto out = at::zeros({batch, len, dim}, u.options());
  auto B_dim = B.dim();
  auto C_dim = C.dim();
  auto delta_ptr = delta2.data_ptr<float>();
  auto u_ptr = u2.data_ptr<float>();
  auto A_ptr = A2.data_ptr<float>();
  auto x_ptr = x.data_ptr<float>();
  auto out_ptr = out.data_ptr<float>();
  bool has_dt_bias = delta_bias.has_value();
  auto delta_bias_ptr =
      has_dt_bias ? delta_bias.value().data_ptr<float>() : nullptr;
  bool has_D = D.has_value();
  auto D_ptr = has_D ? D.value().data_ptr<float>() : nullptr;
  bool has_z = z.has_value();
  auto z_ptr = has_z ? z.value().data_ptr<float>() : nullptr;
  auto z_strideB = has_z ? z.value().stride(0) : 0;
  auto z_strideD = has_z ? z.value().stride(1) : 0;
  auto z_strideL = has_z ? z.value().stride(2) : 0;
  auto delta_strideB = delta2.stride(0);
  auto delta_strideD = delta2.stride(2);
  auto delta_strideL = delta2.stride(1);
  auto u_strideB = u2.stride(0);
  auto u_strideD = u2.stride(2);
  auto u_strideL = u2.stride(1);
  auto A_strideD = A2.stride(1);
  auto A_strideS = A2.stride(0);
  auto x_strideB = x.stride(0);
  auto x_strideD = x.stride(2);
  auto x_strideS = x.stride(1);
  auto out_strideB = out.stride(0);
  auto out_strideD = out.stride(2);
  auto out_strideL = out.stride(1);
  auto B_ptr = B.data_ptr<float>();
  auto C_ptr = C.data_ptr<float>();
  auto B_stride0 = B.stride(0);
  auto B_stride1 = B.stride(1);
  auto B_stride2 = B_dim >= 3 ? B.stride(2) : 0;
  auto B_stride3 = B_dim == 4 ? B.stride(3) : 0;
  auto C_stride0 = C.stride(0);
  auto C_stride1 = C.stride(1);
  auto C_stride2 = C_dim >= 3 ? C.stride(2) : 0;
  auto C_stride3 = C_dim == 4 ? C.stride(3) : 0;
  auto BC_group = B_dim == 4 ? B.size(1) : 1;

  auto threshold = 80.0f;
  auto threshold_vec = at::vec::Vectorized<float>(threshold);
#pragma omp parallel for collapse(2)
  for (auto bi = 0; bi < batch; bi++) {
    for (auto li = 0; li < len; li++) {
      auto di = 0;
      for (; di < dim - (dim % vec_size); di += vec_size) {
        auto dt_vec = at::vec::Vectorized<float>::loadu(
            delta_ptr + bi * delta_strideB + di * delta_strideD +
            li * delta_strideL);
        if (has_dt_bias) {
          auto dt_bias_vec =
              at::vec::Vectorized<float>::loadu(delta_bias_ptr + di);
          dt_vec += dt_bias_vec;
        }
        if (delta_softplus) {
          dt_vec = at::vec::Vectorized<float>::blendv(
              dt_vec.exp().log1p(), dt_vec, dt_vec > threshold_vec);
        }
        dt_vec.store(
            delta_ptr + bi * delta_strideB + di * delta_strideD +
            li * delta_strideL);
      }
      for (; di < dim; di++) {
        auto dt_idx =
            bi * delta_strideB + di * delta_strideD + li * delta_strideL;
        float delta_val = delta_ptr[dt_idx];
        if (has_dt_bias) {
          delta_val += delta_bias_ptr[di];
        }
        if (delta_softplus) {
          delta_val = delta_val > threshold ? delta_val
                                            : std::log1pf(std::exp(delta_val));
        }
        delta_ptr[dt_idx] = delta_val;
      }
    }
  }
#pragma omp parallel for collapse(2)
  for (auto bi = 0; bi < batch; bi++) {
    for (auto di = 0; di < dim; di++) {
      for (auto li = 0; li < len; li++) {
        auto delta_val = delta_ptr
            [bi * delta_strideB + di * delta_strideD + li * delta_strideL];
        auto u_val = u_ptr[bi * u_strideB + di * u_strideD + li * u_strideL];
        auto out_idx = bi * out_strideB + di * out_strideD + li * out_strideL;
        auto dt_u_mul = delta_val * u_val;
        auto out_val = out_ptr[out_idx];
        for (auto dsi = 0; dsi < dstate; dsi++) {
          auto x_idx = bi * x_strideB + di * x_strideD + dsi * x_strideS;
          float deltaA_A_mul =
              delta_val * A_ptr[A_strideD * di + dsi * A_strideS];
          deltaA_A_mul = deltaA_A_mul > threshold ? threshold : deltaA_A_mul;
          float deltaA = std::exp(deltaA_A_mul);
          float x_val = x_ptr[x_idx];
          x_val *= deltaA;
          if (B_dim == 2) { // dim x dstate
            auto B_val = B_ptr[di * B_stride0 + dsi * B_stride1];
            x_val += B_val * dt_u_mul;
          } else if (B_dim == 3) { // batch x dstate x len
            auto B_val =
                B_ptr[bi * B_stride0 + dsi * B_stride1 + li * B_stride2];
            x_val += B_val * dt_u_mul;
          } else { // batch x BC_group x dstate x len
            auto B_val = B_ptr
                [bi * B_stride0 + int(di / BC_group) * B_stride1 +
                 dsi * B_stride2 + li * B_stride3];
            x_val += B_val * dt_u_mul;
          }
          if (C_dim == 2) {
            auto C_val = C_ptr[di * C_stride0 + dsi * C_stride1];
            out_val += x_val * C_val;
          } else if (C_dim == 3) {
            auto C_val =
                C_ptr[bi * C_stride0 + dsi * C_stride1 + li * C_stride2];
            out_val += x_val * C_val;
          } else {
            auto C_val = C_ptr
                [bi * C_stride0 + int(di / BC_group) * C_stride1 +
                 dsi * C_stride2 + li * C_stride3];
            out_val += x_val * C_val;
          }
          x_ptr[x_idx] = x_val;
        }
        if (has_D) {
          out_val += u_val * D_ptr[di];
        }
        if (has_z) {
          auto z_val = z_ptr[bi * z_strideB + di * z_strideD + li * z_strideL];
          out_val *= z_val / (1 + expf(-z_val));
        }
        out_ptr[out_idx] = out_val;
      }
    }
  }
  return std::make_tuple(
      std::move(out.transpose_(1, 2)), std::move(x.transpose_(1, 2)));
}

template <typename T>
at::Tensor selective_state_update_kernel_impl_inner(
    const at::Tensor& state,
    const at::Tensor& x,
    const at::Tensor& dt,
    const at::Tensor& A,
    const at::Tensor& B,
    const at::Tensor& C,
    const c10::optional<at::Tensor>& D,
    const c10::optional<at::Tensor>& z,
    const c10::optional<at::Tensor>& dt_bias,
    bool dt_softplus) {
  using Vec = at::vec::Vectorized<T>;
  using fVec = at::vec::Vectorized<float>;
  auto vec_size = Vec::size();
  auto fvec_size = fVec::size();
  auto has_heads = state.dim() > 3;
  auto batch = state.size(0);
  auto nheads = has_heads ? state.size(1) : 1;
  auto dim = state.size(-2);
  auto dstate = state.size(-1);
  auto out = has_heads ? at::empty({batch, nheads, dim}, state.options())
                       : at::empty({batch, dim}, state.options());
  auto state2 = state.transpose(-2, -1).contiguous();
  auto A2 = A.size(-1) == dim ? A : A.transpose(-2, -1).contiguous();
  auto B_dim = B.dim();
  auto C_dim = C.dim();
  auto B2 = B;
  auto C2 = C;
  auto B_strideS = 1;
  auto C_strideS = 1;
  auto B_strideD = 1;
  auto C_strideD = 1;
  auto B_sizeD = 1;
  auto C_sizeD = 1;
  if (B_dim == 3) {
    B2 = B.transpose(-2, -1).contiguous();
    B_strideS = B2.stride(-2);
    B_strideD = B2.stride(-1);
    B_sizeD = B2.size(-1);
  }
  if (C_dim == 3) {
    C2 = C.transpose(-2, -1).contiguous();
    C_strideS = C2.stride(-2);
    C_strideD = C2.stride(-1);
    C_sizeD = C2.size(-1);
  }
  auto B_strideB = B2.stride(0);
  auto C_strideB = C2.stride(0);
  auto dt_ptr = dt.data_ptr<T>();
  auto A_ptr = A2.data_ptr<float>();
  auto B_ptr = B2.data_ptr<float>();
  auto C_ptr = C2.data_ptr<float>();
  auto state_ptr = state2.data_ptr<T>();
  auto x_ptr = x.data_ptr<T>();
  auto out_ptr = out.data_ptr<T>();
  bool has_dt_bias = dt_bias.has_value();
  bool has_z = z.has_value();
  bool has_D = D.has_value();
  auto dt_bias_ptr = has_dt_bias ? dt_bias.value().data_ptr<float>() : nullptr;
  auto D_ptr = has_D ? D.value().data_ptr<float>() : nullptr;
  auto z_ptr = has_z ? z.value().data_ptr<T>() : nullptr;
  auto dt_bias_stride0 =
      has_heads && has_dt_bias ? dt_bias.value().stride(0) : 0;
  auto z_stride0 = has_z ? z.value().stride(0) : 0;
  auto z_stride1 = has_heads && has_z ? z.value().stride(1) : 0;
  auto D_stride0 =
      !has_heads && has_D && D.value().dim() == 1 ? 0 : D.value().stride(0);

  auto dt_strideB = dt.stride(0);
  auto dt_strideH = has_heads ? dt.stride(1) : 0;
  auto dt_strideD = dt.stride(-1);
  auto state_strideB = state2.stride(0);
  auto state_strideH = has_heads ? state2.stride(1) : 0;
  auto state_strideD = state2.stride(-1);
  auto state_strideS = state2.stride(-2);
  auto x_strideB = x.stride(0);
  auto x_strideH = has_heads ? x.stride(1) : 0;
  auto A_strideH = has_heads ? A2.stride(0) : 0;
  auto A_strideD = A2.stride(-1);
  auto A_strideS = A2.stride(-2);
  auto out_strideB = out.stride(0);
  auto out_strideH = has_heads ? out.stride(1) : 0;
  auto out_strideD = out.stride(-1);
  auto one_vec = Vec(1);
  auto one_vec_fp32 = fVec(1);
  auto threshold = 80.0f;
  auto threshold_vec = fVec(threshold);
#pragma omp parallel for collapse(2)
  for (auto bi = 0; bi < batch; bi++) {
    for (auto hi = 0; hi < nheads; hi++) {
      auto di = 0;
      for (; di < dim - (dim % vec_size); di += vec_size) {
        Vec dt_vec = Vec::loadu(
            dt_ptr + bi * dt_strideB + hi * dt_strideH + di * dt_strideD);
        fVec dt_fvec0, dt_fvec1;
        std::tie(dt_fvec0, dt_fvec1) = at::vec::convert_to_float<T>(dt_vec);
        if (has_dt_bias) {
          fVec dt_bias_fvec0 =
              fVec::loadu(dt_bias_ptr + hi * dt_bias_stride0 + di);
          fVec dt_bias_fvec1 =
              fVec::loadu(dt_bias_ptr + hi * dt_bias_stride0 + di + fvec_size);
          dt_fvec0 += dt_bias_fvec0;
          dt_fvec1 += dt_bias_fvec1;
        }
        if (dt_softplus) {
          dt_fvec0 = fVec::blendv(
              dt_fvec0.exp().log1p(), dt_fvec0, dt_fvec0 > threshold_vec);
          dt_fvec1 = fVec::blendv(
              dt_fvec1.exp().log1p(), dt_fvec1, dt_fvec1 > threshold_vec);
        }
        fVec out_fvec0 = fVec(0);
        fVec out_fvec1 = fVec(0);
        auto x_vec = Vec::loadu(x_ptr + bi * x_strideB + hi * x_strideH + di);
        fVec x_fvec0, x_fvec1;
        std::tie(x_fvec0, x_fvec1) = at::vec::convert_to_float<T>(x_vec);
        fVec dt_x_mul_fvec0 = dt_fvec0 * x_fvec0;
        fVec dt_x_mul_fvec1 = dt_fvec1 * x_fvec1;
        for (auto dsi = 0; dsi < dstate; dsi++) {
          auto state_vec = Vec::loadu(
              state_ptr + bi * state_strideB + hi * state_strideH +
              di * state_strideD + dsi * state_strideS);
          fVec A_fvec0 = fVec::loadu(
              A_ptr + hi * A_strideH + di * A_strideD + dsi * A_strideS);
          fVec A_fvec1 = fVec::loadu(
              A_ptr + hi * A_strideH + di * A_strideD + dsi * A_strideS +
              fvec_size);
          fVec state_fvec0, state_fvec1;
          std::tie(state_fvec0, state_fvec1) =
              at::vec::convert_to_float<T>(state_vec);
          auto dt_A_mul_fvec0 = dt_fvec0 * A_fvec0;
          auto dt_A_mul_fvec1 = dt_fvec1 * A_fvec1;
          auto dA_state_fvec0 = dt_A_mul_fvec0.exp() * state_fvec0;
          auto dA_state_fvec1 = dt_A_mul_fvec1.exp() * state_fvec1;
          fVec B_fvec0, B_fvec1;
          if (B_dim == 2) {
            B_fvec0 = fVec(B_ptr[bi * B_strideB + dsi * B_strideS]);
            B_fvec1 = fVec(B_ptr[bi * B_strideB + dsi * B_strideS]);
          } else {
            B_fvec0 = fVec(B_ptr
                               [bi * B_strideB + (di % B_sizeD) * B_strideD +
                                dsi * B_strideS]);
            B_fvec1 = fVec(
                B_ptr
                    [bi * B_strideB + ((di + fvec_size) % B_sizeD) * B_strideD +
                     dsi * B_strideS]);
          }
          state_fvec0 = dA_state_fvec0 + B_fvec0 * dt_x_mul_fvec0;
          state_fvec1 = dA_state_fvec1 + B_fvec1 * dt_x_mul_fvec1;
          fVec C_fvec0, C_fvec1;
          if (C_dim == 2) {
            C_fvec0 = fVec(C_ptr[bi * C_strideB + dsi * C_strideS]);
            C_fvec1 = fVec(C_ptr[bi * C_strideB + dsi * C_strideS]);
          } else {
            C_fvec0 = fVec(C_ptr
                               [bi * C_strideB + (di % C_sizeD) * C_strideD +
                                dsi * C_strideS]);
            C_fvec1 = fVec(
                C_ptr
                    [bi * C_strideB + ((di + vec_size) % C_sizeD) * C_strideD +
                     dsi * C_strideS]);
          }
          out_fvec0 += state_fvec0 * C_fvec0;
          out_fvec1 += state_fvec1 * C_fvec1;
          state_vec = at::vec::convert_from_float<T>(state_fvec0, state_fvec1);
          state_vec.store(
              state_ptr + bi * state_strideB + hi * state_strideH +
              di * state_strideD + dsi * state_strideS);
        }
        if (has_D) {
          fVec D_fvec0 = fVec::loadu(D_ptr + hi * D_stride0 + di);
          fVec D_fvec1 = fVec::loadu(D_ptr + hi * D_stride0 + di + fvec_size);
          out_fvec0 += x_fvec0 * D_fvec0;
          out_fvec1 += x_fvec1 * D_fvec1;
        }
        if (has_z) {
          auto z_vec = Vec::loadu(z_ptr + bi * z_stride0 + hi * z_stride1 + di);
          fVec z_fvec0, z_fvec1;
          std::tie(z_fvec0, z_fvec1) = at::vec::convert_to_float<T>(z_vec);
          out_fvec0 *= z_fvec0 / (one_vec_fp32 + z_fvec0.neg().exp());
          out_fvec1 *= z_fvec1 / (one_vec_fp32 + z_fvec1.neg().exp());
        }
        Vec out_vec = at::vec::convert_from_float<T>(out_fvec0, out_fvec1);
        out_vec.store(
            out_ptr + bi * out_strideB + hi * out_strideH + di * out_strideD);
      }
      for (; di < dim; di++) {
        float dt_val =
            dt_ptr[bi * dt_strideB + hi * dt_strideH + di * dt_strideD];
        if (has_dt_bias) {
          dt_val += dt_bias_ptr[hi * dt_bias_stride0 + di];
        }
        if (dt_softplus) {
          dt_val = dt_val > threshold ? dt_val : std::log1p(std::exp(dt_val));
        }
        auto out_idx = bi * out_strideB + hi * out_strideH + di * out_strideD;
        float x_val = x_ptr[bi * x_strideB + hi * x_strideH + di];
        float out_val = 0;
        float dt_x_mul = dt_val * x_val;
        for (auto dsi = 0; dsi < dstate; dsi++) {
          auto state_idx = bi * state_strideB + hi * state_strideH +
              di * state_strideD + dsi * state_strideS;
          float A_val =
              A_ptr[hi * A_strideH + di * A_strideD + dsi * A_strideS];
          float B_val = B_ptr
              [bi * B_strideB + (di % B_sizeD) * B_strideD + dsi * B_strideS];
          float C_val = C_ptr
              [bi * C_strideB + (di % C_sizeD) * C_strideD + dsi * C_strideS];
          float dA_A_mul = dt_val * A_val;
          float dA = dA_A_mul > threshold ? dA_A_mul : std::exp(dA_A_mul);
          state_ptr[state_idx] = dA * state_ptr[state_idx] + B_val * dt_x_mul;
          out_val += state_ptr[state_idx] * C_val;
        }
        if (has_D) {
          out_val += x_val * D_ptr[hi * D_stride0 + di];
        }
        if (has_z) {
          float z_val = z_ptr[bi * z_stride0 + hi * z_stride1 + di];
          out_val *= z_val / (1 + expf(-z_val));
        }
        out_ptr[out_idx] = out_val;
      }
    }
  }
  state.copy_(state2.transpose(-2, -1));
  return out;
}

template <>
at::Tensor selective_state_update_kernel_impl_inner<float>(
    const at::Tensor& state,
    const at::Tensor& x,
    const at::Tensor& dt,
    const at::Tensor& A,
    const at::Tensor& B,
    const at::Tensor& C,
    const c10::optional<at::Tensor>& D,
    const c10::optional<at::Tensor>& z,
    const c10::optional<at::Tensor>& dt_bias,
    bool dt_softplus) {
  auto vec_size = at::vec::Vectorized<float>::size();
  auto has_heads = state.dim() > 3;
  auto batch = state.size(0);
  auto nheads = has_heads ? state.size(1) : 1;
  auto dim = state.size(-2);
  auto dstate = state.size(-1);
  auto out = has_heads ? at::empty({batch, nheads, dim}, state.options())
                       : at::empty({batch, dim}, state.options());
  auto state2 = state.transpose(-2, -1).contiguous();
  auto A2 = A.size(-1) == dim ? A : A.transpose(-2, -1).contiguous();
  auto B_dim = B.dim();
  auto C_dim = C.dim();
  auto B2 = B;
  auto C2 = C;
  auto B_strideS = 1;
  auto C_strideS = 1;
  auto B_strideD = 1;
  auto C_strideD = 1;
  auto B_sizeD = 1;
  auto C_sizeD = 1;
  if (B_dim == 3) {
    B2 = B.transpose(-2, -1).contiguous();
    B_strideS = B2.stride(-2);
    B_strideD = B2.stride(-1);
    B_sizeD = B2.size(-1);
  }
  if (C_dim == 3) {
    C2 = C.transpose(-2, -1).contiguous();
    C_strideS = C2.stride(-2);
    C_strideD = C2.stride(-1);
    C_sizeD = C2.size(-1);
  }
  auto B_strideB = B2.stride(0);
  auto C_strideB = C2.stride(0);
  auto dt_ptr = dt.data_ptr<float>();
  auto A_ptr = A2.data_ptr<float>();
  auto B_ptr = B2.data_ptr<float>();
  auto C_ptr = C2.data_ptr<float>();
  auto state_ptr = state2.data_ptr<float>();
  auto x_ptr = x.data_ptr<float>();
  auto out_ptr = out.data_ptr<float>();
  bool has_dt_bias = dt_bias.has_value();
  bool has_z = z.has_value();
  bool has_D = D.has_value();
  auto dt_bias_ptr = has_dt_bias ? dt_bias.value().data_ptr<float>() : nullptr;
  auto D_ptr = has_D ? D.value().data_ptr<float>() : nullptr;
  auto z_ptr = has_z ? z.value().data_ptr<float>() : nullptr;
  auto dt_bias_stride0 =
      has_heads && has_dt_bias ? dt_bias.value().stride(0) : 0;
  auto z_stride0 = has_z ? z.value().stride(0) : 0;
  auto z_stride1 = has_heads && has_z ? z.value().stride(1) : 0;
  auto D_stride0 =
      !has_heads && has_D && D.value().dim() == 1 ? 0 : D.value().stride(0);

  auto dt_strideB = dt.stride(0);
  auto dt_strideH = has_heads ? dt.stride(1) : 0;
  auto dt_strideD = dt.stride(-1);
  auto state_strideB = state2.stride(0);
  auto state_strideH = has_heads ? state2.stride(1) : 0;
  auto state_strideD = state2.stride(-1);
  auto state_strideS = state2.stride(-2);
  auto x_strideB = x.stride(0);
  auto x_strideH = has_heads ? x.stride(1) : 0;
  auto A_strideH = has_heads ? A2.stride(0) : 0;
  auto A_strideD = A2.stride(-1);
  auto A_strideS = A2.stride(-2);
  auto out_strideB = out.stride(0);
  auto out_strideH = has_heads ? out.stride(1) : 0;
  auto out_strideD = out.stride(-1);
  auto one_vec = at::vec::Vectorized<float>(1);
  auto threshold = 80.0f;
  auto threshold_vec = at::vec::Vectorized<float>(threshold);
#pragma omp parallel for collapse(2)
  for (auto bi = 0; bi < batch; bi++) {
    for (auto hi = 0; hi < nheads; hi++) {
      auto di = 0;
      for (; di < dim - (dim % vec_size); di += vec_size) {
        auto dt_vec = at::vec::Vectorized<float>::loadu(
            dt_ptr + bi * dt_strideB + hi * dt_strideH + di * dt_strideD);
        if (has_dt_bias) {
          auto dt_bias_vec = at::vec::Vectorized<float>::loadu(
              dt_bias_ptr + hi * dt_bias_stride0 + di);
          dt_vec += dt_bias_vec;
        }
        if (dt_softplus) {
          dt_vec = at::vec::Vectorized<float>::blendv(
              dt_vec.exp().log1p(), dt_vec, dt_vec > threshold_vec);
        }
        at::vec::Vectorized<float> out_vec(0);
        auto x_vec = at::vec::Vectorized<float>::loadu(
            x_ptr + bi * x_strideB + hi * x_strideH + di);
        auto dt_x_mul = dt_vec * x_vec;
        for (auto dsi = 0; dsi < dstate; dsi++) {
          auto state_vec = at::vec::Vectorized<float>::loadu(
              state_ptr + bi * state_strideB + hi * state_strideH +
              di * state_strideD + dsi * state_strideS);
          auto A_vec = at::vec::Vectorized<float>::loadu(
              A_ptr + hi * A_strideH + di * A_strideD + dsi * A_strideS);
          auto dt_A_mul = dt_vec * A_vec;
          auto dA = dt_A_mul.exp();

          if (B_dim == 2) {
            auto B_vec = at::vec::Vectorized<float>(
                B_ptr[bi * B_strideB + dsi * B_strideS]);
            state_vec = dA * state_vec + B_vec * dt_x_mul;
          } else {
            auto B_vec = at::vec::Vectorized<float>::loadu(
                B_ptr + bi * B_strideB + (di % B_sizeD) * B_strideD +
                dsi * B_strideS);
            state_vec = dA * state_vec + B_vec * dt_x_mul;
          }
          if (C_dim == 2) {
            auto C_vec = at::vec::Vectorized<float>(
                C_ptr[bi * C_strideB + dsi * C_strideS]);
            out_vec += state_vec * C_vec;
          } else {
            auto C_vec = at::vec::Vectorized<float>::loadu(
                C_ptr + bi * C_strideB + (di % C_sizeD) * C_strideD +
                dsi * C_strideS);
            out_vec += state_vec * C_vec;
          }
          state_vec.store(
              state_ptr + bi * state_strideB + hi * state_strideH +
              di * state_strideD + dsi * state_strideS);
        }
        if (has_D) {
          auto D_vec =
              at::vec::Vectorized<float>::loadu(D_ptr + hi * D_stride0 + di);
          out_vec += x_vec * D_vec;
        }
        if (has_z) {
          auto z_vec = at::vec::Vectorized<float>::loadu(
              z_ptr + bi * z_stride0 + hi * z_stride1 + di);
          out_vec *= z_vec / (one_vec + z_vec.neg().exp());
        }
        out_vec.store(
            out_ptr + bi * out_strideB + hi * out_strideH + di * out_strideD);
      }
      for (; di < dim; di++) {
        auto dt_val =
            dt_ptr[bi * dt_strideB + hi * dt_strideH + di * dt_strideD];
        if (has_dt_bias) {
          dt_val += dt_bias_ptr[hi * dt_bias_stride0 + di];
        }
        if (dt_softplus) {
          dt_val = dt_val > threshold ? dt_val : std::log1p(std::exp(dt_val));
        }
        auto out_idx = bi * out_strideB + hi * out_strideH + di * out_strideD;
        auto x_val = x_ptr[bi * x_strideB + hi * x_strideH + di];
        float out_val = 0;
        auto dt_x_mul = dt_val * x_val;
        for (auto dsi = 0; dsi < dstate; dsi++) {
          auto state_idx = bi * state_strideB + hi * state_strideH +
              di * state_strideD + dsi * state_strideS;
          auto A_val = A_ptr[hi * A_strideH + di * A_strideD + dsi * A_strideS];
          auto B_val = B_ptr
              [bi * B_strideB + (di % B_sizeD) * B_strideD + dsi * B_strideS];
          auto C_val = C_ptr
              [bi * C_strideB + (di % C_sizeD) * C_strideD + dsi * C_strideS];
          float dA_A_mul = dt_val * A_val;
          float dA = dA_A_mul > threshold ? threshold : std::exp(dA_A_mul);
          state_ptr[state_idx] = dA * state_ptr[state_idx] + B_val * dt_x_mul;
          out_val += state_ptr[state_idx] * C_val;
        }
        if (has_D) {
          out_val += x_val * D_ptr[hi * D_stride0 + di];
        }
        if (has_z) {
          auto z_val = z_ptr[bi * z_stride0 + hi * z_stride1 + di];
          out_val *= z_val / (1 + expf(-z_val));
        }
        out_ptr[out_idx] = out_val;
      }
    }
  }
  state.copy_(state2.transpose(-2, -1));
  return out;
}

std::tuple<at::Tensor, at::Tensor> selective_scan_kernel_impl(
    const at::Tensor& u,
    const at::Tensor& delta,
    const at::Tensor& A,
    const at::Tensor& B,
    const at::Tensor& C,
    const c10::optional<at::Tensor>& D,
    const c10::optional<at::Tensor>& z,
    const c10::optional<at::Tensor>& delta_bias,
    bool delta_softplus,
    bool return_last_state) {
  if (u.scalar_type() == at::ScalarType::Float) {
    return selective_scan_kernel_inner<float>(
        u, delta, A, B, C, D, z, delta_bias, delta_softplus, return_last_state);
  } else if (u.scalar_type() == at::ScalarType::BFloat16) {
    return selective_scan_kernel_inner<at::BFloat16>(
        u, delta, A, B, C, D, z, delta_bias, delta_softplus, return_last_state);
  } else if (u.scalar_type() == at::ScalarType::Half) {
    return selective_scan_kernel_inner<at::Half>(
        u, delta, A, B, C, D, z, delta_bias, delta_softplus, return_last_state);
  } else {
    TORCH_CHECK(
        false, "Only support bfloat16, float16 and float for selective_scan");
  }
}

at::Tensor selective_state_update_kernel_impl(
    const at::Tensor& state,
    const at::Tensor& x,
    const at::Tensor& dt,
    const at::Tensor& A,
    const at::Tensor& B,
    const at::Tensor& C,
    const c10::optional<at::Tensor>& D,
    const c10::optional<at::Tensor>& z,
    const c10::optional<at::Tensor>& dt_bias,
    bool dt_softplus) {
  if (x.scalar_type() == at::ScalarType::Float) {
    return selective_state_update_kernel_impl_inner<float>(
        state, x, dt, A, B, C, D, z, dt_bias, dt_softplus);
  } else if (x.scalar_type() == at::ScalarType::BFloat16) {
    return selective_state_update_kernel_impl_inner<at::BFloat16>(
        state, x, dt, A, B, C, D, z, dt_bias, dt_softplus);
  } else if (x.scalar_type() == at::ScalarType::Half) {
    return selective_state_update_kernel_impl_inner<at::Half>(
        state, x, dt, A, B, C, D, z, dt_bias, dt_softplus);
  } else {
    TORCH_CHECK(
        false,
        "Only support bfloat16, float16 and float for selective_state_update");
  }
}

} // anonymous namespace
IPEX_REGISTER_DISPATCH(selective_scan_kernel_stub, &selective_scan_kernel_impl);
IPEX_REGISTER_DISPATCH(
    selective_state_update_kernel_stub,
    &selective_state_update_kernel_impl);

} // namespace cpu
} // namespace torch_ipex