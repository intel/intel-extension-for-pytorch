
#include <aten/InstanceNorm.h>

#include <torch/csrc/autograd/function.h>
#include "vec/vec.h"

namespace torch_ipex {
namespace cpu {

namespace {

#if defined(CPU_CAPABILITY_AVX512)
static inline __m512 _mm512_add_reduce_ps(__m512 v) {
  auto perm0 = _mm512_permute_ps(v, _MM_SHUFFLE(2, 3, 0, 1));
  auto m1 = v + perm0;
  auto perm1 = _mm512_permute_ps(m1, _MM_SHUFFLE(1, 0, 3, 2));
  auto m2 = m1 + perm1;
  auto perm2 = _mm512_shuffle_f32x4(m2, m2, _MM_SHUFFLE(2, 3, 0, 1));
  auto m3 = m2 + perm2;
  auto perm3 = _mm512_shuffle_f32x4(m3, m3, _MM_SHUFFLE(1, 0, 3, 2));
  auto m4 = m3 + perm3;
  return m4;
}

inline static __m512 _mm512_mean_reduce_ps(__m512 v, int64_t N) {
  auto rN = _mm512_set1_ps(1. / N);
  auto vsum = _mm512_add_reduce_ps(v);
  return vsum * rN;
}
#endif

#if defined(CPU_CAPABILITY_AVX512)
template <typename T>
inline static __m512 _mm512_loadu_data_ps(T* ptr) {
  return _mm512_loadu_ps(ptr);
}

template <typename T = at::BFloat16>
inline static __m512 _mm512_loadu_data_ps(at::BFloat16* ptr) {
  return cvt_bf16_to_fp32(_mm256_loadu_si256((__m256i*)(ptr)));
}

template <typename T>
inline static __m512 _mm512_mask_loadu_data_ps(__mmask16 k, T* ptr) {
  return _mm512_maskz_loadu_ps(k, ptr);
}

template <typename T = at::BFloat16>
inline static __m512 _mm512_mask_loadu_data_ps(__mmask16 k, at::BFloat16* ptr) {
  return cvt_bf16_to_fp32(_mm256_maskz_loadu_epi16(k, ptr));
}

template <typename T>
inline static void _mm512_storeu_data_ps(T* ptr, __m512 vec) {
  _mm512_storeu_ps(ptr, vec);
}

template <typename T = at::BFloat16>
inline static void _mm512_storeu_data_ps(at::BFloat16* ptr, __m512 vec) {
  _mm256_storeu_si256((__m256i*)(ptr), cvt_fp32_to_bf16(vec));
}

template <typename T>
inline static void _mm512_mask_storeu_data_ps(T* ptr, __mmask16 k, __m512 vec) {
  _mm512_mask_storeu_ps(ptr, k, vec);
}

template <typename T = at::BFloat16>
inline static void _mm512_mask_storeu_data_ps(
    at::BFloat16* ptr,
    __mmask16 k,
    __m512 vec) {
  _mm256_mask_storeu_epi16(ptr, k, cvt_fp32_to_bf16(vec));
}
#endif

#if defined(CPU_CAPABILITY_AVX512)
template <typename T>
void channels_first_forward(
    T* in,
    T* out,
    float& weight,
    float& bias,
    float& m,
    float& v,
    int64_t channel,
    int64_t rl) {
  int64_t d;
  auto vsum = _mm512_setzero_ps();
  auto vsum2 = _mm512_setzero_ps();
  auto* pin = in;

  for (d = 0; d < rl / 16 * 16; d += 16) {
    auto f = _mm512_loadu_data_ps<T>(&pin[d]);
    vsum += f;
    vsum2 += f * f;
  }

  if (d < rl) {
    auto rem = rl - d;
    __mmask16 k = (1 << rem) - 1;
    auto f = _mm512_mask_loadu_data_ps<T>(k, &pin[d]);
    vsum += f;
    vsum2 += f * f;
  }

  auto veps = _mm512_set1_ps(1e-5);
  auto vmean = _mm512_mean_reduce_ps(vsum, rl);
  auto vmean2 = _mm512_mean_reduce_ps(vsum2, rl);
  auto vvar2 = vmean2 - vmean * vmean;

  m = vmean[0];
  v = vvar2[0];
  vvar2 += veps;

  auto r_vvar = 1. / _mm512_sqrt_ps(vvar2);
  auto* pout = out;
  auto w = _mm512_set1_ps(weight);
  auto b = _mm512_set1_ps(bias);

  for (d = 0; d < rl / 16 * 16; d += 16) {
    auto f = _mm512_loadu_data_ps<T>(&pin[d]);
    auto o = (f - vmean) * w * r_vvar + b;
    _mm512_storeu_data_ps<T>(&pout[d], o);
  }
  if (d < rl) {
    auto rem = rl - d;
    __mmask16 k = (1 << rem) - 1;
    auto f = _mm512_mask_loadu_data_ps<T>(k, &pin[d]);
    auto o = (f - vmean) * w * r_vvar + b;
    _mm512_mask_storeu_data_ps<T>(&pout[d], k, o);
  }
}
#endif

#if defined(CPU_CAPABILITY_AVX512)
template <typename T>
std::vector<at::Tensor> instancenorm_forward_channels_first(
    const at::Tensor& input,
    const at::Tensor& weight,
    const at::Tensor& bias) {
  auto in_sz = input.sizes();
  auto channel = in_sz[1];
  int64_t reduce_l;
  if (in_sz.size() == 4)
    reduce_l = in_sz[2] * in_sz[3];
  else
    reduce_l = in_sz[2] * in_sz[3] * in_sz[4];
  auto batch = in_sz[0] * in_sz[1];

  auto mean_t = at::empty(
      batch,
      at::TensorOptions().dtype<float>().memory_format(
          c10::MemoryFormat::Contiguous));
  auto var_t = at::empty(
      batch,
      at::TensorOptions().dtype<float>().memory_format(
          c10::MemoryFormat::Contiguous));
  auto output = at::empty(
      in_sz, input.options().memory_format(input.suggest_memory_format()));

  auto* in_ptr = input.data_ptr();
  auto data_type = input.scalar_type();
  auto* out_ptr = output.data_ptr();
  auto* w_ptr = weight.data_ptr();
  auto* b_ptr = bias.data_ptr();
  auto* m_ptr = mean_t.data_ptr();
  auto* v_ptr = var_t.data_ptr();

#pragma omp parallel for
  for (auto i = 0; i < batch; ++i) {
    auto* bin = reinterpret_cast<T(*)[reduce_l]>(in_ptr);
    auto* w = reinterpret_cast<float(*)>(w_ptr);
    auto* b = reinterpret_cast<float(*)>(b_ptr);
    auto* m = reinterpret_cast<float(*)>(m_ptr);
    auto* v = reinterpret_cast<float(*)>(v_ptr);
    auto* bout = reinterpret_cast<T(*)[reduce_l]>(out_ptr);
    channels_first_forward<T>(
        bin[i],
        bout[i],
        w[i % channel],
        b[i % channel],
        m[i],
        v[i],
        channel,
        reduce_l);
  }
  return {output, mean_t, var_t};
}
#endif

#if defined(CPU_CAPABILITY_AVX512)
template <typename T>
void channels_first_backward(
    T* dout,
    T* in,
    float& weight,
    float& m,
    float& v,
    T* dx,
    float& dw,
    float& db,
    int64_t channel,
    int64_t rl) {
  int64_t d;
  auto dgamma_sum = _mm512_setzero_ps();
  auto dbias_sum = _mm512_setzero_ps();

  auto* pin = in;
  auto* pdout = dout;
  auto* pdx = dx;

  auto vweight = _mm512_set1_ps(weight);
  auto vmean = _mm512_set1_ps(m);
  auto vvar = _mm512_set1_ps(v);
  auto veps = _mm512_set1_ps(1e-5);
  auto r_var = 1. / _mm512_sqrt_ps(vvar + veps);

  for (d = 0; d < rl / 16 * 16; d += 16) {
    auto fin = _mm512_loadu_data_ps<T>(&pin[d]);
    auto fdout = _mm512_loadu_data_ps<T>(&pdout[d]);
    dbias_sum += fdout;
    dgamma_sum += fdout * (fin - vmean);
  }
  if (d < rl) {
    auto rem = rl - d;
    __mmask16 k = (1 << rem) - 1;
    auto fin = _mm512_mask_loadu_data_ps<T>(k, &pin[d]);
    auto fdout = _mm512_mask_loadu_data_ps<T>(k, &pdout[d]);
    dbias_sum += fdout;
    dgamma_sum += fdout * (fin - vmean);
  }
  dgamma_sum *= r_var;

  auto bias_sum = _mm512_mean_reduce_ps(dbias_sum, 1.);
  auto gamma_sum = _mm512_mean_reduce_ps(dgamma_sum, 1.);

  dw = gamma_sum[0];
  db = bias_sum[0];

  auto cdb = _mm512_set1_ps(db / rl);
  auto cdw = _mm512_set1_ps(dw / rl) * r_var;

  for (d = 0; d < rl / 16 * 16; d += 16) {
    auto f = _mm512_loadu_data_ps<T>(&pin[d]);
    auto fo = _mm512_loadu_data_ps<T>(&pdout[d]);
    fo -= cdb + (f - vmean) * cdw;
    fo *= vweight * r_var;
    _mm512_storeu_data_ps<T>(&pdx[d], fo);
  }
  if (d < rl) {
    auto rem = rl - d;
    __mmask16 k = (1 << rem) - 1;
    auto f = _mm512_mask_loadu_data_ps<T>(k, &pin[d]);
    auto fo = _mm512_mask_loadu_data_ps<T>(k, &pdout[d]);
    fo -= cdb + (f - vmean) * cdw;
    fo *= vweight * r_var;
    _mm512_mask_storeu_data_ps<T>(&pdx[d], k, fo);
  }
}

template <typename T>
std::vector<at::Tensor> instancenorm_backward_channels_first(
    const at::Tensor& grad_output,
    const at::Tensor& input,
    const at::Tensor& weight,
    const at::Tensor& mean,
    const at::Tensor& var) {
  auto in_sz = input.sizes();
  auto channel = in_sz[1];
  int64_t reduce_l;
  if (in_sz.size() == 4)
    reduce_l = in_sz[2] * in_sz[3];
  else
    reduce_l = in_sz[2] * in_sz[3] * in_sz[4];
  auto batch = in_sz[0] * in_sz[1];

  auto grad_weight = at::empty(
      batch,
      at::TensorOptions().dtype<float>().memory_format(
          c10::MemoryFormat::Contiguous));
  auto grad_bias = at::empty(
      batch,
      at::TensorOptions().dtype<float>().memory_format(
          c10::MemoryFormat::Contiguous));
  auto grad_input = at::empty(
      in_sz,
      grad_output.options().memory_format(grad_output.suggest_memory_format()));

  auto* dout_ptr = grad_output.data_ptr();
  auto* in_ptr = input.data_ptr();
  auto* w_ptr = weight.data_ptr();
  auto* m_ptr = mean.data_ptr();
  auto* v_ptr = var.data_ptr();

  auto* dx_ptr = grad_input.data_ptr();
  auto* dw_ptr = grad_weight.data_ptr();
  auto* db_ptr = grad_bias.data_ptr();

#pragma omp parallel for
  for (auto i = 0; i < batch; ++i) {
    auto* dout = reinterpret_cast<T(*)[reduce_l]>(dout_ptr);
    auto* bin = reinterpret_cast<T(*)[reduce_l]>(in_ptr);
    auto* w = reinterpret_cast<float(*)>(w_ptr);
    auto* m = reinterpret_cast<float(*)>(m_ptr);
    auto* v = reinterpret_cast<float(*)>(v_ptr);
    auto* dx = reinterpret_cast<T(*)[reduce_l]>(dx_ptr);
    auto* dw = reinterpret_cast<float(*)>(dw_ptr);
    auto* db = reinterpret_cast<float(*)>(db_ptr);
    channels_first_backward<T>(
        dout[i],
        bin[i],
        w[i % channel],
        m[i],
        v[i],
        dx[i],
        dw[i],
        db[i],
        channel,
        reduce_l);
  }

  grad_weight = grad_weight.reshape({in_sz[0], in_sz[1]});
  grad_bias = grad_bias.reshape({in_sz[0], in_sz[1]});
  grad_weight = at::sum(grad_weight, 0);
  grad_bias = at::sum(grad_bias, 0);
  return {grad_input, grad_weight, grad_bias};
}
#endif

#if defined(CPU_CAPABILITY_AVX512)
template <typename T>
void channels_last_mean_var(T* in, float* m, float* v, int64_t c, int64_t bl) {
  auto vnum = c / 16;
  auto vrem = c % 16;
  auto vnum_total = vnum;
  if (vrem > 0)
    vnum_total += 1;

  __m512 sm[vnum_total];
  __m512 smm[vnum_total];
  for (int i = 0; i < vnum_total; ++i) {
    sm[i] = _mm512_setzero_ps();
    smm[i] = _mm512_setzero_ps();
  }

  auto* pin = in;
  auto rbl = _mm512_set1_ps(1.0 / bl);
  for (auto i = 0; i < bl; ++i) {
    int64_t j;
    for (j = 0; j < vnum; ++j) {
      auto f = _mm512_loadu_data_ps<T>(&pin[(i * c + j * 16)]);
      sm[j] += f;
      smm[j] += f * f;
    }
    if (vrem > 0) {
      __mmask16 k = (1 << vrem) - 1;
      auto f = _mm512_mask_loadu_data_ps<T>(k, &pin[(i * c + j * 16)]);
      sm[j] += f;
      smm[j] += f * f;
    }
  }

  int64_t i;
  for (i = 0; i < vnum; ++i) {
    _mm512_storeu_data_ps<float>(&m[i * 16], sm[i] * rbl);
    _mm512_storeu_data_ps<float>(&v[i * 16], smm[i] * rbl);
  }
  if (vrem > 0) {
    __mmask16 k = (1 << vrem) - 1;
    _mm512_mask_storeu_data_ps<float>(&m[i * 16], k, sm[i] * rbl);
    _mm512_mask_storeu_data_ps<float>(&v[i * 16], k, smm[i] * rbl);
  }
}
#endif

#if defined(CPU_CAPABILITY_AVX512)
template <typename T>
void channels_last_norm(
    T* in,
    T* out,
    float* w,
    float* b,
    float* m,
    float* v,
    int64_t c,
    int64_t bl) {
  auto vnum = c / 16;
  auto vrem = c % 16;
  auto vnum_total = vnum;
  if (vrem > 0)
    vnum_total += 1;

  __m512 _m[vnum_total];
  __m512 _v[vnum_total];
  __m512 _w[vnum_total];
  __m512 _b[vnum_total];
  __m512 vscale[vnum_total];
  __m512 vshift[vnum_total];

  for (int i = 0; i < vnum_total; ++i) {
    _m[i] = _mm512_setzero_ps();
    _v[i] = _mm512_setzero_ps();
    _w[i] = _mm512_setzero_ps();
    _b[i] = _mm512_setzero_ps();
    vscale[i] = _mm512_setzero_ps();
    vshift[i] = _mm512_setzero_ps();
  }

  auto veps = _mm512_set1_ps(1e-5);
  int64_t i;
  for (i = 0; i < vnum; ++i) {
    _m[i] = _mm512_loadu_data_ps<float>(&m[i * 16]);
    _v[i] = _mm512_loadu_data_ps<float>(&v[i * 16]) + veps;
    _w[i] = _mm512_loadu_data_ps<float>(&w[i * 16]);
    _b[i] = _mm512_loadu_data_ps<float>(&b[i * 16]);
    vscale[i] = _w[i] * 1.0 / _mm512_sqrt_ps(_v[i]);
    vshift[i] = _mm512_fmsub_ps(_m[i], vscale[i], _b[i]);
  }
  if (vrem > 0) {
    __mmask16 k = (1 << vrem) - 1;
    _m[i] = _mm512_mask_loadu_data_ps<float>(k, &m[i * 16]);
    _v[i] = _mm512_mask_loadu_data_ps<float>(k, &v[i * 16]) + veps;
    _w[i] = _mm512_mask_loadu_data_ps<float>(k, &w[i * 16]);
    _b[i] = _mm512_mask_loadu_data_ps<float>(k, &b[i * 16]);
    vscale[i] = _w[i] * 1.0 / _mm512_sqrt_ps(_v[i]);
    vshift[i] = _mm512_fmsub_ps(_m[i], vscale[i], _b[i]);
  }

  auto* pin = in;
  auto* pout = out;
  for (auto i = 0; i < bl; ++i) {
    int64_t j;
    for (j = 0; j < vnum; ++j) {
      auto f = _mm512_loadu_data_ps<T>(&pin[(i * c + j * 16)]);
      auto o = _mm512_fmsub_ps(f, vscale[j], vshift[j]);
      _mm512_storeu_data_ps<T>(&pout[(i * c + j * 16)], o);
    }
    if (vrem > 0) {
      __mmask16 k = (1 << vrem) - 1;
      auto f = _mm512_mask_loadu_data_ps<T>(k, &pin[(i * c + j * 16)]);
      auto o = _mm512_fmsub_ps(f, vscale[j], vshift[j]);
      _mm512_mask_storeu_data_ps<T>(&pout[(i * c + j * 16)], k, o);
    }
  }
}

template <typename T>
std::vector<at::Tensor> instancenorm_forward_channels_last(
    const at::Tensor& input,
    const at::Tensor& weight,
    const at::Tensor& bias) {
  auto in_sz = input.sizes();
  auto batch = in_sz[0];
  auto channel = in_sz[1];
  auto block_len = in_sz[2];
  int64_t reduce_l;
  if (in_sz.size() == 4)
    reduce_l = in_sz[2] * in_sz[3];
  else
    reduce_l = in_sz[2] * in_sz[3] * in_sz[4];
  auto block_num = reduce_l / block_len;

  auto mean_t = at::empty(
      {batch, block_num, channel},
      at::TensorOptions().dtype<float>().memory_format(
          c10::MemoryFormat::Contiguous));
  auto var_t = at::empty(
      {batch, block_num, channel},
      at::TensorOptions().dtype<float>().memory_format(
          c10::MemoryFormat::Contiguous));
  auto output = at::empty(
      in_sz, input.options().memory_format(input.suggest_memory_format()));

  auto* in_ptr = input.data_ptr();
  auto* out_ptr = output.data_ptr();
  auto* w_ptr = weight.data_ptr();
  auto* b_ptr = bias.data_ptr();
  auto* m_ptr = mean_t.data_ptr();
  auto* v_ptr = var_t.data_ptr();

#pragma omp parallel for
  for (auto i = 0; i < batch * block_num; ++i) {
    auto* bin = reinterpret_cast<T(*)[block_len * channel]>(in_ptr);
    auto* m = reinterpret_cast<float(*)[channel]>(m_ptr);
    auto* v = reinterpret_cast<float(*)[channel]>(v_ptr);
    channels_last_mean_var<T>(bin[i], m[i], v[i], channel, block_len);
  }

  auto mt = at::mean(mean_t, 1).reshape(batch * channel);
  auto mt2 = at::mean(var_t, 1).reshape(batch * channel);
  auto vt = mt2 - mt * mt;
  auto* mt_ptr = mt.data_ptr();
  auto* vt_ptr = vt.data_ptr();

#pragma omp parallel for
  for (auto i = 0; i < batch * block_num; ++i) {
    auto* bin = reinterpret_cast<T(*)[block_len * channel]>(in_ptr);
    auto* w = reinterpret_cast<float(*)>(w_ptr);
    auto* b = reinterpret_cast<float(*)>(b_ptr);
    auto* m = reinterpret_cast<float(*)[channel]>(mt_ptr);
    auto* v = reinterpret_cast<float(*)[channel]>(vt_ptr);
    auto* bout = reinterpret_cast<T(*)[block_len * channel]>(out_ptr);
    channels_last_norm<T>(
        bin[i],
        bout[i],
        w,
        b,
        m[i / block_num],
        v[i / block_num],
        channel,
        block_len);
  }
  return {output, mt, vt};
}
#endif

#if defined(CPU_CAPABILITY_AVX512)
template <typename T>
void channels_last_dwdb(
    T* dout,
    T* in,
    float* m,
    float* v,
    float* dw,
    float* db,
    int64_t c,
    int64_t bl) {
  auto vnum = c / 16;
  auto vrem = c % 16;
  auto vnum_total = vnum;
  if (vrem > 0)
    vnum_total += 1;

  __m512 dgamma_sum[vnum_total];
  __m512 dbias_sum[vnum_total];
  __m512 vmean[vnum_total];
  __m512 r_var[vnum_total];
  for (int i = 0; i < vnum_total; ++i) {
    dgamma_sum[i] = _mm512_setzero_ps();
    dbias_sum[i] = _mm512_setzero_ps();
    vmean[i] = _mm512_setzero_ps();
    r_var[i] = _mm512_setzero_ps();
  }

  auto* pin = in;
  auto* pdout = dout;
  auto* pdw = dw;
  auto* pdb = db;

  auto veps = _mm512_set1_ps(1e-5);
  int64_t i;
  for (i = 0; i < vnum; ++i) {
    vmean[i] = _mm512_loadu_data_ps<float>(&m[i * 16]);
    r_var[i] =
        1. / _mm512_sqrt_ps(_mm512_loadu_data_ps<float>(&v[i * 16]) + veps);
  }
  if (vrem > 0) {
    __mmask16 k = (1 << vrem) - 1;
    vmean[i] = _mm512_mask_loadu_data_ps<float>(k, &m[i * 16]);
    r_var[i] = 1. /
        _mm512_sqrt_ps(_mm512_mask_loadu_data_ps<float>(k, &v[i * 16]) + veps);
  }

  for (int i = 0; i < bl; ++i) {
    int64_t j;
    for (j = 0; j < vnum; ++j) {
      auto fin = _mm512_loadu_data_ps<T>(&pin[(i * c + j * 16)]);
      auto fo = _mm512_loadu_data_ps<T>(&pdout[(i * c + j * 16)]);
      dbias_sum[j] += fo;
      dgamma_sum[j] += fo * (fin - vmean[j]);
    }
    if (vrem > 0) {
      __mmask16 k = (1 << vrem) - 1;
      auto fin = _mm512_mask_loadu_data_ps<T>(k, &pin[(i * c + j * 16)]);
      auto fo = _mm512_mask_loadu_data_ps<T>(k, &pdout[(i * c + j * 16)]);
      dbias_sum[j] += fo;
      dgamma_sum[j] += fo * (fin - vmean[j]);
    }
  }

  for (i = 0; i < vnum; ++i) {
    dgamma_sum[i] *= r_var[i];
    _mm512_storeu_data_ps<float>(&pdw[i * 16], dgamma_sum[i]);
    _mm512_storeu_data_ps<float>(&pdb[i * 16], dbias_sum[i]);
  }
  if (vrem > 0) {
    __mmask16 k = (1 << vrem) - 1;
    dgamma_sum[i] *= r_var[i];
    _mm512_mask_storeu_data_ps<float>(&pdw[i * 16], k, dgamma_sum[i]);
    _mm512_mask_storeu_data_ps<float>(&pdb[i * 16], k, dbias_sum[i]);
  }
}

template <typename T>
void channels_last_dx(
    T* dout,
    T* in,
    float* w,
    float* m,
    float* v,
    T* dx,
    float* dw,
    float* db,
    int64_t c,
    int64_t bl,
    int64_t rl) {
  auto vnum = c / 16;
  auto vrem = c % 16;
  auto vnum_total = vnum;
  if (vrem > 0)
    vnum_total += 1;

  __m512 _w[vnum_total];
  __m512 _m[vnum_total];
  __m512 _r_v[vnum_total];
  __m512 _cdb[vnum_total];
  __m512 _cdw[vnum_total];
  for (int i = 0; i < vnum_total; ++i) {
    _w[i] = _mm512_setzero_ps();
    _m[i] = _mm512_setzero_ps();
    _r_v[i] = _mm512_setzero_ps();
    _cdb[i] = _mm512_setzero_ps();
    _cdw[i] = _mm512_setzero_ps();
  }

  auto* pin = in;
  auto* pdout = dout;
  auto* pdx = dx;

  auto veps = _mm512_set1_ps(1e-5);
  auto vrl = _mm512_set1_ps(rl);

  int64_t i;
  for (i = 0; i < vnum; ++i) {
    _w[i] = _mm512_loadu_data_ps<float>(&w[i * 16]);
    _m[i] = _mm512_loadu_data_ps<float>(&m[i * 16]);
    _r_v[i] =
        1. / _mm512_sqrt_ps(_mm512_loadu_data_ps<float>(&v[i * 16]) + veps);
    _cdb[i] = _mm512_loadu_data_ps<float>(&db[i * 16]) * 1.0 / vrl;
    _cdw[i] = _mm512_loadu_data_ps<float>(&dw[i * 16]) * 1.0 / vrl * _r_v[i];
  }
  if (vrem > 0) {
    __mmask16 k = (1 << vrem) - 1;
    _w[i] = _mm512_mask_loadu_data_ps<float>(k, &w[i * 16]);
    _m[i] = _mm512_mask_loadu_data_ps<float>(k, &m[i * 16]);
    _r_v[i] = 1. /
        _mm512_sqrt_ps(_mm512_mask_loadu_data_ps<float>(k, &v[i * 16]) + veps);
    _cdb[i] = _mm512_mask_loadu_data_ps<float>(k, &db[i * 16]) * 1.0 / vrl;
    _cdw[i] =
        _mm512_mask_loadu_data_ps<float>(k, &dw[i * 16]) * 1.0 / vrl * _r_v[i];
  }

  for (int i = 0; i < bl; ++i) {
    int64_t j;
    for (j = 0; j < vnum; ++j) {
      auto fin = _mm512_loadu_data_ps<T>(&pin[(i * c + j * 16)]);
      auto fo = _mm512_loadu_data_ps<T>(&pdout[(i * c + j * 16)]);
      fo -= _cdb[j] + (fin - _m[j]) * _cdw[j];
      fo *= _w[j] * _r_v[j];
      _mm512_storeu_data_ps<T>(&pdx[(i * c + j * 16)], fo);
    }
    if (vrem > 0) {
      __mmask16 k = (1 << vrem) - 1;
      auto fin = _mm512_mask_loadu_data_ps<T>(k, &pin[(i * c + j * 16)]);
      auto fo = _mm512_mask_loadu_data_ps<T>(k, &pdout[(i * c + j * 16)]);
      fo -= _cdb[j] + (fin - _m[j]) * _cdw[j];
      fo *= _w[j] * _r_v[j];
      _mm512_mask_storeu_data_ps<T>(&pdx[(i * c + j * 16)], k, fo);
    }
  }
}

template <typename T>
std::vector<at::Tensor> instancenorm_backward_channels_last(
    const at::Tensor& grad_output,
    const at::Tensor& input,
    const at::Tensor& weight,
    const at::Tensor& mean,
    const at::Tensor& var) {
  auto in_sz = input.sizes();
  auto batch = in_sz[0];
  auto channel = in_sz[1];
  auto block_len = in_sz[2];
  int64_t reduce_l;
  if (in_sz.size() == 4)
    reduce_l = in_sz[2] * in_sz[3];
  else
    reduce_l = in_sz[2] * in_sz[3] * in_sz[4];
  auto block_num = reduce_l / block_len;

  auto grad_weight = at::empty(
      {batch, block_num, channel},
      at::TensorOptions().dtype<float>().memory_format(
          c10::MemoryFormat::Contiguous));
  auto grad_bias = at::empty(
      {batch, block_num, channel},
      at::TensorOptions().dtype<float>().memory_format(
          c10::MemoryFormat::Contiguous));
  auto grad_input = at::empty(
      in_sz,
      grad_output.options().memory_format(grad_output.suggest_memory_format()));

  auto* dout_ptr = grad_output.data_ptr();
  auto* in_ptr = input.data_ptr();
  auto* w_ptr = weight.data_ptr();
  auto* m_ptr = mean.data_ptr();
  auto* v_ptr = var.data_ptr();

  auto* dx_ptr = grad_input.data_ptr();
  auto* dw_ptr = grad_weight.data_ptr();
  auto* db_ptr = grad_bias.data_ptr();

#pragma omp parallel for
  for (auto i = 0; i < batch * block_num; ++i) {
    auto* dout = reinterpret_cast<T(*)[block_len * channel]>(dout_ptr);
    auto* bin = reinterpret_cast<T(*)[block_len * channel]>(in_ptr);
    auto* m = reinterpret_cast<float(*)[channel]>(m_ptr);
    auto* v = reinterpret_cast<float(*)[channel]>(v_ptr);
    auto* dw = reinterpret_cast<float(*)[channel]>(dw_ptr);
    auto* db = reinterpret_cast<float(*)[channel]>(db_ptr);
    channels_last_dwdb<T>(
        dout[i],
        bin[i],
        m[i / block_num],
        v[i / block_num],
        dw[i],
        db[i],
        channel,
        block_len);
  }

  auto grad_w = at::sum(grad_weight, 1);
  auto grad_b = at::sum(grad_bias, 1);
  dw_ptr = grad_w.data_ptr();
  db_ptr = grad_b.data_ptr();

#pragma omp parallel for
  for (auto i = 0; i < batch * block_num; ++i) {
    auto* dout = reinterpret_cast<T(*)[block_len * channel]>(dout_ptr);
    auto* bin = reinterpret_cast<T(*)[block_len * channel]>(in_ptr);
    auto* w = reinterpret_cast<float(*)>(w_ptr);
    auto* m = reinterpret_cast<float(*)[channel]>(m_ptr);
    auto* v = reinterpret_cast<float(*)[channel]>(v_ptr);
    auto* dx = reinterpret_cast<T(*)[block_len * channel]>(dx_ptr);
    auto* dw = reinterpret_cast<float(*)[channel]>(dw_ptr);
    auto* db = reinterpret_cast<float(*)[channel]>(db_ptr);
    channels_last_dx<T>(
        dout[i],
        bin[i],
        w,
        m[i / block_num],
        v[i / block_num],
        dx[i],
        dw[i / block_num],
        db[i / block_num],
        channel,
        block_len,
        reduce_l);
  }

  grad_w = grad_w.reshape({in_sz[0], in_sz[1]});
  grad_b = grad_b.reshape({in_sz[0], in_sz[1]});
  grad_w = at::sum(grad_w, 0);
  grad_b = at::sum(grad_b, 0);
  return {grad_input, grad_w, grad_b};
}
#endif

std::vector<at::Tensor> InstanceNormKernelImpl(
    const at::Tensor& input,
    const at::Tensor& weight_t,
    const at::Tensor& bias_t,
    const at::Tensor& running_mean,
    const at::Tensor& running_var,
    double eps,
    bool is_channels_last) {
  int channel = input.sizes()[1];
  auto weight = weight_t.defined() ? weight_t : at::ones(channel);
  auto bias = bias_t.defined() ? bias_t : at::zeros(channel);

#if defined(CPU_CAPABILITY_AVX512)
  auto data_type = input.scalar_type();
  if (is_channels_last) {
    if (data_type == c10::ScalarType::BFloat16) {
      return instancenorm_forward_channels_last<at::BFloat16>(
          input, weight, bias);
    } else {
      return instancenorm_forward_channels_last<float>(input, weight, bias);
    }
  } else {
    if (data_type == c10::ScalarType::BFloat16) {
      return instancenorm_forward_channels_first<at::BFloat16>(
          input, weight, bias);
    } else {
      return instancenorm_forward_channels_first<float>(input, weight, bias);
    }
  }
#else
  return {};
#endif
}

std::vector<at::Tensor> InstanceNormBackwardKernelImpl(
    const at::Tensor& grad_output,
    const at::Tensor& input,
    const at::Tensor& weight_t,
    const at::Tensor& mean,
    const at::Tensor& var,
    bool is_channels_last) {
  int channel = input.sizes()[1];
  auto weight = weight_t.defined() ? weight_t : at::ones(channel);

#if defined(CPU_CAPABILITY_AVX512)
  auto data_type = input.scalar_type();
  if (is_channels_last) {
    if (data_type == c10::ScalarType::BFloat16) {
      return instancenorm_backward_channels_last<at::BFloat16>(
          grad_output, input, weight, mean, var);
    } else {
      return instancenorm_backward_channels_last<float>(
          grad_output, input, weight, mean, var);
    }
  } else {
    if (data_type == c10::ScalarType::BFloat16) {
      return instancenorm_backward_channels_first<at::BFloat16>(
          grad_output, input, weight, mean, var);
    } else {
      return instancenorm_backward_channels_first<float>(
          grad_output, input, weight, mean, var);
    }
  }
#else
  return {};
#endif
}

} // namespace

IPEX_REGISTER_DISPATCH(InstanceNormKernel, &InstanceNormKernelImpl);
IPEX_REGISTER_DISPATCH(
    InstanceNormBackwardKernel,
    &InstanceNormBackwardKernelImpl);

} // namespace cpu
} // namespace torch_ipex