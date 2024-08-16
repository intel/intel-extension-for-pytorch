#include "xetla_int4_dequantize.h"

namespace torch_ipex::xpu::xetla {

using fp16 = gpu::xetla::fp16;
using bf16 = gpu::xetla::bf16;

#define INT4_DEQUANTIZE_ASTR_IMPL_FUNC(scalar_t, q_mode, ...)    \
  template cgf_t XETLA_KERNEL_API                                \
  xetla_dequantize_int4_weight<scalar_t, q_mode, ##__VA_ARGS__>( \
      scalar_t * out,                                            \
      const uint32_t* b,                                         \
      const uint32_t* b_zp,                                      \
      const scalar_t* b_scale,                                   \
      const uint32_t n,                                          \
      const uint32_t k);

#define INT4_DEQUANTIZE_ASTR_IMPL_FUNC_GZ(Q_MODE, DEQUANT_S) \
  INT4_DEQUANTIZE_ASTR_IMPL_FUNC(                            \
      fp16,                                                  \
      Q_MODE,                                                \
      16,                                                    \
      128,                                                   \
      16,                                                    \
      128,                                                   \
      32,                                                    \
      DEQUANT_S,                                             \
      static_cast<int>(gpu::xetla::gpu_arch::XeLpg))

#define INT4_DEQUANTIZE_IMPL_GZ_LIST(Q_MODE)     \
  INT4_DEQUANTIZE_ASTR_IMPL_FUNC_GZ(Q_MODE, 16)  \
  INT4_DEQUANTIZE_ASTR_IMPL_FUNC_GZ(Q_MODE, 32)  \
  INT4_DEQUANTIZE_ASTR_IMPL_FUNC_GZ(Q_MODE, 64)  \
  INT4_DEQUANTIZE_ASTR_IMPL_FUNC_GZ(Q_MODE, 128) \
  INT4_DEQUANTIZE_ASTR_IMPL_FUNC_GZ(Q_MODE, 256) \
  INT4_DEQUANTIZE_ASTR_IMPL_FUNC_GZ(Q_MODE, 512) \
  INT4_DEQUANTIZE_ASTR_IMPL_FUNC_GZ(Q_MODE, 1024)

INT4_DEQUANTIZE_IMPL_GZ_LIST(quant_mode::I4_SYM)
INT4_DEQUANTIZE_IMPL_GZ_LIST(quant_mode::I4_ASYM)

} // namespace torch_ipex::xpu::xetla
