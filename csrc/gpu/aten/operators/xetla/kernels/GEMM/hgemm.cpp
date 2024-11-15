#include "../../hgemm.h"

#ifdef USE_XETLA_XE_HPC
#include "./hgemm_xehpc.hpp"
#endif

namespace torch_ipex::xpu::xetla {

XETLA_KERNEL_API int hgemm_find_policy_id(
    const int m,
    const int n,
    const int k,
    const bool is_b_row_major,
    const gpu::xetla::gpu_arch arch_tag) {
  switch (arch_tag) {
#ifdef USE_XETLA_XE_HPC
    case gpu::xetla::gpu_arch::XeHpc:
      return xehpc::hgemm_find_policy_id(m, n, k, is_b_row_major);
#endif
    default:
      break;
  }
  return -1;
}

XETLA_KERNEL_API int hgemm_qkv_find_policy_id(
    const int m,
    const int n,
    const int k,
    const bool is_b_row_major,
    const gpu::xetla::gpu_arch arch_tag) {
  switch (arch_tag) {
#ifdef USE_XETLA_XE_HPC
    case gpu::xetla::gpu_arch::XeHpc:
      return xehpc::hgemm_qkv_find_policy_id(m, n, k, is_b_row_major);
#endif
    default:
      break;
  }
  return -1;
}

XETLA_KERNEL_API cgfs_t hgemm_addmm(
    const int policy_id,
    sycl::half* out,
    const sycl::half* res,
    const sycl::half* a,
    const sycl::half* b,
    float* acc_ptr,
    uint32_t* cnt_ptr,
    const int m,
    const int n,
    const int k,
    const float alpha,
    const float beta,
    const gpu::xetla::gpu_arch arch_tag) {
  switch (arch_tag) {
#ifdef USE_XETLA_XE_HPC
    case gpu::xetla::gpu_arch::XeHpc:
      return xehpc::hgemm_addmm(
          policy_id, out, res, a, b, acc_ptr, cnt_ptr, m, n, k, alpha, beta);
#endif
    default:
      break;
  }
  return {};
}

XETLA_KERNEL_API cgfs_t hgemm_common(
    const int policy_id,
    sycl::half* out,
    const sycl::half* a,
    const sycl::half* b,
    float* acc_ptr,
    uint32_t* cnt_ptr,
    const int m,
    const int n,
    const int k,
    const gpu::xetla::gpu_arch arch_tag) {
  switch (arch_tag) {
#ifdef USE_XETLA_XE_HPC
    case gpu::xetla::gpu_arch::XeHpc:
      return xehpc::hgemm_common(
          policy_id, out, a, b, acc_ptr, cnt_ptr, m, n, k);
#endif
    default:
      break;
  }
  return {};
}

XETLA_KERNEL_API cgfs_t hgemm_res(
    const int policy_id,
    sycl::half* out,
    const sycl::half* a,
    const sycl::half* b,
    const sycl::half* res,
    float* acc_ptr,
    uint32_t* cnt_ptr,
    const int m,
    const int n,
    const int k,
    const float res_factor,
    const gpu::xetla::gpu_arch arch_tag) {
  switch (arch_tag) {
#ifdef USE_XETLA_XE_HPC
    case gpu::xetla::gpu_arch::XeHpc:
      return xehpc::hgemm_res(
          policy_id, out, a, b, res, acc_ptr, cnt_ptr, m, n, k, res_factor);
#endif
    default:
      break;
  }
  return {};
}

XETLA_KERNEL_API cgfs_t hgemm_res_res(
    const int policy_id,
    sycl::half* out,
    const sycl::half* a,
    const sycl::half* b,
    const sycl::half* res0,
    const sycl::half* res1,
    float* acc_ptr,
    uint32_t* cnt_ptr,
    const int m,
    const int n,
    const int k,
    const float res0_factor,
    const float res1_factor,
    const gpu::xetla::gpu_arch arch_tag) {
  switch (arch_tag) {
#ifdef USE_XETLA_XE_HPC
    case gpu::xetla::gpu_arch::XeHpc:
      return xehpc::hgemm_res_res(
          policy_id,
          out,
          a,
          b,
          res0,
          res1,
          acc_ptr,
          cnt_ptr,
          m,
          n,
          k,
          res0_factor,
          res1_factor);
#endif
    default:
      break;
  }
  return {};
}

XETLA_KERNEL_API cgfs_t hgemm_bias(
    const int policy_id,
    sycl::half* out,
    const sycl::half* a,
    const sycl::half* b,
    const sycl::half* bias,
    float* acc_ptr,
    uint32_t* cnt_ptr,
    const int m,
    const int n,
    const int k,
    const float bias_factor,
    const gpu::xetla::gpu_arch arch_tag) {
  switch (arch_tag) {
#ifdef USE_XETLA_XE_HPC
    case gpu::xetla::gpu_arch::XeHpc:
      return xehpc::hgemm_bias(
          policy_id, out, a, b, bias, acc_ptr, cnt_ptr, m, n, k, bias_factor);
#endif
    default:
      break;
  }
  return {};
}

XETLA_KERNEL_API cgfs_t hgemm_bias_res(
    const int policy_id,
    sycl::half* out,
    const sycl::half* a,
    const sycl::half* b,
    const sycl::half* bias,
    const sycl::half* res,
    float* acc_ptr,
    uint32_t* cnt_ptr,
    const int m,
    const int n,
    const int k,
    const float bias_factor,
    const float res_factor,
    const gpu::xetla::gpu_arch arch_tag) {
  switch (arch_tag) {
#ifdef USE_XETLA_XE_HPC
    case gpu::xetla::gpu_arch::XeHpc:
      return xehpc::hgemm_bias_res(
          policy_id,
          out,
          a,
          b,
          bias,
          res,
          acc_ptr,
          cnt_ptr,
          m,
          n,
          k,
          bias_factor,
          res_factor);
#endif
    default:
      break;
  }
  return {};
}

XETLA_KERNEL_API cgfs_t hgemm_bias_res_res(
    const int policy_id,
    sycl::half* out,
    const sycl::half* a,
    const sycl::half* b,
    const sycl::half* bias,
    const sycl::half* res0,
    const sycl::half* res1,
    float* acc_ptr,
    uint32_t* cnt_ptr,
    const int m,
    const int n,
    const int k,
    const float bias_factor,
    const float res0_factor,
    const float res1_factor,
    const gpu::xetla::gpu_arch arch_tag) {
  switch (arch_tag) {
#ifdef USE_XETLA_XE_HPC
    case gpu::xetla::gpu_arch::XeHpc:
      return xehpc::hgemm_bias_res_res(
          policy_id,
          out,
          a,
          b,
          bias,
          res0,
          res1,
          acc_ptr,
          cnt_ptr,
          m,
          n,
          k,
          bias_factor,
          res0_factor,
          res1_factor);
#endif
    default:
      break;
  }
  return {};
}

XETLA_KERNEL_API cgfs_t hgemm_bias_relu(
    const int policy_id,
    sycl::half* out,
    const sycl::half* a,
    const sycl::half* b,
    const sycl::half* bias,
    float* acc_ptr,
    uint32_t* cnt_ptr,
    const int m,
    const int n,
    const int k,
    const float bias_factor,
    const gpu::xetla::gpu_arch arch_tag) {
  switch (arch_tag) {
#ifdef USE_XETLA_XE_HPC
    case gpu::xetla::gpu_arch::XeHpc:
      return xehpc::hgemm_bias_relu(
          policy_id, out, a, b, bias, acc_ptr, cnt_ptr, m, n, k, bias_factor);
#endif
    default:
      break;
  }
  return {};
}

XETLA_KERNEL_API cgfs_t hgemm_bias_gelu(
    const int policy_id,
    sycl::half* out,
    const sycl::half* a,
    const sycl::half* b,
    const sycl::half* bias,
    float* acc_ptr,
    uint32_t* cnt_ptr,
    const int m,
    const int n,
    const int k,
    const float bias_factor,
    const gpu::xetla::gpu_arch arch_tag) {
  switch (arch_tag) {
#ifdef USE_XETLA_XE_HPC
    case gpu::xetla::gpu_arch::XeHpc:
      return xehpc::hgemm_bias_gelu(
          policy_id, out, a, b, bias, acc_ptr, cnt_ptr, m, n, k, bias_factor);
#endif
    default:
      break;
  }
  return {};
}

XETLA_KERNEL_API cgfs_t hgemm_resmul(
    const int policy_id,
    sycl::half* out,
    const sycl::half* a,
    const sycl::half* b,
    const sycl::half* mul,
    float* acc_ptr,
    uint32_t* cnt_ptr,
    const int m,
    const int n,
    const int k,
    const gpu::xetla::gpu_arch arch_tag) {
  switch (arch_tag) {
#ifdef USE_XETLA_XE_HPC
    case gpu::xetla::gpu_arch::XeHpc:
      return xehpc::hgemm_resmul(
          policy_id, out, a, b, mul, acc_ptr, cnt_ptr, m, n, k);
#endif
    default:
      break;
  }
  return {};
}

XETLA_KERNEL_API cgfs_t hgemm_silu(
    const int policy_id,
    sycl::half* out,
    const sycl::half* a,
    const sycl::half* b,
    float* acc_ptr,
    uint32_t* cnt_ptr,
    const int m,
    const int n,
    const int k,
    const gpu::xetla::gpu_arch arch_tag) {
  switch (arch_tag) {
#ifdef USE_XETLA_XE_HPC
    case gpu::xetla::gpu_arch::XeHpc:
      return xehpc::hgemm_silu(policy_id, out, a, b, acc_ptr, cnt_ptr, m, n, k);
#endif
    default:
      break;
  }
  return {};
}

XETLA_KERNEL_API cgfs_t hgemm_qkv(
    const int policy_id,
    sycl::half* out0,
    sycl::half* out1,
    sycl::half* out2,
    const sycl::half* a,
    const sycl::half* b,
    float* acc_ptr,
    uint32_t* cnt_ptr,
    const int m,
    const int n,
    const int k,
    const gpu::xetla::gpu_arch arch_tag) {
  switch (arch_tag) {
#ifdef USE_XETLA_XE_HPC
    case gpu::xetla::gpu_arch::XeHpc:
      return xehpc::hgemm_qkv(
          policy_id, out0, out1, out2, a, b, acc_ptr, cnt_ptr, m, n, k);
#endif
    default:
      break;
  }
  return {};
}

XETLA_KERNEL_API cgfs_t hgemm_qkv_bias(
    const int policy_id,
    sycl::half* out0,
    sycl::half* out1,
    sycl::half* out2,
    const sycl::half* a,
    const sycl::half* b,
    const sycl::half* bias,
    float* acc_ptr,
    uint32_t* cnt_ptr,
    const int m,
    const int n,
    const int k,
    const gpu::xetla::gpu_arch arch_tag) {
  switch (arch_tag) {
#ifdef USE_XETLA_XE_HPC
    case gpu::xetla::gpu_arch::XeHpc:
      return xehpc::hgemm_qkv_bias(
          policy_id, out0, out1, out2, a, b, bias, acc_ptr, cnt_ptr, m, n, k);
#endif
    default:
      break;
  }
  return {};
}

XETLA_KERNEL_API cgfs_t hgemm_qkv_group(
    const int policy_id,
    sycl::half* out0,
    sycl::half* out1,
    sycl::half* out2,
    const sycl::half* a,
    const sycl::half* b,
    float* acc_ptr,
    uint32_t* cnt_ptr,
    const int m,
    const int n,
    const int k,
    const int num_kv_head,
    const int group,
    const int head_dim,
    const gpu::xetla::gpu_arch arch_tag) {
  switch (arch_tag) {
#ifdef USE_XETLA_XE_HPC
    case gpu::xetla::gpu_arch::XeHpc:
      return xehpc::hgemm_qkv_group(
          policy_id,
          out0,
          out1,
          out2,
          a,
          b,
          acc_ptr,
          cnt_ptr,
          m,
          n,
          k,
          num_kv_head,
          group,
          head_dim);
#endif
    default:
      break;
  }
  return {};
}

XETLA_KERNEL_API cgfs_t hgemm_qkv_group_bias(
    const int policy_id,
    sycl::half* out0,
    sycl::half* out1,
    sycl::half* out2,
    const sycl::half* a,
    const sycl::half* b,
    const sycl::half* bias,
    float* acc_ptr,
    uint32_t* cnt_ptr,
    const int m,
    const int n,
    const int k,
    const int num_kv_head,
    const int group,
    const int head_dim,
    const gpu::xetla::gpu_arch arch_tag) {
  switch (arch_tag) {
#ifdef USE_XETLA_XE_HPC
    case gpu::xetla::gpu_arch::XeHpc:
      return xehpc::hgemm_qkv_group_bias(
          policy_id,
          out0,
          out1,
          out2,
          a,
          b,
          bias,
          acc_ptr,
          cnt_ptr,
          m,
          n,
          k,
          num_kv_head,
          group,
          head_dim);
#endif
    default:
      break;
  }
  return {};
}

} // namespace torch_ipex::xpu::xetla
