#include "sgmv.h"
#include "../../xetla_kernel_api.h"

namespace c10 {
struct Half;
struct BFloat16;
} // namespace c10

namespace gpu::xetla {

template <typename T>
struct ToXetlaType {
  using type = T;
};

template <>
struct ToXetlaType<c10::Half> {
  using type = fp16;
};

template <>
struct ToXetlaType<c10::BFloat16> {
  using type = bf16;
};
} // namespace gpu::xetla

namespace torch_ipex::xpu::xetla {

using namespace gpu::xetla;

template <typename output_t, typename input_t, gpu_arch arch_tag>
XETLA_KERNEL_API cgf_t dispatch_sgmv_shrink(
    output_t* outputs,
    input_t* inputs,
    input_t* weights,
    int64_t* seq_start_locs,
    int64_t* seq_lens,
    int64_t* lora_indices,
    uint32_t batches,
    uint32_t max_seq_len,
    uint32_t gemm_k,
    uint32_t gemm_n,
    float scale) {
  using OutputT = ToXetlaType<output_t>::type;
  using InputT = ToXetlaType<input_t>::type;
  using Policy = gpu::xetla::SgmvShrinkPolicy;
  using kernel = SgmvShrinkKernel<OutputT, InputT, Policy, arch_tag>;

  auto result = [=](sycl::handler& cgh) {
    kernel task(
        reinterpret_cast<OutputT*>(outputs),
        reinterpret_cast<InputT*>(inputs),
        reinterpret_cast<InputT*>(weights),
        seq_start_locs,
        seq_lens,
        lora_indices,
        batches,
        max_seq_len,
        gemm_k,
        gemm_n,
        scale);
    cgh.parallel_for(task.get_nd_range(), task);
  };
  return result;
}

template <typename output_t, typename input_t, gpu_arch arch_tag>
XETLA_KERNEL_API cgf_t dispatch_sgmv_expand_with_slice(
    output_t* outputs,
    input_t* inputs,
    output_t* weights,
    int64_t* seq_start_locs,
    int64_t* seq_lens,
    int64_t* lora_indices,
    uint32_t batches,
    uint32_t max_seq_len,
    uint32_t gemm_k,
    uint32_t gemm_n,
    uint32_t slice_offset,
    uint32_t output_hidden,
    bool add_to_output) {
  using OutputT = ToXetlaType<output_t>::type;
  using InputT = ToXetlaType<input_t>::type;
  using Policy = gpu::xetla::SgmvExpandPolicy;
  using kernel = SgmvExpandKernel<OutputT, InputT, Policy, arch_tag>;

  auto result = [=](sycl::handler& cgh) {
    kernel task(
        reinterpret_cast<OutputT*>(outputs),
        reinterpret_cast<InputT*>(inputs),
        reinterpret_cast<OutputT*>(weights),
        seq_start_locs,
        seq_lens,
        lora_indices,
        batches,
        max_seq_len,
        gemm_k,
        gemm_n,
        slice_offset,
        output_hidden,
        add_to_output);
    cgh.parallel_for(task.get_nd_range(), task);
  };
  return result;
}

template <typename output_t, typename input_t>
XETLA_KERNEL_API cgf_t sgmv_shrink(
    gpu_arch arch_tag,
    output_t* outputs,
    input_t* inputs,
    input_t* weights,
    int64_t* seq_start_locs,
    int64_t* seq_lens,
    int64_t* lora_indices,
    uint32_t batches,
    uint32_t max_seq_len,
    uint32_t gemm_k,
    uint32_t gemm_n,
    float scale) {
  switch (arch_tag) {
#ifdef USE_XETLA_XE_HPG
    case gpu_arch::XeHpg:
      return dispatch_sgmv_shrink<output_t, input_t, gpu_arch::XeHpg>(
          outputs,
          inputs,
          weights,
          seq_start_locs,
          seq_lens,
          lora_indices,
          batches,
          max_seq_len,
          gemm_k,
          gemm_n,
          scale);
#endif
#ifdef USE_XETLA_XE_HPC
    case gpu_arch::XeHpc:
      return dispatch_sgmv_shrink<output_t, input_t, gpu_arch::XeHpc>(
          outputs,
          inputs,
          weights,
          seq_start_locs,
          seq_lens,
          lora_indices,
          batches,
          max_seq_len,
          gemm_k,
          gemm_n,
          scale);
#endif
    default:
      printf("Unsupported gpu_arch of sgmv_shrink!!\n\n");
      return {};
  }
}

template <typename output_t, typename input_t>
XETLA_KERNEL_API cgf_t sgmv_expand_with_slice(
    gpu_arch arch_tag,
    output_t* outputs,
    input_t* inputs,
    output_t* weights,
    int64_t* seq_start_locs,
    int64_t* seq_lens,
    int64_t* lora_indices,
    uint32_t batches,
    uint32_t max_seq_len,
    uint32_t gemm_k,
    uint32_t gemm_n,
    uint32_t slice_offset,
    uint32_t output_hidden,
    bool add_to_output) {
  switch (arch_tag) {
#ifdef USE_XETLA_XE_HPG
    case gpu_arch::XeHpg:
      return dispatch_sgmv_expand_with_slice<
          output_t,
          input_t,
          gpu_arch::XeHpg>(
          outputs,
          inputs,
          weights,
          seq_start_locs,
          seq_lens,
          lora_indices,
          batches,
          max_seq_len,
          gemm_k,
          gemm_n,
          slice_offset,
          output_hidden,
          add_to_output);
#endif
#ifdef USE_XETLA_XE_HPC
    case gpu_arch::XeHpc:
      return dispatch_sgmv_expand_with_slice<
          output_t,
          input_t,
          gpu_arch::XeHpc>(
          outputs,
          inputs,
          weights,
          seq_start_locs,
          seq_lens,
          lora_indices,
          batches,
          max_seq_len,
          gemm_k,
          gemm_n,
          slice_offset,
          output_hidden,
          add_to_output);
#endif
    default:
      printf("Unsupported gpu_arch of sgmv_expand!!\n\n");
      return {};
  }
}

#define SGMVSHRINK_INSTANTIATE(io_t, rank_t)                 \
  template cgf_t XETLA_KERNEL_API sgmv_shrink<rank_t, io_t>( \
      gpu_arch,                                              \
      rank_t*,                                               \
      io_t*,                                                 \
      io_t*,                                                 \
      int64_t*,                                              \
      int64_t*,                                              \
      int64_t*,                                              \
      uint32_t,                                              \
      uint32_t,                                              \
      uint32_t,                                              \
      uint32_t,                                              \
      float);

#define SGMVSHRINK_INSTANTIATE_FOR_RANK_TYPES(rank_t) \
  SGMVSHRINK_INSTANTIATE(c10::Half, rank_t)           \
  SGMVSHRINK_INSTANTIATE(c10::BFloat16, rank_t)

SGMVSHRINK_INSTANTIATE_FOR_RANK_TYPES(float)
SGMVSHRINK_INSTANTIATE_FOR_RANK_TYPES(c10::Half)
SGMVSHRINK_INSTANTIATE_FOR_RANK_TYPES(c10::BFloat16)

#define SGMVEXPAND_INSTANTIATE(io_t, rank_t)                            \
  template cgf_t XETLA_KERNEL_API sgmv_expand_with_slice<io_t, rank_t>( \
      gpu_arch,                                                         \
      io_t*,                                                            \
      rank_t*,                                                          \
      io_t*,                                                            \
      int64_t*,                                                         \
      int64_t*,                                                         \
      int64_t*,                                                         \
      uint32_t,                                                         \
      uint32_t,                                                         \
      uint32_t,                                                         \
      uint32_t,                                                         \
      uint32_t,                                                         \
      uint32_t,                                                         \
      bool);

SGMVEXPAND_INSTANTIATE(c10::Half, float)
SGMVEXPAND_INSTANTIATE(c10::BFloat16, float)
SGMVEXPAND_INSTANTIATE(c10::Half, c10::Half)
SGMVEXPAND_INSTANTIATE(c10::BFloat16, c10::BFloat16)

#undef SGMVEXPAND_INSTANTIATE
#undef SGMVSHRINK_INSTANTIATE_FOR_RANK_TYPES
#undef SGMVSHRINK_INSTANTIATE

} // namespace torch_ipex::xpu::xetla