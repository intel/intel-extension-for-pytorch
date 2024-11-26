#pragma once

#include <sycl/sycl.hpp>
#include "xetla_kernel_api.h"

namespace torch_ipex::xpu::xetla {

template <typename output_t, typename input_t>
XETLA_KERNEL_API cgf_t sgmv_shrink(
    gpu::xetla::gpu_arch arch_tag,
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
    float scale);

template <typename output_t, typename input_t>
XETLA_KERNEL_API cgf_t sgmv_expand_with_slice(
    gpu::xetla::gpu_arch arch_tag,
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
    bool add_to_output);

} // namespace torch_ipex::xpu::xetla