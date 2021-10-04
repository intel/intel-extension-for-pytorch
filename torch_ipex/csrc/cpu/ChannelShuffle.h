#pragma once

namespace torch_ipex {
namespace cpu {

template <typename scalar_t>
void cpu_channel_shuffle(
    at::Tensor& output,
    const at::Tensor& input,
    int64_t groups);

template <typename scalar_t, typename func_t>
static void parallel_nd(
    scalar_t* input_data,
    scalar_t* output_data,
    int64_t M,
    int64_t C,
    const func_t& f);

template <typename scalar_t>
void cpu_channel_shuffle_cl(
    at::Tensor& output,
    const at::Tensor& input,
    int64_t groups);

void channel_shuffle_kernel(
    at::Tensor& output,
    const at::Tensor& input,
    int64_t groups);

at::Tensor channel_shuffle(const at::Tensor& self, int64_t groups);

} // namespace cpu
} // namespace torch_ipex