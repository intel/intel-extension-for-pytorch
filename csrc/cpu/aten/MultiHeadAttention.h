#pragma once

#include <ATen/ATen.h>
#include <cpu/kernels/Matmul.h>
#include <cpu/kernels/Mha.h>
#include <cpu/kernels/Softmax.h>
#include <dyndisp/DispatchStub.h>
#include "AddSoftmax.h"
#include "DivSoftmax.h"

namespace torch_ipex {
namespace cpu {

// This operator assumes that the softmax is applied to the last
// dimension.
at::Tensor bert_flash_mha(
    const at::Tensor& qkv,
    const at::Tensor& rel_kv,
    const int64_t& head_num,
    const int64_t& headSize,
    const double& dim_per_head);

at::Tensor sd_flash_mha(
    const at::Tensor& qkv,
    const int64_t& head_num,
    const int64_t& headSize,
    const double& scale);

at::Tensor sd_flash_mha(
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    const int64_t& head_num,
    const int64_t& headSize,
    const double& scale);

namespace {
at::Tensor bert_mha_kernel_impl(
    const at::Tensor& qkv,
    const at::Tensor& rel_kv,
    const int64_t& head_num,
    const int64_t& headSize,
    const double& dim_per_head);

at::Tensor sd_mha_kernel_v1_impl(
    const at::Tensor& qkv,
    const int64_t& head_num,
    const int64_t& headSize,
    const double& scale);

at::Tensor sd_mha_kernel_v2_impl(
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    const int64_t& head_num,
    const int64_t& headSize,
    const double& scale);
} // namespace

using bert_mha_kernel_fn = at::Tensor (*)(
    const at::Tensor&,
    const at::Tensor&,
    const int64_t&,
    const int64_t&,
    const double&);

using sd_mha_kernel_v1_fn = at::Tensor (*)(
    const at::Tensor&,
    const int64_t&,
    const int64_t&,
    const double&);

using sd_mha_kernel_v2_fn = at::Tensor (*)(
    const at::Tensor&,
    const at::Tensor&,
    const at::Tensor&,
    const int64_t&,
    const int64_t&,
    const double&);

IPEX_DECLARE_DISPATCH(bert_mha_kernel_fn, bert_mha_kernel_stub);
IPEX_DECLARE_DISPATCH(sd_mha_kernel_v1_fn, sd_mha_kernel_v1_stub);
IPEX_DECLARE_DISPATCH(sd_mha_kernel_v2_fn, sd_mha_kernel_v2_stub);
} // namespace cpu
} // namespace torch_ipex
