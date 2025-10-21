#pragma once

#include <ATen/ATen.h>
#include <ATen/record_function.h>
#include <runtime/Utils.h>
#include <utils/oneMKLUtils.h>
#include "xetla/GEMM_INT4_marlin.h"

using namespace torch_ipex::xpu::xetla;

#define RECORD_FUNCTION_IMPL(F, m_, n_, k_)            \
  char str__[100];                                     \
  sprintf(str__, "%s(%d, %d, %d)", "" #F, m_, n_, k_); \
  RECORD_FUNCTION(str__, c10::ArrayRef<c10::IValue>({}));

template <
    typename dtype_a,
    typename dtype_b,
    typename dtype_c,
    typename dtype_zp,
    typename dtype_scale>
void launch_hgemm_wint4_marlin(
    dtype_c* out,
    const dtype_a* a,
    const dtype_b* b,
    const dtype_zp* b_zp,
    const dtype_scale* b_scale,
    const uint32_t m,
    const uint32_t n,
    const uint32_t k) {
  auto& q = torch_ipex::xpu::dpcpp::dpcppGetCurrentQueue();
  RECORD_FUNCTION_IMPL(hgemm_wint4_marlin, m, n, k);
  if (m <= 4) {
    auto cgfs = hgemm_wint4_marlin<
        dtype_a,
        dtype_b,
        dtype_c,
        dtype_zp,
        dtype_scale,
        GEMV>(out, a, b, b_zp, b_scale, m, n, k);
    DPCPP_Q_SUBMIT_CGFS(q, cgfs);
  } else if (m <= 32) {
    auto cgfs = hgemm_wint4_marlin<
        dtype_a,
        dtype_b,
        dtype_c,
        dtype_zp,
        dtype_scale,
        GEMV_16>(out, a, b, b_zp, b_scale, m, n, k);
    DPCPP_Q_SUBMIT_CGFS(q, cgfs);
  } else if (m <= 128) {
    auto cgfs = hgemm_wint4_marlin<
        dtype_a,
        dtype_b,
        dtype_c,
        dtype_zp,
        dtype_scale,
        GEMV_32>(out, a, b, b_zp, b_scale, m, n, k);
    DPCPP_Q_SUBMIT_CGFS(q, cgfs);
  } else {
    auto cgfs = hgemm_wint4_marlin<
        dtype_a,
        dtype_b,
        dtype_c,
        dtype_zp,
        dtype_scale,
        GEMM>(out, a, b, b_zp, b_scale, m, n, k);
    DPCPP_Q_SUBMIT_CGFS(q, cgfs);
  }

  return;
}

template <
    typename dtype_a,
    typename dtype_b,
    typename dtype_c,
    typename dtype_zp,
    typename dtype_scale>
void launch_group_hgemm_wint4_marlin(
    dtype_c* out,
    const dtype_a* a,
    const dtype_b* b,
    const dtype_zp* b_zp,
    const dtype_scale* b_scale,
    const dtype_a* bias,
    const int* atomic_buffer,
    const int* total_rows_for_each_expert,
    const int expert_num,
    const uint32_t average_m,
    const uint32_t n,
    const uint32_t k) {
  auto& q = torch_ipex::xpu::dpcpp::dpcppGetCurrentQueue();

  if (average_m <= 4) {
    auto cgfs = group_hgemm_wint4_marlin<
        dtype_a,
        dtype_b,
        dtype_c,
        dtype_zp,
        dtype_scale,
        GEMV>(
        out,
        a,
        b,
        b_zp,
        b_scale,
        bias,
        atomic_buffer,
        total_rows_for_each_expert,
        expert_num,
        n,
        k);
    DPCPP_Q_SUBMIT_CGFS(q, cgfs);
  } else if (average_m <= 32) {
    auto cgfs = group_hgemm_wint4_marlin<
        dtype_a,
        dtype_b,
        dtype_c,
        dtype_zp,
        dtype_scale,
        GEMV_16>(
        out,
        a,
        b,
        b_zp,
        b_scale,
        bias,
        atomic_buffer,
        total_rows_for_each_expert,
        expert_num,
        n,
        k);
    DPCPP_Q_SUBMIT_CGFS(q, cgfs);
  } else if (average_m <= 128) {
    auto cgfs = group_hgemm_wint4_marlin<
        dtype_a,
        dtype_b,
        dtype_c,
        dtype_zp,
        dtype_scale,
        GEMV_32>(
        out,
        a,
        b,
        b_zp,
        b_scale,
        bias,
        atomic_buffer,
        total_rows_for_each_expert,
        expert_num,
        n,
        k);
    DPCPP_Q_SUBMIT_CGFS(q, cgfs);
  } else {
    auto cgfs = group_hgemm_wint4_marlin<
        dtype_a,
        dtype_b,
        dtype_c,
        dtype_zp,
        dtype_scale,
        GEMM>(
        out,
        a,
        b,
        b_zp,
        b_scale,
        bias,
        atomic_buffer,
        total_rows_for_each_expert,
        expert_num,
        n,
        k);
    DPCPP_Q_SUBMIT_CGFS(q, cgfs);
  }

  return;
}
