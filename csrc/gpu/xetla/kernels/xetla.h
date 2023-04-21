#pragma once

#define XETPP_CODE_BASE __ESIMD__
#define ESIMD_XE_HPC

#include <brgemm/xetpp_brgemm.hpp>
#include <core/xetpp_core.hpp>
#include <gemm/xetpp_gemm.hpp>
#include <layer_norm/xetpp_layer_norm_bwd_xe.hpp>
#include <layer_norm/xetpp_layer_norm_config.hpp>
#include <layer_norm/xetpp_layer_norm_fwd_xe.hpp>
#include <reduction/xetpp_reduction.hpp>
#include <reduction/xetpp_reduction_api.hpp>
#include <reduction/xetpp_reduction_config.hpp>
#include <sycl/sycl.hpp>
#include <util/xetpp_util.hpp>

using namespace sycl;
using namespace __XETPP_TILE_NS;
using namespace __XETPP_REDUCTION_NS;
using namespace __XETPP_NS;
using namespace __XETPP_BRGEMM_NS;
using namespace __XETPP_GEMM_NS;
using namespace __XETPP_REDUCTION_NS;
using namespace __XETPP_LAYER_NORM_NS;

template <typename data_type>
inline auto getTypeName() {
  fprintf(stderr, "FAIL: Not implemented specialization\n");
  exit(-1);
}

template <>
inline auto getTypeName<int>() {
  return "_int";
}
template <>
inline auto getTypeName<float>() {
  return "_float";
}
template <>
inline auto getTypeName<uint32_t>() {
  return "_uint32_t";
}
template <>
inline auto getTypeName<double>() {
  return "_double";
}

template <>
inline auto getTypeName<__XETPP_NS::bf16>() {
  return "_bf16";
}

template <>
inline auto getTypeName<__XETPP_NS::fp16>() {
  return "_fp16";
}

template <>
inline auto getTypeName<__XETPP_NS::tf32>() {
  return "_tf32";
}
