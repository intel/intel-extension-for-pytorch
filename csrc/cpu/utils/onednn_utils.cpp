#include "onednn_utils.h"

#include <ideep.hpp>

namespace torch_ipex {
namespace utils {

int onednn_set_verbose(int level) {
  return ideep::utils::set_verbose(level);
}

bool onednn_has_bf16_type_support() {
  return ideep::has_bf16_type_support();
}

bool onednn_has_fp16_type_support() {
  return ideep::has_fp16_type_support();
}

bool onednn_has_fp8_type_support() {
  auto engine = ideep::engine::cpu_engine();
  int64_t M = 2, K = 3, N = 4;
  std::vector<int64_t> src_dims = {M, K};
  std::vector<int64_t> weight_dims = {K, N};
  std::vector<int64_t> dst_dims = {M, N};

  auto src_desc = ideep::tensor::desc(
      src_dims, ideep::tensor::data_type::f8_e4m3, ideep::format_tag::any);
  auto weights_desc = ideep::tensor::desc(
      weight_dims, ideep::tensor::data_type::f8_e4m3, ideep::format_tag::any);
  auto dst_desc = ideep::tensor::desc(
      dst_dims, ideep::tensor::data_type::f32, ideep::format_tag::any);

  try {
    auto primitive_desc = dnnl::matmul::primitive_desc(
        engine, src_desc, weights_desc, dst_desc, ideep::attr_t());
  } catch (dnnl::error& e) {
    if (e.status == dnnl_unimplemented)
      return false;
  }
  return true;
}

} // namespace utils
} // namespace torch_ipex
