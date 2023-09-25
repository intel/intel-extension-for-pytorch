#pragma once

#include <ATen/ATen.h>
#include <ATen/record_function.h>
#include <oneDNN/oneDNN.h>
#include <runtime/Utils.h>
#include <utils/oneMKLUtils.h>
#include "comm/ATDispatch.h"
#include "comm/RegistrationDeclarations.h"
#include "xetla/GEMM_INT4.h"

using namespace xpu::xetla;

#define RECORD_FUNCTION_IMPL(F)                        \
  char str__[100];                                     \
  sprintf(str__, "%s(%d, %d, %d)", "" #F, m_, n_, k_); \
  RECORD_FUNCTION(str__, c10::ArrayRef<c10::IValue>({}));

#define HGEMM_INT4_DISPATCH(F)                                          \
  {                                                                     \
    RECORD_FUNCTION_IMPL(F)                                             \
    F(q,                                                                \
      reinterpret_cast<sycl::half*>(output_->data_ptr<scalar_t>()),     \
      reinterpret_cast<sycl::half*>(input_->data_ptr<scalar_t>()),      \
      weight_->data_ptr<uint8_t>(),                                     \
      weight_zp_->data_ptr<uint8_t>(),                                  \
      reinterpret_cast<sycl::half*>(weight_scl_->data_ptr<scalar_t>()), \
      m_,                                                               \
      n_,                                                               \
      k_);                                                              \
  }

#define HGEMM_INT4_BIAS_DISPATCH(F)                                       \
  {                                                                       \
    RECORD_FUNCTION_IMPL(F)                                               \
    F(q,                                                                  \
      reinterpret_cast<sycl::half*>(output_->data_ptr<scalar_t>()),       \
      reinterpret_cast<sycl::half*>(input_->data_ptr<scalar_t>()),        \
      weight_->data_ptr<uint8_t>(),                                       \
      weight_zp_->data_ptr<uint8_t>(),                                    \
      reinterpret_cast<sycl::half*>(weight_scl_->data_ptr<scalar_t>()),   \
      reinterpret_cast<sycl::half*>(epilogues_[0]->data_ptr<scalar_t>()), \
      m_,                                                                 \
      n_,                                                                 \
      k_);                                                                \
  }

#define HGEMM_INT4_BIAS_RES_RES_DISPATCH(F)                               \
  {                                                                       \
    RECORD_FUNCTION_IMPL(F)                                               \
    F(q,                                                                  \
      reinterpret_cast<sycl::half*>(output_->data_ptr<scalar_t>()),       \
      reinterpret_cast<sycl::half*>(input_->data_ptr<scalar_t>()),        \
      weight_->data_ptr<uint8_t>(),                                       \
      weight_zp_->data_ptr<uint8_t>(),                                    \
      reinterpret_cast<sycl::half*>(weight_scl_->data_ptr<scalar_t>()),   \
      reinterpret_cast<sycl::half*>(epilogues_[0]->data_ptr<scalar_t>()), \
      reinterpret_cast<sycl::half*>(epilogues_[1]->data_ptr<scalar_t>()), \
      reinterpret_cast<sycl::half*>(epilogues_[2]->data_ptr<scalar_t>()), \
      m_,                                                                 \
      n_,                                                                 \
      k_);                                                                \
  }

#define HGEMM_INT4_BIAS_GELU_DISPATCH(F)                                  \
  {                                                                       \
    RECORD_FUNCTION_IMPL(F)                                               \
    F(q,                                                                  \
      reinterpret_cast<sycl::half*>(output_->data_ptr<scalar_t>()),       \
      reinterpret_cast<sycl::half*>(input_->data_ptr<scalar_t>()),        \
      weight_->data_ptr<uint8_t>(),                                       \
      weight_zp_->data_ptr<uint8_t>(),                                    \
      reinterpret_cast<sycl::half*>(weight_scl_->data_ptr<scalar_t>()),   \
      reinterpret_cast<sycl::half*>(epilogues_[0]->data_ptr<scalar_t>()), \
      m_,                                                                 \
      n_,                                                                 \
      k_);                                                                \
  }

#define HGEMM_INT4_RES_DISPATCH(F)                                        \
  {                                                                       \
    RECORD_FUNCTION_IMPL(F)                                               \
    F(q,                                                                  \
      reinterpret_cast<sycl::half*>(output_->data_ptr<scalar_t>()),       \
      reinterpret_cast<sycl::half*>(input_->data_ptr<scalar_t>()),        \
      weight_->data_ptr<uint8_t>(),                                       \
      weight_zp_->data_ptr<uint8_t>(),                                    \
      reinterpret_cast<sycl::half*>(weight_scl_->data_ptr<scalar_t>()),   \
      reinterpret_cast<sycl::half*>(epilogues_[0]->data_ptr<scalar_t>()), \
      m_,                                                                 \
      n_,                                                                 \
      k_);                                                                \
  }

#define HGEMM_INT4_COMMON_DISPATCH_IMPL(DISPATCHER, F) DISPATCHER(F)
#define HGEMM_INT4_COMMON_DISPATCH(F)                                      \
  {                                                                        \
    if (num_epilogues_ == 0)                                               \
      HGEMM_INT4_COMMON_DISPATCH_IMPL(HGEMM_INT4_DISPATCH, hgemm_wint4##F) \
    else if (num_epilogues_ == 1 && epilogue_type_[0] == BIAS)             \
      HGEMM_INT4_COMMON_DISPATCH_IMPL(                                     \
          HGEMM_INT4_BIAS_DISPATCH, hgemm_bias_wint4##F)                   \
    else if (                                                              \
        num_epilogues_ == 3 && epilogue_type_[0] == BIAS &&                \
        epilogue_type_[1] == RES_ADD && epilogue_type_[2] == RES_ADD)      \
      HGEMM_INT4_COMMON_DISPATCH_IMPL(                                     \
          HGEMM_INT4_BIAS_RES_RES_DISPATCH, hgemm_bias_res_res_wint4##F)   \
    else if (                                                              \
        num_epilogues_ == 2 && epilogue_type_[0] == BIAS &&                \
        epilogue_type_[1] == GELU)                                         \
      HGEMM_INT4_COMMON_DISPATCH_IMPL(                                     \
          HGEMM_INT4_BIAS_GELU_DISPATCH, hgemm_bias_gelu_wint4##F)         \
    else if (num_epilogues_ == 1 && epilogue_type_[0] == RES_ADD)          \
      HGEMM_INT4_COMMON_DISPATCH_IMPL(                                     \
          HGEMM_INT4_RES_DISPATCH, hgemm_res_wint4##F)                     \
  }

#define HGEMM_INT4_8M_16384N_4096K_IMPL(gz) \
  HGEMM_INT4_COMMON_DISPATCH(_##8x256_8x16x32##_##gz##_2_)
#define HGEMM_INT4_8M_4096N_4096K_IMPL(gz) \
  HGEMM_INT4_COMMON_DISPATCH(_##8x64_8x16x64##_##gz##_8_)
#define HGEMM_INT4_8M_4096N_16384K_IMPL(gz) \
  HGEMM_INT4_COMMON_DISPATCH(_##8x64_8x16x64##_##gz##_8_)
#define HGEMM_INT4_8M_50400N_4096K_IMPL(gz) \
  HGEMM_INT4_COMMON_DISPATCH(_##8x512_8x16x32##_##gz##_1_)
#define HGEMM_INT4_8M_50416N_4096K_IMPL(gz) \
  HGEMM_INT4_COMMON_DISPATCH(_##8x512_8x16x32##_##gz##_1_)

#define HGEMM_INT4_16M_16384N_4096K_IMPL(gz) \
  HGEMM_INT4_COMMON_DISPATCH(_##16x256_16x16x32##_##gz##_2_)
#define HGEMM_INT4_16M_4096N_4096K_IMPL(gz) \
  HGEMM_INT4_COMMON_DISPATCH(_##16x64_16x16x32##_##gz##_8_)
#define HGEMM_INT4_16M_4096N_16384K_IMPL(gz) \
  HGEMM_INT4_COMMON_DISPATCH(_##16x64_16x16x32##_##gz##_8_)
#define HGEMM_INT4_16M_50400N_4096K_IMPL(gz) \
  HGEMM_INT4_COMMON_DISPATCH(_##16x512_16x16x32##_##gz##_1_)
#define HGEMM_INT4_16M_50416N_4096K_IMPL(gz) \
  HGEMM_INT4_COMMON_DISPATCH(_##16x512_16x16x32##_##gz##_1_)

#define HGEMM_INT4_32M_16384N_4096K_IMPL(gz) \
  HGEMM_INT4_COMMON_DISPATCH(_##32x256_32x16x32##_##gz##_2_)
#define HGEMM_INT4_32M_4096N_4096K_IMPL(gz) \
  HGEMM_INT4_COMMON_DISPATCH(_##32x64_32x16x32##_##gz##_8_)
#define HGEMM_INT4_32M_4096N_16384K_IMPL(gz) \
  HGEMM_INT4_COMMON_DISPATCH(_##32x64_32x16x32##_##gz##_8_)
#define HGEMM_INT4_32M_50400N_4096K_IMPL(gz) \
  HGEMM_INT4_COMMON_DISPATCH(_##32x512_32x16x32##_##gz##_1_)
#define HGEMM_INT4_32M_50416N_4096K_IMPL(gz) \
  HGEMM_INT4_COMMON_DISPATCH(_##32x512_32x16x32##_##gz##_1_)

#define HGEMM_INT4_64M_16384N_4096K_IMPL(gz) \
  HGEMM_INT4_COMMON_DISPATCH(_##64x256_64x16x32##_##gz##_2_)
#define HGEMM_INT4_64M_4096N_4096K_IMPL(gz) \
  HGEMM_INT4_COMMON_DISPATCH(_##32x128_32x16x32##_##gz##_4_)
#define HGEMM_INT4_64M_4096N_16384K_IMPL(gz) \
  HGEMM_INT4_COMMON_DISPATCH(_##32x128_32x16x32##_##gz##_4_)
#define HGEMM_INT4_64M_50400N_4096K_IMPL(gz) \
  HGEMM_INT4_COMMON_DISPATCH(_##64x512_64x16x32##_##gz##_1_)
#define HGEMM_INT4_64M_50416N_4096K_IMPL(gz) \
  HGEMM_INT4_COMMON_DISPATCH(_##64x512_64x16x32##_##gz##_1_)

#define HGEMM_INT4_384M_16384N_4096K_IMPL(gz) \
  HGEMM_INT4_COMMON_DISPATCH(_##64x128_64x16x32##_##gz##_4_)
#define HGEMM_INT4_384M_4096N_4096K_IMPL(gz) \
  HGEMM_INT4_COMMON_DISPATCH(_##64x128_64x16x32##_##gz##_4_)
#define HGEMM_INT4_384M_4096N_16384K_IMPL(gz) \
  HGEMM_INT4_COMMON_DISPATCH(_##64x128_64x16x32##_##gz##_4_)
#define HGEMM_INT4_384M_50400N_4096K_IMPL(gz) \
  HGEMM_INT4_COMMON_DISPATCH(_##64x128_64x16x32##_##gz##_4_)
#define HGEMM_INT4_384M_50416N_4096K_IMPL(gz) \
  HGEMM_INT4_COMMON_DISPATCH(_##64x128_64x16x32##_##gz##_4_)

#define HGEMM_INT4_1024M_16384N_4096K_IMPL(gz) \
  HGEMM_INT4_COMMON_DISPATCH(_##128x256_64x16x32##_##gz##_1_)
#define HGEMM_INT4_1024M_4096N_4096K_IMPL(gz) \
  HGEMM_INT4_COMMON_DISPATCH(_##128x256_64x16x32##_##gz##_1_)
#define HGEMM_INT4_1024M_4096N_16384K_IMPL(gz) \
  HGEMM_INT4_COMMON_DISPATCH(_##128x256_64x16x32##_##gz##_1_)
#define HGEMM_INT4_1024M_50400N_4096K_IMPL(gz) \
  HGEMM_INT4_COMMON_DISPATCH(_##128x512_64x32x32##_##gz##_1_)
#define HGEMM_INT4_1024M_50416N_4096K_IMPL(gz) \
  HGEMM_INT4_COMMON_DISPATCH(_##128x512_64x32x32##_##gz##_1_)

#define HGEMM_INT4_INFM_16384N_4096K_IMPL(gz) \
  HGEMM_INT4_COMMON_DISPATCH(_##256x256_64x32x32##_##gz##_1_)
#define HGEMM_INT4_INFM_4096N_4096K_IMPL(gz) \
  HGEMM_INT4_COMMON_DISPATCH(_##256x256_64x32x32##_##gz##_1_)
#define HGEMM_INT4_INFM_4096N_16384K_IMPL(gz) \
  HGEMM_INT4_COMMON_DISPATCH(_##256x256_64x32x32##_##gz##_1_)
#define HGEMM_INT4_INFM_50400N_4096K_IMPL(gz) \
  HGEMM_INT4_COMMON_DISPATCH(_##256x256_64x32x32##_##gz##_1_)
#define HGEMM_INT4_INFM_50416N_4096K_IMPL(gz) \
  HGEMM_INT4_COMMON_DISPATCH(_##256x256_64x32x32##_##gz##_1_)

#define HGEMM_INT4_M_N_K_DISPATCH(m, n, k, gz) \
  HGEMM_INT4_##m##M_##n##N_##k##K_IMPL(gz)

#define HGEMM_INT4_N_K_DISPATCH(m, gz)             \
  if (n_ == 16384 && k_ == 4096) {                 \
    HGEMM_INT4_M_N_K_DISPATCH(m, 16384, 4096, gz); \
  } else if (n_ == 4096 && k_ == 4096) {           \
    HGEMM_INT4_M_N_K_DISPATCH(m, 4096, 4096, gz);  \
  } else if (n_ == 4096 && k_ == 16384) {          \
    HGEMM_INT4_M_N_K_DISPATCH(m, 4096, 16384, gz); \
  } else if (n_ == 50400 && k_ == 4096) {          \
    HGEMM_INT4_M_N_K_DISPATCH(m, 50400, 4096, gz); \
  } else if (n_ == 50416 && k_ == 4096) {          \
    HGEMM_INT4_M_N_K_DISPATCH(m, 50416, 4096, gz); \
  } else {                                         \
    TORCH_CHECK(false);                            \
  }

#define HGEMM_INT4_M_DISPATCH(m)            \
  if (calib_gz_ == k_ || calib_gz_ == -1) { \
    HGEMM_INT4_N_K_DISPATCH(m, 0);          \
  } else if (calib_gz_ == 128) {            \
    HGEMM_INT4_N_K_DISPATCH(m, 128);        \
  } else {                                  \
    TORCH_CHECK(false);                     \
  }

inline Tensor resize_as_mat1(const Tensor& mat1, const Tensor& output) {
  auto output_ = output.flatten(0, -2);
  int n = output_.sizes()[1];
  auto sizes = mat1.sym_sizes().vec();
  sizes[sizes.size() - 1] = n;
  return output.view_symint(sizes);
}

class HGEMMXetla_INT4 final {
 public:
  enum EpilogueType {
    BIAS = 0,
    RES_ADD,
    GELU,
    RES_MUL,
    SILU,
    SCALED_RES_ADD,
  };

 private:
  enum {
    MAX_EPILOGUES = 4,
  };
  Tensor *input_, *weight_, *output_, *weight_scl_, *weight_zp_;
  Tensor* epilogues_[MAX_EPILOGUES];
  EpilogueType epilogue_type_[MAX_EPILOGUES];
  float pf32[MAX_EPILOGUES];
  int num_epilogues_ = 0;
  bool is_a_row_major_;
  bool is_a_col_major_;
  bool is_b_row_major_;
  bool is_b_col_major_;
  bool fallback_;
  int m_, n_, k_;
  int64_t calib_gz_;

 public:
  HGEMMXetla_INT4() = default;
  bool fallback() const {
    return fallback_;
  }
  HGEMMXetla_INT4& add_matrix_out(const Tensor& output) {
    output_ = const_cast<Tensor*>(&output);
    return *this;
  }
  HGEMMXetla_INT4& add_matrix_inp(const Tensor& input) {
    input_ = const_cast<Tensor*>(&input);
    return *this;
  }
  HGEMMXetla_INT4& add_matrix_wei(const Tensor& b) {
    weight_ = const_cast<Tensor*>(&b);
    return *this;
  }
  HGEMMXetla_INT4& add_matrix_scl(const Tensor& scales) {
    weight_scl_ = const_cast<Tensor*>(&scales);
    return *this;
  }
  HGEMMXetla_INT4& add_matrix_zp(const Tensor& zero_points) {
    weight_zp_ = const_cast<Tensor*>(&zero_points);
    return *this;
  }
  HGEMMXetla_INT4& add_calib_gz(int64_t calib_gz) {
    calib_gz_ = calib_gz;
    return *this;
  }
  HGEMMXetla_INT4& add_epilogue(const Tensor& t, EpilogueType eptype) {
    epilogues_[num_epilogues_] = const_cast<Tensor*>(&t);
    epilogue_type_[num_epilogues_++] = eptype;
    return *this;
  }
  HGEMMXetla_INT4& add_epilogue(
      const Tensor& t,
      EpilogueType eptype,
      const float x) {
    epilogues_[num_epilogues_] = const_cast<Tensor*>(&t);
    pf32[num_epilogues_] = x;
    epilogue_type_[num_epilogues_++] = eptype;
    return *this;
  }

  HGEMMXetla_INT4& build() {
    fallback_ = true;
    if (input_->scalar_type() != kHalf ||
        (weight_->scalar_type() != kByte &&
         weight_->scalar_type() != kQUInt8) ||
        output_->scalar_type() != kHalf)
      return *this;
    if (!(input_->dim() == 2 && weight_->dim() == 2 && output_->dim() == 2))
      return *this;
    is_a_row_major_ = input_->is_contiguous();
    is_a_col_major_ = input_->transpose(0, 1).is_contiguous();
    is_b_row_major_ = weight_->is_contiguous();
    is_b_col_major_ = weight_->transpose(0, 1).is_contiguous();
    auto a_sizes = input_->sizes();
    auto b_sizes = weight_->sizes();
    auto c_sizes = output_->sizes();
    m_ = a_sizes[0];
    k_ = a_sizes[1];
    n_ = b_sizes[1] * 2;
    bool ck0 = b_sizes[0] == k_;
    bool ck1 = c_sizes[0] == m_ && c_sizes[1] == n_;
    bool ck2 = is_a_row_major_;
    for (int i = 0; i < num_epilogues_; i++) {
      switch (epilogue_type_[i]) {
        case BIAS: {
          bool ck = epilogues_[i]->dim() == 1 && epilogues_[i]->is_contiguous();
          ck = ck && epilogues_[i]->sizes()[0] == n_;
          ck = ck && epilogues_[i]->scalar_type() == kHalf;
          if (!ck)
            return *this;
        } break;
        case RES_MUL:
        case RES_ADD: {
          bool ck = epilogues_[i]->dim() == 2;
          ck = ck && epilogues_[i]->sizes()[0] == m_ &&
              epilogues_[i]->sizes()[1] == n_;
          ck = ck && epilogues_[i]->is_contiguous();
          ck = ck && epilogues_[i]->scalar_type() == kHalf;
          if (!ck)
            return *this;
        } break;
        case GELU:
        case SILU: {
        } break;
      }
    }
    fallback_ = false;
    return *this;
  }

  void run() {
    using scalar_t =
        decltype(c10::impl::ScalarTypeToCPPType<ScalarType::Half>::t);
    auto& q = dpcppGetCurrentQueue();
    if (m_ <= 8) {
      HGEMM_INT4_M_DISPATCH(8);
    } else if (m_ <= 16) {
      HGEMM_INT4_M_DISPATCH(16);
    } else if (m_ <= 32) {
      HGEMM_INT4_M_DISPATCH(32);
    } else if (m_ <= 64) {
      HGEMM_INT4_M_DISPATCH(64);
    } else if (m_ <= 384 && n_ == 4096 && (k_ == 4096 || k_ == 16384)) {
      HGEMM_INT4_M_DISPATCH(384);
    } else if (m_ <= 1024) {
      HGEMM_INT4_M_DISPATCH(1024);
    } else {
      HGEMM_INT4_M_DISPATCH(INF);
    }
  }
};
