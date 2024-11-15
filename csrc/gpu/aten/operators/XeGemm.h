#pragma once

#include <ATen/ATen.h>
#include <ATen/record_function.h>
#include <oneDNN/oneDNN.h>
#include <runtime/Utils.h>
#include <utils/oneMKLUtils.h>
#include "comm/ATDispatch.h"
#include "comm/RegistrationDeclarations.h"
#include "utils/ComputeEngine.h"
#include "xetla/hgemm.h"

using namespace torch_ipex::xpu::xetla;

#define RECORD_XeTLA_FUNCTION_IMPL(F, M, N, K)            \
  char str__[100];                                        \
  sprintf(str__, "xetla_%s(%d, %d, %d)", "" #F, M, N, K); \
  RECORD_FUNCTION(str__, c10::ArrayRef<c10::IValue>({}));

static inline bool hgemm_xetla_valid(const Tensor& a, const Tensor& b) {
  return (
      choose_compute_eng(torch_ipex::xpu::COMPUTE_ENG::XETLA, a, b) ==
      torch_ipex::xpu::COMPUTE_ENG::XETLA);
}

static inline bool hgemm_xetla_valid(
    const Tensor& input,
    const Tensor& weight,
    const Tensor& others) {
  bool is_row_major = (input.is_contiguous() && weight.is_contiguous());
  if (!is_row_major)
    return false;

  return (
      choose_compute_eng(
          torch_ipex::xpu::COMPUTE_ENG::XETLA, input, weight, others) ==
      torch_ipex::xpu::COMPUTE_ENG::XETLA);
}

#if 0
static inline uint32_t find_ks_coop_num_y(
    uint32_t slm_kslicing,
    uint32_t sg_m) {
  uint32_t ks_coop_num_y = sg_m;
  while (slm_kslicing % sg_m != 0) {
    ks_coop_num_y = slm_kslicing % sg_m;
    slm_kslicing = sg_m;
    sg_m = ks_coop_num_y;
  }
  return ks_coop_num_y;
}

static inline size_t get_acc_size(
    const uint32_t matrix_m,
    const uint32_t matrix_n) {
  return matrix_m * matrix_n;
};

static size_t get_cnt_size(
    const uint32_t matrix_m,
    const uint32_t matrix_n,
    const uint32_t wg_m,
    const uint32_t wg_n,
    const uint32_t sg_m,
    const uint32_t sg_n,
    const uint32_t slm_kslicing) {
  // return matrix_m * matrix_n;
  size_t group_range_m = (matrix_m + wg_m - 1) / wg_m;
  size_t group_range_n = (matrix_n + wg_n - 1) / wg_n;

  uint32_t wg_size_x = (wg_m + sg_m - 1) / sg_m;
  uint32_t wg_size_y = (wg_n + sg_n - 1) / sg_n;
  // uint32_t ks_coop_num_y = gcd<slm_kslicing, sg_m>::value;
  uint32_t ks_coop_num_y = find_ks_coop_num_y(slm_kslicing, sg_m);
  uint32_t coop_remain_num_x = slm_kslicing / ks_coop_num_y;
  bool has_redundant_wg = (coop_remain_num_x * 16) > sg_n;
  uint32_t tile_size_y = sg_m / ks_coop_num_y;
  uint32_t tile_size_x = has_redundant_wg ? 16 : sg_n / coop_remain_num_x;
  uint32_t ks_coop_num_x = sg_n / tile_size_x;

  uint32_t counter_size = 8;
  return group_range_m * group_range_n * wg_size_x * wg_size_y * ks_coop_num_y *
      ks_coop_num_x * counter_size;
}

static float* get_acc_and_cnt_tensor(
    const c10::MaybeOwned<Tensor> input_,
    const uint32_t m_,
    const uint32_t n_,
    const uint32_t k_,
    const int32_t policy_id,
    Tensor& acc_tensor_,
    Tensor& cnt_tensor_) {
  if (policy_id == -1) {
    return (float*)nullptr;
  }

  auto policy_config = hgemm_policy_traits[policy_id];
  if (policy_config.l3_ks_ != 1) {
    uint32_t wg_m = policy_config.wg_m_;
    uint32_t wg_n = policy_config.wg_n_;
    uint32_t sg_m = policy_config.sg_m_;
    uint32_t sg_n = policy_config.sg_n_;
    uint32_t slm_ks = policy_config.slm_ks_;
    size_t acc_size = get_acc_size(m_, n_);
    size_t cnt_size = get_cnt_size(m_, n_, wg_m, wg_n, sg_m, sg_n, slm_ks);

    acc_tensor_ = at::AtenIpexTypeXPU::empty(
        {acc_size}, input_->options().dtype(at::kFloat), c10::nullopt);
    cnt_tensor_ = at::AtenIpexTypeXPU::empty(
        {cnt_size}, input_->options().dtype(at::kByte), c10::nullopt);
    return acc_tensor_.data_ptr<float>();
  } else {
    return (float*)nullptr;
  }
}
#endif

enum EpilogueType {
  BIAS = 0,
  RES_ADD,
  RELU,
  GELU,
  RES_MUL,
  SILU,
};

static inline bool ptr_align64(void* data_ptr) {
  return ((reinterpret_cast<uint64_t>(data_ptr) & 0x3f) == 0);
}

static inline bool ptr_align4(void* data_ptr) {
  return ((reinterpret_cast<uint64_t>(data_ptr) & 0x3) == 0);
}

class hgemm_ctx_t {
 public:
  c10::MaybeOwned<Tensor> c_, a_, b_;
  int m_, n_, k_, bk_;
  float alpha_ = 1.0f;
  bool is_b_row_major_;
  bool is_a_row_major_;

  inline void add_alpha(const float alpha) {
    alpha_ = alpha;
  }
  inline void add_matrix_c(const Tensor& c) {
    c_ = c10::MaybeOwned<Tensor>::borrowed(c);
  }
  inline void add_matrix_a(const Tensor& a) {
    a_ = c10::MaybeOwned<Tensor>::borrowed(a);
    auto a_for_gemm = a_->flatten(0, -2);
    auto a_sizes = a_for_gemm.sizes();
    m_ = a_sizes[0];
    k_ = a_sizes[1];
    is_a_row_major_ = a_->is_contiguous();
  }
  inline void add_matrix_b(const Tensor& b) {
    b_ = c10::MaybeOwned<Tensor>::borrowed(b);
    auto b_sizes = b_->sizes();
    n_ = b_sizes[1];
    bk_ = b_sizes[0];
    is_b_row_major_ = b_->is_contiguous();
  }
  inline void add_operands(
      const Tensor& out,
      const Tensor& a,
      const Tensor& b,
      const float alpha = 1.0) {
    add_matrix_c(out);
    add_matrix_a(a);
    add_matrix_b(b);
    add_alpha(alpha);
  }

  inline bool fp16() {
    return (c_->scalar_type() == kHalf) && (a_->scalar_type() == kHalf) &&
        (b_->scalar_type() == kHalf);
  }
  inline bool bf16() {
    return (c_->scalar_type() == kBFloat16) &&
        (a_->scalar_type() == kBFloat16) && (b_->scalar_type() == kBFloat16);
  }
  inline bool lda_align8() {
    return ((k_ & 0x3) == 0);
  }
  inline bool ldb_align8() {
    return ((n_ & 0x3) == 0);
  }
  inline bool lda_align4() {
    return ((k_ & 0x1) == 0);
  }
  inline bool ldb_align4() {
    return ((n_ & 0x1) == 0);
  }
  inline bool ld_align4() {
    return (lda_align4() && ldb_align4());
  }
  inline bool ld_align8() {
    return (lda_align8() && ldb_align8());
  }
  inline bool c_base_align64() {
    return ptr_align64(c_->data_ptr());
  }
  inline bool c_base_align4() {
    return ptr_align4(c_->data_ptr());
  }
  inline bool a_base_align64() {
    return ptr_align64(a_->data_ptr());
  }
  inline bool a_base_align4() {
    return ptr_align4(a_->data_ptr());
  }
  inline bool b_base_align64() {
    return ptr_align64(b_->data_ptr());
  }
  inline bool b_base_align4() {
    return ptr_align4(b_->data_ptr());
  }
  inline bool base_align64() {
    return (c_base_align64() && a_base_align64() && b_base_align64());
  }
  inline bool base_align4() {
    return (c_base_align4() && a_base_align4() && b_base_align4());
  }
  inline bool mm_size_match() {
    auto c_for_gemm = c_->flatten(0, -2);
    auto c_sizes = c_for_gemm.sizes();
    return (
        (b_->dim() == 2) && (bk_ == k_) && (c_sizes[0] == m_) &&
        (c_sizes[1] == n_));
  }

  inline bool check_gemm_valid(gpu::xetla::gpu_arch arch_tag) {
    if (!(is_a_row_major_ && is_b_row_major_))
      return false;
    if (!(fp16() && mm_size_match()))
      return false;
    if (arch_tag == gpu::xetla::gpu_arch::XeHpc) {
      return (ld_align8() && base_align64());
    } else {
      return (ld_align4() && base_align4());
    }
  }
};

template <uint32_t MAX_EPILOGUES>
class HGEMM_XETLA {
 private:
  gpu::xetla::gpu_arch arch_tag;
  hgemm_ctx_t ctx;

  static_assert(MAX_EPILOGUES != 0);
  EpilogueType epilogue_types_[MAX_EPILOGUES];
  SmallVector<c10::MaybeOwned<Tensor>, MAX_EPILOGUES> epilogue_tensors_;
  float epilogue_params_[MAX_EPILOGUES];
  int num_epilogues_ = 0;

  int policy_id = -1;
  bool valid_ = false;

 public:
  HGEMM_XETLA() = default;
  HGEMM_XETLA(gpu::xetla::gpu_arch arch_tag_) {
    arch_tag = arch_tag_;
  };

  inline bool valid() const {
    return valid_;
  }

  HGEMM_XETLA& add_operands(
      const Tensor& out,
      const Tensor& a,
      const Tensor& b,
      const float alpha = 1.0) {
    ctx.add_operands(out, a, b, alpha);
    return *this;
  }

  HGEMM_XETLA& add_epilogue(
      const Tensor& t,
      const EpilogueType eptype,
      const float x = 1.0) {
    epilogue_tensors_.push_back(c10::MaybeOwned<Tensor>::borrowed(t));
    epilogue_params_[num_epilogues_] = x;
    epilogue_types_[num_epilogues_++] = eptype;
    return *this;
  }

  HGEMM_XETLA& build() {
    assert(num_epilogues_ == MAX_EPILOGUES);

    if (!ctx.check_gemm_valid(arch_tag))
      return *this;

#define __CHECK(X) \
  if (!(X))        \
    return *this;

#pragma unroll
    for (int i = 0; i < MAX_EPILOGUES; i++) {
      auto eptensor = epilogue_tensors_[i];
      switch (epilogue_types_[i]) {
        case BIAS: {
          __CHECK(
              eptensor->is_contiguous() && eptensor->scalar_type() == kHalf);
          __CHECK(eptensor->numel() == ctx.n_);
        } break;
        case RES_MUL:
        case RES_ADD: {
          __CHECK(
              eptensor->is_contiguous() && eptensor->scalar_type() == kHalf);
          auto eptensor_for_gemm = eptensor->flatten(0, -2);
          auto epsizes = eptensor_for_gemm.sizes();
          __CHECK(epsizes[0] == ctx.m_ && epsizes[1] == ctx.n_);
        } break;
        default:
          break;
      }
    }

#undef __CHECK

    policy_id = hgemm_find_policy_id(
        ctx.m_, ctx.n_, ctx.k_, ctx.is_b_row_major_, arch_tag);
    valid_ = (policy_id >= 0);
    return *this;
  }

  torch_ipex::xpu::xetla::GemmStatus run() {
    assert(valid_);

    auto& q = dpcppGetCurrentQueue();
#if 0
    Tensor acc_tensor_, cnt_tensor_;
    float* acc_ptr = get_acc_and_cnt_tensor(
        a_, m_, n_, k_, policy_id, acc_tensor_, cnt_tensor_);
    uint32_t* cnt_ptr = (acc_ptr == nullptr)
        ? (uint32_t*)nullptr
        : reinterpret_cast<uint32_t*>(cnt_tensor_.data_ptr());
#endif
    auto m_ = ctx.m_;
    auto n_ = ctx.n_;
    auto k_ = ctx.k_;
    auto alpha_ = ctx.alpha_;
    sycl::half* c_ptr_ = reinterpret_cast<sycl::half*>(ctx.c_->data_ptr());
    sycl::half* a_ptr_ = reinterpret_cast<sycl::half*>(ctx.a_->data_ptr());
    sycl::half* b_ptr_ = reinterpret_cast<sycl::half*>(ctx.b_->data_ptr());
    float* acc_ptr = (float*)nullptr;
    uint32_t* cnt_ptr = (uint32_t*)nullptr;
    if constexpr (MAX_EPILOGUES == 1) {
      EpilogueType epilogue1 = epilogue_types_[0];
      if (epilogue1 == RES_ADD) {
        sycl::half* epi1_ptr_ =
            reinterpret_cast<sycl::half*>(epilogue_tensors_[0]->data_ptr());
        if (alpha_ == 1.0f) {
          RECORD_XeTLA_FUNCTION_IMPL(hgemm_res, m_, n_, k_);
          auto cgfs = hgemm_res(
              policy_id,
              c_ptr_,
              a_ptr_,
              b_ptr_,
              epi1_ptr_,
              acc_ptr,
              cnt_ptr,
              m_,
              n_,
              k_,
              epilogue_params_[0],
              arch_tag);
          DPCPP_Q_SUBMIT_CGFS(q, cgfs);
          return torch_ipex::xpu::xetla::GemmStatus::kSuccess;
        } else {
          RECORD_XeTLA_FUNCTION_IMPL(hgemm_addmm, m_, n_, k_);
          auto cgfs = hgemm_addmm(
              policy_id,
              c_ptr_,
              epi1_ptr_,
              a_ptr_,
              b_ptr_,
              acc_ptr,
              cnt_ptr,
              m_,
              n_,
              k_,
              alpha_,
              epilogue_params_[0],
              arch_tag);
          DPCPP_Q_SUBMIT_CGFS(q, cgfs);
          return torch_ipex::xpu::xetla::GemmStatus::kSuccess;
        }
      } else if (epilogue1 == BIAS) {
        assert(alpha_ == 1.0f);
        sycl::half* epi1_ptr_ =
            reinterpret_cast<sycl::half*>(epilogue_tensors_[0]->data_ptr());
        RECORD_XeTLA_FUNCTION_IMPL(hgemm_bias, m_, n_, k_);
        auto cgfs = hgemm_bias(
            policy_id,
            c_ptr_,
            a_ptr_,
            b_ptr_,
            epi1_ptr_,
            acc_ptr,
            cnt_ptr,
            m_,
            n_,
            k_,
            epilogue_params_[0],
            arch_tag);
        DPCPP_Q_SUBMIT_CGFS(q, cgfs);
        return torch_ipex::xpu::xetla::GemmStatus::kSuccess;
      } else if (epilogue1 == RES_MUL) {
        assert(alpha_ == 1.0f);
        RECORD_XeTLA_FUNCTION_IMPL(hgemm_resmul, m_, n_, k_);
        sycl::half* epi1_ptr_ =
            reinterpret_cast<sycl::half*>(epilogue_tensors_[0]->data_ptr());
        auto cgfs = hgemm_resmul(
            policy_id,
            c_ptr_,
            a_ptr_,
            b_ptr_,
            epi1_ptr_,
            acc_ptr,
            cnt_ptr,
            m_,
            n_,
            k_,
            arch_tag);
        DPCPP_Q_SUBMIT_CGFS(q, cgfs);
        return torch_ipex::xpu::xetla::GemmStatus::kSuccess;
      } else if (epilogue1 == SILU) {
        assert(alpha_ == 1.0f);
        RECORD_XeTLA_FUNCTION_IMPL(hgemm_silu, m_, n_, k_);
        auto cgfs = hgemm_silu(
            policy_id,
            c_ptr_,
            a_ptr_,
            b_ptr_,
            acc_ptr,
            cnt_ptr,
            m_,
            n_,
            k_,
            arch_tag);
        DPCPP_Q_SUBMIT_CGFS(q, cgfs);
        return torch_ipex::xpu::xetla::GemmStatus::kSuccess;
      }
    } else if constexpr (MAX_EPILOGUES == 2) {
      auto epilogue1 = epilogue_types_[0];
      auto epilogue2 = epilogue_types_[1];
      sycl::half* epi1_ptr_ =
          reinterpret_cast<sycl::half*>(epilogue_tensors_[0]->data_ptr());
      if (epilogue1 == RES_ADD && epilogue2 == RES_ADD) {
        assert(alpha_ == 1.0f);
        RECORD_XeTLA_FUNCTION_IMPL(hgemm_res_res, m_, n_, k_);
        sycl::half* epi2_ptr_ =
            reinterpret_cast<sycl::half*>(epilogue_tensors_[1]->data_ptr());
        auto cgfs = hgemm_res_res(
            policy_id,
            c_ptr_,
            a_ptr_,
            b_ptr_,
            epi1_ptr_,
            epi2_ptr_,
            acc_ptr,
            cnt_ptr,
            m_,
            n_,
            k_,
            epilogue_params_[0],
            epilogue_params_[1],
            arch_tag);
        DPCPP_Q_SUBMIT_CGFS(q, cgfs);
        return torch_ipex::xpu::xetla::GemmStatus::kSuccess;
      } else if (epilogue1 == BIAS && epilogue2 == RES_ADD) {
        assert(alpha_ == 1.0f);
        RECORD_XeTLA_FUNCTION_IMPL(hgemm_bias_res, m_, n_, k_);
        sycl::half* epi2_ptr_ =
            reinterpret_cast<sycl::half*>(epilogue_tensors_[1]->data_ptr());
        auto cgfs = hgemm_bias_res(
            policy_id,
            c_ptr_,
            a_ptr_,
            b_ptr_,
            epi1_ptr_,
            epi2_ptr_,
            acc_ptr,
            cnt_ptr,
            m_,
            n_,
            k_,
            epilogue_params_[0],
            epilogue_params_[1],
            arch_tag);
        DPCPP_Q_SUBMIT_CGFS(q, cgfs);
        return torch_ipex::xpu::xetla::GemmStatus::kSuccess;
      } else if (epilogue1 == BIAS && epilogue2 == RELU) {
        assert(alpha_ == 1.0f);
        RECORD_XeTLA_FUNCTION_IMPL(hgemm_bias_relu, m_, n_, k_);
        auto cgfs = hgemm_bias_relu(
            policy_id,
            c_ptr_,
            a_ptr_,
            b_ptr_,
            epi1_ptr_,
            acc_ptr,
            cnt_ptr,
            m_,
            n_,
            k_,
            epilogue_params_[0],
            arch_tag);
        DPCPP_Q_SUBMIT_CGFS(q, cgfs);
        return torch_ipex::xpu::xetla::GemmStatus::kSuccess;
      } else if (epilogue1 == BIAS && epilogue2 == GELU) {
        assert(alpha_ == 1.0f);
        RECORD_XeTLA_FUNCTION_IMPL(hgemm_bias_gelu, m_, n_, k_);
        auto cgfs = hgemm_bias_gelu(
            policy_id,
            c_ptr_,
            a_ptr_,
            b_ptr_,
            epi1_ptr_,
            acc_ptr,
            cnt_ptr,
            m_,
            n_,
            k_,
            epilogue_params_[0],
            arch_tag);
        DPCPP_Q_SUBMIT_CGFS(q, cgfs);
        return torch_ipex::xpu::xetla::GemmStatus::kSuccess;
      }
    } else if constexpr (MAX_EPILOGUES == 3) {
      auto epilogue1 = epilogue_types_[0];
      auto epilogue2 = epilogue_types_[1];
      auto epilogue3 = epilogue_types_[2];
      sycl::half* epi1_ptr_ =
          reinterpret_cast<sycl::half*>(epilogue_tensors_[0]->data_ptr());
      sycl::half* epi2_ptr_ =
          reinterpret_cast<sycl::half*>(epilogue_tensors_[1]->data_ptr());
      sycl::half* epi3_ptr_ =
          reinterpret_cast<sycl::half*>(epilogue_tensors_[2]->data_ptr());
      if (epilogue1 == BIAS && epilogue2 == RES_ADD && epilogue3 == RES_ADD) {
        assert(alpha_ == 1.0f);
        RECORD_XeTLA_FUNCTION_IMPL(hgemm_bias_res_res, m_, n_, k_);
        auto cgfs = hgemm_bias_res_res(
            policy_id,
            c_ptr_,
            a_ptr_,
            b_ptr_,
            epi1_ptr_,
            epi2_ptr_,
            epi3_ptr_,
            acc_ptr,
            cnt_ptr,
            m_,
            n_,
            k_,
            epilogue_params_[0],
            epilogue_params_[1],
            epilogue_params_[2],
            arch_tag);
        DPCPP_Q_SUBMIT_CGFS(q, cgfs);
        return torch_ipex::xpu::xetla::GemmStatus::kSuccess;
      }
    }

    return torch_ipex::xpu::xetla::GemmStatus::kError;
  };
};

template <>
class HGEMM_XETLA<0> {
 private:
  hgemm_ctx_t ctx;

  gpu::xetla::gpu_arch arch_tag;
  int policy_id = -1;
  bool valid_ = false;

 public:
  HGEMM_XETLA() = default;
  HGEMM_XETLA(gpu::xetla::gpu_arch arch_tag_) {
    arch_tag = arch_tag_;
  };

  inline bool valid() const {
    return valid_;
  }

  HGEMM_XETLA& add_operands(
      const Tensor& out,
      const Tensor& a,
      const Tensor& b,
      const float alpha = 1.0) {
    ctx.add_operands(out, a, b, alpha);
    return *this;
  }

  HGEMM_XETLA& build() {
    if (!ctx.check_gemm_valid(arch_tag))
      return *this;
    policy_id = hgemm_find_policy_id(
        ctx.m_, ctx.n_, ctx.k_, ctx.is_b_row_major_, arch_tag);
    valid_ = (policy_id >= 0);
    return *this;
  }

  torch_ipex::xpu::xetla::GemmStatus run() {
    assert(valid_);
    auto& q = dpcppGetCurrentQueue();
    auto m_ = ctx.m_;
    auto n_ = ctx.n_;
    auto k_ = ctx.k_;
    auto alpha_ = ctx.alpha_;
    sycl::half* c_ptr_ = reinterpret_cast<sycl::half*>(ctx.c_->data_ptr());
    sycl::half* a_ptr_ = reinterpret_cast<sycl::half*>(ctx.a_->data_ptr());
    sycl::half* b_ptr_ = reinterpret_cast<sycl::half*>(ctx.b_->data_ptr());
    float* acc_ptr = (float*)nullptr;
    uint32_t* cnt_ptr = (uint32_t*)nullptr;
    assert(alpha_ == 1.0f);
    RECORD_XeTLA_FUNCTION_IMPL(hgemm_common, m_, n_, k_);
    auto cgfs = hgemm_common(
        policy_id,
        c_ptr_,
        a_ptr_,
        b_ptr_,
        acc_ptr,
        cnt_ptr,
        m_,
        n_,
        k_,
        arch_tag);
    DPCPP_Q_SUBMIT_CGFS(q, cgfs);
    return torch_ipex::xpu::xetla::GemmStatus::kSuccess;
  }
};

inline Tensor matmul_resize(const Tensor& a, const Tensor& output) {
  if (a.dim() == 2)
    return output;
  auto output_ = output.flatten(0, -2);
  int n = output_.sizes()[1];
  auto sizes = a.sym_sizes().vec();
  sizes[sizes.size() - 1] = n;
  return output.view_symint(sizes);
}

inline std::vector<std::tuple<int, int>> hgemm_split_m(
    const int m,
    const int n) {
  std::vector<std::tuple<int, int>> res;
  constexpr int slice_n = 4096;
  if (m > 4096 && n >= 4096) {
    for (int start_idx = 0; start_idx < m; start_idx += slice_n) {
      int remaining = m - start_idx;
      int len = slice_n < remaining ? slice_n : remaining;
      res.push_back(std::make_tuple(start_idx, len));
    }
  } else {
    res.push_back(std::make_tuple(0, m));
  }
  return res;
}

#undef RECORD_XeTLA_FUNCTION_IMPL
