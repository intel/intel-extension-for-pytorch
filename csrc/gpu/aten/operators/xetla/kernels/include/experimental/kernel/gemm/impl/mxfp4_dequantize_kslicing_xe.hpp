/*******************************************************************************
 * Copyright (c) 2022-2023 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *******************************************************************************/

/// @file
/// C++ API

#pragma once

#include <experimental/kernel/gemm/common.hpp>
#include <experimental/kernel/gemm/dispatch_policy.hpp>

namespace gpu::xetla::kernel {

/// @} xetla_gemm

/// @addtogroup xetla_gemm
/// @{

/// @brief Is the GEMM functor, specialized in bit4 matB kslicing dispatch
/// policy and Xe architecture.
///
/// @tparam num_global_kslicing_ Is the k dim split ratio between groups.
/// @tparam num_local_kslicing_ Is the k dim split ratio within a group.
/// @tparam gemm_t_ Is the gemm functor to compose a GEMM.
/// @tparam epilogue_t_ Is the epilogue functor to compose a GEMM.
template <typename gemm_t_, typename epilogue_t_, typename group_swizzle_>
class mxfp4_gemm_universal_t {
  using gemm_t = gemm_t_;
  using epilogue_t = epilogue_t_;
  using gemm_args_t = typename gemm_t::arguments_t;
  using epilogue_args_t = typename epilogue_t::arguments_t;
  using tile_shape = typename gemm_t::tile_shape;
  using group_swizzle_t = group_swizzle_;
  static constexpr uint32_t wg_tile_m = tile_shape::wg_tile_size_y;
  static constexpr uint32_t wg_tile_n = tile_shape::wg_tile_size_x;
  static constexpr uint32_t sg_tile_m = tile_shape::sg_tile_size_y;
  static constexpr uint32_t sg_tile_n = tile_shape::sg_tile_size_x;
  static constexpr uint32_t wg_size_y = tile_shape::wg_size_y;
  static constexpr uint32_t wg_size_x = tile_shape::wg_size_x;
  static constexpr uint32_t real_wg_tile_m = sg_tile_m * wg_size_y;
  static constexpr uint32_t real_wg_tile_n = sg_tile_n * wg_size_x;

  static constexpr uint32_t k_stride = gemm_t::k_stride;
  static constexpr uint32_t dequant_s = gemm_t::dequant_s;
  static constexpr uint32_t pack_ratio = gemm_t::pack_ratio;
  using work_group_t = typename gemm_t::work_group_t;
  static constexpr uint32_t work_group_size = work_group_t::size;

  static constexpr gpu_arch arch_tag = group_swizzle_t::arch_tag;
  static_assert(arch_tag == gemm_t::arch_tag, "arch_tag should be the same");
  static_assert(
      arch_tag == epilogue_t::arch_tag,
      "arch_tag should be the same");
  static_assert(
      std::is_same<
          typename gemm_t::tile_shape,
          typename epilogue_t::tile_shape>::value,
      "tile_shape should be the same");

  using mem_desc_a_t = typename gemm_t::mem_desc_a_t;
  using mem_desc_b_t = typename gemm_t::mem_desc_b_t;
  using mem_desc_scale_t = typename gemm_t::mem_desc_scale_t;
  using mem_desc_c_t = typename epilogue_t::mem_desc_c_t;
  using matA_base_t = typename mem_desc_a_t::base_t;
  using matB_base_t = typename mem_desc_b_t::base_t;
  using matC_base_t = typename mem_desc_c_t::base_t;
  using scale_base_t = typename mem_desc_scale_t::base_t;

  using dtype_a = typename mem_desc_a_t::dtype;
  using dtype_b = typename mem_desc_b_t::dtype;
  using dtype_c = typename mem_desc_c_t::dtype;
  using dtype_scale = typename mem_desc_scale_t::dtype;
  using matAcc_t = typename gemm_t::matAcc_t;
  using dtype_acc = typename matAcc_t::dtype;

  using mem_desc_acc_t =
      mem_desc_t<dtype_acc, mem_layout::row_major, mem_space::global>;
  using mem_desc_cnt_t =
      mem_desc_t<uint32_t, mem_layout::row_major, mem_space::global>;

  static constexpr uint32_t gemm_nbarr_count = gemm_t::barrier_count;
  static constexpr uint32_t gemm_slm_size = gemm_t::slm_size;

  static constexpr uint32_t epilogue_nbarr_count = epilogue_t::barrier_count;
  static constexpr uint32_t epilogue_slm_size = epilogue_t::slm_size;

  static constexpr uint32_t counter_size = 8;

 public:
  /// @brief GEMM arguments.
  /// This is the interface for users to pass the application-related runtime
  /// variables.
  struct arguments_t {
    /// @brief Is the size of the m dimension of the matrix multiplication (m x
    /// k x n).
    uint32_t matrix_m;
    /// @brief Is the size of the k dimension of the matrix multiplication (m x
    /// k x n).
    uint32_t matrix_k;
    /// @brief Is the size of the n dimension of the matrix multiplication (m x
    /// k x n).
    uint32_t matrix_n;
    /// @brief Is the base address of matrix A.
    matA_base_t matA_base;
    /// @brief Is the leading dimension (pitch) size of the matrix A in memory.
    uint32_t matA_ld;
    /// @brief Is the base address of matrix B.
    matB_base_t matB_base;
    /// @brief Is the leading dimension (pitch) size of the matrix B in memory.
    uint32_t matB_ld;
    /// @brief Is the base address of matrix C.
    matC_base_t matC_base;
    /// @brief Is the leading dimension (pitch) size of the matrix C in memory.
    uint32_t matC_ld;

    scale_base_t scale_base;
    uint32_t scale_ld;
    /// @brief Is the epilogue arguments.
    epilogue_args_t epilogue_args;

    /// @brief Constructs arguments with default method.
    inline arguments_t() = default;

    /// @brief Set for device copyable
    static constexpr bool host_callable = true;

    // Be aware of the risks: Rule of three (copy constructor, copy assignment,
    // destructor) Please check if you need to add self-define destructor
    // ~arguments_t(){}

    /// @brief Constructs arguments with initialization list.
    /// @param matrix_m_ Is the size of the m dimension of the matrix
    /// multiplication (m x k x n).
    /// @param matrix_k_ Is the size of the k dimension of the matrix
    /// multiplication (m x k x n).
    /// @param matrix_n_ Is the size of the n dimension of the matrix
    /// multiplication (m x k x n).
    /// @param matA_base_ Is the base address of matrix A.
    /// @param matA_ld_ Is the leading dimension (pitch) size of the matrix A in
    /// memory.
    /// @param matB_base_ Is the base address of matrix B.
    /// @param matB_ld_ Is the leading dimension (pitch) size of the matrix B in
    /// memory.
    /// @param matC_base_ Is the base address of matrix C.
    /// @param matC_ld_ Is the leading dimension (pitch) size of the matrix C in
    /// memory.
    /// @param epilogue_args_ Is the epilogue arguments.
    inline arguments_t(
        uint32_t matrix_m_,
        uint32_t matrix_k_,
        uint32_t matrix_n_,
        matA_base_t matA_base_,
        uint32_t matA_ld_,
        matB_base_t matB_base_,
        uint32_t matB_ld_,
        matC_base_t matC_base_,
        uint32_t matC_ld_,
        scale_base_t scale_base_,
        uint32_t scale_ld_,
        epilogue_args_t epilogue_args_ = {})
        : matrix_m(matrix_m_),
          matrix_k(matrix_k_),
          matrix_n(matrix_n_),
          matA_base(matA_base_),
          matA_ld(matA_ld_),
          matB_base(matB_base_),
          matB_ld(matB_ld_),
          matC_base(matC_base_),
          matC_ld(matC_ld_),
          scale_base(scale_base_),
          scale_ld(scale_ld_),
          epilogue_args(epilogue_args_) {}
    inline arguments_t(const arguments_t& args)
        : matrix_m(args.matrix_m),
          matrix_k(args.matrix_k),
          matrix_n(args.matrix_n),
          matA_base(args.matA_base),
          matA_ld(args.matA_ld),
          matB_base(args.matB_base),
          matB_ld(args.matB_ld),
          matC_base(args.matC_base),
          matC_ld(args.matC_ld),
          scale_base(args.scale_base),
          scale_ld(args.scale_ld),
          epilogue_args(args.epilogue_args) {}
    // Be aware of the risks: Rule of three (copy constructor, copy assignment,
    // destructor) Please check if you need to add self-define destructor inline
    // ~arguments_t(){}
    inline arguments_t& operator=(const arguments_t& args) {
      this->matrix_m = args.matrix_m;
      this->matrix_k = args.matrix_k;
      this->matrix_n = args.matrix_n;
      this->matA_base = args.matA_base;
      this->matA_ld = args.matA_ld;
      this->matB_base = args.matB_base;
      this->matB_ld = args.matB_ld;
      this->matC_base = args.matC_base;
      this->matC_ld = args.matC_ld;
      this->scale_base = args.scale_base;
      this->scale_ld = args.scale_ld;
      this->epilogue_args = args.epilogue_args;
      return *this;
    }
  };

  /// @brief Gets named_barrier id consumption count.
  /// Users query and get a named_barrier id consumption count in compile time.
  /// @return The count of named barriers required.
  __XETLA_API static constexpr uint32_t get_barrier_count() {
    constexpr uint32_t count = gemm_nbarr_count + epilogue_nbarr_count;
    static_assert(
        count <= 32, "The named_barrier count should be less than 32!");
    return count;
  }

  /// @brief Gets local memory size consumption.
  /// Users query and get a local memory consumption size in compile time.
  /// @return The size of local memory required.
  __XETLA_API static constexpr uint32_t get_slm_size() {
    constexpr uint32_t size = gemm_slm_size + epilogue_slm_size;
    static_assert(
        size <= (128 * 1024),
        "The local memory size should be less than 128KB!");
    return size;
  }

#if (XETLA_CODE_BASE == __ESIMD__)

  /// @brief Host helper function to get the expected local range under the
  /// current GEMM config.
  /// @return Expected local range.
  static sycl::range<3> get_local_range() {
    uint32_t local_range_m = (wg_tile_m + sg_tile_m - 1) / sg_tile_m;
    uint32_t local_range_n = (wg_tile_n + sg_tile_n - 1) / sg_tile_n;
    assert(local_range_m * local_range_n <= 32);
    return sycl::range<3>{1, local_range_m, local_range_n};
  };

  /// @brief Host helper function to get the expected group range under the
  /// current GEMM config.
  /// @param matrix_m Is the size of the m dimension of the matrix
  /// multiplication (m x k x n).
  /// @param matrix_n Is the size of the n dimension of the matrix
  /// multiplication (m x k x n).
  /// @return Expected group range.
  static sycl::range<3> get_group_range(uint32_t matrix_m, uint32_t matrix_n) {
    uint32_t group_range_m = (matrix_m + wg_tile_m - 1) / wg_tile_m;
    uint32_t group_range_n = (matrix_n + wg_tile_n - 1) / wg_tile_n;
    group_swizzle_t::update_group_range(group_range_m, group_range_n);
    return sycl::range<3>{1, group_range_m, group_range_n};
  };

  /// @brief Host helper function to get the expected nd_range under the current
  /// GEMM config.
  /// @param args Is the GEMM arguments for application-related runtime
  /// variables.
  /// @return Expected nd_range.
  static sycl::nd_range<3> get_nd_range(arguments_t& args) {
    sycl::range<3> local_range = get_local_range();
    sycl::range<3> group_range = get_group_range(args.matrix_m, args.matrix_n);
    return sycl::nd_range<3>{group_range * local_range, local_range};
  };

  /// @brief Check if the arguments can be implemented.
  /// @param args Is the GEMM arguments for application-related runtime
  /// variables.
  /// @return Check result.
  static bool can_implement(arguments_t& args) {
    bool implementable = true;
    if (gemm_t::msg_type_a != msg_type::unaligned_2d) {
      if (gemm_t::msg_type_a == msg_type::block_2d) {
        implementable &=
            kernel::block_2d<gpu_arch::XeHpc, dtype_a>::check_tensor(
                (uint64_t)(args.matA_base.base),
                gemm_t::is_col_major_a ? args.matrix_m : args.matrix_k,
                gemm_t::is_col_major_a ? args.matrix_k : args.matrix_m,
                args.matA_ld);
      } else {
        implementable &=
            kernel::general_1d<gpu_arch::XeHpc, dtype_a>::check_alignment(
                args.matA_base.base, args.matA_ld);
      }
    }
    if (gemm_t::msg_type_b != msg_type::unaligned_2d) {
      if (gemm_t::msg_type_b == msg_type::block_2d) {
        implementable &=
            kernel::block_2d<gpu_arch::XeHpc, dtype_b>::check_tensor(
                (uint64_t)(args.matB_base.base),
                args.matB_ld / pack_ratio,
                gemm_t::is_col_major_b ? args.matrix_n : args.matrix_k,
                args.matB_ld / pack_ratio);
      } else {
        implementable &=
            kernel::general_1d<gpu_arch::XeHpc, dtype_b>::check_alignment(
                args.matB_base.base, args.matB_ld / pack_ratio);
      }
    }
    if (epilogue_t::msg_type_c != msg_type::unaligned_2d) {
      if (epilogue_t::msg_type_c == msg_type::block_2d) {
        implementable &=
            kernel::block_2d<gpu_arch::XeHpc, dtype_c>::check_tensor(
                (uint64_t)(args.matC_base.base),
                args.matrix_n,
                args.matrix_m,
                args.matC_ld);
      } else {
        implementable &=
            kernel::general_1d<gpu_arch::XeHpc, dtype_c>::check_alignment(
                args.matC_base.base, args.matC_ld);
      }
    }
    // check for mxfp4 packing, following packing conversion along k dimension
    implementable &= ((args.matrix_k % pack_ratio == 0));
    return implementable;
  }

#endif

  /// @brief Main execution function for GEMM.
  /// @param Is the sycl::nd_item, returns execution related information, such
  /// as workgroup id, subgroup id...
  /// @param args Is the GEMM arguments for application-related runtime
  /// variables.
  /// @param slm_base Is the slm base address.
  /// @param nbarrier_base Is the named barrier base.
  __XETLA_API KERNEL_FUNC void operator()(
      sycl::nd_item<3>& item,
      const arguments_t& args,
      uint32_t slm_base = 0,
      uint32_t nbarrier_base = 0) {
    // set up workgroup level coordinates and boundaries
    work_group_t g(item.get_local_linear_id() % work_group_size);
    uint32_t wg_id = item.get_local_linear_id() / work_group_size;
    group_swizzle_t group_swizzle;
    int start_m = group_swizzle.template get_tile_idx<1>(item) * wg_tile_m;
    int start_n = group_swizzle.template get_tile_idx<2>(item) * wg_tile_n;
    int start_k = 0;
    uint32_t wg_tile_k = args.matrix_k;

    uint32_t boundary_m = (start_m + wg_tile_m) > args.matrix_m
        ? args.matrix_m
        : (start_m + wg_tile_m);
    uint32_t boundary_n = (start_n + wg_tile_n) > args.matrix_n
        ? args.matrix_n
        : (start_n + wg_tile_n);
    uint32_t boundary_k = wg_tile_k;

    mem_desc_a_t mem_desc_a;
    mem_desc_b_t mem_desc_b;
    mem_desc_c_t mem_desc_c;
    mem_desc_scale_t mem_desc_scale;

    mem_desc_a.init(
        args.matA_base,
        {boundary_k, boundary_m, args.matA_ld},
        {start_k, start_m});
    mem_desc_b.init(
        args.matB_base,
        {boundary_n, boundary_k / pack_ratio, args.matB_ld},
        {start_n, start_k / static_cast<int>(pack_ratio)});
    mem_desc_c.init(
        args.matC_base,
        {boundary_n, boundary_m, args.matC_ld},
        {start_n, start_m});

    uint32_t scale_size_y = args.matrix_k / dequant_s;
    int start_y_scale = start_k / dequant_s;
    mem_desc_scale.init(
        args.scale_base,
        {args.matrix_n, scale_size_y, args.scale_ld},
        {start_n, start_y_scale});

    uint32_t gemm_slm_base = slm_base;
    uint32_t gemm_nbarr_base = nbarrier_base;
    uint32_t inner_loop_count = (wg_tile_k + k_stride - 1) / k_stride;
    gemm_args_t gemm_args(
        mem_desc_a, mem_desc_b, inner_loop_count, mem_desc_scale);
    matAcc_t matAcc;
    matAcc.init(0);
    gemm_t gemm;
    gemm(g, matAcc, gemm_args, gemm_slm_base, gemm_nbarr_base);

    epilogue_t epilogue;
    epilogue(g, matAcc, mem_desc_c, args.epilogue_args);
    return;
  };
};

template <typename T, int N, int Start>
inline typename std::enable_if_t<(N == Start), xetla_vector<T, N>>
inclusive_prefix_sum(xetla_vector<T, N> src) {
  return src;
}
template <typename T, int N, int Start>
inline typename std::enable_if_t<(Start != N), xetla_vector<T, N>>
inclusive_prefix_sum(xetla_vector<T, N> src) {
  // assert N is a power of 2
  static_assert((N & (N - 1)) == 0, "N is expected to be power of 2");
  xetla_vector<T, N> dst = src;
  dst.xetla_select<N - Start, 1>(Start) += src.xetla_select<N - Start, 1>(0);
  return inclusive_prefix_sum<T, N, Start * 2>(dst);
}

template <typename gemm_t_, typename epilogue_t_>
class persistent_mxfp4_group_gemm_universal_t {
  using gemm_t = gemm_t_;
  using epilogue_t = epilogue_t_;
  using gemm_args_t = typename gemm_t::arguments_t;
  using epilogue_args_t = typename epilogue_t::arguments_t;
  using tile_shape = typename gemm_t::tile_shape;
  static constexpr uint32_t wg_tile_m = tile_shape::wg_tile_size_y;
  static constexpr uint32_t wg_tile_n = tile_shape::wg_tile_size_x;
  static constexpr uint32_t sg_tile_m = tile_shape::sg_tile_size_y;
  static constexpr uint32_t sg_tile_n = tile_shape::sg_tile_size_x;
  static constexpr uint32_t wg_size_y = tile_shape::wg_size_y;
  static constexpr uint32_t wg_size_x = tile_shape::wg_size_x;
  static constexpr uint32_t real_wg_tile_m = sg_tile_m * wg_size_y;
  static constexpr uint32_t real_wg_tile_n = sg_tile_n * wg_size_x;

  static constexpr uint32_t k_stride = gemm_t::k_stride;
  static constexpr uint32_t dequant_s = gemm_t::dequant_s;
  static constexpr uint32_t pack_ratio = gemm_t::pack_ratio;
  using work_group_t = typename gemm_t::work_group_t;
  static constexpr uint32_t work_group_size = work_group_t::size;

  static_assert(
      std::is_same<
          typename gemm_t::tile_shape,
          typename epilogue_t::tile_shape>::value,
      "tile_shape should be the same");

  using mem_desc_a_t = typename gemm_t::mem_desc_a_t;
  using mem_desc_b_t = typename gemm_t::mem_desc_b_t;
  using mem_desc_scale_t = typename gemm_t::mem_desc_scale_t;
  using mem_desc_c_t = typename epilogue_t::mem_desc_c_t;
  using matA_base_t = typename mem_desc_a_t::base_t;
  using matB_base_t = typename mem_desc_b_t::base_t;
  using matC_base_t = typename mem_desc_c_t::base_t;
  using scale_base_t = typename mem_desc_scale_t::base_t;

  using dtype_a = typename mem_desc_a_t::dtype;
  using dtype_b = typename mem_desc_b_t::dtype;
  using dtype_c = typename mem_desc_c_t::dtype;
  using dtype_scale = typename mem_desc_scale_t::dtype;
  using matAcc_t = typename gemm_t::matAcc_t;
  using dtype_acc = typename matAcc_t::dtype;

  using mem_desc_acc_t =
      mem_desc_t<dtype_acc, mem_layout::row_major, mem_space::global>;
  using mem_desc_cnt_t =
      mem_desc_t<uint32_t, mem_layout::row_major, mem_space::global>;
  using acc_base_t = typename mem_desc_acc_t::base_t;

  static constexpr uint32_t gemm_nbarr_count = gemm_t::barrier_count;
  static constexpr uint32_t gemm_slm_size = gemm_t::slm_size;

  static constexpr uint32_t epilogue_nbarr_count = epilogue_t::barrier_count;
  static constexpr uint32_t epilogue_slm_size = epilogue_t::slm_size;

  static constexpr uint32_t counter_size = 8;
  static constexpr int load_expert_num = 8;
  static constexpr int PERSISTENT_SG_NUMS = 20 * 32;
  static constexpr int atomic_slm_size = 1 * sizeof(int);

 public:
  /// @brief GEMM arguments.
  /// This is the interface for users to pass the application-related runtime
  /// variables.
  struct arguments_t {
    /// @brief Is the size of the m dimension of each expert of the matrix
    const int* total_rows_for_each_expert;
    /// @brief Is the size of the number of experts
    int expert_num;
    /// @brief Is the size of the k dimension of the matrix multiplication (m x
    /// k x n).
    uint32_t matrix_k;
    /// @brief Is the size of the n dimension of the matrix multiplication (m x
    /// k x n).
    uint32_t matrix_n;
    uint32_t matrix_n_pad;
    /// @brief Is the base address of matrix A.
    matA_base_t matA_base;
    /// @brief Is the leading dimension (pitch) size of the matrix A in memory.
    uint32_t matA_ld;
    /// @brief Is the base address of matrix B.
    matB_base_t matB_base;
    /// @brief Is the leading dimension (pitch) size of the matrix B in memory.
    uint32_t matB_ld;
    /// @brief Is the base address of matrix C.
    matC_base_t matC_base;
    /// @brief Is the leading dimension (pitch) size of the matrix C in memory.
    uint32_t matC_ld;

    scale_base_t scale_base;
    uint32_t scale_ld;

    dtype_a* bias;

    int* atomic_buffer;

    /// @brief Is the epilogue arguments.
    epilogue_args_t epilogue_args;

    /// @brief Constructs arguments with default method.
    inline arguments_t() = default;

    /// @brief Set for device copyable
    static constexpr bool host_callable = true;

    // Be aware of the risks: Rule of three (copy constructor, copy assignment,
    // destructor) Please check if you need to add self-define destructor
    // ~arguments_t(){}

    /// @brief Constructs arguments with initialization list.
    /// @param matrix_m_ Is the size of the m dimension of the matrix
    /// multiplication (m x k x n).
    /// @param matrix_k_ Is the size of the k dimension of the matrix
    /// multiplication (m x k x n).
    /// @param matrix_n_ Is the size of the n dimension of the matrix
    /// multiplication (m x k x n).
    /// @param matA_base_ Is the base address of matrix A.
    /// @param matA_ld_ Is the leading dimension (pitch) size of the matrix A in
    /// memory.
    /// @param matB_base_ Is the base address of matrix B.
    /// @param matB_ld_ Is the leading dimension (pitch) size of the matrix B in
    /// memory.
    /// @param matC_base_ Is the base address of matrix C.
    /// @param matC_ld_ Is the leading dimension (pitch) size of the matrix C in
    /// memory.
    /// @param epilogue_args_ Is the epilogue arguments.
    inline arguments_t(
        const int* total_rows_for_each_expert_,
        int expert_num_,
        uint32_t matrix_k_,
        uint32_t matrix_n_,
        matA_base_t matA_base_,
        uint32_t matA_ld_,
        matB_base_t matB_base_,
        uint32_t matB_ld_,
        matC_base_t matC_base_,
        uint32_t matC_ld_,
        scale_base_t scale_base_,
        uint32_t scale_ld_,
        dtype_a* bias_,
        int* atomic_buffer_,
        epilogue_args_t epilogue_args_ = {})
        : total_rows_for_each_expert(total_rows_for_each_expert_),
          expert_num(expert_num_),
          matrix_k(matrix_k_),
          matrix_n(matrix_n_),
          matrix_n_pad((matrix_n_ + wg_tile_n - 1) / wg_tile_n * wg_tile_n),
          matA_base(matA_base_),
          matA_ld(matA_ld_),
          matB_base(matB_base_),
          matB_ld(matB_ld_),
          matC_base(matC_base_),
          matC_ld(matC_ld_),
          scale_base(scale_base_),
          scale_ld(scale_ld_),
          bias(bias_),
          atomic_buffer(atomic_buffer_),
          epilogue_args(epilogue_args_) {}
    inline arguments_t(const arguments_t& args)
        : total_rows_for_each_expert(args.total_rows_for_each_expert),
          expert_num(args.expert_num),
          matrix_k(args.matrix_k),
          matrix_n(args.matrix_n),
          matrix_n_pad(args.matrix_n_pad),
          matA_base(args.matA_base),
          matA_ld(args.matA_ld),
          matB_base(args.matB_base),
          matB_ld(args.matB_ld),
          matC_base(args.matC_base),
          matC_ld(args.matC_ld),
          scale_base(args.scale_base),
          scale_ld(args.scale_ld),
          bias(args.bias),
          atomic_buffer(args.atomic_buffer),
          epilogue_args(args.epilogue_args) {}
    // Be aware of the risks: Rule of three (copy constructor, copy assignment,
    // destructor) Please check if you need to add self-define destructor inline
    // ~arguments_t(){}
    inline arguments_t& operator=(const arguments_t& args) {
      this->total_rows_for_each_expert = args.total_rows_for_each_expert;
      this->expert_num = args.expert_num;
      this->matrix_k = args.matrix_k;
      this->matrix_n = args.matrix_n;
      this->matrix_n_pad = args.matrix_n_pad;
      this->matA_base = args.matA_base;
      this->matA_ld = args.matA_ld;
      this->matB_base = args.matB_base;
      this->matB_ld = args.matB_ld;
      this->matC_base = args.matC_base;
      this->matC_ld = args.matC_ld;
      this->scale_base = args.scale_base;
      this->scale_ld = args.scale_ld;
      this->bias = args.bias;
      this->atomic_buffer = args.atomic_buffer;
      this->epilogue_args = args.epilogue_args;
      return *this;
    }
  };

  /// @brief Gets named_barrier id consumption count.
  /// Users query and get a named_barrier id consumption count in compile time.
  /// @return The count of named barriers required.
  __XETLA_API static constexpr uint32_t get_barrier_count() {
    constexpr uint32_t count = gemm_nbarr_count + epilogue_nbarr_count;
    static_assert(
        count <= 32, "The named_barrier count should be less than 32!");
    return count;
  }

  /// @brief Gets local memory size consumption.
  /// Users query and get a local memory consumption size in compile time.
  /// @return The size of local memory required.
  __XETLA_API static constexpr uint32_t get_slm_size() {
    constexpr uint32_t size =
        gemm_slm_size + epilogue_slm_size + atomic_slm_size;
    static_assert(
        size <= (128 * 1024),
        "The local memory size should be less than 128KB!");
    return size;
  }

#if (XETLA_CODE_BASE == __ESIMD__)

  /// @brief Host helper function to get the expected local range under the
  /// current GEMM config.
  /// @return Expected local range.
  static sycl::range<3> get_local_range() {
    uint32_t local_range_m = (wg_tile_m + sg_tile_m - 1) / sg_tile_m;
    uint32_t local_range_n = (wg_tile_n + sg_tile_n - 1) / sg_tile_n;
    assert(local_range_m * local_range_n <= 32);
    return sycl::range<3>{1, local_range_m, local_range_n};
  };

  /// @brief Host helper function to get the expected group range under the
  /// current GEMM config.
  /// @param total_rows_for_each_expert_h Is the size of the m dimension of each
  /// expert of the matrix
  /// @param expert_num Is the size of the number of experts
  /// @param matrix_n Is the size of the n dimension of the matrix
  /// multiplication (m x k x n).
  /// @return Expected group range.
  static sycl::range<3> get_group_range(sycl::range<3>& local_range) {
    uint32_t local_size = local_range[0] * local_range[1] * local_range[2];
    uint32_t group_size = 1;
    if (PERSISTENT_SG_NUMS > local_size) {
      group_size = PERSISTENT_SG_NUMS / local_size;
    }
    assert(group_size * local_size == PERSISTENT_SG_NUMS);
    return sycl::range<3>{group_size, 1, 1};
  };

  /// @brief Host helper function to get the expected nd_range under the current
  /// GEMM config.
  /// @param args Is the GEMM arguments for application-related runtime
  /// variables.
  /// @return Expected nd_range.
  static sycl::nd_range<3> get_nd_range(arguments_t& args) {
    sycl::range<3> local_range = get_local_range();
    sycl::range<3> group_range = get_group_range(local_range);
    return sycl::nd_range<3>{group_range * local_range, local_range};
  };

#endif

  /// @brief Main execution function for GEMM.
  /// The processing order is 1) set group-level base and boundary, split group
  /// to workgroups -> 2) num_local_kslicing x gemms -> 3) local kslicing -> 4)
  /// num_local_kslicing x epilogues.
  /// @param Is the sycl::nd_item, returns execution related information, such
  /// as workgroup id, subgroup id...
  /// @param args Is the GEMM arguments for application-related runtime
  /// variables.
  /// @param slm_base Is the slm base address.
  /// @param nbarrier_base Is the named barrier base.
  __XETLA_API KERNEL_FUNC void operator()(
      sycl::nd_item<3>& item,
      const arguments_t& args,
      uint32_t slm_base = 0,
      uint32_t nbarrier_base = 0) {
    // set up workgroup level coordinates and boundaries
    work_group_t g(item.get_local_linear_id() % work_group_size);
    int start_k = 0;
    uint32_t wg_tile_k = args.matrix_k;

    int group_id = item.get_group(0);
    int group_range = item.get_group_range(0);
    int local_id = item.get_local_linear_id();

    if (group_id == 0 && local_id == 0) {
      __ESIMD_NS::simd<int, 1> value = 0;
      __ESIMD_NS::simd<uint32_t, 1> offset = 0;
      __ESIMD_NS::atomic_update<__ESIMD_NS::atomic_op::store>(
          args.atomic_buffer, offset, value);
    }

    int atomic_slm_offset = gemm_slm_size + epilogue_slm_size;

    int start_n = (group_id * wg_tile_n) % args.matrix_n_pad;
    int group_m_id = (group_id * wg_tile_n) / args.matrix_n_pad;

    int expert_id = 0;
    int expert_m_id = group_m_id;
    int skip_m = 0;

    int pre_rows = 0;
    int pre_tiles = 0;
    int gemm_m = 0;
    for (int i = 0; i < args.expert_num; i += load_expert_num) {
      xetla_vector<int, load_expert_num> rows_for_experts =
          xetla_load_global<int, load_expert_num>(
              (int*)args.total_rows_for_each_expert, i * sizeof(int));

      xetla_vector<int, load_expert_num> cumsum_rows_for_experts =
          inclusive_prefix_sum<int, load_expert_num, 1>(rows_for_experts);

      xetla_vector<int, load_expert_num> cumsum_tiles_for_experts =
          inclusive_prefix_sum<int, load_expert_num, 1>(
              (rows_for_experts + wg_tile_m - 1) / wg_tile_m);

      cumsum_rows_for_experts += pre_rows;
      cumsum_tiles_for_experts += pre_tiles;

      if (group_m_id >= cumsum_tiles_for_experts[load_expert_num - 1]) {
        pre_rows = cumsum_rows_for_experts[load_expert_num - 1];
        pre_tiles = cumsum_tiles_for_experts[load_expert_num - 1];
        continue;
      }

      while (group_m_id < cumsum_tiles_for_experts[load_expert_num - 1]) {
        xetla_vector<uint32_t, load_expert_num> mask =
            group_m_id >= cumsum_tiles_for_experts;

        uint32_t load_start = sycl::ext::intel::esimd::cbit(
            sycl::ext::intel::esimd::ballot(mask));

        uint32_t expert_start = load_start + i;

        if (load_start == 0) {
          expert_m_id = group_m_id - pre_tiles;
          skip_m = pre_rows;
        } else {
          expert_m_id = group_m_id - cumsum_tiles_for_experts[load_start - 1];
          skip_m = cumsum_rows_for_experts[load_start - 1];
        }
        expert_id = expert_start;
        gemm_m = rows_for_experts[load_start];

        int start_m = skip_m + expert_m_id * wg_tile_m;
        uint32_t boundary_m = (expert_m_id * wg_tile_m + wg_tile_m) > gemm_m
            ? skip_m + gemm_m
            : start_m + wg_tile_m;
        uint32_t boundary_n = (start_n + wg_tile_n) > args.matrix_n
            ? args.matrix_n
            : (start_n + wg_tile_n);
        uint32_t boundary_k = wg_tile_k;

        int64_t offset = static_cast<int64_t>(expert_id) *
            static_cast<int64_t>(args.matrix_n) *
            static_cast<int64_t>(args.matrix_k);
        dtype_b* current_matB_base = args.matB_base.base + offset / pack_ratio;
        dtype_scale* current_scale_base =
            args.scale_base.base + offset / dequant_s;

        mem_desc_a_t mem_desc_a;
        mem_desc_b_t mem_desc_b;
        mem_desc_c_t mem_desc_c;
        mem_desc_scale_t mem_desc_scale;

        mem_desc_a.init(
            args.matA_base,
            {boundary_k, boundary_m, args.matA_ld},
            {start_k, start_m});
        mem_desc_b.init(
            current_matB_base,
            {boundary_n, boundary_k / pack_ratio, args.matB_ld},
            {start_n, start_k / static_cast<int>(pack_ratio)});
        mem_desc_c.init(
            args.matC_base,
            {boundary_n, boundary_m, args.matC_ld},
            {start_n, start_m});

        uint32_t scale_size_y = args.matrix_k / dequant_s;
        int start_y_scale = start_k / dequant_s;
        mem_desc_scale.init(
            current_scale_base,
            {args.matrix_n, scale_size_y, args.scale_ld},
            {start_n, start_y_scale});

        uint32_t gemm_slm_base = slm_base;
        uint32_t gemm_nbarr_base = nbarrier_base;
        uint32_t inner_loop_count = (wg_tile_k + k_stride - 1) / k_stride;
        gemm_args_t gemm_args(
            mem_desc_a, mem_desc_b, inner_loop_count, mem_desc_scale);
        matAcc_t matAcc;
        matAcc.init(0);
        gemm_t gemm;
        gemm(g, matAcc, gemm_args, gemm_slm_base, gemm_nbarr_base);

        if (args.bias != nullptr) {
          // add bias
          dtype_a* current_bias_base =
              args.bias + expert_id * args.matrix_n + start_n;
          static constexpr uint32_t tile_size_x = matAcc_t::tile_size_x;
          static constexpr uint32_t tile_size_y = matAcc_t::tile_size_y;
          static constexpr uint32_t block_size_x = matAcc_t::block_size_x;
          static constexpr uint32_t block_size_y = matAcc_t::block_size_y;
          static constexpr int32_t num_block_x = matAcc_t::num_block_x;
          static constexpr uint32_t block_elems = matAcc_t::block_elems;

          xetla_vector<dtype_a, tile_size_x> bias_vec =
              xetla_load_global<dtype_a, tile_size_x>(
                  current_bias_base,
                  item.get_local_id(2) * sg_tile_n * sizeof(dtype_a));

#pragma unroll
          for (uint32_t i = 0; i < tile_size_y / block_size_y; i++) {
#pragma unroll
            for (uint32_t j = 0; j < num_block_x; j++) {
              auto dst_reg =
                  matAcc.reg
                      .xetla_select<block_elems, 1>(
                          (i * num_block_x + j) * block_elems)
                      .xetla_format<dtype_acc, block_size_y, block_size_x>();
              auto src_reg =
                  bias_vec.xetla_select<block_size_x, 1>(j * block_size_x);
#pragma unroll
              for (uint32_t row_i = 0; row_i < block_size_y; row_i++) {
                dst_reg.row(row_i) =
                    xetla_cvt<dtype_acc, dtype_a, block_size_x>(src_reg) +
                    dst_reg.row(row_i);
              }
            }
          }
          // process the tail
          if constexpr ((tile_size_y % block_size_y) != 0) {
            constexpr uint32_t tail_start_y =
                tile_size_y / block_size_y * block_size_y;
            constexpr int32_t tail_size_y = tile_size_y % block_size_y;
            constexpr int32_t tail_block_elems = tail_size_y * block_size_x;
#pragma unroll
            for (uint32_t j = 0; j < num_block_x; j++) {
              auto dst_reg =
                  matAcc.reg
                      .xetla_select<tail_block_elems, 1>(
                          tail_start_y * tile_size_x + j * tail_block_elems)
                      .xetla_format<dtype_acc, tail_size_y, block_size_x>();
              auto src_reg =
                  bias_vec.xetla_select<block_size_x, 1>(j * block_size_x);
#pragma unroll
              for (uint32_t row_i = 0; row_i < tail_size_y; row_i++) {
                dst_reg.row(row_i) =
                    xetla_cvt<dtype_acc, dtype_a, block_size_x>(src_reg) +
                    dst_reg.row(row_i);
              }
            }
          }
        }
        epilogue_t epilogue;
        epilogue(g, matAcc, mem_desc_c, args.epilogue_args);
        if (local_id == 0) {
          __ESIMD_NS::simd<int, 1> value = 1;
          __ESIMD_NS::simd<uint32_t, 1> offset = 0;
          int old_val = __ESIMD_NS::atomic_update<__ESIMD_NS::atomic_op::add>(
              args.atomic_buffer, offset, value)[0];
          __ESIMD_NS::slm_block_store<int, 1>(atomic_slm_offset, old_val);
        }
        item.barrier(sycl::access::fence_space::local_space);
        group_id = __ESIMD_NS::slm_block_load<int, 1>(atomic_slm_offset)[0];
        group_id += group_range;
        start_n = (group_id * wg_tile_n) % args.matrix_n_pad;
        group_m_id = (group_id * wg_tile_n) / args.matrix_n_pad;
      }
      pre_rows = cumsum_rows_for_experts[load_expert_num - 1];
      pre_tiles = cumsum_tiles_for_experts[load_expert_num - 1];
    }
    return;
  };
};

} // namespace gpu::xetla::kernel
