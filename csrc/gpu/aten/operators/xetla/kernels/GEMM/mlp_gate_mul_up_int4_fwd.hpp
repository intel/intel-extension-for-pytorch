#pragma once
#include "../xetla.h"

namespace gpu::xetla {
namespace mlp {

#define DEF_SCALE_MEM_DESC(NAME, PTR)                           \
  mem_desc_scale_t NAME(                                        \
      args.quant_param.PTR,                                     \
      {args.matrix_n, scale_size_y, args.quant_param.scale_ld}, \
      {start_x_scale, start_y_scale});

#define DEF_ZP_MEM_DESC(NAME, PTR)                    \
  mem_desc_zero_pt_t NAME(                            \
      args.quant_param.PTR,                           \
      {(args.matrix_n + pack_ratio - 1) / pack_ratio, \
       ((args.matrix_k + dequant_s - 1) / dequant_s), \
       args.quant_param.zero_pt_ld / pack_ratio},     \
      {start_x_zero_pt, start_y_zero_pt});

#define DEF_ZP_FP_MEM_DESC(NAME, PTR)                 \
  mem_desc_zero_pt_t NAME(                            \
      args.quant_param.PTR,                           \
      {args.matrix_n,                                 \
       ((args.matrix_k + dequant_s - 1) / dequant_s), \
       args.quant_param.zero_pt_ld},                  \
      {start_x_zero_pt, start_y_zero_pt});

#define DEF_COL_MAJOR_WEI_MEM_DESC(NAME, PTR)                           \
  NAME.init(                                                            \
      args.PTR,                                                         \
      {boundary_n, boundary_k / pack_ratio, args.matB_ld / pack_ratio}, \
      {start_n, int(start_k / pack_ratio)});

#define DEF_ROW_MAJOR_WEI_MEM_DESC(NAME, PTR)                           \
  NAME.init(                                                            \
      args.PTR,                                                         \
      {boundary_n / pack_ratio, boundary_k, args.matB_ld / pack_ratio}, \
      {int(start_n / pack_ratio), start_k});

#define DEF_ACC_MEM_DESC(NAME, PTR)            \
  mem_desc_acc_t NAME(                         \
      args.PTR,                                \
      {boundary_n, boundary_m, args.matrix_n}, \
      {acc_start_x, acc_start_y});

#define ASSIGN_SYM_GEMM_ARG(ARG, WEI, SCALE) \
  ARG = gemm_args_t(mem_desc_a, WEI, inner_loop_start, inner_loop_count, SCALE);

#define ASSIGN_ASYM_GEMM_ARG(ARG, WEI, SCALE, ZP) \
  ARG = gemm_args_t(                              \
      mem_desc_a, WEI, inner_loop_start, inner_loop_count, SCALE, ZP);

template <
    typename tile_shape_acc_,
    typename tile_shape_cnt_,
    typename mem_desc_acc_t_,
    typename mem_desc_cnt_t_,
    uint32_t num_group_reduction,
    uint32_t counter_size,
    gpu_arch arch_tag_>
class global_sum_reduce_two_mat_t {
 public:
  static constexpr gpu_arch arch_tag = arch_tag_;
  using tile_shape_acc = tile_shape_acc_;
  using tile_shape_cnt = tile_shape_cnt_;
  using mem_desc_acc_t = mem_desc_acc_t_;
  using mem_desc_cnt_t = mem_desc_cnt_t_;
  using dtype_acc = typename mem_desc_acc_t::dtype;
  using dtype_cnt = typename mem_desc_cnt_t::dtype;
  inline uint32_t update_reduce_counter(mem_desc_cnt_t& mem_desc_cnt) {
    constexpr uint32_t SIMD = 16;
    uint32_t pitch_in_bytes =
        mem_desc_cnt.shape.stride * sizeof(dtype_cnt) * counter_size;
    uint32_t offset_x = mem_desc_cnt.coord.x;
    uint32_t offset_y = mem_desc_cnt.coord.y;
    uint64_t address = (uint64_t)mem_desc_cnt.base.base +
        offset_y * pitch_in_bytes + offset_x * sizeof(dtype_cnt) * counter_size;
    xetla_vector<uint32_t, SIMD> offsets =
        xetla_vector_gen<uint32_t, SIMD>(0, 1);
    offsets *= sizeof(dtype_cnt);
    xetla_mask<SIMD> pred(0);
    pred[0] = 1;
    xetla_vector<dtype_cnt, SIMD> ret = xetla_atomic_global<
        atomic_op::iinc,
        dtype_cnt,
        SIMD,
        data_size::default_size,
        cache_hint::uncached,
        cache_hint::write_back>((dtype_cnt*)address, offsets, pred);
    return ret[0];
  }

  inline void clean_reduce_counter(mem_desc_cnt_t& mem_desc_cnt) {
    uint32_t pitch_in_bytes =
        mem_desc_cnt.shape.stride * sizeof(dtype_cnt) * counter_size;
    uint32_t offset_x = mem_desc_cnt.coord.x;
    uint32_t offset_y = mem_desc_cnt.coord.y;
    uint64_t address = (uint64_t)mem_desc_cnt.base.base +
        offset_y * pitch_in_bytes + offset_x * sizeof(dtype_cnt) * counter_size;
    xetla_vector<dtype_cnt, 1> zeros(0);

    xetla_store_global<
        dtype_cnt,
        1,
        cache_hint::uncached,
        cache_hint::write_back>((dtype_cnt*)address, 0, zeros);
  }

 private:
  static constexpr uint32_t acc_sg_tile_y = tile_shape_acc::sg_tile_size_y;
  static constexpr uint32_t acc_sg_tile_x = tile_shape_acc::sg_tile_size_x;
  static constexpr uint32_t cnt_sg_tile_y = tile_shape_cnt::sg_tile_size_y;
  static constexpr uint32_t cnt_sg_tile_x = tile_shape_cnt::sg_tile_size_x;
  static constexpr uint32_t wg_size_x = tile_shape_acc::wg_size_x;
  static constexpr uint32_t wg_size_y = tile_shape_acc::wg_size_y;
  static_assert(
      (tile_shape_acc::wg_size_x == tile_shape_cnt::wg_size_x) &&
          (tile_shape_acc::wg_size_y == tile_shape_cnt::wg_size_y),
      "acc and cnt wg shape need to be matched");
  using work_group_t = typename tile_shape_acc::work_group_t;

  /// @brief Updates tile base descriptor based on the tid.
  inline void update_sg_tile_tdesc(
      work_group_t& g,
      mem_desc_acc_t& mem_desc_acc_mat1,
      mem_desc_acc_t& mem_desc_acc_mat2,
      mem_desc_cnt_t& mem_desc_cnt) {
    int32_t sg_idx = g.get_id() % wg_size_x;
    int32_t sg_idy = g.get_id() / wg_size_x;
    int32_t acc_tile_offset_x = sg_idx * acc_sg_tile_x;
    int32_t acc_tile_offset_y = sg_idy * acc_sg_tile_y;
    mem_desc_acc_mat1.update_coord(acc_tile_offset_x, acc_tile_offset_y);
    mem_desc_acc_mat2.update_coord(acc_tile_offset_x, acc_tile_offset_y);
    int32_t cnt_tile_offset_x = sg_idx * cnt_sg_tile_x;
    int32_t cnt_tile_offset_y = sg_idy * cnt_sg_tile_y;
    mem_desc_cnt.update_coord(cnt_tile_offset_x, cnt_tile_offset_y);
  }

 public:
  static constexpr uint32_t barrier_count = 0;
  static constexpr uint32_t slm_size = 0;
  uint32_t reduce_id = 0;

  inline bool is_last_group() {
    return reduce_id == (num_group_reduction - 1);
  }

  template <typename matAcc_t>
  __XETLA_API KERNEL_FUNC void operator()(
      work_group_t& g,
      matAcc_t& matAcc1,
      matAcc_t& matAcc2,
      mem_desc_acc_t mem_desc_acc_mat1,
      mem_desc_acc_t mem_desc_acc_mat2,
      mem_desc_cnt_t mem_desc_cnt,
      [[maybe_unused]] uint32_t slm_base = 0,
      [[maybe_unused]] uint32_t nbarrier_base = 0) {
    static_assert(
        std::is_same<typename matAcc_t::dtype, dtype_acc>::value,
        "matAcc_t::dtype should match with dtype_acc");
    update_sg_tile_tdesc(g, mem_desc_acc_mat1, mem_desc_acc_mat2, mem_desc_cnt);
    using matAcc_tile_desc_t = typename matAcc_t::tile_desc;
    using matAcc_store_payload_t = subgroup::mem_payload_t<
        mem_desc_acc_t,
        matAcc_tile_desc_t,
        msg_type::atomic_add,
        arch_tag>;
    matAcc_store_payload_t matAcc1_store_payload(mem_desc_acc_mat1);
    matAcc_store_payload_t matAcc2_store_payload(mem_desc_acc_mat2);
    subgroup::tile_store<cache_hint::uncached, cache_hint::write_back>(
        matAcc1, matAcc1_store_payload);
    subgroup::tile_store<cache_hint::uncached, cache_hint::write_back>(
        matAcc2, matAcc2_store_payload);
    xetla_fence<
        memory_kind::untyped_global,
        fence_op::none,
        fence_scope::tile>();
    reduce_id = update_reduce_counter(mem_desc_cnt);
    if (reduce_id == (num_group_reduction - 1)) {
      using matAcc_payload_t = subgroup::mem_payload_t<
          mem_desc_acc_t,
          matAcc_tile_desc_t,
          msg_type::block_2d,
          arch_tag>;
      matAcc_payload_t matAcc1_payload(mem_desc_acc_mat1);
      matAcc_payload_t matAcc2_payload(mem_desc_acc_mat2);
      subgroup::tile_load(matAcc1, matAcc1_payload);
      subgroup::tile_load(matAcc2, matAcc2_payload);
      clean_reduce_counter(mem_desc_cnt);
      using mat_zero_t = subgroup::tile_t<dtype_acc, matAcc_tile_desc_t>;
      mat_zero_t mat_zero;
      mat_zero.reg = 0;
      subgroup::tile_store<cache_hint::uncached, cache_hint::write_back>(
          mat_zero, matAcc1_payload);
      subgroup::tile_store<cache_hint::uncached, cache_hint::write_back>(
          mat_zero, matAcc2_payload);
    }
  }
};

/// @brief This is an implementation of part of mlp fusion, that is
/// Mul(post_ops_gate(gate_proj Linear), post_ops_up(up_proj Linear)), note that
/// shape of gate_proj & up_proj weight must be same.
///
/// @tparam num_global_kslicing_ Is the k dim split ratio between groups.
/// @tparam num_local_kslicing_ Is the k dim split ratio within a group.
/// @tparam gemm_t_ Is the gemm functor to compose a GEMM.
/// @tparam post_ops_up_t_ Is the post ops for up_proj linear.
/// @tparam post_ops_gate_t_ Is the post ops for gate_proj linear.
/// @tparam epilogue_t_ Just for write back the output matrix.
template <
    gpu_arch arch_tag,
    int num_global_kslicing_,
    int num_local_kslicing_,
    typename gemm_t_,
    typename post_ops_up_t_,
    typename post_ops_gate_t_,
    typename epilogue_t_>
class mlp_gate_mul_up_int4_fwd_t {
  using gemm_t = gemm_t_;
  using epilogue_t = epilogue_t_;
  using post_ops_up_t = post_ops_up_t_;
  using post_ops_gate_t = post_ops_gate_t_;
  using gemm_args_t = typename gemm_t::arguments_t;
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

  using mem_desc_a_t = typename gemm_t::mem_desc_a_t;
  using mem_desc_b_t = typename gemm_t::mem_desc_b_t;
  using mem_desc_scale_t = typename gemm_t::mem_desc_scale_t;
  using mem_desc_zero_pt_t = typename gemm_t::mem_desc_zero_pt_t;
  using mem_desc_c_t = typename epilogue_t::mem_desc_c_t;
  using matA_base_t = typename mem_desc_a_t::base_t;
  using matB_base_t = typename mem_desc_b_t::base_t;
  using matC_base_t = typename mem_desc_c_t::base_t;
  using scale_base_t = typename mem_desc_scale_t::base_t;
  using zero_pt_base_t = typename mem_desc_zero_pt_t::base_t;
  using matAcc_t = typename gemm_t::matC_t;
  using dtype_acc = typename matAcc_t::dtype;

  using mem_desc_acc_t =
      mem_desc_t<dtype_acc, mem_layout::row_major, mem_space::global>;
  using mem_desc_cnt_t =
      mem_desc_t<uint32_t, mem_layout::row_major, mem_space::global>;
  using acc_base_t = typename mem_desc_acc_t::base_t;
  using cnt_base_t = typename mem_desc_cnt_t::base_t;

  static_assert(
      gemm_t::compute_policy::is_int4_matB_policy,
      "should match with 4bit gemm impl");

  static constexpr uint32_t num_global_kslicing = num_global_kslicing_;
  static constexpr uint32_t num_local_kslicing = num_local_kslicing_;
  static_assert(
      (num_global_kslicing > 0) && (num_local_kslicing > 0),
      "min slicing ratio is 1");

  static_assert(
      (num_local_kslicing & (num_local_kslicing - 1)) == 0,
      "num_local_kslicing should be power of 2!");

  using kslicing_t = group::cooperative_reduce_t<
      reduce_op::sum,
      tile_shape,
      matAcc_t,
      num_local_kslicing,
      arch_tag>;

  using mat_slice_t = typename kslicing_t::mat_slice_t;
  static constexpr uint32_t ks_coop_num_x = kslicing_t::coop_num_x;
  static constexpr uint32_t ks_coop_num_y = kslicing_t::coop_num_y;

  static constexpr uint32_t gemm_nbarr_count = gemm_t::barrier_count;
  static constexpr uint32_t gemm_slm_size = gemm_t::slm_size;

  static constexpr uint32_t kslicing_nbarr_count = kslicing_t::barrier_count;
  static constexpr uint32_t kslicing_slm_size = kslicing_t::slm_size;

  static constexpr uint32_t counter_size = 8;

  static constexpr bool disable_kslicing =
      (num_local_kslicing == 1 && num_global_kslicing == 1);

  static constexpr uint32_t local_range_m =
      (wg_tile_m + sg_tile_m - 1) / sg_tile_m;
  static constexpr uint32_t local_range_n =
      (wg_tile_n + sg_tile_n - 1) / sg_tile_n;
  static_assert(
      local_range_m * local_range_n * num_local_kslicing <=
      arch_attr_t<arch_tag>::thread_per_wg);

  using tile_shape_cnt = group::tile_shape_t<
      ks_coop_num_x * wg_size_x,
      ks_coop_num_y * wg_size_y,
      ks_coop_num_x,
      ks_coop_num_y>;

 public:
  struct quant_param_t {
    scale_base_t up_proj_scale_base;
    scale_base_t gate_proj_scale_base;
    zero_pt_base_t up_proj_zero_pt_base;
    zero_pt_base_t gate_proj_zero_pt_base;
    uint32_t scale_ld;
    uint32_t zero_pt_ld;
  };

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
    /// @brief Is the leading dimension (pitch) size of the matrix A in memory.
    uint32_t matA_ld;
    /// @brief Is the leading dimension (pitch) size of the weight in
    /// memory, note in mlp-gp-up-fusion the shape of gate_proj & up_proj must
    /// be same.
    uint32_t matB_ld;
    /// @brief Is the leading dimension (pitch) size of the matrix out in
    /// memory.
    uint32_t mat_out_ld;
    /// @brief Is the base address of matrix A.
    matA_base_t matA_base;
    /// @brief Is the base address of matrix up_proj.
    matB_base_t mat_up_proj_base;
    /// @brief Is the base address of matrix gate_proj.
    matB_base_t mat_gate_proj_base;
    /// @brief Is the base address of matrix out.
    matC_base_t mat_out_base;
    /// @brief Is the base address of up_proj linear accumulation buffer.
    acc_base_t acc_up_proj_base;
    /// @brief Is the base address of gate_proj linear accumulation buffer.
    acc_base_t acc_gate_proj_base;
    /// @brief Is the base address of counter buffer.
    cnt_base_t cnt_base;
    /// @brief Is the dequant related param, include zp/scale.
    quant_param_t quant_param;

    post_ops_up_t::arguments_t post_ops_up_args;
    post_ops_gate_t::arguments_t post_ops_gate_args;

    /// @brief Constructs arguments with default method.
    inline arguments_t() = default;

    /// @brief Set for device copyable
    static constexpr bool host_callable = true;

    // Be aware of the risks: Rule of three (copy constructor, copy assignment,
    // destructor) Please check if you need to add self-define destructor
    // ~arguments_t(){}

    inline arguments_t(
        uint32_t matrix_m_,
        uint32_t matrix_k_,
        uint32_t matrix_n_,
        matA_base_t matA_base_,
        uint32_t matA_ld_,
        matB_base_t mat_up_proj_base_,
        matB_base_t mat_gate_proj_base_,
        uint32_t matB_ld_,
        matC_base_t mat_out_base_,
        uint32_t mat_out_ld_,
        acc_base_t acc_up_proj_base_,
        acc_base_t acc_gate_proj_base_,
        cnt_base_t cnt_base_,
        quant_param_t quant_param_,
        post_ops_up_t::arguments_t post_ops_up_args_,
        post_ops_gate_t::arguments_t post_ops_gate_args_)
        : matrix_m(matrix_m_),
          matrix_k(matrix_k_),
          matrix_n(matrix_n_),
          matA_ld(matA_ld_),
          matB_ld(matB_ld_),
          mat_out_ld(mat_out_ld_),
          matA_base(matA_base_),
          mat_up_proj_base(mat_up_proj_base_),
          mat_gate_proj_base(mat_gate_proj_base_),
          mat_out_base(mat_out_base_),
          acc_up_proj_base(acc_up_proj_base_),
          acc_gate_proj_base(acc_gate_proj_base_),
          cnt_base(cnt_base_),
          quant_param(quant_param_),
          post_ops_up_args(post_ops_up_args_),
          post_ops_gate_args(post_ops_gate_args_) {}
  };

  /// @brief Gets named_barrier id consumption count.
  /// Users query and get a named_barrier id consumption count in compile time.
  /// @return The count of named barriers required.
  __XETLA_API static constexpr uint32_t get_barrier_count() {
    constexpr uint32_t count =
        gemm_nbarr_count * num_local_kslicing + kslicing_nbarr_count;
    static_assert(
        count <= 32, "The named_barrier count should be less than 32!");
    return count;
  }

  /// @brief Gets local memory size consumption.
  /// Users query and get a local memory consumption size in compile time.
  /// @return The size of local memory required.
  __XETLA_API static constexpr uint32_t get_slm_size() {
    constexpr uint32_t size = 2 *
        (gemm_slm_size * num_local_kslicing +
         kslicing_slm_size); // 2 gemm op in mlp-fusion.
    static_assert(
        size <= arch_attr_t<arch_tag>::local_mem_size,
        "The local memory size excess!");
    return size;
  }

  /// @brief Host helper function to get the expected local range under the
  /// current GEMM config.
  /// @return Expected local range.
  static inline const cl::sycl::range<3> get_local_range() {
    XETLA_PRINTF(
        "Local range: {%d, %d, %d}",
        num_local_kslicing,
        local_range_m,
        local_range_n);
    static const cl::sycl::range<3> local_range =
        cl::sycl::range<3>{num_local_kslicing, local_range_m, local_range_n};
    return local_range;
  };

  /// @brief Host helper function to get the expected group range under the
  /// current GEMM config.
  /// @param matrix_m Is the size of the m dimension of the matrix
  /// multiplication (m x k x n).
  /// @param matrix_n Is the size of the n dimension of the matrix
  /// multiplication (m x k x n).
  /// @return Expected group range.
  static inline cl::sycl::range<3> get_group_range(
      uint32_t matrix_m,
      uint32_t matrix_n) {
    uint32_t group_range_m = (matrix_m + wg_tile_m - 1) / wg_tile_m;
    uint32_t group_range_n = (matrix_n + wg_tile_n - 1) / wg_tile_n;
    XETLA_PRINTF(
        "Group range: {%d, %d, %d}",
        num_global_kslicing,
        group_range_m,
        group_range_n);
    return cl::sycl::range<3>{
        num_global_kslicing, group_range_m, group_range_n};
  };

  /// @brief Host helper function to get the expected nd_range under the current
  /// GEMM config.
  /// @param args Is the GEMM arguments for application-related runtime
  /// variables.
  /// @return Expected nd_range.
  static inline cl::sycl::nd_range<3> get_nd_range(arguments_t& args) {
    const cl::sycl::range<3> local_range = get_local_range();
    cl::sycl::range<3> group_range =
        get_group_range(args.matrix_m, args.matrix_n);
    return cl::sycl::nd_range<3>{group_range * local_range, local_range};
  };

  /// @brief Host helper function to get the expected accumulation buffer size
  /// of the current GEMM config.
  /// @param matrix_m Is the size of the m dimension of the matrix
  /// multiplication (m x k x n).
  /// @param matrix_n Is the size of the n dimension of the matrix
  /// multiplication (m x k x n).
  /// @return Expected accumulation buffer size in unit of elements.
  static size_t get_acc_buf_size(uint32_t matrix_m, uint32_t matrix_n) {
    return matrix_m * matrix_n;
  };

  /// @brief Host helper function to get the expected counter buffer size of the
  /// current GEMM config.
  /// @param matrix_m Is the size of the m dimension of the matrix
  /// multiplication (m x k x n).
  /// @param matrix_n Is the size of the n dimension of the matrix
  /// multiplication (m x k x n).
  /// @return Expected counter buffer size in unit of elements.
  static size_t get_cnt_buf_size(uint32_t matrix_m, uint32_t matrix_n) {
    size_t group_range_m = (matrix_m + wg_tile_m - 1) / wg_tile_m;
    size_t group_range_n = (matrix_n + wg_tile_n - 1) / wg_tile_n;
    return group_range_m * group_range_n * wg_size_y * wg_size_x *
        ks_coop_num_y * ks_coop_num_x * counter_size;
  };

  using global_reduce_sync_t = global_sum_reduce_two_mat_t<
      tile_shape,
      tile_shape_cnt,
      mem_desc_acc_t,
      mem_desc_cnt_t,
      num_global_kslicing,
      counter_size,
      arch_tag>;

  /// @brief Main execution function for MLP-fusion.
  /// The processing order is 1) set group-level base and boundary, split group
  /// to workgroups -> 2) sg-level building up_proj & gate_proj block -> 3)
  /// sg-level reduction -> 4) global-reduce, store up_proj and gate_proj in
  /// tile registers -> 5) the last-global-reduce sg tile-register-level apply
  /// corresponding post ops to gate_proj output tile and up_proj output tile.
  /// -> 6) elt-wise multiply gate_proj output and up_proj output -> 7) epilogue
  /// write-back.
  /// @param Item the sycl::nd_item, returns execution related information, such
  /// as workgroup id, subgroup id...
  /// @param args Is the MLP arguments for application-related runtime
  /// variables.
  /// @param slm_base Is the slm base address.
  /// @param nbarrier_base Is the named barrier base.
  __XETLA_API KERNEL_FUNC void operator()(
      sycl::nd_item<3>& item,
      const arguments_t& args,
      uint32_t slm_base = 0,
      uint32_t nbarrier_base = 0) {
    work_group_t g(item.get_local_linear_id() % work_group_size);
    uint32_t wg_id = item.get_local_linear_id() / work_group_size;
    int start_m = item.get_group(1) * wg_tile_m;
    int start_n = item.get_group(2) * wg_tile_n;
    int start_k = 0;
    uint32_t wg_tile_k = args.matrix_k;
    uint32_t boundary_n = std::min(start_n + wg_tile_n, args.matrix_n);
    uint32_t boundary_m = std::min(start_m + wg_tile_m, args.matrix_m);
    uint32_t boundary_k = wg_tile_k;
    if constexpr (num_global_kslicing > 1) {
      wg_tile_k = (wg_tile_k + num_global_kslicing - 1) / num_global_kslicing;
      start_k = start_k + item.get_group(0) * wg_tile_k;
      boundary_k = (start_k + wg_tile_k) > boundary_k ? boundary_k
                                                      : (start_k + wg_tile_k);
    }
    if constexpr (num_local_kslicing > 1) {
      wg_tile_k = (wg_tile_k + num_local_kslicing - 1) / num_local_kslicing;
      start_k = start_k + wg_id * wg_tile_k;
      boundary_k = (start_k + wg_tile_k) > boundary_k ? boundary_k
                                                      : (start_k + wg_tile_k);
    }

    int start_x_scale = start_n;
    int start_y_scale = start_k / dequant_s;

    int start_x_zero_pt =
        gemm_t::compute_policy::quant_mode == quant_mode::I4_ASYM_FP_ZERO
        ? start_n
        : start_n / pack_ratio;
    int start_y_zero_pt = start_k / dequant_s;

    // set up arguments
    uint32_t gemm_slm_base = slm_base;
    uint32_t gemm_nbarr_base = nbarrier_base;
    if constexpr (num_local_kslicing > 1) {
      gemm_slm_base = slm_base + wg_id * gemm_slm_size;
      gemm_nbarr_base = nbarrier_base + wg_id * gemm_nbarr_count;
    }
    uint32_t up_proj_kslicing_slm_base =
        slm_base + num_local_kslicing * gemm_slm_size;
    uint32_t gate_proj_kslicing_slm_base =
        slm_base + num_local_kslicing * gemm_slm_size + kslicing_slm_size;
    uint32_t kslicing_nbarr_base =
        nbarrier_base + num_local_kslicing * gemm_nbarr_count;
    uint32_t epilogue_slm_base =
        gate_proj_kslicing_slm_base + kslicing_slm_size;
    uint32_t epilogue_nbarr_base = kslicing_nbarr_base + kslicing_nbarr_count;

    mem_desc_a_t mem_desc_a;
    mem_desc_b_t mem_desc_up_proj;
    mem_desc_b_t mem_desc_gate_proj;
    mem_desc_c_t mem_desc_out;

    mem_desc_a.init(
        args.matA_base,
        {boundary_k, boundary_m, args.matA_ld},
        {start_k, start_m});
    if constexpr (gemm_t::is_col_major_b) {
      DEF_COL_MAJOR_WEI_MEM_DESC(mem_desc_up_proj, mat_up_proj_base)
      DEF_COL_MAJOR_WEI_MEM_DESC(mem_desc_gate_proj, mat_gate_proj_base)
    } else {
      DEF_ROW_MAJOR_WEI_MEM_DESC(mem_desc_up_proj, mat_up_proj_base)
      DEF_ROW_MAJOR_WEI_MEM_DESC(mem_desc_gate_proj, mat_gate_proj_base)
    }

    uint32_t scale_size_y = ((args.matrix_k + dequant_s - 1) / dequant_s);

    DEF_SCALE_MEM_DESC(mem_desc_up_proj_scale, up_proj_scale_base)
    DEF_SCALE_MEM_DESC(mem_desc_gate_proj_scale, gate_proj_scale_base)

    uint32_t inner_loop_start = (start_k + k_stride - 1) / k_stride;
    uint32_t inner_loop_count = (wg_tile_k + k_stride - 1) / k_stride;

    gemm_args_t up_proj_args, gate_proj_args;
    if constexpr (gemm_t::compute_policy::quant_mode == quant_mode::I4_SYM) {
      ASSIGN_SYM_GEMM_ARG(
          up_proj_args, mem_desc_up_proj, mem_desc_up_proj_scale)
      ASSIGN_SYM_GEMM_ARG(
          gate_proj_args, mem_desc_gate_proj, mem_desc_gate_proj_scale)
    } else if constexpr (
        gemm_t::compute_policy::quant_mode == quant_mode::I4_ASYM) {
      DEF_ZP_MEM_DESC(mem_desc_up_zero_pt, up_proj_zero_pt_base)
      DEF_ZP_MEM_DESC(mem_desc_gate_zero_pt, gate_proj_zero_pt_base)
      ASSIGN_ASYM_GEMM_ARG(
          up_proj_args,
          mem_desc_up_proj,
          mem_desc_up_proj_scale,
          mem_desc_up_zero_pt)
      ASSIGN_ASYM_GEMM_ARG(
          gate_proj_args,
          mem_desc_gate_proj,
          mem_desc_gate_proj_scale,
          mem_desc_gate_zero_pt)
    } else if constexpr (
        gemm_t::compute_policy::quant_mode == quant_mode::I4_ASYM_FP_ZERO) {
      DEF_ZP_FP_MEM_DESC(mem_desc_up_zero_pt, up_proj_zero_pt_base)
      DEF_ZP_FP_MEM_DESC(mem_desc_gate_zero_pt, gate_proj_zero_pt_base)
      ASSIGN_ASYM_GEMM_ARG(
          up_proj_args,
          mem_desc_up_proj,
          mem_desc_up_proj_scale,
          mem_desc_up_zero_pt)
      ASSIGN_ASYM_GEMM_ARG(
          gate_proj_args,
          mem_desc_gate_proj,
          mem_desc_gate_proj_scale,
          mem_desc_gate_zero_pt)
    } else {
      static_assert(false, "Unsupported quant mode");
    }
    matAcc_t mat_up_proj_Acc, mat_gate_proj_Acc;
    mat_slice_t up_proj_out, gate_proj_out;
    mat_up_proj_Acc.init(0);
    mat_gate_proj_Acc.init(0);
    gemm_t gemm;
    gemm(g, mat_up_proj_Acc, up_proj_args, gemm_slm_base, gemm_nbarr_base);
    gemm(g, mat_gate_proj_Acc, gate_proj_args, gemm_slm_base, gemm_nbarr_base);

    auto silu_mul_write_back = [&] {
      using tile_type =
          std::conditional_t<disable_kslicing, matAcc_t, mat_slice_t>;
      tile_type *gate_tile, *up_tile;
      if constexpr (disable_kslicing) {
        gate_tile = &mat_gate_proj_Acc;
        up_tile = &mat_up_proj_Acc;
      } else {
        gate_tile = &gate_proj_out;
        up_tile = &up_proj_out;
      }
      post_ops_up_t{}(
          *up_tile,
          mem_desc_out.coord,
          args.post_ops_up_args,
          slm_base,
          nbarrier_base);
      post_ops_gate_t{}(
          *gate_tile,
          mem_desc_out.coord,
          args.post_ops_gate_args,
          slm_base,
          nbarrier_base);
      constexpr uint32_t out_tile_size = mat_slice_t::tile_size_x *
          mat_slice_t::tile_size_y; // shape of slice_tile and gate_tile is
                                    // same when disable kslicing.
      gate_tile->reg.xetla_select<out_tile_size, 1>(0) =
          gate_tile->reg.xetla_select<out_tile_size, 1>(0) *
          up_tile->reg.xetla_select<out_tile_size, 1>(0);

      epilogue_t epilogue;
      epilogue(
          g,
          mat_gate_proj_Acc,
          mem_desc_out,
          {},
          epilogue_slm_base,
          epilogue_nbarr_base);
    };

    if constexpr (disable_kslicing) {
      mem_desc_out.init(
          args.mat_out_base,
          {boundary_n, boundary_m, args.mat_out_ld},
          {start_n, start_m});
      silu_mul_write_back();
    } else {
      kslicing_t kslicing(wg_id);
      kslicing(
          g,
          up_proj_out,
          mat_up_proj_Acc,
          up_proj_kslicing_slm_base,
          kslicing_nbarr_base);
      kslicing(
          g,
          gate_proj_out,
          mat_gate_proj_Acc,
          gate_proj_kslicing_slm_base,
          kslicing_nbarr_base);

      if (kslicing.is_valid_post_process_wg()) {
        int32_t coop_offset_x = kslicing.coop_id_x * mat_slice_t::tile_size_x;
        int32_t coop_offset_y = kslicing.coop_id_y * mat_slice_t::tile_size_y;
        int32_t acc_start_x = start_n + coop_offset_x;
        int32_t acc_start_y = start_m + coop_offset_y;
        int32_t cnt_start_x =
            item.get_group(2) * tile_shape_cnt::wg_tile_size_x +
            kslicing.coop_id_x;
        int32_t cnt_start_y =
            item.get_group(1) * tile_shape_cnt::wg_tile_size_y +
            kslicing.coop_id_y;
        uint32_t group_range_x = item.get_group_range(2);
        uint32_t group_range_y = item.get_group_range(1);
        uint32_t cnt_size_x = group_range_x * tile_shape_cnt::wg_tile_size_x;
        uint32_t cnt_size_y = group_range_y * tile_shape_cnt::wg_tile_size_y;

        DEF_ACC_MEM_DESC(mem_desc_up_proj_acc, acc_up_proj_base)
        DEF_ACC_MEM_DESC(mem_desc_gate_proj_acc, acc_gate_proj_base)
        mem_desc_cnt_t mem_desc_cnt(
            args.cnt_base,
            {cnt_size_x, cnt_size_y, cnt_size_x},
            {cnt_start_x, cnt_start_y});

        global_reduce_sync_t global_reduce_sync;
        global_reduce_sync(
            g,
            up_proj_out,
            gate_proj_out,
            mem_desc_up_proj_acc,
            mem_desc_gate_proj_acc,
            mem_desc_cnt);

        mem_desc_out.init(
            args.mat_out_base,
            {boundary_n, boundary_m, args.mat_out_ld},
            {start_n + coop_offset_x, start_m + coop_offset_y});

        if (global_reduce_sync.is_last_group()) {
          silu_mul_write_back();
        }
      }
    }
  }
};
} // namespace mlp
} // namespace gpu::xetla
