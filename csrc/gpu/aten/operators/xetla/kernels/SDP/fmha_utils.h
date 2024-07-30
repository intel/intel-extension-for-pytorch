#pragma once
#include "../xetla.h"

namespace gpu::xetla {

namespace fmha {

template <typename mat_t>
struct tile_mask_t {
  using accum_t = typename mat_t::dtype;
  static constexpr accum_t kNegInfinity = INFINITY * -1;
  static constexpr uint32_t tile_size_x = mat_t::tile_size_x;
  static constexpr uint32_t tile_size_y = mat_t::tile_size_y;
  static constexpr uint32_t block_size_x = mat_t::block_size_x;
  static constexpr uint32_t block_size_y = mat_t::block_size_y;
  static constexpr int32_t num_block_x = mat_t::num_block_x;
  static constexpr uint32_t block_elems = mat_t::block_elems;

  // --------------------- // causal_mask // ---------------------- //

  inline static void causal_mask(
      mat_t& src,
      uint32_t start_x,
      uint32_t start_y) {
#pragma unroll
    for (uint32_t i = 0; i < tile_size_y / block_size_y; i++) {
      uint32_t blk_start_y = start_y + i * block_size_y;
#pragma unroll
      for (int j = 0; j < num_block_x; j++) {
        uint32_t blk_start_x = start_x + j * block_size_x;
        if (blk_start_x + block_size_x > blk_start_y) {
          xetla_vector<uint32_t, block_size_x> blk_seq_x =
              xetla_vector_gen<uint32_t, block_size_x>(blk_start_x, 1);
          auto src_sub =
              src.reg
                  .xetla_select<block_elems, 1>(
                      (i * num_block_x + j) * block_elems)
                  .xetla_format<accum_t, block_size_y, block_size_x>();
#pragma unroll
          for (int k = 0; k < block_size_y; k++) {
            xetla_mask<block_size_x> mask = blk_seq_x > blk_start_y + k;
            src_sub.row(k).xetla_merge(kNegInfinity, mask);
          }
        }
      }
    }

    if constexpr ((tile_size_y % block_size_y) != 0) {
      constexpr uint32_t tail_start_y =
          tile_size_y / block_size_y * block_size_y;
      constexpr uint32_t tail_size_y = tile_size_y % block_size_y;
      constexpr uint32_t tail_block_elems = tail_size_y * block_size_x;

      uint32_t blk_start_y = start_y + tail_start_y;
#pragma unroll
      for (int j = 0; j < num_block_x; j++) {
        uint32_t blk_start_x = start_x + j * block_size_x;
        if (blk_start_x + block_size_x > blk_start_y) {
          xetla_vector<uint32_t, block_size_x> blk_seq_x =
              xetla_vector_gen<uint32_t, block_size_x>(blk_start_x, 1);
          auto src_sub =
              src.reg
                  .xetla_select<tail_block_elems, 1>(
                      tail_start_y * tile_size_x + j * tail_block_elems)
                  .xetla_format<accum_t, tail_size_y, block_size_x>();
#pragma unroll
          for (int k = 0; k < tail_size_y; k++) {
            xetla_mask<block_size_x> mask = blk_seq_x > blk_start_y + k;
            src_sub.row(k).xetla_merge(kNegInfinity, mask);
          }
        }
      }
    }
  }

  // -------------------- // padding_mask // ---------------------- //

  inline static void padding_mask(mat_t& src, int num_keep) {
#pragma unroll
    for (uint32_t i = 0; i < tile_size_y / block_size_y; i++) {
#pragma unroll
      for (uint32_t j = 0; j < num_block_x; j++) {
        int start_x = j * block_size_x;
        int num_keep_blk = std::max(0, num_keep - start_x);

        if (num_keep_blk < int(block_size_x)) {
          xetla_mask<block_size_x> mask =
              xetla_vector_gen<uint32_t, block_size_x>(1, 1) > num_keep_blk;
          auto src_sub =
              src.reg
                  .xetla_select<block_elems, 1>(
                      (i * num_block_x + j) * block_elems)
                  .xetla_format<accum_t, block_size_y, block_size_x>();
#pragma unroll
          for (uint32_t k = 0; k < block_size_y; k++) {
            src_sub.row(k).xetla_merge(kNegInfinity, mask);
          }
        }
      }
    }

    if constexpr ((tile_size_y % block_size_y) != 0) {
      constexpr uint32_t tail_start_y =
          tile_size_y / block_size_y * block_size_y;
      constexpr uint32_t tail_size_y = tile_size_y % block_size_y;
      constexpr uint32_t tail_block_elems = tail_size_y * block_size_x;
#pragma unroll
      for (int j = 0; j < num_block_x; j++) {
        int start_x = j * block_size_x;
        int num_keep_blk = std::max(num_keep - start_x, 0);

        if (num_keep_blk < block_size_x) {
          xetla_mask<block_size_x> mask =
              xetla_vector_gen<uint32_t, block_size_x>(1, 1) > num_keep_blk;
          auto src_sub =
              src.reg
                  .xetla_select<tail_block_elems, 1>(
                      tail_start_y * tile_size_x + j * tail_block_elems)
                  .xetla_format<accum_t, tail_size_y, block_size_x>();
#pragma unroll
          for (int k = 0; k < tail_size_y; k++) {
            src_sub.row(k).xetla_merge(kNegInfinity, mask);
          }
        }
      }
    }
  }
};

// ==================== // group_row_reduce_t // ================== //

template <
    typename mat_t,
    uint32_t kNumSg,
    reduce_op reduce_kind,
    gpu_arch arch_tag>
struct group_row_reduce_t {
  using T = typename mat_t::dtype;
  static constexpr uint32_t kNum = mat_t::tile_desc::tile_size_y;
  static constexpr uint32_t kTotal = kNum * kNumSg;

  // store results of subgroup to slm
  using store_tile_desc =
      subgroup::tile_desc_t<kNum, 1, kNum, 1, reg_layout::tiled>;
  using store_tile_t = subgroup::tile_t<T, store_tile_desc>;
  using store_payload_t = subgroup::mem_payload_t<
      mem_desc_t<T, mem_layout::row_major, mem_space::local>,
      store_tile_desc,
      msg_type::block_1d,
      arch_tag>;
  // load all subgroup results together
  using load_tile_desc =
      subgroup::tile_desc_t<kTotal, 1, kTotal, 1, reg_layout::tiled>;
  using load_tile_t = subgroup::tile_t<T, load_tile_desc>;
  using load_payload_t = subgroup::mem_payload_t<
      mem_desc_t<T, mem_layout::row_major, mem_space::local>,
      load_tile_desc,
      subgroup::msg_type_v<
          load_tile_desc,
          mem_desc_t<T, mem_layout::row_major, mem_space::local>>,
      arch_tag>;

  xetla_nbarrier_t<kNumSg, kNumSg, arch_tag> nbarrier;
  uint32_t slm_base;
  uint32_t sg_id;
  inline group_row_reduce_t() = default;
  inline group_row_reduce_t(
      uint32_t sg_id_,
      uint32_t nbarrier_id,
      uint32_t slm_base_) {
    nbarrier.init_nbarrier(nbarrier_id, nbarrier_role::producer_consumer);
    sg_id = sg_id_;
    slm_base = slm_base_;
  }

  inline KERNEL_FUNC xetla_vector<T, kNum> operator()(mat_t& src) {
    xetla_vector<T, kNum> ret =
        subgroup::tile_reduce<reduce_kind, T, T, 1>(src);
    if constexpr (kNumSg == 1)
      return ret;

    store_tile_t sg_store;
    store_payload_t sg_store_payload(
        slm_base, kTotal, 1, kTotal, sg_id * kNum, 0);
    sg_store.reg = ret;
    subgroup::tile_store(sg_store, sg_store_payload);

    xetla_fence<memory_kind::shared_local>();
    nbarrier.arrive_wait();

    load_tile_t sg_load;
    load_payload_t sg_load_payload(slm_base, kTotal, 1, kTotal, 0, 0);
    subgroup::tile_load(sg_load, sg_load_payload);

    auto data_2d = sg_load.reg.xetla_format<T, kNumSg, kNum>();
    ret = data_2d.row(0);
#pragma unroll
    for (uint32_t i = 1; i < kNumSg; i++) {
      ret = reduce_helper<reduce_kind, T, kNum>(data_2d.row(i), ret);
    }
    return ret;
  }
};

enum class add_type : uint8_t {
  single_line = 0, // add one line of data of given coord to target tile
  single_element = 1 // add single data of given coord to target tile
};

/// @brief Is the bias_add op functor.
/// Load the 1d bias data from memory and get the input from matAcc, update the
/// output in place. Used in epilogue::tile_op or chained_tile_op.
/// @tparam dtype_bias Is the data type of bias buffer.
/// @tparam arch_tag Is the hardware architecture tag.
template <
    typename dtype_bias_,
    gpu_arch arch_tag,
    add_type add_tag = add_type::single_line>
struct bias_add_op_t {};

template <typename dtype_bias_, gpu_arch arch_tag>
struct bias_add_op_t<dtype_bias_, arch_tag, add_type::single_line> {
  using dtype_bias = dtype_bias_;
  using mem_desc_bias_t =
      mem_desc_t<dtype_bias, mem_layout::row_major, mem_space::global>;
  using shape_t = typename mem_desc_bias_t::shape_t;
  using coord_t = typename mem_desc_bias_t::coord_t;
  using base_t = typename mem_desc_bias_t::base_t;

  struct arguments_t {
    base_t base;
    shape_t shape;
    inline arguments_t() = default;
    inline arguments_t(base_t base_, shape_t shape_)
        : base(base_), shape(shape_) {}
  };
  template <typename matAcc_t>
  __XETLA_API KERNEL_FUNC void operator()(
      matAcc_t& matAcc,
      const coord_t& coord,
      const arguments_t& args,
      [[maybe_unused]] uint32_t slm_base = 0,
      [[maybe_unused]] uint32_t nbarrier_base = 0) {
    using dtype_acc = typename matAcc_t::dtype;
    static constexpr uint32_t tile_size_x = matAcc_t::tile_size_x;
    static constexpr uint32_t tile_size_y = matAcc_t::tile_size_y;
    static constexpr uint32_t block_size_x = matAcc_t::block_size_x;
    static constexpr uint32_t block_size_y = matAcc_t::block_size_y;
    static constexpr int32_t num_block_x = matAcc_t::num_block_x;
    static constexpr uint32_t block_elems = matAcc_t::block_elems;

    using bias_tile_desc_t = subgroup::
        tile_desc_t<tile_size_x, 1, block_size_x, 1, reg_layout::tiled>;
    using bias_t = subgroup::tile_t<dtype_bias, bias_tile_desc_t>;
    using bias_payload_t = subgroup::mem_payload_t<
        mem_desc_t<dtype_bias, mem_desc_bias_t::layout, mem_desc_bias_t::space>,
        bias_tile_desc_t,
        subgroup::msg_type_v<
            bias_tile_desc_t,
            mem_desc_t<
                dtype_bias,
                mem_desc_bias_t::layout,
                mem_desc_bias_t::space>>,
        arch_tag>;
    coord_t bias_coord(coord.x, coord.y);
    mem_desc_bias_t mem_desc_bias(args.base, args.shape, bias_coord);
    bias_t bias;
    bias_payload_t bias_payload(mem_desc_bias);
    tile_load<cache_hint::cached, cache_hint::cached>(bias, bias_payload);

#pragma unroll
    for (uint32_t i = 0; i < tile_size_y / block_size_y; i++) {
#pragma unroll
      for (int j = 0; j < num_block_x; j++) {
        auto dst_reg =
            matAcc.reg
                .xetla_select<block_elems, 1>(
                    (i * num_block_x + j) * block_elems)
                .xetla_format<dtype_acc, block_size_y, block_size_x>();
#pragma unroll
        for (uint32_t row_i = 0; row_i < block_size_y; row_i++) {
          auto src_reg =
              bias.reg.xetla_select<block_size_x, 1>(j * block_size_x);
          dst_reg.row(row_i) =
              xetla_cvt<dtype_acc, dtype_bias, block_size_x>(src_reg) +
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
      for (int j = 0; j < num_block_x; j++) {
        auto dst_reg =
            matAcc.reg
                .xetla_select<tail_block_elems, 1>(
                    tail_start_y * tile_size_x + j * tail_block_elems)
                .xetla_format<dtype_acc, tail_size_y, block_size_x>();
#pragma unroll
        for (uint32_t row_i = 0; row_i < tail_size_y; row_i++) {
          auto src_reg =
              bias.reg.xetla_select<block_size_x, 1>(j * block_size_x);
          dst_reg.row(row_i) =
              xetla_cvt<dtype_acc, dtype_bias, block_size_x>(src_reg) +
              dst_reg.row(row_i);
        }
      }
    }
  }
};

template <typename dtype_bias_, gpu_arch arch_tag>
struct bias_add_op_t<dtype_bias_, arch_tag, add_type::single_element> {
  using dtype_bias = dtype_bias_;
  using mem_desc_bias_t =
      mem_desc_t<dtype_bias, mem_layout::row_major, mem_space::global>;
  using shape_t = typename mem_desc_bias_t::shape_t;
  using coord_t = typename mem_desc_bias_t::coord_t;
  using base_t = typename mem_desc_bias_t::base_t;

  struct arguments_t {
    base_t base;
    shape_t shape;
    inline arguments_t() = default;
    inline arguments_t(base_t base_, shape_t shape_)
        : base(base_), shape(shape_) {}
  };
  template <typename matAcc_t>
  __XETLA_API KERNEL_FUNC void operator()(
      matAcc_t& matAcc,
      const coord_t& coord,
      const arguments_t& args,
      [[maybe_unused]] uint32_t slm_base = 0,
      [[maybe_unused]] uint32_t nbarrier_base = 0) {
    using dtype_acc = typename matAcc_t::dtype;
    static constexpr uint32_t tile_size_x = matAcc_t::tile_size_x;
    static constexpr uint32_t tile_size_y = matAcc_t::tile_size_y;
    static constexpr uint32_t block_size_x = matAcc_t::block_size_x;
    static constexpr uint32_t block_size_y = matAcc_t::block_size_y;
    static constexpr int32_t num_block_x = matAcc_t::num_block_x;
    static constexpr uint32_t block_elems = matAcc_t::block_elems;

    dtype_bias* ptr = static_cast<dtype_bias*>(args.base.base);
    int32_t pos_x = coord.x > args.shape.x - 1 ? args.shape.x - 1 : coord.x;
    int32_t pos_y = coord.y > args.shape.y - 1 ? args.shape.y - 1 : coord.y;
    uint32_t offset = (pos_y + pos_x * args.shape.stride) * sizeof(dtype_bias);
    auto bias_data_vector = xetla_load_global<
        dtype_bias,
        16,
        1,
        cache_hint::cached,
        cache_hint::cached>(ptr, offset);
    dtype_acc bias_data =
        xetla_cvt<dtype_acc, dtype_bias, 16>(bias_data_vector)[0];

    {
      auto bias_reg =
          xetla_vector<dtype_acc, block_size_y * block_size_x>(bias_data);
#pragma unroll
      for (uint32_t i = 0; i < tile_size_y / block_size_y; i++) {
#pragma unroll
        for (int j = 0; j < num_block_x; j++) {
          auto dst_reg =
              matAcc.reg
                  .xetla_select<block_elems, 1>(
                      (i * num_block_x + j) * block_elems)
                  .xetla_format<dtype_acc, block_size_y, block_size_x>();

          dst_reg += bias_reg;
        }
      }
    }
    // process the tail
    if constexpr ((tile_size_y % block_size_y) != 0) {
      constexpr uint32_t tail_start_y =
          tile_size_y / block_size_y * block_size_y;
      constexpr int32_t tail_size_y = tile_size_y % block_size_y;
      constexpr int32_t tail_block_elems = tail_size_y * block_size_x;
      auto bias_reg =
          xetla_vector<dtype_acc, tail_size_y * block_size_x>(bias_data);
#pragma unroll
      for (int j = 0; j < num_block_x; j++) {
        auto dst_reg =
            matAcc.reg
                .xetla_select<tail_block_elems, 1>(
                    tail_start_y * tile_size_x + j * tail_block_elems)
                .xetla_format<dtype_acc, tail_size_y, block_size_x>();
        dst_reg += bias_reg;
      }
    }
  }
};

struct tile_mul {
  template <typename dtype, int vec_len>
  static xetla_vector<dtype, vec_len> inline func(
      xetla_vector<dtype, vec_len> vec_data,
      dtype data) {
    return vec_data * data;
  }
};

} // namespace fmha

} // namespace gpu::xetla

namespace gpu::xetla::group {

template <
    typename epilogue_policy,
    typename tile_shape_,
    typename mem_desc_c_t_>
class epilogue_transp_t {};

template <
    typename tile_op_t_,
    typename tile_shape_,
    typename mem_desc_c_t_,
    gpu_arch arch_tag_>
class epilogue_transp_t<
    epilogue_policy_tile_op<tile_op_t_, arch_tag_>,
    tile_shape_,
    mem_desc_c_t_> {
 public:
  using tile_shape = tile_shape_;
  using mem_desc_c_t = mem_desc_c_t_;
  static constexpr gpu_arch arch_tag = arch_tag_;
  static constexpr uint32_t barrier_count = 0;
  static constexpr uint32_t slm_size = 0;

  struct arguments_t {};

 private:
  using work_group_t = typename tile_shape::work_group_t;
  static constexpr uint32_t wg_tile_m = tile_shape::wg_tile_size_y;
  static constexpr uint32_t wg_tile_n = tile_shape::wg_tile_size_x;
  static constexpr uint32_t sg_tile_m = tile_shape::sg_tile_size_y;
  static constexpr uint32_t sg_tile_n = tile_shape::sg_tile_size_x;
  static constexpr uint32_t wg_size_x = tile_shape::wg_size_x;
  static constexpr uint32_t wg_size_y = tile_shape::wg_size_y;
  using dtype_c = typename mem_desc_c_t::dtype;
  static constexpr mem_layout mem_layout_c = mem_desc_c_t::layout;
  static constexpr mem_space mem_space_c = mem_desc_c_t::space;
  static constexpr msg_type msg_type_c =
      (mem_space_c == mem_space::global ? msg_type::block_2d
                                        : msg_type::scatter);
  /// @brief Updates tile base descriptor based on the tid.
  __XETLA_API static void update_sg_tile_tdesc(
      work_group_t& g,
      mem_desc_c_t& mem_desc_c) {
    int32_t sg_idy = g.get_id() % wg_size_x;
    int32_t sg_idx = g.get_id() / wg_size_x;
    int32_t tile_offset_n = sg_idx * sg_tile_m;
    int32_t tile_offset_m = sg_idy * sg_tile_n;
    mem_desc_c.update_coord(tile_offset_n, tile_offset_m);
  }

 public:
  template <typename matAcc_t>
  __XETLA_API KERNEL_FUNC void operator()(
      work_group_t& g,
      matAcc_t& matAcc,
      mem_desc_c_t mem_desc_c,
      [[maybe_unused]] arguments_t args = {},
      [[maybe_unused]] uint32_t slm_base = 0,
      [[maybe_unused]] uint32_t nbarrier_base = 0) {
    static_assert(
        mem_layout_c == mem_layout::row_major &&
            mem_space_c == mem_space::local,
        "layout should be row_major and space should be local");
    using matC_tile_desc_t = subgroup::tile_desc_t<
        matAcc_t::tile_size_x,
        matAcc_t::tile_size_y,
        matAcc_t::block_size_x,
        matAcc_t::block_size_y,
        reg_layout::vnni_tiled_col_major>;

    using matC_t = subgroup::tile_t<dtype_c, matC_tile_desc_t>;
    using matC_payload_t = subgroup::mem_payload_t<
        mem_desc_t<dtype_c, mem_layout_c, mem_space_c>,
        matC_tile_desc_t,
        msg_type_c,
        arch_tag>;

    update_sg_tile_tdesc(g, mem_desc_c);
    matC_payload_t matC_payload(mem_desc_c);
    matC_t matC;
    subgroup::vnni_transform(matC, matAcc);
    subgroup::tile_store(matC, matC_payload);
  }
};

template <
    typename epilogue_policy,
    typename tile_shape,
    typename mem_desc_c,
    class enable = void>
class epilogue_write_back_t {};

// Use cache_hint::write_back for L1
template <typename tile_shape_, typename mem_desc_c_t_, gpu_arch arch_tag_>
class epilogue_write_back_t<
    epilogue_policy_default<arch_tag_>,
    tile_shape_,
    mem_desc_c_t_,
    std::enable_if_t<valid_xe_arch_tag<arch_tag_>>> {
 public:
  using epilogue_policy = epilogue_policy_default<arch_tag_>;
  using tile_shape = tile_shape_;
  using mem_desc_c_t = mem_desc_c_t_;
  static constexpr gpu_arch arch_tag = arch_tag_;
  static constexpr uint32_t barrier_count = 0;
  static constexpr uint32_t slm_size = mem_desc_c_t::is_local
      ? tile_shape::wg_tile_size_x * tile_shape::wg_tile_size_y
      : 0;
  /// @brief Epilogue arguments.
  struct arguments_t {};

 private:
  using work_group_t = typename tile_shape::work_group_t;
  static constexpr uint32_t sg_tile_m = tile_shape::sg_tile_size_y;
  static constexpr uint32_t sg_tile_n = tile_shape::sg_tile_size_x;
  static constexpr uint32_t wg_size_x = tile_shape::wg_size_x;
  static constexpr uint32_t wg_size_y = tile_shape::wg_size_y;
  using dtype_c = typename mem_desc_c_t::dtype;
  static constexpr mem_layout mem_layout_c = mem_desc_c_t::layout;
  static constexpr mem_space mem_space_c = mem_desc_c_t::space;

  /// @brief Updates tile base descriptor based on the tid.
  __XETLA_API static void update_sg_tile_tdesc(
      work_group_t& g,
      mem_desc_c_t& mem_desc_c) {
    int32_t sg_idx = g.get_id() % wg_size_x;
    int32_t sg_idy = g.get_id() / wg_size_x;
    int32_t tile_offset_n = sg_idx * sg_tile_n;
    int32_t tile_offset_m = sg_idy * sg_tile_m;
    mem_desc_c.update_coord(tile_offset_n, tile_offset_m);
  }

 public:
  static constexpr msg_type msg_type_c =
      (mem_space_c == mem_space::global ? msg_type::block_2d
                                        : msg_type::scatter);

  template <typename matAcc_t>
  __XETLA_API KERNEL_FUNC void operator()(
      work_group_t& g,
      matAcc_t& matAcc,
      mem_desc_c_t mem_desc_c,
      [[maybe_unused]] arguments_t args = {},
      [[maybe_unused]] uint32_t slm_base = 0,
      [[maybe_unused]] uint32_t nbarrier_base = 0) {
    using mat_tile_desc = typename matAcc_t::tile_desc;
    using matC_t = subgroup::tile_t<dtype_c, mat_tile_desc>;
    using matC_payload_t = subgroup::
        mem_payload_t<mem_desc_c_t, mat_tile_desc, msg_type_c, arch_tag>;
    update_sg_tile_tdesc(g, mem_desc_c);
    matC_t matC;
    matC_payload_t matC_payload(mem_desc_c);
    subgroup::elemwise_cvt(matC, matAcc);
    subgroup::tile_store(matC, matC_payload);
  }
};

} // namespace gpu::xetla::group
