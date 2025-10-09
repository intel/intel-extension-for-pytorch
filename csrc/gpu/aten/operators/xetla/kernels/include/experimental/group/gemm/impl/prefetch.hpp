#include "xetla.hpp"

namespace gpu::xetla::subgroup {

template <
    typename mem_desc_,
    typename tile_desc_,
    uint32_t cooperative_num_,
    gpu_arch arch_tag,
    typename enable = void>
struct block_prefetch_payload_t {};

/// @brief Is to describe the global memory surface to prefetch data to cache
/// data in global memory will be prefetched into 2d tile
/// @tparam tile_desc_ Is the tile descriptor
/// @tparam dtype Is the data type
/// @tparam mem_layout Is the data layout
/// @tparam num_coop_sg_ Is the thread nums to prefetch data
template <
    typename dtype_,
    uint32_t tile_size_x_,
    uint32_t tile_size_y_,
    uint32_t block_size_x_,
    uint32_t block_size_y_,
    mem_layout mem_layout_,
    uint32_t alignment_,
    uint32_t num_coop_sg_,
    reg_layout reg_layout_,
    gpu_arch arch_tag_>
struct block_prefetch_payload_t<
    mem_desc_t<dtype_, mem_layout_, mem_space::global, alignment_>,
    tile_desc_t<
        tile_size_x_,
        tile_size_y_,
        block_size_x_,
        block_size_y_,
        reg_layout_>,
    num_coop_sg_,
    arch_tag_,
    std::enable_if_t<(arch_tag_ == gpu_arch::XeHpc)>> {
  using dtype = dtype_;
  using mem_desc_t =
      mem_desc_t<dtype_, mem_layout_, mem_space::global, alignment_>;
  using tile_desc = tile_desc_t<
      tile_size_x_,
      tile_size_y_,
      block_size_x_,
      block_size_y_,
      reg_layout_>;
  static constexpr mem_space memory_space = mem_space::global;
  static constexpr mem_layout memory_layout = mem_layout_;
  static constexpr gpu_arch arch_tag = arch_tag_;

 private:
  static constexpr uint32_t tile_size_x = tile_desc::tile_size_x;
  static constexpr uint32_t tile_size_y = tile_desc::tile_size_y;
  static constexpr bool is_col_major = mem_layout_ == mem_layout::col_major;
  static constexpr uint32_t mem_tile_size_w =
      is_col_major ? tile_size_y : tile_size_x;
  static constexpr uint32_t mem_tile_size_h =
      is_col_major ? tile_size_x : tile_size_y;
  using load_store_attr = typename arch_attr_t<
      arch_tag>::template load_store_attr<msg_type::block_2d>;
  static constexpr uint32_t special_prefetch_width =
      load_store_attr::special_prefetch_width_in_bytes / sizeof(dtype);
  static constexpr uint32_t normal_prefetch_width =
      load_store_attr::max_load_width_in_bytes / sizeof(dtype);
  static constexpr bool is_special_prefetch =
      (mem_tile_size_w % special_prefetch_width) == 0;

  static constexpr uint32_t block_size_w = is_special_prefetch
      ? special_prefetch_width
      : (normal_prefetch_width > mem_tile_size_w ? mem_tile_size_w
                                                 : normal_prefetch_width);
  static constexpr uint32_t block_size_h =
      load_store_attr::max_load_height_in_elem;
  // could have over-prefetch, but that's should be fine
  static constexpr uint32_t max_num_block_w =
      (mem_tile_size_w + block_size_w - 1) / block_size_w;
  static constexpr uint32_t num_coop_sg = num_coop_sg_;
  static constexpr uint32_t num_coop_sg_w =
      detail::gcd<num_coop_sg, max_num_block_w>::value;
  static constexpr uint32_t num_coop_sg_h = num_coop_sg / num_coop_sg_w;

  static constexpr uint32_t num_block_w = max_num_block_w / num_coop_sg_w;
  static constexpr uint32_t tile_size_w = block_size_w * num_block_w;
  static constexpr uint32_t tile_size_h =
      (mem_tile_size_h + num_coop_sg_h - 1) / num_coop_sg_h;
  static constexpr uint32_t num_block_h =
      (tile_size_h + block_size_h - 1) / block_size_h;
  using this_payload_t =
      block_prefetch_payload_t<mem_desc_t, tile_desc, num_coop_sg_, arch_tag>;

 public:
  static constexpr uint32_t num_tdesc = num_block_w * num_block_h;
  xetla_vector<uint32_t, num_tdesc * 16> tdesc_prefetch;

  inline block_prefetch_payload_t(const this_payload_t& rhs) {
    this->tdesc_prefetch = rhs.tdesc_prefetch;
  }

  inline block_prefetch_payload_t() = default;

  inline this_payload_t& operator=(const this_payload_t& rhs) {
    this->tdesc_prefetch = rhs.tdesc_prefetch;
    return *this;
  }

  inline block_prefetch_payload_t(mem_desc_t& mem_desc, uint32_t coop_id = 0) {
    xetla_tdescriptor base_tdesc = mem_desc.get_tdesc();
    uint32_t coop_id_x = coop_id % num_coop_sg_w;
    uint32_t coop_id_y = coop_id / num_coop_sg_w;
    xetla_update_tdesc_offsetx(
        base_tdesc.xetla_format<uint32_t>(), coop_id_x * tile_size_w);
    xetla_update_tdesc_offsety(
        base_tdesc.xetla_format<uint32_t>(), coop_id_y * tile_size_h);
    prepare_tdesc(base_tdesc);
  }

  inline block_prefetch_payload_t(
      dtype* p,
      int surface_width,
      int surface_height,
      int surface_pitch,
      int surface_offset_x,
      int surface_offset_y,
      uint32_t coop_id = 0) {
    uint32_t coop_id_x = coop_id % num_coop_sg_w;
    uint32_t coop_id_y = coop_id / num_coop_sg_w;
    xetla_tdescriptor base_tdesc;
    xetla_fill_tdesc(
        base_tdesc.xetla_format<uint32_t>(),
        p,
        surface_width,
        surface_height,
        surface_pitch,
        surface_offset_x + coop_id_x * tile_size_w,
        surface_offset_y + coop_id_y * tile_size_h);
    prepare_tdesc(base_tdesc);
  }

  inline void init(xetla_tdescriptor base_tdesc, uint32_t coop_id = 0) {
    uint32_t coop_id_x = coop_id % num_coop_sg_w;
    uint32_t coop_id_y = coop_id / num_coop_sg_w;
    xetla_update_tdesc_offsetx(
        base_tdesc.xetla_format<uint32_t>(), coop_id_x * tile_size_w);
    xetla_update_tdesc_offsety(
        base_tdesc.xetla_format<uint32_t>(), coop_id_y * tile_size_h);
    prepare_tdesc(base_tdesc);
  }
  // Be aware of the risks: Rule of three (copy constructor, copy assignment,
  // destructor) Please check if you need to add self-define destructor
  // ~block_prefetch_payload_t(){}

  template <tdesc_update_dir update_dir = tdesc_update_dir::x_dir>
  __XETLA_API void update_tdesc(int offset) {
    auto tdesc_2d = tdesc_prefetch.xetla_format<uint32_t, num_tdesc, 16>();
    if constexpr (update_dir == tdesc_update_dir::x_dir) {
#pragma unroll
      for (int i = 0; i < num_tdesc; i++) {
        xetla_update_tdesc_offsetx(tdesc_2d.row(i), offset);
      }
    } else {
#pragma unroll
      for (int i = 0; i < num_tdesc; i++) {
        xetla_update_tdesc_offsety(tdesc_2d.row(i), offset);
      }
    }
  }
  __XETLA_API void set_tdesc_width(uint32_t size) {
    auto tdesc_2d = tdesc_prefetch.xetla_format<uint32_t, num_tdesc, 16>();
#pragma unroll
    for (int i = 0; i < num_tdesc; i++) {
      xetla_set_tdesc_width<dtype>(tdesc_2d.row(i), size);
    }
  }

  __XETLA_API void set_tdesc_pitch(uint32_t size) {
    auto tdesc_2d = tdesc_prefetch.xetla_format<uint32_t, num_tdesc, 16>();
#pragma unroll
    for (int i = 0; i < num_tdesc; i++) {
      xetla_set_tdesc_pitch<dtype>(tdesc_2d.row(i), size);
    }
  }

  __XETLA_API void set_tdesc_height(uint32_t size) {
    auto tdesc_2d = tdesc_prefetch.xetla_format<uint32_t, num_tdesc, 16>();
#pragma unroll
    for (int i = 0; i < num_tdesc; i++) {
      xetla_set_tdesc_height(tdesc_2d.row(i), size);
    }
  }

  __XETLA_API void update_tdesc_base_address(int offset) {
    auto tdesc_2d = tdesc_prefetch.xetla_format<uint32_t, num_tdesc, 16>();
#pragma unroll
    for (int i = 0; i < num_tdesc; i++) {
      xetla_update_tdesc_base_address(tdesc_2d.row(i), offset);
    }
  }

  __XETLA_API void set_tdesc_base_address(uint64_t addr) {
    auto tdesc_2d = tdesc_prefetch.xetla_format<uint32_t, num_tdesc, 16>();
#pragma unroll
    for (int i = 0; i < num_tdesc; i++) {
      gpu::xetla::detail::xetla_set_tensor_base_address(tdesc_2d.row(i), addr);
    }
  }

  __XETLA_API void update_tdesc_base_address_masked(
      int offset,
      uint16_t mask = 1) {
    auto tdesc_2d = tdesc_prefetch.xetla_format<uint32_t, num_tdesc, 16>();
#pragma unroll
    for (int i = 0; i < num_tdesc; i++) {
      xetla_update_tdesc_base_address(tdesc_2d.row(i), offset);
    }

#pragma unroll
    for (int i = 0; i < num_tdesc; i++) {
      xetla_tdesc_mask_op(tdesc_2d.row(i), mask);
    }
  }

  __XETLA_API void set_offset(
      int32_t offset_x,
      int32_t offset_y,
      uint32_t coop_id = 0) {
    uint32_t coop_id_x = coop_id % num_coop_sg_w;
    uint32_t coop_id_y = coop_id / num_coop_sg_w;

    auto tdesc_2d = tdesc_prefetch.xetla_format<uint32_t, num_tdesc, 16>();
    int32_t base_offset_y = offset_y +
        (is_col_major ? coop_id_x * tile_size_w : coop_id_y * tile_size_h);
#pragma unroll
    for (int i = 0; i < num_block_h; i++) {
      auto tdesc_row_2d =
          tdesc_2d.xetla_select<num_block_w, 1, 16, 1>(i * num_block_w, 0);

      int32_t base_offset_x = offset_x +
          (is_col_major ? coop_id_y * tile_size_h : coop_id_x * tile_size_w);
#pragma unroll
      for (int j = 0; j < num_block_w; j++) {
        int32_t offset_width = is_col_major ? base_offset_y : base_offset_x;
        int32_t offset_height = is_col_major ? base_offset_x : base_offset_y;
        gpu::xetla::detail::xetla_set_tensor_offset_x(
            tdesc_row_2d.row(j), offset_width);
        gpu::xetla::detail::xetla_set_tensor_offset_y(
            tdesc_row_2d.row(j), offset_height);

        base_offset_x += block_size_w;
      }
      base_offset_y += block_size_h;
    }
  }

 private:
  __XETLA_API void prepare_tdesc(xetla_tdescriptor base_tdesc) {
    auto tdesc_2d = tdesc_prefetch.xetla_format<uint32_t, num_tdesc, 16>();
    uint32_t base_offset_y = 0;
#pragma unroll
    for (int i = 0; i < tile_size_h / block_size_h; i++) {
      auto tdesc_row_2d =
          tdesc_2d.xetla_select<num_block_w, 1, 16, 1>(i * num_block_w, 0);
      prepare_tile_desc_core<num_block_w, block_size_w, block_size_h>(
          tdesc_row_2d, base_tdesc, base_offset_y);
      base_offset_y += block_size_h;
    }
    if constexpr ((tile_size_h % block_size_h) != 0) {
      constexpr int i = tile_size_h / block_size_h;
      auto tdesc_row_2d =
          tdesc_2d.xetla_select<num_block_w, 1, 16, 1>(i * num_block_w, 0);
      constexpr uint32_t remain_size_y = tile_size_h % block_size_h;
      prepare_tile_desc_core<num_block_w, block_size_w, remain_size_y>(
          tdesc_row_2d, base_tdesc, base_offset_y);
    }
  }

  template <int32_t num_tdesc, uint32_t size_x, uint32_t size_y>
  __XETLA_API static void prepare_tile_desc_core(
      xetla_matrix_ref<uint32_t, num_tdesc, 16> __REF__ tdesc_2d,
      xetla_tdescriptor base_tdesc,
      uint32_t base_offset_y) {
    uint32_t base_offset_x = 0;
#pragma unroll
    for (int j = 0; j < num_tdesc; j++) {
      tdesc_2d.row(j) = base_tdesc;

      constexpr uint32_t block_widthx_widthy_arrlen =
          (size_x - 1) | ((size_y - 1) << 8);
      gpu::xetla::detail::xetla_set_block_widthx_widthy_arrlen(
          tdesc_2d.row(j), block_widthx_widthy_arrlen);

      xetla_update_tdesc_offsetx(tdesc_2d.row(j), base_offset_x);
      xetla_update_tdesc_offsety(tdesc_2d.row(j), base_offset_y);
      base_offset_x += size_x;
    }
  }
};

/// @brief Is prefetch data func, which data located in global memory is
/// prefetched to cache, where has higher bandwidth. e.g. In gemm, prefetch next
/// iteration data for mma consumption. This func is specicalized for block 2d
/// scenario.
/// @tparam payload_t Is the mem_payload_t struct illustrating memory info
/// payload indicates the source of prefetch operation.
/// @tparam L1 Is cache hint for L1 cache.
/// @tparam L2 Is cache hint for L2 cache.
/// @param payload Is the payload object with type payload_t. Contains all the
/// information for prefetches.
template <
    cache_hint L1 = cache_hint::cached,
    cache_hint L2 = cache_hint::cached,
    typename payload_t>
__XETLA_API void block_tile_prefetch(payload_t& payload) {
  using dtype = typename payload_t::dtype;
  static constexpr uint32_t num_tdesc = payload_t::num_tdesc;
  auto tdesc_2d =
      payload.tdesc_prefetch.xetla_format<uint32_t, num_tdesc, 16>();

#pragma unroll
  for (int i = 0; i < num_tdesc; i++) {
    xetla_tprefetch_global<dtype, L1, L2, payload_t::arch_tag>(tdesc_2d.row(i));
  }
}
} // namespace gpu::xetla::subgroup
