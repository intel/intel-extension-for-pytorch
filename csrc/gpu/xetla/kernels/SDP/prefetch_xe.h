#pragma once

#include "subgroup/tile/api.hpp"
#include "subgroup/tile/impl/op_function.hpp"

namespace gpu::xetla::subgroup::ext {

/// @brief Is to illustrate the memory information to prefetch data to cache.
/// @tparam dtype_ Is the data type.
/// @tparam tile_desc_ Is the tile descriptor.
/// @tparam mem_layout_ Is the memory data layout.
/// @tparam mem_space_ Is the memory space.
/// @tparam cooperative_num_ Is the thread nums to prefetch data.
/// @tparam arch_tag Is the hardware architecture tag.
template <
    typename dtype_,
    typename tile_desc_,
    mem_layout mem_layout_,
    mem_space mem_space_,
    uint32_t cooperative_num_,
    gpu_arch arch_tag = gpu_arch::Xe>
struct prefetch_payload_t {};

/// @brief Is to describe the global memory surface to prefetch data to cache
/// data in global memory will be prefetched into 2d tile
/// @tparam tile_desc_ Is the tile descriptor
/// @tparam dtype Is the data type
/// @tparam mem_layout Is the data layout
/// @tparam cooperative_num_ Is the thread nums to prefetch data
template <
    typename dtype_,
    uint32_t tile_size_x_,
    uint32_t tile_size_y_,
    uint32_t block_size_x_,
    uint32_t block_size_y_,
    mem_layout mem_layout_,
    uint32_t cooperative_num_>
struct prefetch_payload_t<
    dtype_,
    tile_desc_t<
        tile_size_x_,
        tile_size_y_,
        block_size_x_,
        block_size_y_,
        reg_layout::tiled>,
    mem_layout_,
    mem_space::global,
    cooperative_num_,
    gpu_arch::Xe> {
  using dtype = dtype_;
  using tile_desc = tile_desc_t<
      tile_size_x_,
      tile_size_y_,
      block_size_x_,
      block_size_y_,
      reg_layout::tiled>;
  static constexpr mem_space memory_space = mem_space::global;
  static constexpr mem_layout memory_layout = mem_layout_;
  static constexpr gpu_arch arch_tag = gpu_arch::Xe;

 private:
  static constexpr uint32_t tile_size_x = tile_desc::tile_size_x;
  static constexpr uint32_t tile_size_y = tile_desc::tile_size_y;
  static constexpr bool is_col_major = mem_layout_ == mem_layout::col_major;
  static constexpr uint32_t mem_tile_size_w =
      is_col_major ? tile_size_y : tile_size_x;
  static constexpr uint32_t mem_tile_size_h =
      is_col_major ? tile_size_x : tile_size_y;
  using arch_attr = arch_attr_t<gpu_arch::Xe>;
  using load_store_attr = arch_attr::load_store_attr;
  static constexpr int32_t max_block_size_w =
      load_store_attr::max_load_width_in_bytes / sizeof(dtype) > mem_tile_size_w
      ? mem_tile_size_w
      : load_store_attr::max_load_width_in_bytes / sizeof(dtype);
  static constexpr int32_t max_block_size_h =
      load_store_attr::max_load_height_in_elem;
  static constexpr uint32_t max_num_block_w =
      (mem_tile_size_w + max_block_size_w - 1) / max_block_size_w;

  static constexpr uint32_t cooperative_num = cooperative_num_;
  static constexpr uint32_t coop_num_w =
      detail::gcd<cooperative_num, max_num_block_w>::value;
  static constexpr uint32_t coop_num_h = cooperative_num / coop_num_w;
  // for x direction, we are fine with oob access, since it's still in the same
  // CL
  static constexpr uint32_t tile_size_w =
      max_block_size_w * max_num_block_w / coop_num_w;
  static constexpr int32_t num_block_w = max_num_block_w / coop_num_w;
  // todo: currently we don't have the oob check for y direction
  static constexpr uint32_t tile_size_h =
      (mem_tile_size_h + coop_num_h - 1) / coop_num_h;
  static constexpr int32_t num_block_h =
      (tile_size_h + max_block_size_h - 1) / max_block_size_h;
  using this_payload_t = prefetch_payload_t<
      dtype,
      tile_desc,
      mem_layout_,
      mem_space::global,
      cooperative_num_,
      gpu_arch::Xe>;

 public:
  static constexpr int32_t num_tdesc = num_block_w * num_block_h;
  xetla_vector<uint32_t, num_tdesc * 16> tdesc_prefetch;

  inline prefetch_payload_t(const this_payload_t& rhs) {
    this->tdesc_prefetch = rhs.tdesc_prefetch;
  }

  inline prefetch_payload_t() = default;

  inline this_payload_t& operator=(const this_payload_t& rhs) {
    this->tdesc_prefetch = rhs.tdesc_prefetch;
    return *this;
  }

  inline prefetch_payload_t(
      mem_desc_t<dtype, mem_layout_, mem_space::global>& mem_desc,
      uint32_t coop_id = 0) {
    xetla_tdescriptor base_tdesc = mem_desc.get_tdesc();
    uint32_t coop_id_x = coop_id % coop_num_w;
    uint32_t coop_id_y = coop_id / coop_num_w;
    xetla_update_tdesc_offsetx(
        base_tdesc.xetla_format<uint32_t>(), coop_id_x * tile_size_w);
    xetla_update_tdesc_offsety(
        base_tdesc.xetla_format<uint32_t>(), coop_id_y * tile_size_h);
    prepare_tdesc(base_tdesc);
  }

  inline prefetch_payload_t(
      dtype* p,
      int surface_width,
      int surface_height,
      int surface_pitch,
      int surface_offset_x,
      int surface_offset_y,
      uint32_t coop_id = 0) {
    uint32_t coop_id_x = coop_id % coop_num_w;
    uint32_t coop_id_y = coop_id / coop_num_w;
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
  // Be aware of the risks: Rule of three (copy constructor, copy assignment,
  // destructor) Please check if you need to add self-define destructor
  // ~prefetch_payload_t(){}

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

 private:
  __XETLA_API void prepare_tdesc(xetla_tdescriptor base_tdesc) {
    auto tdesc_2d = tdesc_prefetch.xetla_format<uint32_t, num_tdesc, 16>();
    uint32_t base_offset_y = 0;
#pragma unroll
    for (int i = 0; i < tile_size_h / max_block_size_h; i++) {
      auto tdesc_row_2d =
          tdesc_2d.xetla_select<num_block_w, 1, 16, 1>(i * num_block_w, 0);
      prepare_tile_desc_core<num_block_w, max_block_size_w, max_block_size_h>(
          tdesc_row_2d, base_tdesc, base_offset_y);
      base_offset_y += max_block_size_h;
    }
    if constexpr ((tile_size_h % max_block_size_h) != 0) {
      constexpr int i = tile_size_h / max_block_size_h;
      auto tdesc_row_2d =
          tdesc_2d.xetla_select<num_block_w, 1, 16, 1>(i * num_block_w, 0);
      constexpr uint32_t remain_size_y = tile_size_h % max_block_size_h;
      prepare_tile_desc_core<num_block_w, max_block_size_w, remain_size_y>(
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

/// @brief Is to describe the memory memory to prefetch data to cache
/// data in global memory will be prefetched into 1d tile
/// @tparam tile_desc_ Is the tile descriptor
/// @tparam dtype Is the data type
/// @tparam mem_layout Is the data layout
/// @tparam cooperative_num_ Is the thread nums to prefetch data
template <
    typename dtype_,
    uint32_t tile_size_x_,
    uint32_t block_size_x_,
    mem_layout mem_layout_,
    uint32_t cooperative_num_>
struct prefetch_payload_t<
    dtype_,
    tile_desc_t<tile_size_x_, 1, block_size_x_, 1, reg_layout::tiled>,
    mem_layout_,
    mem_space::global,
    cooperative_num_,
    gpu_arch::Xe> {
  using dtype = dtype_;
  // CL aligned, so we can use uint64_t
  using prefetch_dtype = uint64_t;
  using tile_desc =
      tile_desc_t<tile_size_x_, 1, block_size_x_, 1, reg_layout::tiled>;
  static constexpr mem_space memory_space = mem_space::global;
  static constexpr mem_layout memory_layout = mem_layout_;
  static constexpr gpu_arch arch_tag = gpu_arch::Xe;

 public:
  // Fetches the entire CL.
  static constexpr uint32_t cacheline_elems = 64 / sizeof(dtype);
  static constexpr uint32_t mem_block_nums =
      (tile_desc::tile_size_x + cacheline_elems - 1) / cacheline_elems;
  static constexpr uint32_t cooperative_num = cooperative_num_;

  // For mem_tile_nums < cooperative_num cases, mem_tile_size_x will be CL
  // length which might lead to illegal read. there are cooperative_num threads
  // to prefetch mem_block_nums each thread will prefetch mem_tile_size_x
  // elements
  static constexpr uint32_t mem_tile_size_x = mem_block_nums > cooperative_num
      ? (mem_block_nums + cooperative_num - 1) /
          cooperative_num* cacheline_elems
      : 0;
  using this_payload_t = prefetch_payload_t<
      dtype,
      tile_desc,
      mem_layout_,
      mem_space::global,
      cooperative_num_,
      gpu_arch::Xe>;

  // Fixed prefetch_dtype, close this assertion
  // static_assert(sizeof(prefetch_dtype) >= 4,
  //         "prefetch dtype size should at least DW aligned");

 public:
  static constexpr uint32_t scale_factor =
      sizeof(prefetch_dtype) / sizeof(dtype);
  uint32_t base_offset;
  prefetch_dtype* base_ptr;
  uint32_t pitch_in_bytes;

  inline prefetch_payload_t(const this_payload_t& rhs) {
    this->base_offset = rhs.base_offset;
    this->base_ptr = rhs.base_ptr;
    this->pitch_in_bytes = rhs.pitch_in_bytes;
  }

  inline prefetch_payload_t() = default;

  inline this_payload_t& operator=(const this_payload_t& rhs) {
    this->base_offset = rhs.base_offset;
    this->base_ptr = rhs.base_ptr;
    this->pitch_in_bytes = rhs.pitch_in_bytes;
    return *this;
  }

  inline prefetch_payload_t(
      mem_desc_t<dtype, mem_layout_, mem_space::global>& mem_desc,
      uint32_t coop_id = 0) {
    pitch_in_bytes = mem_desc.shape.stride * sizeof(dtype);
    uint32_t offset_x = mem_desc.coord.x;
    uint32_t offset_y = mem_desc.coord.y;
    base_offset = offset_y * pitch_in_bytes + offset_x * sizeof(dtype);
    dtype* ptr_temp = mem_desc.base.base;
    base_ptr = (prefetch_dtype*)ptr_temp +
        (coop_id % cooperative_num) * mem_tile_size_x;
  }

  inline prefetch_payload_t(
      dtype* p,
      int surface_width,
      int surface_height,
      int surface_pitch,
      int surface_offset_x,
      int surface_offset_y,
      uint32_t coop_id = 0) {
    pitch_in_bytes = surface_pitch * sizeof(dtype);
    uint32_t offset_x = surface_offset_x;
    uint32_t offset_y = surface_offset_y;
    base_offset = offset_y * pitch_in_bytes + offset_x * sizeof(dtype);
    base_ptr =
        (prefetch_dtype*)p + (coop_id % cooperative_num) * mem_tile_size_x;
  }

  template <tdesc_update_dir update_dir = tdesc_update_dir::x_dir>
  __XETLA_API void update_tdesc(int offset) {
    if constexpr (update_dir == tdesc_update_dir::x_dir) {
      base_offset += offset * sizeof(dtype);
    } else {
      base_offset += offset * pitch_in_bytes;
    }
  }
};

/// @brief Is to describe the memory infomation to prefetch data to cache
/// data located in shared local memory, nothing will do.
/// @tparam tile_desc_ Is the tile descriptor
/// @tparam dtype Is the data type
/// @tparam mem_layout Is the data layout
/// @tparam cooperative_num_ Is the thread nums to prefetch data
template <
    typename dtype_,
    typename tile_desc_,
    mem_layout mem_layout_,
    uint32_t cooperative_num_>
struct prefetch_payload_t<
    dtype_,
    tile_desc_,
    mem_layout_,
    mem_space::local,
    cooperative_num_,
    gpu_arch::Xe> {
  using dtype = dtype_;
  using tile_desc = tile_desc_;
  static constexpr mem_space memory_space = mem_space::local;
  static constexpr mem_layout memory_layout = mem_layout_;
  static constexpr gpu_arch arch_tag = gpu_arch::Xe;

  inline prefetch_payload_t(
      mem_desc_t<dtype, mem_layout_, mem_space::local>& mem_desc,
      uint32_t coop_id = 0) {}

  inline prefetch_payload_t(
      dtype* p,
      int surface_width,
      int surface_height,
      int surface_pitch,
      int surface_offset_x,
      int surface_offset_y,
      uint32_t coop_id = 0) {}

  template <tdesc_update_dir update_dir = tdesc_update_dir::x_dir>
  __XETLA_API void update_tdesc(int offset) {}
};

/// @brief Is prefetch data func, which data located in global memory is
/// prefetched to cache, where has higher bandwidth. e.g. In gemm, prefetch next
/// iteration data for mma consumption. This func is specicalized for block 2d
/// scenario.
/// @tparam payload_t Is the mem_payload_t struct illustrating memory info
/// payload indicates the source of prefetch operation.
/// @tparam L1 Is cache hint for L1 cache.
/// @tparam L3 Is cache hint for L3 cache.
/// @param payload Is the payload object with type payload_t. Contains all the
/// information for prefetches.
template <
    cache_hint L1 = cache_hint::cached,
    cache_hint L3 = cache_hint::cached,
    typename payload_t>
__XETLA_API typename std::enable_if_t<
    detail::check_prefetch_type<payload_t>::is_global_2d_xe>
tile_prefetch(payload_t& payload) {
  using dtype = typename payload_t::dtype;
  static constexpr uint32_t num_tdesc = payload_t::num_tdesc;
  auto tdesc_2d =
      payload.tdesc_prefetch.xetla_format<uint32_t, num_tdesc, 16>();

#pragma unroll
  for (int i = 0; i < num_tdesc; i++) {
    xetla_tprefetch_global<dtype, L1, L3>(tdesc_2d.row(i));
  }
}

/// @brief Is prefetch data func, which data located in global memory is
/// prefetched to cache, where has higher bandwidth. e.g. In gemm, prefetch next
/// iteration data for mma consumption. This func is specicalized for block 1d
/// scenario.
/// @tparam payload_t Is the mem_payload_t struct illustrating memory info
/// payload indicates the source of prefetch operation
/// @tparam L1 Is cache hint for L1 cache.
/// @tparam L3 Is cache hint for L3 cache.
/// @param payload Is the payload object with type payload_t. Contains all the
/// information for prefetches.
template <
    cache_hint L1 = cache_hint::cached,
    cache_hint L3 = cache_hint::cached,
    typename payload_t>
__XETLA_API typename std::enable_if_t<
    detail::check_prefetch_type<payload_t>::is_global_block_1d_xe>
tile_prefetch(payload_t& payload) {
  using dtype = typename payload_t::dtype;
  using tile_desc = typename payload_t::tile_desc;
  using prefetch_dtype = typename payload_t::prefetch_dtype;
  constexpr uint32_t prefetch_len =
      payload_t::mem_tile_size_x / payload_t::scale_factor;
  if constexpr (prefetch_len >= 64) {
#pragma unroll
    for (int j = 0; j < prefetch_len / 64; j++) {
      uint32_t offset_x = j * 64 * payload_t::scale_factor;
      uint32_t address_offset = offset_x * sizeof(dtype);
      xetla_prefetch_global<
          prefetch_dtype,
          64,
          data_size::default_size,
          L1,
          L3>(payload.base_ptr, payload.base_offset + address_offset);
    }
  }
  constexpr uint32_t tail_len = prefetch_len % 64;
  uint32_t tail_offset = prefetch_len / 64 * 64 * payload_t::scale_factor;
  detail::process_1d_tail<tail_len, 32, L1, L3, payload_t>(
      payload, tail_offset);
}

/// @brief Is prefetch data func.
/// Current shared local memory prefetch is not supported yet. Only used to keep
/// the consistency with global prefetch.
/// @tparam payload_t Is the mem_payload_t struct illustrating memory info.
/// @tparam L1 Is cache hint for L1 cache.
/// @tparam L3 Is cache hint for L3 cache.
/// @param payload Is the payload object with type payload_t. Contains all the
/// information for prefetches.
template <
    cache_hint L1 = cache_hint::cached,
    cache_hint L3 = cache_hint::cached,
    typename payload_t>
__XETLA_API typename std::enable_if_t<
    detail::check_prefetch_type<payload_t>::is_local_xe>
tile_prefetch(payload_t& payload) {}

} // namespace gpu::xetla::subgroup::ext
