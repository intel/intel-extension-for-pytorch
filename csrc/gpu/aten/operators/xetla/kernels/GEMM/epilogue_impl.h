#pragma once

#include <utils/DPCPP.h>
#include "../xetla.h"

namespace xpu {
namespace xetla {

using namespace cl::sycl;
using namespace gpu;
using namespace gpu::xetla;
using namespace gpu::xetla::group;
using namespace gpu::xetla::kernel;
using namespace gpu::xetla::subgroup;

namespace epilogue_impl {

template <typename dtype_in_>
struct alpha_beta_op_t {
  using dtype_in = dtype_in_;
  using mem_desc_in_t =
      mem_desc_t<dtype_in, mem_layout::row_major, mem_space::global>;
  using shape_t = typename mem_desc_in_t::shape_t;
  using coord_t = typename mem_desc_in_t::coord_t;
  using base_t = typename mem_desc_in_t::base_t;

  struct arguments_t {
    shape_t shape;
    base_t base;
    float alpha, beta;
    inline arguments_t() = default;
    inline arguments_t(base_t base_, shape_t shape_, float alpha_, float beta_)
        : base(base_), shape(shape_), alpha(alpha_), beta(beta_) {}
  };
  template <typename matAcc_t>
  __XETLA_API KERNEL_FUNC void operator()(
      matAcc_t& matAcc,
      const coord_t& coord,
      const arguments_t& args,
      uint32_t slm_base = 0,
      uint32_t nbarrier_base = 0) {
    using dtype_acc = typename matAcc_t::dtype;
    static constexpr uint32_t tile_size_x = matAcc_t::tile_size_x;
    static constexpr uint32_t tile_size_y = matAcc_t::tile_size_y;
    static constexpr uint32_t block_size_x = matAcc_t::block_size_x;
    static constexpr uint32_t block_size_y = matAcc_t::block_size_y;
    static constexpr int32_t num_block_x = matAcc_t::num_block_x;
    static constexpr int32_t num_block_y = matAcc_t::num_block_y;
    static constexpr uint32_t tile_elems = matAcc_t::tile_elems;
    static constexpr uint32_t block_elems = matAcc_t::block_elems;

    using mat_in_tile_desc_t = tile_desc_t<
        tile_size_x,
        tile_size_y,
        block_size_x,
        block_size_y,
        reg_layout::tiled>;
    using mat_in_tile_t = tile_t<dtype_in, mat_in_tile_desc_t>;
    using mat_in_payload_t = mem_payload_t<
        dtype_in,
        mat_in_tile_desc_t,
        msg_type_v<mat_in_tile_desc_t, mem_desc_in_t::space>,
        mem_desc_in_t::layout,
        mem_desc_in_t::space,
        gpu_arch::Xe>;
    using mat_in_tile_acc_t = tile_t<dtype_acc, mat_in_tile_desc_t>;
    mem_desc_in_t mem_desc_in(args.base, args.shape, coord);
    mat_in_tile_t mat_in;
    mat_in_payload_t mat_in_payload(mem_desc_in);
    tile_load<cache_hint::cached, cache_hint::cached>(mat_in, mat_in_payload);
    mat_in_tile_acc_t mat_in_acc;
    elemwise_cvt(mat_in_acc, mat_in);

#pragma unroll
    for (int i = 0; i < tile_size_y / block_size_y; i++) {
#pragma unroll
      for (int j = 0; j < num_block_x; j++) {
        auto dst_reg = matAcc.reg.xetla_select<block_elems, 1>(
            (i * num_block_x + j) * block_elems);
        auto src_reg = mat_in_acc.reg.xetla_select<block_elems, 1>(
            (i * num_block_x + j) * block_elems);
        dst_reg = args.beta * src_reg + args.alpha * dst_reg;
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
        auto dst_reg = matAcc.reg.xetla_select<tail_block_elems, 1>(
            tail_start_y * tile_size_x + j * tail_block_elems);
        auto src_reg = mat_in_acc.reg.xetla_select<tail_block_elems, 1>(
            tail_start_y * tile_size_x + j * tail_block_elems);
        dst_reg = args.beta * src_reg + args.alpha * dst_reg;
      }
    }
  }
};

template <typename dtype_in_>
struct res_op_t {
  using dtype_in = dtype_in_;
  using mem_desc_in_t =
      mem_desc_t<dtype_in, mem_layout::row_major, mem_space::global>;
  using shape_t = typename mem_desc_in_t::shape_t;
  using coord_t = typename mem_desc_in_t::coord_t;
  using base_t = typename mem_desc_in_t::base_t;

  struct arguments_t {
    shape_t shape;
    base_t base;
    float x;
    inline arguments_t() = default;
    inline arguments_t(base_t base_, shape_t shape_, float x_)
        : base(base_), shape(shape_), x(x_) {}
  };
  template <typename matAcc_t>
  __XETLA_API KERNEL_FUNC void operator()(
      matAcc_t& matAcc,
      const coord_t& coord,
      const arguments_t& args,
      uint32_t slm_base = 0,
      uint32_t nbarrier_base = 0) {
    using dtype_acc = typename matAcc_t::dtype;
    static constexpr uint32_t tile_size_x = matAcc_t::tile_size_x;
    static constexpr uint32_t tile_size_y = matAcc_t::tile_size_y;
    static constexpr uint32_t block_size_x = matAcc_t::block_size_x;
    static constexpr uint32_t block_size_y = matAcc_t::block_size_y;
    static constexpr int32_t num_block_x = matAcc_t::num_block_x;
    static constexpr int32_t num_block_y = matAcc_t::num_block_y;
    static constexpr uint32_t tile_elems = matAcc_t::tile_elems;
    static constexpr uint32_t block_elems = matAcc_t::block_elems;

    using mat_in_tile_desc_t = tile_desc_t<
        tile_size_x,
        tile_size_y,
        block_size_x,
        block_size_y,
        reg_layout::tiled>;
    using mat_in_tile_t = tile_t<dtype_in, mat_in_tile_desc_t>;
    using mat_in_payload_t = mem_payload_t<
        dtype_in,
        mat_in_tile_desc_t,
        msg_type_v<mat_in_tile_desc_t, mem_desc_in_t::space>,
        mem_desc_in_t::layout,
        mem_desc_in_t::space,
        gpu_arch::Xe>;
    using mat_in_tile_acc_t = tile_t<dtype_acc, mat_in_tile_desc_t>;
    mem_desc_in_t mem_desc_in(args.base, args.shape, coord);
    mat_in_tile_t mat_in;
    mat_in_payload_t mat_in_payload(mem_desc_in);
    tile_load<cache_hint::cached, cache_hint::cached>(mat_in, mat_in_payload);
    mat_in_tile_acc_t mat_in_acc;
    elemwise_cvt(mat_in_acc, mat_in);

#pragma unroll
    for (int i = 0; i < tile_size_y / block_size_y; i++) {
#pragma unroll
      for (int j = 0; j < num_block_x; j++) {
        auto dst_reg = matAcc.reg.xetla_select<block_elems, 1>(
            (i * num_block_x + j) * block_elems);
        auto src_reg = mat_in_acc.reg.xetla_select<block_elems, 1>(
            (i * num_block_x + j) * block_elems);
        dst_reg = dst_reg + args.x * src_reg;
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
        auto dst_reg = matAcc.reg.xetla_select<tail_block_elems, 1>(
            tail_start_y * tile_size_x + j * tail_block_elems);
        auto src_reg = mat_in_acc.reg.xetla_select<tail_block_elems, 1>(
            tail_start_y * tile_size_x + j * tail_block_elems);
        dst_reg = dst_reg + args.x * src_reg;
      }
    }
  }
};

template <typename dtype_bias_>
struct bias_op_t {
  using dtype_bias = dtype_bias_;
  using mem_desc_bias_t =
      mem_desc_t<dtype_bias, mem_layout::row_major, mem_space::global>;
  using shape_t = typename mem_desc_bias_t::shape_t;
  using coord_t = typename mem_desc_bias_t::coord_t;
  using base_t = typename mem_desc_bias_t::base_t;

  struct arguments_t {
    shape_t shape;
    base_t base;
    float x;
    inline arguments_t() = default;
    inline arguments_t(base_t base_, shape_t shape_, float x_)
        : base(base_), shape(shape_), x(x_) {}
  };
  template <typename matAcc_t>
  __XETLA_API KERNEL_FUNC void operator()(
      matAcc_t& matAcc,
      const coord_t& coord,
      const arguments_t& args,
      uint32_t slm_base = 0,
      uint32_t nbarrier_base = 0) {
    using dtype_acc = typename matAcc_t::dtype;
    static constexpr uint32_t tile_size_x = matAcc_t::tile_size_x;
    static constexpr uint32_t tile_size_y = matAcc_t::tile_size_y;
    static constexpr uint32_t block_size_x = matAcc_t::block_size_x;
    static constexpr uint32_t block_size_y = matAcc_t::block_size_y;
    static constexpr int32_t num_block_x = matAcc_t::num_block_x;
    static constexpr int32_t num_block_y = matAcc_t::num_block_y;
    static constexpr uint32_t tile_elems = matAcc_t::tile_elems;
    static constexpr uint32_t block_elems = matAcc_t::block_elems;

    using bias_tile_desc_t =
        tile_desc_t<tile_size_x, 1, block_size_x, 1, reg_layout::tiled>;
    using bias_t = tile_t<dtype_bias, bias_tile_desc_t>;
    using bias_payload_t = mem_payload_t<
        dtype_bias,
        bias_tile_desc_t,
        msg_type_v<bias_tile_desc_t, mem_desc_bias_t::space>,
        mem_desc_bias_t::layout,
        mem_desc_bias_t::space,
        gpu_arch::Xe>;
    coord_t bias_coord(coord.x, 0);
    mem_desc_bias_t mem_desc_bias(args.base, args.shape, bias_coord);
    bias_t bias;
    bias_payload_t bias_payload(mem_desc_bias);
    tile_load<cache_hint::cached, cache_hint::cached>(bias, bias_payload);

#pragma unroll
    for (int i = 0; i < tile_size_y / block_size_y; i++) {
#pragma unroll
      for (int j = 0; j < num_block_x; j++) {
        auto dst_reg =
            matAcc.reg
                .xetla_select<block_elems, 1>(
                    (i * num_block_x + j) * block_elems)
                .xetla_format<dtype_acc, block_size_y, block_size_x>();
#pragma unroll
        for (int row_i = 0; row_i < block_size_y; row_i++) {
          auto src_reg =
              args.x * bias.reg.xetla_select<block_size_x, 1>(j * block_size_x);
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
        for (int row_i = 0; row_i < tail_size_y; row_i++) {
          auto src_reg =
              args.x * bias.reg.xetla_select<block_size_x, 1>(j * block_size_x);
          dst_reg.row(row_i) =
              xetla_cvt<dtype_acc, dtype_bias, block_size_x>(src_reg) +
              dst_reg.row(row_i);
        }
      }
    }
  }
};

struct silu_op_t {
  struct arguments_t {};
  template <typename matAcc_t, typename coord_t>
  __XETLA_API KERNEL_FUNC void operator()(
      matAcc_t& matAcc,
      const coord_t& coord,
      const arguments_t& args,
      uint32_t slm_base = 0,
      uint32_t nbarrier_base = 0) {
    using dtype = typename matAcc_t::dtype;
    matAcc.reg = matAcc.reg / (1.f + xetla_exp<dtype>(-1.f * matAcc.reg));
  }
};

} // namespace epilogue_impl

} // namespace xetla
} // namespace xpu
