#pragma once

#include "../../xetla_kernel_api.h"
#include "../xetla.h"
#include "epilogue_impl.h"

using namespace gpu;
namespace torch_ipex::xpu::xetla {

template <
    typename dtype_a,
    typename dtype_b,
    typename dtype_c,
    typename dtype_scale,
    uint32_t periodic_sync_interval,
    uint32_t prefetch_distance,
    uint32_t wg_tile_m,
    uint32_t wg_tile_n,
    uint32_t sg_tile_m,
    uint32_t sg_tile_n,
    uint32_t sg_tile_k,
    uint32_t dequant_s,
    DequantMode dequant_mode>
struct hgemm_mxfp4_marlin_func {
  using data_type_a = dtype_a;
  using data_type_b = dtype_b;
  using data_type_c = dtype_c;
  using data_type_scale = dtype_scale;
  using data_type_acc_in = bf16;
  using data_type_acc = float;

  static constexpr gpu_arch arch_tag = gpu_arch::XeHpc;
  using tile_shape = gpu::xetla::group::
      tile_shape_t<wg_tile_n, wg_tile_m, sg_tile_n, sg_tile_m>;
  //   static constexpr uint32_t periodic_sync_interval = 1;
  //   static constexpr uint32_t prefetch_distance = 3;

  using mem_desc_a_t =
      xetla::mem_desc_t<data_type_a, mem_layout::row_major, mem_space::global>;
  using mem_desc_b_t =
      xetla::mem_desc_t<data_type_b, mem_layout::row_major, mem_space::global>;
  using mem_desc_c_t =
      xetla::mem_desc_t<data_type_c, mem_layout::row_major, mem_space::global>;

  using compute_attr = gpu::xetla::group::
      compute_attr_t<data_type_acc_in, data_type_acc_in, data_type_acc>;
  using perf_tuning_knob = gpu::xetla::group::
      perf_tuning_knob_t<sg_tile_k, prefetch_distance, periodic_sync_interval>;
  using compute_policy = gpu::xetla::group::compute_policy_mxfp4_dequantize<
      compute_attr,
      perf_tuning_knob,
      data_type_scale,
      dequant_s,
      dequant_mode,
      arch_tag>;
  using gemm_t = gpu::xetla::group::
      gemm_t<compute_policy, tile_shape, mem_desc_a_t, mem_desc_b_t>;

  using epilogue_t = gpu::xetla::group::epilogue_t<
      gpu::xetla::group::epilogue_policy_default<arch_tag>,
      tile_shape,
      mem_desc_c_t>;
  using group_swizzle = gpu::xetla::kernel::group_swizzle_default<arch_tag>;

  using gemm_op_t = gpu::xetla::kernel::
      mxfp4_gemm_universal_t<gemm_t, epilogue_t, group_swizzle>;

  static const char* func_name() {
    return "hgemm_mxfp4_marlin_func";
  }

  template <typename gemm_op_t>
  struct RunKernelFunctor {
    KERNEL_MAIN void operator()(nd_item<3> item) const {
      slm_barrier_init<gemm_op_t>();
      gemm_op_t gemm_op;
      gemm_op(item, gemm_arg);
    }
    RunKernelFunctor(typename gemm_op_t::arguments_t gemm_arg_)
        : gemm_arg(gemm_arg_) {}

   private:
    typename gemm_op_t::arguments_t gemm_arg;
  };

  static inline cgf_t run(
      data_type_a* A,
      data_type_b* B,
      data_type_c* C,
      uint32_t matrix_m,
      uint32_t matrix_n,
      uint32_t matrix_k,
      data_type_scale* scale_ptr,
      typename epilogue_t::arguments_t epilogue_args = {}) {
    // set up gemm arguments
    typename gemm_op_t::arguments_t gemm_arg(
        matrix_m,
        matrix_k,
        matrix_n,
        A,
        matrix_k,
        B,
        matrix_n,
        C,
        matrix_n,
        scale_ptr,
        matrix_n);

    sycl::nd_range<3> NDRange = gemm_op_t::get_nd_range(gemm_arg);

    RunKernelFunctor<gemm_op_t> kfn(gemm_arg);
    return [=](sycl::handler& cgh) {
      cgh.parallel_for<decltype(kfn)>(NDRange, kfn);
    };
  }
};

} // namespace torch_ipex::xpu::xetla
