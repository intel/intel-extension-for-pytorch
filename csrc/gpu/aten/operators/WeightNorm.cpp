#include <ATen/ATen.h>
#include <oneDNN/oneDNN.h>
#include <cmath>
#include <oneapi/dpl/tuple>
#include "BatchKernel.h"
#include "Loops.h"
#include "Reduce.h"
#include "Resize.h"
#include "comm/AccumulateType.h"
#include "comm/RegistrationDeclarations.h"
#include "comm/TensorOptions.h"

using namespace xpu::dpcpp::detail;

namespace at {
namespace AtenIpexTypeXPU {

namespace {
template <typename T>
struct ReduceAdd {
  T operator()(const T a, const T b) const {
    return a + b;
  }
};

template <class ScalarTypeInfo, class AccTypeInfo>
static inline void launch_weight_norm_reduce_kernel_(
    ScalarTypeInfo& iinfo,
    AccTypeInfo& oinfo,
    BatchKernelConfig& cfg,
    bool need_squre,
    bool is_final) {
  using scalar_t = typename ScalarTypeInfo::scalar_t;
  using accscalar_t = typename AccTypeInfo::scalar_t;
  using vec_t = at::detail::Array<accscalar_t, 1>;
  auto cgf = DPCPP_Q_CGF(__cgh) {
    dpcpp_local_acc_t<accscalar_t> shared(cfg.group_size().size(), __cgh);
    auto kfn = DPCPP_Q_KFN(sycl::nd_item<2> item) {
      auto id = cfg.get_item_desc(item);

      int64_t si, bi, ldr_pi, str_pi, ldr_lid, ldr_off, str_lid, str_off;
      si = id.glb_batch % cfg.stride_;
      bi = id.glb_batch / cfg.stride_;
      ldr_pi = id.chunk * id.chunk_size + id.chunk_off;
      str_pi = id.chunk;
      ldr_lid = si + ldr_pi * cfg.stride_ + bi * cfg.problem_ * cfg.stride_;
      ldr_off = IndexToOffset<scalar_t, int64_t>::get(
          ldr_lid,
          iinfo,
          IndexToOffset<scalar_t, int64_t>::NON_STRICT_CONTIGUOUS);
      str_lid = si + str_pi * cfg.stride_ + bi * id.chunk_num * cfg.stride_;
      str_off = IndexToOffset<accscalar_t, int64_t>::get(
          str_lid,
          oinfo,
          IndexToOffset<accscalar_t, int64_t>::NON_STRICT_CONTIGUOUS);

      accscalar_t value = 0;
      if (id.glb_problem < cfg.problem_ && id.glb_batch < cfg.problem_batch_) {
        value = (accscalar_t)iinfo.data[ldr_off];
        if (need_squre)
          value *= value;
      }

      if (cfg.problem_along_x_) {
        value = group_x_reduce(
            item, shared, vec_t(value), ReduceAdd<accscalar_t>())[0];
      } else {
        value = group_y_reduce(
            item, shared, vec_t(value), ReduceAdd<accscalar_t>())[0];
      }

      if (id.glb_problem < cfg.problem_ && id.glb_batch < cfg.problem_batch_) {
        if (id.chunk_off == 0) {
          oinfo.data[str_off] = is_final ? sqrtf(value) : value;
        }
      }
    };
    __cgh.parallel_for(
        cl::sycl::nd_range<2>(cfg.global_size(), cfg.group_size()), kfn);
  };
  DPCPP_Q_SUBMIT(dpcppGetCurrentQueue(), cgf);
}

template <class ScalarTypeInfo, class AccTypeInfo>
static inline void weight_norm_reduce_(
    ScalarTypeInfo& vinfo,
    AccTypeInfo& ninfo,
    int dim_after_collapse,
    bool need_square) {
  int64_t batch = vinfo.outerSize(dim_after_collapse);
  int64_t problem = vinfo.sizes[dim_after_collapse];
  int64_t stride = vinfo.innerSize(dim_after_collapse);
  bool problem_along_x = vinfo.strides[dim_after_collapse] == 1 ? true : false;

  BatchKernelConfig cfg = {
      batch, problem, stride, batch * stride, problem_along_x};

  if (cfg.problem_ <= cfg.problem_wg_range_) {
    launch_weight_norm_reduce_kernel_(vinfo, ninfo, cfg, need_square, true);
    return;
  }

  Tensor carrier = at::empty(
      {cfg.batch_, cfg.problem_glb_range_ / cfg.problem_wg_range_, cfg.stride_},
      map_options<typename AccTypeInfo::scalar_t>());
  auto cinfo = getTensorInfo<typename AccTypeInfo::scalar_t, int64_t>(carrier);
  launch_weight_norm_reduce_kernel_(vinfo, cinfo, cfg, need_square, false);

  weight_norm_reduce_(cinfo, ninfo, 1, false);
  return;
}

template <class ScalarTypeInfo, class AccTypeInfo>
static inline void segment_weight_norm_(
    ScalarTypeInfo& vinfo,
    ScalarTypeInfo& ginfo,
    ScalarTypeInfo& winfo,
    AccTypeInfo& ninfo,
    int dim_after_collapse) {
  // segment reduce for statistics
  weight_norm_reduce_(vinfo, ninfo, dim_after_collapse, true);

  // normalization
  int64_t batch = vinfo.outerSize(dim_after_collapse);
  int64_t problem = vinfo.sizes[dim_after_collapse];
  int64_t stride = vinfo.innerSize(dim_after_collapse);
  bool problem_along_x = vinfo.strides[dim_after_collapse] == 1 ? true : false;

  BatchKernelConfig cfg = {
      batch, problem, stride, batch * stride, problem_along_x};

  using scalar_t = typename ScalarTypeInfo::scalar_t;
  using accscalar_t = typename AccTypeInfo::scalar_t;
  auto cgf = DPCPP_Q_CGF(__cgh) {
    auto kfn = DPCPP_Q_KFN(sycl::nd_item<2> item) {
      auto id = cfg.get_item_desc(item);

      int64_t si, bi, pi, w_lid, v_off, w_off, n_lid, n_off, g_off;
      si = id.glb_batch % cfg.stride_;
      bi = id.glb_batch / cfg.stride_;
      pi = id.chunk * id.chunk_size + id.chunk_off;

      w_lid = si + pi * cfg.stride_ + bi * cfg.problem_ * cfg.stride_;
      n_lid = id.glb_batch;

      v_off = IndexToOffset<scalar_t, int64_t>::get(
          w_lid,
          vinfo,
          IndexToOffset<scalar_t, int64_t>::NON_STRICT_CONTIGUOUS);

      w_off = IndexToOffset<scalar_t, int64_t>::get(
          w_lid,
          winfo,
          IndexToOffset<scalar_t, int64_t>::NON_STRICT_CONTIGUOUS);

      g_off = IndexToOffset<scalar_t, int64_t>::get(
          n_lid,
          ginfo,
          IndexToOffset<scalar_t, int64_t>::NON_STRICT_CONTIGUOUS);

      n_off = IndexToOffset<accscalar_t, int64_t>::get(
          n_lid,
          ninfo,
          IndexToOffset<accscalar_t, int64_t>::NON_STRICT_CONTIGUOUS);

      if (id.glb_problem < cfg.problem_ && id.glb_batch < cfg.problem_batch_) {
        winfo.data[w_off] =
            (1.f / ninfo.data[n_off]) * vinfo.data[v_off] * ginfo.data[g_off];
      }
    };
    __cgh.parallel_for(
        cl::sycl::nd_range<2>(cfg.global_size(), cfg.group_size()), kfn);
  };
  DPCPP_Q_SUBMIT(dpcppGetCurrentQueue(), cgf);

  return;
}

template <class ScalarTypeInfo, class AccTypeInfo>
static inline void weight_norm_(
    ScalarTypeInfo& vinfo,
    ScalarTypeInfo& ginfo,
    ScalarTypeInfo& winfo,
    AccTypeInfo& ninfo,
    int dim_after_collapse) {
  int64_t batch = vinfo.outerSize(dim_after_collapse);
  int64_t problem = vinfo.sizes[dim_after_collapse];
  int64_t stride = vinfo.innerSize(dim_after_collapse);
  bool problem_along_x = vinfo.strides[dim_after_collapse] == 1 ? true : false;

  BatchKernelConfig cfg = {
      batch,
      problem,
      stride,
      batch * stride,
      problem_along_x,
      BatchKernelConfig::Policy::pLoop};

  using scalar_t = typename ScalarTypeInfo::scalar_t;
  using accscalar_t = typename AccTypeInfo::scalar_t;
  using vec_t = at::detail::Array<accscalar_t, 1>;
  auto cgf = DPCPP_Q_CGF(__cgh) {
    int wg_size = cfg.group_size().size();
    int batch_wg_range = wg_size / cfg.problem_wg_range_;
    dpcpp_local_acc_t<accscalar_t> shared(wg_size, __cgh);
    auto kfn = DPCPP_Q_KFN(sycl::nd_item<2> item) {
      auto id = cfg.get_item_desc(item);

      int64_t n_lid, n_off, g_off;
      n_lid = id.glb_batch;

      g_off = IndexToOffset<scalar_t, int64_t>::get(
          n_lid,
          ginfo,
          IndexToOffset<scalar_t, int64_t>::NON_STRICT_CONTIGUOUS);

      n_off = IndexToOffset<accscalar_t, int64_t>::get(
          n_lid,
          ninfo,
          IndexToOffset<accscalar_t, int64_t>::NON_STRICT_CONTIGUOUS);

      int64_t si = id.glb_batch % cfg.stride_;
      int64_t bi = id.glb_batch / cfg.stride_;
      int64_t pi = id.chunk_off;
      bi = si + bi * cfg.problem_ * cfg.stride_;

      accscalar_t value = 0;
      if (id.glb_batch < cfg.problem_batch_) {
        for (int pi_ = pi; pi_ < cfg.problem_; pi_ += cfg.problem_wg_range_) {
          int64_t v_lid, v_off;
          v_lid = bi + pi_ * cfg.stride_;

          v_off = IndexToOffset<scalar_t, int64_t>::get(
              v_lid,
              vinfo,
              IndexToOffset<scalar_t, int64_t>::NON_STRICT_CONTIGUOUS);

          accscalar_t v = (accscalar_t)vinfo.data[v_off];
          value += v * v;
        }
      }

      if (cfg.problem_along_x_) {
        value = group_x_reduce(
            item, shared, vec_t(value), ReduceAdd<accscalar_t>())[0];
      } else {
        value = group_y_reduce(
            item, shared, vec_t(value), ReduceAdd<accscalar_t>())[0];
      }

      int n_slid = (int)id.glb_batch % batch_wg_range;
      if (id.glb_batch < cfg.problem_batch_ && id.chunk_off == 0) {
        value = sqrtf(value);
        ninfo.data[n_off] = value;
        shared[n_slid] = value;
      }
      // Here using slm instead. If using ugm, need fence w/
      // order:acq_rel & scope:workgroup & space:global_mem.
      item.barrier(dpcpp_local_fence);

      if (id.glb_batch < cfg.problem_batch_) {
        for (int pi_ = pi; pi_ < cfg.problem_; pi_ += cfg.problem_wg_range_) {
          int64_t v_lid, v_off, w_off;
          v_lid = bi + pi_ * cfg.stride_;

          v_off = IndexToOffset<scalar_t, int64_t>::get(
              v_lid,
              vinfo,
              IndexToOffset<scalar_t, int64_t>::NON_STRICT_CONTIGUOUS);

          w_off = IndexToOffset<scalar_t, int64_t>::get(
              v_lid,
              winfo,
              IndexToOffset<scalar_t, int64_t>::NON_STRICT_CONTIGUOUS);

          winfo.data[w_off] =
              (1.f / shared[n_slid]) * vinfo.data[v_off] * ginfo.data[g_off];
        }
      }
    };
    __cgh.parallel_for(
        cl::sycl::nd_range<2>(cfg.global_size(), cfg.group_size()), kfn);
  };
  DPCPP_Q_SUBMIT(dpcppGetCurrentQueue(), cgf);

  return;
}

template <
    bool is_first,
    class ScalarType1Info,
    class ScalarType2Info,
    class AccTypeInfo>
static inline void launch_weight_norm_backward_reduce_kernel_(
    ScalarType1Info& i1info,
    ScalarType2Info& i2info,
    AccTypeInfo& oinfo,
    BatchKernelConfig& cfg) {
  using scalar1_t = typename ScalarType1Info::scalar_t;
  using scalar2_t = typename ScalarType2Info::scalar_t;
  using accscalar_t = typename AccTypeInfo::scalar_t;
  using vec_t = at::detail::Array<accscalar_t, 1>;
  auto cgf = DPCPP_Q_CGF(__cgh) {
    dpcpp_local_acc_t<accscalar_t> shared(cfg.group_size().size(), __cgh);
    auto kfn = DPCPP_Q_KFN(sycl::nd_item<2> item) {
      auto id = cfg.get_item_desc(item);

      int64_t si, bi, i_pi, o_pi, i_lid, i1_off, i2_off, o_lid, o_off;
      si = id.glb_batch % cfg.stride_;
      bi = id.glb_batch / cfg.stride_;
      i_pi = id.chunk * id.chunk_size + id.chunk_off;
      o_pi = id.chunk;

      i_lid = si + i_pi * cfg.stride_ + bi * cfg.problem_ * cfg.stride_;
      i1_off = IndexToOffset<scalar1_t, int64_t>::get(
          i_lid,
          i1info,
          IndexToOffset<scalar1_t, int64_t>::NON_STRICT_CONTIGUOUS);

      if (is_first) {
        i2_off = IndexToOffset<scalar2_t, int64_t>::get(
            i_lid,
            i2info,
            IndexToOffset<scalar2_t, int64_t>::NON_STRICT_CONTIGUOUS);
      }

      o_lid = si + o_pi * cfg.stride_ + bi * id.chunk_num * cfg.stride_;
      o_off = IndexToOffset<accscalar_t, int64_t>::get(
          o_lid,
          oinfo,
          IndexToOffset<accscalar_t, int64_t>::NON_STRICT_CONTIGUOUS);

      accscalar_t value = 0;
      if (id.glb_problem < cfg.problem_ && id.glb_batch < cfg.problem_batch_) {
        if (is_first) {
          auto value1 = (accscalar_t)i1info.data[i1_off];
          auto value2 = (accscalar_t)i2info.data[i2_off];
          value = value1 * value2;
        } else {
          value = (accscalar_t)i1info.data[i1_off];
        }
      }

      if (cfg.problem_along_x_) {
        value = group_x_reduce(
            item, shared, vec_t(value), ReduceAdd<accscalar_t>())[0];
      } else {
        value = group_y_reduce(
            item, shared, vec_t(value), ReduceAdd<accscalar_t>())[0];
      }

      if (id.glb_problem < cfg.problem_ && id.glb_batch < cfg.problem_batch_) {
        if (id.chunk_off == 0) {
          oinfo.data[o_off] = value;
        }
      }
    };
    __cgh.parallel_for(
        cl::sycl::nd_range<2>(cfg.global_size(), cfg.group_size()), kfn);
  };
  DPCPP_Q_SUBMIT(dpcppGetCurrentQueue(), cgf);
}

template <class ScalarType1Info, class ScalarType2Info, class AccTypeInfo>
static inline void weight_norm_backward_reduce_(
    ScalarType1Info& vinfo,
    ScalarType2Info& gwinfo,
    AccTypeInfo& rinfo,
    int dim_after_collapse,
    bool is_first) {
  int64_t batch = vinfo.outerSize(dim_after_collapse);
  int64_t problem = vinfo.sizes[dim_after_collapse];
  int64_t stride = vinfo.innerSize(dim_after_collapse);
  bool problem_along_x = vinfo.strides[dim_after_collapse] == 1 ? true : false;

  BatchKernelConfig cfg = {
      batch, problem, stride, batch * stride, problem_along_x};

  if (cfg.problem_ <= cfg.problem_wg_range_) {
    if (is_first)
      launch_weight_norm_backward_reduce_kernel_<true>(
          vinfo, gwinfo, rinfo, cfg);
    else
      launch_weight_norm_backward_reduce_kernel_<false>(
          vinfo, gwinfo, rinfo, cfg);
    return;
  }

  Tensor carrier = at::empty(
      {cfg.batch_, cfg.problem_glb_range_ / cfg.problem_wg_range_, cfg.stride_},
      map_options<typename AccTypeInfo::scalar_t>());
  auto cinfo = getTensorInfo<typename AccTypeInfo::scalar_t, int64_t>(carrier);
  if (is_first)
    launch_weight_norm_backward_reduce_kernel_<true>(vinfo, gwinfo, cinfo, cfg);
  else
    launch_weight_norm_backward_reduce_kernel_<false>(
        vinfo, gwinfo, cinfo, cfg);

  weight_norm_backward_reduce_(cinfo, gwinfo, rinfo, 1, false);
  return;
}

template <class ScalarTypeInfo, class AccTypeInfo>
static inline void segment_weight_norm_backward_(
    ScalarTypeInfo& vinfo,
    ScalarTypeInfo& ginfo,
    ScalarTypeInfo& gwinfo,
    AccTypeInfo& ninfo,
    ScalarTypeInfo& gvinfo,
    ScalarTypeInfo& gginfo,
    AccTypeInfo& rinfo,
    int dim_after_collapse) {
  // segment reduce
  weight_norm_backward_reduce_(vinfo, gwinfo, rinfo, dim_after_collapse, true);

  // compute gradient
  int64_t batch = vinfo.outerSize(dim_after_collapse);
  int64_t problem = vinfo.sizes[dim_after_collapse];
  int64_t stride = vinfo.innerSize(dim_after_collapse);
  bool problem_along_x = vinfo.strides[dim_after_collapse] == 1 ? true : false;

  BatchKernelConfig cfg = {
      batch, problem, stride, batch * stride, problem_along_x};

  using scalar_t = typename ScalarTypeInfo::scalar_t;
  using accscalar_t = typename AccTypeInfo::scalar_t;
  auto cgf = DPCPP_Q_CGF(__cgh) {
    auto kfn = DPCPP_Q_KFN(sycl::nd_item<2> item) {
      auto id = cfg.get_item_desc(item);

      int64_t si, bi, pi, gv_lid, gv_off, gw_off, v_off, gg_lid, g_off, n_off,
          gg_off, r_off;
      si = id.glb_batch % cfg.stride_;
      bi = id.glb_batch / cfg.stride_;
      pi = id.chunk * id.chunk_size + id.chunk_off;

      gv_lid = si + pi * cfg.stride_ + bi * cfg.problem_ * cfg.stride_;
      gg_lid = id.glb_batch;

      v_off = IndexToOffset<scalar_t, int64_t>::get(
          gv_lid,
          vinfo,
          IndexToOffset<scalar_t, int64_t>::NON_STRICT_CONTIGUOUS);

      gw_off = IndexToOffset<scalar_t, int64_t>::get(
          gv_lid,
          gwinfo,
          IndexToOffset<scalar_t, int64_t>::NON_STRICT_CONTIGUOUS);

      gv_off = IndexToOffset<scalar_t, int64_t>::get(
          gv_lid,
          gvinfo,
          IndexToOffset<scalar_t, int64_t>::NON_STRICT_CONTIGUOUS);

      g_off = IndexToOffset<scalar_t, int64_t>::get(
          gg_lid,
          ginfo,
          IndexToOffset<scalar_t, int64_t>::NON_STRICT_CONTIGUOUS);

      n_off = IndexToOffset<accscalar_t, int64_t>::get(
          gg_lid,
          ninfo,
          IndexToOffset<accscalar_t, int64_t>::NON_STRICT_CONTIGUOUS);

      r_off = IndexToOffset<accscalar_t, int64_t>::get(
          gg_lid,
          rinfo,
          IndexToOffset<accscalar_t, int64_t>::NON_STRICT_CONTIGUOUS);

      gg_off = IndexToOffset<scalar_t, int64_t>::get(
          gg_lid,
          gginfo,
          IndexToOffset<scalar_t, int64_t>::NON_STRICT_CONTIGUOUS);

      if (id.glb_problem < cfg.problem_ && id.glb_batch < cfg.problem_batch_) {
        accscalar_t g = ginfo.data[g_off];
        accscalar_t gw = gwinfo.data[gw_off];
        accscalar_t v = vinfo.data[v_off];
        accscalar_t n = 1.f / ninfo.data[n_off];
        accscalar_t r = rinfo.data[r_off];
        accscalar_t gg = r * n;
        accscalar_t n3 = n * n * n;
        accscalar_t gv = g * (n * gw - n3 * v * r);

        gvinfo.data[gv_off] = static_cast<scalar_t>(gv);
        if (id.chunk == 0 && id.chunk_off == 0)
          gginfo.data[gg_off] = static_cast<scalar_t>(gg);
      }
    };
    __cgh.parallel_for(
        cl::sycl::nd_range<2>(cfg.global_size(), cfg.group_size()), kfn);
  };
  DPCPP_Q_SUBMIT(dpcppGetCurrentQueue(), cgf);

  return;
}

template <class ScalarTypeInfo, class AccTypeInfo>
static inline void weight_norm_backward_(
    ScalarTypeInfo& vinfo,
    ScalarTypeInfo& ginfo,
    ScalarTypeInfo& gwinfo,
    AccTypeInfo& ninfo,
    ScalarTypeInfo& gvinfo,
    ScalarTypeInfo& gginfo,
    int dim_after_collapse) {
  int64_t batch = vinfo.outerSize(dim_after_collapse);
  int64_t problem = vinfo.sizes[dim_after_collapse];
  int64_t stride = vinfo.innerSize(dim_after_collapse);
  bool problem_along_x = vinfo.strides[dim_after_collapse] == 1 ? true : false;

  BatchKernelConfig cfg = {
      batch,
      problem,
      stride,
      batch * stride,
      problem_along_x,
      BatchKernelConfig::Policy::pLoop};

  using scalar_t = typename ScalarTypeInfo::scalar_t;
  using accscalar_t = typename AccTypeInfo::scalar_t;
  using vec_t = at::detail::Array<accscalar_t, 1>;
  auto cgf = DPCPP_Q_CGF(__cgh) {
    int wg_size = cfg.group_size().size();
    int batch_wg_range = wg_size / cfg.problem_wg_range_;
    dpcpp_local_acc_t<accscalar_t> shared(wg_size, __cgh);
    auto kfn = DPCPP_Q_KFN(sycl::nd_item<2> item) {
      auto id = cfg.get_item_desc(item);

      int64_t n_lid, n_off, g_off, gg_off;
      n_lid = id.glb_batch;

      g_off = IndexToOffset<scalar_t, int64_t>::get(
          n_lid,
          ginfo,
          IndexToOffset<scalar_t, int64_t>::NON_STRICT_CONTIGUOUS);

      gg_off = IndexToOffset<scalar_t, int64_t>::get(
          n_lid,
          gginfo,
          IndexToOffset<scalar_t, int64_t>::NON_STRICT_CONTIGUOUS);

      n_off = IndexToOffset<accscalar_t, int64_t>::get(
          n_lid,
          ninfo,
          IndexToOffset<accscalar_t, int64_t>::NON_STRICT_CONTIGUOUS);

      int64_t si = id.glb_batch % cfg.stride_;
      int64_t bi = id.glb_batch / cfg.stride_;
      int64_t pi = id.chunk_off;
      bi = si + bi * cfg.problem_ * cfg.stride_;

      accscalar_t value = 0;
      if (id.glb_batch < cfg.problem_batch_) {
        for (int pi_ = pi; pi_ < cfg.problem_; pi_ += cfg.problem_wg_range_) {
          int64_t v_lid, v_off, gw_off;
          v_lid = bi + pi_ * cfg.stride_;

          v_off = IndexToOffset<scalar_t, int64_t>::get(
              v_lid,
              vinfo,
              IndexToOffset<scalar_t, int64_t>::NON_STRICT_CONTIGUOUS);

          gw_off = IndexToOffset<scalar_t, int64_t>::get(
              v_lid,
              gwinfo,
              IndexToOffset<scalar_t, int64_t>::NON_STRICT_CONTIGUOUS);

          accscalar_t v = (accscalar_t)vinfo.data[v_off];
          accscalar_t gw = (accscalar_t)gwinfo.data[gw_off];
          value += v * gw;
        }
      }

      if (cfg.problem_along_x_) {
        value = group_x_reduce(
            item, shared, vec_t(value), ReduceAdd<accscalar_t>())[0];
      } else {
        value = group_y_reduce(
            item, shared, vec_t(value), ReduceAdd<accscalar_t>())[0];
      }

      int n_slid = (int)id.glb_batch % batch_wg_range;
      if (id.glb_batch < cfg.problem_batch_ && id.chunk_off == 0) {
        shared[n_slid] = value;
      }
      item.barrier(dpcpp_local_fence);

      if (id.glb_batch < cfg.problem_batch_) {
        for (int pi_ = pi; pi_ < cfg.problem_; pi_ += cfg.problem_wg_range_) {
          int64_t v_lid, v_off, gw_off, gv_off;
          v_lid = bi + pi_ * cfg.stride_;

          v_off = IndexToOffset<scalar_t, int64_t>::get(
              v_lid,
              vinfo,
              IndexToOffset<scalar_t, int64_t>::NON_STRICT_CONTIGUOUS);

          gw_off = IndexToOffset<scalar_t, int64_t>::get(
              v_lid,
              gwinfo,
              IndexToOffset<scalar_t, int64_t>::NON_STRICT_CONTIGUOUS);

          gv_off = IndexToOffset<scalar_t, int64_t>::get(
              v_lid,
              gvinfo,
              IndexToOffset<scalar_t, int64_t>::NON_STRICT_CONTIGUOUS);

          accscalar_t g = ginfo.data[g_off];
          accscalar_t gw = gwinfo.data[gw_off];
          accscalar_t v = vinfo.data[v_off];
          accscalar_t n = 1.f / ninfo.data[n_off];
          accscalar_t r = shared[n_slid];
          accscalar_t gg = r * n;
          accscalar_t n3 = n * n * n;
          accscalar_t gv = g * (n * gw - n3 * v * r);

          gvinfo.data[gv_off] = static_cast<scalar_t>(gv);
          if (id.chunk_off == 0)
            gginfo.data[gg_off] = static_cast<scalar_t>(gg);
        }
      }
    };
    __cgh.parallel_for(
        cl::sycl::nd_range<2>(cfg.global_size(), cfg.group_size()), kfn);
  };
  DPCPP_Q_SUBMIT(dpcppGetCurrentQueue(), cgf);

  return;
}

} // namespace

std::tuple<Tensor, Tensor> _weight_norm_interface(
    const Tensor& v,
    const Tensor& g,
    int64_t dim) {
  TORCH_INTERNAL_ASSERT(
      dim == 0 || dim == v.dim() - 1,
      "fused kernels can only be applied for first or last dim");

  at::ScalarType scalar_acc_t = g.scalar_type() == at::ScalarType::Half
      ? at::ScalarType::Float
      : g.scalar_type();
  auto norms = at::empty(
      g.sizes(), g.options().dtype(scalar_acc_t), g.suggest_memory_format());
  auto w = at::empty(v.sizes(), v.options(), v.suggest_memory_format());

  IPEX_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      v.scalar_type(),
      "aten::weight_norm",
      [&] {
        auto vinfo = getTensorInfo<scalar_t, int64_t>(v);
        int dim_after_collapse = vinfo.collapseDims(dim);

        auto ginfo = getTensorInfo<scalar_t, int64_t>(g);
        ginfo.collapseDims();

        auto winfo = getTensorInfo<scalar_t, int64_t>(w);
        winfo.collapseDims(dim);

        auto ninfo =
            getTensorInfo<AccumulateType<scalar_t>::type, int64_t>(norms);
        ninfo.collapseDims();

        dim_after_collapse = 1 - dim_after_collapse; // remain dim

        int64_t batch = vinfo.outerSize(dim_after_collapse);
        int64_t problem = vinfo.sizes[dim_after_collapse];
        int64_t stride = vinfo.innerSize(dim_after_collapse);
        bool problem_along_x =
            vinfo.strides[dim_after_collapse] == 1 ? true : false;
        if (BatchKernelConfig::Policy::pSegment ==
            BatchKernelConfig::suggest_policy(
                batch, problem, stride, problem_along_x)) {
          segment_weight_norm_(vinfo, ginfo, winfo, ninfo, dim_after_collapse);
        } else {
          weight_norm_(vinfo, ginfo, winfo, ninfo, dim_after_collapse);
        }
      });

  return {w, norms};
}

std::tuple<Tensor, Tensor> _weight_norm_interface_backward(
    const Tensor& grad_w,
    const Tensor& saved_v,
    const Tensor& saved_g,
    const Tensor& saved_norms,
    int64_t dim) {
  TORCH_CHECK(saved_v.is_contiguous(), "saved_v must be contiguous");
  TORCH_CHECK(saved_g.is_contiguous(), "saved_g must be contiguous");
  TORCH_CHECK(saved_norms.is_contiguous(), "saved_norms must be contiguous");
  TORCH_CHECK(
      dim == 0 || dim == saved_v.dim() - 1,
      "fused kernels can only be applied for first or last dim")
  auto grad_v = at::empty_like(saved_v, c10::get_contiguous_memory_format());
  auto grad_g = at::empty_like(saved_g, c10::get_contiguous_memory_format());

  IPEX_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      saved_v.scalar_type(),
      "aten::weight_norm",
      [&] {
        auto vinfo = getTensorInfo<scalar_t, int64_t>(saved_v);
        int dim_after_collapse = vinfo.collapseDims(dim);

        auto ginfo = getTensorInfo<scalar_t, int64_t>(saved_g);
        ginfo.collapseDims();

        auto gwinfo = getTensorInfo<scalar_t, int64_t>(grad_w);
        gwinfo.collapseDims(dim);

        auto ninfo =
            getTensorInfo<AccumulateType<scalar_t>::type, int64_t>(saved_norms);
        ninfo.collapseDims();

        auto gvinfo = getTensorInfo<scalar_t, int64_t>(grad_v);
        gvinfo.collapseDims(dim);

        auto gginfo = getTensorInfo<scalar_t, int64_t>(grad_g);
        gginfo.collapseDims();

        dim_after_collapse = 1 - dim_after_collapse; // remain dim

        int64_t batch = vinfo.outerSize(dim_after_collapse);
        int64_t problem = vinfo.sizes[dim_after_collapse];
        int64_t stride = vinfo.innerSize(dim_after_collapse);
        bool problem_along_x =
            vinfo.strides[dim_after_collapse] == 1 ? true : false;
        if (BatchKernelConfig::Policy::pSegment ==
            BatchKernelConfig::suggest_policy(
                batch, problem, stride, problem_along_x)) {
          auto reduce =
              at::empty_like(saved_g, c10::get_contiguous_memory_format());
          auto rinfo =
              getTensorInfo<AccumulateType<scalar_t>::type, int64_t>(reduce);
          rinfo.collapseDims();

          segment_weight_norm_backward_(
              vinfo,
              ginfo,
              gwinfo,
              ninfo,
              gvinfo,
              gginfo,
              rinfo,
              dim_after_collapse);
        } else {
          weight_norm_backward_(
              vinfo, ginfo, gwinfo, ninfo, gvinfo, gginfo, dim_after_collapse);
        }
      });

  return {grad_v, grad_g};
}

} // namespace AtenIpexTypeXPU
} // namespace at
