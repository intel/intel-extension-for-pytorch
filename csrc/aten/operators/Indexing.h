#pragma once
#include "ATen/ATen.h"
#include "BatchKernel.h"
#include "comm/Atomics.h"
#include "core/detail/TensorInfo.h"

using namespace at::AtenIpexTypeXPU;

template <class SrcInfo, class DstInfo, class IdxInfo, class FuncType>
class IndexKernelConfig : public BatchKernelConfig {
 public:
  using ValType = typename SrcInfo::scalar_t;
  using IdxType = typename IdxInfo::scalar_t;

  IndexKernelConfig() = delete;
  IndexKernelConfig(
      SrcInfo& sinfo,
      DstInfo& dinfo,
      IdxInfo& iinfo,
      ValType alpha,
      int64_t index_num,
      int64_t indexing_dimension_size,
      bool indexing_dst,
      FuncType func,
      int64_t batch,
      int64_t problem,
      int64_t stride,
      int64_t problem_batch,
      bool problem_along_x)
      : BatchKernelConfig(
            batch,
            problem,
            stride,
            problem_batch,
            problem_along_x),
        sinfo_(sinfo),
        dinfo_(dinfo),
        iinfo_(iinfo),
        alpha_(alpha),
        index_num_(index_num),
        indexing_dimension_size_(indexing_dimension_size),
        indexing_dst_(indexing_dst),
        func_(func) {}

  template <class TarInfo>
  static inline void indexing_problem_mapping(
      TarInfo& tinfo,
      IdxInfo& iinfo,
      int dim,
      int64_t index_num,
      int64_t indexing_dimension_size,
      int64_t& batch,
      int64_t& problem,
      int64_t& stride,
      int64_t& problem_batch,
      bool& problem_along_x) {
    int64_t outer = tinfo.outerSize(dim);
    int64_t inner = tinfo.innerSize(dim);

    if (inner == 1) {
      problem = outer;
      stride = indexing_dimension_size;
      batch = 1;
      problem_batch = index_num;
      problem_along_x = tinfo.strides[dim] == 1 ? false : true;
    } else if (outer == 1) {
      problem = inner;
      stride = 1;
      batch = indexing_dimension_size;
      problem_batch = index_num;
      problem_along_x = tinfo.strides[tinfo.dims - 1] == 1 ? true : false;
    } else {
      problem = inner;
      stride = 1;
      batch = outer * indexing_dimension_size;
      problem_batch = outer * index_num;
      problem_along_x = tinfo.strides[tinfo.dims - 1] == 1 ? true : false;
    }
    return;
  }

  static IndexKernelConfig<SrcInfo, DstInfo, IdxInfo, FuncType> make_config(
      SrcInfo& src_info,
      DstInfo& dst_info,
      IdxInfo& index_info,
      ValType alpha,
      int64_t dim,
      bool indexing_dst,
      FuncType func) {
    int64_t index_num = index_info.sizes[0];
    int64_t indexing_dimension_size;

    bool problem_along_x;
    int64_t batch, problem, stride, problem_batch;

    TORCH_CHECK(
        indexing_dst || src_info.data != nullptr,
        "Indexing kernel backbone does not support null src ...");

    if (indexing_dst) {
      indexing_dimension_size = dst_info.sizes[dim];
      indexing_problem_mapping(
          dst_info,
          index_info,
          dim,
          index_num,
          indexing_dimension_size,
          batch,
          problem,
          stride,
          problem_batch,
          problem_along_x);
    } else {
      indexing_dimension_size = src_info.sizes[dim];
      indexing_problem_mapping(
          src_info,
          index_info,
          dim,
          index_num,
          indexing_dimension_size,
          batch,
          problem,
          stride,
          problem_batch,
          problem_along_x);
    }

    return {
        src_info,
        dst_info,
        index_info,
        alpha,
        index_num,
        indexing_dimension_size,
        indexing_dst,
        func,
        batch,
        problem,
        stride,
        problem_batch,
        problem_along_x};
  }

 public:
  SrcInfo sinfo_; // sinfo_.data could be nullptr, while indexing along dst.
  DstInfo dinfo_;
  IdxInfo iinfo_;
  ValType alpha_;
  int64_t index_num_;
  int64_t indexing_dimension_size_;
  bool indexing_dst_;
  FuncType func_;
};

template <class IdxConfig>
class IndexKernel {
 public:
  using ValType = typename IdxConfig::ValType;
  using IdxType = typename IdxConfig::IdxType;

  IndexKernel() = delete;
  IndexKernel(IdxConfig& cfg) : cfg_(cfg) {}

  DPCPP_DEVICE void init_global_batch_info(
      BatchKernelConfig::ItemDesc& id,
      int64_t& idx_logical_off,
      int64_t& glb_batch_group,
      int64_t& glb_batch_group_loc_off) const {
    idx_logical_off = id.glb_batch % cfg_.index_num_;
    int64_t idx_off = IndexToOffset<IdxType, int64_t>::get(
        idx_logical_off,
        cfg_.iinfo_,
        IndexToOffset<IdxType, int64_t>::NON_STRICT_CONTIGUOUS);
    glb_batch_group = id.glb_batch / cfg_.index_num_;
    glb_batch_group_loc_off = cfg_.iinfo_.data[idx_off];
    glb_batch_group_loc_off = glb_batch_group_loc_off >= 0
        ? glb_batch_group_loc_off
        : cfg_.indexing_dimension_size_ + glb_batch_group_loc_off;
  }

  DPCPP_DEVICE int64_t indexing_logical_off(
      BatchKernelConfig::ItemDesc& id,
      int64_t glb_batch_group,
      int64_t glb_batch_group_loc_off) const {
    int64_t si, pi, bi;
    int64_t glb_batch_group_glb_off =
        glb_batch_group * cfg_.indexing_dimension_size_ +
        glb_batch_group_loc_off;
    if (cfg_.batch_ != 1) {
      si = 0;
      pi = id.chunk * id.chunk_size + id.chunk_off;
      bi = glb_batch_group_glb_off;
    } else {
      si = glb_batch_group_glb_off;
      pi = id.chunk * id.chunk_size + id.chunk_off;
      bi = 0;
    }
    return si + pi * cfg_.stride_ + bi * cfg_.problem_ * cfg_.stride_;
  }

  DPCPP_DEVICE int64_t fixing_logical_off(
      BatchKernelConfig::ItemDesc& id,
      int64_t glb_batch_group,
      int64_t idx_logical_off) const {
    int64_t si, pi, bi, stride;
    int64_t glb_batch_group_glb_off =
        glb_batch_group * cfg_.index_num_ + idx_logical_off;
    if (cfg_.batch_ != 1) {
      si = 0;
      pi = id.chunk * id.chunk_size + id.chunk_off;
      bi = glb_batch_group_glb_off;
      stride = 1;
    } else {
      si = glb_batch_group_glb_off;
      pi = id.chunk * id.chunk_size + id.chunk_off;
      bi = 0;
      stride = cfg_.index_num_;
    }
    return si + pi * stride + bi * stride * cfg_.problem_;
  }

  DPCPP_DEVICE void operator()(DPCPP::nd_item<2> item) const {
    auto id = cfg_.get_item_desc(item);

    if (id.glb_problem >= cfg_.problem_ ||
        id.glb_batch >= cfg_.problem_batch_) {
      return;
    }

    // index kernel has three operands,
    // 1. index operand
    // 2. operand indexing on
    // 3. operand has fixing size as index (optional)
    int64_t indexing_si, indexing_pi, indexing_bi;
    int64_t fixing_si, fixing_pi, fixing_bi;
    int64_t idx_logical_off, glb_batch_group, glb_batch_group_loc_off;
    int64_t glb_indexing_logical_off, glb_fixing_logical_off;
    int64_t glb_indexing_off, glb_fixing_off;
    int64_t dst_off, src_off;

    init_global_batch_info(
        id, idx_logical_off, glb_batch_group, glb_batch_group_loc_off);

    glb_indexing_logical_off =
        indexing_logical_off(id, glb_batch_group, glb_batch_group_loc_off);

    if (cfg_.sinfo_.data != nullptr && cfg_.sinfo_.data != nullptr) {
      glb_fixing_logical_off =
          fixing_logical_off(id, glb_batch_group, idx_logical_off);
    }

    if (cfg_.indexing_dst_) {
      // index_copy, index_add, index_fill
      dst_off = IndexToOffset<ValType, int64_t>::get(
          glb_indexing_logical_off,
          cfg_.dinfo_,
          IndexToOffset<ValType, int64_t>::NON_STRICT_CONTIGUOUS);
      if (cfg_.sinfo_.data != nullptr) {
        src_off = IndexToOffset<ValType, int64_t>::get(
            glb_fixing_logical_off,
            cfg_.sinfo_,
            IndexToOffset<ValType, int64_t>::NON_STRICT_CONTIGUOUS);
      }
    } else {
      // index_select
      src_off = IndexToOffset<ValType, int64_t>::get(
          glb_indexing_logical_off,
          cfg_.sinfo_,
          IndexToOffset<ValType, int64_t>::NON_STRICT_CONTIGUOUS);
      dst_off = IndexToOffset<ValType, int64_t>::get(
          glb_fixing_logical_off,
          cfg_.dinfo_,
          IndexToOffset<ValType, int64_t>::NON_STRICT_CONTIGUOUS);
    }

    cfg_.func_(
        cfg_.dinfo_.data, cfg_.sinfo_.data, dst_off, src_off, cfg_.alpha_);
  }

 private:
  IdxConfig cfg_;
};

template <class IdxConfig>
static inline void launch_index_kernel(IdxConfig& cfg) {
  auto& queue = dpcppGetCurrentQueue();
  auto cgf = DPCPP_Q_CGF(__cgh) {
    IndexKernel<IdxConfig> idx_ker(cfg);
    __cgh.parallel_for(
        DPCPP::nd_range<2>(cfg.global_size(), cfg.group_size()), idx_ker);
  };
  DPCPP_Q_SUBMIT(queue, cgf);
}

template <typename ValType>
class IndexFillOperator {
 public:
  DPCPP_DEVICE void operator()(
      ValType* dst,
      ValType* src,
      int64_t dst_off,
      int64_t src_off,
      ValType alpha) const {
    dst[dst_off] = alpha;
  }
};

template <class ValInfo, class IdxInfo>
static inline void _index_fill_kernel(
    ValInfo& self_info,
    IdxInfo& index_info,
    int64_t dim,
    typename ValInfo::scalar_t val) {
  using scalar_t = typename ValInfo::scalar_t;
  using DstInfo = ValInfo;
  using SrcInfo = ValInfo;
  auto src_info = SrcInfo();
  auto cfg = IndexKernelConfig<
      SrcInfo,
      DstInfo,
      IdxInfo,
      IndexFillOperator<scalar_t>>::
      make_config(
          src_info,
          self_info,
          index_info,
          val,
          dim,
          true,
          IndexFillOperator<scalar_t>());
  launch_index_kernel(cfg);
}

template <typename ValType>
class IndexCopyOperator {
 public:
  DPCPP_DEVICE void operator()(
      ValType* dst,
      ValType* src,
      int64_t dst_off,
      int64_t src_off,
      ValType alpha) const {
    dst[dst_off] = src[src_off];
  }
};

template <class SrcInfo, class DstInfo, class IdxInfo>
static inline void _index_copy_kernel(
    SrcInfo& src_info,
    DstInfo& dst_info,
    IdxInfo& index_info,
    int64_t dim) {
  using scalar_t = typename SrcInfo::scalar_t;
  auto cfg = IndexKernelConfig<
      SrcInfo,
      DstInfo,
      IdxInfo,
      IndexCopyOperator<scalar_t>>::
      make_config(
          src_info,
          dst_info,
          index_info,
          0,
          dim,
          true,
          IndexCopyOperator<scalar_t>());
  launch_index_kernel(cfg);
}

template <typename ValType>
class IndexAddOperator {
 public:
  DPCPP_DEVICE void operator()(
      ValType* dst,
      ValType* src,
      int64_t dst_off,
      int64_t src_off,
      ValType alpha) const {
    atomicAdd(
        (dpcpp_global_ptr_pt<ValType>)(dst + dst_off), src[src_off] * alpha);
  }
};

template <class SrcInfo, class DstInfo, class IdxInfo>
static inline void _index_add_kernel(
    SrcInfo& src_info,
    DstInfo& dst_info,
    IdxInfo& index_info,
    typename SrcInfo::scalar_t alpha,
    int64_t dim) {
  using scalar_t = typename SrcInfo::scalar_t;
  auto cfg =
      IndexKernelConfig<SrcInfo, DstInfo, IdxInfo, IndexAddOperator<scalar_t>>::
          make_config(
              src_info,
              dst_info,
              index_info,
              alpha,
              dim,
              true,
              IndexAddOperator<scalar_t>());
  launch_index_kernel(cfg);
}

template <typename ValType>
class IndexSelectOperator {
 public:
  DPCPP_DEVICE void operator()(
      ValType* dst,
      ValType* src,
      int64_t dst_off,
      int64_t src_off,
      ValType alpha) const {
    dst[dst_off] = src[src_off];
  }
};

template <class SrcInfo, class DstInfo, class IdxInfo>
static inline void _index_select_kernel(
    SrcInfo& src_info,
    DstInfo& dst_info,
    IdxInfo& index_info,
    int64_t dim) {
  using scalar_t = typename SrcInfo::scalar_t;
  auto cfg = IndexKernelConfig<
      SrcInfo,
      DstInfo,
      IdxInfo,
      IndexSelectOperator<scalar_t>>::
      make_config(
          src_info,
          dst_info,
          index_info,
          0,
          dim,
          false,
          IndexSelectOperator<scalar_t>());
  launch_index_kernel(cfg);
}
