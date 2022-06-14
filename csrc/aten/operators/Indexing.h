#pragma once
#include "ATen/ATen.h"
#include "BatchKernel.h"
#include "core/detail/TensorInfo.h"

template <typename T, class ValInfo, class IdxInfo>
class IndexFillConfig : public BatchKernelConfig {
 public:
  IndexFillConfig() = delete;
  IndexFillConfig(
      ValInfo& vinfo,
      IdxInfo& iinfo,
      int64_t index_num,
      int64_t indexing_dimension_size,
      int64_t batch,
      int64_t problem,
      int64_t stride,
      int64_t problem_batch,
      bool problem_along_x,
      T val)
      : BatchKernelConfig(
            batch,
            problem,
            stride,
            problem_batch,
            problem_along_x),
        vinfo_(vinfo),
        iinfo_(iinfo),
        index_num_(index_num),
        indexing_dimension_size_(indexing_dimension_size),
        val_(val) {}

  static IndexFillConfig<T, ValInfo, IdxInfo> make_config(
      ValInfo& self_info,
      IdxInfo& index_info,
      int64_t fill_dim,
      T val) {
    int64_t index_num = index_info.sizes[0];
    int64_t indexing_dimension_size = self_info.sizes[fill_dim];
    int64_t outer = self_info.outerSize(fill_dim);
    int64_t inner = self_info.innerSize(fill_dim);
    bool problem_along_x;

    int64_t batch, problem, stride, problem_batch;
    if (inner == 1) {
      problem = outer;
      stride = indexing_dimension_size;
      batch = 1;
      problem_batch = index_num;
      problem_along_x = self_info.strides[fill_dim] == 1 ? true : false;
    } else if (outer == 1) {
      problem = inner;
      stride = 1;
      batch = indexing_dimension_size;
      problem_batch = index_num;
      problem_along_x =
          self_info.strides[self_info.dims - 1] == 1 ? true : false;
    } else {
      problem = inner;
      stride = 1;
      batch = outer * indexing_dimension_size;
      problem_batch = outer * index_num;
      problem_along_x =
          self_info.strides[self_info.dims - 1] == 1 ? true : false;
    }

    return {
        self_info,
        index_info,
        index_num,
        indexing_dimension_size,
        batch,
        problem,
        stride,
        problem_batch,
        problem_along_x,
        val};
  }

 public:
  ValInfo vinfo_;
  IdxInfo iinfo_;
  int64_t index_num_;
  int64_t indexing_dimension_size_;
  T val_;
};

template <typename T, class ValInfo, class IdxInfo>
static inline void _index_fill_kernel(
    ValInfo& self_info,
    IdxInfo& index_info,
    int64_t dim,
    T val) {
  auto& queue = dpcppGetCurrentQueue();

  auto cfg = IndexFillConfig<T, ValInfo, IdxInfo>::make_config(
      self_info, index_info, dim, val);

  auto cgf = DPCPP_Q_CGF(__cgh) {
    auto kfn = DPCPP_Q_KFN(DPCPP::nd_item<2> item) {
      auto id = cfg.get_item_desc(item);

      int64_t si, pi, bi, idx_off, idx_logical_off, glb_batch_group,
          glb_batch_group_size, glb_batch_group_loc_off,
          glb_batch_group_glb_off, glb_off, glb_logical_off;
      if (id.glb_problem < cfg.problem_ && id.glb_batch < cfg.problem_batch_) {
        idx_logical_off = id.glb_batch % cfg.index_num_;
        idx_off = IndexToOffset<typename IdxInfo::scalar_t, int64_t>::get(
            idx_logical_off, cfg.iinfo_);
        glb_batch_group = id.glb_batch / cfg.index_num_;
        glb_batch_group_size = cfg.indexing_dimension_size_;
        glb_batch_group_loc_off = cfg.iinfo_.data[idx_off];
        glb_batch_group_loc_off = glb_batch_group_loc_off >= 0
            ? glb_batch_group_loc_off
            : glb_batch_group_size + glb_batch_group_loc_off;
        glb_batch_group_glb_off =
            glb_batch_group * glb_batch_group_size + glb_batch_group_loc_off;
        if (cfg.batch_ != 1) {
          si = 0;
          pi = id.chunk * id.chunk_size + id.chunk_off;
          bi = glb_batch_group_glb_off;
        } else {
          si = glb_batch_group_glb_off;
          pi = id.chunk * id.chunk_size + id.chunk_off;
          bi = 0;
        }
        glb_logical_off =
            si + pi * cfg.stride_ + bi * cfg.problem_ * cfg.stride_;
        glb_off = IndexToOffset<typename ValInfo::scalar_t, int64_t>::get(
            glb_logical_off,
            cfg.vinfo_,
            IndexToOffset<typename ValInfo::scalar_t, int64_t>::
                NON_STRICT_CONTIGUOUS);

        cfg.vinfo_.data[glb_off] = cfg.val_;
      }
    };
    __cgh.parallel_for(
        DPCPP::nd_range<2>(cfg.global_size(), cfg.group_size()), kfn);
  };
  DPCPP_Q_SUBMIT(queue, cgf);
}
