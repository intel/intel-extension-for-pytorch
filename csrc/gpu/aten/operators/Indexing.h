#pragma once
#include <c10/macros/Macros.h>
#include <stdio.h>
#include "ATen/ATen.h"
#include "BatchKernel.h"
#include "Loops.h"
#include "comm/Atomics.h"
#include "core/detail/TensorInfo.h"
using namespace at::AtenIpexTypeXPU;
using namespace xpu::dpcpp::detail;

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

  DPCPP_DEVICE void operator()(sycl::nd_item<2> item) const {
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
        cfg_.dinfo_.data,
        cfg_.sinfo_.data,
        dst_off,
        src_off,
        glb_batch_group_loc_off,
        cfg_.alpha_);
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
        sycl::nd_range<2>(cfg.global_size(), cfg.group_size()), idx_ker);
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
      int64_t idx,
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
      int64_t idx,
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
      int64_t idx,
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
      int64_t idx,
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

// DPCPP suggest: it’s possible (and even desirable) to oversubscribe tasks to
// device;
constexpr int OVER_SUBSCRIBE_DSS_FACTOR = 16;

template <typename func_t>
void dpcpp_small_index_kernel_impl(
    TensorIterator& iter,
    IntArrayRef index_size,
    IntArrayRef index_stride,
    IntArrayRef non_index_size,
    IntArrayRef non_index_stride,
    const func_t f) {
  auto numel = iter.numel();
  auto indices_size = iter.tensor(2).size(-1);
  auto& dpcpp_queue = dpcppGetCurrentQueue();
  auto dev_id = dpcppGetDeviceIdOfCurrentQueue();
  int64_t max_group_num = dpcppMaxDSSNum(dev_id) * OVER_SUBSCRIBE_DSS_FACTOR;

  auto total_index_iter = numel / indices_size;
  max_group_num = std::min(int64_t(total_index_iter / 2), max_group_num);

  // process the tail
  auto group_index_iter =
      (total_index_iter + max_group_num - 1) / max_group_num;
  auto group_num_tail = group_index_iter * max_group_num - total_index_iter;
  auto group_num = max_group_num - group_num_tail;
  auto group_numel = group_index_iter * indices_size;
  auto group_numel_tail = (group_index_iter - 1) * indices_size;

  auto wgroup_size = dpcppMaxWorkGroupSize(dev_id);
  wgroup_size = std::min(decltype(wgroup_size)(group_numel), wgroup_size);
  auto global_size = max_group_num * wgroup_size;

  size_t num_non_indices = non_index_size.size();
  at::detail::Array<int64_t, MAX_TENSORINFO_DIMS> src_sizes(0);
  at::detail::Array<int64_t, MAX_TENSORINFO_DIMS> src_strides(0);
  for (size_t i = 0; i < num_non_indices; ++i) {
    src_sizes[i] = non_index_size[i];
    src_strides[i] = non_index_stride[i];
  }
  auto src_strides0 = non_index_stride[0];

  size_t num_indices = index_size.size();
  at::detail::Array<int64_t, MAX_TENSORINFO_DIMS> sizes(0);
  at::detail::Array<int64_t, MAX_TENSORINFO_DIMS> strides(0);
  for (size_t i = 0; i < num_indices; i++) {
    sizes[i] = index_size[i];
    strides[i] = index_stride[i];
  }

  int64_t element_size_bytes = iter.tensor(1).element_size();
  int64_t indice_size_bytes = iter.tensor(2).element_size();
  auto cgf = DPCPP_Q_CGF(__cgh) {
    auto out_data = (char*)iter.data_ptr(0);
    auto in_data = (char*)iter.data_ptr(1);
    using index_buf_type = decltype((char*)iter.data_ptr(0));
    at::detail::Array<index_buf_type, MAX_TENSORINFO_DIMS> index_ptrs;
    for (size_t i = 0; i < num_indices; i++) {
      index_ptrs[i] = (char*)iter.data_ptr(i + 2);
    }

    dpcpp_local_acc_t<int64_t, 1> local_offset(indices_size, __cgh);
    auto kfn = DPCPP_Q_KFN(sycl::nd_item<1> item_id) {
      auto local_id = item_id.get_local_id(0);
      auto group_id = item_id.get_group(0);

      // construct a indices_size table on SLM
      for (int64_t local_index = local_id; local_index < indices_size;
           local_index += wgroup_size) {
        int64_t offset = 0;
        for (size_t i = 0; i < num_indices; i++) {
          int64_t index =
              *(int64_t*)(index_ptrs[i] + local_index * indice_size_bytes);
          SYCL_KERNEL_ASSERT(
              index >= -sizes[i] && index < sizes[i] && "index out of bounds");
          if (index < 0) {
            index += sizes[i];
          }
          offset += index * strides[i];
        }
        local_offset[local_index] = offset;
      }

      // calculate the number of workloads on each group
      auto group_linear_id = group_id * group_numel;
      auto group_numel_range = group_numel;
      if (group_num_tail && group_id >= group_num) {
        group_linear_id =
            group_num * group_numel + (group_id - group_num) * group_numel_tail;
        group_numel_range = group_numel_tail;
      }
      auto out_ptr = out_data;
      auto in_ptr = in_data;
      item_id.barrier(sycl::access::fence_space::local_space);

      // compute the in/out/indices offsets and perform memory copy
      for (int64_t local_index = local_id; local_index < group_numel_range;
           local_index += wgroup_size) {
        auto linear_id = group_linear_id + local_index;
        auto out_offset = linear_id * element_size_bytes;
        auto src_linear_id = linear_id / indices_size;
        int64_t in_offset = 0;
        for (int i = num_non_indices - 1; i > 0; --i) {
          in_offset += (src_linear_id % src_sizes[i]) * src_strides[i];
          src_linear_id /= src_sizes[i];
        }
        in_offset += src_linear_id * src_strides0;

        auto offset = local_offset[local_index % indices_size];
        f(out_ptr + out_offset, in_ptr + in_offset, offset);
      }
    };
    __cgh.parallel_for(
        sycl::nd_range<1>(
            sycl::range<1>(global_size), sycl::range<1>(wgroup_size)),
        kfn);
  };
  DPCPP_Q_SUBMIT(dpcpp_queue, cgf);
}

template <typename func_t>
void dpcpp_index_kernel_impl(
    TensorIterator& iter,
    IntArrayRef index_size,
    IntArrayRef index_stride,
    const func_t f) {
  size_t num_indices = index_size.size();
  auto numel = iter.numel();
  at::detail::Array<int64_t, MAX_TENSORINFO_DIMS> sizes(0);
  at::detail::Array<int64_t, MAX_TENSORINFO_DIMS> strides(0);
  for (size_t i = 0; i < num_indices; i++) {
    sizes[i] = index_size[i];
    strides[i] = index_stride[i];
  }

  auto& dpcpp_queue = dpcppGetCurrentQueue();

  auto cgf = DPCPP_Q_CGF(__cgh) {
    auto out_data = (char*)iter.data_ptr(0);
    auto in_data = (char*)iter.data_ptr(1);
    using index_buf_type = decltype((char*)iter.data_ptr(0));
    at::detail::Array<index_buf_type, MAX_TENSORINFO_DIMS> index_ptrs;
    for (size_t i = 0; i < num_indices; i++) {
      index_ptrs[i] = (char*)iter.data_ptr(i + 2);
    }

    auto offset_calc = make_offset_calculator<3>(iter);
    auto kfn = DPCPP_Q_KFN(sycl::item<1> item_id) {
      auto linear_idx = item_id.get_linear_id();
      auto offsets = offset_calc.get(linear_idx);
      auto out_ptr = out_data + offsets[0];
      auto in_ptr = in_data + offsets[1];
      int64_t offset = 0;
      //#pragma unroll
      for (size_t i = 0; i < num_indices; i++) {
        int64_t index = *(int64_t*)(index_ptrs[i] + offsets[2]);
        SYCL_KERNEL_ASSERT(
            index >= -sizes[i] && index < sizes[i] && "index out of bounds");
        if (index < 0) {
          index += sizes[i];
        }
        offset += index * strides[i];
      }
      f(out_ptr, in_ptr, offset);
    };
    __cgh.parallel_for(sycl::range</*dim=*/1>(numel), kfn);
  };
  DPCPP_Q_SUBMIT(dpcpp_queue, cgf);
}

template <typename func_t>
void dpcpp_index_kernel(
    TensorIterator& iter,
    IntArrayRef index_size,
    IntArrayRef index_stride,
    IntArrayRef non_index_size,
    IntArrayRef non_index_stride,
    const func_t f) {
  auto numel = iter.numel();

  if (numel == 0) {
    return;
  }

  size_t num_indices = index_size.size();
  TORCH_INTERNAL_ASSERT(num_indices == index_stride.size());
  TORCH_INTERNAL_ASSERT(
      num_indices == static_cast<size_t>(iter.ntensors()) - 2);
  TORCH_INTERNAL_ASSERT(num_indices <= MAX_TENSORINFO_DIMS);

  // the dpcpp_small_index_kernel_impl is applied for last several successive
  // dims indexing of an input tensor Taking 3-dims tensor input
  // (input.shape=[x,y,z]) for example: input[:,:,idx] or input[:,idx1,idx2]
  // when input tensor satisfies the following conditions, the
  // small_index_kernel path will be selected: 1.there are common indices such
  // as input[:,:,idx] and input[:,idx1,idx2] instead of
  //   input[idx0,idx1,idx2], input[idx0,idx1,:], input[idx0,:,idx2],
  //   input[idx0,:,:], input[:,idx1,:]
  // 2.the common indices numel should larger than 2 times of the
  // dpcppMaxComputeUnitSize (then we can get memory access benifit) 3.the
  // workloads in each group should larger than the maximum number of workitem
  // (ensure all the workitem activate) 4.the indices_table size should
  // satisfied the SLM limit condition

  // check whether the current case satisfying the condition 1
  // for 3-dims input:
  // Taking input[idx0,:,idx2] for example, the indices_sizes=[sz,1,sz]
  // While the satified case is input[:,idx1,idx2], indices_sizes=[1,sz,sz]
  bool small_index = non_index_size.size() != 0 && iter.tensor(1).dim() == 3;
  auto indices_sizes = iter.tensor(2).sizes();
  for (size_t i = 1; i < num_indices; ++i) {
    if (indices_sizes[i - 1] > indices_sizes[i]) {
      small_index = false;
      break;
    }
  }
  if (small_index) {
    auto& dpcpp_queue = dpcppGetCurrentQueue();
    auto dev_id = dpcppGetDeviceIdOfCurrentQueue();
    int64_t max_group_num = dpcppMaxDSSNum(dev_id);
    auto wgroup_size = dpcppMaxWorkGroupSize(dev_id);
    auto indices_size = iter.tensor(2).size(-1);
    auto total_index_iter = numel / indices_size;
    auto local_index = numel / max_group_num;

    // the max_local_mem_size = 65536B (64KB)
    auto max_local_mem_size = dpcppLocalMemSize(dev_id);
    auto indice_table_size = indices_size * sizeof(int64_t);

    // check whether the current case satisfying conditions 2,3,4
    small_index =
        (total_index_iter > 2 * max_group_num && local_index > wgroup_size &&
         indice_table_size < max_local_mem_size * 0.5);
    if (small_index) {
      dpcpp_small_index_kernel_impl<func_t>(
          iter, index_size, index_stride, non_index_size, non_index_stride, f);
      return;
    }
  }

  if (!iter.can_use_32bit_indexing()) {
    for (auto& sub_iter : iter.with_32bit_indexing()) {
      dpcpp_index_kernel(
          sub_iter, index_size, index_stride, IntArrayRef{}, IntArrayRef{}, f);
    }
    return;
  }

  dpcpp_index_kernel_impl<func_t>(iter, index_size, index_stride, f);
}
