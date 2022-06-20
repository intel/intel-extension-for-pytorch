#include <ATen/ATen.h>
#include <ATen/native/TensorIterator.h>

#include <ATen/MemoryOverlap.h>
#include <core/Memory.h>
#include <core/Stream.h>
#include <core/TensorImplUtils.h>
#include <core/detail/IndexUtils.h>
#include <core/detail/TensorInfo.h>
#include <runtime/Utils.h>
#include <utils/DPCPP.h>
#include <utils/Helpers.h>
#include "PSTLFunctions.h"
#include "comm/ATDispatch.h"
#include "comm/ApplyUtils.h"
#include "comm/Atomics.h"
#include "comm/MathReduce.h"
#include "comm/Numerics.h"

#include "Indexing.h"
#include "IndexingUtils.h"
#include "Loops.h"
#include "ParttenScan.h"

using namespace xpu::dpcpp;

namespace at {
namespace AtenIpexTypeXPU {

Tensor sum(const Tensor& self, c10::optional<ScalarType> dtype);

namespace impl {

template <typename scalar_t>
void indexSelect(
    const Tensor& dst,
    const Tensor& src,
    int dim,
    const Tensor& indices) {
  at::assert_no_internal_overlap(dst);
  at::assert_no_overlap(dst, src);
  at::assert_no_overlap(dst, indices);

  dim = at::maybe_wrap_dim(dim, src);
  int srcDims = src.dim();
  int dstDims = dst.dim();
  int idxDims = indices.dim();

  TORCH_CHECK(
      srcDims <= MAX_DPCPPTORCH_DIMS && srcDims > 0,
      "src tensor dim should be > 0 and < ",
      MAX_DPCPPTORCH_DIMS);
  TORCH_CHECK(
      dstDims <= MAX_DPCPPTORCH_DIMS && dstDims > 0,
      "dst tensor dim should be > 0 and < ",
      MAX_DPCPPTORCH_DIMS);
  TORCH_CHECK(
      idxDims <= MAX_DPCPPTORCH_DIMS && idxDims > 0,
      "index tensor dim should be > 0 and < ",
      MAX_DPCPPTORCH_DIMS);
  TORCH_CHECK(
      idxDims <= 1, "Index is supposed to be an empty tensor or a vector");
  TORCH_CHECK(
      dim >= -1 && dim < srcDims,
      "Indexing dim should be >= -1 and < dims - 1");
  TORCH_CHECK(srcDims > 0, "Source tensor is empty");
  TORCH_CHECK(
      indices.scalar_type() == ScalarType::Long ||
          indices.scalar_type() == ScalarType::Int,
      "index_select(): Expected dtype int32 or int64 for index but got: ",
      indices.scalar_type());
  TORCH_CHECK(
      src.scalar_type() == dst.scalar_type(),
      "index_select(): Source and result must have the same scalar type");

  IPEX_DISPATCH_INDEX_TYPES(indices.scalar_type(), "indexSelect", [&] {
    TensorInfo<index_t, int64_t> indices_info =
        getTensorInfo<index_t, int64_t>(indices);
    indices_info.collapseDims();

    auto new_size = src.sizes().vec();
    new_size[dim] = indices_info.sizes[0];
    dst.resize_(new_size);

    ptrdiff_t dst_num_elem = dst.numel();
    if (dst_num_elem == 0) {
      return;
    }

    TensorInfo<scalar_t, int64_t> dst_info =
        getTensorInfo<scalar_t, int64_t>(dst);
    TensorInfo<scalar_t, int64_t> src_info =
        getTensorInfo<scalar_t, int64_t>(src);
    int new_indexing_dim = src_info.collapseDims(dim);

    _index_select_kernel(src_info, dst_info, indices_info, new_indexing_dim);
  });
  return;
}

template <typename scalar_t>
void nonzero(Tensor& tensor, const Tensor& self_) {
  Tensor self = self_.contiguous();

  const int64_t num_dim = self.dim() == 0 ? 1 : self.dim();
  TORCH_CHECK(num_dim <= MAX_TENSORINFO_DIMS, "dim exceed max allowed dim");

  int64_t N = self.numel();

  if (N > 0) {
    Tensor idx_flat = at::empty(
        {N}, tensor.options().memory_format(LEGACY_CONTIGUOUS_MEMORY_FORMAT));
    Tensor range = at::empty(
        {N}, tensor.options().memory_format(LEGACY_CONTIGUOUS_MEMORY_FORMAT));

    scalar_t* self_begin = self.data_ptr<scalar_t>();
    int64_t* idx_flat_begin = idx_flat.data_ptr<int64_t>();
    int64_t* range_begin = nullptr;

    auto idx_flat_end = xpu::pstl::copy_if<int64_t>(
        range_begin, range_begin + N, idx_flat_begin, [=](int64_t x) {
          return Numerics<scalar_t>::ne(self_begin[x], scalar_t(0));
        });

    auto num_nonzeros = std::distance(idx_flat_begin, idx_flat_end);

    Tensor tensor_ = tensor.resize_({num_nonzeros, num_dim}).contiguous();
    if (num_nonzeros > 0 && num_dim > 0) {
      int64_t* tensor_begin = tensor_.data_ptr<int64_t>();

      // preload sizes tensor for index calculation
      int64_t sizes[MAX_TENSORINFO_DIMS];
      int64_t divisor[MAX_TENSORINFO_DIMS];
      sizes[num_dim - 1] = self.size(num_dim - 1);
      divisor[num_dim - 1] = 1;
      for (auto dim = num_dim - 2; dim >= 0; dim--) {
        sizes[dim] = self.size(dim);
        divisor[dim] = sizes[dim + 1] * divisor[dim + 1];
      }

      const int64_t N = num_nonzeros * num_dim;
      auto& dpcpp_queue = dpcppGetCurrentQueue();
      const auto dev_id = dpcppGetDeviceIdOfCurrentQueue();
      const auto wgroup_size = std::min(dpcppMaxWorkGroupSize(dev_id), N);
      const auto ngroups = (N + wgroup_size - 1) / wgroup_size;

      // restore flatten idx to indices
      auto cgf = DPCPP_Q_CGF(__cgh) {
        auto kfn = DPCPP_Q_KFN(sycl::nd_item<1> item_id) {
          auto global_id = item_id.get_global_linear_id();

          if (global_id < N) {
            auto index = global_id / num_dim;
            auto dim = global_id % num_dim;
            tensor_begin[global_id] =
                idx_flat_begin[index] / divisor[dim] % sizes[dim];
          }
        };

        __cgh.parallel_for(
            sycl::nd_range<1>(ngroups * wgroup_size, wgroup_size), kfn);
      };
      DPCPP_Q_SUBMIT(dpcpp_queue, cgf);

      // Support non-contiguous/outplace cases
      // TODO: Next step, we will give state of art algo/implementation.
      // Non-contiguous/outplace cases performance will be covered there.
      if (tensor.data_ptr() != tensor_.data_ptr()) {
        tensor.copy_(tensor_);
      }
    }
  } else {
    tensor = tensor.resize_({N, num_dim}).contiguous();
  }
}

template <typename scalar_t>
void _index_add(
    Tensor& dst,
    int64_t dim,
    const Tensor& indices,
    const Tensor& src,
    const Scalar& alpha) {
  scalar_t alpha_val = alpha.to<scalar_t>();
  dim = maybe_wrap_dim(dim, dst.dim());

  auto numIndices = indices.numel();
  TORCH_CHECK(
      indices.dim() <= 1, "index_add_(): Index is supposed to be a vector");
  TORCH_CHECK(
      indices.scalar_type() == ScalarType::Long,
      "index_add_(): Expected dtype int64 for index");
  TORCH_CHECK(
      dst.scalar_type() == src.scalar_type(),
      "index_add_(): self and source must have the same scalar type");
  TORCH_CHECK(
      dim == 0 || dim < src.dim(),
      "index_add_(): Indexing dim ",
      dim,
      " is out of bounds of tensor");
  TORCH_CHECK(
      numIndices == (src.dim() == 0 ? 1 : src.size(dim)),
      "index_add_(): Number of indices should be equal to self.size(dim)");

  at::assert_no_internal_overlap(dst);
  at::assert_no_overlap(dst, indices);
  at::assert_no_overlap(dst, src);

  TORCH_CHECK(
      dst.dim() <= MAX_DPCPPTORCH_DIMS,
      "tensor has too many (>",
      DPCPPTORCH_DIM_WARNING,
      ") dims");
  TORCH_CHECK(
      src.dim() <= MAX_DPCPPTORCH_DIMS,
      "tensor has too many (>",
      DPCPPTORCH_DIM_WARNING,
      ") dims");
  TORCH_CHECK(
      indices.dim() <= MAX_DPCPPTORCH_DIMS,
      "tensor has too many (>",
      DPCPPTORCH_DIM_WARNING,
      ") dims");

  // See Note [Enabling Deterministic Operations]
  if (globalContext().deterministicAlgorithms()) {
    // TODO: enable deterministic algorithm
    TORCH_CHECK(
        false, "index_add is not implemented with deterministic algorithm.")
  }

  // The `src` is partitioned into two parts:
  // -the size of each slice we are indexing, which is the
  // total size of the tensor ignoring dimension `dim`;
  // -the number of indices we are choosing, which is the total size
  // of the tensor `indices`.

  int dstDims = dst.dim() == 0 ? 1 : dst.dim();
  int srcDims = src.dim() == 0 ? 1 : src.dim();
  ptrdiff_t dstSliceSize = 1;
  for (int d = 0; d < dstDims; d++) {
    if (d != dim) {
      dstSliceSize *= dst.dim() == 0 ? 1 : dst.size(d);
    }
  }

  ptrdiff_t srcSliceSize = 1;
  bool mismatch = false;

  if (dstDims != srcDims)
    mismatch = true;

  for (int d = 0; d < srcDims; d++) {
    if (d != dim) {
      srcSliceSize *= src.dim() == 0 ? 1 : src.size(d);
      if (!mismatch &&
          (dst.dim() == 0 ? 1 : dst.size(d)) !=
              (src.dim() == 0 ? 1 : src.size(d)))
        mismatch = true;
    }
  }

  TORCH_CHECK(
      dstSliceSize == srcSliceSize,
      "Source/destination tensor have different slice sizes");

  if (mismatch) {
    static bool warningShown = false;
    if (!warningShown) {
      warningShown = true;
      fprintf(
          stderr,
          "Warning: source/destination slices have same size but different "
          "shape for an index operation. This behavior is deprecated.\n");
    }
  }

  ptrdiff_t sliceSize = dstSliceSize;
  ptrdiff_t srcTotalSize = src.numel();
  int64_t dstAddDimSize = dst.dim() == 0 ? 1 : dst.size(dim);

  if (sliceSize == 0) {
    return;
  }

  TensorInfo<int64_t, int64_t> indices_info =
      getTensorInfo<int64_t, int64_t>(indices);
  indices_info.collapseDims();

  TensorInfo<scalar_t, int64_t> src_info =
      getTensorInfo<scalar_t, int64_t>(src);

  TensorInfo<scalar_t, int64_t> dst_info =
      getTensorInfo<scalar_t, int64_t>(dst);
  int new_indexing_dim = dst_info.collapseDims(dim);

  _index_add_kernel(
      src_info, dst_info, indices_info, alpha_val, new_indexing_dim);
}

template <typename scalar_t>
void _index_fill(
    Tensor& dst,
    int64_t dim,
    const Tensor& indices,
    Scalar val_scalar) {
  auto val = val_scalar.to<scalar_t>();
  auto dst_ptr = dst.data_ptr<scalar_t>();
  auto idx_ptr = indices.data_ptr<int64_t>();
  int64_t indexing = dst.size(dim);
  int64_t inner = dst.stride(dim);
  int64_t outter = dst.numel() / (indexing * inner);

  TensorInfo<int64_t, int64_t> indices_info =
      getTensorInfo<int64_t, int64_t>(indices);
  indices_info.collapseDims();

  TensorInfo<scalar_t, int64_t> dst_info =
      getTensorInfo<scalar_t, int64_t>(dst);
  int dim_after_collapse = dst_info.collapseDims(dim);

  _index_fill_kernel(dst_info, indices_info, dim_after_collapse, val);

  return;
}

template <typename scalar_t>
void _index_copy(
    Tensor& dst,
    int64_t dim,
    const Tensor& indices,
    const Tensor& source) {
  TORCH_CHECK(dst.dim() <= MAX_DPCPPTORCH_DIMS, DPCPPTORCH_DIM_WARNING);
  TORCH_CHECK(indices.dim() <= MAX_DPCPPTORCH_DIMS, DPCPPTORCH_DIM_WARNING);
  TORCH_CHECK(source.dim() <= MAX_DPCPPTORCH_DIMS, DPCPPTORCH_DIM_WARNING);

  // The `src` is partitioned into two parts:
  // -the size of each slice we are indexing, which is the
  // total size of the tensor ignoring dimension `dim`;
  // -the number of indices we are choosing, which is the total size
  // of the tensor `indices`.
  int dstDims = dst.dim() == 0 ? 1 : dst.dim();

  TORCH_CHECK(dim >= 0 && dim < dstDims, "Indexing dim is out of bounds");

  ptrdiff_t sliceSize = 1;
  for (int d = 0; d < dstDims; d++) {
    if (d != dim) {
      sliceSize *= dst.dim() == 0 ? 1 : dst.size(d);
    }
  }
  ptrdiff_t dstTotalSize = dst.numel();
  ptrdiff_t numIndices = indices.numel();

  if (sliceSize == 0) {
    return;
  }

  TensorInfo<int64_t, int64_t> indices_info =
      getTensorInfo<int64_t, int64_t>(indices);
  indices_info.collapseDims();

  TensorInfo<scalar_t, int64_t> src_info =
      getTensorInfo<scalar_t, int64_t>(source);

  TensorInfo<scalar_t, int64_t> dst_info =
      getTensorInfo<scalar_t, int64_t>(dst);
  auto collapse_dim = (dst.dim() == 0) ? -1 : dim;
  int new_indexing_dim = dst_info.collapseDims(collapse_dim);
  _index_copy_kernel(src_info, dst_info, indices_info, new_indexing_dim);
}

template <typename scalar_t>
void Diag(Tensor& dst, const Tensor& src, int64_t k) {
  int nDimension = src.dim() == 0 ? 1 : src.dim();
  TORCH_CHECK(
      (nDimension == 2) || (nDimension == 1), "expected a matrix or a vector");

  if (nDimension == 2) {
    int64_t stride0 = src.stride(0);
    int64_t stride1 = src.stride(1);
    int64_t size0 = src.size(0);
    int64_t size1 = src.size(1);
    int64_t size = (k > 0) ? sycl::min((int64_t)size0, (int64_t)size1 - k)
                           : sycl::min((int64_t)size0 + k, (int64_t)size1);
    int64_t size_[1] = {size};
    TensorImpl_resizeNd(TensorImpl_Unwrap(dst), 1, size_, nullptr);
    if (size > 0) {
      auto in = src.data_ptr<scalar_t>();
      auto out = dst.data_ptr<scalar_t>();
      int64_t strideSelf = dst.dim() == 0 ? 1 : dst.stride(0);
      int64_t start = (k >= 0 ? k * stride1 : -k * stride0);
      auto& dpcpp_queue = dpcppGetCurrentQueue();

      auto cgf = DPCPP_Q_CGF(cgh) {
        auto kfn = DPCPP_Q_KFN(sycl::item<1> item_id) {
          size_t id = item_id.get_id(0);
          const int64_t bOffset = start + (stride0 + stride1) * id;
          out[strideSelf * id] = in[bOffset];
        };
        cgh.parallel_for(sycl::range<1>(dst.numel()), kfn);
      };
      DPCPP_Q_SUBMIT(dpcpp_queue, cgf);
    }
  } else {
    int64_t totalElements = src.numel();
    int64_t size = (k > 0) ? totalElements + k : totalElements - k;
    int64_t strideSrc = src.dim() == 0 ? 1 : src.stride(0);
    int64_t size_[2] = {size, size};
    TensorImpl_resizeNd(TensorImpl_Unwrap(dst), 2, size_, nullptr);
    dst.zero_();
    if (size > 0) {
      auto in = src.data_ptr<scalar_t>();
      auto out = dst.data_ptr<scalar_t>();
      int64_t stride0 = dst.stride(0);
      int64_t stride1 = dst.stride(1);
      int64_t start = (k >= 0 ? k * stride1 : -k * stride0);
      auto& dpcpp_queue = dpcppGetCurrentQueue();

      auto cgf = DPCPP_Q_CGF(cgh) {
        auto kfn = DPCPP_Q_KFN(sycl::item<1> item_id) {
          size_t id = item_id.get_id(0);
          const int64_t aOffset = start + (stride0 + stride1) * id;
          out[aOffset] = in[strideSrc * id];
        };
        cgh.parallel_for(sycl::range<1>(src.numel()), kfn);
      };
      DPCPP_Q_SUBMIT(dpcpp_queue, cgf);
    }
  }
}

template <typename scalar_t, typename mask_t>
void MaskedScatter(Tensor& tensor, const Tensor& mask_, const Tensor& src) {
  c10::MaybeOwned<Tensor> mask =
      expand_inplace(tensor, mask_, "masked_scatter_");
  auto maskSize = (*mask).numel();
  auto tensorSize = tensor.numel();
  auto srcSize = src.numel();

  // `mask` and `tensor` must have the same number of elements
  TORCH_CHECK(
      maskSize == tensorSize,
      "mask and tensor must have the same number of elements");

  // Determine our output size
  c10::optional<ScalarType> dtype;
  auto totalElements = at::AtenIpexTypeXPU::sum(*mask, dtype).item().to<int>();

  // The number of `1` elements present in the mask must be <= the
  // number of elements available in `src`
  if (totalElements > srcSize) {
    TORCH_CHECK(false, "source nElements must be == mask `1` elements");
  }

  Tensor maskLong = at::empty((*mask).sizes(), (*mask).options().dtype(kLong));
  maskLong.copy_(*mask);

  // Use a prefix sum to determine the output locations of the masked elements
  Tensor maskPrefixSum =
      at::empty((*mask).sizes(), (*mask).options().dtype(kLong));

  auto maskLong_size = maskLong.numel() * (maskLong.dtype().itemsize());
  auto maskPrefixSum_size =
      maskPrefixSum.numel() * (maskPrefixSum.dtype().itemsize());
  int64_t size = maskLong.numel();

  auto& dpcpp_queue = dpcppGetCurrentQueue();
  int64_t local_range = static_cast<int64_t>(1);
  int64_t global_range = static_cast<int64_t>(1);
  if (size != 0) {
    int64_t wg_size = dpcppMaxWorkGroupSize();
    local_range = size < wg_size ? size : wg_size;
    global_range = ((size + local_range - 1) / local_range) * local_range;
  }

  auto acc_maskLong_data = maskLong.data_ptr<int64_t>();
  auto acc_maskPrefixSum_data = maskPrefixSum.data_ptr<int64_t>();

  auto maskLong_ptr = acc_maskLong_data;
  auto maskPrefixSum_ptr = acc_maskPrefixSum_data;
  xpu::pstl::exclusive_scan(
      maskLong_ptr,
      maskLong_ptr + size,
      maskPrefixSum_ptr,
      static_cast<int64_t>(0));

  Tensor contigSrc = src.contiguous();
  Tensor contigMask = (*mask).contiguous();

  // command group function
  // copy src to tensor according to mask
  auto cgfMaskedScatter = DPCPP_Q_CGF(cgh) {
    auto src_data = contigSrc.data_ptr<scalar_t>();
    auto mask_data = static_cast<mask_t*>(contigMask.data_ptr());
    auto maskPrefixSum_data = maskPrefixSum.data_ptr<int64_t>();
    auto tensor_data = tensor.data_ptr<scalar_t>();

    // kernel function
    auto kfn = DPCPP_Q_KFN(sycl::nd_item<1> item) {
      int64_t linear_index = item.get_global_linear_id();
      if (linear_index < size) {
        if (mask_data[linear_index]) {
          tensor_data[linear_index] =
              src_data[maskPrefixSum_data[linear_index]];
        }
      }
    };

    cgh.parallel_for(
        sycl::nd_range<1>(
            sycl::range<1>(global_range), sycl::range<1>(local_range)),
        kfn);
  };

  // submit to DPCPP queue
  DPCPP_Q_SUBMIT(dpcpp_queue, cgfMaskedScatter);
}

template <typename scalar_t, typename mask_t>
void MaskedSelect(Tensor& tensor, const Tensor& src, const Tensor& mask) {
  TORCH_CHECK(mask.numel() == src.numel(), "sizes do not match");

  // Determine our output size
  c10::optional<ScalarType> dtype;
  int totalElements = at::AtenIpexTypeXPU::sum(mask, dtype).item().to<int>();
  int64_t real_sizes[1] = {(int64_t)totalElements};
  if (totalElements == 0) {
    TensorImpl_resizeNd(TensorImpl_Unwrap(tensor), 1, real_sizes, nullptr);
    return;
  }

  Tensor tensorContig = tensor.contiguous();

  TensorImpl_resizeNd(TensorImpl_Unwrap(tensorContig), 1, real_sizes, nullptr);
  if (&tensor != &tensorContig) {
    TensorImpl_resizeNd(TensorImpl_Unwrap(tensor), 1, real_sizes, nullptr);
  }

  Tensor maskLong = at::empty({0}, mask.options().dtype(kLong));
  maskLong.resize_(mask.sizes());
  maskLong.copy_(mask);

  // Use a prefix sum to determine the output locations of the masked elements
  Tensor maskPrefixSum = at::empty(mask.sizes(), mask.options().dtype(kLong));

  auto maskLong_size = maskLong.numel() * (maskLong.dtype().itemsize());
  auto maskPrefixSum_size =
      maskPrefixSum.numel() * (maskPrefixSum.dtype().itemsize());
  int64_t size = maskLong.numel();

  auto acc_maskLong_ptr = maskLong.data_ptr<int64_t>();
  auto acc_maskPrefixSum_ptr = maskPrefixSum.data_ptr<int64_t>();
  xpu::pstl::inclusive_scan<int64_t>(
      acc_maskLong_ptr,
      acc_maskLong_ptr + size,
      acc_maskPrefixSum_ptr,
      (int64_t)0);

  auto& dpcpp_queue = dpcppGetCurrentQueue();
  int64_t local_range = static_cast<int64_t>(1);
  int64_t global_range = static_cast<int64_t>(1);
  if (size != 0) {
    int64_t wg_size = dpcppMaxWorkGroupSize();
    local_range = size < wg_size ? size : wg_size;
    global_range = ((size + local_range - 1) / local_range) * local_range;
  }

  TensorInfo<scalar_t, uint64_t> src_info =
      getTensorInfo<scalar_t, uint64_t>(src);
  src_info.collapseDims();

  TensorInfo<mask_t, uint64_t> mask_info =
      getTensorInfo<mask_t, uint64_t>(mask);
  mask_info.collapseDims();

  // command group function
  auto cgfMaskedSelect = DPCPP_Q_CGF(cgh) {
    auto acc_src_data = src.data_ptr<scalar_t>();
    auto acc_mask_data = static_cast<mask_t*>(mask.data_ptr());
    auto acc_maskPrefixSum_data = maskPrefixSum.data_ptr<int64_t>();
    auto acc_tensor_data = tensorContig.data_ptr<scalar_t>();

    // kernel function per work-item
    auto kfn = DPCPP_Q_KFN(sycl::nd_item<1> item) {
      int64_t linear_index = item.get_global_linear_id();

      auto src_ptr = acc_src_data;
      auto mask_ptr = acc_mask_data;
      auto maskPrefix_ptr = acc_maskPrefixSum_data;
      auto tensor_ptr = acc_tensor_data;

      if (linear_index < size) {
        // The mask tensor maybe not contiguos.
        auto mask_offset =
            IndexToOffset<mask_t, uint64_t>().get(linear_index, mask_info);
        if (mask_ptr[mask_offset]) {
          // The src tensor maybe not contiguos.
          auto src_offset =
              IndexToOffset<scalar_t, uint64_t>().get(linear_index, src_info);
          tensor_ptr[maskPrefix_ptr[linear_index] - 1] = src_ptr[src_offset];
        }
      }
    };
    cgh.parallel_for(
        sycl::nd_range<1>(
            sycl::range<1>(global_range), sycl::range<1>(local_range)),
        kfn);
  };

  // submit to DPCPP queue
  DPCPP_Q_SUBMIT(dpcpp_queue, cgfMaskedSelect);

  if (!tensor.is_same(tensorContig)) {
    tensor.copy_(tensorContig);
  }
}

template <int N>
struct alignas(N) OpaqueType {
  char data[N];
};

template <typename scalar_t, typename Func>
void put(Tensor& self, const Tensor& index, const Tensor& source, Func f) {
  auto numel = index.numel();
  auto out_numel = self.numel();
  size_t scalar_bytes = sizeof(scalar_t);

  TensorInfo<scalar_t, uint64_t> out_info =
      getTensorInfo<scalar_t, uint64_t>(self);
  out_info.collapseDims();

  TensorInfo<long, uint64_t> indices_info =
      getTensorInfo<long, uint64_t>(index);
  indices_info.collapseDims();

  TensorInfo<scalar_t, uint64_t> source_info =
      getTensorInfo<scalar_t, uint64_t>(source);
  source_info.collapseDims();

  auto& dpcpp_queue = dpcppGetCurrentQueue();

  auto cgf = DPCPP_Q_CGF(__cgh) {
    auto out_data = self.data_ptr<scalar_t>();
    auto indices_data = index.data_ptr<long>();
    auto source_data = source.data_ptr<scalar_t>();

    auto kfn = DPCPP_Q_KFN(sycl::item<1> item_id) {
      auto out_ptr = (char*)out_data;
      auto indices_ptr = indices_data;
      auto source_ptr = (char*)source_data;

      auto linear_idx = item_id.get_id(0);
      auto idx_offset =
          IndexToOffset<long, uint64_t>::get(linear_idx, indices_info);

      auto index = indices_ptr[idx_offset];
      if (index > out_numel) {
        /*error handle*/
        return;
      }

      auto src_offset =
          IndexToOffset<scalar_t, uint64_t>::get(linear_idx, source_info);
      src_offset *= scalar_bytes;
      auto out_offset = IndexToOffset<scalar_t, uint64_t>::get(index, out_info);
      out_offset *= scalar_bytes;

      f(out_ptr, source_ptr + src_offset, out_offset);
    };

    __cgh.parallel_for(sycl::range</*dim=*/1>(numel), kfn);
  };
  DPCPP_Q_SUBMIT(dpcpp_queue, cgf);
}

void index(
    TensorIterator& iter,
    IntArrayRef index_size,
    IntArrayRef index_stride,
    IntArrayRef non_index_size,
    IntArrayRef non_index_stride) {
  IPEX_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(
      at::ScalarType::BFloat16,
      at::ScalarType::Half,
      at::ScalarType::Bool,
      iter.dtype(),
      "index",
      [&] {
        using dtype = OpaqueType<sizeof(scalar_t)>;
        dpcpp_index_kernel(
            iter,
            index_size,
            index_stride,
            non_index_size,
            non_index_stride,
            [](char* out_data, char* in_data, int64_t offset) {
              *(dtype*)out_data = *(dtype*)(in_data + offset);
            });
      });
}

void index_put_impl(
    TensorIterator& iter,
    IntArrayRef index_size,
    IntArrayRef index_stride,
    bool accumulate) {
  if (accumulate) {
    IPEX_DISPATCH_ATOMIC_ALL_TYPES_AND_COMPLEX(iter.dtype(), "index_put", [&] {
      dpcpp_index_kernel(
          iter,
          index_size,
          index_stride,
          IntArrayRef{},
          IntArrayRef{},
          [](char* out_data, char* in_data, int64_t offset) {
            dpcpp_global_ptr_pt<scalar_t> out_ptr =
                (dpcpp_global_ptr_pt<scalar_t>)(out_data + offset);
            auto in = *(scalar_t*)in_data;
            atomicAdd(out_ptr, in);
          });
    });
  } else {
    IPEX_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(
        at::ScalarType::BFloat16,
        at::ScalarType::Half,
        at::ScalarType::Bool,
        iter.dtype(),
        "index_put",
        [&] {
          using dtype = OpaqueType<sizeof(scalar_t)>;
          dpcpp_index_kernel(
              iter,
              index_size,
              index_stride,
              IntArrayRef{},
              IntArrayRef{},
              [](char* out_data, char* in_data, int64_t offset) {
                *(dtype*)(out_data + offset) = *(dtype*)in_data;
              });
        });
  }
}

template <typename scalar_t>
void take_dpcpp(Tensor& dst, const Tensor& src, const Tensor& index) {
  TORCH_CHECK(src.dim() <= MAX_DPCPPTORCH_DIMS, DPCPPTORCH_DIM_WARNING);
  TORCH_CHECK(dst.dim() <= MAX_DPCPPTORCH_DIMS, DPCPPTORCH_DIM_WARNING);
  TORCH_CHECK(index.dim() <= MAX_DPCPPTORCH_DIMS, DPCPPTORCH_DIM_WARNING);
  TORCH_CHECK(
      !(src.numel() == 0 && index.numel() != 0),
      "tried to take from an empty tensor");

  dst = dst.resize_as_(index);

  ptrdiff_t dst_num_elem = dst.numel();
  if (dst_num_elem == 0) {
    return;
  }

  TensorInfo<scalar_t, int64_t> src_info =
      getTensorInfo<scalar_t, int64_t>(src);
  src_info.collapseDims();

  TensorInfo<scalar_t, int64_t> dst_info =
      getTensorInfo<scalar_t, int64_t>(dst);
  dst_info.collapseDims();

  TensorInfo<int64_t, int64_t> idx_info =
      getTensorInfo<int64_t, int64_t>(index);
  idx_info.collapseDims();

  auto& dpcpp_queue = dpcppGetCurrentQueue();
  auto dev_id = dpcppGetDeviceIdOfCurrentQueue();
  auto wgroup_size = dpcppMaxWorkGroupSize(dev_id);
  auto wgroup_range = (dst_num_elem + wgroup_size - 1) / wgroup_size;

  auto cgf = DPCPP_Q_CGF(cgh) {
    auto src_data = src.data_ptr<scalar_t>();
    auto dst_data = dst.data_ptr<scalar_t>();
    auto idx_data = index.data_ptr<int64_t>();

    auto kfn = DPCPP_Q_KFN(sycl::nd_item<1> item) {
      auto linear_idx = item.get_global_linear_id();
      if (linear_idx < dst_num_elem) {
        auto idx_offset = linear_idx;
        IndexToOffset<int64_t, int64_t>::get(
            linear_idx,
            idx_info,
            IndexToOffset<int64_t, int64_t>::NON_STRICT_CONTIGUOUS);
        auto idx = idx_data[idx_offset];
        auto source_offset = idx;
        IndexToOffset<scalar_t, int64_t>::get(
            idx,
            src_info,
            IndexToOffset<scalar_t, int64_t>::NON_STRICT_CONTIGUOUS);
        auto dst_offset = linear_idx;
        IndexToOffset<scalar_t, int64_t>::get(
            linear_idx,
            dst_info,
            IndexToOffset<scalar_t, int64_t>::NON_STRICT_CONTIGUOUS);

        dst_data[dst_offset] = src_data[source_offset];
      }
    };

    cgh.parallel_for(
        sycl::nd_range<1>({wgroup_range * wgroup_size}, {wgroup_size}), kfn);
  };

  DPCPP_Q_SUBMIT(dpcpp_queue, cgf);
}

static std::tuple<bool, Tensor> canDispatchToMaskedFill(
    const Tensor& self,
    const torch::List<c10::optional<at::Tensor>>& indices,
    const Tensor& value) {
  if (!(value.numel() == 1 && value.device().is_cpu())) {
    return std::make_tuple(false, Tensor());
  }
  int64_t num_ind = 0;
  Tensor mask;
  auto self_device = self.device();
  for (const c10::optional<Tensor> i : indices) {
    if (!i.has_value() || !(*i).defined()) {
      num_ind++;
    } else {
      Tensor index = std::move(*i);
      if ((index.scalar_type() != kByte && index.scalar_type() != kBool) ||
          index.device() != self_device || mask.defined()) {
        return std::make_tuple(false, Tensor());
      } else {
        mask = index;
        for (int64_t j = 0; j < index.dim(); j++) {
          int64_t srcIdx = num_ind + j;
          TORCH_CHECK_INDEX(
              index.size(j) == self.size(srcIdx),
              "The shape of the mask ",
              index.sizes(),
              " at index ",
              j,
              " does not match the shape of the indexed tensor ",
              self.sizes(),
              " at index ",
              srcIdx);
        }
        num_ind += mask.ndimension();
      }
    }
  }
  for (int64_t i = num_ind; i < self.ndimension(); i++) {
    mask = mask.unsqueeze(-1);
  }
  return std::make_tuple(true, mask);
}
} // namespace impl

Tensor& index_select_out(
    Tensor& out,
    const Tensor& self,
    int64_t dim,
    const Tensor& index) {
  IPEX_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      at::ScalarType::Bool,
      self.scalar_type(),
      "indexSelect",
      [=]() { impl::indexSelect<scalar_t>(out, self, dim, index); });
  return out;
}

Tensor index_select(const Tensor& self, int64_t dim, const Tensor& index) {
  auto out = at::empty({0}, self.options());
  return at::AtenIpexTypeXPU::index_select_out(out, self, dim, index);
}

Tensor& nonzero_out(Tensor& out, const Tensor& self) {
  IPEX_DISPATCH_ALL_TYPES_AND3(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      at::ScalarType::Bool,
      self.scalar_type(),
      "indexSelect",
      [&]() { impl::nonzero<scalar_t>(out, self); });
  return out;
}

Tensor nonzero(const at::Tensor& self) {
  auto out = at::empty({0}, self.options().dtype(kLong));
  return at::AtenIpexTypeXPU::nonzero_out(out, self);
}

Tensor count_nonzero(const Tensor& self, IntArrayRef dims) {
  return (self != 0).sum(dims);
}

Tensor& index_add_(
    Tensor& self,
    int64_t dim,
    const Tensor& index,
    const Tensor& source,
    const Scalar& alpha) {
  IPEX_DISPATCH_ATOMIC_ALL_TYPES_AND_COMPLEX(
      self.scalar_type(), "index_add_", [&] {
        impl::_index_add<scalar_t>(self, dim, index, source, alpha);
      });
  return self;
}

Tensor& _index_copy_(
    Tensor& self,
    int64_t dim,
    const Tensor& index,
    const Tensor& source) {
  // Handle the case when self /source is 0-dim
  Tensor self_nonzero = self.dim() == 0 ? self.unsqueeze(0) : self;
  Tensor source_nonzero = source.dim() == 0 ? source.unsqueeze(0) : source;

  IPEX_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(
      at::ScalarType::Half,
      at::ScalarType::Bool,
      at::ScalarType::BFloat16,
      self.scalar_type(),
      "index_copy",
      [&]() { impl::_index_copy<scalar_t>(self, dim, index, source); });
  return self;
}

Tensor& index_copy_(
    Tensor& self,
    int64_t dim,
    const Tensor& index,
    const Tensor& source) {
  dim = maybe_wrap_dim(dim, self.dim());
  TORCH_CHECK_INDEX(
      index.dim() < 2,
      "index_copy_(): Index should have dimension 1 or 0 (got ",
      index.dim(),
      ")");
  at::assert_no_internal_overlap(self);
  at::assert_no_overlap(self, index);
  at::assert_no_overlap(self, source);

  int64_t numIndices = index.numel();
  if (source.dim() == 0 && numIndices != 1) {
    TORCH_CHECK_INDEX(
        false,
        "index_copy_(): When source is scalar, index should have one element (got ",
        numIndices,
        ")");
  } else if (
      (source.dim() != self.dim()) && (source.dim() != 0 && self.dim() != 0)) {
    TORCH_CHECK_INDEX(
        false,
        "index_copy_(): When source and destination are not scalars, their dimensionality must match. Source dimensionality (",
        source.dim(),
        "), destination dimensionality (",
        self.dim(),
        ")");
  }

  TORCH_CHECK(
      index.scalar_type() == ScalarType::Long,
      "index_copy_(): Expected a long tensor for index, but got ",
      index.scalar_type())
  TORCH_CHECK(
      self.scalar_type() == source.scalar_type(),
      "index_copy_(): self and source expected to have the same dtype, but got (self) ",
      self.scalar_type(),
      " and (source) ",
      source.scalar_type());
  TORCH_CHECK(
      self.device() == source.device() && self.device() == index.device(),
      "index_copy_(): self, index and source expected to be in the same device, but got (self) ",
      self.device(),
      ", (index) ",
      index.device(),
      ", and (source) ",
      source.device());

  // Check that source and destination slices have the same size
  auto selfSlicedSizes = self.sizes().vec();
  if (selfSlicedSizes.size() > 0) {
    selfSlicedSizes.erase(selfSlicedSizes.begin() + dim);
  }
  auto sourceSlicedSizes = source.sizes().vec();
  if (sourceSlicedSizes.size() > 0) {
    sourceSlicedSizes.erase(sourceSlicedSizes.begin() + dim);
  }
  if (selfSlicedSizes.size() != sourceSlicedSizes.size() ||
      !std::equal(
          selfSlicedSizes.begin(),
          selfSlicedSizes.end(),
          sourceSlicedSizes.begin())) {
    std::stringstream ss;
    ss << "index_copy_(): Source/destination tensor must have same slice shapes. ";
    ss << "Destination slice shape: " << selfSlicedSizes << " at dimension "
       << dim;
    ss << " and source slice shape: " << sourceSlicedSizes
       << " at dimension 0.";
    TORCH_CHECK(false, ss.str());
  }
  TORCH_CHECK_INDEX(
      source.dim() == 0 || numIndices == source.size(dim),
      "index_copy_(): Number of indices (",
      numIndices,
      ") should be equal to source.size(dim) (",
      source.size(dim),
      ")");

  // See Note [Enabling Deterministic Operations]
  if (globalContext().deterministicAlgorithms()) {
    // TODO: enable deterministic algorithm
    TORCH_CHECK(
        false, "index_copy is not implemented with deterministic algorithm.");
  }

  return at::_index_copy_(self, dim, index, source);
}

Tensor& index_fill_(
    Tensor& self,
    int64_t dim,
    const Tensor& index,
    const Scalar& value) {
  TORCH_CHECK_INDEX(
      index.scalar_type() == ScalarType::Long,
      "index_fill_(): Expected dtype int64 for index.");

  at::assert_no_overlap(self, index);
  if (at::has_internal_overlap(self) == at::MemOverlap::Yes) {
    TORCH_WARN(
        "Use of index_fill_ on expanded tensors is deprecated. "
        "Please clone() the tensor before performing this operation. "
        "This also applies to advanced indexing e.g. tensor[mask] = scalar");
  }

  if (!self.is_complex() && value.isComplex()) {
    TORCH_CHECK(
        false,
        "index_fill_(): Converting complex Scalar to non-complex type is not supported");
  }

  // Handle the case when `self` is 0-dim
  Tensor self_nonzero_dim = (self.dim() == 0) ? self.unsqueeze(-1) : self;

  TORCH_CHECK(index.dim() <= 1, "Index has to be a vector/scalar");
  TORCH_CHECK(self.dim() <= MAX_DPCPPTORCH_DIMS, DPCPPTORCH_DIM_WARNING);
  TORCH_CHECK(index.dim() <= MAX_DPCPPTORCH_DIMS, DPCPPTORCH_DIM_WARNING);

  // The `src` is partitioned into two parts:
  // -the size of each slice we are indexing, which is the
  // total size of the tensor ignoring dimension `dim`;
  // -the number of indices we are choosing, which is the total size
  // of the tensor `indices`.
  int selfDims = self.dim() == 0 ? 1 : self.dim();

  TORCH_CHECK(dim >= 0 && dim < selfDims, "Indexing dim is out of bounds");

  ptrdiff_t sliceSize = 1;
  for (int d = 0; d < selfDims; d++) {
    if (d != dim) {
      sliceSize *= self.dim() == 0 ? 1 : self.size(d);
    }
  }
  if (sliceSize == 0) {
    return self;
  }

  IPEX_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(
      at::ScalarType::Half,
      at::ScalarType::Bool,
      at::ScalarType::BFloat16,
      self.scalar_type(),
      "index_fill",
      [&]() {
        impl::_index_fill<scalar_t>(self_nonzero_dim, dim, index, value);
      });
  return self;
}

Tensor& index_fill_(
    Tensor& self,
    int64_t dim,
    const Tensor& index,
    const Tensor& value) {
  TORCH_CHECK(
      value.dim() == 0,
      "index_fill_ only supports a 0-dimensional value tensor, but got tensor "
      "with ",
      value.dim(),
      " dimension(s).");
  return at::AtenIpexTypeXPU::index_fill_(self, dim, index, value.item());
}

Tensor& diag_out(Tensor& out, const Tensor& self, int64_t diagonal) {
  IPEX_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      at::ScalarType::Bool,
      self.scalar_type(),
      "Diag",
      [&]() { impl::Diag<scalar_t>(out, self, diagonal); });
  return out;
}

Tensor diag(const Tensor& self, int64_t diagonal) {
  Tensor out = at::empty({0}, self.options());
  return at::AtenIpexTypeXPU::diag_out(out, self, diagonal);
}

Tensor trace(const Tensor& self) {
  TORCH_CHECK(self.dim() == 2, "expected a matrix");
  Tensor diag = at::AtenIpexTypeXPU::diag(self, 0);
  optional<ScalarType> dtype;
  Tensor out = at::AtenIpexTypeXPU::sum(diag, dtype);
  return out;
}

template <typename mask_t>
void masked_fill_kernel(TensorIterator& iter, const Scalar& value) {
  IPEX_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(
      at::ScalarType::BFloat16,
      at::ScalarType::Half,
      at::ScalarType::Bool,
      iter.dtype(),
      "masked_fill_",
      [&] {
        const auto value_ = value.to<scalar_t>();
        dpcpp_kernel_for_tensor_iter(
            iter, [=](scalar_t self, mask_t mask) -> scalar_t {
              if (mask) {
                return value_;
              }
              return self;
            });
      });
}

Tensor& masked_fill_(Tensor& self, const Tensor& mask, const Scalar& value) {
  TORCH_CHECK(
      self.device() == mask.device(),
      "expected self and mask to be on the same device, but got mask on ",
      mask.device(),
      " and self on ",
      self.device());
  TORCH_CHECK(
      mask.scalar_type() == kByte || mask.scalar_type() == kBool,
      "expected mask dtype to be Bool but got ",
      mask.scalar_type());
  auto maybe_outnames =
      namedinference::broadcast_to_outnames(self, mask, "masked_fill_");
  if (at::has_internal_overlap(self) == MemOverlap::Yes) {
    TORCH_WARN(
        "Use of masked_fill_ on expanded tensors is deprecated. "
        "Please clone() the tensor before performing this operation. "
        "This also applies to advanced indexing e.g. tensor[mask] = scalar");
  }
  at::assert_no_partial_overlap(self, mask);
  c10::MaybeOwned<Tensor> b_mask = expand_inplace(self, mask, "masked_fill_");

  auto iter = TensorIteratorConfig()
                  .set_check_mem_overlap(false)
                  .check_all_same_dtype(false)
                  .resize_outputs(false)
                  .add_output(self)
                  .add_input(self)
                  .add_input(*b_mask)
                  .build();

  if (mask.dtype() == at::ScalarType::Byte) {
    TORCH_WARN(
        "masked_fill_ received a mask with dtype torch.uint8, this behavior is now deprecated,"
        "please use a mask with dtype torch.bool instead.");
    masked_fill_kernel<uint8_t>(iter, value);
  } else {
    masked_fill_kernel<bool>(iter, value);
  }
  return self;
}

Tensor& masked_fill_(Tensor& self, const Tensor& mask, const Tensor& value) {
  return at::AtenIpexTypeXPU::masked_fill_(self, mask, value.item());
}

Tensor& masked_scatter_(
    Tensor& self,
    const Tensor& mask,
    const Tensor& source) {
  at::assert_no_internal_overlap(self);
  TORCH_CHECK(
      self.scalar_type() == source.scalar_type(),
      "masked_scatter: expected self and source to have same dtypes but got",
      self.scalar_type(),
      " and ",
      source.scalar_type());

  IPEX_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(
      at::ScalarType::Bool,
      at::ScalarType::BFloat16,
      at::ScalarType::Half,
      self.scalar_type(),
      "MaskedScatter",
      [&]() {
        if (mask.dtype() == ScalarType::Bool) {
          impl::MaskedScatter<scalar_t, bool>(self, mask, source);
        } else if (mask.dtype() == ScalarType::Byte) {
          TORCH_WARN(
              "masked_scatter_ received a mask with dtype torch.uint8, this behavior is now deprecated,"
              "please use a mask with dtype torch.bool instead.");
          impl::MaskedScatter<scalar_t, uint8_t>(self, mask, source);
        } else {
          AT_ERROR(
              "masked_scatter: expected BoolTensor or ByteTensor for mask");
        }
      });
  return self;
}

Tensor& masked_select_out(Tensor& out, const Tensor& self, const Tensor& mask) {
  TORCH_CHECK(
      self.scalar_type() == out.scalar_type(),
      "masked_select(): self and result must have the same scalar type")

  at::assert_no_internal_overlap(out);
  at::assert_no_overlap(out, self);
  at::assert_no_overlap(out, mask);

  c10::MaybeOwned<Tensor> b_self, b_mask;
  std::tie(b_self, b_mask) = expand_outplace(self, mask, "masked_select_out");
  IPEX_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(
      at::ScalarType::Bool,
      at::ScalarType::BFloat16,
      at::ScalarType::Half,
      self.scalar_type(),
      "MaskedSelect",
      [&]() {
        if (mask.dtype() == ScalarType::Bool) {
          impl::MaskedSelect<scalar_t, bool>(out, *b_self, *b_mask);
        } else if (mask.dtype() == ScalarType::Byte) {
          TORCH_WARN(
              "indexing with dtype torch.uint8 is now deprecated, "
              "please use a mask with dtype torch.bool instead.");
          impl::MaskedSelect<scalar_t, uint8_t>(out, *b_self, *b_mask);
        } else {
          AT_ERROR("masked_select: expected BoolTensor or ByteTensor for mask");
        }
      });
  return out;
}

Tensor masked_select(const Tensor& self, const Tensor& mask) {
  Tensor out = at::empty({0}, self.options());
  return at::AtenIpexTypeXPU::masked_select_out(out, self, mask);
}

Tensor& put_(
    Tensor& self,
    const Tensor& index_,
    const Tensor& source_,
    bool accumulate) {
  TORCH_CHECK(
      index_.numel() == source_.numel(),
      "indices number must be same as the source number");
  TORCH_CHECK(
      index_.dtype() == kLong,
      "indices number must be same as the source number");
  TORCH_CHECK(
      self.dtype() == source_.dtype(),
      "out and source must be the same tpye. got:",
      self.dtype(),
      " and ",
      source_.dtype());

  at::assert_no_internal_overlap(self);
  at::assert_no_overlap(self, index_);
  at::assert_no_overlap(self, source_);

  Tensor index;
  Tensor source;
  // Ensure index is on the same device as self
  if (index_.device() != self.device()) {
    index = index_.to(self.device());
  } else {
    index = index_;
  }

  // Ensure source is on the same device as self
  if (source_.device() != self.device()) {
    source = source_.to(self.device());
  } else {
    source = source_;
  }

  if (accumulate) {
    IPEX_DISPATCH_ATOMIC_ALL_TYPES_AND_COMPLEX(self.scalar_type(), "put_", [&] {
      impl::put<scalar_t>(
          self,
          index,
          source,
          [](char* out_data, char* in_data, uint64_t offset) {
            dpcpp_global_ptr_pt<scalar_t> out_ptr =
                (dpcpp_global_ptr_pt<scalar_t>)(out_data + offset);
            auto in = *(scalar_t*)in_data;
            atomicAdd(out_ptr, in);
          });
    });
  } else {
    IPEX_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(
        at::ScalarType::BFloat16,
        at::ScalarType::Half,
        at::ScalarType::Bool,
        self.scalar_type(),
        "put_",
        [&] {
          using dtype = impl::OpaqueType<sizeof(scalar_t)>;
          impl::put<scalar_t>(
              self,
              index,
              source,
              [](char* out_data, char* in_data, uint64_t offset) {
                *(dtype*)(out_data + offset) = *(dtype*)in_data;
              });
        });
  }

  return self;
}

Tensor index(
    const Tensor& self,
    const c10::List<c10::optional<Tensor>>& indices) {
  TORCH_CHECK(
      indices.size() <= (size_t)self.dim(),
      "too many indices for tensor of dimension ",
      self.dim(),
      " (got ",
      indices.size(),
      ")");

  auto info = make_info(self, indices);
  auto iter = make_index_iterator(info);
  impl::index(
      iter,
      info.indexed_sizes,
      info.indexed_strides,
      info.non_indexed_sizes,
      info.non_indexed_strides);
  return iter.output();
}

Tensor& _index_put_impl_(
    Tensor& self,
    const c10::List<c10::optional<Tensor>>& indices,
    const Tensor& value,
    bool accumulate,
    bool unsafe) {
  TORCH_CHECK(
      indices.size() <= (size_t)self.dim(),
      "too many indices for tensor of dimension ",
      self.dim(),
      " (got ",
      indices.size(),
      ")");
  if (at::has_internal_overlap(self) == MemOverlap::Yes) {
    TORCH_WARN(
        "Use of index_put_ on expanded tensors is deprecated. "
        "Please clone() the tensor before performing this operation. "
        "This also applies to advanced indexing e.g. tensor[indices] = tensor");
  }
  if (!accumulate) {
    auto masked_fill_dispatch =
        impl::canDispatchToMaskedFill(self, indices, value);
    if (std::get<0>(masked_fill_dispatch)) {
      return self.masked_fill_(std::get<1>(masked_fill_dispatch), value.item());
    }
  }
  auto value_ = value;
  if (value.device() != self.device() && value.numel() == 1 &&
      value.dim() == 0) {
    value_ = value.to(self.device());
  }
  at::assert_no_overlap(self, value);
  // NOLINTNEXTLINE(performance-implicit-conversion-in-loop)
  for (const c10::optional<Tensor>& index : indices) {
    if (index.has_value()) {
      at::assert_no_overlap(self, *index);
    }
  }

  // See Note [Enabling Deterministic Operations]
  if (accumulate && globalContext().deterministicAlgorithms()) {
    // TODO: enable deterministic algorithm and change the condition check.
    TORCH_CHECK(
        false, "index_put is not implemented with deterministic algorithm.");
  }

  auto info = make_info(self, indices);
  auto iter = make_index_put_iterator(info, value);
  impl::index_put_impl(
      iter, info.indexed_sizes, info.indexed_strides, accumulate);
  return self;
}

Tensor& take_out(const Tensor& self, const Tensor& index, Tensor& out) {
  // Type and device checks
  TORCH_CHECK(
      index.scalar_type() == ScalarType::Long,
      "take(): Expected a long tensor for index, but got ",
      index.scalar_type());
  TORCH_CHECK(
      self.scalar_type() == out.scalar_type(),
      "take(): self and out expected to havethe same dtype, but got self.dtype = ",
      self.scalar_type(),
      " and out.dtype = ",
      out.scalar_type());
  TORCH_CHECK(
      self.device() == out.device() && self.device() == index.device(),
      "take(): self, index and out expected to be in the same device, but got self.device = ",
      self.device(),
      ", index.device = ",
      index.device(),
      ", and out.device = ",
      out.device());

  // index checks
  TORCH_CHECK_INDEX(
      !(self.numel() == 0 && index.numel() != 0),
      "take(): tried to take from an empty tensor");

  at::assert_no_internal_overlap(out);
  at::assert_no_overlap(out, index);
  at::assert_no_overlap(out, self);

  IPEX_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(
      at::ScalarType::BFloat16,
      at::ScalarType::Half,
      at::ScalarType::Bool,
      self.scalar_type(),
      "take",
      [&]() { impl::take_dpcpp<scalar_t>(out, self, index); });

  return out;
}

Tensor take(const Tensor& self, const Tensor& index) {
  Tensor out = at::empty({0}, self.options());
  return at::AtenIpexTypeXPU::take_out(self, index, out);
}

} // namespace AtenIpexTypeXPU
} // namespace at
