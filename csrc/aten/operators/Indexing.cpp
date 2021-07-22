#include <ATen/ATen.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/AtenIpexTypeXPU.h>

#include <runtime/Utils.h>
#include <utils/DPCPP.h>
#include <utils/Helpers.h>
#include <core/Memory.h>
#include <core/Stream.h>
#include <core/TensorImplUtils.h>
#include <core/detail/IndexUtils.h>
#include <core/detail/TensorInfo.h>
#include "comm/ApplyUtils.h"
#include "comm/Atomics.h"
#include "comm/MathReduce.h"
#include "comm/Numerics.h"
#include "comm/ATDispatch.h"

#include "IndexingUtils.h"
#include "Loops.h"
#include "ParttenScan.h"

#ifdef USE_ONEDPL
#include <oneapi/dpl/algorithm>
#include <oneapi/dpl/execution>
#include <oneapi/dpl/numeric>
#include <oneapi/dpl/iterator>
#endif


using namespace xpu::dpcpp;

namespace at {
namespace AtenIpexTypeXPU {
namespace impl {

DPCPP_DEF_K1(index_select_ker);
template <typename scalar_t>
void indexSelect(
    Tensor& dst,
    const Tensor& src,
    int dim,
    const Tensor& indices) {
  int srcDims = src.dim() == 0 ? 1 : src.dim();
  int dstDims = dst.dim() == 0 ? 1 : dst.dim();
  int idxDims = indices.dim() == 0 ? 1 : indices.dim();

  TORCH_CHECK(srcDims <= MAX_DPCPPTORCH_DIMS, DPCPPTORCH_DIM_WARNING);
  TORCH_CHECK(dstDims <= MAX_DPCPPTORCH_DIMS, DPCPPTORCH_DIM_WARNING);
  TORCH_CHECK(idxDims <= MAX_DPCPPTORCH_DIMS, DPCPPTORCH_DIM_WARNING);
  TORCH_CHECK(
      idxDims <= 1, "Index is supposed to be an empty tensor or a vector");
  TORCH_CHECK(dim < srcDims, "Indexing dim is out of bounds");
  TORCH_CHECK(srcDims > 0, "Source tensor is empty");

  TORCH_CHECK(
      indices.scalar_type() == ScalarType::Long,
      "index_select(): Expected dtype int64 for index");
  TORCH_CHECK(
      src.scalar_type() == dst.scalar_type(),
      "index_select(): Source and result must have the same scalar type");

  TensorInfo<int64_t, unsigned int> indices_info =
      getTensorInfo<int64_t, unsigned int>(indices);
  indices_info.collapseDims();

  auto new_size = src.sizes().vec();
  new_size[dim] = indices_info.sizes[0];
  dst.resize_(new_size);

  ptrdiff_t dst_num_elem = dst.numel();
  if (dst_num_elem == 0) {
    return;
  }

  TensorInfo<scalar_t, unsigned int> dst_info =
      getTensorInfo<scalar_t, unsigned int>(dst);
  int dst_select_dim = dst_info.collapseDims(dim);
  dst_info.reduceDim(dst_select_dim);

  TensorInfo<scalar_t, unsigned int> src_info =
      getTensorInfo<scalar_t, unsigned int>(src);
  int src_select_dim = src_info.collapseDims(dim);
  src_info.reduceDim(src_select_dim);

  // The `src` is partitioned into two parts:
  // -the size of each slice we are indexing, which is the
  // total size of the tensor ignoring dimension `dim`;
  // -the number of indices we are choosing, which is the total size
  // of the tensor `indices`.
  // TODO: if the slice number is to large. Need to balance the work group and
  // work item number.
  // Make the work balance based on the MCU number.
  // auto __mcu = dpcppMaxComputeUnitSize(dev_id);
  uint64_t num_slices = indices.numel();

  auto slice_size = dst_num_elem / num_slices;

  auto& dpcpp_queue = dpcppGetCurrentQueue();
  auto dev_id = dpcppGetDeviceIdOfCurrentQueue();
  auto wgroup_size = dpcppMaxWorkGroupSize(dev_id);
  wgroup_size = std::min(decltype(wgroup_size)(slice_size), wgroup_size);

  auto n_work_item_iter = (slice_size + wgroup_size - 1) / wgroup_size;

  auto cgf = DPCPP_Q_CGF(__cgh) {
    auto src_data = src.data_ptr<scalar_t>();
    auto dst_data = dst.data_ptr<scalar_t>();
    auto idx_data = indices.data_ptr<int64_t>();

    __cgh.parallel_for<DPCPP_K(index_select_ker, scalar_t)>(
      DPCPP::nd_range</*dim=*/1>(
        DPCPP::range</*dim=*/1>(num_slices*wgroup_size),
        DPCPP::range</*dim=*/1>(wgroup_size)),
        [=](DPCPP::nd_item<1> item_id) {
          auto src_ptr = src_data;
          auto dst_ptr = dst_data;
          auto idx_ptr = idx_data;

          auto dst_slice_id = item_id.get_group(0);

          auto slice_off = IndexToOffset<int64_t, unsigned int>::get(
              dst_slice_id, indices_info);
          auto src_slice_id = idx_ptr[slice_off] /* - TH_INDEX_BASE*/;

          auto g_src_ptr =
              src_ptr + src_slice_id * src_info.strides[src_select_dim];
          auto g_dst_ptr =
              dst_ptr + dst_slice_id * dst_info.strides[dst_select_dim];

          auto ii_ = item_id.get_local_id(0);
          auto src_offset_ =
              IndexToOffset<scalar_t, unsigned int>::get(ii_, src_info);
          auto dst_offset_ =
              IndexToOffset<scalar_t, unsigned int>::get(ii_, dst_info);

          g_dst_ptr[dst_offset_] = g_src_ptr[src_offset_];

          for (int iter = 1; iter < n_work_item_iter; iter++) {
            auto __inner_idx = iter * wgroup_size + ii_;
            if (__inner_idx < slice_size) {
              src_offset_ = IndexToOffset<scalar_t, unsigned int>::get(
                  __inner_idx, src_info);
              dst_offset_ = IndexToOffset<scalar_t, unsigned int>::get(
                  __inner_idx, dst_info);

              g_dst_ptr[dst_offset_] = g_src_ptr[src_offset_];
            }
          }
        });
  };

  DPCPP_Q_ASYNC_SUBMIT(dpcpp_queue, cgf);
  return;
}

DPCPP_DEF_K1(nonzero_kernel);

template <typename T>
struct NonZeroOp {
  NonZeroOp() {}
  bool operator()(T lhs) const {
    if (Numerics<T>::ne(lhs, ScalarConvert<float, T>::to(0.0))) {
      return true;
    } else {
      return false;
    }
  }
};

#ifdef USE_ONEDPL
template <typename Iterator>
class strided_range {
public:
  using difference_type = typename std::iterator_traits<Iterator>::difference_type;

  struct stride_functor
  {
    using reference = typename std::iterator_traits<Iterator>::reference;
    difference_type stride;
    Iterator base;

    stride_functor(difference_type stride, Iterator begin)
      : stride(stride), base(begin) {}

    reference operator()(const difference_type& i) const
    {
      return base[stride * i];
    }
  };

  using CountingIterator = typename oneapi::dpl::counting_iterator<difference_type>;
  // type of the strided_range iterator
  using TransformIterator = typename oneapi::dpl::transform_iterator<CountingIterator, stride_functor>;

  // construct strided_range for the range [first,last)
  strided_range(Iterator first, Iterator last, difference_type stride)
    : first(first), last(last), stride(stride) {}

  TransformIterator begin(void) const
  {
    return TransformIterator(CountingIterator(0),
                             stride_functor(stride, first));
  }

  TransformIterator end(void) const
  {
    return begin() + ((last - first) + (stride - 1)) / stride;
  }

protected:
  Iterator first;
  Iterator last;
  difference_type stride;
};
#endif


template <typename scalar_t>
void nonzero(Tensor& tensor, const Tensor& self_) {
  auto self = self_.contiguous();

  int64_t num_dim = self.dim() == 0 ? 1 : self.dim();
  int64_t N = self.numel();

  // First to resize out tensor to full elements row
  tensor = tensor.resize_({N, num_dim}).contiguous();

  // Prepare input tensor strides for calculating result index
  if (N > 0) {
#if defined(USE_ONEDPL)
    auto& dpcpp_queue = dpcppGetCurrentQueue();
    auto policy = oneapi::dpl::execution::make_device_policy(dpcpp_queue);
    auto tensor_begin = tensor.data_ptr<long>();
    auto self_begin = self.data_ptr<scalar_t>();
    strided_range<decltype(tensor_begin)> strided_tensor(tensor_begin, tensor_begin + N*num_dim, num_dim);
    oneapi::dpl::counting_iterator<int64_t> idxfirst(0);
    auto start = oneapi::dpl::make_zip_iterator(idxfirst, self_begin);
    auto dend =
      std::copy_if(policy, start, start + N,
        oneapi::dpl::make_transform_iterator(strided_tensor.begin(), [](auto& x) {return std::forward_as_tuple(x, std::ignore);}),
        [](auto h){
          using std::get;
          return NonZeroOp<scalar_t>{}(get<1>(h));
        });

    auto num_nonzeros = std::distance(strided_tensor.begin(), dend.base());

    if (num_nonzeros > 0 && num_dim > 0) {
      int64_t div = 1;
      for (int dim = num_dim - 1; dim >= 0; dim--) {
        strided_range<decltype(tensor_begin)> stride_dim(tensor_begin + dim,
                                                         tensor_begin + N * num_dim, num_dim);
        auto dim_size = self.size(dim);
        std::transform(policy,
                       strided_tensor.begin(),
                       strided_tensor.begin() + num_nonzeros,
                       stride_dim.begin(),
                       [=](long linear_id) {
                         return (linear_id / div) % dim_size;
                       });
        div *= dim_size;
      }
    }
  tensor.resize_({num_nonzeros, num_dim});
#else
    throw std::runtime_error("no oneDPL found when compile. USM nonzero not supported");
#endif

  }
}

DPCPP_DEF_K1(index_add_ker);
template <typename scalar_t>
void indexAdd(
    Tensor& dst,
    int64_t dim,
    const Tensor& indices,
    const Tensor& src) {
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

  TORCH_CHECK(dst.dim() <= MAX_DPCPPTORCH_DIMS, DPCPPTORCH_DIM_WARNING);
  TORCH_CHECK(src.dim() <= MAX_DPCPPTORCH_DIMS, DPCPPTORCH_DIM_WARNING);

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

  TensorInfo<int64_t, unsigned int> indices_info =
      getTensorInfo<int64_t, unsigned int>(indices);
  indices_info.collapseDims();

  TensorInfo<scalar_t, unsigned int> dst_info =
      getTensorInfo<scalar_t, unsigned int>(dst);
  int dst_add_dim = dst_info.collapseDims(dim);
  dst_info.reduceDim(dst_add_dim);

  TensorInfo<scalar_t, unsigned int> src_info =
      getTensorInfo<scalar_t, unsigned int>(src);
  int src_add_dim = src_info.collapseDims(dim);
  src_info.reduceDim(src_add_dim);

  auto& dpcpp_queue = dpcppGetCurrentQueue();
  auto dev_id = dpcppGetDeviceIdOfCurrentQueue();
  auto wgroup_size = dpcppMaxWorkGroupSize(dev_id);
  wgroup_size = std::min(decltype(wgroup_size)(sliceSize), wgroup_size);

  auto n_work_item_iter = (sliceSize + wgroup_size - 1) / wgroup_size;

  auto cgf = DPCPP_Q_CGF(__cgh) {
    auto src_data = src.data_ptr<scalar_t>();
    auto dst_data = dst.data_ptr<scalar_t>();
    auto idx_data = indices.data_ptr<long>();

    __cgh.parallel_for<DPCPP_K(index_add_ker, scalar_t)>(
      DPCPP::nd_range</*dim=*/1>(
        DPCPP::range</*dim=*/1>(numIndices*wgroup_size),
        DPCPP::range</*dim=*/1>(wgroup_size)),
        [=](DPCPP::nd_item<1> item_id) {
          auto src_ptr = src_data;
          auto dst_ptr = dst_data;
          auto idx_ptr = idx_data;

          auto dst_slice_id = item_id.get_group(0);
          // auto slice_off = IndexToOffset<int64_t, unsigned
          // int>::get(dst_slice_id, indices_info);
          auto g_idx_ptr = idx_ptr;
          auto g_dst_ptr =
              dst_ptr + g_idx_ptr[dst_slice_id] * dst_info.strides[dst_add_dim];
          auto g_src_ptr =
              src_ptr + dst_slice_id * src_info.strides[src_add_dim];

          auto ii_ = item_id.get_local_id(0);
          auto dst_offset_ =
              IndexToOffset<scalar_t, unsigned int>::get(ii_, dst_info);
          auto src_offset_ =
              IndexToOffset<scalar_t, unsigned int>::get(ii_, src_info);
          g_dst_ptr[dst_offset_] += g_src_ptr[src_offset_];

          for (int iter = 1; iter < n_work_item_iter; iter++) {
            auto idx_offset_ =
                IndexToOffset<int64_t, unsigned int>::get(iter, indices_info);
            auto __inner_idx = g_idx_ptr[idx_offset_] * wgroup_size + ii_;
            auto __src_idx = idx_offset_ * wgroup_size + ii_;

            if (__src_idx < srcTotalSize) {
              dst_offset_ = IndexToOffset<scalar_t, unsigned int>::get(
                  __inner_idx, dst_info);
              src_offset_ = IndexToOffset<scalar_t, unsigned int>::get(
                  __src_idx, src_info);
              g_dst_ptr[dst_offset_] += g_src_ptr[src_offset_];
            }
          }
        });
  };

  DPCPP_Q_ASYNC_SUBMIT(dpcpp_queue, cgf);
}

DPCPP_DEF_K1(index_fill_ker);
template <typename scalar_t>
void indexFill(
    Tensor& dst,
    int64_t dim,
    const Tensor& indices,
    Scalar val_scalar) {
  TORCH_CHECK(dst.dim() <= MAX_DPCPPTORCH_DIMS, DPCPPTORCH_DIM_WARNING);
  TORCH_CHECK(indices.dim() <= MAX_DPCPPTORCH_DIMS, DPCPPTORCH_DIM_WARNING);

  // The `src` is partitioned into two parts:
  // -the size of each slice we are indexing, which is the
  // total size of the tensor ignoring dimension `dim`;
  // -the number of indices we are choosing, which is the total size
  // of the tensor `indices`.
  int dstDims = dst.dim() == 0 ? 1 : dst.dim();

  TORCH_CHECK(indices.dim() <= 1, "expecting vector of indices");
  TORCH_CHECK(dim >= 0 && dim < dstDims, "Indexing dim is out of bounds");

  auto val = val_scalar.to<scalar_t>();
  ptrdiff_t sliceSize = 1;
  for (int d = 0; d < dstDims; d++) {
    if (d != dim) {
      sliceSize *= dst.dim() == 0 ? 1 : dst.size(d);
    }
  }
  ptrdiff_t dstTotalSize = dst.numel();
  int64_t dstFillDimSize = dst.dim() == 0 ? 1 : dst.dim();
  ptrdiff_t numIndices = indices.numel();

  if (sliceSize == 0) {
    return;
  }

  TensorInfo<int64_t, unsigned int> indices_info =
      getTensorInfo<int64_t, unsigned int>(indices);
  indices_info.collapseDims();

  TensorInfo<scalar_t, unsigned int> dst_info =
      getTensorInfo<scalar_t, unsigned int>(dst);
  int dst_fill_dim = dst_info.collapseDims(dim);
  dst_info.reduceDim(dst_fill_dim);

  auto& dpcpp_queue = dpcppGetCurrentQueue();
  auto dev_id = dpcppGetDeviceIdOfCurrentQueue();
  auto wgroup_size = dpcppMaxWorkGroupSize(dev_id);
  wgroup_size = std::min(decltype(wgroup_size)(sliceSize), wgroup_size);
  auto n_work_item_iter = (sliceSize + wgroup_size - 1) / wgroup_size;

  auto cgf = DPCPP_Q_CGF(__cgh) {
    auto dst_data =
        dst.data_ptr<scalar_t>();
    auto idx_data = indices.data_ptr<long>();

    __cgh.parallel_for<DPCPP_K(index_fill_ker, scalar_t)>(
      DPCPP::nd_range</*dim=*/1>(
        DPCPP::range</*dim=*/1>(numIndices*wgroup_size),
        DPCPP::range</*dim=*/1>(wgroup_size)),
        [=](DPCPP::nd_item<1> item_id) {
          auto dst_ptr = dst_data;
          auto idx_ptr = idx_data;

          auto dst_slice_id = item_id.get_group(0);
          // auto slice_off = IndexToOffset<int64_t, unsigned
          // int>::get(dst_slice_id, indices_info);
          auto g_idx_ptr = idx_ptr;
          auto g_dst_ptr = dst_ptr +
              g_idx_ptr[dst_slice_id] * dst_info.strides[dst_fill_dim];

          auto ii_ = item_id.get_local_id(0);
          auto dst_offset_ =
              IndexToOffset<scalar_t, unsigned int>::get(ii_, dst_info);
          g_dst_ptr[dst_offset_] = val;

          for (int iter = 1; iter < n_work_item_iter; iter++) {
            auto idx_offset_ =
                IndexToOffset<int64_t, unsigned int>::get(iter, indices_info);
            auto __inner_idx = g_idx_ptr[idx_offset_] * wgroup_size + ii_;

            if (__inner_idx < dstTotalSize) {
              dst_offset_ = IndexToOffset<scalar_t, unsigned int>::get(
                  __inner_idx, dst_info);

              g_dst_ptr[dst_offset_] = val;
            }
            }
        });
  };

  DPCPP_Q_ASYNC_SUBMIT(dpcpp_queue, cgf);
}


DPCPP_DEF_K1(index_copy_ker);
template <typename scalar_t>
void indexCopy(
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
  int64_t dstFillDimSize = dst.dim() == 0 ? 1 : dst.dim();
  ptrdiff_t numIndices = indices.numel();

  if (sliceSize == 0) {
    return;
  }

  TensorInfo<int64_t, unsigned int> indices_info =
      getTensorInfo<int64_t, unsigned int>(indices);
  indices_info.collapseDims();

  TensorInfo<scalar_t, unsigned int> src_info =
      getTensorInfo<scalar_t, unsigned int>(source);
  int src_dim = src_info.collapseDims(0);
  src_info.reduceDim(src_dim);

  TensorInfo<scalar_t, unsigned int> dst_info =
      getTensorInfo<scalar_t, unsigned int>(dst);
  int dst_fill_dim = dst_info.collapseDims(dim);
  dst_info.reduceDim(dst_fill_dim);

  auto& dpcpp_queue = dpcppGetCurrentQueue();
  auto dev_id = dpcppGetDeviceIdOfCurrentQueue();
  auto wgroup_size = dpcppMaxWorkGroupSize(dev_id);
  wgroup_size = std::min(decltype(wgroup_size)(sliceSize), wgroup_size);
  auto n_work_item_iter = (sliceSize + wgroup_size - 1) / wgroup_size;

  auto cgf = DPCPP_Q_CGF(cgh) {
    auto dst_data = dst.data_ptr<scalar_t>();
    auto src_data = source.data_ptr<scalar_t>();
    auto idx_data = indices.data_ptr<long>();

    auto kfn = DPCPP_Q_KFN(DPCPP::nd_item<1> item_id) {
        auto dst_ptr = dst_data;
        auto src_ptr = src_data;
        auto idx_ptr = idx_data;

        auto dst_slice_id = item_id.get_group(0);
        auto g_idx_ptr = idx_ptr;
        auto g_dst_ptr = dst_ptr + g_idx_ptr[dst_slice_id] * dst_info.strides[dst_fill_dim];
        auto g_src_ptr = src_ptr + dst_slice_id * src_info.strides[src_dim];

        auto ii_ = item_id.get_local_id(0);
        auto dst_offset_ = IndexToOffset<scalar_t, unsigned int>::get(ii_, dst_info);
        auto src_offset_ = IndexToOffset<scalar_t, unsigned int>::get(ii_, src_info);
        g_dst_ptr[dst_offset_] = g_src_ptr[src_offset_];

        for (int iter = 1; iter < n_work_item_iter; iter++) {
          auto idx_offset_ =
              IndexToOffset<int64_t, unsigned int>::get(iter, indices_info);
          auto __inner_idx = g_idx_ptr[idx_offset_] * wgroup_size + ii_;

          if (__inner_idx < dstTotalSize) {
            dst_offset_ = IndexToOffset<scalar_t, unsigned int>::get(
                __inner_idx, dst_info);

            g_dst_ptr[dst_offset_] = src_ptr[ii_];
          }
        }

    };

    cgh.parallel_for<DPCPP_K(
      index_copy_ker, scalar_t)>(
      DPCPP::nd_range<1>(DPCPP::range<1>(numIndices * wgroup_size), DPCPP::range<1>(wgroup_size)), kfn);
  };

  DPCPP_Q_ASYNC_SUBMIT(dpcpp_queue, cgf);
}

DPCPP_DEF_K1(diag_from_dpcpp_ker);
DPCPP_DEF_K1(diag_to_dpcpp_ker);
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
    int64_t size = (k > 0) ? DPCPP::min((int64_t)size0, (int64_t)size1 - k)
                           : DPCPP::min((int64_t)size0 + k, (int64_t)size1);
    int64_t size_[1] = {size};
    TensorImpl_resizeNd(TensorImpl_Unwrap(dst), 1, size_, nullptr);
    if (size > 0) {
      int64_t strideSelf = dst.dim() == 0 ? 1 : dst.stride(0);
      int64_t start = (k >= 0 ? k * stride1 : -k * stride0);
      static const auto write_mode = DPCPP::access::mode::discard_write;
      static const auto read_mode = DPCPP::access::mode::read;
      auto& dpcpp_queue = dpcppGetCurrentQueue();

      auto cgf = DPCPP_Q_CGF(cgh) {
        auto in_data = src.data_ptr<scalar_t>();
        auto out_data = dst.data_ptr<scalar_t>();
        auto kfn = DPCPP_Q_KFN(DPCPP::item<1> item_id) {
          size_t id = item_id.get_id(0);
          auto in_ptr = in_data;
          auto out_ptr = out_data;
          const int64_t bOffset = start + (stride0 + stride1) * id;
          out_ptr[strideSelf * id] = in_ptr[bOffset];
        };
        cgh.parallel_for<DPCPP_K(diag_from_dpcpp_ker, scalar_t)>(
            DPCPP::range<1>(dst.numel()), kfn);
      };
      DPCPP_Q_ASYNC_SUBMIT(dpcpp_queue, cgf);
    }
  } else {
    int64_t totalElements = src.numel();
    int64_t size = (k > 0) ? totalElements + k : totalElements - k;
    int64_t strideSrc = src.dim() == 0 ? 1 : src.stride(0);
    int64_t size_[2] = {size, size};
    TensorImpl_resizeNd(TensorImpl_Unwrap(dst), 2, size_, nullptr);
    dst.zero_();
    if (size > 0) {
      int64_t stride0 = dst.stride(0);
      int64_t stride1 = dst.stride(1);
      int64_t start = (k >= 0 ? k * stride1 : -k * stride0);
      static const auto write_mode = DPCPP::access::mode::discard_write;
      static const auto read_mode = DPCPP::access::mode::read;
      auto& dpcpp_queue = dpcppGetCurrentQueue();

      auto cgf = DPCPP_Q_CGF(cgh) {
        auto in_data = src.data_ptr<scalar_t>();
        auto out_data = dst.data_ptr<scalar_t>();
        auto kfn = DPCPP_Q_KFN(DPCPP::item<1> item_id) {
          size_t id = item_id.get_id(0);
          auto in_ptr = in_data;
          auto out_ptr = out_data;
          const int64_t aOffset = start + (stride0 + stride1) * id;
          out_ptr[aOffset] = in_ptr[strideSrc * id];
        };
        cgh.parallel_for<DPCPP_K(diag_to_dpcpp_ker, scalar_t)>(
            DPCPP::range<1>(dst.numel()), kfn);
      };
      DPCPP_Q_ASYNC_SUBMIT(dpcpp_queue, cgf);
    }
  }
}

template <typename T, typename MaskT>
struct TensorMaskedFillOp {
  TensorMaskedFillOp(T v) : value(v) {}
  inline void operator()(T& t, MaskT& mask) const {
    if (mask) {
      t = value;
    }
  }

  T value;
};

template <typename scalar_t>
void MaskedFillBool(Tensor& tensor, const Tensor& mask, Scalar value_scalar) {
  auto value = value_scalar.to<scalar_t>();
  TORCH_CHECK(tensor.numel() == mask.numel(), "sizes do not match");
  DPCPP_tensor_apply2<scalar_t, bool>(
      tensor, mask, TensorMaskedFillOp<scalar_t, bool>(value));
}

template <typename scalar_t>
void MaskedFill(Tensor& tensor, const Tensor& mask, Scalar value_scalar) {
  auto value = value_scalar.to<scalar_t>();
  TORCH_CHECK(tensor.numel() == mask.numel(), "sizes do not match");
  DPCPP_tensor_apply2<scalar_t, uint8_t>(
      tensor, mask, TensorMaskedFillOp<scalar_t, unsigned char>(value));
}

DPCPP_DEF_K1(maskedScatter_scan_dpcpp_ker);
DPCPP_DEF_K1(TensorMaskedScatterOp);
template <typename scalar_t>
void MaskedScatter(Tensor& tensor, const Tensor& mask_, const Tensor& src) {
  Tensor mask;
  std::tie(mask) = expand_inplace(tensor, mask_, "masked_scatter_");
  auto maskSize = mask.numel();
  auto tensorSize = tensor.numel();
  auto srcSize = src.numel();

  // `mask` and `tensor` must have the same number of elements
  TORCH_CHECK(
      maskSize == tensorSize,
      "mask and tensor must have the same number of elements");

  // Determine our output size
  c10::optional<ScalarType> dtype;
  auto totalElements = at::AtenIpexTypeXPU::sum(mask, dtype).item().to<int>();

  // The number of `1` elements present in the mask must be <= the
  // number of elements available in `src`
  if (totalElements > srcSize) {
    TORCH_CHECK(false, "source nElements must be == mask `1` elements");
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

  auto& dpcpp_queue = dpcppGetCurrentQueue();
  int64_t rng, GRange, tileSize;
  parallel_for_setup(size, tileSize, rng, GRange);

  // command group functions
  auto cgf = DPCPP_Q_CGF(cgh) {
    auto acc_maskLong_data = maskLong.data_ptr<int64_t>();
    auto acc_maskPrefixSum_data = maskPrefixSum.data_ptr<int64_t>();

    // kernel function per work-item
    auto kfn = DPCPP_Q_KFN() {
      auto maskLong_ptr = acc_maskLong_data;
      auto maskPrefixSum_ptr = acc_maskPrefixSum_data;
      dpcpp_exclusive_scan(
          maskLong_ptr,
          maskLong_ptr + size,
          maskPrefixSum_ptr,
          static_cast<int64_t>(0),
          AddOp<int64_t>());
    };
    // kick off kernel
    // (TODO) single_task need replaced due to low efficiency
    cgh.single_task<DPCPP_K(maskedScatter_scan_dpcpp_ker, scalar_t)>(kfn);
  };
  // submit to DPCPP queue
  DPCPP_Q_ASYNC_SUBMIT(dpcpp_queue, cgf);

  Tensor contigSrc = src.contiguous();

  // command group function
  // copy src to tensor according to mask
  auto cgfMaskedScatter = DPCPP_Q_CGF(cgh) {
    auto acc_src_data =
        contigSrc.data_ptr<scalar_t>();
    auto acc_mask_data = mask.data_ptr<bool>();
    auto acc_maskPrefixSum_data =
        maskPrefixSum.data_ptr<int64_t>();
    auto acc_tensor_data =
        tensor.data_ptr<scalar_t>();

    // kernel function
    auto kfn = DPCPP_Q_KFN(DPCPP::nd_item<1> item) {
      int64_t linear_index = item.get_global_linear_id();
      auto src_ptr = acc_src_data;
      auto mask_ptr = acc_mask_data;
      auto maskPrefix_ptr = acc_maskPrefixSum_data;
      auto tensor_ptr = acc_tensor_data;
      if (linear_index < size) {
        if (mask_ptr[linear_index]) {
          tensor_ptr[linear_index] = src_ptr[maskPrefix_ptr[linear_index]];
        }
      }
    };

    cgh.parallel_for<DPCPP_K(TensorMaskedScatterOp, scalar_t)>(
        DPCPP::nd_range<1>(DPCPP::range<1>(GRange), DPCPP::range<1>(tileSize)),
        kfn);
  };

  // submit to DPCPP queue
  DPCPP_Q_ASYNC_SUBMIT(dpcpp_queue, cgfMaskedScatter);
}

DPCPP_DEF_K1(maskedSelect_scan_dpcpp_ker);
DPCPP_DEF_K1(TensorMaskedSelectOp);
template <typename scalar_t>
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

  auto& dpcpp_queue = dpcppGetCurrentQueue();
  int64_t rng, GRange, tileSize;
  parallel_for_setup(size, tileSize, rng, GRange);

  // command group functions
  auto cgf = DPCPP_Q_CGF(cgh) {
    auto acc_maskLong_data =
        maskLong.data_ptr<int64_t>();
    auto acc_maskPrefixSum_data = maskPrefixSum.data_ptr<int64_t>();

    // kernel function per work-item
    auto kfn = DPCPP_Q_KFN() {
      auto maskLong_ptr = acc_maskLong_data;
      auto maskPrefixSum_ptr = acc_maskPrefixSum_data;
      dpcpp_inclusive_scan(
          maskLong_ptr,
          maskLong_ptr + size,
          maskPrefixSum_ptr,
          AddOp<int64_t>());
    };
    // kick off kernel
    // (TODO) single_task need replaced due to low efficiency
    cgh.single_task<DPCPP_K(maskedSelect_scan_dpcpp_ker, scalar_t)>(kfn);
  };

  // submit to DPCPP queue
  DPCPP_Q_ASYNC_SUBMIT(dpcpp_queue, cgf);

  TensorInfo<scalar_t, uint64_t> src_info =
      getTensorInfo<scalar_t, uint64_t>(src);
  src_info.collapseDims();

  TensorInfo<bool, uint64_t> mask_info = getTensorInfo<bool, uint64_t>(mask);
  mask_info.collapseDims();

  // command group function
  auto cgfMaskedSelect = DPCPP_Q_CGF(cgh) {
    auto acc_src_data = src.data_ptr<scalar_t>();
    auto acc_mask_data = mask.data_ptr<bool>();
    auto acc_maskPrefixSum_data = maskPrefixSum.data_ptr<int64_t>();
    auto acc_tensor_data = tensorContig.data_ptr<scalar_t>();

    // kernel function per work-item
    auto kfn = DPCPP_Q_KFN(DPCPP::nd_item<1> item) {
      int64_t linear_index = item.get_global_linear_id();

      auto src_ptr = acc_src_data;
      auto mask_ptr = acc_mask_data;
      auto maskPrefix_ptr = acc_maskPrefixSum_data;
      auto tensor_ptr = acc_tensor_data;

      if (linear_index < size) {
        // The mask tensor maybe not contiguos.
        auto mask_offset =
            IndexToOffset<bool, uint64_t>().get(linear_index, mask_info);
        if (mask_ptr[mask_offset]) {
          // The src tensor maybe not contiguos.
          auto src_offset =
              IndexToOffset<scalar_t, uint64_t>().get(linear_index, src_info);
          tensor_ptr[maskPrefix_ptr[linear_index] - 1] = src_ptr[src_offset];
        }
      }
    };
    cgh.parallel_for<DPCPP_K(TensorMaskedSelectOp, scalar_t)>(
        DPCPP::nd_range<1>(DPCPP::range<1>(GRange), DPCPP::range<1>(tileSize)),
        kfn);
  };

  // submit to DPCPP queue
  DPCPP_Q_ASYNC_SUBMIT(dpcpp_queue, cgfMaskedSelect);

  if (&tensor != &tensorContig) {
    tensor.copy_(tensorContig);
  }
}

template <int N>
struct alignas(N) OpaqueType {
  char data[N];
};

DPCPP_DEF_K1(dpcpp_put_kernel);
template <typename scalar_t, typename kernel_name, typename Func>
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

    auto kfn = DPCPP_Q_KFN(DPCPP::item<1> item_id) {
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

    __cgh.parallel_for<DPCPP_K(dpcpp_put_kernel, kernel_name)>(
        DPCPP::range</*dim=*/1>(numel), kfn);
  };
  DPCPP_Q_ASYNC_SUBMIT(dpcpp_queue, cgf);
}

DPCPP_DEF_K1(index_kernel);
void index(
    TensorIterator& iter,
    IntArrayRef index_size,
    IntArrayRef index_stride,
    IntArrayRef non_index_size,
    IntArrayRef non_index_stride) {
  IPEX_DISPATCH_ALL_TYPES_AND3(
      at::ScalarType::BFloat16,
      at::ScalarType::Half,
      at::ScalarType::Bool,
      iter.dtype(),
      "index",
      [&] {
        using dtype = OpaqueType<sizeof(scalar_t)>;
        dpcpp_index_kernel<DPCPP_K(index_kernel, scalar_t)>(
            iter,
            index_size,
            index_stride,
            non_index_size,
            non_index_stride,
            [](char *out_data,
               char *in_data,
               int64_t offset) {
              *(dtype*)out_data = *(dtype*)(in_data + offset);
            });
      });
}

DPCPP_DEF_K1(index_put_kernel);
void index_put(
    TensorIterator& iter,
    IntArrayRef index_size,
    IntArrayRef index_stride,
    bool accumulate) {
  if (accumulate) {
    IPEX_DISPATCH_ATOMIC_ALL_TYPES(iter.dtype(), "index_put", [&] {
      dpcpp_index_kernel<DPCPP_K(
          index_put_kernel,
          scalar_t,
          /*accumulate=*/bool)>(
          iter,
          index_size,
          index_stride,
          IntArrayRef{},
          IntArrayRef{},
          [](char *out_data,
             char *in_data,
             int64_t offset) {
            dpcpp_global_ptr_pt<scalar_t> out_ptr =
                (dpcpp_global_ptr_pt<scalar_t>)(out_data + offset);
            auto in = *(scalar_t*)in_data;
            atomicAdd(out_ptr, in);
          });
    });
  } else {
    IPEX_DISPATCH_ALL_TYPES_AND3(
      at::ScalarType::BFloat16,
      at::ScalarType::Half,
      at::ScalarType::Bool,
      iter.dtype(),
      "index_put",
      [&] {
        using dtype = OpaqueType<sizeof(scalar_t)>;
        dpcpp_index_kernel<DPCPP_K(index_put_kernel, scalar_t)>(
            iter,
            index_size,
            index_stride,
            IntArrayRef{},
            IntArrayRef{},
            [](char *out_data,
               char *in_data,
               int64_t offset) {
              *(dtype*)(out_data + offset) = *(dtype*)in_data;
            });
      });
  }
}

DPCPP_DEF_K1(take_ker);
template <typename scalar_t>
void Take(Tensor& dst, const Tensor& src, const Tensor& index) {
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

  auto& dpcpp_queue = dpcppGetCurrentQueue();
  auto cgf = DPCPP_Q_CGF(cgh) {
    auto src_data = src.data_ptr<scalar_t>();
    auto dst_data = dst.data_ptr<scalar_t>();
    auto idx_data = index.data_ptr<int64_t>();

    auto kfn = DPCPP_Q_KFN(DPCPP::item<1> item) {
      auto src_ptr = src_data;
      auto dst_ptr = dst_data;
      auto idx_ptr = idx_data;

      auto idx = item.get_linear_id();
      auto offset = idx_ptr[idx];
      dst_ptr[idx] = src_ptr[offset];
    };

    cgh.parallel_for<DPCPP_K(take_ker, scalar_t)>(
        DPCPP::range<1>(dst_num_elem), kfn);
  };

  DPCPP_Q_ASYNC_SUBMIT(dpcpp_queue, cgf);
}

} // namespace impl

Tensor& index_select_out(
    Tensor& out,
    const Tensor& self,
    int64_t dim,
    const Tensor& index) {
  IPEX_DISPATCH_ALL_TYPES_AND3(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      at::ScalarType::Bool,
      self.scalar_type(),
      "indexSelect",
      [&]() { impl::indexSelect<scalar_t>(out, self, dim, index); });
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

Tensor& index_add_(
    Tensor& self,
    int64_t dim,
    const Tensor& index,
    const Tensor& source) {
  IPEX_DISPATCH_ALL_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      self.scalar_type(),
      "indexAdd",
      [&]() { impl::indexAdd<scalar_t>(self, dim, index, source); });
  return self;
}

Tensor& index_copy_(Tensor& self, int64_t dim, const Tensor& index, const Tensor& source) {
  dim = maybe_wrap_dim(dim, self.dim());
  TORCH_CHECK_INDEX(index.dim() < 2, "index_copy_(): Index should have dimension 1 or 0 (got ", index.dim(), ")");

  int64_t numIndices = index.numel();
  if (source.dim() == 0 && numIndices != 1) {
    TORCH_CHECK_INDEX(false, "index_copy_(): When source is scalar, index should have one element (got ", numIndices, ")");
  } else if ((source.dim() != self.dim()) && (source.dim() != 0 && self.dim() != 0)) {
    TORCH_CHECK_INDEX(false, "index_copy_(): When source and destination are not scalars, their dimensionality must match. Source dimensionality (",
                   source.dim(), "), destination dimensionality (", self.dim(), ")");
  }

  TORCH_CHECK_INDEX(index.scalar_type() == ScalarType::Long, "index_copy_(): Expected LongTensor for index");
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
      !std::equal(selfSlicedSizes.begin(), selfSlicedSizes.end(),
                  sourceSlicedSizes.begin())) {
    std::stringstream ss;
    ss << "index_copy_(): Source/destination tensor must have same slice shapes. ";
    ss << "Destination slice shape: " << selfSlicedSizes << " at dimension " << dim;
    ss << " and source slice shape: " << sourceSlicedSizes << " at dimension 0.";
    TORCH_CHECK(false, ss.str());
  }
  TORCH_CHECK_INDEX(source.dim() == 0 || numIndices == source.size(dim),
          "index_copy_(): Number of indices (", numIndices, ") should be equal to source.size(dim) (", source.size(dim), ")");

  IPEX_DISPATCH_ALL_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      self.scalar_type(),
      "indexCopy",
      [&]() { impl::indexCopy<scalar_t>(self, dim, index, source); });
  return self;
}

Tensor& index_fill_(
    Tensor& self,
    int64_t dim,
    const Tensor& index,
    Scalar value) {
  IPEX_DISPATCH_ALL_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      self.scalar_type(),
      "indexFill",
      [&]() { impl::indexFill<scalar_t>(self, dim, index, value); });
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
  IPEX_DISPATCH_ALL_TYPES_AND3(
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

Tensor& masked_fill_(Tensor& self, const Tensor& mask_, Scalar value) {
  Tensor mask;
  std::tie(mask) = expand_inplace(self, mask_, "masked_fill_");
  IPEX_DISPATCH_ALL_TYPES_AND2(
      at::ScalarType::BFloat16,
      at::ScalarType::Half,
      self.scalar_type(),
      "MaskedFill",
      [&]() {
        if (mask.dtype() == at::ScalarType::Byte) {
          impl::MaskedFill<scalar_t>(self, mask, value);
        } else {
          impl::MaskedFillBool<scalar_t>(self, mask, value);
        }
      });
  return self;
}

Tensor& masked_fill_(Tensor& self, const Tensor& mask, const Tensor& value) {
  return at::AtenIpexTypeXPU::masked_fill_(self, mask, value.item());
}

Tensor& masked_scatter_(
    Tensor& self,
    const Tensor& mask,
    const Tensor& source) {
  IPEX_DISPATCH_ALL_TYPES_AND2(
      at::ScalarType::BFloat16,
      at::ScalarType::Half,
      self.scalar_type(),
      "MaskedScatter",
      [&]() { impl::MaskedScatter<scalar_t>(self, mask, source); });
  return self;
}

Tensor& masked_select_out(Tensor& out, const Tensor& self, const Tensor& mask) {
  Tensor b_self, b_mask;
  std::tie(b_self, b_mask) = expand_outplace(self, mask, "masked_select_out");
  IPEX_DISPATCH_ALL_TYPES_AND2(
      at::ScalarType::BFloat16,
      at::ScalarType::Half,
      self.scalar_type(),
      "MaskedSelect",
      [&]() { impl::MaskedSelect<scalar_t>(out, b_self, b_mask); });
  return out;
}

Tensor masked_select(const Tensor& self, const Tensor& mask) {
  Tensor out = at::empty({0}, self.options());
  return at::AtenIpexTypeXPU::masked_select_out(out, self, mask);
}

DPCPP_DEF_K1(dpcpp_put_kernel);
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
    IPEX_DISPATCH_ATOMIC_ALL_TYPES(self.scalar_type(), "put_", [&] {
      impl::put<
          scalar_t,
          DPCPP_K(dpcpp_put_kernel, scalar_t, /*accumulate=*/bool)>(
          self,
          index,
          source,
          [](char* out_data,
             char* in_data,
             uint64_t offset) {
            dpcpp_global_ptr_pt<scalar_t> out_ptr =
                (dpcpp_global_ptr_pt<scalar_t>)(out_data + offset);
            auto in = *(scalar_t*)in_data;
            atomicAdd(out_ptr, in);
          });
    });
  } else {
    IPEX_DISPATCH_ALL_TYPES_AND3(
      at::ScalarType::BFloat16,
      at::ScalarType::Half,
      at::ScalarType::Bool,
      self.scalar_type(),
      "put_",
      [&] {
        using dtype = impl::OpaqueType<sizeof(scalar_t)>;
        impl::put<scalar_t, DPCPP_K(dpcpp_put_kernel, scalar_t)>(
            self,
            index,
            source,
            [](char* out_data,
               char* in_data,
               uint64_t offset) {
              *(dtype*)(out_data + offset) = *(dtype*)in_data;
            });
      });
  }

  return self;
}

Tensor index(const Tensor& self, TensorList indices) {
  TORCH_CHECK(
      indices.size() <= (size_t)self.dim(),
      "too many indices for tensor of dimension ",
      self.dim(),
      " (got ",
      indices.size(),
      ")");

  auto info = make_info(self, indices);
  auto iter = make_index_iterator(info);
  impl::index(iter, info.indexed_sizes, info.indexed_strides,
      info.non_indexed_sizes, info.non_indexed_strides);
  return iter.output();
}

Tensor& _index_put_impl_(
    Tensor& self,
    TensorList indices,
    const Tensor& value,
    const bool accumulate,
    const bool unsafe) {
  TORCH_CHECK(
      indices.size() <= (size_t)self.dim(),
      "too many indices for tensor of dimension ",
      self.dim(),
      " (got ",
      indices.size(),
      ")");

  auto info = make_info(self, indices);
  auto iter = make_index_put_iterator(info, value);
  impl::index_put(iter, info.indexed_sizes, info.indexed_strides, accumulate);
  return self;
}

Tensor& take_out(Tensor& out, const Tensor& self, const Tensor& index) {
  IPEX_DISPATCH_ALL_TYPES_AND2(
      at::ScalarType::BFloat16,
      at::ScalarType::Half,
      self.scalar_type(),
      "Take",
      [&]() { impl::Take<scalar_t>(out, self, index); });

  return out;
}

Tensor take(const Tensor& self, const Tensor& index) {
  Tensor out = at::empty({0}, self.options());
  return at::AtenIpexTypeXPU::take_out(out, self, index);
}

} // namespace AtenIpexTypeXPU
} // namespace at
