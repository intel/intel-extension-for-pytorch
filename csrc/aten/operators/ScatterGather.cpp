#include <ATen/ATen.h>
#include <ATen/MemoryOverlap.h>

#include <core/TensorImplUtils.h>
#include <core/detail/IndexUtils.h>
#include <core/detail/TensorInfo.h>
#include <utils/DPCPP.h>

#include "Loops.h"
#include "ReduceOpsUtils.h"
#include "ScatterGather.h"
#include "comm/ATDispatch.h"
#include "comm/RegistrationDeclarations.h"

constexpr int n_elems_per_work_item = UNROLLED_ELEM_PER_WORK_ITEM;

using namespace xpu::dpcpp::detail;
using namespace xpu::dpcpp;

namespace at {
namespace AtenIpexTypeXPU {
namespace impl {

#define RUN(TYPE, DIMS, REAL)                   \
  THDPCPPTensor_gatherKernel<TYPE, REAL, DIMS>( \
      tensorInfo, srcInfo, indexInfo, dim, (TYPE)totalElements);

template <typename scalar_t>
void Gather(
    Tensor& tensor,
    const Tensor& src,
    int64_t dim,
    const Tensor& index) {
  dim = maybe_wrap_dim(dim, tensor.dim());
  TORCH_CHECK(
      TensorImpl_nDimensionLegacyNoScalars(TensorImpl_Unwrap(index)) ==
          TensorImpl_nDimensionLegacyNoScalars(TensorImpl_Unwrap(src)),
      "Index tensor must have same dimensions as input tensor");
  TORCH_CHECK(
      TensorImpl_nDimensionLegacyNoScalars(TensorImpl_Unwrap(index)) ==
          TensorImpl_nDimensionLegacyNoScalars(TensorImpl_Unwrap(tensor)),
      "Index tensor must have same dimensions as output tensor");
  TORCH_CHECK(
      dim < TensorImpl_nDimensionLegacyNoScalars(TensorImpl_Unwrap(tensor)),
      "Index dimension is out of bounds");
  TORCH_CHECK(
      TensorImpl_nDimensionLegacyNoScalars(TensorImpl_Unwrap(src)) ==
          TensorImpl_nDimensionLegacyNoScalars(TensorImpl_Unwrap(tensor)),
      "Input tensor must have same dimensions as output tensor");

  for (int d = 0;
       d < TensorImpl_nDimensionLegacyNoScalars(TensorImpl_Unwrap(tensor));
       d++) {
    if (d != dim) {
      TORCH_CHECK(
          TensorImpl_sizeLegacyNoScalars(TensorImpl_Unwrap(tensor), d) ==
              TensorImpl_sizeLegacyNoScalars(TensorImpl_Unwrap(src), d),
          "Input tensor must have same size as output tensor apart from the "
          "specified dimension");
    }
  }

  TORCH_CHECK(
      TensorImpl_nDimensionLegacyNoScalars(TensorImpl_Unwrap(tensor)) <=
          MAX_DPCPPTORCH_DIMS,
      DPCPPTORCH_DIM_WARNING);

  const ptrdiff_t totalElements = index.numel();

  Tensor oldTensor;
  if (maybeOverlappingIndices(tensor)) {
    oldTensor = tensor;
    tensor = tensor.contiguous();
  }

  if (totalElements > 0) {
    if (canUse32BitIndexMath(tensor) && canUse32BitIndexMath(src) &&
        canUse32BitIndexMath(index)) {
      TensorInfo<scalar_t, unsigned int> tensorInfo =
          getTensorInfo<scalar_t, unsigned int>(tensor);
      TensorInfo<scalar_t, unsigned int> srcInfo =
          getTensorInfo<scalar_t, unsigned int>(src);
      TensorInfo<int64_t, unsigned int> indexInfo =
          getTensorInfo<int64_t, unsigned int>(index);
      // Specialize for a small number of dimensions.
      switch (indexInfo.dims) {
        case 1:
          RUN(unsigned int, 1, scalar_t);
          break;
        case 2:
          RUN(unsigned int, 2, scalar_t);
          break;
        case 3:
          RUN(unsigned int, 3, scalar_t);
          break;
        default:
          RUN(unsigned int, -1, scalar_t);
          break;
      }
    } else {
      TensorInfo<scalar_t, uint64_t> tensorInfo =
          getTensorInfo<scalar_t, uint64_t>(tensor);
      TensorInfo<scalar_t, uint64_t> srcInfo =
          getTensorInfo<scalar_t, uint64_t>(src);
      TensorInfo<int64_t, uint64_t> indexInfo =
          getTensorInfo<int64_t, uint64_t>(index);

      RUN(uint64_t, -1, scalar_t)
    }
  }

  if (oldTensor.defined()) {
    oldTensor.copy_(tensor);
    // TODO:copyIgnoringOverlaps Called when we are copying into an overlapping
    // index `dst`
  }
}

#undef RUN

#define RUN(TYPE, DIMS, REAL)                   \
  THSyclTensor_scatterKernel<TYPE, REAL, DIMS>( \
      tensorInfo, srcInfo, indexInfo, dim, (TYPE)totalElements);

template <typename scalar_t>
void Scatter(
    Tensor& tensor,
    int64_t dim,
    const Tensor& index,
    const Tensor& src) {
  dim = maybe_wrap_dim(dim, tensor.dim());
  int index_ndim_legacy_all =
      TensorImpl_nDimensionLegacyAll(TensorImpl_Unwrap(index));
  TORCH_CHECK(
      dim < TensorImpl_nDimensionLegacyNoScalars(TensorImpl_Unwrap(tensor)),
      "Index dimension is out of bounds");
  TORCH_CHECK(
      index_ndim_legacy_all == 0 ||
          TensorImpl_nDimensionLegacyNoScalars(TensorImpl_Unwrap(index)) ==
              TensorImpl_nDimensionLegacyNoScalars(TensorImpl_Unwrap(src)),
      "Index tensor must be either empty or have same dimensions as input "
      "tensor");
  TORCH_CHECK(
      TensorImpl_nDimensionLegacyNoScalars(TensorImpl_Unwrap(src)) ==
          TensorImpl_nDimensionLegacyNoScalars(TensorImpl_Unwrap(tensor)),
      "Input tensor must have same dimensions as output tensor");

  // no-op if index is empty
  if (index_ndim_legacy_all == 0)
    return;

  for (int d = 0;
       d < TensorImpl_nDimensionLegacyNoScalars(TensorImpl_Unwrap(tensor));
       d++) {
    int64_t indexSizeD =
        TensorImpl_sizeLegacyNoScalars(TensorImpl_Unwrap(index), d);
    if (d != dim) {
      TORCH_CHECK(
          indexSizeD <=
              TensorImpl_sizeLegacyNoScalars(TensorImpl_Unwrap(tensor), d),
          "Index tensor must not have larger size than output tensor "
          "apart from the specified dimension ",
          dim,
          ", but got index ",
          index.sizes(),
          " output ",
          tensor.sizes());
    }
    TORCH_CHECK(
        indexSizeD <= TensorImpl_sizeLegacyNoScalars(TensorImpl_Unwrap(src), d),
        "Index tensor must not have larger size than input tensor, but "
        "got index ",
        index.sizes(),
        "input",
        src.sizes());
  }

  TORCH_CHECK(tensor.dim() <= MAX_DPCPPTORCH_DIMS, DPCPPTORCH_DIM_WARNING);

  const ptrdiff_t totalElements = index.numel();

  Tensor oldTensor;
  if (maybeOverlappingIndices(tensor)) {
    oldTensor = tensor;
    tensor = tensor.contiguous();
  }

  if (totalElements > 0) {
    if (canUse32BitIndexMath(tensor) && canUse32BitIndexMath(src) &&
        canUse32BitIndexMath(index)) {
      TensorInfo<scalar_t, unsigned int> tensorInfo =
          getTensorInfo<scalar_t, unsigned int>(tensor);
      TensorInfo<scalar_t, unsigned int> srcInfo =
          getTensorInfo<scalar_t, unsigned int>(src);
      TensorInfo<int64_t, unsigned int> indexInfo =
          getTensorInfo<int64_t, unsigned int>(index);
      // Specialize for a small number of dimensions.
      switch (indexInfo.dims) {
        case 1:
          RUN(unsigned int, 1, scalar_t);
          break;
        case 2:
          RUN(unsigned int, 2, scalar_t);
          break;
        case 3:
          RUN(unsigned int, 3, scalar_t);
          break;
        default:
          RUN(unsigned int, -1, scalar_t);
          break;
      }
    } else {
      TensorInfo<scalar_t, uint64_t> tensorInfo =
          getTensorInfo<scalar_t, uint64_t>(tensor);
      TensorInfo<scalar_t, uint64_t> srcInfo =
          getTensorInfo<scalar_t, uint64_t>(src);
      TensorInfo<int64_t, uint64_t> indexInfo =
          getTensorInfo<int64_t, uint64_t>(index);

      RUN(uint64_t, -1, scalar_t)
    }
  }

  if (oldTensor.defined()) {
    oldTensor.copy_(tensor);
    // TODO:copyIgnoringOverlaps Called when we are copying into an overlapping
    // index `dst`
  }
}

#undef RUN

#define RUN(TYPE, DIMS, REAL)                       \
  THSyclTensor_scatterFillKernel<TYPE, REAL, DIMS>( \
      tensorInfo, indexInfo, value, dim, (TYPE)totalElements);

template <typename scalar_t>
void ScatterFill(
    Tensor& tensor,
    int64_t dim,
    const Tensor& index,
    Scalar value_scalar) {
  dim = maybe_wrap_dim(dim, tensor.dim());
  int index_ndim_legacy_all =
      TensorImpl_nDimensionLegacyAll(TensorImpl_Unwrap(index));
  TORCH_CHECK(
      dim < TensorImpl_nDimensionLegacyNoScalars(TensorImpl_Unwrap(tensor)),
      "Index dimension is out of bounds");
  TORCH_CHECK(
      index_ndim_legacy_all == 0 ||
          TensorImpl_nDimensionLegacyNoScalars(TensorImpl_Unwrap(index)) ==
              TensorImpl_nDimensionLegacyNoScalars(TensorImpl_Unwrap(tensor)),
      "Index tensor must be either empty or have same dimensions as output "
      "tensor");

  auto value = value_scalar.to<scalar_t>();
  // no-op if index is empty
  if (index_ndim_legacy_all == 0)
    return;

  for (int d = 0;
       d < TensorImpl_nDimensionLegacyNoScalars(TensorImpl_Unwrap(tensor));
       d++) {
    int64_t indexSizeD =
        TensorImpl_sizeLegacyNoScalars(TensorImpl_Unwrap(index), d);
    if (d != dim) {
      TORCH_CHECK(
          indexSizeD <=
              TensorImpl_sizeLegacyNoScalars(TensorImpl_Unwrap(tensor), d),
          "Index tensor must not have larger size than output tensor "
          "apart from the specified dimension ",
          dim,
          ", but got index ",
          index.sizes(),
          " output ",
          tensor.sizes());
    }
  }

  TORCH_CHECK(tensor.dim() <= MAX_DPCPPTORCH_DIMS, DPCPPTORCH_DIM_WARNING);

  const ptrdiff_t totalElements = index.numel();

  Tensor oldTensor;
  if (maybeOverlappingIndices(tensor)) {
    oldTensor = tensor;
    tensor = tensor.contiguous();
  }

  if (canUse32BitIndexMath(tensor) && canUse32BitIndexMath(index)) {
    TensorInfo<scalar_t, unsigned int> tensorInfo =
        getTensorInfo<scalar_t, unsigned int>(tensor);
    TensorInfo<int64_t, unsigned int> indexInfo =
        getTensorInfo<int64_t, unsigned int>(index);

    // Specialize for a small number of dimensions.
    switch (indexInfo.dims) {
      case 1:
        RUN(unsigned int, 1, scalar_t);
        break;
      case 2:
        RUN(unsigned int, 2, scalar_t);
        break;
      case 3:
        RUN(unsigned int, 3, scalar_t);
        break;
      default:
        RUN(unsigned int, -1, scalar_t);
        break;
    }
  } else {
    TensorInfo<scalar_t, uint64_t> tensorInfo =
        getTensorInfo<scalar_t, uint64_t>(tensor);
    TensorInfo<int64_t, uint64_t> indexInfo =
        getTensorInfo<int64_t, uint64_t>(index);

    RUN(uint64_t, -1, scalar_t);
  }

  if (oldTensor.defined()) {
    oldTensor.copy_(tensor);
    // TODO:copyIgnoringOverlaps Called when we are copying into an overlapping
    // index `dst`
  }
}

#undef RUN

#define RUN(TYPE, DIMS, REAL)                      \
  THSyclTensor_scatterAddKernel<TYPE, REAL, DIMS>( \
      tensorInfo, srcInfo, indexInfo, dim, (TYPE)totalElements);

template <typename scalar_t>
typename std::enable_if<
    IS_FLOAT32(scalar_t) || IS_BFLOAT16(scalar_t) || IS_INT(scalar_t) ||
        IS_INT64(scalar_t) || IS_DOUBLE(scalar_t) || IS_COMPLEX(scalar_t),
    void>::type
ScatterAdd(
    Tensor& tensor,
    int64_t dim,
    const Tensor& index,
    const Tensor& src) {
  dim = maybe_wrap_dim(dim, tensor.dim());
  int index_ndim_legacy_all =
      TensorImpl_nDimensionLegacyAll(TensorImpl_Unwrap(index));
  TORCH_CHECK(
      dim < TensorImpl_nDimensionLegacyNoScalars(TensorImpl_Unwrap(tensor)),
      "Index dimension is out of bounds");
  TORCH_CHECK(
      index_ndim_legacy_all == 0 ||
          TensorImpl_nDimensionLegacyNoScalars(TensorImpl_Unwrap(index)) ==
              TensorImpl_nDimensionLegacyNoScalars(TensorImpl_Unwrap(src)),
      "Index tensor must be either empty or have same dimensions as input "
      "tensor");
  TORCH_CHECK(
      TensorImpl_nDimensionLegacyNoScalars(TensorImpl_Unwrap(src)) ==
          TensorImpl_nDimensionLegacyNoScalars(TensorImpl_Unwrap(tensor)),
      "Input tensor must have same dimensions as output tensor");

  // no-op if index is empty
  if (index_ndim_legacy_all == 0)
    return;

  for (int d = 0;
       d < TensorImpl_nDimensionLegacyNoScalars(TensorImpl_Unwrap(tensor));
       d++) {
    int64_t indexSizeD =
        TensorImpl_sizeLegacyNoScalars(TensorImpl_Unwrap(index), d);
    if (d != dim) {
      TORCH_CHECK(
          indexSizeD <=
              TensorImpl_sizeLegacyNoScalars(TensorImpl_Unwrap(tensor), d),
          "Index tensor must not have larger size than output tensor "
          "apart from the specified dimension ",
          dim,
          ", but got index ",
          index.sizes(),
          " output ",
          tensor.sizes());
    }
    TORCH_CHECK(
        indexSizeD <= TensorImpl_sizeLegacyNoScalars(TensorImpl_Unwrap(src), d),
        "Index tensor must not have larger size than input tensor, but "
        "got index ",
        index.sizes(),
        "input",
        src.sizes());
  }

  TORCH_CHECK(tensor.dim() <= MAX_DPCPPTORCH_DIMS, DPCPPTORCH_DIM_WARNING);

  const ptrdiff_t totalElements = index.numel();

  Tensor oldTensor;
  if (maybeOverlappingIndices(tensor)) {
    oldTensor = tensor;
    tensor = tensor.contiguous();
  }

  if (totalElements > 0) {
    if (canUse32BitIndexMath(tensor) && canUse32BitIndexMath(src) &&
        canUse32BitIndexMath(index)) {
      TensorInfo<scalar_t, unsigned int> tensorInfo =
          getTensorInfo<scalar_t, unsigned int>(tensor);
      TensorInfo<scalar_t, unsigned int> srcInfo =
          getTensorInfo<scalar_t, unsigned int>(src);
      TensorInfo<int64_t, unsigned int> indexInfo =
          getTensorInfo<int64_t, unsigned int>(index);
      // Specialize for a small number of dimensions.
      switch (indexInfo.dims) {
        case 1:
          RUN(unsigned int, 1, scalar_t);
          break;
        case 2:
          RUN(unsigned int, 2, scalar_t);
          break;
        case 3:
          RUN(unsigned int, 3, scalar_t);
          break;
        default:
          RUN(unsigned int, -1, scalar_t);
          break;
      }
    } else {
      TensorInfo<scalar_t, uint64_t> tensorInfo =
          getTensorInfo<scalar_t, uint64_t>(tensor);
      TensorInfo<scalar_t, uint64_t> srcInfo =
          getTensorInfo<scalar_t, uint64_t>(src);
      TensorInfo<int64_t, uint64_t> indexInfo =
          getTensorInfo<int64_t, uint64_t>(index);

      RUN(uint64_t, -1, scalar_t)
    }
  }

  if (oldTensor.defined()) {
    oldTensor.copy_(tensor);
    // TODO:copyIgnoringOverlaps Called when we are copying into an overlapping
    // index `dst`
  }
}

template <typename scalar_t>
typename std::enable_if<
    !(IS_FLOAT32(scalar_t) || IS_BFLOAT16(scalar_t) || IS_INT(scalar_t) ||
      IS_INT64(scalar_t) || IS_DOUBLE(scalar_t) || IS_COMPLEX(scalar_t)),
    void>::type
ScatterAdd(
    Tensor& tensor,
    int64_t dim,
    const Tensor& index,
    const Tensor& src) {
  TORCH_CHECK(
      0,
      "scatter add only supports float, bfloat16, int, int64 and double type");
}
#undef RUN

// essentialy rewritten related to legacy::launch_kernel parts
template <int n_elems_per_work_item, typename func_t>
void _scatter_gather_elementwise_kernel(int total_n_elems, func_t f) {
  int group_items = dpcppMaxWorkGroupSize(dpcppGetDeviceIdOfCurrentQueue());
  int group_work_size = n_elems_per_work_item * group_items;
  int num_groups = (total_n_elems + group_work_size - 1) / group_work_size;
  auto& dpcpp_queue = dpcppGetCurrentQueue();

  auto cgf = DPCPP_Q_CGF(cgh) {
    cgh.parallel_for(
        sycl::nd_range<1>(
            sycl::range<1>(num_groups * group_items),
            sycl::range<1>(group_items)),
        [=](sycl::nd_item<1> itemId) {
          int idx = itemId.get_global_id()[0];
#pragma unroll
          for (int i = 0; i < n_elems_per_work_item; ++i) {
            if (idx < total_n_elems) {
              f(idx);
              idx += itemId.get_global_range()[0];
            }
          }
        });
  };

  DPCPP_Q_SUBMIT(dpcpp_queue, cgf);
}

template <int n_elems_per_work_item, typename func_t>
static void _launch_scatter_gather_kernel(
    int64_t total_n_elems,
    const func_t& f) {
  TORCH_INTERNAL_ASSERT(
      total_n_elems >= 0 &&
      total_n_elems <= std::numeric_limits<int32_t>::max());
  if (total_n_elems == 0) {
    return;
  }

  _scatter_gather_elementwise_kernel<n_elems_per_work_item, func_t>(
      total_n_elems, f);
}

template <typename scalar_t>
struct _dpcpp_scatter_fill_internal_kernel {
  template <typename func_t>
  void operator()(
      TensorIterator& iter,
      scalar_t src_val,
      int64_t index_size,
      int64_t index_stride,
      const func_t& f) {
    if (!iter.can_use_32bit_indexing()) {
      for (auto& sub_iter : iter.with_32bit_indexing()) {
        _dpcpp_scatter_fill_internal_kernel<scalar_t>()(
            sub_iter, src_val, index_size, index_stride, f);
      }
      return;
    }

    char* self_ptr = (char*)iter.data_ptr(0);
    char* index_ptr = (char*)iter.data_ptr(1);

    auto offset_calc = make_offset_calculator<2>(iter);
    auto loop = [=](int i) {
      auto offsets = offset_calc.get(i);

      int64_t idx_dim = *(int64_t*)(index_ptr + offsets[1]);

      char* self_data = self_ptr + offsets[0];

      f((scalar_t*)self_data + idx_dim * index_stride, (scalar_t*)&src_val);
    };

    _launch_scatter_gather_kernel<n_elems_per_work_item>(iter.numel(), loop);
  }
}; // struct _dpcpp_scatter_fill_internal_kernel

template <bool is_scatter_like, typename scalar_t>
struct _dpcpp_scatter_gather_internal_kernel {
  template <typename func_t>
  void operator()(
      TensorIterator& iter,
      int64_t index_size,
      int64_t index_stride,
      const func_t& f) {
    if (!iter.can_use_32bit_indexing()) {
      for (auto& sub_iter : iter.with_32bit_indexing()) {
        _dpcpp_scatter_gather_internal_kernel<is_scatter_like, scalar_t>()(
            sub_iter, index_size, index_stride, f);
      }
      return;
    }

    char* self_ptr = (char*)iter.data_ptr(0);
    char* src_ptr = (char*)iter.data_ptr(1);
    char* index_ptr = (char*)iter.data_ptr(2);

    auto offset_calc = make_offset_calculator<3>(iter);
    auto loop = [=](int i) {
      auto offsets = offset_calc.get(i);

      int64_t idx_dim = *(int64_t*)(index_ptr + offsets[2]);

      char* self_data = self_ptr + offsets[0];
      char* src_data = src_ptr + offsets[1];

      f((scalar_t*)self_data + (is_scatter_like ? idx_dim * index_stride : 0),
        (scalar_t*)src_data + (is_scatter_like ? 0 : idx_dim * index_stride));
    };

    _launch_scatter_gather_kernel<n_elems_per_work_item>(iter.numel(), loop);
  }
}; // struct _dpcpp_scatter_gather_internal_kernel

template <bool cast_to_opaque = true>
struct dpcpp_scatter_fill_base_kernel {
  template <typename func_t>
  void operator()(
      const Tensor& self,
      int64_t dim,
      const Tensor& index,
      Scalar src,
      const std::string& method_name,
      const func_t& f) {
    at::assert_no_internal_overlap(self);

    auto index_sizes = ensure_nonempty_vec(index.sizes().vec());

    // restride self such that
    // self.shape = index.shape and
    // self.stride[dim] = 0
    auto self_restrided = restride_dim(self, dim, index_sizes);

    auto iter = TensorIteratorConfig()
                    .set_check_mem_overlap(false)
                    .check_all_same_dtype(false)
                    .resize_outputs(false)
                    .add_output(self_restrided)
                    .add_input(index)
                    .build();

    auto index_size = ensure_nonempty_size(self, dim);
    auto index_stride = ensure_nonempty_stride(self, dim);

    IPEX_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(
        at::ScalarType::Half,
        at::ScalarType::Bool,
        at::ScalarType::BFloat16,
        iter.dtype(),
        "dpcpp_scatter_fill_base_kernel_func",
        [&] {
          using dtype = typename std::conditional<
              cast_to_opaque,
              OpaqueType<sizeof(scalar_t)>,
              scalar_t>::type;

          auto src_scalar_val = src.to<scalar_t>();
          auto src_val = *(dtype*)&src_scalar_val;

          _dpcpp_scatter_fill_internal_kernel<dtype>()(
              iter, src_val, index_size, index_stride, f);
        });
  }

  void operator()(
      const Tensor& self,
      int64_t dim,
      const Tensor& index,
      Scalar src,
      const std::string& method_name,
      const ReduceMultiply& f) {
    at::assert_no_internal_overlap(self);

    auto index_sizes = ensure_nonempty_vec(index.sizes().vec());

    // restride self such that
    // self.shape = index.shape and
    // self.stride[dim] = 0
    auto self_restrided = restride_dim(self, dim, index_sizes);

    auto iter = TensorIteratorConfig()
                    .set_check_mem_overlap(false)
                    .check_all_same_dtype(false)
                    .resize_outputs(false)
                    .add_output(self_restrided)
                    .add_input(index)
                    .build();

    auto index_size = ensure_nonempty_size(self, dim);
    auto index_stride = ensure_nonempty_stride(self, dim);

    IPEX_DISPATCH_ALL_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        iter.dtype(),
        "dpcpp_scatter_fill_base_kernel_reduce_multiply",
        [&] {
          using dtype = typename std::conditional<
              cast_to_opaque,
              OpaqueType<sizeof(scalar_t)>,
              scalar_t>::type;

          auto src_scalar_val = src.to<scalar_t>();
          auto src_val = *(dtype*)&src_scalar_val;

          _dpcpp_scatter_fill_internal_kernel<dtype>()(
              iter, src_val, index_size, index_stride, f);
        });
  }
}; // struct dpcpp_scatter_fill_base_kernel

template <bool is_scatter_like = true, bool cast_to_opaque = true>
struct dpcpp_scatter_gather_base_kernel {
  template <typename func_t>
  void operator()(
      const Tensor& self,
      int64_t dim,
      const Tensor& index,
      const Tensor& src,
      const std::string& method_name,
      const func_t& f) {
    at::assert_no_internal_overlap(self);

    auto index_sizes = ensure_nonempty_vec(index.sizes().vec());
    auto self_strides = ensure_nonempty_vec(self.strides().vec());
    auto src_strides = ensure_nonempty_vec(src.strides().vec());

    // restride self and src such that
    // self.shape = src.shape = index.shape
    //
    // restride stride[dim] such that
    // if (is_scatter_like) self.stride[dim] = 0
    // else src.stride[dim] = 0
    auto self_restrided = is_scatter_like
        ? restride_dim(self, dim, index_sizes)
        : self.as_strided(index_sizes, self_strides);
    auto src_restrided = is_scatter_like
        ? src.as_strided(index_sizes, src_strides)
        : restride_dim(src, dim, index_sizes);

    auto iter = TensorIteratorConfig()
                    .set_check_mem_overlap(false)
                    .check_all_same_dtype(false)
                    .resize_outputs(false)
                    .add_output(self_restrided)
                    .add_input(src_restrided)
                    .add_input(index)
                    .build();

    auto self_dim_stride = ensure_nonempty_stride(self, dim);
    auto self_dim_size = ensure_nonempty_size(self, dim);

    auto src_dim_stride = ensure_nonempty_stride(src, dim);
    auto src_dim_size = ensure_nonempty_size(src, dim);

    auto index_size = is_scatter_like ? self_dim_size : src_dim_size;
    auto index_stride = is_scatter_like ? self_dim_stride : src_dim_stride;

    IPEX_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(
        at::ScalarType::Half,
        at::ScalarType::Bool,
        at::ScalarType::BFloat16,
        iter.dtype(),
        "dpcpp_scatter_gather_base_kernel_func",
        [&] {
          using dtype = typename std::conditional<
              cast_to_opaque,
              OpaqueType<sizeof(scalar_t)>,
              scalar_t>::type;

          _dpcpp_scatter_gather_internal_kernel<is_scatter_like, dtype>()(
              iter, index_size, index_stride, f);
        });
  }

  void operator()(
      const Tensor& self,
      int64_t dim,
      const Tensor& index,
      const Tensor& src,
      const std::string& method_name,
      const ReduceMultiply& f) {
    at::assert_no_internal_overlap(self);

    auto index_sizes = ensure_nonempty_vec(index.sizes().vec());
    auto self_strides = ensure_nonempty_vec(self.strides().vec());
    auto src_strides = ensure_nonempty_vec(src.strides().vec());

    // restride self and src such that
    // self.shape = src.shape = index.shape
    //
    // restride stride[dim] such that
    // if (is_scatter_like) self.stride[dim] = 0
    // else src.stride[dim] = 0
    auto self_restrided = is_scatter_like
        ? restride_dim(self, dim, index_sizes)
        : self.as_strided(index_sizes, self_strides);
    auto src_restrided = is_scatter_like
        ? src.as_strided(index_sizes, src_strides)
        : restride_dim(src, dim, index_sizes);

    auto iter = TensorIteratorConfig()
                    .set_check_mem_overlap(false)
                    .check_all_same_dtype(false)
                    .resize_outputs(false)
                    .add_output(self_restrided)
                    .add_input(src_restrided)
                    .add_input(index)
                    .build();

    auto self_dim_stride = ensure_nonempty_stride(self, dim);
    auto self_dim_size = ensure_nonempty_size(self, dim);

    auto src_dim_stride = ensure_nonempty_stride(src, dim);
    auto src_dim_size = ensure_nonempty_size(src, dim);

    auto index_size = is_scatter_like ? self_dim_size : src_dim_size;
    auto index_stride = is_scatter_like ? self_dim_stride : src_dim_stride;

    IPEX_DISPATCH_ALL_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        iter.dtype(),
        "dpcpp_scatter_gather_base_kernel_reduce_multiply",
        [&] {
          using dtype = typename std::conditional<
              cast_to_opaque,
              OpaqueType<sizeof(scalar_t)>,
              scalar_t>::type;

          _dpcpp_scatter_gather_internal_kernel<is_scatter_like, dtype>()(
              iter, index_size, index_stride, f);
        });
  }
}; // struct dpcpp_scatter_gather_base_kernel

} // namespace impl

Tensor& scatter_(
    Tensor& self,
    int64_t dim,
    const Tensor& index,
    const Tensor& src) {
  at::assert_no_internal_overlap(self);
  at::assert_no_overlap(self, index);
  at::assert_no_overlap(self, src);
  IPEX_DISPATCH_ALL_TYPES_AND_COMPLEX_AND2(
      at::ScalarType::BFloat16,
      at::ScalarType::Bool,
      self.scalar_type(),
      "Scatter",
      [&]() { impl::Scatter<scalar_t>(self, dim, index, src); });
  return self;
}

Tensor scatter(
    const Tensor& self,
    int64_t dim,
    const Tensor& index,
    const Tensor& src) {
  at::assert_no_internal_overlap(self);
  Tensor out = at::empty_like(self);
  out.copy_(self);
  at::assert_no_internal_overlap(out);
  at::assert_no_overlap(out, index);
  at::assert_no_overlap(out, src);

  IPEX_DISPATCH_ALL_TYPES_AND_COMPLEX_AND2(
      at::ScalarType::BFloat16,
      at::ScalarType::Bool,
      out.scalar_type(),
      "Scatter",
      [&]() { impl::Scatter<scalar_t>(out, dim, index, src); });
  return out;
}

Tensor scatter(
    const Tensor& self,
    int64_t dim,
    const Tensor& index,
    const Scalar& value) {
  at::assert_no_internal_overlap(self);
  Tensor out = at::empty_like(self);
  out.copy_(self);
  at::assert_no_internal_overlap(out);
  at::assert_no_overlap(out, index);
  IPEX_DISPATCH_ALL_TYPES_AND3(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      at::ScalarType::Bool,
      out.scalar_type(),
      "ScatterFill",
      [&]() { impl::ScatterFill<scalar_t>(out, dim, index, value); });
  return out;
}

Tensor& scatter_out(
    const Tensor& self,
    int64_t dim,
    const Tensor& index,
    const Scalar& value,
    Tensor& out) {
  at::assert_no_internal_overlap(self);
  if (!self.is_same(out)) {
    out.copy_(self);
  }
  at::assert_no_internal_overlap(out);
  at::assert_no_overlap(out, index);
  IPEX_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      at::ScalarType::Bool,
      self.scalar_type(),
      "ScatterFill",
      [&]() { impl::ScatterFill<scalar_t>(out, dim, index, value); });
  return out;
}

Tensor& scatter_add_out(
    const Tensor& self,
    int64_t dim,
    const Tensor& index,
    const Tensor& src,
    Tensor& out) {
  at::assert_no_internal_overlap(self);
  if (!self.is_same(out)) {
    out.copy_(self);
  }
  at::assert_no_internal_overlap(out);
  at::assert_no_overlap(out, index);
  at::assert_no_overlap(out, src);
  IPEX_DISPATCH_ALL_TYPES_AND_COMPLEX_AND2(
      at::ScalarType::BFloat16,
      at::ScalarType::Bool,
      self.scalar_type(),
      "ScatterAdd",
      [&]() { impl::ScatterAdd<scalar_t>(out, dim, index, src); });
  return out;
}

Tensor& gather_out(
    Tensor& out,
    const Tensor& self,
    int64_t dim,
    const Tensor& index,
    bool sparse_grad) {
  at::assert_no_internal_overlap(self);
  out.resize_(index.sizes());

  bool check_result = out.defined();
  if (check_result) {
    at::assert_no_internal_overlap(out);
    at::assert_no_overlap(out, self);
    at::assert_no_partial_overlap(out, index);
  }
  IPEX_DISPATCH_ALL_TYPES_AND_COMPLEX_AND2(
      at::ScalarType::BFloat16,
      at::ScalarType::Bool,
      self.scalar_type(),
      "Gather",
      [&]() { impl::Gather<scalar_t>(out, self, dim, index); });
  return out;
}

Tensor gather(
    const Tensor& self,
    int64_t dim,
    const Tensor& index,
    bool sparse_grad) {
  Tensor out = at::empty({0}, self.options());
  return at::AtenIpexTypeXPU::gather_out(out, self, dim, index, sparse_grad);
}

void scatter_dpcpp_kernel(
    const Tensor& self,
    int64_t dim,
    const Tensor& index,
    const Tensor& src) {
  impl::dpcpp_scatter_gather_base_kernel<>()(
      self, dim, index, src, "scatter_dpcpp_", tensor_assign);
}

void scatter_reduce_dpcpp_kernel(
    const Tensor& self,
    const int64_t dim,
    const Tensor& index,
    const Tensor& src,
    const SCATTER_GATHER_OP& reduce) {
  switch (reduce) {
    case SCATTER_GATHER_OP::REDUCE_ADD:
      impl::dpcpp_scatter_gather_base_kernel<true, false>()(
          self, dim, index, src, "scatter_reduce_dpcpp_add_", reduce_add);
      break;
    case SCATTER_GATHER_OP::REDUCE_MULTIPLY:
      impl::dpcpp_scatter_gather_base_kernel<true, false>()(
          self,
          dim,
          index,
          src,
          "scatter_reduce_dpcpp_multiply_",
          reduce_multiply);
      break;
  }
}

void scatter_fill_dpcpp_kernel(
    const Tensor& self,
    int64_t dim,
    const Tensor& index,
    const Scalar& src) {
  impl::dpcpp_scatter_fill_base_kernel<>()(
      self, dim, index, src, "scatter_fill_dpcpp_", tensor_assign);
}

void scatter_scalar_reduce_dpcpp_kernel(
    const Tensor& self,
    const int64_t dim,
    const Tensor& index,
    const Scalar& value,
    const SCATTER_GATHER_OP& reduce) {
  switch (reduce) {
    case SCATTER_GATHER_OP::REDUCE_ADD:
      impl::dpcpp_scatter_fill_base_kernel<false>()(
          self, dim, index, value, "scatter_fill_dpcpp_add_", reduce_add);
      break;
    case SCATTER_GATHER_OP::REDUCE_MULTIPLY:
      impl::dpcpp_scatter_fill_base_kernel<false>()(
          self,
          dim,
          index,
          value,
          "scatter_fill_dpcpp_multiply_",
          reduce_multiply);
      break;
  }
}

// scatter.reduce_out
Tensor& scatter_out(
    const Tensor& self,
    int64_t dim,
    const Tensor& index,
    const Tensor& src,
    c10::string_view reduce,
    Tensor& out) {
  scatter_impl(
      self,
      dim,
      index,
      src,
      out,
      scatter_reduce_dpcpp_kernel,
      scatter_dpcpp_kernel,
      reduce);
  return out;
}

// scatter.src_out
Tensor& scatter_out(
    const Tensor& self,
    int64_t dim,
    const Tensor& index,
    const Tensor& src,
    Tensor& out) {
  scatter_impl(
      self,
      dim,
      index,
      src,
      out,
      scatter_reduce_dpcpp_kernel,
      scatter_dpcpp_kernel);
  return out;
}

// scatter.value_reduce_out
Tensor& scatter_out(
    const Tensor& self,
    int64_t dim,
    const Tensor& index,
    const Scalar& value,
    c10::string_view reduce,
    Tensor& out) {
  scatter_impl(
      self,
      dim,
      index,
      value,
      out,
      scatter_scalar_reduce_dpcpp_kernel,
      scatter_fill_dpcpp_kernel,
      reduce);
  return out;
}

} // namespace AtenIpexTypeXPU
} // namespace at
