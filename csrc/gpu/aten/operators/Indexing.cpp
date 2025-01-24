#include "Indexing.h"
#include <ATen/ATen.h>
#include <ATen/MemoryOverlap.h>
#include <ATen/ceil_div.h>
#include <ATen/native/TensorIterator.h>
#include <core/Memory.h>
#include <core/detail/IndexUtils.h>
#include <core/detail/TensorInfo.h>
#include <runtime/Utils.h>
#ifdef USE_OVERRIDE_OP
#include <ATen/DeviceGuard.h>
#include <ATen/core/op_registration/adaption.h>
#include <utils/CustomOperatorRegistration.h>
#endif
#include <utils/DPCPP.h>
#include <utils/Helpers.h>
#include <iostream>
#include "IndexingUtils.h"
#include "Loops.h"
#include "PSTLFunctions.h"
#include "ParttenScan.h"
#include "SortingDeviceRadixSort.h"
#include "comm/ATDispatch.h"
#include "comm/AccumulateType.h"
#include "comm/ApplyUtils.h"
#include "comm/Atomics.h"
#include "comm/MathReduce.h"
#include "comm/Numerics.h"

using namespace torch_ipex::xpu::dpcpp;

namespace at {
namespace AtenIpexTypeXPU {

namespace impl {

// Pretend that the scalar tensor is in fact a one-element vector.
template <typename T, typename IndexType>
torch_ipex::xpu::dpcpp::detail::TensorInfo<T, IndexType> tensorInfoIfScalar(
    torch_ipex::xpu::dpcpp::detail::TensorInfo<T, IndexType> ti) {
  if (ti.dims == 0) {
    ti.dims = 1;
    ti.sizes[0] = 1;
    ti.strides[0] = 1;
  }
  return ti;
}

template <typename scalar_t>
void indexSelect(
    const Tensor& dst,
    const Tensor& src,
    int dim,
    const Tensor& indices) {
  IPEX_DISPATCH_INDEX_TYPES(indices.scalar_type(), "indexSelect", [&] {
    TensorInfo<index_t, int64_t> indices_info =
        tensorInfoIfScalar(getTensorInfo<index_t, int64_t>(indices));
    indices_info.collapseDims();

    TensorInfo<scalar_t, int64_t> dst_info =
        tensorInfoIfScalar(getTensorInfo<scalar_t, int64_t>(dst));
    TensorInfo<scalar_t, int64_t> src_info =
        tensorInfoIfScalar(getTensorInfo<scalar_t, int64_t>(src.contiguous()));
    int new_indexing_dim = src_info.collapseDims(dim);

    if (dst.is_contiguous() && indices.is_contiguous())
      _index_select_kernel<
          decltype(src_info),
          decltype(dst_info),
          decltype(indices_info),
          /* TrivialOffCal */ true>(
          src_info, dst_info, indices_info, new_indexing_dim);
    else
      _index_select_kernel<
          decltype(src_info),
          decltype(dst_info),
          decltype(indices_info),
          /* TrivialOffCal */ false>(
          src_info, dst_info, indices_info, new_indexing_dim);
  });
  return;
}
template <typename scalar_t>
void index_select_impl(
    const Tensor& dst,
    const Tensor& src,
    int dim,
    const Tensor& indices) {
  at::assert_no_internal_overlap(dst);
  at::assert_no_overlap(dst, src);
  at::assert_no_overlap(dst, indices);

  dim = at::maybe_wrap_dim(dim, src);
  int srcDims = src.dim() == 0 ? 1 : src.dim();
  int dstDims = dst.dim();
  int idxDims = indices.dim();

  TORCH_CHECK(
      srcDims <= MAX_DPCPPTORCH_DIMS,
      "src tensor dim should be < ",
      MAX_DPCPPTORCH_DIMS);
  TORCH_CHECK(
      dstDims <= MAX_DPCPPTORCH_DIMS,
      "dst tensor dim should be < ",
      MAX_DPCPPTORCH_DIMS);
  TORCH_CHECK(
      idxDims <= MAX_DPCPPTORCH_DIMS,
      "index tensor dim should be < ",
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

  auto new_size = src.sizes().vec();

  if (src.dim() > 0) {
    new_size[dim] = indices.numel();
  }

  at::native::resize_output(dst, new_size);

  ptrdiff_t dst_num_elem = dst.numel();
  if (dst_num_elem == 0) {
    return;
  }

  if (!canUse32BitIndexMath(dst)) {
    auto MaxInt32 = std::numeric_limits<int32_t>::max();
    int32_t iter_number = (dst_num_elem + MaxInt32 - 1) / MaxInt32;
    int64_t slice_offset = indices.numel() / iter_number;

    int64_t start_id = 0;
    int64_t end_id = 0;
    for (int32_t i = 0; i < iter_number; i++) {
      start_id = 0 + slice_offset * i;
      end_id = start_id + slice_offset;
      if (end_id <= indices.numel()) {
        indexSelect<scalar_t>(
            dst.slice(dim, start_id, end_id),
            src,
            dim,
            indices.slice(0, start_id, end_id));
      } else {
        indexSelect<scalar_t>(
            dst.slice(dim, start_id), src, dim, indices.slice(0, start_id));
      }
    }
    if (end_id < indices.numel()) {
      indexSelect<scalar_t>(
          dst.slice(dim, end_id), src, dim, indices.slice(0, end_id));
    }
  } else {
    indexSelect<scalar_t>(dst, src, dim, indices);
  }
  return;
}

struct NonzeroKernelFunctor {
  void operator()(sycl::nd_item<1> item_id) const {
    auto global_id = item_id.get_global_linear_id();

    if (global_id < N) {
      auto index = global_id / num_dim;
      auto dim = global_id % num_dim;
      tensor_begin[global_id] =
          idx_flat_begin[index] / divisor[dim] % sizes[dim];
    }
  }
  NonzeroKernelFunctor(
      int64_t N_,
      const int64_t num_dim_,
      int64_t* tensor_begin_,
      int64_t* idx_flat_begin_,
      int64_t* divisor_,
      int64_t* sizes_)
      : N(N_),
        num_dim(num_dim_),
        tensor_begin(tensor_begin_),
        idx_flat_begin(idx_flat_begin_) {
    for (auto dim = num_dim - 1; dim >= 0; dim--) {
      sizes[dim] = sizes_[dim];
      divisor[dim] = divisor_[dim];
    }
  }

 private:
  int64_t N;
  const int64_t num_dim;
  int64_t* tensor_begin;
  int64_t* idx_flat_begin;
  int64_t divisor[MAX_TENSORINFO_DIMS];
  int64_t sizes[MAX_TENSORINFO_DIMS];
};

template <typename scalar_t>
struct nonzero_copy_if_functor {
  auto operator()(int64_t x) const {
    return Numerics<scalar_t>::ne(self_begin[x], scalar_t(0));
  }
  nonzero_copy_if_functor(scalar_t* self_begin) : self_begin(self_begin) {}

 private:
  scalar_t* self_begin;
};
template <>
struct nonzero_copy_if_functor<bool> {
  bool operator()(int64_t x) const {
    // Using data type conversion to break deduce of execution chain in bool.
    // Bool operations will be removed in the compiler optimization.
    // The function returns a bool variable with one byte value stored i    n
    // self_begin_ not 1 specified here.
    volatile int in = (int)self_begin_[x];
    bool res = in != int(0) ? 1 : 0;
    return res;
  }
  nonzero_copy_if_functor(bool* self_begin) : self_begin_(self_begin) {}

 private:
  bool* self_begin_;
};

template <typename scalar_t>
void nonzero(Tensor& tensor, const Tensor& self_) {
  Tensor self = self_.contiguous();
  const int64_t num_dim = self.dim();
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

    nonzero_copy_if_functor<scalar_t> f(self_begin);
    auto idx_flat_end = torch_ipex::xpu::pstl::copy_if<int64_t>(
        range_begin, range_begin + N, idx_flat_begin, f);

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
        NonzeroKernelFunctor kfn(
            N, num_dim, tensor_begin, idx_flat_begin, divisor, sizes);
        __cgh.parallel_for<decltype(kfn)>(
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
struct DiagKernelFunctor {
  void operator()(sycl::item<1> item_id) const {
    size_t id = item_id.get_id(0);
    const int64_t bOffset = start + (stride0 + stride1) * id;
    out[strideSelf * id] = in[bOffset];
  }
  DiagKernelFunctor(
      scalar_t* in_,
      scalar_t* out_,
      int64_t strideSelf_,
      int64_t start_,
      int64_t stride0_,
      int64_t stride1_)
      : in(in_),
        out(out_),
        strideSelf(strideSelf_),
        start(start_),
        stride0(stride0_),
        stride1(stride1_) {}

 private:
  scalar_t* in;
  scalar_t* out;
  int64_t strideSelf;
  int64_t start;
  int64_t stride0;
  int64_t stride1;
};

template <typename scalar_t>
struct DiagKernelFunctor2 {
  void operator()(sycl::item<1> item_id) const {
    size_t id = item_id.get_id(0);
    const int64_t aOffset = start + (stride0 + stride1) * id;
    out[aOffset] = in[strideSrc * id];
  }
  DiagKernelFunctor2(
      scalar_t* in_,
      scalar_t* out_,
      int64_t strideSrc_,
      int64_t start_,
      int64_t stride0_,
      int64_t stride1_)
      : in(in_),
        out(out_),
        strideSrc(strideSrc_),
        start(start_),
        stride0(stride0_),
        stride1(stride1_) {}

 private:
  scalar_t* in;
  scalar_t* out;
  int64_t strideSrc;
  int64_t start;
  int64_t stride0;
  int64_t stride1;
};

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
    resize_output(dst, {size});
    if (size > 0) {
      auto in = src.data_ptr<scalar_t>();
      auto out = dst.data_ptr<scalar_t>();
      int64_t strideSelf = dst.dim() == 0 ? 1 : dst.stride(0);
      int64_t start = (k >= 0 ? k * stride1 : -k * stride0);
      auto& dpcpp_queue = dpcppGetCurrentQueue();

      auto cgf = DPCPP_Q_CGF(cgh) {
        DiagKernelFunctor<scalar_t> kfn(
            in, out, strideSelf, start, stride0, stride1);
        cgh.parallel_for<decltype(kfn)>(sycl::range<1>(dst.numel()), kfn);
      };
      DPCPP_Q_SUBMIT(dpcpp_queue, cgf);
    }
  } else {
    int64_t totalElements = src.numel();
    int64_t size = (k > 0) ? totalElements + k : totalElements - k;
    int64_t strideSrc = src.dim() == 0 ? 1 : src.stride(0);
    resize_output(dst, {size, size});
    dst.zero_();
    if (size > 0) {
      auto in = src.data_ptr<scalar_t>();
      auto out = dst.data_ptr<scalar_t>();
      int64_t stride0 = dst.stride(0);
      int64_t stride1 = dst.stride(1);
      int64_t start = (k >= 0 ? k * stride1 : -k * stride0);
      auto& dpcpp_queue = dpcppGetCurrentQueue();

      auto cgf = DPCPP_Q_CGF(cgh) {
        DiagKernelFunctor2<scalar_t> kfn(
            in, out, strideSrc, start, stride0, stride1);
        cgh.parallel_for<decltype(kfn)>(sycl::range<1>(src.numel()), kfn);
      };
      DPCPP_Q_SUBMIT(dpcpp_queue, cgf);
    }
  }
}

template <int N>
struct alignas(N) OpaqueType {
  char data[N];
};

template <typename scalar_t, typename Func>
struct PutKernelFunctor {
  void operator()(sycl::item<1> item_id) const {
    auto out_ptr = (char*)out_data;
    auto indices_ptr = indices_data;
    auto source_ptr = (char*)source_data;

    auto linear_idx = item_id.get_id(0);
    auto idx_offset =
        IndexToOffset<int64_t, uint64_t>::get(linear_idx, indices_info);

    auto index = indices_ptr[idx_offset];
    if (index < 0) {
      index += out_numel;
    }

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
  }
  PutKernelFunctor(
      Func f_,
      int64_t out_numel_,
      size_t scalar_bytes_,
      TensorInfo<scalar_t, uint64_t> out_info_,
      TensorInfo<int64_t, uint64_t> indices_info_,
      TensorInfo<scalar_t, uint64_t> source_info_,
      scalar_t* out_data_,
      int64_t* indices_data_,
      scalar_t* source_data_)
      : f(f_),
        out_numel(out_numel_),
        scalar_bytes(scalar_bytes_),
        out_info(out_info_),
        indices_info(indices_info_),
        source_info(source_info_),
        out_data(out_data_),
        indices_data(indices_data_),
        source_data(source_data_) {}

 private:
  Func f;
  int64_t out_numel;
  size_t scalar_bytes;
  TensorInfo<scalar_t, uint64_t> out_info;
  TensorInfo<int64_t, uint64_t> indices_info;
  TensorInfo<scalar_t, uint64_t> source_info;
  scalar_t* out_data;
  int64_t* indices_data;
  scalar_t* source_data;
};

template <typename scalar_t, typename Func>
void put(Tensor& self, const Tensor& index, const Tensor& source, Func f) {
  auto numel = index.numel();
  if (numel == 0)
    return;

  auto out_numel = self.numel();
  size_t scalar_bytes = sizeof(scalar_t);

  TensorInfo<scalar_t, uint64_t> out_info =
      getTensorInfo<scalar_t, uint64_t>(self);
  out_info.collapseDims();

  TensorInfo<int64_t, uint64_t> indices_info =
      getTensorInfo<int64_t, uint64_t>(index);
  indices_info.collapseDims();

  TensorInfo<scalar_t, uint64_t> source_info =
      getTensorInfo<scalar_t, uint64_t>(source);
  source_info.collapseDims();

  auto& dpcpp_queue = dpcppGetCurrentQueue();

  auto cgf = DPCPP_Q_CGF(__cgh) {
    auto out_data = self.data_ptr<scalar_t>();
    auto indices_data = index.data_ptr<int64_t>();
    auto source_data = source.data_ptr<scalar_t>();

    PutKernelFunctor<scalar_t, Func> kfn(
        f,
        out_numel,
        scalar_bytes,
        out_info,
        indices_info,
        source_info,
        out_data,
        indices_data,
        source_data);

    __cgh.parallel_for<decltype(kfn)>(sycl::range</*dim=*/1>(numel), kfn);
  };
  DPCPP_Q_SUBMIT(dpcpp_queue, cgf);
}

template <typename dtype>
struct index_functor {
  void operator()(char* out_data, char* in_data, int64_t offset) const {
    *(dtype*)out_data = *(dtype*)(in_data + offset);
  }
};

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
        index_functor<dtype> f;
        dpcpp_index_kernel(
            iter,
            index_size,
            index_stride,
            non_index_size,
            non_index_stride,
            f);
      });
}

template <typename scalar_t, typename accscalar_t>
struct IndexPutDeterministicKernelFunctor {
  void operator()(sycl::nd_item<2> item) const {
    auto id = cfg.get_item_desc(item);

    if (id.glb_batch >= cfg.problem_batch_ || id.glb_problem >= cfg.problem_)
      return;

    int64_t idx = sorted_indices[id.glb_batch];
    if (id.glb_batch != 0 && idx == sorted_indices[id.glb_batch - 1])
      return;

    int64_t pi_ = id.glb_problem;
    int64_t si_ = pi_ % stride;
    int64_t bi_ = pi_ / stride;
    int64_t s_gid = si_ + idx * stride + bi_ * stride_before;
    int64_t v_stride = si_ + bi_ * v_stride_before;

    accscalar_t acc;
    if (accumulate)
      acc = self[s_gid];
    for (int64_t inner_idx = id.glb_batch;
         inner_idx < cfg.problem_batch_ && sorted_indices[inner_idx] == idx;
         inner_idx++) {
      int64_t idx_orig = indices[inner_idx];
      int64_t v_gid = idx_orig * stride + v_stride;
      if (accumulate) {
        acc += (accscalar_t)value[v_gid];
      } else {
        self[s_gid] = value[v_gid];
        break;
      }
    }
    if (accumulate)
      self[s_gid] = acc;
  }
  IndexPutDeterministicKernelFunctor(
      int64_t* sorted_indices_,
      int64_t* indices_,
      scalar_t* value_,
      scalar_t* self_,
      int64_t numel_,
      int64_t stride_,
      int64_t stride_before_,
      int64_t outer_dim_,
      bool accumulate_,
      int64_t v_stride_before_,
      BatchKernelConfig cfg_)
      : sorted_indices(sorted_indices_),
        indices(indices_),
        value(value_),
        self(self_),
        numel(numel_),
        stride(stride_),
        stride_before(stride_before_),
        outer_dim(outer_dim_),
        accumulate(accumulate_),
        v_stride_before(v_stride_before_),
        cfg(cfg_) {}

 private:
  int64_t* sorted_indices;
  int64_t* indices;
  scalar_t* value;
  scalar_t* self;
  int64_t numel;
  int64_t stride;
  int64_t stride_before;
  int64_t outer_dim;
  bool accumulate;
  int64_t v_stride_before;
  BatchKernelConfig cfg;
};

template <typename scalar_t>
void index_put_deterministic_kernel(
    int64_t* sorted_indices,
    int64_t* indices,
    scalar_t* value,
    scalar_t* self,
    int64_t numel,
    int64_t stride,
    int64_t stride_before,
    int64_t outer_dim,
    bool accumulate) {
  if (outer_dim * stride == 0 || numel == 0) {
    return;
  }
  int64_t v_stride_before = numel * stride;
  BatchKernelConfig cfg = {
      /* num of indices      */ numel,
      /* num of elements to put per indices */ outer_dim * stride,
      1,
      numel,
      true,
      {BatchKernelConfig::Policy::pSegment,
       BatchKernelConfig::Policy::pAggressiveSplit}};

  // align with precision of CPU backend.
  using accscalar_t = scalar_t; /* acc_type<scalar_t>; */
  auto cgf = DPCPP_Q_CGF(cgh) {
    IndexPutDeterministicKernelFunctor<scalar_t, accscalar_t> kfn(
        sorted_indices,
        indices,
        value,
        self,
        numel,
        stride,
        stride_before,
        outer_dim,
        accumulate,
        v_stride_before,
        cfg);
    cgh.parallel_for<decltype(kfn)>(
        sycl::nd_range<2>(cfg.global_size(), cfg.group_size()), kfn);
  };

  DPCPP_Q_SUBMIT(dpcppGetCurrentQueue(), cgf);
} // namespace impl

static Tensor wrapIndexOnce(
    const Tensor& index,
    int64_t dim,
    int64_t dim_size,
    bool check_range = true) {
  if (index.numel() != 0 && check_range) {
    auto max_idx = index.max().item<int64_t>();
    auto min_idx = index.min().item<int64_t>();
    if (max_idx >= dim_size) {
      TORCH_CHECK_INDEX(
          false,
          "index ",
          max_idx,
          " is out of bounds for dimension ",
          dim,
          " with size ",
          dim_size);
    }
    if (min_idx < -dim_size) {
      TORCH_CHECK_INDEX(
          false,
          "index ",
          min_idx,
          " is out of bounds for dimension ",
          dim,
          " with size ",
          dim_size);
    }
  }
  return index.remainder(dim_size);
}

static std::vector<int64_t> computeLinearStride(const Tensor& tensor) {
  // computes the stride as if tensor were contiguous
  auto sizes = tensor.sizes();
  std::vector<int64_t> stride(tensor.dim());
  stride[tensor.dim() - 1] = 1;
  std::partial_sum(
      sizes.rbegin(),
      sizes.rend() - 1,
      stride.rbegin() + 1,
      std::multiplies<int64_t>());
  return stride;
}

static std::tuple<Tensor, int64_t, int64_t, int64_t> computeLinearIndex(
    const Tensor& src,
    TensorList indices,
    bool check_range) {
  auto strides = computeLinearStride(src);
  const auto& device = src.options().device();

  // Compute the linear index by multiplying the indexing tensors by the
  // stride and summing them. All the indexing tensors have the same shape at
  // this point. We also compute the number of dimensions before and after
  // that are not being index.
  Tensor linearIndex;
  int64_t nElemBefore = 1, nElemAfter = 1, strideBefore = 0;
  for (const auto i : c10::irange(src.dim())) {
    if (indices[i].defined()) {
      // Cast index to the longType matching src's device
      // This allows us to support ie indexing a xpu tensor with a cpu tensor
      Tensor index =
          (wrapIndexOnce(indices[i], i, src.size(i), check_range) * strides[i])
              .to(device);
      if (linearIndex.defined()) {
        linearIndex += index;
      } else {
        linearIndex = index;
        if (i > 0) {
          strideBefore = src.stride(i - 1); // stride after undefined dimensions
        }
      }
    } else if (linearIndex.defined()) {
      nElemAfter *= src.size(i);
    } else {
      nElemBefore *= src.size(i);
    }
  }
  return std::make_tuple(
      std::move(linearIndex), nElemBefore, strideBefore, nElemAfter);
}

static std::
    tuple<Tensor, Tensor, int64_t, int64_t, int64_t, std::vector<int64_t>>
    makeLinearIndex(
        Tensor self,
        const c10::List<c10::optional<at::Tensor>>& orig,
        bool check_range) {
  checkIndexTensorTypes(orig, /*allow_int*/ true);
  // first expand BoolTensor (masks) or ByteTensor (masks) into 1 or more
  // LongTensors
  auto indices = expandTensors(self, orig);
  for (auto& i : indices) {
    if (i.defined() && i.dtype() == at::kInt) {
      i = i.to(at::kLong);
    }
  }
  // next broadcast all index tensors together
  indices = expand_outplace(indices);
  // add missing null Tensors so that it matches self.dim()
  while (indices.size() < (size_t)self.dim()) {
    indices.emplace_back();
  }
  // if the non-null indices are not all adjacent, transpose self and indices
  // together so that they're adjacent at the front
  std::vector<int64_t> inversePerm;
  if (!hasContiguousSubspace(indices)) {
    std::tie(self, indices, inversePerm) =
        transposeToFrontAndInvPerm(self, indices);
  }
  int64_t nElemBefore, strideBefore, nElemAfter;
  Tensor linearIndex;
  std::tie(linearIndex, nElemBefore, strideBefore, nElemAfter) =
      computeLinearIndex(self, indices, check_range);
  return std::make_tuple(
      linearIndex, self, nElemBefore, strideBefore, nElemAfter, inversePerm);
}

void index_put_deterministic_impl(
    Tensor& self,
    const c10::List<c10::optional<Tensor>>& indices,
    const Tensor& value,
    bool accumulate,
    bool unsafe) {
  if (indices.size() > (size_t)self.dim()) {
    TORCH_CHECK_INDEX(
        false,
        "too many indices for tensor of dimension ",
        self.dim(),
        " (got ",
        indices.size(),
        ")");
  }
  bool self_contiguous = self.is_contiguous();
  auto self_ = self_contiguous ? self : self.contiguous();
  Tensor linearIndex, src, expandedValue = value;
  int64_t nElemBefore, strideBefore, sliceSize;
  std::vector<int64_t> inversePerm;
  std::tie(
      linearIndex, src, nElemBefore, strideBefore, sliceSize, inversePerm) =
      makeLinearIndex(self_, indices, !unsafe);
  int64_t num_indices = linearIndex.numel();

  if (expandedValue.numel() < num_indices * nElemBefore * sliceSize) {
    auto expanded_size = at::DimVector(expandedValue.sizes());
    auto size1 = expandedValue.sizes();
    auto size2 = linearIndex.sizes();
    if (are_expandable(size1, size2)) {
      expanded_size = infer_size_dimvector(size1, size2);
    }
    if (nElemBefore > 1) {
      expanded_size.insert(expanded_size.begin(), nElemBefore);
    }
    if (sliceSize > 1) {
      expanded_size.insert(expanded_size.end(), sliceSize);
    }
    expandedValue = expandedValue.expand(expanded_size);
  }
  expandedValue = expandedValue.contiguous();

  if (num_indices > 0 && sliceSize > 0) {
    const bool permuted = !src.is_contiguous();
    auto src_ = permuted ? src.contiguous() : src;
    linearIndex = linearIndex.reshape(-1);
    auto sorted_indices =
        at::empty_like(linearIndex, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
    auto orig_indices =
        at::empty_like(linearIndex, LEGACY_CONTIGUOUS_MEMORY_FORMAT);

    linearIndex.divide_(sliceSize, "trunc");

    sorted_indices.copy_(linearIndex);
    torch_ipex::xpu::pstl::iota(
        orig_indices.data_ptr<int64_t>(),
        orig_indices.data_ptr<int64_t>() + linearIndex.numel(),
        (int64_t)0);
    torch_ipex::xpu::pstl::sort<int64_t, int64_t>(
        linearIndex.data_ptr<int64_t>(),
        sorted_indices.data_ptr<int64_t>(),
        orig_indices.data_ptr<int64_t>(),
        linearIndex.numel(),
        false);
    TORCH_INTERNAL_ASSERT(
        linearIndex.numel() * sliceSize * nElemBefore == expandedValue.numel(),
        "number of flattened indices did not match number of elements in the value tensor: ",
        linearIndex.numel() * sliceSize * nElemBefore,
        " vs ",
        expandedValue.numel());
    IPEX_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(
        at::ScalarType::Half,
        at::ScalarType::Bool,
        at::ScalarType::BFloat16,
        expandedValue.scalar_type(),
        "index_put_deterministic_kernel",
        [&] {
          index_put_deterministic_kernel<scalar_t>(
              sorted_indices.data_ptr<int64_t>(),
              orig_indices.data_ptr<int64_t>(),
              expandedValue.data_ptr<scalar_t>(),
              src_.data_ptr<scalar_t>(),
              num_indices,
              sliceSize,
              strideBefore,
              nElemBefore,
              accumulate);
        });
    if (permuted)
      self.copy_(src_.permute(inversePerm));
  }
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
    const Tensor& self,
    int64_t dim,
    const Tensor& index,
    Tensor& out) {
  TORCH_CHECK(self.is_xpu(), "self must be a XPU tensor.");
  TORCH_CHECK(out.is_xpu(), "out must be a XPU tensor.");

  IPEX_DISPATCH_ALL_TYPES_AND_COMPLEX_AND5(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      at::ScalarType::Bool,
      at::ScalarType::Float8_e4m3fn,
      at::ScalarType::Float8_e5m2,
      self.scalar_type(),
      "index_select_impl",
      [=]() { impl::index_select_impl<scalar_t>(out, self, dim, index); });
  return out;
}

Tensor index_select(const Tensor& self, int64_t dim, const Tensor& index) {
  auto out = at::empty({0}, self.options());
  return at::AtenIpexTypeXPU::index_select_out(self, dim, index, out);
}

Tensor& nonzero_out(const Tensor& self, Tensor& out) {
  TORCH_CHECK(
      self.numel() < std::numeric_limits<int>::max(),
      "nonzero is not supported for tensors with more than INT_MAX elements, \
      See https://github.com/pytorch/pytorch/issues/51871");
  TORCH_CHECK(
      out.dtype() == at::kLong,
      "Expected object of scalar type ",
      at::kLong,
      " as out, but got ",
      out.dtype());
  TORCH_CHECK(
      self.device() == out.device(),
      "expected self and out to be on the same device, but got out on ",
      out.device(),
      " and self on ",
      self.device());
  TORCH_CHECK(
      self.dim() <= 16,
      "nonzero is not supported for tensor with more than ",
      16,
      " dimensions");

  IPEX_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(
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
  return at::AtenIpexTypeXPU::nonzero_out(self, out);
}

at::Tensor index_copy_meta(
    const at::Tensor& self,
    int64_t& dim,
    const at::Tensor& index,
    const at::Tensor& source,
    at::Tensor& out) {
  dim = maybe_wrap_dim(dim, self.dim());

  at::Tensor& result = out;

  // Memory overlap checks need to be done after resizing (if required) is done.
  // But it only makes sense to do these checks when result was defined, hence
  // the boolean variable `check_result` here.
  // For more details, see:
  // https://github.com/pytorch/pytorch/pull/63312#discussion_r694794832 and
  // https://github.com/pytorch/pytorch/issues/63837
  bool check_result = result.defined();
  at::TensorIterator iter = at::TensorIteratorConfig()
                                .add_output(result)
                                .add_borrowed_const_input(self)
                                .build();
  iter.set_output_raw_strided(
      0,
      self.sizes(),
      {},
      self.options(),
      result.has_names() ? result.names() : DimnameList{});
  if (check_result) {
    at::assert_no_internal_overlap(result);
    at::assert_no_overlap(result, index);
    at::assert_no_overlap(result, source);
  }

  TORCH_CHECK_INDEX(
      index.dim() < 2,
      "index_copy_(): Index should have dimension 1 or 0 (got ",
      index.dim(),
      ")");

  int64_t numIndices = index.numel();
  if (source.dim() == 0 && numIndices != 1) {
    TORCH_CHECK_INDEX(
        false,
        "index_copy_(): When source is scalar, index should have "
        "one element (got ",
        numIndices,
        ")");
  } else if (
      (source.dim() != self.dim()) && (source.dim() != 0 && self.dim() != 0)) {
    TORCH_CHECK_INDEX(
        false,
        "index_copy_(): When source and destination are not scalars, their "
        "dimensionality must match. Source dimensionality (",
        source.dim(),
        "), destination dimensionality (",
        self.dim(),
        ")");
  }

  TORCH_CHECK(
      index.scalar_type() == ScalarType::Long,
      "index_copy_(): Expected a long tensor for index, but got ",
      index.scalar_type());
  TORCH_CHECK(
      self.scalar_type() == source.scalar_type(),
      "index_copy_(): self and source expected to have the same dtype, "
      "but got (self) ",
      self.scalar_type(),
      " and (source) ",
      source.scalar_type());
  TORCH_CHECK(
      self.device() == source.device() && self.device() == index.device(),
      "index_copy_(): self, index and source expected to be in the "
      "same device, but got (self) ",
      self.device(),
      ", (index) ",
      index.device(),
      ", and (source) ",
      source.device());

  // Check that source and destination slices have the same size
  auto selfSlicedSizes = self.sizes().vec();
  if (!selfSlicedSizes.empty()) {
    selfSlicedSizes.erase(selfSlicedSizes.begin() + dim);
  }
  auto sourceSlicedSizes = source.sizes().vec();
  if (!sourceSlicedSizes.empty()) {
    sourceSlicedSizes.erase(sourceSlicedSizes.begin() + dim);
  }
  if (selfSlicedSizes.size() != sourceSlicedSizes.size() ||
      !std::equal(
          selfSlicedSizes.begin(),
          selfSlicedSizes.end(),
          sourceSlicedSizes.begin())) {
    std::stringstream ss;
    ss << "index_copy_(): Source/destination tensor must have same slice "
          "shapes. ";
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

  return iter.output();
}

template <typename scalar_t>
void index_copy_impl(
    at::Tensor& dst,
    int64_t dim,
    const at::Tensor& indices,
    const at::Tensor& source) {
  TORCH_CHECK(dst.dim() <= MAX_TENSORINFO_DIMS, DPCPPTORCH_DIM_WARNING);
  TORCH_CHECK(indices.dim() <= MAX_TENSORINFO_DIMS, DPCPPTORCH_DIM_WARNING);
  TORCH_CHECK(source.dim() <= MAX_TENSORINFO_DIMS, DPCPPTORCH_DIM_WARNING);

  // The `src` is partitioned into two parts:
  // -the size of each slice we are indexing, which is the
  // total size of the tensor ignoring dimension `dim`;
  // -the number of indices we are choosing, which is the total size
  // of the tensor `indices`.
  int64_t dstDims = dst.dim() == 0 ? 1 : dst.dim();

  TORCH_CHECK(dim >= 0 && dim < dstDims, "Indexing dim is out of bounds");

  ptrdiff_t sliceSize = 1;
  for (int d = 0; d < dstDims; d++) {
    if (d != dim) {
      sliceSize *= dst.dim() == 0 ? 1 : dst.size(d);
    }
  }
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

void index_copy_kernel(
    int64_t dim,
    const at::Tensor& index,
    const at::Tensor& source,
    at::Tensor& out) {
  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND5(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      at::ScalarType::Bool,
      at::ScalarType::Float8_e4m3fn,
      at::ScalarType::Float8_e5m2,
      out.scalar_type(),
      "index_copy",
      [&]() { index_copy_impl<scalar_t>(out, dim, index, source); });
}

Tensor& index_copy_out(
    const at::Tensor& self,
    int64_t dim,
    const at::Tensor& index,
    const at::Tensor& source,
    at::Tensor& out) {
  out = index_copy_meta(self, dim, index, source, out);

  if (!out.is_same(self))
    out.copy_(self);

  if (index.numel() == 0) {
    return out;
  }

  // See Note [Enabling Deterministic Operations]
  if (globalContext().deterministicAlgorithms()) {
    torch::List<std::optional<Tensor>> indices;
    indices.reserve(dim + 1);
    for (const auto i : c10::irange(dim)) {
      (void)i;
      indices.emplace_back();
    }
    indices.emplace_back(index);
    out.index_put_(indices, source, false);
    return out;
  }

  index_copy_kernel(dim, index, source, out);
  return out;
}

#ifdef USE_OVERRIDE_OP
at::Tensor& index_copy_(
    at::Tensor& self,
    int64_t dim,
    const at::Tensor& index,
    const at::Tensor& source) {
  return index_copy_out(self, dim, index, source, self);
}
#endif

Tensor& diag_out(const Tensor& self, int64_t diagonal, Tensor& out) {
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
  return at::AtenIpexTypeXPU::diag_out(self, diagonal, out);
}

Tensor _unsafe_index(
    const Tensor& self,
    const torch::List<c10::optional<Tensor>>& indices) {
  // Disallow boolean indexing since it leads to dynamic output shapes
  for (auto i : c10::irange(indices.size())) {
    auto index = indices.get(i);
    if (index.has_value()) {
      auto dtype = index->scalar_type();
      TORCH_CHECK(
          dtype == kLong || dtype == kInt,
          "_unsafe_index found unexpected index type ",
          dtype);
    }
  }
  return at::index(self, indices);
}

template <typename scalar_t>
struct _index_put_impl_functor {
  void operator()(char* out_data, char* in_data, int64_t offset) const {
    dpcpp_global_ptr_pt<scalar_t> out_ptr =
        (dpcpp_global_ptr_pt<scalar_t>)(out_data + offset);
    auto in = *(scalar_t*)in_data;
    atomicAdd(out_ptr, in);
  }
};

template <typename dtype>
struct _index_put_impl_functor_2 {
  void operator()(char* out_data, char* in_data, int64_t offset) const {
    *(dtype*)(out_data + offset) = *(dtype*)in_data;
  }
};

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

  if (globalContext().deterministicAlgorithms()) {
    impl::index_put_deterministic_impl(
        self, indices, value, accumulate, unsafe);
    return self;
  }

  if (accumulate) {
    if (self.scalar_type() == at::kBFloat16) {
      impl::index_put_deterministic_impl(
          self, indices, value, accumulate, unsafe);
    } else {
      auto info = make_info(self, indices);
      auto iter = make_index_put_iterator(info, value);
      IPEX_DISPATCH_ATOMIC_ALL_TYPES_AND_COMPLEX(
          iter.dtype(), "index_put_non_deterministic_acc_kernel", [&] {
            _index_put_impl_functor<scalar_t> f;
            dpcpp_index_kernel(
                iter,
                info.indexed_sizes,
                info.indexed_strides,
                IntArrayRef{},
                IntArrayRef{},
                f);
          });
    }
  } else {
    auto info = make_info(self, indices);
    auto iter = make_index_put_iterator(info, value);
    IPEX_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(
        at::ScalarType::BFloat16,
        at::ScalarType::Half,
        at::ScalarType::Bool,
        iter.dtype(),
        "index_put_non_deterministic_non_acc_kernel",
        [&] {
          using dtype = impl::OpaqueType<sizeof(scalar_t)>;
          _index_put_impl_functor_2<dtype> f;
          dpcpp_index_kernel(
              iter,
              info.indexed_sizes,
              info.indexed_strides,
              IntArrayRef{},
              IntArrayRef{},
              f);
        });
  }
  return self;
}

} // namespace AtenIpexTypeXPU
} // namespace at

#ifdef USE_OVERRIDE_OP
namespace {
at::Tensor& wrapper_XPU___index_put_impl_(
    at::Tensor& self,
    const c10::List<c10::optional<at::Tensor>>& indices,
    const at::Tensor& values,
    bool accumulate,
    bool unsafe) {
  // No device check
  const OptionalDeviceGuard device_guard(device_of(self));

  return at::AtenIpexTypeXPU::_index_put_impl_(
      self, indices, values, accumulate, unsafe);
}

at::Tensor wrapper_XPU__index_select(
    const at::Tensor& self,
    int64_t dim,
    const at::Tensor& index) {
  c10::optional<Device> common_device = nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(
      common_device, self, "wrapper_XPU__index_select", "self");
  c10::impl::check_and_update_common_device(
      common_device, index, "wrapper_XPU__index_select", "index");
  const OptionalDeviceGuard device_guard(device_of(self));

  return at::AtenIpexTypeXPU::index_select(self, dim, index);
}

at::Tensor& wrapper_XPU_out_index_select_out(
    const at::Tensor& self,
    int64_t dim,
    const at::Tensor& index,
    at::Tensor& out) {
  c10::optional<Device> common_device = nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(
      common_device, out, "wrapper_XPU_out_index_select_out", "out");
  c10::impl::check_and_update_common_device(
      common_device, self, "wrapper_XPU_out_index_select_out", "self");
  c10::impl::check_and_update_common_device(
      common_device, index, "wrapper_XPU_out_index_select_out", "index");
  const OptionalDeviceGuard device_guard(device_of(self));

  return at::AtenIpexTypeXPU::index_select_out(self, dim, index, out);
}

at::Tensor wrapper_XPU__nonzero(const at::Tensor& self) {
  c10::optional<Device> common_device = nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(
      common_device, self, "wrapper_XPU__nonzero", "self");
  const OptionalDeviceGuard device_guard(device_of(self));

  return at::AtenIpexTypeXPU::nonzero(self);
}

at::Tensor& wrapper_XPU_out_nonzero_out(
    const at::Tensor& self,
    at::Tensor& out) {
  c10::optional<Device> common_device = nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(
      common_device, out, "wrapper_XPU_out_nonzero_out", "out");
  c10::impl::check_and_update_common_device(
      common_device, self, "wrapper_XPU_out_nonzero_out", "self");
  const OptionalDeviceGuard device_guard(device_of(self));

  return at::AtenIpexTypeXPU::nonzero_out(self, out);
}

at::Tensor& wrapper_XPU_index_copy_(
    at::Tensor& self,
    int64_t dim,
    const at::Tensor& index,
    const at::Tensor& source) {
  std::optional<Device> common_device = std::nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(
      common_device, self, "wrapper_XPU_index_copy_", "self");
  c10::impl::check_and_update_common_device(
      common_device, index, "wrapper_XPU_index_copy_", "index");
  c10::impl::check_and_update_common_device(
      common_device, source, "wrapper_XPU_index_copy_", "source");

  const OptionalDeviceGuard device_guard(device_of(self));

  return at::AtenIpexTypeXPU::index_copy_(self, dim, index, source);
}

IPEX_TORCH_LIBRARY_IMPL(aten, XPU, m) {
  m.impl("_index_put_impl_", TORCH_FN((&wrapper_XPU___index_put_impl_)));
  m.impl("index_select", TORCH_FN((&wrapper_XPU__index_select)));
  m.impl("index_select.out", TORCH_FN((&wrapper_XPU_out_index_select_out)));
  m.impl("nonzero", TORCH_FN((&wrapper_XPU__nonzero)));
  m.impl("nonzero.out", TORCH_FN((&wrapper_XPU_out_nonzero_out)));
  m.impl("index_copy_", TORCH_FN((&wrapper_XPU_index_copy_)));
}
} // namespace
#endif
