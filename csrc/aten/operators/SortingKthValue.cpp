#include <ATen/ATen.h>
#include <ATen/MemoryOverlap.h>
#include <ATen/native/ReduceOpsUtils.h>
#include <ATen/native/SortingUtils.h>
#include <c10/macros/Macros.h>

#include <core/Memory.h>
#include <core/detail/IndexUtils.h>
#include <core/detail/TensorInfo.h>
#include <runtime/Utils.h>
#include <utils/DPCPP.h>
#include "ReduceOpsUtils.h"
#include "SortingCommon.h"
#include "SortingRadixSelect.h"
#include "comm/ATDispatch.h"
#include "comm/ApplyUtils.h"
#include "comm/Atomics.h"
#include "comm/Numerics.h"
#include "comm/RegistrationDeclarations.h"

using namespace at::native;
using namespace xpu::dpcpp::detail;
using namespace xpu::dpcpp;

namespace at {
namespace AtenIpexTypeXPU {
namespace impl {

template <typename scalar_t, typename index_t, int Dim>
void gatherKthValue(
    TensorInfo<scalar_t, index_t> input,
    index_t inputSliceSize,
    index_t k,

    index_t numInputSlices,
    index_t inputWithinSliceStride,

    TensorInfo<scalar_t, index_t> kthValue,
    TensorInfo<int64_t, index_t> indices) {
  // Indices are limited to integer fp precision, so counts can fit in
  // int32, regardless of index_t

  auto& dpcpp_queue = dpcppGetCurrentQueue();
  auto dev_id = dpcppGetDeviceIdOfCurrentQueue();
  int64_t local_size = dpcppMaxWorkGroupSize(dev_id);

  auto cgf = DPCPP_Q_CGF(cgh) {
    auto in_data = input.data;
    auto kth_data = kthValue.data;
    auto indices_data = indices.data;

    auto smem = dpcpp_local_acc_t<int>(32, cgh);
    auto kfn = DPCPP_Q_KFN(sycl::nd_item<1> item) {
      index_t slice = item.get_group_linear_id();

      // Find the start offset for our slice
      auto sliceStartIndex =
          IndexToOffset<scalar_t, index_t, Dim>::get(slice, input);
      auto kthValueSliceStartIndex =
          IndexToOffset<scalar_t, index_t, Dim>::get(slice, kthValue);
      auto indicesSliceStartIndex =
          IndexToOffset<int64_t, index_t, Dim>::get(slice, indices);

      scalar_t* inputSliceStart = in_data + sliceStartIndex;
      scalar_t* kthValueSliceStart = kth_data + kthValueSliceStartIndex;
      int64_t* indicesSliceStart = indices_data + indicesSliceStartIndex;

      // Find the k-th highest element in our input
      scalar_t kValue = ScalarConvert<int, scalar_t>::to(0);
      radixSelect<
          scalar_t,
          typename TopKTypeConfig<scalar_t>::RadixType,
          index_t,
          false>(
          (dpcpp_global_ptr_pt<scalar_t>)inputSliceStart,
          k,
          inputSliceSize,
          inputWithinSliceStride,
          smem,
          &kValue,
          item);

      // Find the index of the k-th highest element
      index_t kValueIndex = 0;
      bool foundKValue = false;

      for (index_t i = item.get_local_id(0); i < inputSliceSize;
           i += item.get_local_range(0)) {
        bool inRange = (i < inputSliceSize);
        scalar_t v = inRange ? inputSliceStart[i * inputWithinSliceStride]
                             : static_cast<scalar_t>(0);
        bool isKValue = inRange && Numerics<scalar_t>::eq(v, kValue);

        if (isKValue) {
          kValueIndex = i;
          foundKValue = true;
          break;
        }
      }
      if (foundKValue) {
        kthValueSliceStart[0] = kValue;
        indicesSliceStart[0] = kValueIndex;
      }
    };

    cgh.parallel_for(
        sycl::nd_range<1>(
            sycl::range<1>(numInputSlices * local_size),
            sycl::range<1>(local_size)),
        kfn);
  };

  DPCPP_Q_SUBMIT(dpcpp_queue, cgf);
}

// kernel to find the median, and its index, of the values along dimension dim
template <typename scalar_t, typename index_t, int Dim>
void gatherMedian(
    TensorInfo<scalar_t, index_t> values,
    TensorInfo<int64_t, index_t> indices,
    TensorInfo<scalar_t, index_t> input,
    index_t inputSliceSize,
    index_t numInputSlices,
    index_t inputWithinSliceStride,
    bool ignore_nan) {
  // Shared memory for the subroutine RadixSelect. Note that RadixSelect
  // converts the floating point type to int with the same relative ordering.

  auto& dpcpp_queue = dpcppGetCurrentQueue();
  auto dev_id = dpcppGetDeviceIdOfCurrentQueue();
  int64_t local_size = dpcppMaxWorkGroupSize(dev_id);

  auto cgf = DPCPP_Q_CGF(cgh) {
    auto values_data = values.data;
    auto indices_data = indices.data;
    auto in_data = input.data;

    auto smem = dpcpp_local_acc_t<int>(32, cgh);
    auto num_nan = dpcpp_local_acc_t<index_t>(1, cgh);

    auto kfn = DPCPP_Q_KFN(sycl::nd_item<1> item) {
      index_t slice = item.get_group_linear_id();

      // Finds the start offset for our slice
      index_t valuesSliceStartIndex =
          IndexToOffset<scalar_t, index_t, Dim>::get(slice, values);
      index_t indicesSliceStartIndex =
          IndexToOffset<int64_t, index_t, Dim>::get(slice, indices);
      index_t inputSliceStartIndex =
          IndexToOffset<scalar_t, index_t, Dim>::get(slice, input);

      scalar_t* valuesSliceStart = values_data + valuesSliceStartIndex;
      int64_t* indicesSliceStart = indices_data + indicesSliceStartIndex;
      scalar_t* inputSliceStart = in_data + inputSliceStartIndex;

      index_t nan_count = 0;
      for (index_t i = item.get_local_id(0); i < inputSliceSize;
           i += item.get_local_range(0)) {
        scalar_t val = inputSliceStart[i * inputWithinSliceStride];
        nan_count += Numerics<scalar_t>::isnan(val) ? 1 : 0;
      }

      // Counts number of nan values
      // This code performs a parallel sum reduction
      if (item.get_local_id(0) == 0) {
        num_nan[0] = 0;
      }

      item.barrier(dpcpp_local_fence);
      if (nan_count > 0) {
        atomicAdd(num_nan.get_pointer().get(), nan_count);
      }
      item.barrier(dpcpp_local_fence);

      // For torch.median, if we found nan set k to last index so the computed
      // value is nan, otherwise set k to the middle element of the non-nan
      // values
      index_t k = (!ignore_nan && num_nan[0] > 0)
          ? inputSliceSize - 1
          : (inputSliceSize - num_nan[0] - 1) / 2;

      // Find the median
      scalar_t median = static_cast<scalar_t>(0);
      radixSelect<
          scalar_t,
          typename TopKTypeConfig<scalar_t>::RadixType,
          index_t,
          false>(
          (dpcpp_global_ptr_pt<scalar_t>)inputSliceStart,
          k + 1,
          inputSliceSize,
          inputWithinSliceStride,
          smem,
          &median,
          item);

      valuesSliceStart[0] = median;

      // Find the index of the median value in the slice
      for (index_t i = item.get_local_id(0); i < inputSliceSize;
           i += item.get_local_range(0)) {
        scalar_t val = inputSliceStart[i * inputWithinSliceStride];
        if (Numerics<scalar_t>::eq(val, median)) {
          indicesSliceStart[0] = i;
          break;
        }
      }
    };

    cgh.parallel_for(
        sycl::nd_range<1>(
            sycl::range<1>(numInputSlices * local_size),
            sycl::range<1>(local_size)),
        kfn);
  };

  DPCPP_Q_SUBMIT(dpcpp_queue, cgf);
}

struct KthValueLauncher {
  int64_t k;

  KthValueLauncher(int64_t k) : k(k) {}

  template <typename scalar_t, typename index_t, int all_dims>
  inline void launch(
      TensorInfo<scalar_t, index_t> values_info,
      int collapse_values_dim,
      TensorInfo<int64_t, index_t> indices_info,
      int collapse_indices_dim,
      TensorInfo<scalar_t, index_t> self_info,
      int collapse_self_dim,
      int64_t num_slices,
      int64_t slice_size) {
    gatherKthValue<scalar_t, index_t, all_dims>(
        self_info,
        slice_size,
        k,
        num_slices,
        /* The actual dimension that the k-selection is running in */
        /* may have changed from collapseDims() */
        self_info.strides[collapse_self_dim],
        values_info,
        indices_info);
  }
};

struct MedianLauncher {
  bool ignore_nan;

  MedianLauncher(bool ignore_nan) : ignore_nan(ignore_nan) {}

  template <typename scalar_t, typename index_t, int all_dims>
  inline void launch(
      TensorInfo<scalar_t, index_t> values_info,
      int collapse_values_dim,
      TensorInfo<int64_t, index_t> indices_info,
      int collapse_indices_dim,
      TensorInfo<scalar_t, index_t> self_info,
      int collapse_self_dim,
      int64_t num_slices,
      int64_t slice_size) {
    gatherMedian<scalar_t, index_t, all_dims>(
        values_info,
        indices_info,
        self_info,
        slice_size,
        num_slices,
        self_info.strides[collapse_self_dim],
        ignore_nan);
  }
};

std::tuple<Tensor&, Tensor&> median_with_indices_impl(
    Tensor& values,
    Tensor& indices,
    const Tensor& self,
    int64_t dim,
    bool keepdim,
    bool ignore_nan) {
  at::globalContext().alertNotDeterministic("median XPU with indices output");

  dim = at::maybe_wrap_dim(dim, self.dim());
  Tensor in = self.dim() > 0 ? self.contiguous() : self.unsqueeze(0);

  TORCH_CHECK(
      self.dim() <= MAX_TENSORINFO_DIMS,
      "cannot operate on more than ",
      MAX_TENSORINFO_DIMS,
      " dimensions");

  std::vector<int64_t> out_shape = self.sizes().vec();
  zero_numel_check_dims(self, dim, "median()");
  if (self.dim() > 0) {
    if (keepdim) {
      out_shape[dim] = 1;
    } else {
      out_shape.erase(out_shape.begin() + dim);
    }
  }

  values.resize_(out_shape);
  indices.resize_(out_shape);

  if (self.numel() > 0) {
    Tensor vals = keepdim && self.dim() > 0 ? values : values.unsqueeze(dim);
    Tensor inds = keepdim && self.dim() > 0 ? indices : indices.unsqueeze(dim);
    IPEX_DISPATCH_ALL_TYPES_AND(
        at::ScalarType::Half, self.scalar_type(), "median_out_impl", [&] {
          if (canUse32BitIndexMath(vals) && canUse32BitIndexMath(inds) &&
              canUse32BitIndexMath(in)) {
            run_launcher<scalar_t, uint32_t>(
                vals, inds, in, dim, MedianLauncher(ignore_nan));
          } else {
            run_launcher<scalar_t, uint64_t>(
                vals, inds, in, dim, MedianLauncher(ignore_nan));
          }
        });
  }
  return std::forward_as_tuple(values, indices);
}

Tensor median_impl(const Tensor& self, bool ignore_nan) {
  int64_t size = self.numel();
  // Return nan for empty tensors
  if (size <= 0) {
    return at::full({}, std::numeric_limits<float>::quiet_NaN())
        .to(self.options());
  }

  // Sort input tensor to efficiently query for median element
  Tensor sorted = std::get<0>(self.flatten().sort());

  if (!ignore_nan) {
    // For torch.median return either the middle element or nan (sorted as
    // largest) if there are any
    int64_t k = (size - 1) / 2;
    return at::where(sorted[-1].isnan(), sorted[-1], sorted[k]);
  } else {
    // For torch.nanmedian return the middle element among the non-nan values
    Tensor k = ((size - 1) - sorted.isnan().sum()) / 2;
    return sorted[k.toType(kLong)];
  }
}

template <typename scalar_t>
void kthvalue_template(
    const Tensor& self,
    int64_t k,
    int64_t dim_,
    bool keepdim,
    Tensor& values,
    Tensor& indices) {
  int64_t dim = maybe_wrap_dim(dim_, self.dim());
  at::assert_no_overlap(self, values);
  TORCH_CHECK(
      self.numel() > 0,
      "cannot perform reduction function kthvalue",
      " on tensor with no elements because the operation does not have "
      "an identity");

  if (self.dim() > 0) {
    int64_t slicesize = self.size(dim);
    TORCH_CHECK(k >= 1 && k <= slicesize, "selected number k out of range");
  } else {
    TORCH_CHECK(k <= 1, "selected number k out of range");
  }

  _reduction_with_indices_allocate_or_resize_output(
      values, indices, self, dim, keepdim);
  if (self.dim() == 0 && self.numel() == 1) {
    values.copy_(self);
    indices.zero_();
    return;
  }

  TORCH_CHECK(
      self.dim() <= MAX_TENSORINFO_DIMS,
      "cannot operate on more than ",
      MAX_TENSORINFO_DIMS,
      " dimensions");

  // Based on required index size, run the algorithm with the
  // appropriate index type
  if (canUse32BitIndexMath(self) && canUse32BitIndexMath(values) &&
      canUse32BitIndexMath(indices)) {
    run_launcher<scalar_t, uint32_t>(
        values, indices, self, dim, KthValueLauncher(k));
  } else {
    run_launcher<scalar_t, uint64_t>(
        values, indices, self, dim, KthValueLauncher(k));
  }

  if (!keepdim) {
    values.squeeze_(dim);
    indices.squeeze_(dim);
  }
}

template <typename scalar_t>
std::tuple<Tensor&, Tensor&> mode_out_template(
    Tensor& values,
    Tensor& indices,
    const Tensor& self,
    int64_t dim,
    bool keepdim) {
  auto self_ = self.contiguous();
  auto& dpcpp_queue = dpcppGetCurrentQueue();
  auto total_threads = values.size(0);

  // vertical
  /*  0 | 0 |  1  | 0 | 1
      5 | 2 | -1  | 1 | 3
      1 | 0 |  4  | 1 | 3
      2 | 1 |  3  | 2 | 2
      2 | 1 |  0  | 4 | 4
  */
  if (dim == 0 && self_.dim() != 1) {
    auto length = self_.size(1);
    auto cgf = DPCPP_Q_CGF(cgh) {
      auto self_data = self_.data_ptr<scalar_t>();
      auto values_data = values.data_ptr<scalar_t>();
      auto indices_data = indices.data_ptr<long>();

      cgh.parallel_for(sycl::range<1>(total_threads), [=](sycl::item<1> item) {
        auto self_ptr = self_data;
        auto values_ptr = values_data;
        auto indices_ptr = indices_data;

        auto id = item.get_linear_id();
        scalar_t value;
        auto indices = 0, max_count = 0;
        bool allCountOne = true;

        for (int i = 0; i < length; ++i) {
          auto elm = self_ptr[i * total_threads + id];
          auto elm_count = 1;

          for (int j = i + 1; j < length; ++j) {
            if (elm == self_ptr[j * total_threads + id]) {
              allCountOne = false;
              elm_count++;
              if (elm_count > max_count) {
                value = elm;
                max_count = elm_count;
                indices = j;
              } else if (elm_count == max_count && elm < value) {
                value = elm;
                max_count = elm_count;
                indices = j;
              }
            }
          }
        }

        // evey element only occur just once, we seek out
        if (allCountOne) {
          for (int i = 0; i < length; ++i) {
            auto elm = self_ptr[i * total_threads + id];
            if (elm < value) {
              value = elm;
              indices = i;
            }
          }
        }

        values_ptr[id] = value;
        indices_ptr[id] = indices;
      });
    };
    DPCPP_Q_SUBMIT(dpcpp_queue, cgf);

  } else {
    /*  self_.dim == 1 or horizontal seek mode.
        0 0  1 0 1
      --------------
        5 2 -1 1 3
      --------------
        1 0  4 1 3
      --------------
        2 1  3 2 2
      --------------
        2 1  0 4 4
      --------------
    */
    auto cgf = DPCPP_Q_CGF(cgh) {
      auto length = self_.size(0);
      auto self_data = self_.data_ptr<scalar_t>();
      auto values_data = values.data_ptr<scalar_t>();
      auto indices_data = indices.data_ptr<long>();

      cgh.parallel_for(sycl::range<1>(total_threads), [=](sycl::item<1> item) {
        auto self_ptr = self_data;
        auto values_ptr = values_data;
        auto indices_ptr = indices_data;

        auto id = item.get_linear_id();
        scalar_t value;
        auto indices = 0, max_count = 0;
        bool allCountOne = true;

        for (int i = 0; i < length; ++i) {
          auto elm = self_ptr[id * total_threads + i];
          auto elm_count = 1;

          for (int j = i + 1; j < length; ++j) {
            if (elm == self_ptr[id * total_threads + j]) {
              allCountOne = false;
              elm_count++;
              if (elm_count > max_count) {
                value = elm;
                max_count = elm_count;
                indices = j;
              } else if (elm_count == max_count && elm < value) {
                value = elm;
                max_count = elm_count;
                indices = j;
              }
            }
          }
        }

        // evey element only occur just once, we seek out
        if (allCountOne) {
          for (int i = 0; i < length; ++i) {
            auto elm = self_ptr[i * total_threads + id];
            if (elm < value) {
              value = elm;
              indices = i;
            }
          }
        }

        values_ptr[id] = value;
        indices_ptr[id] = indices;
      });
    };
    DPCPP_Q_SUBMIT(dpcpp_queue, cgf);
  }
  return std::forward_as_tuple(values, indices);
}

} // namespace impl

std::tuple<Tensor&, Tensor&> mode_out(
    const Tensor& self,
    int64_t dim,
    bool keepdim,
    Tensor& values,
    Tensor& indices) {
  TORCH_CHECK(
      self.layout() == Layout::Strided,
      "mode only supports strided layout, got: ",
      self.layout());
  TORCH_CHECK(
      self.device() == values.device(),
      "expected device '",
      self.device(),
      "' but got '",
      values.device(),
      "' for values output");
  TORCH_CHECK(
      self.device() == indices.device(),
      "expected device '",
      self.device(),
      "' but got '",
      indices.device(),
      "' for indices output");
  TORCH_CHECK(
      self.scalar_type() == values.scalar_type(),
      "expected scalar type '",
      self.scalar_type(),
      "' but got '",
      values.scalar_type(),
      "' for values output");
  TORCH_CHECK(
      indices.scalar_type() == ScalarType::Long,
      "expected scalar type '",
      ScalarType::Long,
      "' but got '",
      indices.scalar_type(),
      "' for indices output");

  IPEX_DISPATCH_ALL_TYPES_AND(
      at::ScalarType::Half, self.scalar_type(), "mode", [&] {
        impl::mode_out_template<scalar_t>(values, indices, self, dim, keepdim);
      });
  return std::forward_as_tuple(values, indices);
}

std::tuple<Tensor, Tensor> mode(const Tensor& self, int64_t dim, bool keepdim) {
  // https://pytorch.org/docs/stable/generated/torch.mode.html?highlight=mode#torch.mode
  // Mode means seeking out the value and indices, whose element appears most
  // often in chosen dimensions of tensor. If evey element appears just once,
  // then return the smallest element and its indices. For now, we only
  // implement less than 2 dimensions Tensor and no keepdim.
  TORCH_CHECK(
      self.dim() > 0 && self.dim() <= 2,
      "Input Tensor's dimension must be within (0, 2]");

  TORCH_CHECK(dim == -1 || dim == 0 || dim == 1, "Input dim must be -1, 0, 1");

  Tensor values, indices;

  if (self.dim() != 1) {
    auto size = self.size(dim);
    values = at::empty(size, self.options());
    indices = at::empty(size, self.options().dtype(kLong));
    auto ans =
        at::AtenIpexTypeXPU::mode_out(self, dim, keepdim, values, indices);
    values = std::get<0>(ans);
    indices = std::get<1>(ans);
  } else {
    values = at::empty({1}, self.options());
    indices = at::empty({1}, self.options().dtype(kLong));
    auto ans =
        at::AtenIpexTypeXPU::mode_out(self, dim, keepdim, values, indices);
    values = std::get<0>(ans).squeeze();
    indices = std::get<1>(ans).squeeze();
  }

  return std::forward_as_tuple(values, indices);
}

Tensor median(const Tensor& self) {
  return impl::median_impl(self, /*ignore_nan=*/false);
}

std::tuple<Tensor&, Tensor&> median_out(
    const Tensor& self,
    int64_t dim,
    bool keepdim,
    Tensor& values,
    Tensor& indices) {
  return impl::median_with_indices_impl(
      values, indices, self, dim, keepdim, /*ignore_nan=*/false);
}

std::tuple<Tensor&, Tensor&> kthvalue_out(
    const Tensor& self,
    int64_t k,
    int64_t dim,
    bool keepdim,
    Tensor& values,
    Tensor& indices) {
  IPEX_DISPATCH_ALL_TYPES_AND(
      at::ScalarType::Half, self.scalar_type(), "kthvalue", [&] {
        impl::kthvalue_template<scalar_t>(
            self, k, dim, keepdim, values, indices);
      });
  return std::forward_as_tuple(values, indices);
}

std::tuple<Tensor&, Tensor&> nanmedian_out(
    const Tensor& self,
    int64_t dim,
    bool keepdim,
    Tensor& values,
    Tensor& indices) {
  return impl::median_with_indices_impl(
      values, indices, self, dim, keepdim, /*ignore_nan=*/true);
}

Tensor nanmedian(const Tensor& self) {
  return impl::median_impl(self, /*ignore_nan=*/true);
}

} // namespace AtenIpexTypeXPU
} // namespace at
