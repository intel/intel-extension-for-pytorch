#include <ATen/ATen.h>
#include <ATen/native/SortingUtils.h>
#include <c10/macros/Macros.h>

#include <core/ApplyUtils.h>
#include <core/DPCPP.h>
#include <core/detail/IndexUtils.h>
#include <core/detail/TensorInfo.h>
#include <utils/Numerics.h>
#include <utils/ATDispatch.h>
#include "SortingCommon.h"
#include "SortingRadixSelect.h"

using namespace at::dpcpp;
using namespace at::native;

namespace at {
namespace AtenIpexTypeDPCPP {
namespace impl {

DPCPP_DEF_K2(
    gatherKthValueKernelName,
    typename scalar_t,
    typename index_t,
    int Dim);

template <typename scalar_t, typename index_t, int Dim>
void gatherKthValue(
    dpcpp::detail::TensorInfo<scalar_t, index_t> input,
    index_t inputSliceSize,
    index_t k,

    index_t numInputSlices,
    index_t inputWithinSliceStride,

    dpcpp::detail::TensorInfo<scalar_t, index_t> kthValue,
    dpcpp::detail::TensorInfo<int64_t, index_t> indices) {
  // Indices are limited to integer fp precision, so counts can fit in
  // int32, regardless of index_t

  auto dpcpp_queue = dpcppGetCurrentQueue();
  int64_t local_size =
      dpcpp_queue.get_device().template get_info<dpcpp_dev_max_wgroup_size>();

  auto cgf = DPCPP_Q_CGF(cgh) {
    auto in_data = get_buffer<dpcpp_r_mode>(cgh, input.data);
    auto kth_data = get_buffer<dpcpp_w_mode>(cgh, kthValue.data);
    auto indices_data = get_buffer<dpcpp_w_mode>(cgh, indices.data);

    auto smem = dpcpp_local_acc_t<int>(32, cgh);
    auto kfn = DPCPP_Q_KFN(DPCPP::nd_item<1> item) {
      index_t slice = item.get_group_linear_id();

      // Find the start offset for our slice
      auto sliceStartIndex =
          dpcpp::detail::IndexToOffset<scalar_t, index_t, Dim>::get(
              slice, input);
      auto kthValueSliceStartIndex =
          dpcpp::detail::IndexToOffset<scalar_t, index_t, Dim>::get(
              slice, kthValue);
      auto indicesSliceStartIndex =
          dpcpp::detail::IndexToOffset<int64_t, index_t, Dim>::get(
              slice, indices);

      scalar_t* inputSliceStart = get_pointer(in_data) + sliceStartIndex;
      scalar_t* kthValueSliceStart = get_pointer(kth_data) + kthValueSliceStartIndex;
      int64_t* indicesSliceStart = get_pointer(indices_data) + indicesSliceStartIndex;

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

    cgh.parallel_for<DPCPP_K(gatherKthValueKernelName, scalar_t, index_t, Dim)>(
        DPCPP::nd_range<1>(
            DPCPP::range<1>(numInputSlices * local_size),
            DPCPP::range<1>(local_size)),
        kfn);
  };

  DPCPP_Q_ASYNC_SUBMIT(dpcpp_queue, cgf);
}

struct KthValueLauncher {
  int64_t k;

  KthValueLauncher(int64_t k) : k(k) {}

  template <typename scalar_t, typename index_t, int all_dims>
  inline void launch(
      dpcpp::detail::TensorInfo<scalar_t, index_t> values_info,
      int collapse_values_dim,
      dpcpp::detail::TensorInfo<int64_t, index_t> indices_info,
      int collapse_indices_dim,
      dpcpp::detail::TensorInfo<scalar_t, index_t> self_info,
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

// this does not reduce to median with dim beause we don't want to copy twice
template <typename scalar_t>
Tensor median_template(const Tensor& self) {
  TORCH_CHECK(self.numel() > 0, "median cannot be called with empty tensor");
  if (self.dim() == 0 && self.numel() == 1) {
    return self.clone();
  }
  auto self_copy = self.clone().view(-1);
  auto values = at::empty({1}, self.options());
  auto indices = at::empty({1}, self.options().dtype(kLong));
  TORCH_CHECK(
      self.dim() <= MAX_TENSORINFO_DIMS,
      "cannot operate on more than ",
      MAX_TENSORINFO_DIMS,
      " dimensions");

  // Based on required index size, run the algorithm with the
  // appropriate index type
  if (dpcpp::detail::canUse32BitIndexMath(self) &&
      dpcpp::detail::canUse32BitIndexMath(values) &&
      dpcpp::detail::canUse32BitIndexMath(indices)) {
    run_launcher<scalar_t, uint32_t>(
        values,
        indices,
        self_copy,
        0,
        KthValueLauncher((self_copy.size(0) + 1) / 2)); // KthValue is 1-based
  } else {
    run_launcher<scalar_t, uint64_t>(
        values,
        indices,
        self_copy,
        0,
        KthValueLauncher((self_copy.size(0) + 1) / 2)); // KthValue is 1-based
  }
  return values.view({});
}

template <typename scalar_t>
void kthvalue_template(
    Tensor& values,
    Tensor& indices,
    const Tensor& self,
    int64_t k,
    int64_t dim_,
    bool keepdim) {
  int64_t dim = maybe_wrap_dim(dim_, self.dim());
  int64_t slicesize = self.size(dim);
  TORCH_CHECK(
      self.numel() > 0,
      "cannot perform reduction function kthvalue",
      " on tensor with no elements because the operation does not have "
      "an identity");
  TORCH_CHECK(k >= 1 && k <= slicesize, "selected number k out of range");

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
  if (dpcpp::detail::canUse32BitIndexMath(self) &&
      dpcpp::detail::canUse32BitIndexMath(values) &&
      dpcpp::detail::canUse32BitIndexMath(indices)) {
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

} // namespace impl

Tensor median(const Tensor& self) {
  return IPEX_DISPATCH_ALL_TYPES_AND(
      at::ScalarType::Half, self.scalar_type(), "median", [&] {
        return impl::median_template<scalar_t>(self);
      });
}

std::tuple<Tensor&, Tensor&> kthvalue_out(
    Tensor& values,
    Tensor& indices,
    const Tensor& self,
    int64_t k,
    int64_t dim,
    bool keepdim) {
  IPEX_DISPATCH_ALL_TYPES_AND(
      at::ScalarType::Half, self.scalar_type(), "kthvalue", [&] {
        impl::kthvalue_template<scalar_t>(
            values, indices, self, k, dim, keepdim);
      });
  return std::forward_as_tuple(values, indices);
}

} // namespace AtenIpexTypeDPCPP
} // namespace at
