#include <ATen/ATen.h>
#include <ATen/native/SortingUtils.h>
#include <assert.h>
#include <c10/macros/Macros.h>
#include <c10/dpcpp/SYCL.h>
#include <stdlib.h>
#include <core/SYCLApplyUtils.h>
#include <core/detail/IndexUtils.h>
#include <core/detail/TensorInfo.h>
#include <ATen/native/dpcpp/SortingCommon.h>
#include <ATen/native/dpcpp/SortingRadixSelect.h>

#include <functions/Numerics.h>


DP_DEF_K2(gatherKthValueKernelName, typename scalar_t, typename index_t, int Dim);

namespace at {
namespace native {

template <typename scalar_t, typename index_t, int Dim>
void gatherKthValue(
    sycl::detail::TensorInfo<scalar_t, index_t> input,
    index_t inputSliceSize,
    index_t k,

    index_t numInputSlices,
    index_t inputWithinSliceStride,

    sycl::detail::TensorInfo<scalar_t, index_t> kthValue,
    sycl::detail::TensorInfo<int64_t, index_t> indices) {
    // Indices are limited to integer fp precision, so counts can fit in
    // int32, regardless of index_t

    auto sycl_queue = c10::sycl::syclGetCurrentQueue();
    int64_t local_size = sycl_queue.get_device(). template get_info<dp_dev_max_wgroup_size>();

    auto cgf = DP_Q_CGF(cgh) {
      auto in_acc = c10::sycl::SYCLAccessor<dp_r_mode>(cgh, input.data);
      auto kth_acc = c10::sycl::SYCLAccessor<dp_w_mode>(cgh, kthValue.data);
      auto indices_acc = c10::sycl::SYCLAccessor<dp_w_mode>(cgh, indices.data);

      auto smem = dp_local_acc_t<int>(32, cgh);
        auto kfn = DP_Q_KFN(DP::nd_item<1> item) {

          index_t slice = item.get_group_linear_id(); 

          // Find the start offset for our slice
          auto sliceStartIndex =
              sycl::detail::IndexToOffset<scalar_t, index_t, Dim>::get(slice, input);
          auto kthValueSliceStartIndex =
              sycl::detail::IndexToOffset<scalar_t, index_t, Dim>::get(slice, kthValue);
          auto indicesSliceStartIndex =
              sycl::detail::IndexToOffset<int64_t, index_t, Dim>::get(slice, indices);

          scalar_t* inputSliceStart = in_acc.template get_pointer<scalar_t>() + sliceStartIndex;
          scalar_t* kthValueSliceStart = kth_acc.template get_pointer<scalar_t>() + kthValueSliceStartIndex;
          int64_t* indicesSliceStart = indices_acc.template get_pointer<int64_t>() + indicesSliceStartIndex;


          // Find the k-th highest element in our input
          scalar_t kValue = ScalarConvert<int, scalar_t>::to(0);
          radixSelect<scalar_t, typename TopKTypeConfig<scalar_t>::RadixType, index_t, false>(
             (dp_global_ptr_pt<scalar_t>)inputSliceStart,
             k,
             inputSliceSize,
             inputWithinSliceStride,
             smem,
             &kValue,
             item);

          // Find the index of the k-th highest element
          index_t kValueIndex = 0;
          bool foundKValue = false;
  
          for (index_t i = item.get_local_id(0); i < inputSliceSize; i += item.get_local_range(0)) {
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

      cgh.parallel_for<DP_K(gatherKthValueKernelName, scalar_t, index_t, Dim)> (
        DP::nd_range<1>(DP::range<1>(numInputSlices * local_size), DP::range<1>(local_size)), kfn);
    };

    DP_Q_ASYNC_SUBMIT(sycl_queue, cgf);
}

struct KthValueLauncher {
  int64_t k;

  KthValueLauncher(int64_t k) : k(k) {}


  template <typename scalar_t, typename index_t, int all_dims>
  inline void launch(
      sycl::detail::TensorInfo<scalar_t, index_t> values_info,
      int collapse_values_dim,
      sycl::detail::TensorInfo<int64_t, index_t> indices_info,
      int collapse_indices_dim,
      sycl::detail::TensorInfo<scalar_t, index_t> self_info,
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
Tensor median_sycl_template(const Tensor& self) {
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
  if (sycl::detail::canUse32BitIndexMath(self) &&
      sycl::detail::canUse32BitIndexMath(values) &&
      sycl::detail::canUse32BitIndexMath(indices)) {
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

Tensor median_sycl(const Tensor& self) {
  return AT_DISPATCH_ALL_TYPES_AND(at::ScalarType::Half, self.scalar_type(), "median", [&] {
    return native::median_sycl_template<scalar_t>(self);
  });
}

} // namespace native
} // namespace at
