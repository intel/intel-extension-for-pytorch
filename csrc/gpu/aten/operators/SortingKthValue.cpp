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
using namespace torch_ipex::xpu::dpcpp::detail;
using namespace torch_ipex::xpu::dpcpp;

namespace at {
namespace AtenIpexTypeXPU {
namespace impl {

template <typename scalar_t, typename index_t, int Dim>
struct GatherKthValueKernelFunctor {
  void operator()(sycl::nd_item<1> item) const {
    index_t slice = item.get_group_linear_id();

    // Find the start offset for our slice
    auto sliceStartIndex = IndexToOffset<scalar_t, index_t>::get(slice, input);
    auto kthValueSliceStartIndex =
        IndexToOffset<scalar_t, index_t>::get(slice, kthValue);
    auto indicesSliceStartIndex =
        IndexToOffset<int64_t, index_t>::get(slice, indices);

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
  }
  GatherKthValueKernelFunctor(
      TensorInfo<scalar_t, index_t> input_,
      index_t inputSliceSize_,
      index_t k_,
      index_t numInputSlices_,
      index_t inputWithinSliceStride_,
      TensorInfo<scalar_t, index_t> kthValue_,
      TensorInfo<int64_t, index_t> indices_,
      scalar_t* in_data_,
      scalar_t* kth_data_,
      int64_t* indices_data_,
      dpcpp_local_acc_t<int> smem_)
      : input(input_),
        inputSliceSize(inputSliceSize_),
        k(k_),
        numInputSlices(numInputSlices_),
        inputWithinSliceStride(inputWithinSliceStride_),
        kthValue(kthValue_),
        indices(indices_),
        in_data(in_data_),
        kth_data(kth_data_),
        indices_data(indices_data_),
        smem(smem_) {}

 private:
  TensorInfo<scalar_t, index_t> input;
  index_t inputSliceSize;
  index_t k;
  index_t numInputSlices;
  index_t inputWithinSliceStride;
  TensorInfo<scalar_t, index_t> kthValue;
  TensorInfo<int64_t, index_t> indices;
  scalar_t* in_data;
  scalar_t* kth_data;
  int64_t* indices_data;
  dpcpp_local_acc_t<int> smem;
};

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
    GatherKthValueKernelFunctor<scalar_t, index_t, Dim> kfn(
        input,
        inputSliceSize,
        k,
        numInputSlices,
        inputWithinSliceStride,
        kthValue,
        indices,
        in_data,
        kth_data,
        indices_data,
        smem);
    cgh.parallel_for<decltype(kfn)>(
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

template <typename scalar_t>
void kthvalue_template(
    const Tensor& self,
    int64_t k,
    int64_t dim_,
    bool keepdim,
    Tensor& values,
    Tensor& indices) {
  int64_t dim = maybe_wrap_dim(dim_, self.dim(), /*wrap_scalar=*/true);
  zero_numel_check_dims(self, dim, "kthvalue()");
  at::assert_no_overlap(self, values);

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
} // namespace impl

std::tuple<Tensor&, Tensor&> kthvalue_out(
    const Tensor& self,
    int64_t k,
    int64_t dim,
    bool keepdim,
    Tensor& values,
    Tensor& indices) {
  IPEX_DISPATCH_ALL_TYPES_AND2(
      at::ScalarType::BFloat16,
      at::ScalarType::Half,
      self.scalar_type(),
      "kthvalue",
      [&] {
        impl::kthvalue_template<scalar_t>(
            self, k, dim, keepdim, values, indices);
      });
  return std::forward_as_tuple(values, indices);
}

} // namespace AtenIpexTypeXPU
} // namespace at
