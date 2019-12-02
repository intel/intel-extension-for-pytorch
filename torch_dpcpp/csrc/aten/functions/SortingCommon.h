#include <ATen/ATen.h>
#include <ATen/native/SortingUtils.h>
#include <assert.h>
#include <c10/macros/Macros.h>
#include <c10/dpcpp/SYCL.h>
#include <stdlib.h>
#include <core/SYCLApplyUtils.h>
#include <core/detail/TensorInfo.h>
#include <THDP/THSYCLDeviceUtils.h> // only for THCRoundUp?
#include <THDP/THSYCLScanUtils.h>

#include <utils/MathReduce.h> // AddOp


namespace at {
namespace native {

// Maximum size per grid dimension that we assume (compute capability >= 2.0)

template <typename scalar_t, typename index_t, typename Launcher>
void run_launcher(
    Tensor& values,
    Tensor& indices,
    const Tensor& self,
    int64_t dim,
    Launcher l) {
  auto self_info = sycl::detail::getTensorInfo<scalar_t, index_t>(self);
  auto values_info = sycl::detail::getTensorInfo<scalar_t, index_t>(values);
  auto indices_info = sycl::detail::getTensorInfo<int64_t, index_t>(indices);

  int64_t slice_size = self.size(dim);
  /* We use these structures solely to find the offset to */
  /* each slice we are operating on */
  self_info.reduceDim(dim);
  values_info.reduceDim(dim);
  indices_info.reduceDim(dim);

  /* Collapse all other dims */
  int collapse_self_dim = self_info.collapseDims(dim);
  int collapse_values_dim = values_info.collapseDims(dim);
  int collapse_indices_dim = indices_info.collapseDims(dim);

  int64_t num_slices = 1;
  for (int i = 0; i < self_info.dims; ++i) {
    num_slices *= self_info.sizes[i];
  }

  /* This is used as a template parameter to calculate indices. */
  /* We only specialize it if all collapsed dim sizes are the */
  /* same; otherwise, we use -1 which is the specialization */
  /* parameter for arbitrary dimensions */
  int all_dims = self_info.dims;
  if (values_info.dims != all_dims || indices_info.dims != all_dims) {
    all_dims = -1;
  }

  if (all_dims == 1) {
    l.template launch<scalar_t, index_t, 1>(
        values_info,
        collapse_values_dim,
        indices_info,
        collapse_indices_dim,
        self_info,
        collapse_self_dim,
        num_slices,
        slice_size);
  } else if (all_dims == 2) {
    l.template launch<scalar_t, index_t, 2>(
        values_info,
        collapse_values_dim,
        indices_info,
        collapse_indices_dim,
        self_info,
        collapse_self_dim,
        num_slices,
        slice_size);
  } else if (all_dims == 3) {
    l.template launch<scalar_t, index_t, 3>(
        values_info,
        collapse_values_dim,
        indices_info,
        collapse_indices_dim,
        self_info,
        collapse_self_dim,
        num_slices,
        slice_size);
  } else {
    l.template launch<scalar_t, index_t, -1>(
        values_info,
        collapse_values_dim,
        indices_info,
        collapse_indices_dim,
        self_info,
        collapse_self_dim,
        num_slices,
        slice_size);
  }
}

} // namespace native
} // namespace at
