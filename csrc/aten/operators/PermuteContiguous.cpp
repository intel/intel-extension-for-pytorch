#include <ATen/ATen.h>
#include <ATen/CPUApplyUtils.h>
#include <ATen/DeviceGuard.h>
#include <ATen/NativeFunctions.h>
#include <ATen/record_function.h>
#include <grp.h>
#include <oneapi/dnnl/dnnl.hpp>

#include <oneDNN/oneDNN.h>

namespace at {
namespace AtenIpexTypeXPU {

at::Tensor permute_contiguous(
    const at::Tensor& self,
    at::IntArrayRef dims,
    at::MemoryFormat dim_contiguous) {
  Tensor result;
  // plain format tensor will go through naitve permute contiguous pass
  if (DPCPPTensorContext::get_tensor_ctx(self).is_plain()) {
    result = at::native::permute(self, dims).contiguous(dim_contiguous);
    return result;
  }
  // block format tensor will be reordered to plain format in this fusion, and
  // it mainly consists of 4 steps.

  // 1. run some checks and calculate the output tensor shape.
  auto nDims = self.dim();
  TORCH_CHECK(
      dims.size() == (size_t)nDims, "number of dims don't match in permute");
  auto oldSizes = self.sizes();
  auto oldStrides = self.strides();
  DimVector newSizes(nDims);
  DimVector newStrides(nDims);
  std::vector<bool> seen(nDims);
  for (const auto i : c10::irange(nDims)) {
    auto dim = at::maybe_wrap_dim(dims[i], nDims);
    TORCH_CHECK(!seen[dim], "repeated dim in permute");
    seen[dim] = true;
    newSizes[i] = oldSizes[dim];
    newStrides[i] = oldStrides[dim];
  }
  DimVector revert_dim(nDims);

  // 2.calculate reverse permute index for new tensor.
  for (const auto i : c10::irange(nDims)) {
    revert_dim[dims[i]] = i;
  }
  if (self.is_quantized()) {
    result = at::_empty_affine_quantized(
        newSizes,
        self.options(),
        self.q_scale(),
        self.q_zero_point(),
        dim_contiguous);

  } else {
    result = at::empty(newSizes, self.options(), dim_contiguous);
  }

  // 3.permute the new contiguous tensor to same shape against input.
  Tensor permute_one = at::native::permute(result, revert_dim);

  // 4.reorder the input tensor to plain format and put it into the new tensor,
  // which will be contiguous in the shape of the desire output one.
  ::xpu::oneDNN::reorder(self, permute_one);
  result = at::native::permute(permute_one, dims);
  return result;
}
} // namespace AtenIpexTypeXPU
} // namespace at
