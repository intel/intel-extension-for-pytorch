#include <ATen/ATen.h>

#include <core/TensorImplUtils.h>

using namespace at::dpcpp;

namespace at {
namespace AtenIpexTypeDPCPP {
namespace impl {

void check_shape_except_dim(Tensor& first, Tensor& second, int dimension) {
  int first_dims = first.dim();
  int second_dims = second.dim();
  TORCH_CHECK(
      first_dims == second_dims, "Tensors must have same number of dimensions");
  for (int dim = 0; dim < first_dims; dim++) {
    if (dim == dimension) {
      continue;
    }
    int64_t first_dim_size = first.size(dim);
    int64_t second_dim_size = second.size(dim);
    TORCH_CHECK(
        first_dim_size == second_dim_size,
        "Sizes of tensors must match except in dimension");
  }
}

static void cat(
    Tensor& result,
    TensorList inputs,
    int numInputs,
    int dimension) {
  // previously, size [0] tensors were the only possible empty tensors; thus, it
  // wasn't possible
  // to cat empty tensors unless all the other tensors were 1-dimensional, so we
  // allowed these tensors
  // to be "skipped".  We maintain this behavior for backwards compatibility,
  // but only for this specific
  // size (i.e. other empty sizes are not skipped).
  // FIXME: warn if this is the case
  int i, j;
  int64_t offset;
  bool hasSkippedInput = false;
  Tensor notSkippedTensor; // non-owning reference
  auto should_skip = [](const Tensor& t) {
    return !t.defined() && t.dim() == 1;
  };
  int nDims = 0;

  for (i = 0; i < numInputs; i++) {
    if (should_skip(inputs[i])) {
      hasSkippedInput = true;
      continue;
    }
    nDims = inputs[i].dim();
    notSkippedTensor = inputs[i];
  }

  // If all inputs are empty tensors, return an empty tensor
  if (!notSkippedTensor.defined()) {
    return;
  }

  TORCH_CHECK(numInputs > 0, "invalid number of inputs");
  TORCH_CHECK(dimension >= 0, "invalid dimension");

  std::vector<int64_t> size(nDims);

  // Compute size of the result in the cat dimension
  int64_t cat_dim_size = 0;
  for (int i = 0; i < numInputs; i++) {
    Tensor tensor = inputs[i];
    if (should_skip(tensor)) {
      continue;
    }
    check_shape_except_dim(notSkippedTensor, tensor, dimension);
    cat_dim_size += tensor.size(dimension);
  }

  // Compute the size of the result
  for (int dim = 0; dim < nDims; dim++) {
    int64_t result_dim_size = notSkippedTensor.size(dim);
    if (dim == dimension) {
      result_dim_size = cat_dim_size;
    }
    size[dim] = result_dim_size;
  }
  result.resize_(size);

  offset = 0;
  for (j = 0; j < numInputs; j++) {
    if (should_skip(inputs[j]))
      continue;
    int64_t dimSize = inputs[j].size(dimension);
    auto nt = at::empty_like(result);
    auto result_impl = TensorImpl_Unwrap(result);
    TensorImpl_setStorageNd(
        TensorImpl_Unwrap(nt),
        TensorImpl_getStoragePtr(result_impl),
        result.storage_offset(),
        result.dim(),
        TensorImpl_getSizePtr(result_impl),
        TensorImpl_getStridePtr(result_impl));
    nt = nt.narrow(dimension, offset, dimSize);
    nt.copy_(inputs[j]);
    offset += dimSize;
  }
}

} // namespace impl

Tensor& _cat_out(Tensor& out, TensorList tensors, int64_t dim) {
  impl::cat(out, tensors, tensors.size(), dim);
  return out;
}

Tensor _cat(TensorList tensors, int64_t dim) {
  auto out = at::empty({0}, tensors[0].options());
  return at::AtenIpexTypeDPCPP::_cat_out(out, tensors, dim);
}

} // namespace AtenIpexTypeDPCPP
} // namespace at
