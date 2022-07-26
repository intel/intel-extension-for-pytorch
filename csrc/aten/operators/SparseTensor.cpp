#include <ATen/NativeFunctions.h>
#include <ATen/SparseTensorUtils.h>
#include <core/Memory.h>
#include <core/detail/ListUtils.h>
#include <runtime/Utils.h>

#include "BitonicMergeSort.h"
#include "PSTLFunctions.h"
#include "comm/ATDispatch.h"
#include "comm/AccumulateType.h"
#include "comm/Numerics.h"

using namespace xpu::dpcpp;
using namespace at::sparse;

namespace at {
namespace AtenIpexTypeSparseXPU {
namespace impl {

template <typename scalar_t>
void coalesce_values_kernel(
    Tensor segment_offsets,
    Tensor value_indices,
    Tensor values,
    Tensor newValues,
    int64_t nnz,
    int64_t newNnz,
    int64_t stride) {
  using accscalar_t = AtenIpexTypeXPU::acc_type<scalar_t>;

  auto& queue = dpcppGetCurrentQueue();
  const int num_group_0 = CeilDiv(newNnz, (int64_t)4);
  const int num_group_1 = CeilDiv(stride, (int64_t)64);

  auto cgf = DPCPP_Q_CGF(cgh) {
    auto segment_offsets_data = segment_offsets.data_ptr<int64_t>();
    auto value_indices_data = value_indices.data_ptr<int64_t>();
    auto values_data = values.data_ptr<scalar_t>();
    auto newValues_data = newValues.data_ptr<scalar_t>();
    auto kfn = DPCPP_Q_KFN(DPCPP::nd_item<2> item) {
      auto segment_offsets_ptr = segment_offsets_data;
      auto value_indices_ptr = value_indices_data;
      auto values_ptr = values_data;
      auto newValues_ptr = newValues_data;

      int seg = item.get_global_id()[0];

      if (seg < newNnz) {
        const int newValueRow = seg * stride;
        const int begin = segment_offsets_ptr[seg];
        const int end = (seg < newNnz - 1) ? segment_offsets_ptr[seg + 1] : nnz;
        const int featureDim = item.get_global_id()[1];

        accscalar_t tmp = 0;
        for (int row = begin; row < end; row++) {
          const int valueRow = ((int)value_indices_ptr[row]) * stride;
          if (featureDim < stride) {
            tmp += static_cast<accscalar_t>(values_ptr[valueRow + featureDim]);
          }
        }
        if (featureDim < stride) {
          newValues_ptr[newValueRow + featureDim] = static_cast<scalar_t>(tmp);
        }
      }
    };

    // kick off kernel
    cgh.parallel_for(
        DPCPP::nd_range<2>(
            DPCPP::range<2>(num_group_0 * 4, num_group_1 * 64),
            DPCPP::range<2>(4, 64)),
        kfn);
  };
  DPCPP_Q_SUBMIT(queue, cgf);
}
} // namespace impl

Tensor _sparse_coo_tensor_with_dims_and_tensors(
    int64_t sparse_dim,
    int64_t dense_dim,
    IntArrayRef size,
    const Tensor& indices,
    const Tensor& values,
    c10::optional<at::ScalarType> dtype,
    c10::optional<at::Layout> layout,
    c10::optional<at::Device> device,
    c10::optional<bool> pin_memory) {
  TensorOptions options =
      TensorOptions().dtype(dtype).layout(layout).device(device).pinned_memory(
          pin_memory);
  return at::native::new_with_dims_and_tensor_sparse(
      sparse_dim,
      dense_dim,
      size,
      indices,
      values,
      options.dtype().toScalarType(),
      options.layout(),
      options.device(),
      options.pinned_memory());
}

Tensor empty(
    IntArrayRef size,
    const TensorOptions& options,
    c10::optional<MemoryFormat> memory_format) {
  return at::native::empty_sparse(
      size,
      options.dtype().toScalarType(),
      options.layout(),
      options.device(),
      options.pinned_memory(),
      memory_format);
}

Tensor _indices(const Tensor& self) {
  return at::native::_indices_sparse(self);
}

Tensor _values(const Tensor& self) {
  return at::native::_values_sparse(self);
}

Tensor& copy_sparse_to_sparse_(
    Tensor& self,
    const Tensor& src,
    bool non_blocking) {
  return at::native::copy_sparse_(self, src, non_blocking);
}

Tensor& _coalesced_(Tensor& self, bool coalesced) {
  return at::native::_coalesced_sparse_(self, coalesced);
}

bool is_coalesced(const Tensor& self) {
  return at::native::is_coalesced_sparse(self);
}

int64_t dense_dim(const Tensor& self) {
  return at::native::dense_dim_sparse(self);
}

int64_t sparse_dim(const Tensor& self) {
  return at::native::sparse_dim_sparse(self);
}

int64_t _nnz(const Tensor& self) {
  return at::native::_nnz_sparse(self);
}

SparseTensor coalesce(const SparseTensor& self) {
  if (self.is_coalesced()) {
    return self;
  }
  return at::_coalesce(self);
}

Tensor _coalesce(const Tensor& self) {
  int64_t nnz = self._nnz();
  if (self.is_coalesced()) {
    return self;
  }
  // NOTE: Since `coalesce` is not an in-place operation when `is_coalesced` is
  // false, we should keep the original tensor intact and do coalesce on a copy
  // of the tensor
  if (nnz < 2) {
    SparseTensor dst = self.clone();
    dst._coalesced_(true);
    return dst;
  }

  Tensor values = self._values();

  int64_t sparse_dim = self.sparse_dim();
  int64_t newNnz;

  // indices will be modified by Thrust, so we have to clone or use new storage
  // here.
  Tensor indices1D = flatten_indices(self._indices(), self.sizes(), true);

  Tensor origIndices = at::empty({nnz}, self._indices().options());
  Tensor uniqueOffsets = at::empty({nnz}, self._indices().options());

  auto origIndices_ptr = origIndices.data_ptr<int64_t>();
  auto uniqueOffsets_ptr = uniqueOffsets.data_ptr<int64_t>();

  xpu::pstl::iota<int64_t>(origIndices_ptr, origIndices_ptr + nnz, (int64_t)0);
  xpu::pstl::iota<int64_t>(
      uniqueOffsets_ptr, uniqueOffsets_ptr + nnz, (int64_t)0);

  auto indices1D_ptr = indices1D.data_ptr<int64_t>();
  xpu::pstl::merge_sort<int64_t, int64_t>(
      indices1D_ptr,
      origIndices_ptr,
      indices1D.size(0),
      [](int64_t a, int64_t b) { return Numerics<int64_t>::lt(a, b); });

  auto indices1D_end = indices1D_ptr;
  auto uniqueOffsets_end = uniqueOffsets_ptr;
  std::tie(indices1D_end, uniqueOffsets_end) =
      xpu::pstl::unique_with_zip<int64_t, int64_t, int64_t>(
          indices1D_ptr,
          indices1D_ptr + nnz,
          uniqueOffsets_ptr,
          [](auto lhs, auto rhs) { return Numerics<int64_t>::eq(lhs, rhs); });
  newNnz = std::distance(indices1D_ptr, indices1D_end);

  indices1D.resize_({1, newNnz});
  auto newValues_size = values.sizes().vec();
  newValues_size[0] = newNnz;
  Tensor newValues = at::empty(newValues_size, values.options());

  if (newValues.numel() > 0) {
    values = values.contiguous();
    int64_t stride = xpu::dpcpp::detail::prod_intlist(values.sizes().slice(1));
    IPEX_DISPATCH_ALL_TYPES_AND2(
        at::ScalarType::BFloat16,
        at::ScalarType::Half,
        values.scalar_type(),
        "coalesce",
        [&]() {
          impl::coalesce_values_kernel<scalar_t>(
              uniqueOffsets,
              origIndices,
              values,
              newValues,
              nnz,
              newNnz,
              stride);
        });
  }

  Tensor newIndices;
  if (sparse_dim == 1) {
    newIndices = indices1D;
  } else {
    newIndices = at::empty({sparse_dim, newNnz}, origIndices.options());
    for (int64_t d = sparse_dim - 1; d >= 0; d--) {
      // NB: Not a select, so I can preserve the outer dimension
      Tensor indicesSlice = newIndices.narrow(0, d, 1);
      // Note for the porting guide: THCTensor_(copy) does NOT do normal
      // broadcasting logic; instead, it will blast the elements from one
      // to the other so long as the numel is the same
      indicesSlice.copy_(indices1D);
      indices1D.floor_divide_(self.size(d));
      indicesSlice.add_(indices1D, -self.size(d));
    }
  }
  ////////////////////////////////////////////////////////////
  // We can use unsafe sparse tensor constructor because the indices do not
  // need to be revalidated as we do not add or change indices, just remove
  // duplicates.
  SparseTensor dst =
      at::_sparse_coo_tensor_unsafe(newIndices, newValues, self.sizes())
          ._coalesced_(true);

  return dst;
}

Tensor sparse_mask(const Tensor& self, const Tensor& mask) {
  SparseTensor r = at::empty({0}, self.options().layout(kSparse));
  TORCH_CHECK(mask.is_coalesced(), "sparse_mask: mask is uncoalesced");
  TORCH_CHECK(
      mask.sizes().equals(self.sizes()),
      "sparse_mask: operands have incompatible sizes; self has size ",
      self.sizes(),
      " but mask has size ",
      mask.sizes());
  r.resize_as_(mask);
  if (mask._nnz() == 0) {
    return r.zero_();
  }
  Tensor mask_indices = mask._indices();
  Tensor mask_values = mask._values();
  Tensor r_values = at::empty(mask_values.sizes(), r._values().options());
  alias_into_sparse(
      r, mask_indices.clone(at::MemoryFormat::Contiguous), r_values);
  r._coalesced_(mask.is_coalesced());
  if (self.numel() ==
      0) { // if t is an empty tensor, there is no need to mask its elements
    return r;
  }

  // Get a flattened sparse indices, similar to NOTE [ Flatten Sparse Indices ].
  // Keeping this implementation because it is faster than flatten_indices()
  Tensor indices = at::zeros({mask._nnz()}, mask_indices.options());
  for (int64_t d = 0; d < mask.sparse_dim(); d++) {
    indices.mul_(mask.size(d));
    // This used to use a buffer but I deoptimized it
    indices.add_(mask_indices.select(0, d));
  }

  std::vector<int64_t> view_size(1 + mask.dense_dim());
  view_size[0] = -1;
  for (int64_t d = 0; d < mask.dense_dim(); d++) {
    view_size[d + 1] = mask.size(mask.sparse_dim() + d);
  }

  Tensor self_view;
  if (self.is_contiguous())
    self_view = self.view(view_size);
  else
    self_view = self.contiguous().view(view_size);
  // TODO: Re-audit this; it used to be an indexSelect directly into r_values
  at::index_select_out(r_values, self_view, 0, indices);

  return r;
}

Tensor empty(
    IntArrayRef size,
    c10::optional<at::ScalarType> dtype,
    c10::optional<at::Layout> layout,
    c10::optional<at::Device> device,
    c10::optional<bool> pin_memory,
    c10::optional<MemoryFormat> optional_memory_format) {
  TensorOptions options =
      TensorOptions().dtype(dtype).layout(layout).device(device).pinned_memory(
          pin_memory);
  return empty(size, options, optional_memory_format);
}
} // namespace AtenIpexTypeSparseXPU
} // namespace at
