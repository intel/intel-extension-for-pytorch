#include <ATen/NativeFunctions.h>
#include <ATen/SparseTensorUtils.h>
#include <core/Memory.h>
#include <runtime/Utils.h>

#include "comm/ATDispatch.h"
#include "comm/AccumulateType.h"
#include "comm/Numerics.h"
#include "comm/PSTLFunctions.h"

#ifdef USE_ONEDPL
#include <oneapi/dpl/algorithm>
#include <oneapi/dpl/execution>
#include <oneapi/dpl/iterator>
#include <oneapi/dpl/numeric>
#endif

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
    const TensorOptions& options) {
  return at::native::new_with_dims_and_tensor_sparse(
      sparse_dim, dense_dim, size, indices, values, options);
}

Tensor empty(
    IntArrayRef size,
    const TensorOptions& options,
    c10::optional<MemoryFormat> memory_format) {
  return at::native::empty_sparse(size, options, memory_format);
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

Tensor coalesce(const Tensor& self) {
#ifndef USE_ONEDPL
  throw std::runtime_error("no oneDPL found when compile");
#else
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
  LongTensor indices1D = flatten_indices(self._indices(), self.sizes(), true);

  LongTensor origIndices = at::empty({nnz}, self._indices().options());
  LongTensor uniqueOffsets = at::empty({nnz}, self._indices().options());

  auto& dpcpp_queue = dpcppGetCurrentQueue();
  auto policy = oneapi::dpl::execution::make_device_policy(dpcpp_queue);

  {
    auto countIterI = oneapi::dpl::counting_iterator<int64_t>(0);
    auto countIterO = oneapi::dpl::counting_iterator<int64_t>(0);

    auto origIndices_ptr = origIndices.data_ptr<int64_t>();
    auto uniqueOffsets_ptr = uniqueOffsets.data_ptr<int64_t>();

    std::copy(policy, countIterI, countIterI + nnz, origIndices_ptr);
    std::copy(policy, countIterO, countIterO + nnz, uniqueOffsets_ptr);

    auto indices1D_ptr = indices1D.data_ptr<int64_t>();
    auto zipped_indices =
        oneapi::dpl::make_zip_iterator(indices1D_ptr, origIndices_ptr);
    std::sort(
        policy, zipped_indices, zipped_indices + nnz, [](auto lhs, auto rhs) {
          using std::get;
          return get<0>(lhs) < get<0>(rhs);
        });
    auto zipped_uniqueOffsets =
        oneapi::dpl::make_zip_iterator(indices1D_ptr, uniqueOffsets_ptr);
    auto newEnd = at::AtenIpexTypeXPU::unique(
        zipped_uniqueOffsets,
        zipped_uniqueOffsets + nnz,
        [](auto lhs, auto rhs) {
          using std::get;
          return get<0>(lhs) == get<0>(rhs);
        });
    newNnz = std::distance(zipped_uniqueOffsets, newEnd);
  }

  indices1D.resize_({1, newNnz});
  auto newValues_size = values.sizes().vec();
  newValues_size[0] = newNnz;
  Tensor newValues = at::empty(newValues_size, values.options());

  if (newValues.numel() > 0) {
    values = values.contiguous();
    int64_t stride = at::prod_intlist(values.sizes().slice(1));
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

  LongTensor newIndices;
  if (sparse_dim == 1) {
    newIndices = indices1D;
  } else {
    newIndices = at::empty({sparse_dim, newNnz}, origIndices.options());
    for (int64_t d = sparse_dim - 1; d >= 0; d--) {
      // NB: Not a select, so I can preserve the outer dimension
      LongTensor indicesSlice = newIndices.narrow(0, d, 1);
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

#endif
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
  LongTensor mask_indices = mask._indices();
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
  LongTensor indices = at::zeros({mask._nnz()}, mask_indices.options());
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

} // namespace AtenIpexTypeSparseXPU
} // namespace at
