#include <ATen/NativeFunctions.h>
#include <ATen/SparseTensorUtils.h>
#include <core/Memory.h>
#include <core/detail/ListUtils.h>
#include <core/detail/TensorInfo.h>
#include <runtime/Utils.h>
#include <utils/DPCPP.h>

#include "../comm/ATDispatch.h"
#include "../comm/Numerics.h"

#define I_INFO(tensor) \
  xpu::dpcpp::detail::getTensorInfo<int64_t, uint64_t>(tensor)
#define V_INFO(tensor) \
  xpu::dpcpp::detail::getTensorInfo<scalar_t, uint64_t>(tensor)

using namespace xpu::dpcpp;
using namespace at::sparse;

inline void alpha_check(const ScalarType dtype, const Scalar& alpha) {
  TORCH_CHECK(
      !alpha.isBoolean() || dtype == ScalarType::Bool,
      "Boolean alpha only supported for Boolean results.");
  TORCH_CHECK(
      isFloatingType(dtype) || isComplexType(dtype) || alpha.isIntegral(true),
      "For integral input tensors, argument alpha must not be a floating point number.");
  TORCH_CHECK(
      isComplexType(dtype) || !alpha.isComplex(),
      "For non-complex input tensors, argument alpha must not be a complex number.")
}

namespace at {
namespace AtenIpexTypeSparseXPU {

using xpu::dpcpp::detail::TensorInfo;
using indexT = int64_t;

namespace impl {

template <typename Op, typename IndexType, typename Real>
void sparse_elementwise_kernel(
    Op op,
    TensorInfo<Real, IndexType> dense,
    TensorInfo<indexT, IndexType> indices,
    TensorInfo<Real, IndexType> values,
    const IndexType nnz) {
  auto& queue = dpcppGetCurrentQueue();
  auto dev_id = dpcppGetDeviceIdOfCurrentQueue();
  IndexType group_size = (IndexType)dpcppMaxWorkGroupSize(dev_id);
  IndexType target_global_size = (IndexType)dpcppMaxWorkItemsPerTile(dev_id);
  auto max_work_group_num = target_global_size / group_size;

  auto num_groups = CeilDiv(nnz, group_size);
  if (num_groups > max_work_group_num)
    num_groups = max_work_group_num;
  auto total_items = num_groups * group_size;

  auto cgf = DPCPP_Q_CGF(cgh) {
    auto kfn = DPCPP_Q_KFN(sycl::nd_item<1> item) {
      IndexType ind_skip = indices.strides[0];
      IndexType ind_nnz_skip = indices.strides[1];
      IndexType value_size = values.strides[0];

      for (IndexType linearId = (IndexType)item.get_group_linear_id();
           linearId < nnz;
           linearId += (IndexType)item.get_group_range()[0]) {
        IndexType index = 0;
        for (IndexType d = 0; d < indices.sizes[0]; d++) {
          index = dense.sizes[d] * index +
              indices.data[d * ind_skip + linearId * ind_nnz_skip];
        }
        Real* dst = dense.data + index * value_size;
        Real* src = values.data + linearId * value_size;
        for (IndexType linearId2 = (IndexType)item.get_local_id()[0];
             linearId2 < value_size;
             linearId2 += (IndexType)item.get_local_range()[0]) {
          op(dst + linearId2, src + linearId2);
        }
      }
    };

    // kick off for kernel
    cgh.parallel_for(
        sycl::nd_range<1>(
            sycl::range<1>(total_items), sycl::range<1>(group_size)),
        kfn);
  };
  DPCPP_Q_SUBMIT(queue, cgf);
}

template <typename Op, typename IndexType, typename Real>
void sparse_elementwise_kernel_scalar(
    Op op,
    TensorInfo<Real, IndexType> dense,
    TensorInfo<indexT, IndexType> indices,
    TensorInfo<Real, IndexType> values,
    const IndexType nnz) {
  auto& queue = dpcppGetCurrentQueue();
  auto dev_id = dpcppGetDeviceIdOfCurrentQueue();
  IndexType group_size = (IndexType)dpcppMaxWorkGroupSize(dev_id);
  IndexType target_global_size = (IndexType)dpcppMaxWorkItemsPerTile(dev_id);
  auto max_work_group_num = target_global_size / group_size;

  auto num_groups = CeilDiv(nnz * group_size, group_size);
  if (num_groups > max_work_group_num)
    num_groups = max_work_group_num;
  auto total_items = num_groups * group_size;

  auto cgf = DPCPP_Q_CGF(cgh) {
    auto kfn = DPCPP_Q_KFN(sycl::nd_item<1> item) {
      IndexType ind_skip = indices.strides[0];
      IndexType ind_nnz_skip = indices.strides[1];
      IndexType value_skip = values.strides[0];

      for (IndexType linearId = (IndexType)item.get_group_linear_id() *
                   (IndexType)item.get_local_range()[0] +
               (IndexType)item.get_local_id()[0];
           linearId < nnz;
           linearId += (IndexType)item.get_group_range()[0] *
               (IndexType)item.get_local_range()[0]) {
        IndexType index = 0;
        for (IndexType d = 0; d < indices.sizes[0]; d++) {
          index = dense.sizes[d] * index +
              indices.data[d * ind_skip + linearId * ind_nnz_skip];
        }
        op(dense.data + index, values.data + linearId * value_skip);
      }
    };
    // kick off for kernel
    cgh.parallel_for(
        sycl::nd_range<1>(
            sycl::range<1>(total_items), sycl::range<1>(group_size)),
        kfn);
  };
  DPCPP_Q_SUBMIT(queue, cgf);
}
} // namespace impl

SparseTensor& mul_out_sparse_zerodim(
    SparseTensor& r,
    const SparseTensor& t,
    const Tensor& value) {
  AT_ASSERT(r.is_sparse());
  AT_ASSERT(t.is_sparse());
  AT_ASSERT(value.dim() == 0);

  if (is_same_tensor(r, t)) {
    r._values().mul_(value);
  } else {
    r.resize_as_(t);
    auto indices = r._indices();
    indices.resize_as_(t._indices());
    indices.copy_(t._indices());
    Tensor r_values = r._values();
    at::mul_out(r_values, t._values(), value);
    at::sparse::get_sparse_impl(r)->set_nnz_and_narrow(t._nnz());
    r._coalesced_(t.is_coalesced());
  }
  return r;
}

SparseTensor& mul_out_sparse_scalar(
    SparseTensor& r,
    const SparseTensor& t,
    const Scalar& value) {
  return mul_out_sparse_zerodim(r, t, native::wrapped_scalar_tensor(value));
}

template <typename T>
struct TensorCAddOp {
  TensorCAddOp(T v) : val(v) {}

  void operator()(T* out, T* in) const {
    *out += val * *in;
  }

  void operator()(T* out, T* in1, T* in2) const {
    *out = *in1 + val * *in2;
  }

  T val;
};

Tensor& add_out_dense_sparse(
    Tensor& r_,
    const Tensor& dense,
    const SparseTensor& sparse,
    const at::Scalar& value) {
  TORCH_CHECK(
      dense.sizes().equals(sparse.sizes()),
      "add: expected 'self' and 'other' to have same size, but self has size ",
      dense.sizes(),
      " while other has size ",
      sparse.sizes(),
      " (FYI: dense-sparse addition does not currently support broadcasting)");

  const int64_t nnz = sparse._nnz();
  if (nnz == 0) {
    r_.resize_as_(dense);
    r_.copy_(dense);
    return r_;
  }

  auto commonDtype = at::result_type(dense, sparse);
  TORCH_CHECK(
      canCast(commonDtype, r_.scalar_type()),
      "Can't convert result type ",
      commonDtype,
      " to output ",
      r_.scalar_type());

  Tensor r = r_;
  if (r_.scalar_type() != commonDtype) {
    r = at::empty_like(dense, r_.options().dtype(commonDtype));
  }

  Tensor dense_buffer = dense.to(commonDtype);
  Tensor values = sparse._values().to(commonDtype);

  if (is_same_tensor(r, dense_buffer)) {
    TORCH_CHECK(
        r_.is_contiguous(),
        "add: dense-sparse addition with a non-contiguous output tensor does not work; shout if you need it (see https://github.com/pytorch/pytorch/issues/1521 )");
  } else {
    r.resize_as_(dense);
    r.copy_(dense_buffer);
  }

  Tensor indices = sparse._indices();
  int64_t nDim = dense.dim();
  int64_t nDimI = sparse.sparse_dim();

  if (values.numel() == 0) {
    return r_;
  }

  if (sparse.is_coalesced()) {
    if (sparse.dense_dim() == 0) {
      IPEX_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(
          at::ScalarType::Bool,
          at::ScalarType::Half,
          at::ScalarType::BFloat16,
          commonDtype,
          "add_out_dense_sparse",
          [&] {
            impl::sparse_elementwise_kernel_scalar(
                TensorCAddOp<scalar_t>(value.to<scalar_t>()),
                V_INFO(r),
                I_INFO(indices),
                V_INFO(values),
                static_cast<uint64_t>(nnz));
          });
    } else {
      // sparseElementwiseKernel needs values to be contiguous too
      values = values.contiguous();

      IPEX_DISPATCH_ALL_TYPES_AND_COMPLEX_AND2(
          at::ScalarType::Half,
          at::ScalarType::BFloat16,
          commonDtype,
          "add_out_dense_sparse",
          [&] {
            impl::sparse_elementwise_kernel(
                TensorCAddOp<scalar_t>(value.to<scalar_t>()),
                V_INFO(r),
                I_INFO(indices),
                V_INFO(values),
                static_cast<uint64_t>(nnz));
          });
    }
  } else {
    Tensor indices1D = flatten_indices(indices, sparse.sizes(), 0);

    int64_t view_rows = 1;
    int64_t view_columns = 1;
    for (int i = 0; i < nDimI; i++) {
      view_rows *= r.size(i);
    }
    for (int i = nDimI; i < nDim; i++) {
      view_columns *= r.size(i);
    }

    Tensor r_view = r.view({view_rows, view_columns});
    values = values.reshape({nnz, view_columns});
    r_view.index_add_(0, indices1D, values, value);
  }

  r_.copy_(r);
  return r_;
}

SparseTensor& add_out_sparse(
    const SparseTensor& t,
    const SparseTensor& src,
    const Scalar& value,
    SparseTensor& r_) {
  if (!t.is_sparse()) {
    return add_out_dense_sparse(r_, t, src, value);
  }

  TORCH_CHECK(
      src.is_sparse(),
      "add(sparse, dense) is not supported. Use add(dense, sparse) instead.");
  auto commonDtype = at::result_type(t, src);
  TORCH_CHECK(
      canCast(commonDtype, r_.scalar_type()),
      "Can't convert result type ",
      commonDtype,
      " to output ",
      r_.scalar_type());

  TORCH_CHECK(
      t.sizes().equals(src.sizes()),
      "add: expected 'self' and 'other' to have same size, but ",
      t.sizes(),
      " != ",
      src.sizes());

  if (src._nnz() == 0) {
    return copy_sparse_to_sparse_(r_, t);
  }
  if (t._nnz() == 0) {
    return mul_out_sparse_scalar(r_, src, value);
  }

  TORCH_CHECK(
      is_same_density(t, src),
      "add: expected 'self' and 'other' to have same density, but 'self' has ",
      t.sparse_dim(),
      " sparse dimensions while 'other' has ",
      src.sparse_dim(),
      " sparse dimensions");

  // We deliberately choose to simply concat the indices and values tensors
  // rather than merging them. This removes the need to synchronously fetch nnz
  // at the end of the operation, at the cost of having a non-coalesced result.
  // This trade-off is preferable for the common use-case of gradient
  // accumulation.
  Tensor t_indices_ = t._indices();
  Tensor s_indices_ = src._indices();

  Tensor t_values_ = t._values().to(commonDtype);
  Tensor s_values_ = src._values().to(commonDtype);

  IPEX_DISPATCH_ALL_TYPES_AND_COMPLEX_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      commonDtype,
      "add_out_sparse_kernel",
      [&] {
        if (value.to<scalar_t>() != scalar_t(1)) {
          s_values_ = s_values_.mul(value);
        }
      });
  Tensor r_indices_ = at::cat({t_indices_, s_indices_}, 1);
  Tensor r_values_ = at::cat({t_values_, s_values_}, 0);

  if (r_.scalar_type() != commonDtype) {
    SparseTensor promoted = at::empty({0}, r_.options().dtype(commonDtype));
    promoted.resize_as_(src);
    alias_into_sparse(promoted, r_indices_, r_values_);
    // performs the addition under the common dtype.
    promoted = promoted.coalesce();
    r_values_ = promoted._values().to(r_.scalar_type());
    r_indices_ = promoted._indices();
  } else {
    r_.resize_as_(src);
  }

  alias_into_sparse(r_, r_indices_, r_values_);

  // Prevent unbounded growth of nnz
  if (r_._nnz() > r_.numel()) {
    auto c = r_.coalesce();
    alias_into_sparse(r_, c._indices(), c._values());
  }

  return r_;
}

Tensor& add_out(
    const Tensor& self,
    const Tensor& other,
    const Scalar& alpha,
    Tensor& out) {
  return add_out_sparse(self, other, alpha, out);
}

Tensor add(const Tensor& self, const Tensor& other, const Scalar& alpha) {
  TORCH_CHECK(
      !(self.is_sparse() && !other.is_sparse()),
      "add(sparse, dense) is not supported. Use add(dense, sparse) instead.");
  auto commonDtype = at::result_type(self, other);
  alpha_check(commonDtype, alpha);
  Tensor result = at::empty({0}, self.options().dtype(commonDtype));
  return at::add_out(result, self, other, alpha); // redispatch!
}

Tensor& add_(Tensor& self, const Tensor& other, const Scalar& alpha) {
  return at::add_out(self, self, other, alpha); // redispatch!
}

} // namespace AtenIpexTypeSparseXPU
} // namespace at
