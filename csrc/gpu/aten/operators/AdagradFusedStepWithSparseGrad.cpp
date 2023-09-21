#include <ATen/ATen.h>
#include <ATen/DeviceGuard.h>
#include <ATen/Functions.h>
#include <ATen/native/Activation.h>
#include <ATen/native/SparseTensorUtils.h>
#include <ATen/record_function.h>
#include <core/detail/ListUtils.h>
#include <torch/library.h>
#include "Loops.h"
#include "LoopsTemplates.h"
#include "PSTLFunctions.h"
#include "SparseTensorUtils.h"
#include "comm/ATDispatch.h"
#include "comm/AccumulateType.h"
#include "comm/ApplyUtils.h"
#include "comm/Numerics.h"

#include <aten/operators/MemoryAccess.h>
#include "utils/CustomOperatorRegistration.h"

using namespace xpu::dpcpp;
using namespace xpu::dpcpp::detail;
using namespace xpu::dpcpp;
using namespace at::sparse;

namespace at {
namespace AtenIpexTypeXPU {
namespace impl {

#define MinLocalCol 32
// The MaxLoop can reduce the number of groups. And, the
// adagrad_fused_step_nocoalesced_large_group_kernel_impl can get great perf.
// However. adagrad_fused_step_nocoalesced_large_group_kernel_impl will get bad
// perf due to huge number of total loop(MaxLoop * nnz/newNnz). So, we limits
// MaxLoop to 16.
#define MaxLoop 16

static inline int64_t log2_ceil(int64_t value) {
  int64_t log2_value = 0;
  while ((1 << log2_value) < value)
    ++log2_value;
  return log2_value;
}

static inline int64_t log2_floor(int64_t value) {
  int64_t log2_value = 0;
  while ((1 << log2_value) < value)
    ++log2_value;
  if (1 << log2_value > value)
    --log2_value;
  return log2_value;
}

static inline void get_wgroup_size_for_coalesced_tensor(
    int64_t nnz,
    int64_t stride,
    int64_t& global_size_row,
    int64_t& global_size_col,
    int64_t& local_size_row,
    int64_t& local_size_col) {
  auto dev_id = dpcppGetDeviceIdOfCurrentQueue();
  int64_t maxWGSize = dpcppMaxWorkGroupSize(dev_id);
  auto max_group = dpcppMaxWorkItemsPerTile(dev_id) / maxWGSize;

  int64_t exp2_floor = 1 << log2_floor(stride);
  if (exp2_floor >= maxWGSize) {
    local_size_col = maxWGSize;
  } else if (
      CeilDiv(nnz, max_group) ==
      CeilDiv(nnz * CeilDiv(stride, exp2_floor), max_group)) {
    local_size_col = exp2_floor;
  } else {
    local_size_col = stride;
  }
  local_size_row = maxWGSize / local_size_col;

  global_size_row = CeilDiv(nnz, local_size_row);
  global_size_col = CeilDiv(stride, local_size_col);
}

template <typename scalar_t, typename IndexType>
void adagrad_fused_step_coalesced_kernel_impl(
    Tensor param,
    Tensor state_sum,
    Tensor param_indices,
    Tensor values,
    int64_t nnz,
    int64_t stride_grad,
    int64_t stride_param,
    float lr,
    float eps) {
  using accscalar_t = AtenIpexTypeXPU::acc_type<scalar_t>;

  auto& queue = dpcppGetCurrentQueue();
  int64_t global_size_row = 0, global_size_col = 0, local_size_row = 0,
          local_size_col = 0;
  get_wgroup_size_for_coalesced_tensor(
      nnz,
      stride_grad,
      global_size_row,
      global_size_col,
      local_size_row,
      local_size_col);

  auto cgf = DPCPP_Q_CGF(cgh) {
    auto param_data = param.data_ptr<scalar_t>();
    auto state_sum_data = state_sum.data_ptr<scalar_t>();
    auto param_indices_data = param_indices.data_ptr<int64_t>();
    auto values_data = values.data_ptr<scalar_t>();
    auto kfn = DPCPP_Q_KFN(sycl::nd_item<2> item) {
      auto param_ptr = param_data;
      auto state_sum_ptr = state_sum_data;
      auto param_indices_ptr = param_indices_data;
      auto values_ptr = values_data;

      IndexType seg = item.get_global_id()[0];
      const uint64_t featureDim = item.get_global_id()[1];

      if (seg < nnz && featureDim < stride_grad) {
        const uint64_t newValueRow =
            stride_param * param_indices_ptr[seg] * stride_grad;

        accscalar_t tmp = 0, std = 0;
        const uint64_t valueRow = seg * stride_grad;
        tmp = static_cast<accscalar_t>(values_ptr[valueRow + featureDim]);
        state_sum_data[newValueRow + featureDim] += Numerics<accscalar_t>::pow(
            static_cast<accscalar_t>(tmp), accscalar_t(2.0));
        std = Numerics<accscalar_t>::sqrt(
                  state_sum_data[newValueRow + featureDim]) +
            static_cast<accscalar_t>(eps);
        param_data[newValueRow + featureDim] +=
            tmp / std * static_cast<accscalar_t>(-lr);
      }
    };

    // kick off kernel
    cgh.parallel_for(
        sycl::nd_range<2>(
            sycl::range<2>(
                global_size_row * local_size_row,
                global_size_col * local_size_col),
            sycl::range<2>(local_size_row, local_size_col)),
        kfn);
  };
  DPCPP_Q_SUBMIT(queue, cgf);
}

static inline void get_wgroup_size(
    int64_t nnz,
    int64_t newNnz,
    int64_t stride,
    int64_t& global_size_row,
    int64_t& global_size_col,
    int64_t& local_size_row,
    int64_t& local_size_col) {
  auto dev_id = dpcppGetDeviceIdOfCurrentQueue();
  int64_t maxWGSize = dpcppMaxWorkGroupSize(dev_id);
  int64_t avg_num = CeilDiv(nnz, newNnz);

  int64_t exp2_ceil = 1 << log2_ceil(stride);
  if (avg_num * stride < maxWGSize) {
    local_size_col = exp2_ceil;
    if (avg_num * exp2_ceil < maxWGSize) {
      local_size_row = avg_num;
    } else {
      local_size_row = maxWGSize / exp2_ceil;
    }
  } else if (stride <= MinLocalCol) {
    local_size_row = maxWGSize / exp2_ceil;
    local_size_col = exp2_ceil;
  } else {
    local_size_col = MinLocalCol;
    while (avg_num * (local_size_col << 1) <= maxWGSize) {
      local_size_col = local_size_col << 1;
    }
    local_size_row = maxWGSize / local_size_col;
  }
  global_size_row = newNnz;
  global_size_col = CeilDiv(stride, local_size_col);
}

// Cycle sum of a unique index can be processed by multi-items. And, the result
// will be saved by reducing multi-items.
template <typename scalar_t, typename IndexType>
void adagrad_fused_step_nocoalesced_kernel_impl(
    Tensor param,
    Tensor state_sum,
    Tensor segment_offsets,
    Tensor value_indices,
    Tensor param_indices,
    Tensor values,
    int64_t nnz,
    int64_t newNnz,
    int64_t stride_grad,
    int64_t stride_param,
    float lr,
    float eps) {
  using accscalar_t = AtenIpexTypeXPU::acc_type<scalar_t>;

  auto& queue = dpcppGetCurrentQueue();
  int64_t global_size_row = 0, global_size_col = 0, local_size_row = 0,
          local_size_col = 0;
  get_wgroup_size(
      nnz,
      newNnz,
      stride_grad,
      global_size_row,
      global_size_col,
      local_size_row,
      local_size_col);

  auto cgf = DPCPP_Q_CGF(cgh) {
    auto param_data = param.data_ptr<scalar_t>();
    auto state_sum_data = state_sum.data_ptr<scalar_t>();
    auto segment_offsets_data = segment_offsets.data_ptr<int64_t>();
    auto value_indices_data = value_indices.data_ptr<int64_t>();
    auto param_indices_data = param_indices.data_ptr<int64_t>();
    auto values_data = values.data_ptr<scalar_t>();

    auto local_sum = dpcpp_local_acc_t<accscalar_t, 2>(
        sycl::range<2>{local_size_row, local_size_col}, cgh);

    auto kfn = DPCPP_Q_KFN(sycl::nd_item<2> item) {
      uint64_t stride_grad_remainder = item.get_global_id()[1] / local_size_col;
      uint64_t stride_grad_div = item.get_local_id(1);
      IndexType seg = item.get_global_id()[0] / local_size_row;
      IndexType reduce_id = item.get_local_id(0);

      const uint64_t featureDim =
          stride_grad_remainder * local_size_col + stride_grad_div;

      auto param_ptr = param_data;
      auto state_sum_ptr = state_sum_data;
      auto segment_offsets_ptr = segment_offsets_data;
      auto value_indices_ptr = value_indices_data;
      auto param_indices_ptr = param_indices_data;
      auto values_ptr = values_data;

      if (seg < newNnz && featureDim < stride_grad) {
        const IndexType begin = segment_offsets_ptr[seg];
        const IndexType end =
            (seg < newNnz - 1) ? segment_offsets_ptr[seg + 1] : nnz;

        accscalar_t tmp = 0, std = 0;
        for (IndexType row = begin + reduce_id; row < end;
             row += local_size_row) {
          const uint64_t valueRow =
              ((IndexType)value_indices_ptr[row]) * stride_grad;
          tmp += static_cast<accscalar_t>(values_ptr[valueRow + featureDim]);
        }

        local_sum[reduce_id][stride_grad_div] = tmp;
        item.barrier(dpcpp_local_fence);

        tmp = 0;
        const uint64_t newValueRow =
            stride_param * param_indices_ptr[seg] * stride_grad;
        if (reduce_id == 0) {
          for (IndexType i = 0; i < local_size_row; ++i) {
            tmp += local_sum[i][stride_grad_div];
          }

          state_sum_data[newValueRow + featureDim] +=
              Numerics<accscalar_t>::pow(
                  static_cast<accscalar_t>(tmp), accscalar_t(2.0));
          std = Numerics<accscalar_t>::sqrt(
                    state_sum_data[newValueRow + featureDim]) +
              static_cast<accscalar_t>(eps);
          param_data[newValueRow + featureDim] +=
              tmp / std * static_cast<accscalar_t>(-lr);
        }
      }
    };

    // kick off kernel
    cgh.parallel_for(
        sycl::nd_range<2>(
            sycl::range<2>(
                global_size_row * local_size_row,
                global_size_col * local_size_col),
            sycl::range<2>(local_size_row, local_size_col)),
        kfn);
  };
  DPCPP_Q_SUBMIT(queue, cgf);
}

static inline void get_wgroup_size_large(
    int64_t nnz,
    int64_t newNnz,
    int64_t stride,
    int64_t& global_size_row,
    int64_t& global_size_col,
    int64_t& local_size_row,
    int64_t& local_size_col,
    int64_t& loop) {
  auto dev_id = dpcppGetDeviceIdOfCurrentQueue();
  int64_t maxWGSize = dpcppMaxWorkGroupSize(dev_id);
  auto max_group = dpcppMaxWorkItemsPerTile(dev_id) / maxWGSize;

  int64_t exp2_floor = 1 << log2_floor(stride);
  local_size_col = maxWGSize > exp2_floor ? exp2_floor : maxWGSize;
  local_size_row = maxWGSize / local_size_col;

  loop = newNnz / (local_size_row * max_group) > MaxLoop ? MaxLoop
      : newNnz / (local_size_row * max_group) > 0
      ? newNnz / (local_size_row * max_group)
      : 1;
  global_size_row = CeilDiv(newNnz, local_size_row * loop);
  global_size_col = CeilDiv(stride, local_size_col);
}

// When the group is huge, we can reduce the number of groups by dividing the
// loop.
template <typename scalar_t, typename IndexType>
void adagrad_fused_step_nocoalesced_large_group_kernel_impl(
    Tensor param,
    Tensor state_sum,
    Tensor segment_offsets,
    Tensor value_indices,
    Tensor param_indices,
    Tensor values,
    int64_t nnz,
    int64_t newNnz,
    int64_t stride_grad,
    int64_t stride_param,
    float lr,
    float eps) {
  using accscalar_t = AtenIpexTypeXPU::acc_type<scalar_t>;
  auto& queue = dpcppGetCurrentQueue();
  int64_t global_size_row = 0, global_size_col = 0, local_size_row = 0,
          local_size_col = 0, loop = 1;
  get_wgroup_size_large(
      nnz,
      newNnz,
      stride_grad,
      global_size_row,
      global_size_col,
      local_size_row,
      local_size_col,
      loop);

  auto cgf = DPCPP_Q_CGF(cgh) {
    auto param_data = param.data_ptr<scalar_t>();
    auto state_sum_data = state_sum.data_ptr<scalar_t>();
    auto segment_offsets_data = segment_offsets.data_ptr<int64_t>();
    auto value_indices_data = value_indices.data_ptr<int64_t>();
    auto param_indices_data = param_indices.data_ptr<int64_t>();
    auto values_data = values.data_ptr<scalar_t>();

    auto kfn = DPCPP_Q_KFN(sycl::nd_item<2> item) {
      auto param_ptr = param_data;
      auto state_sum_ptr = state_sum_data;
      auto segment_offsets_ptr = segment_offsets_data;
      auto value_indices_ptr = value_indices_data;
      auto param_indices_ptr = param_indices_data;
      auto values_ptr = values_data;

      uint64_t stride_grad_remainder = item.get_global_id()[1] / local_size_col;
      uint64_t stride_grad_div = item.get_local_id(1);
      IndexType seg_remainder = item.get_global_id()[0] / local_size_row;
      IndexType seg_div = item.get_local_id(0);

      uint64_t featureDim =
          stride_grad_remainder * local_size_col + stride_grad_div;
      for (int l = 0; l < loop; ++l) {
        IndexType seg = seg_remainder * local_size_row * loop +
            l * local_size_row + seg_div;
        const uint64_t newValueRow =
            stride_param * param_indices_ptr[seg] * stride_grad;
        if (seg < newNnz && featureDim < stride_grad) {
          const IndexType begin = segment_offsets_ptr[seg];
          const IndexType end =
              (seg < newNnz - 1) ? segment_offsets_ptr[seg + 1] : nnz;

          accscalar_t tmp = 0, std = 0;
          for (IndexType row = begin; row < end; row++) {
            const uint64_t valueRow =
                ((IndexType)value_indices_ptr[row]) * stride_grad;
            tmp += static_cast<accscalar_t>(values_ptr[valueRow + featureDim]);
          }
          state_sum_data[newValueRow + featureDim] +=
              Numerics<accscalar_t>::pow(
                  static_cast<accscalar_t>(tmp), accscalar_t(2.0));
          std = Numerics<accscalar_t>::sqrt(
                    state_sum_data[newValueRow + featureDim]) +
              static_cast<accscalar_t>(eps);
          param_data[newValueRow + featureDim] +=
              tmp / std * static_cast<accscalar_t>(-lr);
        }
      }
    };

    // kick off kernel
    cgh.parallel_for(
        sycl::nd_range<2>(
            sycl::range<2>(
                global_size_row * local_size_row,
                global_size_col * local_size_col),
            sycl::range<2>(local_size_row, local_size_col)),
        kfn);
  };
  DPCPP_Q_SUBMIT(queue, cgf);
}

static inline bool is_large_group_condition(
    int64_t nnz,
    int64_t newNnz,
    int64_t stride_grad) {
  auto dev_id = dpcppGetDeviceIdOfCurrentQueue();
  int64_t maxWGSize = dpcppMaxWorkGroupSize(dev_id);
  auto max_group = dpcppMaxWorkItemsPerTile(dev_id) / maxWGSize;
  int64_t global_size_row_not_large = 0, global_size_col_not_large = 0,
          local_size_row_not_large = 0, local_size_col_not_large = 0;
  int64_t global_size_row_large = 0, global_size_col_large = 0,
          local_size_row_large = 0, local_size_col_large = 0, loop = 1;
  get_wgroup_size(
      nnz,
      newNnz,
      stride_grad,
      global_size_row_not_large,
      global_size_col_not_large,
      local_size_row_not_large,
      local_size_col_not_large);
  get_wgroup_size_large(
      nnz,
      newNnz,
      stride_grad,
      global_size_row_large,
      global_size_col_large,
      local_size_row_large,
      local_size_col_not_large,
      loop);
  int64_t total_loop_not_large =
      CeilDiv(
          global_size_row_not_large * global_size_col_not_large, max_group) *
      CeilDiv(CeilDiv(nnz, newNnz), global_size_row_not_large);
  int64_t total_loop_large =
      CeilDiv(global_size_row_large * global_size_col_large, max_group) * loop *
      CeilDiv(nnz, newNnz);
  return (total_loop_not_large > total_loop_large);
}

template <typename scalar_t, typename IndexType>
void adagrad_fused_step_nocoalesced_kernel(
    Tensor param,
    Tensor state_sum,
    Tensor segment_offsets,
    Tensor value_indices,
    Tensor param_indices,
    Tensor values,
    int64_t nnz,
    int64_t newNnz,
    int64_t stride_grad,
    int64_t stride_param,
    float lr,
    float eps) {
  bool is_large_group = is_large_group_condition(nnz, newNnz, stride_grad);
  if (is_large_group) {
    adagrad_fused_step_nocoalesced_large_group_kernel_impl<scalar_t, IndexType>(
        param,
        state_sum,
        segment_offsets,
        value_indices,
        param_indices,
        values,
        nnz,
        newNnz,
        stride_grad,
        stride_param,
        lr,
        eps);
  } else {
    adagrad_fused_step_nocoalesced_kernel_impl<scalar_t, IndexType>(
        param,
        state_sum,
        segment_offsets,
        value_indices,
        param_indices,
        values,
        nnz,
        newNnz,
        stride_grad,
        stride_param,
        lr,
        eps);
  }
}

} // namespace impl

std::tuple<at::Tensor, at::Tensor> adagrad_fused_step_kernel_stub(
    at::Tensor& param_,
    at::Tensor& state_sum_,
    const at::Tensor& param2_,
    const at::Tensor& grad_,
    float lr,
    float eps) {
  auto param = param_.contiguous();
  // auto param2 = param2_.contiguous();
  auto state_sum = state_sum_.contiguous();
  int64_t nnz = grad_._nnz();
  int64_t sparse_dim = grad_.sparse_dim();
  if (grad_.is_coalesced() || nnz <= 2) {
    auto grad = grad_.is_coalesced() ? grad_ : grad_.coalesce();
    Tensor indices1D = AtenIpexTypeSparseXPU::flatten_indices(
        grad._indices(), grad.sizes(), true);
    Tensor values = grad._values();
    auto values_size = values.sizes().vec();
    if (std::accumulate(
            values_size.begin(),
            values_size.end(),
            1ll,
            std::multiplies<int64_t>()) > 0) {
      values = values.contiguous();
      int64_t stride_grad =
          xpu::dpcpp::detail::prod_intlist(values.sizes().slice(1));
      int64_t stride_param = xpu::dpcpp::detail::prod_intlist(
          param.sizes().slice(0, sparse_dim - 1));
      IPEX_DISPATCH_FLOATING_TYPES_AND2(
          at::ScalarType::BFloat16,
          at::ScalarType::Half,
          values.scalar_type(),
          "adagrad_fused_step_coalesced",
          [&]() {
            IPEX_DISPATCH_INDEX_TYPES(
                indices1D.scalar_type(), "adagrad_fused_step_coalesced", [&] {
                  impl::adagrad_fused_step_coalesced_kernel_impl<
                      scalar_t,
                      index_t>(
                      param,
                      state_sum,
                      indices1D,
                      values,
                      nnz,
                      stride_grad,
                      stride_param,
                      lr,
                      eps);
                });
          });
    }
  } else {
    Tensor values = grad_._values();
    int64_t newNnz = 0;

    Tensor indices1D = AtenIpexTypeSparseXPU::flatten_indices(
        grad_._indices(), grad_.sizes(), true);

    Tensor origIndices = at::empty({nnz}, grad_._indices().options());
    Tensor uniqueOffsets = at::empty({nnz}, grad_._indices().options());

    auto origIndices_ptr = origIndices.data_ptr<int64_t>();
    auto uniqueOffsets_ptr = uniqueOffsets.data_ptr<int64_t>();

    xpu::pstl::iota<int64_t>(
        origIndices_ptr, origIndices_ptr + nnz, (int64_t)0);
    xpu::pstl::iota<int64_t>(
        uniqueOffsets_ptr, uniqueOffsets_ptr + nnz, (int64_t)0);

    auto indices1D_ptr = indices1D.data_ptr<int64_t>();
    xpu::pstl::sort<int64_t, int64_t>(
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
    auto values_size = values.sizes().vec();
    values_size[0] = newNnz;

    if (std::accumulate(
            values_size.begin(),
            values_size.end(),
            1ll,
            std::multiplies<int64_t>()) > 0) {
      values = values.contiguous();
      int64_t stride_grad =
          xpu::dpcpp::detail::prod_intlist(values.sizes().slice(1));
      int64_t stride_param = xpu::dpcpp::detail::prod_intlist(
          param.sizes().slice(0, sparse_dim - 1));
      IPEX_DISPATCH_FLOATING_TYPES_AND2(
          at::ScalarType::BFloat16,
          at::ScalarType::Half,
          values.scalar_type(),
          "adagrad_fused_step_nocoalesced",
          [&]() {
            IPEX_DISPATCH_INDEX_TYPES(
                indices1D.scalar_type(), "adagrad_fused_step_nocoalesced", [&] {
                  impl::
                      adagrad_fused_step_nocoalesced_kernel<scalar_t, index_t>(
                          param,
                          state_sum,
                          uniqueOffsets,
                          origIndices,
                          indices1D,
                          values,
                          nnz,
                          newNnz,
                          stride_grad,
                          stride_param,
                          lr,
                          eps);
                });
          });
    }
  }

  if (!param_.is_contiguous()) {
    param_.copy_(param);
  }
  if (!state_sum_.is_contiguous()) {
    state_sum_.copy_(state_sum);
  }
  // if (!param2_.is_contiguous()) {
  //  param2_.copy_(param2);
  //}

  return std::make_tuple(param_, state_sum_);
}

std::tuple<at::Tensor, at::Tensor> adagrad_fused_step_with_sparse_grad(
    at::Tensor& param_,
    at::Tensor& state_sum_,
    const at::Tensor& param2_,
    const at::Tensor& grad_,
    double lr,
    double eps) {
  RECORD_FUNCTION(
      "torch_ipex::adagrad_fused_step_with_sparse_grad",
      c10::ArrayRef<c10::IValue>({}));

  TORCH_CHECK(eps >= 0, "Expect eps >= 0.0, got ", eps);

  TORCH_CHECK(
      param_.sizes() == grad_.sizes(),
      "Expect param and grad_ have the same sizes, param sizes: ",
      param_.sizes(),
      "; grad_ sizes: ",
      grad_.sizes());
  TORCH_CHECK(
      param_.sizes() == state_sum_.sizes(),
      "Expect param and state_sum have the same sizes, param sizes: ",
      param_.sizes(),
      "; state_sum sizes: ",
      state_sum_.sizes());
  // TORCH_CHECK(
  //    param2_.numel() == 0 || param_.sizes() == param2_.sizes(),
  //    "Expect param and param2_ have the same sizes, param sizes: ",
  //    param_.sizes(),
  //    "; param2_ sizes: ",
  //    param2_.sizes());

  TORCH_CHECK(grad_.is_sparse(), "Expect grad_ is a sparse tensor.");
  TORCH_CHECK(
      !(grad_.scalar_type() == at::ScalarType::BFloat16),
      "Not supports grad_ is a tensor.");

  return adagrad_fused_step_kernel_stub(
      param_, state_sum_, param2_, grad_, lr, eps);
}

} // namespace AtenIpexTypeXPU
} // namespace at

namespace {
IPEX_LIBRARY_FRAGMENT() {
  IPEX_OP_REGISTER_DISPATCH(
      "adagrad_fused_step_with_sparse_grad",
      at::AtenIpexTypeXPU::adagrad_fused_step_with_sparse_grad,
      c10::DispatchKey::SparseXPU);
}
} // namespace
