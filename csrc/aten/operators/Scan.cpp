#include <ATen/ATen.h>

#include <core/Memory.h>
#include <runtime/Utils.h>
#include <utils/DPCPP.h>
#include <utils/Helpers.h>
#include "comm/ATDispatch.h"
#include "comm/MathReduce.h"
#include "comm/Numerics.h"
#include "comm/PSTLFunctions.h"
#include "comm/RegistrationDeclarations.h"

using namespace at::detail;
using namespace xpu::dpcpp;

namespace at {
namespace AtenIpexTypeXPU {

typedef enum {
  UNDEF = 0,
  SCAN_BIN_ADD = 1,
  SCAN_BIN_PROD = 2,
} ScanBinOpType;

namespace impl {

template <int scan_type, class InputIt, class OutputIt, class T, class BinaryOp>
DPCPP_DEVICE static inline OutputIt _scan_kernel(
    InputIt first,
    InputIt last,
    OutputIt d_first,
    const int64_t scan_dim,
    const int64_t scan_dim_size,
    const int64_t stride,
    const int64_t batch,
    T init,
    ScanBinOpType scanBinOpType,
    BinaryOp binary_op) {
  T default_value = 0;
  if (SCAN_BIN_ADD == scanBinOpType) {
    default_value = 0;
  }
  if (SCAN_BIN_PROD == scanBinOpType) {
    default_value = 1;
  }
  const auto N = std::distance(first, last);

  auto& dpcpp_queue = dpcppGetCurrentQueue();
  const auto dev_id = dpcppGetDeviceIdOfCurrentQueue();
  auto wgroup_size = dpcppMaxWorkGroupSize(dev_id);

  // Treat NDRange as 3D cuboid
  const auto column = (1 == stride) ? scan_dim_size : stride;
  const auto row = (1 == stride) ? stride : scan_dim_size;
  const auto depth = batch;

  // For innermost scenario, treat workgroup as 1D arch to get largest coverage
  // for work group For outer scenario, treat workgroup as 2D arch to achieve
  // memory coalescing access
  const auto wgroup_size_col = (1 == stride) ? wgroup_size : 32;
  const auto wgroup_size_row =
      (1 == stride) ? 1 : (wgroup_size / wgroup_size_col);
  const auto wgroup_size_depth = 1; // It's always 1
  // re-calculate workgroup size
  wgroup_size = wgroup_size_col * wgroup_size_row * wgroup_size_depth;

  // calculate workgroup num in 3 directions
  const auto wgroup_num_col = (column + wgroup_size_col - 1) / wgroup_size_col;
  const auto wgroup_num_row = (row + wgroup_size_row - 1) / wgroup_size_row;
  const auto wgroup_num_depth = depth;
  // overall workgroup num
  const auto wgroup_num = wgroup_num_col * wgroup_num_row * wgroup_num_depth;

  auto options = map_options<T>();

  // ------ convergence sceanrio: ------
  // (stride == 1) column <= wgroup_size_col
  // (stride > 1) row <= wgroup_size_row
  if ((1 == stride && column <= wgroup_size_col) ||
      (stride > 1 && row <= wgroup_size_row)) {
    auto cur_column =
        (1 == stride) ? column : (wgroup_num_col * wgroup_size_col);
    auto cur_row =
        (1 == stride) ? row : (1 /*wgroup_num_row*/ * wgroup_size_row);
    auto cur_depth =
        (1 == stride) ? depth : (wgroup_num_depth * wgroup_size_depth);

    auto cur_wgroup_size_col = (1 == stride) ? column : wgroup_size_col;
    auto cur_wgroup_size_row =
        (1 == stride) ? wgroup_size_row : wgroup_size_row;
    auto cur_wgroup_size_depth = 1;

    // Kogge-Stone addr algorithm;
    auto cgf = DPCPP_Q_CGF(__cgh) {
      auto slm_size = (1 == stride) ? cur_column * cur_wgroup_size_row
                                    : cur_row * cur_wgroup_size_col;
      dpcpp_local_acc_t<T> local_scan(slm_size, __cgh);

      auto kfn = DPCPP_Q_KFN(DPCPP::nd_item<3> item_id) {
        // wgroup shape is (cur_wgroup_size_row, cur_wgroup_size_col,
        // cur_wgroup_size_depth)
        auto local_id_col = item_id.get_local_id(0);
        auto local_id_row = item_id.get_local_id(1);
        auto local_id_depth = item_id.get_local_id(2);

        auto global_id_col = item_id.get_global_id(0);
        auto global_id_row = item_id.get_global_id(1);
        auto global_id_depth = item_id.get_global_id(2);

        auto group_id_col = item_id.get_group(0);
        auto group_id_row = item_id.get_group(1);
        auto group_id_depth = item_id.get_group(2);

        auto global_mem_offset = global_id_col + global_id_row * column +
            global_id_depth * column * row;

        // local_id_depth is always 0, so don't need it below
        // for scenario where stride > 1, need transpose move from global memory
        // to SLM
        auto slm_offset = (1 == stride)
            ? (local_id_col + local_id_row * cur_wgroup_size_col)
            : (local_id_row + local_id_col * cur_wgroup_size_row);
        // initialize local_input [[global memeory ------> SLM]]
        auto cur_init = init;

        auto local_id_orth = (1 == stride) ? local_id_col : local_id_row;
        if (global_id_col < column && global_id_row < row &&
            global_id_depth < depth) {
          if (scan_type == 1) {
            local_scan[slm_offset] = first[global_mem_offset];
          } else {
            if (local_id_orth > 0)
              local_scan[slm_offset] = first[global_mem_offset - stride];
            else
              local_scan[slm_offset] = default_value;
          }
          if (local_id_orth == 0)
            local_scan[slm_offset] =
                binary_op(local_scan[slm_offset], cur_init);
        }

        group_barrier(item_id.get_group());

        // body of KS algo    [[calculation on SLM]]
        // auto ks_dim_size = (1 == stride) ? column : row;
        auto ks_dim_size = (1 == stride) ? wgroup_size_col : wgroup_size_row;
        auto ks_id = (1 == stride) ? local_id_col : local_id_row;
        for (auto __k = 1; __k < ks_dim_size; __k <<= 1) {
          auto tmp =
              (ks_id >= __k) ? local_scan[slm_offset - __k] : default_value;
          group_barrier(item_id.get_group());
          local_scan[slm_offset] = binary_op(local_scan[slm_offset], tmp);
          group_barrier(item_id.get_group());
        }

        // flush result into dst [[SLM -------> global memory]]
        if (global_id_col < column && global_id_row < row &&
            global_id_depth < depth) {
          d_first[global_mem_offset] = local_scan[slm_offset];
        }
      };
      __cgh.parallel_for(
          DPCPP::nd_range</*dim=*/3>(
              DPCPP::range<3>(cur_column, cur_row, cur_depth),
              DPCPP::range<3>(
                  cur_wgroup_size_col,
                  cur_wgroup_size_row,
                  cur_wgroup_size_depth)),
          kfn);
    };
    DPCPP_Q_SUBMIT(dpcpp_queue, cgf);

    return d_first + N;
  }

  Tensor carry = (1 == stride)
      ? at::empty(
            {wgroup_num_col, wgroup_num_row, wgroup_num_depth, wgroup_size_row},
            options)
      : at::empty({wgroup_num_row, column, wgroup_num_depth}, options);

  T* carry_ptr = carry.data_ptr<T>();

  //========== N > wgroup_size ============
  // 1. do exclusive_scan on each workgroups
  auto cgf_1 = DPCPP_Q_CGF(__cgh) {
    dpcpp_local_acc_t<T> local_scan(wgroup_size, __cgh);

    auto kfn = DPCPP_Q_KFN(DPCPP::nd_item<3> item_id) {
      // wgroup shape is (wgroup_size_row, wgroup_size_col, wgroup_size_depth)
      auto local_id_col = item_id.get_local_id(0);
      auto local_id_row = item_id.get_local_id(1);
      auto local_id_depth = item_id.get_local_id(2);

      auto global_id_col = item_id.get_global_id(0);
      auto global_id_row = item_id.get_global_id(1);
      auto global_id_depth = item_id.get_global_id(2);

      auto group_id_col = item_id.get_group(0);
      auto group_id_row = item_id.get_group(1);
      auto group_id_depth = item_id.get_group(2);

      auto global_mem_offset = global_id_col + global_id_row * column +
          global_id_depth * column * row;

      // local_id_depth is always 0, so don't need it below
      auto slm_offset = (1 == stride)
          ? (local_id_col + local_id_row * wgroup_size_col)
          : (local_id_row + local_id_col * wgroup_size_row);

      auto carry_offset = (1 == stride)
          ? (group_id_col + group_id_row * wgroup_num_col +
             group_id_depth * wgroup_num_col * wgroup_num_row + local_id_row)
          : (group_id_col * wgroup_size_col + group_id_row * column +
             group_id_depth * column * wgroup_num_row + local_id_col);

      // initialize local_input [[global memory ------> SLM]]
      auto cur_init = default_value;
      if (1 == stride) {
        // innermost scenarios
        // 2D tensor(scan_dim=1)
        // 3D tensor(scan_dim=2)
        cur_init = (0 == group_id_col) ? init : default_value;
      } else { // stride > 1, need perform transpose copy
        // outer scenario
        // 2D tensor(scan_dim=0) and 3D tensor(scan_dim=0) scenarios
        // 3D tensor(scan_dim=1) scenario
        cur_init = (0 == group_id_row) ? init : default_value;
      }

      // local_id orthogonal direction
      auto local_id_orth = (1 == stride) ? local_id_col : local_id_row;
      // wgroup_size orthogonal direction
      auto wgroup_size_orth = (1 == stride) ? wgroup_size_col : wgroup_size_row;
      if (global_id_col < column && global_id_row < row &&
          global_id_depth < depth) {
        if (scan_type == 1) { // inclusive
          local_scan[slm_offset] = first[global_mem_offset];
        } else { // scan_type == 0 // exclusive
          if (local_id_orth > 0)
            local_scan[slm_offset] = first[global_mem_offset - stride];
          else
            local_scan[slm_offset] = default_value;
        }

        if (local_id_orth == 0)
          local_scan[slm_offset] = binary_op(local_scan[slm_offset], cur_init);

        if (local_id_orth == wgroup_size_orth - 1) {
          carry_ptr[carry_offset] = first[global_mem_offset];
        }
      }
      group_barrier(item_id.get_group());

      // body of KS algo [[calculation on SLM]]
      for (auto __k = 1; __k < wgroup_size_orth; __k <<= 1) {
        auto tmp = (local_id_orth >= __k) ? local_scan[slm_offset - __k]
                                          : default_value;
        group_barrier(item_id.get_group());
        local_scan[slm_offset] = binary_op(local_scan[slm_offset], tmp);
        group_barrier(item_id.get_group());
      }

      // flush result into dst [[SLM ------> global memory]]
      if (global_id_col < column && global_id_row < row &&
          global_id_depth < depth) {
        d_first[global_mem_offset] = local_scan[slm_offset];
        if (local_id_orth == wgroup_size_orth - 1) {
          if (scan_type == 1)
            carry_ptr[carry_offset] = local_scan[slm_offset];
          else
            carry_ptr[carry_offset] =
                binary_op(carry_ptr[carry_offset], local_scan[slm_offset]);
        }
      }
    };

    __cgh.parallel_for(
        DPCPP::nd_range</*dim=*/3>(
            DPCPP::range<3>(
                wgroup_num_col * wgroup_size_col,
                wgroup_num_row * wgroup_size_row,
                wgroup_num_depth * wgroup_size_depth),
            DPCPP::range<3>(
                wgroup_size_col, wgroup_size_row, wgroup_size_depth)),
        kfn);
  };
  DPCPP_Q_SUBMIT(dpcpp_queue, cgf_1);

  // 2. recursion for carry
  const auto M = (1 == stride) ? wgroup_num * wgroup_size_row
                               : wgroup_num_row * column * wgroup_num_depth;
  const auto carry_scan_dim = scan_dim;
  const auto carry_scan_dim_size = (1 == stride)
      ? (scan_dim_size + wgroup_size_col - 1) / wgroup_size_col
      : (scan_dim_size + wgroup_size_row - 1) / wgroup_size_row;
  const auto carry_stride = stride;
  const auto carry_batch = batch;

  _scan_kernel<0>(
      carry_ptr,
      carry_ptr + M,
      carry_ptr,
      carry_scan_dim,
      carry_scan_dim_size,
      carry_stride,
      carry_batch,
      default_value,
      scanBinOpType,
      binary_op);

  // 3. reduce among all work groups and flush data to dst
  auto cgf_3 = DPCPP_Q_CGF(__cgh) {
    auto local_carry_size = (1 == stride) ? wgroup_size_row : wgroup_size_col;
    dpcpp_local_acc_t<T> local_carry(local_carry_size, __cgh);

    auto kfn = DPCPP_Q_KFN(DPCPP::nd_item<3> item_id) {
      auto local_id_col = item_id.get_local_id(0);
      auto local_id_row = item_id.get_local_id(1);
      auto local_id_depth = item_id.get_local_id(2);

      auto global_id_col = item_id.get_global_id(0);
      auto global_id_row = item_id.get_global_id(1);
      auto global_id_depth = item_id.get_global_id(2);

      auto group_id_col = item_id.get_group(0);
      auto group_id_row = item_id.get_group(1);
      auto group_id_depth = item_id.get_group(2);

      auto global_mem_offset = global_id_col + global_id_row * column +
          global_id_depth * column * row;

      auto carry_offset = (1 == stride)
          ? (group_id_col + group_id_row * wgroup_num_col +
             group_id_depth * wgroup_num_col * wgroup_num_row + local_id_row)
          : (group_id_col * wgroup_size_col + group_id_row * column +
             group_id_depth * column * wgroup_num_row + local_id_col);

      // [[global memory ---> SLM]]
      if (1 == stride) {
        if (local_id_col == 0)
          local_carry[local_id_row] = carry_ptr[carry_offset];
      } else { // stride > 1
        // transpose
        if (local_id_row == 0)
          local_carry[local_id_col] = carry_ptr[carry_offset];
      }
      group_barrier(item_id.get_group());

      // [[add from SLM to global memory]]
      if (global_id_col < column && global_id_row < row &&
          global_id_depth < depth) {
        d_first[global_mem_offset] = binary_op(
            d_first[global_mem_offset],
            local_carry[(1 == stride) ? local_id_row : local_id_col]);
      }
    };
    __cgh.parallel_for(
        DPCPP::nd_range</*dim=*/3>(
            DPCPP::range<3>(
                wgroup_num_col * wgroup_size_col,
                wgroup_num_row * wgroup_size_row,
                wgroup_num_depth * wgroup_size_depth /* 1 */),
            DPCPP::range<3>(
                wgroup_size_col, wgroup_size_row, wgroup_size_depth /* 1 */)),
        kfn);
  };
  DPCPP_Q_SUBMIT(dpcpp_queue, cgf_3);

  return d_first + N;
}

template <typename T, class InputIt, class OutputIt, class BinaryOp>
DPCPP_DEVICE static inline OutputIt _inclusive_scan(
    InputIt first,
    InputIt last,
    OutputIt d_first,
    const int64_t scan_dim,
    const int64_t scan_dim_size,
    const int64_t stride,
    const int64_t batch,
    T init,
    ScanBinOpType scanBinOpType,
    BinaryOp binary_op) {
  RECORD_FUNCTION("_inclusive_scan_xpu", {});
  return _scan_kernel<1>(
      first,
      last,
      d_first,
      scan_dim,
      scan_dim_size,
      stride,
      batch,
      init,
      scanBinOpType,
      binary_op);
}

template <typename scalar_t, class BinaryOp>
void scanDimCore(
    Tensor& tgt,
    Tensor& src,
    int dimension,
    scalar_t init,
    ScanBinOpType scanBinOpType,
    BinaryOp binary_op) {
  auto& dpcpp_queue = dpcppGetCurrentQueue();
  const auto dev_id = dpcppGetDeviceIdOfCurrentQueue();
  const int wgroup_size = dpcppMaxWorkGroupSize(dev_id);

  auto totalElements = tgt.numel();
  int64_t scan_dim_size = src.size(dimension);
  int64_t stride = src.stride(dimension);
  int64_t batch = totalElements / (scan_dim_size * stride);

  auto src_loc_begin = src.data_ptr<scalar_t>();
  auto tgt_loc_begin = tgt.data_ptr<scalar_t>();
  _inclusive_scan<scalar_t>(
      src_loc_begin,
      src_loc_begin + totalElements,
      tgt_loc_begin,
      dimension,
      scan_dim_size,
      stride,
      batch,
      init,
      scanBinOpType,
      binary_op);
}

template <typename scalar_t, class BinaryFunction>
typename std::enable_if<!IS_HALF(scalar_t), void>::type scanDim(
    Tensor& self_,
    const Tensor& src_,
    int dimension,
    scalar_t init,
    ScanBinOpType scanBinOpType,
    BinaryFunction binary_op) {
  int ndim = src_.dim() == 0 ? 1 : src_.dim();
  dimension = maybe_wrap_dim(dimension, src_.dim());
  TORCH_CHECK(
      dimension >= 0 && dimension < ndim,
      "dimension ",
      dimension,
      " out of range");

  self_.resize_as_(src_);
  auto self = self_.contiguous();
  auto src = src_.contiguous();

  scanDimCore<scalar_t>(self, src, dimension, init, scanBinOpType, binary_op);

  self_.copy_(self);
}

template <typename scalar_t, class BinaryFunction>
typename std::enable_if<IS_HALF(scalar_t), void>::type scanDim(
    Tensor& self_,
    const Tensor& src_,
    int dimension,
    scalar_t init,
    ScanBinOpType scanBinOpType,
    BinaryFunction binary_op) {
  int ndim = src_.dim() == 0 ? 1 : src_.dim();
  dimension = maybe_wrap_dim(dimension, src_.dim());
  TORCH_CHECK(
      dimension >= 0 && dimension < ndim,
      "dimension ",
      dimension,
      " out of range");

  self_.resize_as_(src_);
  auto self = self_.contiguous();
  auto src = src_.contiguous();

  scanDimCore<scalar_t>(self, src, dimension, init, scanBinOpType, binary_op);

  self_.copy_(self);
}

} // namespace impl

Tensor& cumsum_out(
    const Tensor& self,
    int64_t dim,
    c10::optional<at::ScalarType> dtype,
    Tensor& out) {
  IPEX_DISPATCH_ALL_TYPES_AND(
      at::ScalarType::Half, self.scalar_type(), "cumsum", [&]() {
        impl::scanDim<scalar_t>(
            out,
            self,
            dim,
            ScalarConvert<float, scalar_t>::to(0.0),
            SCAN_BIN_ADD,
            AddOp<scalar_t>());
      });
  return out;
}

Tensor& cumprod_out(
    const Tensor& self,
    int64_t dim,
    c10::optional<at::ScalarType> dtype,
    Tensor& out) {
  IPEX_DISPATCH_ALL_TYPES_AND(
      at::ScalarType::Half, self.scalar_type(), "cumprod", [&]() {
        impl::scanDim<scalar_t>(
            out,
            self,
            dim,
            ScalarConvert<float, scalar_t>::to(1.0),
            SCAN_BIN_PROD,
            MulOp<scalar_t>());
      });
  return out;
}

} // namespace AtenIpexTypeXPU
} // namespace at
