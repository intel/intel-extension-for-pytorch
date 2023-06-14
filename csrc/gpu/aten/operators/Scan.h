#pragma once
#include <core/detail/TensorInfo.h>
#include <utils/DPCPP.h>
#include "BatchKernel.h"
#include "comm/Numerics.h"
#include "comm/TensorOptions.h"

#include <runtime/Utils.h>
#include <utils/oneMKLUtils.h>
#include "comm/ATDispatch.h"
#include "comm/AccumulateType.h"
#include "comm/Math.h"
using namespace xpu::dpcpp::detail;

namespace at {
namespace AtenIpexTypeXPU {
typedef enum {
  EXCLUSIVE_TYPE = 0,
  INCLUSIVE_TYPE = 1,
} ScanType;

template <typename scalar_t, typename idx_t, typename BinaryOperation>
void binary_op_update(
    const scalar_t lhs,
    scalar_t& rhs,
    const idx_t lhs_idx,
    idx_t& rhs_idx,
    BinaryOperation binary_op) {
  if (!at::_isnan(rhs) && (at::_isnan(lhs) || !binary_op(rhs, lhs))) {
    rhs = lhs;
    rhs_idx = lhs_idx;
  }
}

// group x scan by using up down sweep algorithm(call uds for short)
template <
    class LSConfig,
    class T,
    class BinaryFunction,
    bool TrivialOffCal = false>
T inline group_x_scan_by_uds_for_loop_scan(
    sycl::nd_item<2> item,
    const T pre_max_carr,
    int64_t base_off_batch,
    int64_t base_off_problem,
    dpcpp_local_ptr<T> slm,
    LSConfig cfg) {
  using InputInfo = typename LSConfig::InputInfoType;
  using OutputInfo = typename LSConfig::OutputInfoType;

  int64_t glb_ldr_off_0, glb_ldr_off_1, glb_str_off_0, glb_str_off_1,
      glb_ldr_logical_off_0, glb_ldr_logical_off_1, glb_str_logical_off_0,
      glb_str_logical_off_1;

  const auto sub_group = item.get_sub_group();
  const auto sub_group_size = sub_group.get_local_range()[0];

  typename LSConfig::item_desc id = cfg.get_item_desc(item);

  const auto lix = item.get_local_id(1);
  const auto liy = item.get_local_id(0);
  const auto rx = item.get_local_range(1);
  const auto ry = item.get_local_range(0);

  uint32_t ix0 = base_off_problem + lix;
  uint32_t ix1 = base_off_problem + rx + lix;
  uint32_t glb0 = base_off_batch * cfg.problem_ + ix0;
  uint32_t glb1 = base_off_batch * cfg.problem_ + ix1;
  if constexpr (TrivialOffCal) {
    glb_ldr_off_0 = glb0;
    glb_ldr_off_1 = glb1;
    glb_str_off_0 = glb0;
    glb_str_off_1 = glb1;
  } else {
    glb_ldr_logical_off_0 = glb0;
    glb_ldr_off_0 = IndexToOffset<typename InputInfo::scalar_t, int64_t>::get(
        glb_ldr_logical_off_0,
        cfg.input_,
        IndexToOffset<typename InputInfo::scalar_t, int64_t>::
            NON_STRICT_CONTIGUOUS);

    glb_ldr_logical_off_1 = glb1;
    glb_ldr_off_1 = IndexToOffset<typename InputInfo::scalar_t, int64_t>::get(
        glb_ldr_logical_off_1,
        cfg.input_,
        IndexToOffset<typename InputInfo::scalar_t, int64_t>::
            NON_STRICT_CONTIGUOUS);

    glb_str_logical_off_0 = glb0;
    glb_str_off_0 = IndexToOffset<typename OutputInfo::scalar_t, int64_t>::get(
        glb_str_logical_off_0,
        cfg.output_,
        IndexToOffset<typename OutputInfo::scalar_t, int64_t>::
            NON_STRICT_CONTIGUOUS);

    glb_str_logical_off_1 = glb1;
    glb_str_off_1 = IndexToOffset<typename OutputInfo::scalar_t, int64_t>::get(
        glb_str_logical_off_1,
        cfg.output_,
        IndexToOffset<typename OutputInfo::scalar_t, int64_t>::
            NON_STRICT_CONTIGUOUS);
  }
  // TODO: opti for bank conflict elemination
  // Read data from global memory to shared local memory
  // Each work item load 2 elements from global device memory to shared local
  // memory
  if (base_off_batch < cfg.batch_) {
    if (ix0 < cfg.problem_) {
      slm[liy * rx * 2 + lix] = cfg.input_.data[glb_ldr_off_0];
    } else {
      slm[liy * rx * 2 + lix] = cfg.init_;
    }

    if (ix1 < cfg.problem_) {
      slm[liy * rx * 2 + rx + lix] = cfg.input_.data[glb_ldr_off_1];
    } else {
      slm[liy * rx * 2 + rx + lix] = cfg.init_;
    }

    // Add the total value of all previous work groups to the first value of
    // this work group.
    if (0 == lix) {
      slm[liy * rx * 2 + lix] =
          cfg.func_(slm[liy * rx * 2 + lix], pre_max_carr);
    }
  }
  item.barrier(dpcpp_local_fence);

  // Parallel reduction (Up-sweep)
  for (uint32_t s = rx, d = 1; s >= 1; s >>= 1, d <<= 1) {
    if (base_off_batch < cfg.batch_ && lix < s) {
      uint32_t offset = liy * rx * 2 + (2 * lix + 1) * d - 1;
      slm[offset + d] = cfg.func_(slm[offset], slm[offset + d]);
    }
    if (sub_group_size != cfg.wg_range_x_) {
      item.barrier(dpcpp_local_fence);
    }
  }

  // Down-sweep
  for (uint32_t s = 2, d = rx / 2; d >= 1; s <<= 1, d >>= 1) {
    if (base_off_batch < cfg.batch_ && lix < s - 1) {
      uint32_t offset = liy * rx * 2 + 2 * (lix + 1) * d - 1;
      slm[offset + d] = cfg.func_(slm[offset], slm[offset + d]);
    }
    if (sub_group_size != cfg.wg_range_x_) {
      item.barrier(dpcpp_local_fence);
    }
  }

  // Write back from shared local memory to global memory
  if (base_off_batch < cfg.batch_) {
    if (ix0 < cfg.problem_) {
      cfg.output_.data[glb_str_off_0] = slm[liy * rx * 2 + lix];
    }
    if (ix1 < cfg.problem_) {
      cfg.output_.data[glb_str_off_1] = slm[liy * rx * 2 + rx + lix];
    }
  }

  // each work item would return current max carr
  return slm[liy * rx * 2 + 2 * rx - 1];
}

// group x scan by using up down sweep algorithm(call uds for short)
template <
    class LSConfig,
    class T,
    class BinaryFunction,
    bool TrivialOffCal = false>
void inline group_x_scan_by_uds_for_loop_scan_with_indices(
    sycl::nd_item<2> item,
    T& pre_max_carr,
    int64_t& pre_idx_carr,
    int64_t base_off_batch,
    int64_t base_off_problem,
    dpcpp_local_ptr<T> slm,
    dpcpp_local_ptr<int64_t> slm_idx,
    LSConfig cfg) {
  using InputInfo = typename LSConfig::InputInfoType;
  using OutputInfo = typename LSConfig::OutputInfoType;
  using IndicesInfo = typename LSConfig::IndicesInfoType;

  int64_t glb_ldr_off_0, glb_ldr_off_1, glb_str_off_0, glb_str_off_1,
      glb_ldr_logical_off_0, glb_ldr_logical_off_1, glb_str_logical_off_0,
      glb_str_logical_off_1, glb_idx_off_0, glb_idx_off_1;

  const auto sub_group = item.get_sub_group();
  const auto sub_group_size = sub_group.get_local_range()[0];

  typename LSConfig::item_desc id = cfg.get_item_desc(item);

  const auto lix = item.get_local_id(1);
  const auto liy = item.get_local_id(0);
  const auto rx = item.get_local_range(1);
  const auto ry = item.get_local_range(0);

  uint32_t ix0 = base_off_problem + lix;
  uint32_t ix1 = base_off_problem + rx + lix;
  uint32_t glb0 = base_off_batch * cfg.problem_ + ix0;
  uint32_t glb1 = base_off_batch * cfg.problem_ + ix1;
  if constexpr (TrivialOffCal) {
    glb_ldr_off_0 = glb0;
    glb_ldr_off_1 = glb1;
    glb_str_off_0 = glb0;
    glb_str_off_1 = glb1;
    glb_idx_off_0 = glb0;
    glb_idx_off_1 = glb1;
  } else {
    glb_ldr_logical_off_0 = glb0;
    glb_ldr_off_0 = IndexToOffset<typename InputInfo::scalar_t, int64_t>::get(
        glb_ldr_logical_off_0,
        cfg.input_,
        IndexToOffset<typename InputInfo::scalar_t, int64_t>::
            NON_STRICT_CONTIGUOUS);

    glb_ldr_logical_off_1 = glb1;
    glb_ldr_off_1 = IndexToOffset<typename InputInfo::scalar_t, int64_t>::get(
        glb_ldr_logical_off_1,
        cfg.input_,
        IndexToOffset<typename InputInfo::scalar_t, int64_t>::
            NON_STRICT_CONTIGUOUS);

    glb_str_logical_off_0 = glb0;
    glb_str_off_0 = IndexToOffset<typename OutputInfo::scalar_t, int64_t>::get(
        glb_str_logical_off_0,
        cfg.output_,
        IndexToOffset<typename OutputInfo::scalar_t, int64_t>::
            NON_STRICT_CONTIGUOUS);

    glb_str_logical_off_1 = glb1;
    glb_str_off_1 = IndexToOffset<typename OutputInfo::scalar_t, int64_t>::get(
        glb_str_logical_off_1,
        cfg.output_,
        IndexToOffset<typename OutputInfo::scalar_t, int64_t>::
            NON_STRICT_CONTIGUOUS);

    glb_idx_off_0 = IndexToOffset<typename IndicesInfo::scalar_t, int64_t>::get(
        glb0,
        cfg.indices_,
        IndexToOffset<typename IndicesInfo::scalar_t, int64_t>::
            NON_STRICT_CONTIGUOUS);

    glb_idx_off_1 = IndexToOffset<typename IndicesInfo::scalar_t, int64_t>::get(
        glb1,
        cfg.indices_,
        IndexToOffset<typename IndicesInfo::scalar_t, int64_t>::
            NON_STRICT_CONTIGUOUS);
  }
  // TODO: opti for bank conflict elemination
  // Read data from global memory to shared local memory
  // Each work item load 2 elements from global device memory to shared local
  // memory
  if (base_off_batch < cfg.batch_) {
    if (ix0 < cfg.problem_) {
      slm[liy * rx * 2 + lix] = cfg.input_.data[glb_ldr_off_0];
      slm_idx[liy * rx * 2 + lix] = ix0;
    } else {
      slm[liy * rx * 2 + lix] = cfg.init_;
    }
    if (ix1 < cfg.problem_) {
      slm[liy * rx * 2 + rx + lix] = cfg.input_.data[glb_ldr_off_1];
      slm_idx[liy * rx * 2 + rx + lix] = ix1;
    } else {
      slm[liy * rx * 2 + rx + lix] = cfg.init_;
    }

    // Add the total value of all previous work groups to the first value of
    // this work group.
    if (0 == lix) {
      auto offset = liy * rx * 2 + lix;
      binary_op_update(
          pre_max_carr, slm[offset], pre_idx_carr, slm_idx[offset], cfg.func_);
    }
  }
  item.barrier(dpcpp_local_fence);

  // Parallel reduction (Up-sweep)
  for (uint32_t s = rx, d = 1; s >= 1; s >>= 1, d <<= 1) {
    if (base_off_batch < cfg.batch_ && lix < s) {
      uint32_t offset = liy * rx * 2 + (2 * lix + 1) * d - 1;
      binary_op_update(
          slm[offset],
          slm[offset + d],
          slm_idx[offset],
          slm_idx[offset + d],
          cfg.func_);
    }
    if (sub_group_size != cfg.wg_range_x_) {
      item.barrier(dpcpp_local_fence);
    }
  }

  // Down-sweep
  for (uint32_t s = 2, d = rx / 2; d >= 1; s <<= 1, d >>= 1) {
    if (base_off_batch < cfg.batch_ && lix < s - 1) {
      uint32_t offset = liy * rx * 2 + 2 * (lix + 1) * d - 1;
      binary_op_update(
          slm[offset],
          slm[offset + d],
          slm_idx[offset],
          slm_idx[offset + d],
          cfg.func_);
    }
    if (sub_group_size != cfg.wg_range_x_) {
      item.barrier(dpcpp_local_fence);
    }
  }

  // Write back from shared local memory to global memory
  if (base_off_batch < cfg.batch_) {
    if (ix0 < cfg.problem_) {
      cfg.output_.data[glb_str_off_0] = slm[liy * rx * 2 + lix];
      cfg.indices_.data[glb_idx_off_0] = slm_idx[liy * rx * 2 + lix];
    }
    if (ix1 < cfg.problem_) {
      cfg.output_.data[glb_str_off_1] = slm[liy * rx * 2 + rx + lix];
      cfg.indices_.data[glb_idx_off_1] = slm_idx[liy * rx * 2 + rx + lix];
    }
  }

  pre_max_carr = slm[liy * rx * 2 + 2 * rx - 1];
  pre_idx_carr = slm_idx[liy * rx * 2 + 2 * rx - 1];
}

template <class T, class BinaryFunction>
T subgroup_scan(sycl::nd_item<2> item, T value, BinaryFunction func) {
  const auto sg = item.get_sub_group();
  const auto lane = sg.get_local_linear_id();
  const auto sg_size = sg.get_local_range()[0];

  T tmp = 0;
  for (int offset = 1; offset < sg_size; offset <<= 1) {
    tmp = sg.shuffle_up(value, offset);
    if (lane >= offset)
      value = func(value, tmp);
  }

  return value;
}

template <class T, class BinaryFunction>
void subgroup_scan_with_indices(
    sycl::nd_item<2> item,
    T& value,
    int64_t& idx,
    BinaryFunction func) {
  const auto sg = item.get_sub_group();
  const auto lane = sg.get_local_linear_id();
  const auto sg_size = sg.get_local_range()[0];

  T tmp = 0;
  int64_t tmp_idx = idx;
  for (int offset = 1; offset < sg_size; offset <<= 1) {
    tmp = sg.shuffle_up(value, offset);
    tmp_idx = sg.shuffle_up(idx, offset);
    if (lane >= offset) {
      binary_op_update(tmp, value, tmp_idx, idx, func);
    }
  }
}

template <class T, class BinaryFunction>
T group_x_scan(
    sycl::nd_item<2> item,
    T value,
#ifndef SG_SCAN
    dpcpp_local_ptr<T> slm,
#else
    dpcpp_local_ptr<T> carrier,
#endif
    T init,
    BinaryFunction func) {
#ifndef SG_SCAN
  const auto lix = item.get_local_id(1);
  const auto liy = item.get_local_id(0);
  const auto rx = item.get_local_range(1);
  const auto ry = item.get_local_range(0);

  slm[liy * rx + lix] = value;
  for (int offset = 1; offset < rx; offset <<= 1) {
    item.barrier(dpcpp_local_fence);
    if (lix >= offset)
      value = func(slm[liy * rx + (lix - offset)], slm[liy * rx + lix]);
    item.barrier(dpcpp_local_fence);

    if (lix >= offset)
      slm[liy * rx + lix] = value;
  }

#else
  const auto wg_size = item.get_local_range(1);
  const auto sg = item.get_sub_group();
  const auto sgi = sg.get_group_linear_id();
  const auto sg_size = sg.get_local_range()[0];
  const auto lane = sg.get_local_linear_id();
  const auto sg_num = (wg_size + sg_size - 1) / sg_size;

  value = subgroup_scan<T, BinaryFunction>(item, value, func);
  if (lane == sg_size - 1) {
    carrier[sgi] = value;
  }

  item.barrier(dpcpp_local_fence);
  for (int i = 0; i < sg_num; i++) {
    if (sgi > i) {
      value = func(value, carrier[i]);
    }
  }
#endif

  return value;
}

template <class T, class BinaryFunction>
void group_x_scan_with_indices(
    sycl::nd_item<2> item,
    T& value,
    int64_t& idx,
#ifndef SG_SCAN
    dpcpp_local_ptr<T> slm,
    dpcpp_local_ptr<int64_t> slm_idx,
#else
    dpcpp_local_ptr<T> carrier,
    dpcpp_local_ptr<int64_t> carrier_idx,
#endif
    T init,
    BinaryFunction func) {
#ifndef SG_SCAN
  const auto lix = item.get_local_id(1);
  const auto liy = item.get_local_id(0);
  const auto rx = item.get_local_range(1);
  const auto ry = item.get_local_range(0);

  slm[liy * rx + lix] = value;
  slm_idx[liy * rx + lix] = idx;
  for (int offset = 1; offset < rx; offset <<= 1) {
    item.barrier(dpcpp_local_fence);
    if (lix >= offset) {
      binary_op_update(
          slm[liy * rx + (lix - offset)],
          value,
          slm_idx[liy * rx + (lix - offset)],
          idx,
          func);
    }
    item.barrier(dpcpp_local_fence);

    if (lix >= offset) {
      slm[liy * rx + lix] = value;
      slm_idx[liy * rx + lix] = idx;
    }
  }

#else
  const auto wg_size = item.get_local_range(1);
  const auto sg = item.get_sub_group();
  const auto sgi = sg.get_group_linear_id();
  const auto sg_size = sg.get_local_range()[0];
  const auto lane = sg.get_local_linear_id();
  const auto sg_num = (wg_size + sg_size - 1) / sg_size;

  subgroup_scan_with_indices<T, BinaryFunction>(item, value, idx, func);
  if (lane == sg_size - 1) {
    carrier[sgi] = value;
    carrier_idx[sgi] = idx;
  }

  item.barrier(dpcpp_local_fence);
  for (int i = 0; i < sg_num; i++) {
    if (sgi > i) {
      binary_op_update(carrier[i], value, carrier_idx[i], idx, func);
    }
  }
#endif
}

template <class T, class BinaryFunction>
T group_y_scan(
    sycl::nd_item<2> item,
    T value,
    dpcpp_local_ptr<T> temp,
    BinaryFunction func) {
  const auto lix = item.get_local_id(1);
  const auto liy = item.get_local_id(0);
  const auto rx = item.get_local_range(1);
  const auto ry = item.get_local_range(0);

  temp[liy * rx + lix] = value;
  for (int offset = 1; offset < ry; offset <<= 1) {
    item.barrier(dpcpp_local_fence);
    if (liy >= offset)
      value = func(temp[(liy - offset) * rx + lix], temp[liy * rx + lix]);
    item.barrier(dpcpp_local_fence);

    if (liy >= offset)
      temp[liy * rx + lix] = value;
  }

  return value;
}

template <class T, class BinaryFunction>
void group_y_scan_with_indices(
    sycl::nd_item<2> item,
    T& value,
    int64_t& idx,
    dpcpp_local_ptr<T> temp,
    dpcpp_local_ptr<int64_t> temp_idx,
    BinaryFunction func) {
  const auto lix = item.get_local_id(1);
  const auto liy = item.get_local_id(0);
  const auto rx = item.get_local_range(1);
  const auto ry = item.get_local_range(0);

  temp[liy * rx + lix] = value;
  temp_idx[liy * rx + lix] = idx;
  for (int offset = 1; offset < ry; offset <<= 1) {
    item.barrier(dpcpp_local_fence);
    if (liy >= offset) {
      binary_op_update(
          temp[(liy - offset) * rx + lix],
          value,
          temp_idx[(liy - offset) * rx + lix],
          idx,
          func);
    }
    item.barrier(dpcpp_local_fence);

    if (liy >= offset) {
      temp[liy * rx + lix] = value;
      temp_idx[liy * rx + lix] = idx;
    }
  }
}

template <
    class InputInfo,
    class OutputInfo,
    class IndicesInfo,
    typename T,
    class BinaryFunction>
class LoopScanConfig {
 public:
  using arg_t = T;
  using func_t = BinaryFunction;
  using InputInfoType = InputInfo;
  using OutputInfoType = OutputInfo;
  using IndicesInfoType = OutputInfo;

  LoopScanConfig() = delete;
  LoopScanConfig(
      InputInfo input_info,
      OutputInfo output_info,
      IndicesInfo indices_info,
      int64_t batch,
      int64_t problem,
      T init,
      ScanType type,
      BinaryFunction func)
      : input_(input_info),
        output_(output_info),
        indices_(indices_info),
        batch_(batch),
        problem_(problem),
        stride_(1),
        init_(init),
        type_(type),
        func_(func),
        glb_range_x_(0),
        glb_range_y_(0),
        wg_range_x_(0),
        wg_range_y_(0) {
    auto dev_id = dpcppGetDeviceIdOfCurrentQueue();
    int64_t wg_size = dpcppMaxWorkItemsPerEU(dev_id);
    wg_range_x_ = 32;
    while (problem_ <= wg_range_x_ >> 1) {
      wg_range_x_ = wg_range_x_ >> 1;
    }
    wg_range_y_ = wg_size / wg_range_x_;
    const auto target_global_size = dpcppMaxWorkItemsPerTile(dev_id);
    ;
    const int max_work_group_num = target_global_size / wg_size;
    const int wg_number =
        std::min(max_work_group_num, CeilDiv((int)batch_, wg_range_y_));
    glb_range_x_ = wg_range_x_;
    glb_range_y_ = wg_range_y_ * wg_number;

    // For up down sweep algorithm, each work-item handle two elements.
    // This means that one work group would handle 2 times work group size
    // elements.
    loops_batch = (batch_ + glb_range_y_ - 1) / glb_range_y_;
    loops_problem = (problem_ + (wg_range_x_ * 2) - 1) / (wg_range_x_ * 2);
  }

  static LoopScanConfig<InputInfo, OutputInfo, IndicesInfo, T, BinaryFunction>
  make_config(
      InputInfo& input_info,
      OutputInfo& output_info,
      IndicesInfo& indices_info,
      int scan_dim,
      T init,
      ScanType type,
      BinaryFunction func) {
    int64_t batch = input_info.outerSize(scan_dim);
    int64_t stride = input_info.innerSize(scan_dim);
    int64_t problem = input_info.sizes[scan_dim];
    return {
        input_info,
        output_info,
        indices_info,
        batch,
        problem,
        init,
        type,
        func};
  }

  sycl::range<2> global_size() const {
    return {glb_range_y_, glb_range_x_};
  }

  sycl::range<2> group_size() const {
    return {wg_range_y_, wg_range_x_};
  }

  void set_type(ScanType other) {
    type_ = other;
  }

  struct item_desc {
    /* parallel batch, not tensor batch */ int64_t glb_batch;
    /* current global assignment id */ int64_t glb_problem;
  };

  item_desc get_item_desc(sycl::nd_item<2> item) const {
    auto giy = item.get_global_id(0);
    auto gix = item.get_global_id(1);

    return {giy, gix};
  }

 public:
  InputInfo input_;
  OutputInfo output_;
  IndicesInfo indices_;
  int64_t batch_;
  int64_t problem_;
  int64_t stride_;
  T init_;
  int loops_batch;
  int loops_problem;
  ScanType type_;
  BinaryFunction func_;
  int glb_range_x_;
  int glb_range_y_;
  int wg_range_x_;
  int wg_range_y_;
};

template <
    class LSConfig,
    class T,
    class BinaryFunction,
    bool TrivialOffCal = false>
class loop_scan_kernel {
 public:
  loop_scan_kernel(const LSConfig& cfg) : cfg(cfg) {}

  void lrun(
      sycl::nd_item<2> item,
      dpcpp_local_acc_t<T> slm,
      dpcpp_local_acc_t<T> max_carr) const {
    const int loops_batch = cfg.loops_batch;
    const int loops_problem = cfg.loops_problem;
    const auto group_size_x = cfg.wg_range_x_;
    const auto liy = item.get_local_id(0);

    for (int k = 0,
             base_off_batch_group = item.get_group(0) * item.get_local_range(0);
         k < loops_batch && base_off_batch_group < cfg.batch_;
         k++, base_off_batch_group += cfg.glb_range_y_) {
      max_carr[liy] = cfg.init_;
      int64_t base_off_batch = k * cfg.glb_range_y_ + item.get_global_id(0);
      for (int i = 0; i < loops_problem; ++i) {
        // calculate base addr offset for each loop
        int64_t base_off_problem = i * group_size_x * 2;
        max_carr[liy] = group_x_scan_by_uds_for_loop_scan<
            LSConfig,
            T,
            BinaryFunction,
            TrivialOffCal>(
            item, max_carr[liy], base_off_batch, base_off_problem, slm, cfg);
      }
    }
  }

 private:
  LSConfig cfg;
};

template <typename SKer, typename T>
class loop_scan {
 public:
  loop_scan(
      const SKer& ker,
      dpcpp_local_acc_t<T> shared,
      dpcpp_local_acc_t<T> max_carr)
      : ker_(ker), shared_(shared), max_carr_(max_carr) {}

  void operator()(sycl::nd_item<2> item) const {
    ker_.lrun(item, shared_, max_carr_);
  }

 private:
  SKer ker_;
  dpcpp_local_acc_t<T> shared_;
  dpcpp_local_acc_t<T> max_carr_;
};

template <typename LSConfig, bool TrivialOffCal = false>
static inline void launch_loop_scan(const LSConfig& cfg) {
  auto& queue = dpcppGetCurrentQueue();
  auto cgf = DPCPP_Q_CGF(__cgh) {
    // up_down_sweep demands 2 times work group size
    int slm_size = cfg.wg_range_x_ * cfg.wg_range_y_ * 2;
    int carr_size = cfg.wg_range_y_;
    dpcpp_local_acc_t<typename LSConfig::arg_t> shared(slm_size, __cgh);
    dpcpp_local_acc_t<typename LSConfig::arg_t> max_carr(carr_size, __cgh);
    loop_scan_kernel<
        LSConfig,
        typename LSConfig::arg_t,
        typename LSConfig::func_t,
        TrivialOffCal>
        ker(cfg);
    loop_scan sscan(ker, shared, max_carr);
    __cgh.parallel_for(
        sycl::nd_range<2>(cfg.global_size(), cfg.group_size()), sscan);
  };
  DPCPP_Q_SUBMIT(queue, cgf);
}

template <
    class LSConfig,
    class T,
    class BinaryFunction,
    bool TrivialOffCal = false>
class loop_scan_kernel_with_indices {
 public:
  loop_scan_kernel_with_indices(const LSConfig& cfg) : cfg(cfg) {}

  void lrun(
      sycl::nd_item<2> item,
      dpcpp_local_acc_t<T> slm,
      dpcpp_local_acc_t<int64_t> slm_idx) const {
    const int loops_batch = cfg.loops_batch;
    const int loops_problem = cfg.loops_problem;
    const auto group_size_x = cfg.wg_range_x_;
    const auto liy = item.get_local_id(0);

    for (int k = 0,
             base_off_batch_group = item.get_group(0) * item.get_local_range(0);
         k < loops_batch && base_off_batch_group < cfg.batch_;
         k++, base_off_batch_group += cfg.glb_range_y_) {
      T pre_max_carr = cfg.init_;
      int64_t pre_idx_carr = 0;
      int64_t base_off_batch = k * cfg.glb_range_y_ + item.get_global_id(0);
      for (int i = 0; i < loops_problem; ++i) {
        // calculate base addr offset for each loop
        int64_t base_off_problem = i * group_size_x * 2;
        group_x_scan_by_uds_for_loop_scan_with_indices<
            LSConfig,
            T,
            BinaryFunction,
            TrivialOffCal>(
            item,
            pre_max_carr,
            pre_idx_carr,
            base_off_batch,
            base_off_problem,
            slm,
            slm_idx,
            cfg);
      }
    }
  }

 private:
  LSConfig cfg;
};

template <typename SKer, typename T>
class loop_scan_with_indices {
 public:
  loop_scan_with_indices(
      const SKer& ker,
      dpcpp_local_acc_t<T> shared,
      dpcpp_local_acc_t<int64_t> shared_idx)
      : ker_(ker), shared_(shared), shared_idx_(shared_idx) {}

  void operator()(sycl::nd_item<2> item) const {
    ker_.lrun(item, shared_, shared_idx_);
  }

 private:
  SKer ker_;
  dpcpp_local_acc_t<T> shared_;
  dpcpp_local_acc_t<int64_t> shared_idx_;
};

template <typename LSConfig, bool TrivialOffCal = false>
static inline void launch_loop_scan_with_indices(const LSConfig& cfg) {
  auto& queue = dpcppGetCurrentQueue();
  auto cgf = DPCPP_Q_CGF(__cgh) {
    // up_down_sweep demands 2 times work group size
    int slm_size = cfg.wg_range_x_ * cfg.wg_range_y_ * 2;
    int carr_size = cfg.wg_range_y_;
    dpcpp_local_acc_t<typename LSConfig::arg_t> shared(slm_size, __cgh);
    dpcpp_local_acc_t<int64_t> shared_idx(slm_size, __cgh);
    loop_scan_kernel_with_indices<
        LSConfig,
        typename LSConfig::arg_t,
        typename LSConfig::func_t,
        TrivialOffCal>
        ker(cfg);
    loop_scan_with_indices sscan(ker, shared, shared_idx);
    __cgh.parallel_for(
        sycl::nd_range<2>(cfg.global_size(), cfg.group_size()), sscan);
  };
  DPCPP_Q_SUBMIT(queue, cgf);
}

template <
    class InputInfo,
    class OutputInfo,
    class IndicesInfo,
    typename T,
    class BinaryFunction>
class SegmentScanConfig : public BatchKernelConfig {
 public:
  using arg_t = T;
  using func_t = BinaryFunction;
  using InputInfoType = InputInfo;
  using OutputInfoType = OutputInfo;
  using IndicesInfoType = IndicesInfo;

  SegmentScanConfig() = delete;
  SegmentScanConfig(
      InputInfo input_info,
      OutputInfo output_info,
      IndicesInfo indices_info,
      int64_t batch,
      int64_t problem,
      int64_t stride,
      bool problem_along_x,
      T init,
      ScanType type,
      BinaryFunction func)
      : BatchKernelConfig(
            batch,
            problem,
            stride,
            batch * stride,
            problem_along_x),
        iinfo_(input_info),
        oinfo_(output_info),
        idxinfo_(indices_info),
        init_(init),
        type_(type),
        func_(func),
        carrier_(nullptr),
        carrier_idx_(nullptr) {}

  static SegmentScanConfig<
      InputInfo,
      OutputInfo,
      IndicesInfo,
      T,
      BinaryFunction>
  make_config(
      InputInfo& input_info,
      OutputInfo& output_info,
      IndicesInfo& indices_info,
      int scan_dim,
      T init,
      ScanType type,
      BinaryFunction func) {
    int64_t batch = input_info.outerSize(scan_dim);
    int64_t stride = input_info.innerSize(scan_dim);
    int64_t problem = input_info.sizes[scan_dim];
    int64_t problem_along_x = input_info.strides[scan_dim] == 1 ? true : false;
    return {
        input_info,
        output_info,
        indices_info,
        batch,
        problem,
        stride,
        problem_along_x,
        init,
        type,
        func};
  }

  int64_t carrier_size() {
    return problem_glb_range_ / problem_wg_range_ * batch_ * stride_;
  }

  void set_carrier(T* other) {
    carrier_ = other;
  }

  void set_carrier_idx(int64_t* other) {
    carrier_idx_ = other;
  }

  void set_type(ScanType other) {
    type_ = other;
  }

 public:
  InputInfo iinfo_;
  OutputInfo oinfo_;
  IndicesInfo idxinfo_;
  T init_;
  ScanType type_;
  BinaryFunction func_;
  /* contiguous temp buffer */ T* carrier_;
  /* contiguous temp buffer */ int64_t* carrier_idx_;
};

template <
    class SSConfig,
    class T,
    class BinaryFunction,
    bool TrivialOffCal = false,
    bool TrivialIdxCal = false>
class segment_scan_kernel {
 public:
  using InputInfo = typename SSConfig::InputInfoType;
  using OutputInfo = typename SSConfig::OutputInfoType;

  segment_scan_kernel(const SSConfig& cfg) : cfg(cfg) {}

 public:
  void srun(sycl::nd_item<2> item, dpcpp_local_acc_t<T> slm) const {
    auto id = cfg.get_item_desc(item);
    int64_t si, pi, bi, glb_ldr_off, glb_str_off, glb_str_off_0,
        glb_ldr_logical_off, glb_str_logical_off, crr_off;

    int64_t e = cfg.type_ == INCLUSIVE_TYPE ? 0 : 1;
    if constexpr (TrivialIdxCal) {
      glb_ldr_logical_off = item.get_global_linear_id();
      glb_str_logical_off = glb_ldr_logical_off + e;
      crr_off = id.chunk;
    } else {
      si = id.glb_batch % cfg.stride_;
      bi = id.glb_batch / cfg.stride_;
      pi = id.chunk * id.chunk_size + id.chunk_off;
      glb_ldr_logical_off =
          si + pi * cfg.stride_ + bi * cfg.problem_ * cfg.stride_;
      glb_str_logical_off =
          si + (pi + e) * cfg.stride_ + bi * cfg.problem_ * cfg.stride_;
      crr_off = si + id.chunk * cfg.stride_ + bi * id.chunk_num * cfg.stride_;
    }

    if constexpr (TrivialOffCal) {
      glb_ldr_off = glb_ldr_logical_off;
      glb_str_off = glb_str_logical_off;
      glb_str_off_0 = glb_ldr_logical_off;
    } else {
      glb_ldr_off = IndexToOffset<typename InputInfo::scalar_t, int64_t>::get(
          glb_ldr_logical_off,
          cfg.iinfo_,
          IndexToOffset<typename InputInfo::scalar_t, int64_t>::
              NON_STRICT_CONTIGUOUS);
      glb_str_off = IndexToOffset<typename OutputInfo::scalar_t, int64_t>::get(
          glb_str_logical_off,
          cfg.oinfo_,
          IndexToOffset<typename InputInfo::scalar_t, int64_t>::
              NON_STRICT_CONTIGUOUS);
      glb_str_off_0 =
          IndexToOffset<typename OutputInfo::scalar_t, int64_t>::get(
              glb_ldr_logical_off,
              cfg.oinfo_,
              IndexToOffset<typename InputInfo::scalar_t, int64_t>::
                  NON_STRICT_CONTIGUOUS);
    }
    T value = cfg.init_;
    if (id.glb_problem < cfg.problem_ && id.glb_batch < cfg.problem_batch_) {
      value = cfg.iinfo_.data[glb_ldr_off];
    }

    if (cfg.problem_along_x_) {
      // so far assign all work items along problem dimension
      // sg_shuffle benefits reduce on the dimension
      value = group_x_scan<T, BinaryFunction>(
          item, value, slm, cfg.init_, cfg.func_);
    } else {
      // parallel prefix reduce
      value = group_y_scan<T, BinaryFunction>(item, value, slm, cfg.func_);
    }

    if (id.glb_batch < cfg.problem_batch_) {
      if (cfg.type_ == INCLUSIVE_TYPE) {
        if (id.glb_problem < cfg.problem_) {
          cfg.oinfo_.data[glb_str_off] = value;
        }
      } else {
        if (id.glb_problem < cfg.problem_ - 1 &&
            id.chunk_off < id.chunk_size - 1) {
          cfg.oinfo_.data[glb_str_off] = value;
        }
        if (id.glb_problem < cfg.problem_ && id.chunk_off == 0) {
          cfg.oinfo_.data[glb_str_off_0] = cfg.init_;
        }
      }

      if (cfg.carrier_ != nullptr && id.chunk_off == id.chunk_size - 1) {
        cfg.carrier_[crr_off] = value;
      }
    }
  }

 private:
  SSConfig cfg;
};

template <
    class SSConfig,
    class T,
    class BinaryFunction,
    bool TrivialOffCal = false,
    bool TrivialIdxCal = false,
    bool is_idx_carried = false>
class segment_scan_kernel_with_indices {
 public:
  using InputInfo = typename SSConfig::InputInfoType;
  using OutputInfo = typename SSConfig::OutputInfoType;
  using IndicesInfo = typename SSConfig::IndicesInfoType;

  segment_scan_kernel_with_indices(const SSConfig& cfg) : cfg(cfg) {}

 public:
  void srun(
      sycl::nd_item<2> item,
      dpcpp_local_acc_t<T> slm,
      dpcpp_local_acc_t<int64_t> slm_idx) const {
    auto id = cfg.get_item_desc(item);
    int64_t si, pi, bi, glb_ldr_off, glb_str_off, glb_str_off_0, glb_idx_off,
        glb_idx_off_0, glb_ldr_logical_off, glb_str_logical_off, crr_off,
        glb_idx_logical_off;

    int64_t e = cfg.type_ == INCLUSIVE_TYPE ? 0 : 1;
    if constexpr (TrivialIdxCal) {
      glb_ldr_logical_off = item.get_global_linear_id();
      glb_str_logical_off = glb_ldr_logical_off + e;
      glb_idx_logical_off = glb_ldr_logical_off + e;
      pi = id.chunk * id.chunk_size + id.chunk_off;
      crr_off = id.chunk;
    } else {
      si = id.glb_batch % cfg.stride_;
      bi = id.glb_batch / cfg.stride_;
      pi = id.chunk * id.chunk_size + id.chunk_off;
      glb_ldr_logical_off =
          si + pi * cfg.stride_ + bi * cfg.problem_ * cfg.stride_;
      glb_str_logical_off =
          si + (pi + e) * cfg.stride_ + bi * cfg.problem_ * cfg.stride_;
      glb_idx_logical_off =
          si + (pi + e) * cfg.stride_ + bi * cfg.problem_ * cfg.stride_;
      crr_off = si + id.chunk * cfg.stride_ + bi * id.chunk_num * cfg.stride_;
    }

    if constexpr (TrivialOffCal) {
      glb_ldr_off = glb_ldr_logical_off;
      glb_str_off = glb_str_logical_off;
      glb_str_off_0 = glb_ldr_logical_off;
      glb_idx_off = glb_idx_logical_off;
      glb_idx_off_0 = glb_ldr_logical_off;
    } else {
      glb_ldr_off = IndexToOffset<typename InputInfo::scalar_t, int64_t>::get(
          glb_ldr_logical_off,
          cfg.iinfo_,
          IndexToOffset<typename InputInfo::scalar_t, int64_t>::
              NON_STRICT_CONTIGUOUS);
      glb_str_off = IndexToOffset<typename OutputInfo::scalar_t, int64_t>::get(
          glb_str_logical_off,
          cfg.oinfo_,
          IndexToOffset<typename InputInfo::scalar_t, int64_t>::
              NON_STRICT_CONTIGUOUS);
      glb_str_off_0 =
          IndexToOffset<typename OutputInfo::scalar_t, int64_t>::get(
              glb_ldr_logical_off,
              cfg.oinfo_,
              IndexToOffset<typename InputInfo::scalar_t, int64_t>::
                  NON_STRICT_CONTIGUOUS);
      glb_idx_off = IndexToOffset<typename IndicesInfo::scalar_t, int64_t>::get(
          glb_idx_logical_off,
          cfg.idxinfo_,
          IndexToOffset<typename InputInfo::scalar_t, int64_t>::
              NON_STRICT_CONTIGUOUS);
      glb_idx_off_0 =
          IndexToOffset<typename IndicesInfo::scalar_t, int64_t>::get(
              glb_ldr_logical_off,
              cfg.oinfo_,
              IndexToOffset<typename InputInfo::scalar_t, int64_t>::
                  NON_STRICT_CONTIGUOUS);
    }
    T value = cfg.init_;
    int64_t idx = pi;
    if (id.glb_problem < cfg.problem_ && id.glb_batch < cfg.problem_batch_) {
      value = cfg.iinfo_.data[glb_ldr_off];
      if constexpr (is_idx_carried) {
        idx = cfg.idxinfo_.data[glb_ldr_off];
      }
    }

    if (cfg.problem_along_x_) {
      // so far assign all work items along problem dimension
      // sg_shuffle benefits reduce on the dimension
      group_x_scan_with_indices<T, BinaryFunction>(
          item, value, idx, slm, slm_idx, cfg.init_, cfg.func_);
    } else {
      // parallel prefix reduce
      group_y_scan_with_indices<T, BinaryFunction>(
          item, value, idx, slm, slm_idx, cfg.func_);
    }

    if (id.glb_batch < cfg.problem_batch_) {
      if (cfg.type_ == INCLUSIVE_TYPE) {
        if (id.glb_problem < cfg.problem_) {
          cfg.oinfo_.data[glb_str_off] = value;
          cfg.idxinfo_.data[glb_idx_off] = idx;
        }
      } else {
        if (id.glb_problem < cfg.problem_ - 1 &&
            id.chunk_off < id.chunk_size - 1) {
          cfg.oinfo_.data[glb_str_off] = value;
          cfg.idxinfo_.data[glb_idx_off] = idx;
        }
        if (id.glb_problem < cfg.problem_ && id.chunk_off == 0) {
          cfg.oinfo_.data[glb_str_off_0] = cfg.init_;
          cfg.idxinfo_.data[glb_idx_off_0] = pi;
        }
      }

      if (cfg.carrier_ != nullptr && cfg.carrier_idx_ != nullptr &&
          id.chunk_off == id.chunk_size - 1) {
        cfg.carrier_[crr_off] = value;
        cfg.carrier_idx_[crr_off] = idx;
      }
    }
  }

 private:
  SSConfig cfg;
};

template <typename SKer, typename T>
class segment_scan {
 public:
  segment_scan(const SKer& ker, dpcpp_local_acc_t<T> shared)
      : ker_(ker), shared_(shared) {}

  void operator()(sycl::nd_item<2> item) const {
    ker_.srun(item, shared_);
  }

 private:
  SKer ker_;
  dpcpp_local_acc_t<T> shared_;
};

template <typename SKer, typename T>
class segment_scan_with_indices {
 public:
  segment_scan_with_indices(
      const SKer& ker,
      dpcpp_local_acc_t<T> shared,
      dpcpp_local_acc_t<int64_t> shared_idx)
      : ker_(ker), shared_(shared), shared_idx_(shared_idx) {}

  void operator()(sycl::nd_item<2> item) const {
    ker_.srun(item, shared_, shared_idx_);
  }

 private:
  SKer ker_;
  dpcpp_local_acc_t<T> shared_;
  dpcpp_local_acc_t<int64_t> shared_idx_;
};

template <
    typename SSConfig,
    bool TrivialOffCal = false,
    bool TrivialIdxCal = false>
static inline void launch_segment_scan(const SSConfig& cfg) {
  auto& queue = dpcppGetCurrentQueue();

  auto cgf = DPCPP_Q_CGF(__cgh) {
    // SLM for sub group cumsum/cumprod in one workgroup. Allow that
    // min sub_group_size is 8, herein we use wg_size / 8 to allocate
    // enough share local memory for simd32, simd16 and simd8.
    int min_sg_size = dpcppMinSubGroupSize();
    int carrier_size = (cfg.problem_wg_range_ + min_sg_size - 1) / min_sg_size;
    int wg_size = cfg.wg_range_x_ * cfg.wg_range_y_;
#ifndef SG_SCAN
    int slm_size = wg_size;
#else
    int slm_size = cfg.problem_along_x_ ? carrier_size : wg_size;
#endif
    dpcpp_local_acc_t<typename SSConfig::arg_t> shared(slm_size, __cgh);
    segment_scan_kernel<
        SSConfig,
        typename SSConfig::arg_t,
        typename SSConfig::func_t,
        TrivialOffCal,
        TrivialIdxCal>
        ker(cfg);
    segment_scan gscan(ker, shared);
    __cgh.parallel_for(
        sycl::nd_range<2>(cfg.global_size(), cfg.group_size()), gscan);
  };
  DPCPP_Q_SUBMIT(queue, cgf);
}

template <
    typename SSConfig,
    bool TrivialOffCal = false,
    bool TrivialIdxCal = false,
    bool is_idx_carried = false>
static inline void launch_segment_scan_with_indices(const SSConfig& cfg) {
  auto& queue = dpcppGetCurrentQueue();

  auto cgf = DPCPP_Q_CGF(__cgh) {
    // SLM for sub group cumsum/cumprod in one workgroup. Allow that
    // min sub_group_size is 8, herein we use wg_size / 8 to allocate
    // enough share local memory for simd32, simd16 and simd8.
    int min_sg_size = dpcppMinSubGroupSize();
    int carrier_size = (cfg.problem_wg_range_ + min_sg_size - 1) / min_sg_size;
    int wg_size = cfg.wg_range_x_ * cfg.wg_range_y_;
#ifndef SG_SCAN
    int slm_size = wg_size;
#else
    int slm_size = cfg.problem_along_x_ ? carrier_size : wg_size;
#endif
    //    sycl::stream sout(1024, 256, __cgh);

    dpcpp_local_acc_t<typename SSConfig::arg_t> shared(slm_size, __cgh);
    dpcpp_local_acc_t<int64_t> shared_idx(slm_size, __cgh);
    segment_scan_kernel_with_indices<
        SSConfig,
        typename SSConfig::arg_t,
        typename SSConfig::func_t,
        TrivialOffCal,
        TrivialIdxCal,
        is_idx_carried>
        ker(cfg);
    segment_scan_with_indices gscan(ker, shared, shared_idx);
    __cgh.parallel_for(
        sycl::nd_range<2>(cfg.global_size(), cfg.group_size()), gscan);
  };
  DPCPP_Q_SUBMIT(queue, cgf);
}

template <class SSConfig, bool TrivialIdxCal = false>
static inline void accumulate_carrier(const SSConfig& cfg) {
  TORCH_CHECK(
      cfg.carrier_ != nullptr, "scan: nullptr carrier in accumulation ...");
  auto& queue = dpcppGetCurrentQueue();

  auto cgf = DPCPP_Q_CGF(__cgh) {
    auto kfn = DPCPP_Q_KFN(sycl::nd_item<2> item) {
      auto id = cfg.get_item_desc(item);
      int64_t si, pi, bi, glb_off, crr_off;
      if constexpr (TrivialIdxCal) {
        glb_off = item.get_global_linear_id();
        crr_off = id.chunk;
      } else {
        si = id.glb_batch % cfg.stride_;
        bi = id.glb_batch / cfg.stride_;
        pi = id.chunk * id.chunk_size + id.chunk_off;
        glb_off = si + pi * cfg.stride_ + bi * cfg.problem_ * cfg.stride_;
        crr_off = si + id.chunk * cfg.stride_ + bi * id.chunk_num * cfg.stride_;
      }
      if (id.glb_problem < cfg.problem_ && id.glb_batch < cfg.problem_batch_) {
        cfg.oinfo_.data[glb_off] =
            cfg.func_(cfg.oinfo_.data[glb_off], cfg.carrier_[crr_off]);
      }
    };
    __cgh.parallel_for(
        sycl::nd_range<2>(cfg.global_size(), cfg.group_size()), kfn);
  };
  DPCPP_Q_SUBMIT(queue, cgf);
}

template <class SSConfig, bool TrivialIdxCal = false>
static inline void accumulate_carrier_with_indices(const SSConfig& cfg) {
  TORCH_CHECK(
      cfg.carrier_ != nullptr, "scan: nullptr carrier in accumulation ...");
  TORCH_CHECK(
      cfg.carrier_idx_ != nullptr,
      "scan_with_indices: nullptr carrier in accumulation ...");
  auto& queue = dpcppGetCurrentQueue();

  auto cgf = DPCPP_Q_CGF(__cgh) {
    auto kfn = DPCPP_Q_KFN(sycl::nd_item<2> item) {
      auto id = cfg.get_item_desc(item);
      int64_t si, pi, bi, glb_off, crr_off;
      if constexpr (TrivialIdxCal) {
        glb_off = item.get_global_linear_id();
        crr_off = id.chunk;
      } else {
        si = id.glb_batch % cfg.stride_;
        bi = id.glb_batch / cfg.stride_;
        pi = id.chunk * id.chunk_size + id.chunk_off;
        glb_off = si + pi * cfg.stride_ + bi * cfg.problem_ * cfg.stride_;
        crr_off = si + id.chunk * cfg.stride_ + bi * id.chunk_num * cfg.stride_;
      }
      if (id.glb_problem < cfg.problem_ && id.glb_batch < cfg.problem_batch_) {
        binary_op_update(
            cfg.carrier_[crr_off],
            cfg.oinfo_.data[glb_off],
            cfg.carrier_idx_[crr_off],
            cfg.idxinfo_.data[glb_off],
            cfg.func_);
      }
    };
    __cgh.parallel_for(
        sycl::nd_range<2>(cfg.global_size(), cfg.group_size()), kfn);
  };
  DPCPP_Q_SUBMIT(queue, cgf);
}

template <typename T, class BinaryOp, int Power2ScanSize>
void inclusivePrefixScan(
    T* smem,
    BinaryOp binop,
    const sycl::nd_item<1>& item_id) {
  // Reduce step ("upsweep")
  int threadIdx = item_id.get_local_id(0);
  for (int stride = 1; stride < Power2ScanSize; stride <<= 1) {
    int index = (threadIdx + 1) * stride * 2 - 1;
    if (index < Power2ScanSize) {
      smem[index] = binop(smem[index], smem[index - stride]);
    }
    item_id.barrier(dpcpp_local_fence);
  }

  // Post-reduce step ("downsweep")
  for (int stride = Power2ScanSize / 4; stride > 0; stride >>= 1) {
    int index = (threadIdx + 1) * stride * 2 - 1;
    if ((index + stride) < Power2ScanSize) {
      smem[index + stride] = binop(smem[index + stride], smem[index]);
    }
    item_id.barrier(dpcpp_local_fence);
  }
}

// Inclusive prefix sum using shared Memory
template <typename T, class BinaryFunction>
void inclusivePrefixScan(
    const dpcpp_local_acc_t<T>& smem,
    T in,
    T* out,
    BinaryFunction binop,
    const sycl::nd_item<1>& item_id) {
  // FIXME: this is slow
  int threadIdx = item_id.get_local_id(0);
  smem[threadIdx] = in;
  item_id.barrier(dpcpp_local_fence);
  for (int64_t offset = 1;
       offset < static_cast<int64_t>(item_id.get_local_range(0));
       offset *= 2) {
    T val = 0;

    if (threadIdx >= offset) {
      val = binop(smem[threadIdx - offset], smem[threadIdx]);
    }
    item_id.barrier(dpcpp_local_fence);

    if (threadIdx >= offset) {
      smem[threadIdx] = val;
    }
    item_id.barrier(dpcpp_local_fence);
  }
  *out = smem[threadIdx];
  item_id.barrier(dpcpp_local_fence);
}

// Exclusive prefix sum using shared memory
template <typename T, class BinaryFunction>
void exclusivePrefixScan(
    const dpcpp_local_acc_t<T>& smem,
    T in,
    T* out,
    T* carry,
    BinaryFunction binop,
    const sycl::nd_item<1>& item_id) {
  inclusivePrefixScan<T, BinaryFunction>(smem, in, out, binop, item_id);

  *out -= in;
  *carry = smem[item_id.get_local_range(0) - 1];

  // Prevent write-after-read dependencies on smem usage above if necessary
  item_id.barrier(dpcpp_local_fence);
}

static inline bool dispatch_to_loop_scan_kernel(
    const int64_t problem,
    const int64_t stride,
    const int64_t batch) {
  // stride > 1 scenario
  if (stride > 1) {
    // If stride > 1, we use batch scan anyway.
    return false;
  }

  // 1 == stride scenario
  if (batch > 128 && problem < 16384 /*1024 * 16*/) {
    // Only if batch > 128, and problem is not so big, we use loop
    // scan kernel to avoid so many global memory access operations.
    // If batch is so small, or problem is so big, we use batch scan
    // to increase work group mumber to increase device coverage.
    return true;
  }

  return false;
}

template <
    ScanType Type,
    bool TrivialOffCal,
    typename T,
    class InputInfo,
    class OutputInfo,
    class BinaryFunction>
static inline void _loop_scan_kernel(
    InputInfo& input_info,
    OutputInfo& output_info,
    int dim_after_collapse,
    T init,
    BinaryFunction func) {
  auto cfg =
      LoopScanConfig<InputInfo, OutputInfo, OutputInfo, T, BinaryFunction>::
          make_config(
              input_info,
              output_info,
              output_info,
              dim_after_collapse,
              init,
              Type,
              func);
  TORCH_CHECK(1 == cfg.stride_);
  launch_loop_scan<decltype(cfg), TrivialOffCal>(cfg);

  return;
}

template <
    ScanType Type,
    bool TrivialOffCal,
    typename T,
    class InputInfo,
    class OutputInfo,
    class IndicesInfo,
    class BinaryFunction>
static inline void _loop_scan_kernel_with_indices(
    InputInfo& input_info,
    OutputInfo& output_info,
    IndicesInfo& indices_info,
    int dim_after_collapse,
    T init,
    BinaryFunction func) {
  auto cfg =
      LoopScanConfig<InputInfo, OutputInfo, IndicesInfo, T, BinaryFunction>::
          make_config(
              input_info,
              output_info,
              indices_info,
              dim_after_collapse,
              init,
              Type,
              func);
  TORCH_CHECK(1 == cfg.stride_);
  launch_loop_scan_with_indices<decltype(cfg), TrivialOffCal>(cfg);

  return;
}

template <
    ScanType Type,
    bool TrivialOffCal,
    bool TrivialIdxCal,
    typename T,
    class InputInfo,
    class OutputInfo,
    class BinaryFunction>
static inline void _segment_scan_kernel(
    InputInfo& input_info,
    OutputInfo& output_info,
    int dim_after_collapse,
    T init,
    BinaryFunction func) {
  auto cfg = SegmentScanConfig<
      InputInfo,
      OutputInfo,
      OutputInfo /*not used*/,
      T,
      BinaryFunction>::
      make_config(
          input_info,
          output_info,
          output_info /*not used*/,
          dim_after_collapse,
          init,
          Type,
          func);
  // 0. recursive convergence
  if (cfg.problem_ <= cfg.problem_wg_range_) {
    cfg.set_carrier(nullptr);
    launch_segment_scan<decltype(cfg), TrivialOffCal, TrivialIdxCal>(cfg);
    return;
  }

  // 1. inclusive scan in each chunk
  Tensor carrier_holder = at::empty(
      {cfg.batch_, cfg.problem_glb_range_ / cfg.problem_wg_range_, cfg.stride_},
      map_options<T>());
  TensorInfo<T, int64_t> carrier_info =
      getTensorInfo<T, int64_t>(carrier_holder);
  cfg.set_carrier(carrier_info.data);
  launch_segment_scan<decltype(cfg), TrivialOffCal, TrivialIdxCal>(cfg);

  // 2. recursion for carrier
  _segment_scan_kernel<EXCLUSIVE_TYPE, TrivialOffCal, TrivialIdxCal>(
      carrier_info, carrier_info, 1, init, func);

  // 3. accumulate among all chunk
  accumulate_carrier<decltype(cfg), TrivialIdxCal>(cfg);

  return;
}

template <
    ScanType Type,
    bool TrivialOffCal,
    bool TrivialIdxCal,
    bool is_idx_carried,
    typename T,
    class InputInfo,
    class OutputInfo,
    class IndicesInfo,
    class BinaryFunction>
static inline void _segment_scan_kernel_with_indices(
    InputInfo& input_info,
    OutputInfo& output_info,
    IndicesInfo& indices_info,
    int dim_after_collapse,
    T init,
    BinaryFunction func) {
  auto cfg =
      SegmentScanConfig<InputInfo, OutputInfo, IndicesInfo, T, BinaryFunction>::
          make_config(
              input_info,
              output_info,
              indices_info,
              dim_after_collapse,
              init,
              Type,
              func);
  // 0. recursive convergence
  if (cfg.problem_ <= cfg.problem_wg_range_) {
    cfg.set_carrier(nullptr);
    cfg.set_carrier_idx(nullptr);
    launch_segment_scan_with_indices<
        decltype(cfg),
        TrivialOffCal,
        TrivialIdxCal,
        is_idx_carried>(cfg);
    return;
  }

  // 1. inclusive scan in each chunk
  Tensor carrier_holder = at::empty(
      {cfg.batch_, cfg.problem_glb_range_ / cfg.problem_wg_range_, cfg.stride_},
      map_options<T>());
  TensorInfo<T, int64_t> carrier_info =
      getTensorInfo<T, int64_t>(carrier_holder);
  cfg.set_carrier(carrier_info.data);

  Tensor carrier_idx_holder = at::empty(
      {cfg.batch_, cfg.problem_glb_range_ / cfg.problem_wg_range_, cfg.stride_},
      map_options<int64_t>());
  TensorInfo<int64_t, int64_t> carrier_idx_info =
      getTensorInfo<int64_t, int64_t>(carrier_idx_holder);
  cfg.set_carrier_idx(carrier_idx_info.data);
  launch_segment_scan_with_indices<
      decltype(cfg),
      TrivialOffCal,
      TrivialIdxCal,
      is_idx_carried>(cfg);

  // 2. recursion for carrier
  _segment_scan_kernel_with_indices<
      EXCLUSIVE_TYPE,
      TrivialOffCal,
      TrivialIdxCal,
      true>(carrier_info, carrier_info, carrier_idx_info, 1, init, func);

  // 3. accumulate among all chunk
  accumulate_carrier_with_indices<decltype(cfg), TrivialIdxCal>(cfg);

  return;
}

template <
    ScanType Type,
    typename scalar_t,
    typename oscalar_t,
    class BinaryFunction>
void scan(
    Tensor& self,
    const Tensor& input,
    int dimension,
    scalar_t init,
    BinaryFunction func) {
  self.resize_as_(input);
  if (input.dim() == 0) {
    self.fill_(input);
    return;
  } else if (input.numel() == 0) {
    self.zero_();
    return;
  }
  auto input_ = input.contiguous();
  dimension = maybe_wrap_dim(dimension, input_.dim());
  TORCH_CHECK(
      dimension >= 0 && dimension < input_.dim(),
      "dimension ",
      dimension,
      " out of range");

  TensorInfo<scalar_t, int64_t> input_info =
      getTensorInfo<scalar_t, int64_t>(input_);
  int dim_after_collapse = input_info.collapseDims(dimension);

  TensorInfo<oscalar_t, int64_t> output_info =
      getTensorInfo<oscalar_t, int64_t>(self);
  output_info.collapseDims(dimension);

  int64_t batch = input_info.outerSize(dim_after_collapse);
  int64_t stride = input_info.innerSize(dim_after_collapse);
  int64_t problem = input_info.sizes[dim_after_collapse];

  if (dispatch_to_loop_scan_kernel(problem, stride, batch)) {
    _loop_scan_kernel<Type, true>(
        input_info, output_info, dim_after_collapse, init, func);
  } else {
    if (batch == 1 && stride == 1) {
      _segment_scan_kernel<Type, true, true>(
          input_info, output_info, dim_after_collapse, init, func);
    } else {
      _segment_scan_kernel<Type, true, false>(
          input_info, output_info, dim_after_collapse, init, func);
    }
  }
}

template <
    ScanType Type,
    typename scalar_t,
    typename oscalar_t,
    class BinaryFunction>
void scan_with_indices(
    const Tensor& self_,
    Tensor& values,
    Tensor& indices,
    int dimension,
    scalar_t init,
    BinaryFunction func) {
  int ndim = self_.dim();
  auto self = self_.contiguous();
  TORCH_INTERNAL_ASSERT(values.is_contiguous() && indices.is_contiguous());

  dimension = maybe_wrap_dim(dimension, self.dim());
  TORCH_CHECK(
      dimension >= 0 && dimension < self.dim(),
      "dimension ",
      dimension,
      " out of range");

  TensorInfo<scalar_t, int64_t> input_info =
      getTensorInfo<scalar_t, int64_t>(self);
  int dim_after_collapse = input_info.collapseDims(dimension);

  TensorInfo<oscalar_t, int64_t> output_info =
      getTensorInfo<oscalar_t, int64_t>(values);
  output_info.collapseDims(dimension);
  TensorInfo<int64_t, int64_t> indices_info =
      getTensorInfo<int64_t, int64_t>(indices);

  int64_t batch = input_info.outerSize(dim_after_collapse);
  int64_t stride = input_info.innerSize(dim_after_collapse);
  int64_t problem = input_info.sizes[dim_after_collapse];

  if (dispatch_to_loop_scan_kernel(problem, stride, batch)) {
    _loop_scan_kernel_with_indices<Type, true>(
        input_info, output_info, indices_info, dim_after_collapse, init, func);
  } else {
    if (batch == 1 && stride == 1) {
      _segment_scan_kernel_with_indices<Type, true, true, false>(
          input_info,
          output_info,
          indices_info,
          dim_after_collapse,
          init,
          func);
    } else {
      _segment_scan_kernel_with_indices<Type, true, false, false>(
          input_info,
          output_info,
          indices_info,
          dim_after_collapse,
          init,
          func);
    }
  }
}
} // namespace AtenIpexTypeXPU
} // namespace at
