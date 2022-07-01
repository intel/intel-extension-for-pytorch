#pragma once
#include <core/detail/TensorInfo.h>
#include <utils/DPCPP.h>
#include "BatchKernel.h"

using namespace xpu::dpcpp::detail;

typedef enum {
  EXCLUSIVE_TYPE = 0,
  INCLUSIVE_TYPE = 1,
} ScanType;

// group x scan by using up down sweep algorithm(call uds for short)
template <class LSConfig, class T, class BinaryFunction>
DPCPP_DEVICE T group_x_scan_by_uds_for_loop_scan(
    DPCPP::nd_item<2> item,
    const T pre_max_carr,
    int64_t base_off,
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

  uint32_t ix0 = base_off + lix;
  uint32_t ix1 = base_off + rx + lix;

  glb_ldr_logical_off_0 = ix0 + id.glb_batch * cfg.problem_;
  glb_ldr_off_0 = IndexToOffset<typename InputInfo::scalar_t, int64_t>::get(
      glb_ldr_logical_off_0,
      cfg.input_,
      IndexToOffset<typename InputInfo::scalar_t, int64_t>::
          NON_STRICT_CONTIGUOUS);

  glb_ldr_logical_off_1 = ix1 + id.glb_batch * cfg.problem_;
  glb_ldr_off_1 = IndexToOffset<typename InputInfo::scalar_t, int64_t>::get(
      glb_ldr_logical_off_1,
      cfg.input_,
      IndexToOffset<typename InputInfo::scalar_t, int64_t>::
          NON_STRICT_CONTIGUOUS);

  glb_str_logical_off_0 = ix0 + id.glb_batch * cfg.problem_;
  glb_str_off_0 = IndexToOffset<typename OutputInfo::scalar_t, int64_t>::get(
      glb_str_logical_off_0,
      cfg.output_,
      IndexToOffset<typename OutputInfo::scalar_t, int64_t>::
          NON_STRICT_CONTIGUOUS);

  glb_str_logical_off_1 = ix1 + id.glb_batch * cfg.problem_;
  glb_str_off_1 = IndexToOffset<typename OutputInfo::scalar_t, int64_t>::get(
      glb_str_logical_off_1,
      cfg.output_,
      IndexToOffset<typename OutputInfo::scalar_t, int64_t>::
          NON_STRICT_CONTIGUOUS);

  // TODO: opti for bank conflict elemination
  // Read data from global memory to shared local memory
  // Each work item load 2 elements from global device memory to shared local
  // memory
  if (id.glb_batch < cfg.batch_) {
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
    if (id.glb_batch < cfg.batch_ && lix < s) {
      uint32_t offset = liy * rx * 2 + (2 * lix + 1) * d - 1;
      slm[offset + d] = cfg.func_(slm[offset], slm[offset + d]);
    }
    if (sub_group_size != cfg.wg_range_x_) {
      item.barrier(dpcpp_local_fence);
    }
  }

  // Down-sweep
  for (uint32_t s = 2, d = rx / 2; d >= 1; s <<= 1, d >>= 1) {
    if (id.glb_batch < cfg.batch_ && lix < s - 1) {
      uint32_t offset = liy * rx * 2 + 2 * (lix + 1) * d - 1;
      slm[offset + d] = cfg.func_(slm[offset], slm[offset + d]);
    }
    if (sub_group_size != cfg.wg_range_x_) {
      item.barrier(dpcpp_local_fence);
    }
  }

  // Write back from shared local memory to global memory
  if (id.glb_batch < cfg.batch_) {
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

template <class T, class BinaryFunction>
DPCPP_DEVICE T
subgroup_scan(DPCPP::nd_item<2> item, T value, BinaryFunction func) {
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
DPCPP_DEVICE T group_x_scan(
    DPCPP::nd_item<2> item,
    T value,
#ifndef SG_SCAN
    dpcpp_local_ptr<T> slm,
#else
    dpcpp_local_ptr<T> carrier,
#endif
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
DPCPP_DEVICE T group_y_scan(
    DPCPP::nd_item<2> item,
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

template <class InputInfo, class OutputInfo, typename T, class BinaryFunction>
class LoopScanConfig {
 public:
  using arg_t = T;
  using func_t = BinaryFunction;
  using InputInfoType = InputInfo;
  using OutputInfoType = OutputInfo;

  LoopScanConfig() = delete;
  LoopScanConfig(
      InputInfo input_info,
      OutputInfo output_info,
      int64_t batch,
      int64_t problem,
      T init,
      ScanType type,
      BinaryFunction func)
      : input_(input_info),
        output_(output_info),
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
    int64_t wg_size = dpcppMaxWorkGroupSize(dpcppGetDeviceIdOfCurrentQueue());
    wg_range_x_ = 32;
    wg_range_y_ = wg_size / wg_range_x_;

    glb_range_x_ = wg_range_x_;
    glb_range_y_ =
        wg_range_y_ * (int)((batch_ + wg_range_y_ - 1) / wg_range_y_);

    // For up down sweep algorithm, each work-item handle two elements.
    // This means that one work group would handle 2 times work group size
    // elements.
    loops_ = (problem_ + (wg_range_x_ * 2) - 1) / (wg_range_x_ * 2);
  }

  static LoopScanConfig<InputInfo, OutputInfo, T, BinaryFunction> make_config(
      InputInfo& input_info,
      OutputInfo& output_info,
      int scan_dim,
      T init,
      ScanType type,
      BinaryFunction func) {
    int64_t batch = input_info.outerSize(scan_dim);
    int64_t stride = input_info.innerSize(scan_dim);
    int64_t problem = input_info.sizes[scan_dim];
    return {input_info, output_info, batch, problem, init, type, func};
  }

  DPCPP::range<2> global_size() const {
    return {glb_range_y_, glb_range_x_};
  }

  DPCPP::range<2> group_size() const {
    return {wg_range_y_, wg_range_x_};
  }

  void set_type(ScanType other) {
    type_ = other;
  }

  struct item_desc {
    /* parallel batch, not tensor batch */ int64_t glb_batch;
    /* current global assignment id */ int64_t glb_problem;
  };

  item_desc get_item_desc(DPCPP::nd_item<2> item) const {
    auto giy = item.get_global_id(0);
    auto gix = item.get_global_id(1);

    return {giy, gix};
  }

 public:
  InputInfo input_;
  OutputInfo output_;
  int64_t batch_;
  int64_t problem_;
  int64_t stride_;
  T init_;
  int loops_;
  ScanType type_;
  BinaryFunction func_;
  int glb_range_x_;
  int glb_range_y_;
  int wg_range_x_;
  int wg_range_y_;
};

template <class LSConfig, class T, class BinaryFunction>
class loop_scan_kernel {
 public:
  loop_scan_kernel(const LSConfig& cfg) : cfg(cfg) {}

  DPCPP_DEVICE void lrun(
      DPCPP::nd_item<2> item,
      dpcpp_local_acc_t<T> slm,
      dpcpp_local_acc_t<T> max_carr) const {
    const int loops = cfg.loops_;
    const auto group_size_x = cfg.wg_range_x_;
    const auto liy = item.get_local_id(0);
    max_carr[liy] = cfg.init_;

    for (int i = 0; i < loops; ++i) {
      // calculate base addr offset for each loop
      int64_t base_off = i * group_size_x * 2;
      max_carr[liy] =
          group_x_scan_by_uds_for_loop_scan<LSConfig, T, BinaryFunction>(
              item, max_carr[liy], base_off, slm, cfg);
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

  void operator()(DPCPP::nd_item<2> item) const {
    ker_.lrun(item, shared_, max_carr_);
  }

 private:
  SKer ker_;
  dpcpp_local_acc_t<T> shared_;
  dpcpp_local_acc_t<T> max_carr_;
};

template <typename LSConfig>
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
        typename LSConfig::func_t>
        ker(cfg);

    loop_scan sscan(ker, shared, max_carr);
    __cgh.parallel_for(
        DPCPP::nd_range<2>(cfg.global_size(), cfg.group_size()), sscan);
  };
  DPCPP_Q_SUBMIT(queue, cgf);
}

template <class InputInfo, class OutputInfo, typename T, class BinaryFunction>
class SegmentScanConfig : public BatchKernelConfig {
 public:
  using arg_t = T;
  using func_t = BinaryFunction;
  using InputInfoType = InputInfo;
  using OutputInfoType = OutputInfo;

  SegmentScanConfig() = delete;
  SegmentScanConfig(
      InputInfo input_info,
      OutputInfo output_info,
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
        init_(init),
        type_(type),
        func_(func),
        carrier_(nullptr) {}

  static SegmentScanConfig<InputInfo, OutputInfo, T, BinaryFunction> make_config(
      InputInfo& input_info,
      OutputInfo& output_info,
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

  void set_type(ScanType other) {
    type_ = other;
  }

 public:
  InputInfo iinfo_;
  OutputInfo oinfo_;
  T init_;
  ScanType type_;
  BinaryFunction func_;
  /* contiguous temp buffer */ T* carrier_;
};

template <class SSConfig, class T, class BinaryFunction>
class segment_scan_kernel {
 public:
  using InputInfo = typename SSConfig::InputInfoType;
  using OutputInfo = typename SSConfig::OutputInfoType;

  segment_scan_kernel(const SSConfig& cfg) : cfg(cfg) {}

 public:
  DPCPP_DEVICE void srun(DPCPP::nd_item<2> item, dpcpp_local_acc_t<T> slm)
      const {
    auto id = cfg.get_item_desc(item);
    int64_t si, pi, bi, glb_ldr_off, glb_str_off, glb_str_off_0,
        glb_ldr_logical_off, glb_str_logical_off, crr_off;

    si = id.glb_batch % cfg.stride_;
    bi = id.glb_batch / cfg.stride_;
    pi = id.chunk * id.chunk_size + id.chunk_off;

    int64_t e = cfg.type_ == INCLUSIVE_TYPE ? 0 : 1;
    glb_ldr_logical_off =
        si + pi * cfg.stride_ + bi * cfg.problem_ * cfg.stride_;
    glb_str_logical_off =
        si + (pi + e) * cfg.stride_ + bi * cfg.problem_ * cfg.stride_;
    crr_off = si + id.chunk * cfg.stride_ + bi * id.chunk_num * cfg.stride_;

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
    glb_str_off_0 = IndexToOffset<typename OutputInfo::scalar_t, int64_t>::get(
        glb_ldr_logical_off,
        cfg.oinfo_,
        IndexToOffset<typename InputInfo::scalar_t, int64_t>::
            NON_STRICT_CONTIGUOUS);

    T value = cfg.init_;
    if (id.glb_problem < cfg.problem_ && id.glb_batch < cfg.problem_batch_) {
      value = cfg.iinfo_.data[glb_ldr_off];
    }

    if (cfg.problem_along_x_) {
      // so far assign all work items along problem dimension
      // sg_shuffle benefits reduce on the dimension
      value = group_x_scan<T, BinaryFunction>(item, value, slm, cfg.func_);
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

template <typename SKer, typename T>
class segment_scan {
 public:
  segment_scan(const SKer& ker, dpcpp_local_acc_t<T> shared)
      : ker_(ker), shared_(shared) {}

  void operator()(DPCPP::nd_item<2> item) const {
    ker_.srun(item, shared_);
  }

 private:
  SKer ker_;
  dpcpp_local_acc_t<T> shared_;
};

template <typename SSConfig>
static inline void launch_segment_scan(const SSConfig& cfg) {
  auto& queue = dpcppGetCurrentQueue();

  auto cgf = DPCPP_Q_CGF(__cgh) {
    // SLM for sub group cumsum/cumprod in one workgroup. Allow that
    // min sub_group_size is 8, herein we use wg_size / 8 to allocate
    // enough share local memory for simd32, simd16 and simd8.
    int carrier_size = (cfg.problem_wg_range_ + MIN_SG_SIZE - 1) / MIN_SG_SIZE;
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
        typename SSConfig::func_t>
        ker(cfg);
    segment_scan gscan(ker, shared);
    __cgh.parallel_for(
        DPCPP::nd_range<2>(cfg.global_size(), cfg.group_size()), gscan);
  };
  DPCPP_Q_SUBMIT(queue, cgf);
}

template <class SSConfig>
static inline void accumulate_carrier(const SSConfig& cfg) {
  TORCH_CHECK(
      cfg.carrier_ != nullptr, "scan: nullptr carrier in accumulation ...");
  auto& queue = dpcppGetCurrentQueue();

  auto cgf = DPCPP_Q_CGF(__cgh) {
    auto kfn = DPCPP_Q_KFN(DPCPP::nd_item<2> item) {
      auto id = cfg.get_item_desc(item);
      int64_t si, pi, bi, glb_off, crr_off;

      si = id.glb_batch % cfg.stride_;
      bi = id.glb_batch / cfg.stride_;
      pi = id.chunk * id.chunk_size + id.chunk_off;
      glb_off = si + pi * cfg.stride_ + bi * cfg.problem_ * cfg.stride_;
      crr_off = si + id.chunk * cfg.stride_ + bi * id.chunk_num * cfg.stride_;

      if (id.glb_problem < cfg.problem_ && id.glb_batch < cfg.problem_batch_) {
        cfg.oinfo_.data[glb_off] =
            cfg.func_(cfg.oinfo_.data[glb_off], cfg.carrier_[crr_off]);
      }
    };
    __cgh.parallel_for(
        DPCPP::nd_range<2>(cfg.global_size(), cfg.group_size()), kfn);
  };
  DPCPP_Q_SUBMIT(queue, cgf);
}

template <typename T, class BinaryOp, int Power2ScanSize>
DPCPP_DEVICE void inclusivePrefixScan(
    T* smem,
    BinaryOp binop,
    const DPCPP::nd_item<1>& item_id) {
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
DPCPP_DEVICE void inclusivePrefixScan(
    const dpcpp_local_acc_t<T>& smem,
    T in,
    T* out,
    BinaryFunction binop,
    const DPCPP::nd_item<1>& item_id) {
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
DPCPP_DEVICE void exclusivePrefixScan(
    const dpcpp_local_acc_t<T>& smem,
    T in,
    T* out,
    T* carry,
    BinaryFunction binop,
    const DPCPP::nd_item<1>& item_id) {
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
