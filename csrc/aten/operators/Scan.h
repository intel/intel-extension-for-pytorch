#pragma once
#include <core/detail/TensorInfo.h>
#include <utils/DPCPP.h>
#include "BatchKernel.h"

using namespace xpu::dpcpp::detail;

typedef enum {
  EXCLUSIVE_TYPE = 0,
  INCLUSIVE_TYPE = 1,
} ScanType;

template <class InputInfo, class OutputInfo, typename T, class BinaryFunction>
class ScanConfig : public BatchKernelConfig {
 public:
  using arg_t = T;
  using func_t = BinaryFunction;
  using InputInfoType = InputInfo;
  using OutputInfoType = OutputInfo;

  ScanConfig() = delete;
  ScanConfig(
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

  static ScanConfig<InputInfo, OutputInfo, T, BinaryFunction> make_config(
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

template <class SConfig, class T, class BinaryFunction>
class group_scan_kernel {
 public:
  using InputInfo = typename SConfig::InputInfoType;
  using OutputInfo = typename SConfig::OutputInfoType;

  group_scan_kernel(const SConfig& cfg) : cfg(cfg) {}

 public:
  DPCPP_DEVICE T
  subgroup_scan(DPCPP::nd_item<2> item, T value, BinaryFunction func) const {
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

  DPCPP_DEVICE T group_x_scan(
      DPCPP::nd_item<2> item,
      T value,
#ifndef SG_SCAN
      dpcpp_local_ptr<T> temp,
#else
      dpcpp_local_ptr<T> carrier,
#endif
      BinaryFunction func) const {
#ifndef SG_SCAN
    const auto lix = item.get_local_id(1);
    const auto liy = item.get_local_id(0);
    const auto rx = item.get_local_range(1);
    const auto ry = item.get_local_range(0);

    temp[liy * rx + lix] = value;
    for (int offset = 1; offset < rx; offset <<= 1) {
      item.barrier(dpcpp_local_fence);
      if (lix >= offset)
        value = func(temp[liy * rx + (lix - offset)], temp[liy * rx + lix]);
      item.barrier(dpcpp_local_fence);

      if (lix >= offset)
        temp[liy * rx + lix] = value;
    }
#else
    const auto wg_size = item.get_local_range(1);
    const auto sg = item.get_sub_group();
    const auto sgi = sg.get_group_linear_id();
    const auto sg_size = sg.get_local_range()[0];
    const auto lane = sg.get_local_linear_id();
    const auto sg_num = (wg_size + sg_size - 1) / sg_size;

    value = subgroup_scan(item, value, func);
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

  DPCPP_DEVICE T group_y_scan(
      DPCPP::nd_item<2> item,
      T value,
      dpcpp_local_ptr<T> temp,
      BinaryFunction func) const {
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

  DPCPP_DEVICE void run(DPCPP::nd_item<2> item, dpcpp_local_acc_t<T> slm)
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
      value = group_x_scan(item, value, slm, cfg.func_);
    } else {
      // parallel prefix reduce
      value = group_y_scan(item, value, slm, cfg.func_);
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
  SConfig cfg;
};

template <typename SKer, typename T>
class group_scan {
 public:
  group_scan(const SKer& ker, dpcpp_local_acc_t<T> shared)
      : ker_(ker), shared_(shared) {}

  void operator()(DPCPP::nd_item<2> item) const {
    ker_.run(item, shared_);
  }

 private:
  SKer ker_;
  dpcpp_local_acc_t<T> shared_;
};

template <typename SConfig>
static inline void launch_group_scan(const SConfig& cfg) {
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
    dpcpp_local_acc_t<typename SConfig::arg_t> shared(slm_size, __cgh);
    group_scan_kernel<
        SConfig,
        typename SConfig::arg_t,
        typename SConfig::func_t>
        ker(cfg);
    group_scan gscan(ker, shared);
    __cgh.parallel_for(
        DPCPP::nd_range<2>(cfg.global_size(), cfg.group_size()), gscan);
  };
  DPCPP_Q_SUBMIT(queue, cgf);
}

template <class SConfig>
static inline void accumulate_carrier(const SConfig& cfg) {
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
