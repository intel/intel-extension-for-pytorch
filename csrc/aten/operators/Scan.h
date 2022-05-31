#include <utils/DPCPP.h>

// Collection of in-kernel scan / prefix sum utilities

// Inclusive Scan via an upsweep/downsweep mechanism. Assumes:
//
// 1. Power2ScanSize is a power of 2. This code still works for collections that
// do not exactly contain a power of 2 number of elements, simply round up to
// the
// nearest power of 2 and then call.
//
// 2. That there are two-elements per thread, i.e. the size of the smem storage
// is 2 * blockDim.x * sizeof(T).
//
// Consider a (+)-Scan on the following elements:
//
// Upsweep:
//
//    0  1  2  3  4  5  6  7
//       1     5     9    13
//             6          22
//                        28
//
// Downsweep:
//                  15
//         3     10    21
#define DEFAULT_SG_SIZE 32
#define MIN_SG_SIZE 8

typedef enum {
  EXCLUSIVE_TYPE = 0,
  INCLUSIVE_TYPE = 1,
} ScanType;

template <class InputIt, class OutputIt, class T, class BinaryFunction>
class scan_config {
 public:
  using arg_t = T;
  using func_t = BinaryFunction;

  scan_config() = delete;
  scan_config(
      InputIt input,
      OutputIt output,
      int64_t batch,
      int64_t problem,
      int64_t stride,
      T init,
      ScanType type,
      BinaryFunction func)
      : input_(input),
        output_(output),
        batch_(batch),
        problem_(problem),
        stride_(stride),
        init_(init),
        type_(type),
        scan_along_x_([&]() -> bool {
          if (stride == 1)
            return true;
          else
            return false;
        }()),
        func_(func),
        problem_wg_range_(0),
        problem_glb_range_(0),
        glb_range_x_(0),
        glb_range_y_(0),
        wg_range_x_(0),
        wg_range_y_(0),
        carrier_(nullptr) {
    int64_t wg_size = dpcppMaxWorkGroupSize(dpcppGetDeviceIdOfCurrentQueue());
    wg_range_x_ = DEFAULT_SG_SIZE;
    wg_range_y_ = wg_size / wg_range_x_;

    // ensure input loading along contiguous dimension
    if (!scan_along_x_) {
      while (problem_ <= wg_range_y_ >> 1 && wg_range_x_ <= wg_size) {
        wg_range_y_ = wg_range_y_ >> 1;
        wg_range_x_ = wg_size / wg_range_y_;
      }
    }

    if (scan_along_x_) {
      glb_range_x_ =
          int64_t((problem_ + wg_range_x_ - 1) / wg_range_x_) * wg_range_x_;
      glb_range_y_ =
          int64_t((batch_ * stride_ + wg_range_y_ - 1) / wg_range_y_) *
          wg_range_y_;
    } else {
      glb_range_x_ =
          int64_t((batch_ * stride_ + wg_range_x_ - 1) / wg_range_x_) *
          wg_range_x_;
      glb_range_y_ =
          int64_t((problem_ + wg_range_y_ - 1) / wg_range_y_) * wg_range_y_;
    }

    problem_wg_range_ = scan_along_x_ ? wg_range_x_ : wg_range_y_;
    problem_glb_range_ = scan_along_x_ ? glb_range_x_ : glb_range_y_;
  }

  DPCPP::range<2> global_size() const {
    return {glb_range_y_, glb_range_x_};
  }

  DPCPP::range<2> group_size() const {
    return {wg_range_y_, wg_range_x_};
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

  struct item_desc {
    /* chunk id along problem dim */ int64_t chunk;
    /* local task to scan */ int64_t chunk_size;
    /* current local assignment id */ int64_t chunk_off;
    /* how many chunks along problem dim */ int64_t chunk_num;
    /* parallel batch, not tensor batch */ int64_t glb_batch;
    /* current global assignment id */ int64_t glb_problem;
  };

  item_desc get_item_desc(DPCPP::nd_item<2> item) const {
    auto lix = item.get_local_id(1);
    auto liy = item.get_local_id(0);
    auto lrx = item.get_local_range(1);
    auto lry = item.get_local_range(0);
    auto wgrx = item.get_group_range(1);
    auto wgry = item.get_group_range(0);
    auto gix = item.get_global_id(1);
    auto giy = item.get_global_id(0);
    auto gx = item.get_group(1);
    auto gy = item.get_group(0);

    if (scan_along_x_) {
      return {gx, lrx, lix, wgrx, giy, gix};
    } else {
      return {gy, lry, liy, wgry, gix, giy};
    }
  }

 public:
  InputIt input_;
  OutputIt output_;
  int64_t batch_;
  int64_t problem_;
  int64_t stride_;
  T init_;
  ScanType type_;
  bool scan_along_x_;
  BinaryFunction func_;
  int problem_wg_range_;
  int problem_glb_range_;
  int glb_range_x_;
  int glb_range_y_;
  int wg_range_x_;
  int wg_range_y_;
  T* carrier_;
};

template <class SConfig, class T, class BinaryFunction>
class group_scan_kernel {
 public:
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
    typename SConfig::item_desc id = cfg.get_item_desc(item);
    int64_t si, pi, bi, glb_ldr_off, glb_str_off, crr_off;

    si = id.glb_batch % cfg.stride_;
    bi = id.glb_batch / cfg.stride_;
    pi = id.chunk * id.chunk_size + id.chunk_off;

    int64_t e = cfg.type_ == INCLUSIVE_TYPE ? 0 : 1;
    glb_ldr_off = si + pi * cfg.stride_ + bi * cfg.problem_ * cfg.stride_;
    glb_str_off = si + (pi + e) * cfg.stride_ + bi * cfg.problem_ * cfg.stride_;
    crr_off = si + id.chunk * cfg.stride_ + bi * id.chunk_num * cfg.stride_;

    T value = cfg.init_;
    if (id.glb_problem < cfg.problem_ &&
        id.glb_batch < cfg.batch_ * cfg.stride_) {
      value = cfg.input_[glb_ldr_off];
    }

    if (cfg.scan_along_x_) {
      // so far assign all work items along problem dimension
      // sg_shuffle benefits reduce on the dimension
      value = group_x_scan(item, value, slm, cfg.func_);
    } else {
      // parallel prefix reduce
      value = group_y_scan(item, value, slm, cfg.func_);
    }

    if (id.glb_batch < cfg.batch_ * cfg.stride_) {
      if (cfg.type_ == INCLUSIVE_TYPE) {
        if (id.glb_problem < cfg.problem_) {
          cfg.output_[glb_str_off] = value;
        }
      } else {
        if (id.glb_problem < cfg.problem_ - 1 &&
            id.chunk_off < id.chunk_size - 1) {
          cfg.output_[glb_str_off] = value;
        }
        if (id.glb_problem < cfg.problem_ && id.chunk_off == 0) {
          cfg.output_[glb_ldr_off] = cfg.init_;
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
    int slm_size = cfg.scan_along_x_ ? carrier_size : wg_size;
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
      typename SConfig::item_desc id = cfg.get_item_desc(item);
      int64_t si, pi, bi, glb_off, crr_off;

      si = id.glb_batch % cfg.stride_;
      bi = id.glb_batch / cfg.stride_;
      pi = id.chunk * id.chunk_size + id.chunk_off;
      glb_off = si + pi * cfg.stride_ + bi * cfg.problem_ * cfg.stride_;
      crr_off = si + id.chunk * cfg.stride_ + bi * id.chunk_num * cfg.stride_;

      if (id.glb_problem < cfg.problem_ &&
          id.glb_batch < cfg.batch_ * cfg.stride_) {
        cfg.output_[glb_off] =
            cfg.func_(cfg.output_[glb_off], cfg.carrier_[crr_off]);
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
