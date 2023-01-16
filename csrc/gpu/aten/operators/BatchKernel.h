#pragma once
#include <runtime/Utils.h>
#include <utils/DPCPP.h>

using namespace xpu::dpcpp;

class BatchKernelConfig {
 public:
  enum class Policy : uint8_t {
    pSegment = 0,
    pLoop,
    pAdaptive,
    /* pVector */
  };

 public:
  BatchKernelConfig() = delete;
  BatchKernelConfig(
      int64_t batch,
      int64_t problem,
      int64_t stride,
      int64_t problem_batch,
      bool problem_along_x,
      Policy policy = Policy::pSegment)
      : batch_(batch),
        problem_(problem),
        stride_(stride),
        problem_batch_(problem_batch),
        problem_along_x_(problem_along_x),
        policy_(policy),
        problem_wg_range_(0),
        problem_glb_range_(0),
        problem_range_(0),
        batch_glb_range_(0),
        batch_range_(0),
        glb_range_x_(0),
        glb_range_y_(0),
        wg_range_x_(0),
        wg_range_y_(0) {
    int64_t wg_size = dpcppMaxWorkGroupSize();
    int64_t sg_size = dpcppMaxSubGroupSize();
    wg_range_x_ = sg_size;
    wg_range_y_ = wg_size / wg_range_x_;

    if (problem_batch_ == 0)
      problem_batch_ = batch_ * stride_;

    // ensure assigning successive work items along contiguous (small stride)
    // dimension
    auto range_bound_x = problem_along_x_ ? problem_ : problem_batch_;
    auto range_bound_y = problem_along_x_ ? problem_batch_ : problem_;
    while (range_bound_y <= wg_range_y_ >> 1 && wg_range_x_ <= wg_size) {
      wg_range_y_ = wg_range_y_ >> 1;
      wg_range_x_ = wg_size / wg_range_y_;
    }

    while (range_bound_x <= wg_range_x_ >> 1 && sg_size <= wg_range_x_ >> 1) {
      wg_range_x_ = wg_range_x_ >> 1;
    }

    if (policy_ == Policy::pAdaptive) {
      int64_t target_glb_range = dpcppMaxWorkItemsPerTile() /
          (wg_range_x_ * wg_range_y_) * (wg_range_x_ * wg_range_y_);
      if (problem_along_x_) {
        glb_range_y_ = wg_range_y_;

        glb_range_x_ = target_glb_range / glb_range_y_;
        if (glb_range_x_ > range_bound_x) {
          auto wg_num_x = std::max<int64_t>(range_bound_x / wg_range_x_, 1);
          glb_range_x_ = wg_range_x_ * wg_num_x;
        }

        glb_range_y_ = target_glb_range / glb_range_x_;
        if (glb_range_y_ > range_bound_y) {
          auto wg_num_y = std::max<int64_t>(range_bound_y / wg_range_y_, 1);
          glb_range_y_ = wg_range_y_ * wg_num_y;
        }
      } else {
        glb_range_x_ = wg_range_x_;

        glb_range_y_ = target_glb_range / glb_range_x_;
        if (glb_range_y_ > range_bound_y) {
          auto wg_num_y = std::max<int64_t>(range_bound_y / wg_range_y_, 1);
          glb_range_y_ = wg_range_y_ * wg_num_y;
        }

        glb_range_x_ = target_glb_range / glb_range_y_;
        if (glb_range_x_ > range_bound_x) {
          auto wg_num_x = std::max<int64_t>(range_bound_x / wg_range_x_, 1);
          glb_range_x_ = wg_range_x_ * wg_num_x;
        }
      }
    } else {
      if (problem_along_x_) {
        glb_range_x_ = policy_ == Policy::pLoop
            ? wg_range_x_
            : int64_t((problem_ + wg_range_x_ - 1) / wg_range_x_) * wg_range_x_;
        glb_range_y_ =
            int64_t((problem_batch_ + wg_range_y_ - 1) / wg_range_y_) *
            wg_range_y_;
      } else {
        glb_range_x_ =
            int64_t((problem_batch_ + wg_range_x_ - 1) / wg_range_x_) *
            wg_range_x_;
        glb_range_y_ = policy_ == Policy::pLoop
            ? wg_range_y_
            : int64_t((problem_ + wg_range_y_ - 1) / wg_range_y_) * wg_range_y_;
      }
    }

    problem_wg_range_ = problem_along_x_ ? wg_range_x_ : wg_range_y_;
    problem_glb_range_ = problem_along_x_ ? glb_range_x_ : glb_range_y_;
    batch_glb_range_ = problem_along_x_ ? glb_range_y_ : glb_range_x_;
    problem_range_ = problem_along_x_
        ? (problem_ + glb_range_x_ - 1) / glb_range_x_ * glb_range_x_
        : (problem_ + glb_range_y_ - 1) / glb_range_y_ * glb_range_y_;
    batch_range_ = problem_along_x_
        ? (problem_batch_ + glb_range_y_ - 1) / glb_range_y_ * glb_range_y_
        : (problem_batch_ + glb_range_x_ - 1) / glb_range_x_ * glb_range_x_;
  }

  sycl::range<2> global_size() const {
    return {glb_range_y_, glb_range_x_};
  }

  sycl::range<2> group_size() const {
    return {wg_range_y_, wg_range_x_};
  }

  struct ItemDesc {
    /* chunk id along problem dim */ int64_t chunk;
    /* problem chunk size */ int64_t chunk_size;
    /* offsite in current chunk */ int64_t chunk_off;
    /* how many active chunks along problem dim */ int64_t chunk_num;
    /* global batch id */ int64_t glb_batch;
    /* global problem id */ int64_t glb_problem;
  };

  DPCPP_DEVICE ItemDesc get_item_desc(sycl::nd_item<2> item) const {
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

    // ItemDesc::glb_problem is meaningless, if policy is loop for all.
    if (problem_along_x_) {
      return {gx, lrx, lix, wgrx, giy, gix};
    } else {
      return {gy, lry, liy, wgry, gix, giy};
    }
  }

  // iterate over problems and batchs for `pAdaptive` policy
  // # update workload status inplace in `desc`.
  // # prioritize problem iteration.
  DPCPP_DEVICE bool next(sycl::nd_item<2> item, ItemDesc& desc) const {
    auto next_problem = desc.glb_problem + problem_glb_range_;
    auto next_batch = desc.glb_batch + batch_glb_range_;
    auto cur_chunk = desc.chunk;

    // WA: break deduce chain, or offline compiler gets crash, due to,
    // massive and deep divergence level
    desc = get_item_desc(item);

    // iterate over problem
    if (next_problem < problem_range_) {
      desc.glb_problem = next_problem;
      desc.chunk = cur_chunk + desc.chunk_num;
      return true;
    }

    // iterate over batch
    if (next_batch < batch_range_) {
      desc.glb_batch = next_batch;
      return true;
    }

    return false;
  }

  static Policy suggest_policy(
      int64_t batch,
      int64_t problem,
      int64_t stride,
      bool problem_along_x,
      bool bypass_adaptive_policy = true) {
    auto target_wi_num = dpcppMaxWorkItemsPerTile();

    if (!bypass_adaptive_policy && batch * problem * stride >= target_wi_num) {
      return Policy::pAdaptive;
    }

    BatchKernelConfig cfg_ = {
        batch, problem, stride, batch * stride, problem_along_x, Policy::pLoop};
    int64_t wg_num = (cfg_.glb_range_x_ / cfg_.wg_range_x_) *
        (cfg_.glb_range_y_ / cfg_.wg_range_y_);
    int64_t wg_size = cfg_.wg_range_x_ * cfg_.wg_range_y_;

    if (wg_size * (wg_num + 1) > target_wi_num) {
      return Policy::pLoop;
    }

    return Policy::pSegment;
  }

 public:
  /* logical shape desc */ int64_t batch_;
  /* logical shape desc */ int64_t problem_;
  /* logical shape desc */ int64_t stride_;
  /* logical active batch */ int64_t problem_batch_;
  bool problem_along_x_;
  Policy policy_;
  int problem_wg_range_;
  int problem_glb_range_;
  int problem_range_;
  int batch_glb_range_;
  int batch_range_;
  int glb_range_x_;
  int glb_range_y_;
  int wg_range_x_;
  int wg_range_y_;
};
