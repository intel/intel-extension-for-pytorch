#pragma once
#include <utils/DPCPP.h>

#define DEFAULT_SG_SIZE 32
#define MIN_SG_SIZE 8

class BatchKernelConfig {
 public:
  BatchKernelConfig() = delete;
  BatchKernelConfig(
      int64_t batch,
      int64_t problem,
      int64_t stride,
      int64_t problem_batch,
      bool problem_along_x)
      : batch_(batch),
        problem_(problem),
        stride_(stride),
        problem_batch_(problem_batch),
        problem_along_x_(problem_along_x),
        problem_wg_range_(0),
        problem_glb_range_(0),
        glb_range_x_(0),
        glb_range_y_(0),
        wg_range_x_(0),
        wg_range_y_(0) {
    int64_t wg_size = dpcppMaxWorkGroupSize(dpcppGetDeviceIdOfCurrentQueue());
    wg_range_x_ = DEFAULT_SG_SIZE;
    wg_range_y_ = wg_size / wg_range_x_;

    // ensure assigning successive work items along contiguous (small stride)
    // dimension
    if (!problem_along_x_) {
      while (problem_ <= wg_range_y_ >> 1 && wg_range_x_ <= wg_size) {
        wg_range_y_ = wg_range_y_ >> 1;
        wg_range_x_ = wg_size / wg_range_y_;
      }
    }

    if (problem_batch_ == 0)
      problem_batch_ = batch_ * stride_;
    if (problem_along_x_) {
      glb_range_x_ =
          int64_t((problem_ + wg_range_x_ - 1) / wg_range_x_) * wg_range_x_;
      glb_range_y_ = int64_t((problem_batch_ + wg_range_y_ - 1) / wg_range_y_) *
          wg_range_y_;
    } else {
      glb_range_x_ = int64_t((problem_batch_ + wg_range_x_ - 1) / wg_range_x_) *
          wg_range_x_;
      glb_range_y_ =
          int64_t((problem_ + wg_range_y_ - 1) / wg_range_y_) * wg_range_y_;
    }

    problem_wg_range_ = problem_along_x_ ? wg_range_x_ : wg_range_y_;
    problem_glb_range_ = problem_along_x_ ? glb_range_x_ : glb_range_y_;
  }

  DPCPP::range<2> global_size() const {
    return {glb_range_y_, glb_range_x_};
  }

  DPCPP::range<2> group_size() const {
    return {wg_range_y_, wg_range_x_};
  }

  struct item_desc {
    /* chunk id along problem dim */ int64_t chunk;
    /* problem chunk size */ int64_t chunk_size;
    /* offsite in current chunk */ int64_t chunk_off;
    /* how many chunks along problem dim */ int64_t chunk_num;
    /* global batch id */ int64_t glb_batch;
    /* global problem id */ int64_t glb_problem;
  };

  DPCPP_DEVICE item_desc get_item_desc(DPCPP::nd_item<2> item) const {
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

    if (problem_along_x_) {
      return {gx, lrx, lix, wgrx, giy, gix};
    } else {
      return {gy, lry, liy, wgry, gix, giy};
    }
  }

 public:
  /* logical shape desc */ int64_t batch_;
  /* logical shape desc */ int64_t problem_;
  /* logical shape desc */ int64_t stride_;
  /* logical active batch */ int64_t problem_batch_;
  bool problem_along_x_;
  int problem_wg_range_;
  int problem_glb_range_;
  int glb_range_x_;
  int glb_range_y_;
  int wg_range_x_;
  int wg_range_y_;
};
