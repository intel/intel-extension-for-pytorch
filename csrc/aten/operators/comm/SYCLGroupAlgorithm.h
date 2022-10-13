#pragma once

#include <utils/DPCPP.h>

template <typename T, typename sg_t>
inline T GroupReduceSum(sg_t& sg, T val) {
  auto sg_size = sg.get_local_range()[0];
  for (int offset = (sg_size >> 1); offset > 0; offset >>= 1) {
    val += sg.shuffle_down(val, offset);
  }
  return val;
}

template <typename T, typename item_t>
inline T GroupReduceSum(item_t& item, T val, T* shared) {
  auto thread_idx = item.get_local_id(0);
  auto group_size = item.get_local_range(0);
  auto sg = item.get_sub_group();
  auto sg_size = sg.get_local_range()[0];
  int lid = thread_idx % sg_size;
  int wid = thread_idx / sg_size;
  val = GroupReduceSum(sg, val);
  item.barrier(dpcpp_local_fence);
  if (lid == 0) {
    shared[wid] = val;
  }
  item.barrier(dpcpp_local_fence);
  val = (thread_idx < group_size / sg_size) ? shared[lid] : T(0);
  if (wid == 0) {
    val = GroupReduceSum(sg, val);
  }
  return val;
}
