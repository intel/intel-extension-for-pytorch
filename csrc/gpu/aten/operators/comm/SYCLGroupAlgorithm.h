#pragma once

#include <utils/DPCPP.h>

template <typename T, typename sg_t>
inline T GroupReduceSumSGSizeEqualstoNumSG(sg_t& sg, T val) {
  auto sg_size = sg.get_local_range()[0];
  for (int offset = (sg_size >> 1); offset > 0; offset >>= 1) {
    val += sycl::shift_group_left(sg, val, offset);
  }
  return val;
}

// function GroupReduceSumSGSizeEqualstoNumSG will firstly reduce elements in
// each subgroups, after that it will store the results of each subgroup into a
// subgroup, and reduce this subgroup for the final result. So, pls notice, when
// using this method, the maximun work_group size should be equals to sub_group
// size * sub_group size, or some element will not be calculated into the final
// result.
template <typename T, typename item_t>
inline T GroupReduceSumSGSizeEqualstoNumSG(item_t& item, T val, T* shared) {
  auto thread_idx = item.get_local_id(0);
  auto group_size = item.get_local_range(0);
  auto sg = item.get_sub_group();
  auto sg_size = sg.get_local_range()[0];
  int lid = thread_idx % sg_size;
  int wid = thread_idx / sg_size;
  val = GroupReduceSumSGSizeEqualstoNumSG(sg, val);
  item.barrier(dpcpp_local_fence);
  if (lid == 0) {
    shared[wid] = val;
  }
  item.barrier(dpcpp_local_fence);
  val = (thread_idx < group_size / sg_size) ? shared[lid] : T(0);
  if (wid == 0) {
    val = GroupReduceSumSGSizeEqualstoNumSG(sg, val);
  }
  return val;
}
