#include <ATen/ATen.h>
#include <ATen/Functions.h>
#include <ATen/native/Activation.h>
#include <ATen/record_function.h>
#include <torch/library.h>

#include <core/Memory.h>
#include <runtime/Utils.h>
#include <utils/DPCPP.h>

#include "RandomEngine.h"
#include "comm/ATDispatch.h"
#include "comm/AccumulateType.h"
#include "comm/ApplyUtils.h"
#include "comm/Numerics.h"
#include "comm/TensorOptions.h"

#include <aten/operators/MemoryAccess.h>
#include "utils/CustomOperatorRegistration.h"

using namespace xpu::dpcpp;
using namespace xpu::dpcpp::detail;

namespace at {
namespace AtenIpexTypeXPU {

constexpr int items_per_group = sizeof(unsigned long long) * 8;

namespace impl {

template <typename T>
inline T nms_max(const T a, const T b) {
  return a > b ? a : b;
}

template <typename T>
inline T nms_min(const T a, const T b) {
  return a < b ? a : b;
}

template <typename T>
inline bool devIoU(T const* const a, T const* const b, const float threshold) {
  T left = nms_max(a[0], b[0]), right = nms_min(a[2], b[2]);
  T top = nms_max(a[1], b[1]), bottom = nms_min(a[3], b[3]);
  T width = nms_max(right - left, (T)0), height = nms_max(bottom - top, (T)0);
  using acc_T = acc_type<T>;
  acc_T interS = (acc_T)width * height;
  acc_T Sa = ((acc_T)a[2] - a[0]) * (a[3] - a[1]);
  acc_T Sb = ((acc_T)b[2] - b[0]) * (b[3] - b[1]);
  return (interS / (Sa + Sb - interS)) > threshold;
}

template <typename T, typename accT>
void nms_kernel_impl(
    sycl::nd_item<2>& item,
    accT acc,
    int n_boxes,
    float iou_threshold,
    const T* dev_boxes,
    unsigned long long* dev_mask) {
  int row_start = item.get_group(0);
  int col_start = item.get_group(1);

  if (row_start > col_start)
    return;

  const int row_size =
      nms_min(n_boxes - row_start * items_per_group, items_per_group);
  const int col_size =
      nms_min(n_boxes - col_start * items_per_group, items_per_group);

  auto block_boxes = (T*)acc.get_pointer().get(); // items_per_group * 4
  if (item.get_local_id(1) < col_size) {
    block_boxes[item.get_local_id(1) * 4 + 0] =
        dev_boxes[(items_per_group * col_start + item.get_local_id(1)) * 4 + 0];
    block_boxes[item.get_local_id(1) * 4 + 1] =
        dev_boxes[(items_per_group * col_start + item.get_local_id(1)) * 4 + 1];
    block_boxes[item.get_local_id(1) * 4 + 2] =
        dev_boxes[(items_per_group * col_start + item.get_local_id(1)) * 4 + 2];
    block_boxes[item.get_local_id(1) * 4 + 3] =
        dev_boxes[(items_per_group * col_start + item.get_local_id(1)) * 4 + 3];
  }
  item.barrier(dpcpp_local_fence);

  if (item.get_local_id(1) < row_size) {
    const int cur_box_idx = items_per_group * row_start + item.get_local_id(1);
    const T* cur_box = dev_boxes + cur_box_idx * 4;
    int i = 0;
    unsigned long long t = 0;
    int start = 0;
    if (row_start == col_start) {
      start = item.get_local_id(1) + 1;
    }
    for (i = start; i < col_size; i++) {
      if (devIoU<T>(cur_box, block_boxes + i * 4, iou_threshold)) {
        t |= 1ULL << i;
      }
    }
    const int col_blocks = (n_boxes + items_per_group - 1) / items_per_group;
    dev_mask[cur_box_idx * col_blocks + col_start] = t;
  }
}

} // namespace impl

at::Tensor nms_kernel(
    const at::Tensor& dets,
    const at::Tensor& scores,
    double iou_threshold_) {
  float iou_threshold = (float)iou_threshold_;
  TORCH_CHECK(dets.is_xpu(), "dets must be a XPU tensor");
  TORCH_CHECK(scores.is_xpu(), "scores must be a XPU tensor");

  TORCH_CHECK(
      dets.dim() == 2, "boxes should be a 2d tensor, got ", dets.dim(), "D");
  TORCH_CHECK(
      dets.size(1) == 4,
      "boxes should have 4 elements in dimension 1, got ",
      dets.size(1));
  TORCH_CHECK(
      scores.dim() == 1,
      "scores should be a 1d tensor, got ",
      scores.dim(),
      "D");
  TORCH_CHECK(
      dets.size(0) == scores.size(0),
      "boxes and scores should have same number of elements in ",
      "dimension 0, got ",
      dets.size(0),
      " and ",
      scores.size(0))

  if (dets.numel() == 0) {
    return at::empty({0}, dets.options().dtype(at::kLong));
  }

  auto order_t = std::get<1>(
      scores.sort(/*stable=*/true, /*dim=*/0, /* descending=*/true));
  auto dets_sorted = dets.index_select(0, order_t).contiguous();

  int dets_num = dets.size(0);

  int col_blocks = (dets_num + items_per_group - 1) / items_per_group;

  at::Tensor mask =
      at::empty({dets_num * col_blocks}, dets.options().dtype(at::kLong));

  auto& dpcpp_queue = dpcppGetCurrentQueue();
  IPEX_DISPATCH_FLOATING_TYPES_AND_HALF(
      dets_sorted.scalar_type(), "nms_kernel", [&] {
        auto cgf = DPCPP_Q_CGF(cgh) {
          auto dets_sorted_ptr = (scalar_t*)dets_sorted.data_ptr();
          auto mask_ptr = (unsigned long long*)mask.data_ptr();
          auto slm = dpcpp_local_acc_t<float>(items_per_group * 4, cgh);
          auto kfn = DPCPP_Q_KFN(sycl::nd_item<2> item) {
            impl::nms_kernel_impl<scalar_t>(
                item, slm, dets_num, iou_threshold, dets_sorted_ptr, mask_ptr);
          };
          cgh.parallel_for(
              sycl::nd_range<2>(
                  sycl::range<2>(col_blocks, col_blocks * items_per_group),
                  sycl::range<2>(1, items_per_group)),
              kfn);
        };
        DPCPP_Q_SUBMIT(dpcpp_queue, cgf);
      });

  at::Tensor mask_cpu = mask.to(at::kCPU);
  unsigned long long* mask_host = (unsigned long long*)mask_cpu.data_ptr();

  std::vector<unsigned long long> remv(col_blocks);
  memset(&remv[0], 0, sizeof(unsigned long long) * col_blocks);

  at::Tensor keep =
      at::empty({dets_num}, dets.options().dtype(at::kLong).device(at::kCPU));
  int64_t* keep_out = (int64_t*)keep.data_ptr();

  int num_to_keep = 0;
  for (int i = 0; i < dets_num; i++) {
    int nblock = i / items_per_group;
    int inblock = i % items_per_group;

    if (!(remv[nblock] & (1ULL << inblock))) {
      keep_out[num_to_keep++] = i;
      unsigned long long* p = mask_host + i * col_blocks;
      for (int j = nblock; j < col_blocks; j++) {
        remv[j] |= p[j];
      }
    }
  }

  return order_t.index(
      {keep.narrow(/*dim=*/0, /*start=*/0, /*length=*/num_to_keep)
           .to(order_t.device(), keep.scalar_type())});
}

} // namespace AtenIpexTypeXPU
} // namespace at

namespace {
IPEX_LIBRARY_FRAGMENT() {
  IPEX_OP_REGISTER("nms.xpu", at::AtenIpexTypeXPU::nms_kernel);
}
} // namespace
