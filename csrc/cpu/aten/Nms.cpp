// Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#include "Nms.h"
#include <ATen/Parallel.h>
#include <c10/util/Exception.h>
#include <immintrin.h>
#include <torch/csrc/autograd/function.h>
#include <algorithm>
#include "autocast/autocast_mode.h"
#include "cpu/kernels/Softmax.h"

namespace torch_ipex {
namespace cpu {

IPEX_DEFINE_DISPATCH(nms_cpu_kernel_stub);
IPEX_DEFINE_DISPATCH(batch_score_nms_cpu_kernel_stub);
IPEX_DEFINE_DISPATCH(rpn_nms_cpu_kernel_stub);
IPEX_DEFINE_DISPATCH(box_head_nms_cpu_kernel_stub);

} // namespace cpu
} // namespace torch_ipex

namespace torch_ipex {

at::Tensor nms(
    const at::Tensor& dets,
    const at::Tensor& scores,
    const double threshold,
    const bool sorted) {
#if defined(IPEX_DISP_OP)
  printf("IpexExternal::nms\n");
#endif
  RECORD_FUNCTION("IpexExternal::nms", c10::ArrayRef<c10::IValue>({}));

  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(dets.layout() == c10::kStrided);
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(scores.layout() == c10::kStrided);

  // pointer to cpu::nms_cpu_kernel_impl(dets, scores, threshold, sorted);
  auto&& result =
      cpu::nms_cpu_kernel_stub(kCPU, dets, scores, threshold, sorted);

  static_cast<void>(result); // Avoid warnings in case not used
  return result;
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> batch_score_nms(
    const at::Tensor& dets,
    const at::Tensor& scores,
    const double threshold,
    const int64_t max_output) {
#if defined(IPEX_DISP_OP)
  printf("IpexExternal::batch_score_nms\n");
#endif
  RECORD_FUNCTION(
      "IpexExternal::batch_score_nms", c10::ArrayRef<c10::IValue>({}));

  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(dets.layout() == c10::kStrided);
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(scores.layout() == c10::kStrided);

  /*
  pointer to cpu::batch_score_nms_cpu_kernel_impl(dets, scores, threshold,
  max_output);
  */
  auto&& result = cpu::batch_score_nms_cpu_kernel_stub(
      kCPU, dets, scores, threshold, max_output);

  static_cast<void>(result); // Avoid warnings in case not used
  return result;
}

std::tuple<std::vector<at::Tensor>, std::vector<at::Tensor>> rpn_nms(
    const at::Tensor& batch_dets,
    const at::Tensor& batch_scores,
    const std::vector<std::tuple<int64_t, int64_t>>& image_shapes,
    const int64_t min_size,
    const double threshold,
    const int64_t max_output) {
#if defined(IPEX_DISP_OP)
  printf("IpexExternal::rpn_nms\n");
#endif
  RECORD_FUNCTION("IpexExternal::rpn_nms", c10::ArrayRef<c10::IValue>({}));

  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(batch_dets.layout() == c10::kStrided);
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(batch_scores.layout() == c10::kStrided);

  /*
  pointer to cpu::rpn_nms_cpu_kernel_impl(
      batch_dets, batch_scores, image_shapes, min_size, threshold, max_output);
  */
  auto&& result = cpu::rpn_nms_cpu_kernel_stub(
      kCPU,
      batch_dets,
      batch_scores,
      image_shapes,
      min_size,
      threshold,
      max_output);

  static_cast<void>(result); // Avoid warnings in case not used
  return result;
}

std::tuple<
    std::vector<at::Tensor>,
    std::vector<at::Tensor>,
    std::vector<at::Tensor>>
box_head_nms(
    const std::vector<at::Tensor>& batch_bboxes,
    const std::vector<at::Tensor>& batch_scores,
    const std::vector<std::tuple<int64_t, int64_t>>& image_shapes,
    const double score_thresh,
    const double threshold,
    const int64_t detections_per_img,
    const int64_t num_classes) {
#if defined(IPEX_DISP_OP)
  printf("IpexExternal::box_head_nms\n");
#endif
  RECORD_FUNCTION("IpexExternal::box_head_nms", c10::ArrayRef<c10::IValue>({}));

  /*
  pointer to cpu::box_head_nms_cpu_kernel_impl(
      batch_bboxes,
      batch_scores,
      image_shapes,
      score_thresh,
      threshold,
      detections_per_img,
      num_classes);
  */
  auto&& result = cpu::box_head_nms_cpu_kernel_stub(
      kCPU,
      batch_bboxes,
      batch_scores,
      image_shapes,
      score_thresh,
      threshold,
      detections_per_img,
      num_classes);

  static_cast<void>(result); // Avoid warnings in case not used
  return result;
}

template <typename scalar_t>
at::Tensor scale_back_batch_kernel(
    const at::Tensor& _ipex_bboxes_in,
    const at::Tensor& _ipex_dboxes_xywh,
    const float scale_xy,
    const float scale_wh) {
  //_ipex_bboxes_in: [BS, number_boxes, 4], for example: [1, 15130, 4]
  auto _ipex_bboxes_in_conti = _ipex_bboxes_in.contiguous();
  auto _ipex_dboxes_xywh_conti = _ipex_dboxes_xywh.contiguous();
  int64_t batch_size = _ipex_bboxes_in.size(0);
  int64_t boxes_per_image = _ipex_bboxes_in.size(1);
  int64_t ndets = batch_size * boxes_per_image; // batchsize * boxes per image
  at::Tensor output = at::empty(
      {_ipex_bboxes_in.size(0),
       _ipex_bboxes_in.size(1),
       _ipex_bboxes_in.size(2)},
      _ipex_bboxes_in.options());
  auto output_conti = output.contiguous();

  auto* input_data = _ipex_bboxes_in_conti.data_ptr<scalar_t>();
  auto* output_data = output_conti.data_ptr<scalar_t>();
  auto* input_dboxes_xywh_data = _ipex_dboxes_xywh_conti.data_ptr<double>();

#ifdef _OPENMP
#if (_OPENMP >= 201307)
#pragma omp parallel for simd schedule( \
    static) if (omp_get_max_threads() > 1 && !omp_in_parallel())
#else
#pragma omp parallel for schedule( \
    static) if (omp_get_max_threads() > 1 && !omp_in_parallel())
#endif
#endif
  for (int64_t k = 0; k < ndets; k++) {
    int64_t i = k / boxes_per_image;
    int64_t j = k % boxes_per_image;

    int64_t index = i * boxes_per_image * 4 + j * 4;

    // bboxes_in[:, :, :2] = self.scale_xy*bboxes_in[:, :, :2]
    output_data[index] = input_data[index] * scale_xy;
    output_data[index + 1] = input_data[index + 1] * scale_xy;
    // bboxes_in[:, :, 2:] = self.scale_wh*bboxes_in[:, :, 2:]
    output_data[index + 2] = input_data[index + 2] * scale_wh;
    output_data[index + 3] = input_data[index + 3] * scale_wh;

    int64_t index_dboxes_xywh = j * 4;
    // bboxes_in[:, :, :2] = bboxes_in[:, :, :2]*self.dboxes_xywh[:, :, 2:] +
    // self.dboxes_xywh[:, :, :2]
    output_data[index] =
        output_data[index] * input_dboxes_xywh_data[index_dboxes_xywh + 2] +
        input_dboxes_xywh_data[index_dboxes_xywh];
    output_data[index + 1] =
        output_data[index + 1] * input_dboxes_xywh_data[index_dboxes_xywh + 3] +
        input_dboxes_xywh_data[index_dboxes_xywh + 1];
    // bboxes_in[:, :, 2:] = bboxes_in[:, :, 2:].exp()*self.dboxes_xywh[:, :,
    // 2:]
    output_data[index + 2] = exp(output_data[index + 2]) *
        input_dboxes_xywh_data[index_dboxes_xywh + 2];
    output_data[index + 3] = exp(output_data[index + 3]) *
        input_dboxes_xywh_data[index_dboxes_xywh + 3];

    /*
    # Transform format to ltrb
    l, t, r, b = bboxes_in[:, :, 0] - 0.5*bboxes_in[:, :, 2],\
                  bboxes_in[:, :, 1] - 0.5*bboxes_in[:, :, 3],\
                  bboxes_in[:, :, 0] + 0.5*bboxes_in[:, :, 2],\
                  bboxes_in[:, :, 1] + 0.5*bboxes_in[:, :, 3]

    bboxes_in[:, :, 0] = l
    bboxes_in[:, :, 1] = t
    bboxes_in[:, :, 2] = r
    bboxes_in[:, :, 3] = b
    */

    auto l = output_data[index] - 0.5 * output_data[index + 2];
    auto t = output_data[index + 1] - 0.5 * output_data[index + 3];
    auto r = output_data[index] + 0.5 * output_data[index + 2];
    auto b = output_data[index + 1] + 0.5 * output_data[index + 3];
    output_data[index] = l;
    output_data[index + 1] = t;
    output_data[index + 2] = r;
    output_data[index + 3] = b;
  }
  return output;
}

std::tuple<at::Tensor, at::Tensor> parallel_scale_back_batch(
    const at::Tensor& bboxes_in,
    const at::Tensor& scores_in,
    const at::Tensor& dboxes_xywh,
    const double scale_xy,
    const double scale_wh) {
#if defined(IPEX_DISP_OP)
  printf("IpexExternal::parallel_scale_back_batch\n");
#endif
  RECORD_FUNCTION(
      "IpexExternal::parallel_scale_back_batch",
      c10::ArrayRef<c10::IValue>({}));

  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(bboxes_in.layout() == c10::kStrided);
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(dboxes_xywh.layout() == c10::kStrided);

  at::Tensor bbox_result;
  AT_DISPATCH_FLOATING_TYPES(bboxes_in.scalar_type(), "scale_back_batch", [&] {
    bbox_result = scale_back_batch_kernel<scalar_t>(
        bboxes_in, dboxes_xywh, scale_xy, scale_wh);
  });

  auto&& scores_result = torch_ipex::cpu::dil_softmax(scores_in, -1);

  return std::tuple<at::Tensor, at::Tensor>(bbox_result, scores_result);
}
} // namespace torch_ipex

namespace {
static auto dispatch =
    torch::RegisterOperators()
        .op("torch_ipex::nms", &torch_ipex::nms)
        .op("torch_ipex::batch_score_nms", &torch_ipex::batch_score_nms)
        .op("torch_ipex::rpn_nms", &torch_ipex::rpn_nms)
        .op("torch_ipex::box_head_nms", &torch_ipex::box_head_nms)
        .op("torch_ipex::parallel_scale_back_batch",
            &torch_ipex::parallel_scale_back_batch);
}

namespace torch_ipex {
namespace autocast {
at::Tensor nms(
    const at::Tensor& dets,
    const at::Tensor& scores,
    const double threshold,
    const bool sorted) {
  c10::impl::ExcludeDispatchKeyGuard no_autocastCPU(DispatchKey::AutocastCPU);
  static auto op = torch::Dispatcher::singleton()
                       .findSchemaOrThrow("torch_ipex::nms", "")
                       .typed<decltype(nms)>();
  return op.call(
      cpu_cached_cast(at::kFloat, dets),
      cpu_cached_cast(at::kFloat, scores),
      threshold,
      sorted);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> batch_score_nms(
    const at::Tensor& dets,
    const at::Tensor& scores,
    const double threshold,
    const int64_t max_output) {
  c10::impl::ExcludeDispatchKeyGuard no_autocastCPU(DispatchKey::AutocastCPU);
  static auto op = torch::Dispatcher::singleton()
                       .findSchemaOrThrow("torch_ipex::batch_score_nms", "")
                       .typed<decltype(batch_score_nms)>();
  return op.call(
      cpu_cached_cast(at::kFloat, dets),
      cpu_cached_cast(at::kFloat, scores),
      threshold,
      max_output);
}

std::tuple<std::vector<at::Tensor>, std::vector<at::Tensor>> rpn_nms(
    const at::Tensor& batch_dets,
    const at::Tensor& batch_scores,
    const std::vector<std::tuple<int64_t, int64_t>>& image_shapes,
    const int64_t min_size,
    const double threshold,
    const int64_t max_output) {
  c10::impl::ExcludeDispatchKeyGuard no_autocastCPU(DispatchKey::AutocastCPU);
  static auto op = torch::Dispatcher::singleton()
                       .findSchemaOrThrow("torch_ipex::rpn_nms", "")
                       .typed<decltype(rpn_nms)>();
  return op.call(
      cpu_cached_cast(at::kFloat, batch_dets),
      cpu_cached_cast(at::kFloat, batch_scores),
      image_shapes,
      min_size,
      threshold,
      max_output);
}

std::tuple<
    std::vector<at::Tensor>,
    std::vector<at::Tensor>,
    std::vector<at::Tensor>>
box_head_nms(
    const std::vector<at::Tensor>& batch_bboxes,
    const std::vector<at::Tensor>& batch_scores,
    const std::vector<std::tuple<int64_t, int64_t>>& image_shapes,
    const double score_thresh,
    const double threshold,
    const int64_t detections_per_img,
    const int64_t num_classes) {
  c10::impl::ExcludeDispatchKeyGuard no_autocastCPU(DispatchKey::AutocastCPU);
  static auto op = torch::Dispatcher::singleton()
                       .findSchemaOrThrow("torch_ipex::box_head_nms", "")
                       .typed<decltype(box_head_nms)>();
  return op.call(
      cpu_cached_cast(at::kFloat, batch_bboxes),
      cpu_cached_cast(at::kFloat, batch_scores),
      image_shapes,
      score_thresh,
      threshold,
      detections_per_img,
      num_classes);
}

std::tuple<at::Tensor, at::Tensor> parallel_scale_back_batch(
    const at::Tensor& bboxes_in,
    const at::Tensor& scores_in,
    const at::Tensor& dboxes_xywh,
    const double scale_xy,
    const double scale_wh) {
  c10::impl::ExcludeDispatchKeyGuard no_autocastCPU(DispatchKey::AutocastCPU);
  static auto op =
      torch::Dispatcher::singleton()
          .findSchemaOrThrow("torch_ipex::parallel_scale_back_batch", "")
          .typed<decltype(parallel_scale_back_batch)>();
  return op.call(
      cpu_cached_cast(at::kFloat, bboxes_in),
      cpu_cached_cast(at::kFloat, scores_in),
      cpu_cached_cast(at::kFloat, dboxes_xywh),
      scale_xy,
      scale_wh);
}

TORCH_LIBRARY_IMPL(torch_ipex, AutocastCPU, m) {
  m.impl("nms", torch_ipex::autocast::nms);
  m.impl("batch_score_nms", torch_ipex::autocast::batch_score_nms);
  m.impl("rpn_nms", torch_ipex::autocast::rpn_nms);
  m.impl("box_head_nms", torch_ipex::autocast::box_head_nms);
  m.impl(
      "parallel_scale_back_batch",
      torch_ipex::autocast::parallel_scale_back_batch);
}

} // namespace autocast
} // namespace torch_ipex
