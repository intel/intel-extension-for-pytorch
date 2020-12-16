// Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#include "ExternalOPs.h"
#include "torch_ipex/csrc/aten_ipex_bridge.h"
#include "aten/aten.hpp"
#include <ATen/Parallel.h>
#include <algorithm>
#include <c10/util/Exception.h>
#include <torch/csrc/autograd/function.h>
#include <torch/csrc/autograd/record_function.h>
namespace torch_ipex {

/*
 When calculating the Intersection over Union:
  MaskRCNN: bias = 1
  SSD-Resnet34: bias = 0
*/
template <typename scalar_t>
at::Tensor nms_cpu_kernel(const at::Tensor& dets,
                          const at::Tensor& scores,
                          const float threshold, float bias=1) {
  AT_ASSERTM(!dets.type().is_cuda(), "dets must be a CPU tensor");
  AT_ASSERTM(!scores.type().is_cuda(), "scores must be a CPU tensor");
  AT_ASSERTM(dets.type() == scores.type(), "dets should have the same type as scores");

  if (dets.numel() == 0) {
    return at::empty({0}, dets.options().dtype(at::kLong).device(at::kCPU));
  }

  auto x1_t = dets.select(1, 0).contiguous();
  auto y1_t = dets.select(1, 1).contiguous();
  auto x2_t = dets.select(1, 2).contiguous();
  auto y2_t = dets.select(1, 3).contiguous();

  at::Tensor areas_t = (x2_t - x1_t + bias) * (y2_t - y1_t + bias);

  auto order_t = std::get<1>(scores.sort(0, /* descending=*/true));

  auto ndets = dets.size(0);
  at::Tensor suppressed_t = at::zeros({ndets}, dets.options().dtype(at::kByte).device(at::kCPU));

  auto suppressed = suppressed_t.data<uint8_t>();
  auto order = order_t.data<int64_t>();
  auto x1 = x1_t.data<scalar_t>();
  auto y1 = y1_t.data<scalar_t>();
  auto x2 = x2_t.data<scalar_t>();
  auto y2 = y2_t.data<scalar_t>();
  auto areas = areas_t.data<scalar_t>();
  for (int64_t _i = 0; _i < ndets; _i++) {
    auto i = order[_i];
    if (suppressed[i] == 1)
      continue;
    auto ix1 = x1[i];
    auto iy1 = y1[i];
    auto ix2 = x2[i];
    auto iy2 = y2[i];
    auto iarea = areas[i];

#ifdef _OPENMP
#if (_OPENMP >= 201307)
# pragma omp parallel for simd
#else
# pragma omp parallel for schedule(static)
#endif
#endif
    for (int64_t _j = _i + 1; _j < ndets; _j++) {
      auto j = order[_j];
      if (suppressed[j] == 1)
        continue;
      auto xx1 = std::max(ix1, x1[j]);
      auto yy1 = std::max(iy1, y1[j]);
      auto xx2 = std::min(ix2, x2[j]);
      auto yy2 = std::min(iy2, y2[j]);

      auto w = std::max(static_cast<scalar_t>(0), xx2 - xx1 + bias);
      auto h = std::max(static_cast<scalar_t>(0), yy2 - yy1 + bias);
      auto inter = w * h;
      auto ovr = inter / (iarea + areas[j] - inter);
      if (ovr >= threshold)
        suppressed[j] = 1;
   }
  }
  return at::nonzero(suppressed_t == 0).squeeze(1);
}

std::vector<at::Tensor> remove_empty(std::vector<at::Tensor>& candidate) {
  std::vector<at::Tensor> valid_candidate;
  for (int64_t i = 0; i < candidate.size(); i++) {
    if (candidate[i].defined()) {
      valid_candidate.push_back(candidate[i]);
    }
  }
  return valid_candidate;
}

template <typename scalar_t>
std::tuple<at::Tensor, at::Tensor, at::Tensor> batch_score_nms_kernel(const at::Tensor& dets,
                          const at::Tensor& scores,
                          const float threshold, int max_num=200) {
  // Reference to: https://github.com/mlcommons/inference/blob/0f096a18083c3fd529c1fbf97ebda7bc3f1fda70/others/cloud/single_stage_detector/pytorch/utils.py#L163
  auto ndets = dets.size(0);
  auto nscore = scores.size(1);

  std::vector<at::Tensor> scores_split = scores.split(1, 1);

  std::vector<at::Tensor> bboxes_out(nscore);
  std::vector<at::Tensor> scores_out(nscore);
  std::vector<at::Tensor> labels_out(nscore);

#ifdef _OPENMP
#if (_OPENMP >= 201307)
# pragma omp parallel for simd
#else
# pragma omp parallel for schedule(static)
#endif
#endif
  // skip background (i = 0)
  for (int64_t i = 1; i < nscore; i++) {
    at::Tensor score = scores_split[i].squeeze(1);

    at::Tensor mask_index = at::nonzero(score > 0.05).squeeze(1);
    at::Tensor bboxes = at::index_select(dets, /*dim*/0, mask_index);
    score = at::index_select(score, /*dim*/0, mask_index);
    
    if (score.size(0) == 0) {
      continue;
    }

    at::Tensor score_sorted, score_idx_sorted;
    std::tie(score_sorted, score_idx_sorted) = score.sort(0);

    // # select max_num indices
    score_idx_sorted = score_idx_sorted.slice(/*dim*/0, /*start*/std::max(score.size(0) - max_num, static_cast<int64_t>(0)), /*end*/score.size(0));

    at::Tensor keep = nms_cpu_kernel<scalar_t>(at::index_select(bboxes, /*dim*/0, score_idx_sorted), at::index_select(score, /*dim*/0, score_idx_sorted), threshold, /*bias*/0);
    at::Tensor candidates = at::index_select(score_idx_sorted, /*dim*/0, keep);

    bboxes_out[i] = at::index_select(bboxes, /*dim*/0, candidates);
    scores_out[i] = at::index_select(score, /*dim*/0, candidates);
    // TODO optimize the fill_
    labels_out[i] = at::empty({candidates.sizes()}).fill_(i);
  }

  std::vector<at::Tensor> valid_bboxes_out = remove_empty(bboxes_out);
  std::vector<at::Tensor> valid_scores_out = remove_empty(scores_out);
  std::vector<at::Tensor> valid_labels_out = remove_empty(labels_out);

  at::Tensor bboxes_out_ = at::cat(valid_bboxes_out, 0);
  at::Tensor labels_out_ = at::cat(valid_labels_out, 0);
  at::Tensor scores_out_ = at::cat(valid_scores_out, 0);

  return std::make_tuple(bboxes_out_, labels_out_, scores_out_);
}

at::Tensor nms_cpu(const at::Tensor& dets,
               const at::Tensor& scores,
               const float threshold) {
  at::Tensor result;
  AT_DISPATCH_FLOATING_TYPES(dets.type(), "nms", [&] {
    result = nms_cpu_kernel<scalar_t>(dets, scores, threshold);
  });
  return result;
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> batch_score_nms_cpu(const at::Tensor& dets,
               const at::Tensor& scores,
               const float threshold) {
  std::tuple<at::Tensor, at::Tensor, at::Tensor> result;
  AT_DISPATCH_FLOATING_TYPES(dets.type(), "batch_score_nms", [&] {
    result = batch_score_nms_kernel<scalar_t>(dets, scores, threshold);
  });
  return result;
}

at::Tensor IpexExternal::nms(const at::Tensor& dets,
               const at::Tensor& scores,
               const float threshold) {
#if defined(IPEX_DISP_OP)
  printf("IpexExternal::nms\n");
#endif
#if defined(IPEX_PROFILE_OP)
  RECORD_FUNCTION("IpexExternal::nms", std::vector<c10::IValue>({dets, scores}), torch::autograd::Node::peek_at_next_sequence_nr());
#endif
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(dets.layout() == c10::kStrided);
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(scores.layout() == c10::kStrided);
  auto&& _ipex_dets = bridge::shallowFallbackToCPUTensor(dets);
  auto&& _ipex_scores = bridge::shallowFallbackToCPUTensor(scores);
  auto&& _ipex_result = nms_cpu(_ipex_dets, _ipex_scores, threshold);
  static_cast<void>(_ipex_result); // Avoid warnings in case not used
  return bridge::shallowUpgradeToDPCPPTensor(_ipex_result);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> IpexExternal::batch_score_nms(const at::Tensor& dets,
               const at::Tensor& scores,
               const float threshold) {
#if defined(IPEX_DISP_OP)
  printf("IpexExternal::batch_score_nms\n");
#endif
#if defined(IPEX_PROFILE_OP)
  RECORD_FUNCTION("IpexExternal::batch_score_nms", std::vector<c10::IValue>({dets, scores}), torch::autograd::Node::peek_at_next_sequence_nr());
#endif
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(dets.layout() == c10::kStrided);
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(scores.layout() == c10::kStrided);
  auto&& _ipex_dets = bridge::shallowFallbackToCPUTensor(dets);
  auto&& _ipex_scores = bridge::shallowFallbackToCPUTensor(scores);
  auto&& _ipex_result = batch_score_nms_cpu(_ipex_dets, _ipex_scores, threshold);
  static_cast<void>(_ipex_result); // Avoid warnings in case not used
  return std::tuple<at::Tensor,at::Tensor,at::Tensor>(bridge::shallowUpgradeToDPCPPTensor(std::get<0>(_ipex_result)), bridge::shallowUpgradeToDPCPPTensor(std::get<1>(_ipex_result)), bridge::shallowUpgradeToDPCPPTensor(std::get<2>(_ipex_result)));
}
}