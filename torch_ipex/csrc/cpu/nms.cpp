// Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#include "ExtendOPs.h"
#include <ATen/Parallel.h>
#include <algorithm>
#include <c10/util/Exception.h>
#include <torch/csrc/autograd/function.h>
#include "torch_ipex/csrc/autocast_mode.h"
#include "torch_ipex/csrc/autocast_verbose.h"
#include <immintrin.h>

namespace torch_ipex {
/*
 When calculating the Intersection over Union:
  MaskRCNN: bias = 1
  SSD-Resnet34: bias = 0
*/
template <typename scalar_t, bool sorted>
at::Tensor nms_cpu_kernel(const at::Tensor& dets,
                          const at::Tensor& scores,
                          const float threshold, float bias=1.0);

template <typename scalar_t, bool sorted>
at::Tensor nms_cpu_kernel(const at::Tensor& dets,
                          const at::Tensor& scores,
                          const float threshold, float bias) {
  AT_ASSERTM(!dets.is_cuda(), "dets must be a CPU tensor");
  AT_ASSERTM(!scores.is_cuda(), "scores must be a CPU tensor");
  AT_ASSERTM(dets.scalar_type() == scores.scalar_type(), "dets should have the same type as scores");

  if (dets.numel() == 0) {
    return at::empty({0}, dets.options().dtype(at::kLong).device(at::kCPU));
  }

  auto x1_t = dets.select(1, 0).contiguous();
  auto y1_t = dets.select(1, 1).contiguous();
  auto x2_t = dets.select(1, 2).contiguous();
  auto y2_t = dets.select(1, 3).contiguous();

  at::Tensor areas_t = (x2_t - x1_t + bias) * (y2_t - y1_t + bias);

  auto ndets = dets.size(0);
  // If scores and dets are already sorted in descending order, we don't need to sort it again.
  auto order_t = sorted ? at::arange(0, ndets, scores.options().dtype(at::kLong)) :
                          std::get<1>(scores.sort(0, /* descending=*/true));

  at::Tensor suppressed_t = at::zeros({ndets}, dets.options().dtype(at::kByte).device(at::kCPU));

  auto suppressed = suppressed_t.data_ptr<uint8_t>();
  auto order = order_t.data_ptr<int64_t>();
  auto x1 = x1_t.data_ptr<scalar_t>();
  auto y1 = y1_t.data_ptr<scalar_t>();
  auto x2 = x2_t.data_ptr<scalar_t>();
  auto y2 = y2_t.data_ptr<scalar_t>();
  auto areas = areas_t.data_ptr<scalar_t>();
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
# pragma omp parallel for simd schedule(static) if (omp_get_max_threads() > 1 && !omp_in_parallel())
#else
# pragma omp parallel for schedule(static) if (omp_get_max_threads() > 1 && !omp_in_parallel())
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

#ifdef CPU_AVX512
// Optimized nms_cpu_kernel specialized for data type: float32 and sorted_score
template <>
at::Tensor nms_cpu_kernel</*scalar_t*/float, /*sorted*/true>(const at::Tensor& dets,
                          const at::Tensor& scores,
                          const float threshold, float bias) {
  AT_ASSERTM(!dets.is_cuda(), "dets must be a CPU tensor");
  AT_ASSERTM(!scores.is_cuda(), "scores must be a CPU tensor");
  AT_ASSERTM(dets.scalar_type() == scores.scalar_type(), "dets should have the same type as scores");

  AT_ASSERTM(dets.sizes().size() == 2, "dets should have 2 dimension");
  AT_ASSERTM(scores.sizes().size() == 1, "scores should have 1 dimension");

  at::Tensor dets_bbox_number_in_lastdim = dets.permute({1, 0}).contiguous();
  AT_ASSERTM(dets_bbox_number_in_lastdim.size(1) == scores.size(0), "dets should have number of bboxs as scores");
  AT_ASSERTM(dets_bbox_number_in_lastdim.size(0) == 4, "each bbox in dets should have 4 coordinates");

  if (dets_bbox_number_in_lastdim.numel() == 0) {
    return at::empty({0}, dets_bbox_number_in_lastdim.options().dtype(at::kLong).device(at::kCPU));
  }

  auto x1_t = dets_bbox_number_in_lastdim.select(0, 0).contiguous();
  auto y1_t = dets_bbox_number_in_lastdim.select(0, 1).contiguous();
  auto x2_t = dets_bbox_number_in_lastdim.select(0, 2).contiguous();
  auto y2_t = dets_bbox_number_in_lastdim.select(0, 3).contiguous();

  auto ndets = dets_bbox_number_in_lastdim.size(1);
  auto ndets_up_scale = (ndets/16+1)*16;
  auto ndets_down_scale = (ndets/16)*16;
  at::Tensor&& areas_t = at::zeros({ndets}, dets_bbox_number_in_lastdim.options()).contiguous();
  auto areas = areas_t.data_ptr<float>();
  auto x1 = x1_t.data_ptr<float>();
  auto y1 = y1_t.data_ptr<float>();
  auto x2 = x2_t.data_ptr<float>();
  auto y2 = y2_t.data_ptr<float>();
  __m512 m512_zero = _mm512_setzero_ps();
  __m512 m512_bias = _mm512_set1_ps(bias);
  __m128i m128_zeroi = _mm_setzero_si128();

  // Step1: Calculate the area of all bbox's
#ifdef _OPENMP
#if (_OPENMP >= 201307)
# pragma omp parallel for simd schedule(static) if (omp_get_max_threads() > 1 && !omp_in_parallel())
#else
# pragma omp parallel for schedule(static) if (omp_get_max_threads() > 1 && !omp_in_parallel())
#endif
#endif
  for (int i = 0; i < ndets_up_scale; i += 16) {
    __m512 m512_x1;
    __m512 m512_x2;
    __m512 m512_y1;
    __m512 m512_y2;
    __m512 m512_result;
    if (i < ndets_down_scale) {
      // vector
      m512_x1 = _mm512_loadu_ps(x1+i);
      m512_x2 = _mm512_loadu_ps(x2+i);
      m512_y1 = _mm512_loadu_ps(y1+i);
      m512_y2 = _mm512_loadu_ps(y2+i);
      if (bias == 0) {
        m512_result = _mm512_mul_ps(_mm512_sub_ps(m512_x2, m512_x1), _mm512_sub_ps(m512_y2, m512_y1));
      } else {
        m512_result = _mm512_mul_ps(_mm512_add_ps(_mm512_sub_ps(m512_x2, m512_x1), m512_bias),
                                   _mm512_add_ps(_mm512_sub_ps(m512_y2, m512_y1), m512_bias));
      }
      _mm512_storeu_ps(areas+i, m512_result);
    } else {
      // tail case
      unsigned short left_idx = ndets - ndets_down_scale;
      __mmask16 mask = (1 << left_idx) - 1; //0x03ff;
      m512_x1 = _mm512_mask_loadu_ps(m512_zero, mask, x1+i);
      m512_x2 = _mm512_mask_loadu_ps(m512_zero, mask, x2+i);
      m512_y1 = _mm512_mask_loadu_ps(m512_zero, mask, y1+i);
      m512_y2 = _mm512_mask_loadu_ps(m512_zero, mask, y2+i);
      if (bias == 0) {
        m512_result = _mm512_mask_mul_ps(m512_zero, mask,
                                        _mm512_mask_sub_ps(m512_zero, mask, m512_x2, m512_x1),
                                        _mm512_mask_sub_ps(m512_zero, mask, m512_y2, m512_y1));
      } else {
        m512_result = _mm512_mask_mul_ps(m512_zero, mask,
                                        _mm512_mask_add_ps(m512_zero, mask, _mm512_mask_sub_ps(m512_zero, mask, m512_x2, m512_x1), m512_bias),
                                        _mm512_mask_add_ps(m512_zero, mask, _mm512_mask_sub_ps(m512_zero, mask, m512_y2, m512_y1), m512_bias));
      }
      _mm512_mask_storeu_ps(areas+i, mask, m512_result);
    }
  }
  // Step2: Go through the NMS flow
  at::Tensor suppressed_t = at::zeros({ndets}, dets_bbox_number_in_lastdim.options().dtype(at::kByte).device(at::kCPU));
  auto suppressed = suppressed_t.data_ptr<uint8_t>();
  for (int64_t i = 0; i < ndets; i++) {
    if (suppressed[i] == 1)
      continue;
    auto ix1 = x1[i];
    auto iy1 = y1[i];
    auto ix2 = x2[i];
    auto iy2 = y2[i];
    auto iarea = areas[i];

    __m512 m512_ix1 = _mm512_set1_ps(ix1);
    __m512 m512_ix2 = _mm512_set1_ps(ix2);
    __m512 m512_iy1 = _mm512_set1_ps(iy1);
    __m512 m512_iy2 = _mm512_set1_ps(iy2);
    __m512 m512_iarea = _mm512_set1_ps(iarea);
    __m512 m512_threshold = _mm512_set1_ps(threshold);

    auto ndets_i_up_scale = ((ndets-i-1)/16+1)*16;
    auto ndets_i_down_scale = ((ndets-i-1)/16)*16;

#ifdef _OPENMP
#if (_OPENMP >= 201307)
# pragma omp parallel for simd schedule(static) if (omp_get_max_threads() > 1 && !omp_in_parallel())
#else
# pragma omp parallel for schedule(static) if (omp_get_max_threads() > 1 && !omp_in_parallel())
#endif
#endif
    for (int64_t _j = 0; _j < ndets_i_up_scale; _j += 16) {
      if (_j < ndets_i_down_scale) {
        int64_t j = _j + i + 1;
        __m512 m512_x1 = _mm512_loadu_ps(x1+j);
        __m512 m512_x2 = _mm512_loadu_ps(x2+j);
        __m512 m512_y1 = _mm512_loadu_ps(y1+j);
        __m512 m512_y2 = _mm512_loadu_ps(y2+j);

        __m512 m512_xx1 = _mm512_max_ps(m512_ix1, m512_x1);
        __m512 m512_yy1 = _mm512_max_ps(m512_iy1, m512_y1);
        __m512 m512_xx2 = _mm512_min_ps(m512_ix2, m512_x2);
        __m512 m512_yy2 = _mm512_min_ps(m512_iy2, m512_y2);

        __m512 m512_w;
        __m512 m512_h;
        if (bias == 0) {
          m512_w = _mm512_max_ps(m512_zero, _mm512_sub_ps(m512_xx2, m512_xx1));
          m512_h = _mm512_max_ps(m512_zero, _mm512_sub_ps(m512_yy2, m512_yy1));
        } else {
          m512_w = _mm512_max_ps(m512_zero, _mm512_add_ps(_mm512_sub_ps(m512_xx2, m512_xx1), m512_bias));
          m512_h = _mm512_max_ps(m512_zero, _mm512_add_ps(_mm512_sub_ps(m512_yy2, m512_yy1), m512_bias));
        }

        __m512 m512_inter = _mm512_mul_ps(m512_w, m512_h);
        __m512 m512_areas = _mm512_loadu_ps(areas+j);
        __m512 m512_over =  _mm512_div_ps(m512_inter, _mm512_sub_ps(_mm512_add_ps(m512_iarea, m512_areas), m512_inter));
        __mmask16 mask_sus = _mm512_cmp_ps_mask(m512_over, m512_threshold, _CMP_GE_OS);
        __m128i res_sus = _mm_mask_set1_epi8(m128_zeroi, mask_sus, 1);
        _mm_mask_storeu_epi8(suppressed+j, mask_sus, res_sus);

      } else {
        // Tail case
        int64_t j = _j + i + 1;
        int64_t idx_left = ndets - j;
        __mmask16 load_mask = (1 << idx_left) - 1;

        __m512 m512_x1 = _mm512_mask_loadu_ps(m512_zero, load_mask, x1+j);
        __m512 m512_x2 = _mm512_mask_loadu_ps(m512_zero, load_mask, x2+j);
        __m512 m512_y1 = _mm512_mask_loadu_ps(m512_zero, load_mask, y1+j);
        __m512 m512_y2 = _mm512_mask_loadu_ps(m512_zero, load_mask, y2+j);

        __m512 m512_xx1 = _mm512_mask_max_ps(m512_zero, load_mask, m512_ix1, m512_x1);
        __m512 m512_yy1 = _mm512_mask_max_ps(m512_zero, load_mask, m512_iy1, m512_y1);
        __m512 m512_xx2 = _mm512_mask_min_ps(m512_zero, load_mask, m512_ix2, m512_x2);
        __m512 m512_yy2 = _mm512_mask_min_ps(m512_zero, load_mask, m512_iy2, m512_y2);

        __m512 m512_w;
        __m512 m512_h;
        if (bias == 0) {
          m512_w = _mm512_mask_max_ps(m512_zero, load_mask, m512_zero,
                                     _mm512_mask_sub_ps(m512_zero, load_mask, m512_xx2, m512_xx1));
          m512_h = _mm512_mask_max_ps(m512_zero, load_mask, m512_zero,
                                     _mm512_mask_sub_ps(m512_zero, load_mask, m512_yy2, m512_yy1));
        } else {
          m512_w = _mm512_mask_max_ps(m512_zero, load_mask, m512_zero,
                                     _mm512_mask_add_ps(m512_zero, load_mask,
                                     _mm512_mask_sub_ps(m512_zero, load_mask, m512_xx2, m512_xx1), m512_bias));
          m512_h = _mm512_mask_max_ps(m512_zero, load_mask, m512_zero,
                                     _mm512_mask_add_ps(m512_zero, load_mask,
                                     _mm512_mask_sub_ps(m512_zero, load_mask, m512_yy2, m512_yy1), m512_bias));
        }
        __m512 m512_inter = _mm512_mask_mul_ps(m512_zero, load_mask, m512_w, m512_h);
        __m512 m512_areas = _mm512_mask_loadu_ps(m512_zero, load_mask, areas+j);
        __m512 m512_over =  _mm512_mask_div_ps(m512_zero, load_mask, m512_inter,
                                         _mm512_mask_sub_ps(m512_zero, load_mask,
                                         _mm512_mask_add_ps(m512_zero, load_mask, m512_iarea, m512_areas), m512_inter));
        __mmask16 mask_sus = _mm512_mask_cmp_ps_mask(load_mask, m512_over, m512_threshold, _CMP_GE_OS);
        __m128i res_sus = _mm_mask_set1_epi8(m128_zeroi, mask_sus, 1);
        _mm_mask_storeu_epi8(suppressed+j, mask_sus, res_sus);

      }
    }
  }
  return at::nonzero(suppressed_t == 0).squeeze(1);
}
#endif

std::vector<at::Tensor> remove_empty(std::vector<at::Tensor>& candidate, int64_t start, int64_t end) {
  std::vector<at::Tensor> valid_candidate;
  for (int64_t i = start; i < end; i++) {
    if (candidate[i].defined()) {
      valid_candidate.push_back(candidate[i]);
    }
  }
  return valid_candidate;
}

template <typename scalar_t>
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>
batch_score_nms_kernel(const at::Tensor &batch_dets,
                       const at::Tensor &batch_scores, const float threshold,
                       const int max_output = 200) {
  // Reference to: https://github.com/mlcommons/inference/blob/0f096a18083c3fd529c1fbf97ebda7bc3f1fda70/others/cloud/single_stage_detector/pytorch/utils.py#L163
  // batch_dets: (batchsize, num_bbox, 4) For example: batch_dets: (1, 15130, 4)
  // batch_scores: (batchsize, num_bbox, label_num) For example: batch_scores: (1, 15130, 81)
  auto nbatch = batch_scores.size(0); // number of batches
  auto ndets = batch_scores.size(1); // number of boxes
  auto nscore = batch_scores.size(2); // number of labels

  auto nbatch_x_nscore = nbatch * nscore; // (number of batches) * (number of labels)
  std::vector<at::Tensor> bboxes_out(nbatch_x_nscore);
  std::vector<at::Tensor> scores_out(nbatch_x_nscore);
  std::vector<at::Tensor> labels_out(nbatch_x_nscore);

#ifdef _OPENMP
#if (_OPENMP >= 201307)
# pragma omp parallel for simd schedule(static) if (omp_get_max_threads() > 1 && !omp_in_parallel())
#else
# pragma omp parallel for schedule(static) if (omp_get_max_threads() > 1 && !omp_in_parallel())
#endif
#endif
  // skip background (i = 0)
  for (int index = 0; index < nbatch_x_nscore; index++) {
    // Parallel in the dimentaion of: batch * nscore
    auto bs = index / nscore;
    auto i = index % nscore;

    // skip background (i = 0)
    if (i == 0) {
      continue;
    }

    at::Tensor dets = batch_dets[bs].squeeze(0); // dets for boxes per image: (num_bbox, 4); For example: (15130, 4)
    at::Tensor scores = batch_scores[bs].squeeze(0); // scores for boxes per image: (num_bbox, 81); For example: (15130, 81)

    at::Tensor score = scores.slice(1, i, i+1).squeeze(1); // score for boxes per image per class: (num_bbox); For example: (15130)

    at::Tensor mask_index = at::nonzero(score > 0.05).squeeze(1);
    at::Tensor bboxes = at::index_select(dets, /*dim*/0, mask_index);
    score = at::index_select(score, /*dim*/0, mask_index);

    if (score.size(0) == 0) {
      continue;
    }

    at::Tensor score_sliced, score_idx_sorted;
    // select max_output highest' score and bboxes
    std::tie(score_sliced, score_idx_sorted) = at::topk(score, (max_output>score.size(0))?score.size(0):max_output, 0);
    at::Tensor bboxes_sliced = at::index_select(bboxes, /*dim*/0, score_idx_sorted);

    at::Tensor keep = nms_cpu_kernel<scalar_t, /*sorted*/true>(bboxes_sliced, score_sliced, threshold, /*bias*/0);

    bboxes_out[index] = at::index_select(bboxes_sliced, /*dim*/0, keep);
    scores_out[index] = at::index_select(score_sliced, /*dim*/0, keep);
    // TODO optimize the fill_
    labels_out[index] = at::empty({keep.sizes()}).fill_(i);
  }

  std::vector<at::Tensor> output_bboxes_(nbatch);
  std::vector<at::Tensor> output_labels_(nbatch);
  std::vector<at::Tensor> output_scores_(nbatch);
  std::vector<at::Tensor> output_length_(nbatch);
#ifdef _OPENMP
#if (_OPENMP >= 201307)
# pragma omp parallel for simd schedule(static) if (omp_get_max_threads() > 1 && !omp_in_parallel())
#else
# pragma omp parallel for schedule(static) if (omp_get_max_threads() > 1 && !omp_in_parallel())
#endif
#endif
  for (int bs = 0; bs < nbatch; bs++) {
    // Post process the tensors to get the top max_output(number) for each Batchsize
    std::vector<at::Tensor> valid_bboxes_out = remove_empty(bboxes_out, bs*nscore, (bs+1)*nscore);
    std::vector<at::Tensor> valid_scores_out = remove_empty(scores_out, bs*nscore, (bs+1)*nscore);
    std::vector<at::Tensor> valid_labels_out = remove_empty(labels_out, bs*nscore, (bs+1)*nscore);

    at::Tensor bboxes_out_ = at::cat(valid_bboxes_out, 0);
    at::Tensor labels_out_ = at::cat(valid_labels_out, 0);
    at::Tensor scores_out_ = at::cat(valid_scores_out, 0);

    std::tuple<at::Tensor, at::Tensor> sort_result = scores_out_.sort(0);
    at::Tensor max_ids = std::get<1>(sort_result);
    max_ids = max_ids.slice(/*dim*/0, /*start*/std::max(max_ids.size(0) - max_output, static_cast<int64_t>(0)), /*end*/max_ids.size(0));
    output_bboxes_[bs] = bboxes_out_.index_select(/*dim*/ 0, /*index*/ max_ids);
    output_labels_[bs] = labels_out_.index_select(/*dim*/ 0, /*index*/ max_ids);
    output_scores_[bs] = scores_out_.index_select(/*dim*/ 0, /*index*/ max_ids);
    output_length_[bs] = torch::tensor(max_ids.size(0), {torch::kInt32});
  }
  return std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>(
      at::cat(output_bboxes_), at::cat(output_labels_), at::cat(output_scores_),
      at::stack(output_length_));
}

template <typename scalar_t>
std::tuple<std::vector<at::Tensor>, std::vector<at::Tensor>> rpn_nms_kernel(const at::Tensor& batch_dets,
                          const at::Tensor& batch_scores,
                          const std::vector<std::tuple<int64_t, int64_t>>& image_shapes,
                          const int min_size,
                          const float threshold,
                          const int max_output) {
  auto nbatch = batch_dets.size(0); // number of batches

  std::vector<at::Tensor> bboxes_out(nbatch);
  std::vector<at::Tensor> scores_out(nbatch);

#ifdef _OPENMP
#if (_OPENMP >= 201307)
# pragma omp parallel for simd schedule(static) if (omp_get_max_threads() > 1 && !omp_in_parallel())
#else
# pragma omp parallel for schedule(static) if (omp_get_max_threads() > 1 && !omp_in_parallel())
#endif
#endif
  for (int i = 0; i < nbatch; i++) {
    at::Tensor dets = batch_dets[i].squeeze(0); // dets for boxes per image: (num_box, 4); For example: (15130, 4)
    at::Tensor scores = batch_scores[i].squeeze(0); // scores for boxes per image: (num_box); For example: (15130)
    auto image_shape = image_shapes[i]; // image shape: (2)
    dets.slice(1, 0, 1).clamp_(0, std::get<0>(image_shape) - 1);
    dets.slice(1, 1, 2).clamp_(0, std::get<1>(image_shape) - 1);
    dets.slice(1, 2, 3).clamp_(0, std::get<0>(image_shape) - 1);
    dets.slice(1, 3, 4).clamp_(0, std::get<1>(image_shape) - 1);
    at::Tensor keep_index = at::nonzero((dets.slice(1, 2, 3).squeeze(1) - dets.slice(1, 0, 1).squeeze(1) + 1 >= min_size) & (dets.slice(1, 3, 4).squeeze(1) - dets.slice(1, 1, 2).squeeze(1) + 1 >= min_size)).squeeze(1);
    dets = at::index_select(dets, 0, keep_index);
    scores = at::index_select(scores, 0, keep_index);
    if (threshold > 0) {
      at::Tensor keep = nms_cpu_kernel<scalar_t, /*sorted*/true>(dets, scores, threshold);
      if (max_output > 0) {
        keep = keep.slice(0, 0, max_output);
      }
      bboxes_out[i] = dets.index_select(0, keep);
      scores_out[i] = scores.index_select(0, keep);
    } else {
      bboxes_out[i] = dets;
      scores_out[i] = scores;
    }
  }
  return std::make_tuple(bboxes_out, scores_out);
}

template <typename scalar_t>
std::tuple<std::vector<at::Tensor>, std::vector<at::Tensor>, std::vector<at::Tensor>> box_head_nms_kernel(const std::vector<at::Tensor>& batch_bboxes,
                          const std::vector<at::Tensor>& batch_scores,
                          const std::vector<std::tuple<int64_t, int64_t>>& image_shapes,
                          const float score_thresh,
                          const float threshold,
                          const int detections_per_img,
                          const int num_classes) {
  auto nbatch = batch_scores.size(); // number of batches
  auto nbatch_x_nclass = nbatch * num_classes; // (number of batches) * (number of labels)

  std::vector<at::Tensor> bboxes_out(nbatch_x_nclass);
  std::vector<at::Tensor> scores_out(nbatch_x_nclass);
  std::vector<at::Tensor> labels_out(nbatch_x_nclass);

#ifdef _OPENMP
#if (_OPENMP >= 201307)
# pragma omp parallel for simd schedule(static) if (omp_get_max_threads() > 1 && !omp_in_parallel())
#else
# pragma omp parallel for schedule(static) if (omp_get_max_threads() > 1 && !omp_in_parallel())
#endif
#endif
  for (int bs = 0; bs < nbatch; bs++) {
    at::Tensor bboxes = batch_bboxes[bs].reshape({-1, 4});
    at::Tensor scores = batch_scores[bs];
    auto image_shape = image_shapes[bs];
    bboxes.slice(1, 0, 1).clamp_(0, std::get<0>(image_shape) - 1);
    bboxes.slice(1, 1, 2).clamp_(0, std::get<1>(image_shape) - 1);
    bboxes.slice(1, 2, 3).clamp_(0, std::get<0>(image_shape) - 1);
    bboxes.slice(1, 3, 4).clamp_(0, std::get<1>(image_shape) - 1);
    bboxes = bboxes.reshape({-1, num_classes * 4});
    scores = scores.reshape({-1, num_classes});
    at::Tensor indexes = scores > score_thresh;

    for (int j = 1; j < num_classes; j++) {
      at::Tensor index = at::nonzero(indexes.slice(1, j, j + 1).squeeze(1)).squeeze(1);
      at::Tensor score = scores.slice(1, j, j + 1).squeeze(1).index_select(0, index);
      at::Tensor bbox = bboxes.slice(1, j * 4, (j + 1) * 4).index_select(0, index);
      if (score.size(0) == 0) {
        continue;
      }
      auto iter = bs * num_classes + j;
      if (threshold > 0) {
        at::Tensor keep = nms_cpu_kernel<scalar_t, /*sorted*/false>(bbox, score, threshold);
        bboxes_out[iter] = bbox.index_select(0, keep);
        scores_out[iter] = score.index_select(0, keep);
        labels_out[iter] = at::full({keep.sizes()}, j, torch::kInt64);
      } else {
        bboxes_out[iter] = bbox;
        scores_out[iter] = score;
        labels_out[iter] = at::full({score.sizes()}, j, torch::kInt64);
      }
    }
  }

  std::vector<at::Tensor> bboxes_out_(nbatch);
  std::vector<at::Tensor> scores_out_(nbatch);
  std::vector<at::Tensor> labels_out_(nbatch);

#ifdef _OPENMP
#if (_OPENMP >= 201307)
# pragma omp parallel for simd schedule(static) if (omp_get_max_threads() > 1 && !omp_in_parallel())
#else
# pragma omp parallel for schedule(static) if (omp_get_max_threads() > 1 && !omp_in_parallel())
#endif
#endif
  for (int bs = 0; bs < nbatch; bs++) {
    std::vector<at::Tensor> valid_bboxes_out = remove_empty(bboxes_out, bs*num_classes, (bs+1)*num_classes);
    std::vector<at::Tensor> valid_scores_out = remove_empty(scores_out, bs*num_classes, (bs+1)*num_classes);
    std::vector<at::Tensor> valid_labels_out = remove_empty(labels_out, bs*num_classes, (bs+1)*num_classes);
    if (valid_bboxes_out.size() > 0) {
      bboxes_out_[bs] = at::cat(valid_bboxes_out, 0);
      scores_out_[bs] = at::cat(valid_scores_out, 0);
      labels_out_[bs] = at::cat(valid_labels_out, 0);
    } else {
      bboxes_out_[bs] = at::empty({0, 4}, torch::kFloat);
      scores_out_[bs] = at::empty({0}, torch::kFloat);
      labels_out_[bs] = at::empty({0}, torch::kInt64);
    }
    auto number_of_detections = bboxes_out_[bs].size(0);
    if (number_of_detections > detections_per_img && detections_per_img > 0) {
      auto out_ = scores_out_[bs].kthvalue(number_of_detections - detections_per_img + 1);
      at::Tensor keep = at::nonzero(scores_out_[bs] >= std::get<0>(out_).item()).squeeze(1);
      bboxes_out_[bs] = bboxes_out_[bs].index_select(0, keep);
      scores_out_[bs] = scores_out_[bs].index_select(0, keep);
      labels_out_[bs] = labels_out_[bs].index_select(0, keep);
    }
  }
  return std::make_tuple(bboxes_out_, scores_out_, labels_out_);
}

at::Tensor nms_cpu(const at::Tensor& dets,
               const at::Tensor& scores,
               const float threshold,
               const bool sorted) {
  at::Tensor result;
  AT_DISPATCH_FLOATING_TYPES(dets.scalar_type(), "nms", [&] {
    result = sorted ? nms_cpu_kernel<scalar_t, true>(dets, scores, threshold) : nms_cpu_kernel<scalar_t, false>(dets, scores, threshold);
  });
  return result;
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>
batch_score_nms_cpu(const at::Tensor &dets, const at::Tensor &scores,
                    const float threshold, const int max_output) {
  std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> result;
  AT_DISPATCH_FLOATING_TYPES(dets.scalar_type(), "batch_score_nms", [&] {
    result = batch_score_nms_kernel<scalar_t>(dets, scores, threshold, max_output);
  });
  return result;
}

std::tuple<std::vector<at::Tensor>, std::vector<at::Tensor>> rpn_nms_cpu(const at::Tensor& batch_dets,
                          const at::Tensor& batch_scores,
                          const std::vector<std::tuple<int64_t, int64_t>>& image_shapes,
                          const int min_size,
                          const float threshold,
                          const int max_output) {
  std::tuple<std::vector<at::Tensor>, std::vector<at::Tensor>> result;
  AT_DISPATCH_FLOATING_TYPES(batch_dets.scalar_type(), "rpn_nms", [&] {
    result = rpn_nms_kernel<scalar_t>(batch_dets, batch_scores, image_shapes, min_size, threshold, max_output);
  });
  return result;
}

std::tuple<std::vector<at::Tensor>, std::vector<at::Tensor>, std::vector<at::Tensor>> box_head_nms_cpu(const std::vector<at::Tensor>& batch_bboxes,
                          const std::vector<at::Tensor>& batch_scores,
                          const std::vector<std::tuple<int64_t, int64_t>>& image_shapes,
                          const float score_thresh,
                          const float threshold,
                          const int detections_per_img,
                          const int num_classes) {
  std::tuple<std::vector<at::Tensor>, std::vector<at::Tensor>, std::vector<at::Tensor>> result;
  AT_DISPATCH_FLOATING_TYPES(batch_bboxes[0].scalar_type(), "box_head_nms", [&] {
    result = box_head_nms_kernel<scalar_t>(batch_bboxes, batch_scores, image_shapes, score_thresh, threshold, detections_per_img, num_classes);
  });
  return result;
}

at::Tensor AtenIpexTypeExt::nms(const at::Tensor& dets,
               const at::Tensor& scores,
               const double threshold,
               const bool sorted) {
#if defined(IPEX_DISP_OP)
  printf("IpexExternal::nms\n");
#endif
#if defined(IPEX_PROFILE_OP)
  RECORD_FUNCTION("IpexExternal::nms", std::vector<c10::IValue>({}));
#endif
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(dets.layout() == c10::kStrided);
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(scores.layout() == c10::kStrided);
  auto&& result = nms_cpu(dets, scores, threshold, sorted);
  static_cast<void>(result); // Avoid warnings in case not used
  return result;
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>
AtenIpexTypeExt::batch_score_nms(const at::Tensor &dets,
                                 const at::Tensor &scores,
                                 const double threshold,
                                 const int64_t max_output) {
#if defined(IPEX_DISP_OP)
  printf("IpexExternal::batch_score_nms\n");
#endif
#if defined(IPEX_PROFILE_OP)
  RECORD_FUNCTION("IpexExternal::batch_score_nms", std::vector<c10::IValue>({}));
#endif
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(dets.layout() == c10::kStrided);
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(scores.layout() == c10::kStrided);
  auto&& result = batch_score_nms_cpu(dets, scores, threshold, max_output);
  static_cast<void>(result); // Avoid warnings in case not used
  return result;
}

std::tuple<std::vector<at::Tensor>, std::vector<at::Tensor>> AtenIpexTypeExt::rpn_nms(const at::Tensor& batch_dets,
                          const at::Tensor& batch_scores,
                          const std::vector<std::tuple<int64_t, int64_t>>& image_shapes,
                          const int64_t min_size,
                          const double threshold,
                          const int64_t max_output) {
#if defined(IPEX_DISP_OP)
  printf("IpexExternal::rpn_nms\n");
#endif
#if defined(IPEX_PROFILE_OP)
  RECORD_FUNCTION("IpexExternal::rpn_nms", std::vector<c10::IValue>({}));
#endif
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(batch_dets.layout() == c10::kStrided);
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(batch_scores.layout() == c10::kStrided);
  auto&& result = rpn_nms_cpu(batch_dets, batch_scores, image_shapes, min_size, threshold, max_output);
  static_cast<void>(result); // Avoid warnings in case not used
  return result;
}

std::tuple<std::vector<at::Tensor>, std::vector<at::Tensor>, std::vector<at::Tensor>> AtenIpexTypeExt::box_head_nms(
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
#if defined(IPEX_PROFILE_OP)
  RECORD_FUNCTION("IpexExternal::box_head_nms", std::vector<c10::IValue>({}));
#endif
  auto&& result = box_head_nms_cpu(batch_bboxes, batch_scores, image_shapes, score_thresh, threshold, detections_per_img, num_classes);
  static_cast<void>(result); // Avoid warnings in case not used
  return result;
}

template <typename scalar_t>
at::Tensor scale_back_batch_kernel(const at::Tensor& _ipex_bboxes_in,
                                         const at::Tensor& _ipex_dboxes_xywh,
                                         const float scale_xy,
                                         const float scale_wh) {
  //_ipex_bboxes_in: [BS, number_boxes, 4], for example: [1, 15130, 4]
  auto _ipex_bboxes_in_conti = _ipex_bboxes_in.contiguous();
  auto _ipex_dboxes_xywh_conti = _ipex_dboxes_xywh.contiguous();
  int64_t batch_size = _ipex_bboxes_in.size(0);
  int64_t boxes_per_image = _ipex_bboxes_in.size(1);
  int64_t ndets = batch_size * boxes_per_image; // batchsize * boxes per image
  at::Tensor output = at::empty({_ipex_bboxes_in.size(0), _ipex_bboxes_in.size(1), _ipex_bboxes_in.size(2)}, _ipex_bboxes_in.options());
  auto output_conti = output.contiguous();

  auto* input_data = _ipex_bboxes_in_conti.data_ptr<scalar_t>();
  auto* output_data = output_conti.data_ptr<scalar_t>();
  auto* input_dboxes_xywh_data = _ipex_dboxes_xywh_conti.data_ptr<double>();

#ifdef _OPENMP
#if (_OPENMP >= 201307)
# pragma omp parallel for simd schedule(static) if (omp_get_max_threads() > 1 && !omp_in_parallel())
#else
# pragma omp parallel for schedule(static) if (omp_get_max_threads() > 1 && !omp_in_parallel())
#endif
#endif
  for (int64_t k = 0; k < ndets; k++) {
    int64_t i = k / boxes_per_image;
    int64_t j = k % boxes_per_image;

    int64_t index = i * boxes_per_image * 4 + j * 4;

    // bboxes_in[:, :, :2] = self.scale_xy*bboxes_in[:, :, :2]
    output_data[index] = input_data[index] * scale_xy;
    output_data[index+1] = input_data[index+1] * scale_xy;
    // bboxes_in[:, :, 2:] = self.scale_wh*bboxes_in[:, :, 2:]
    output_data[index+2] = input_data[index+2] * scale_wh;
    output_data[index+3] = input_data[index+3] * scale_wh;

    int64_t index_dboxes_xywh = j * 4;
    // bboxes_in[:, :, :2] = bboxes_in[:, :, :2]*self.dboxes_xywh[:, :, 2:] + self.dboxes_xywh[:, :, :2]
    output_data[index] = output_data[index] * input_dboxes_xywh_data[index_dboxes_xywh+2] + input_dboxes_xywh_data[index_dboxes_xywh];
    output_data[index+1] = output_data[index+1] * input_dboxes_xywh_data[index_dboxes_xywh+3] + input_dboxes_xywh_data[index_dboxes_xywh+1];
    // bboxes_in[:, :, 2:] = bboxes_in[:, :, 2:].exp()*self.dboxes_xywh[:, :, 2:]
    output_data[index+2] = exp(output_data[index+2])*input_dboxes_xywh_data[index_dboxes_xywh+2];
    output_data[index+3] = exp(output_data[index+3])*input_dboxes_xywh_data[index_dboxes_xywh+3];

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

    auto l = output_data[index] - 0.5 * output_data[index+2];
    auto t = output_data[index+1] - 0.5 * output_data[index+3];
    auto r = output_data[index] + 0.5 * output_data[index+2];
    auto b = output_data[index+1] + 0.5 * output_data[index+3];
    output_data[index] = l;
    output_data[index+1] = t;
    output_data[index+2] = r;
    output_data[index+3] = b;
  }
  return output;
}

std::tuple<at::Tensor, at::Tensor> AtenIpexTypeExt::parallel_scale_back_batch(const at::Tensor& bboxes_in,
                                                                                 const at::Tensor& scores_in,
                                                                                 const at::Tensor& dboxes_xywh,
                                                                                 const double scale_xy,
                                                                                 const double scale_wh){
#if defined(IPEX_DISP_OP)
  printf("IpexExternal::parallel_scale_back_batch\n");
#endif
#if defined(IPEX_PROFILE_OP)
  RECORD_FUNCTION("IpexExternal::parallel_scale_back_batch", std::vector<c10::IValue>({}));
#endif
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(bboxes_in.layout() == c10::kStrided);
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(dboxes_xywh.layout() == c10::kStrided);

  at::Tensor bbox_result;
  AT_DISPATCH_FLOATING_TYPES(bboxes_in.scalar_type(), "scale_back_batch", [&] {
    bbox_result = scale_back_batch_kernel<scalar_t>(bboxes_in, dboxes_xywh, scale_xy, scale_wh);
  });

  auto&& scores_result = at::softmax(scores_in, -1);

  return std::tuple<at::Tensor, at::Tensor>(bbox_result, scores_result);
}
} // namespace torch_ipex

namespace {
static auto dispatch =
    torch::RegisterOperators()
        .op("torch_ipex::nms", &torch_ipex::AtenIpexTypeExt::nms)
        .op("torch_ipex::batch_score_nms", &torch_ipex::AtenIpexTypeExt::batch_score_nms)
        .op("torch_ipex::rpn_nms", &torch_ipex::AtenIpexTypeExt::rpn_nms)
        .op("torch_ipex::box_head_nms", &torch_ipex::AtenIpexTypeExt::box_head_nms)
        .op("torch_ipex::parallel_scale_back_batch", &torch_ipex::AtenIpexTypeExt::parallel_scale_back_batch);
}

namespace torch_ipex {
namespace autocast {
at::Tensor nms(const at::Tensor& dets,
               const at::Tensor& scores,
               const double threshold,
               const bool sorted) {
  c10::impl::ExcludeDispatchKeyGuard no_autocastCPU(DispatchKey::AutocastCPU);
  static auto op = torch::Dispatcher::singleton()
    .findSchemaOrThrow("torch_ipex::nms", "")
    .typed<decltype(nms)>();
#if defined(ENABLE_AUTOCAST_VERBOSE)
  verbose::OpNameGuard op_name("nms");
#endif
  return op.call(cpu_cached_cast(at::kFloat, dets), cpu_cached_cast(at::kFloat, scores), threshold, sorted);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>
batch_score_nms(const at::Tensor &dets, const at::Tensor &scores,
                const double threshold, const int64_t max_output) {
  c10::impl::ExcludeDispatchKeyGuard no_autocastCPU(DispatchKey::AutocastCPU);
  static auto op = torch::Dispatcher::singleton()
    .findSchemaOrThrow("torch_ipex::batch_score_nms", "")
    .typed<decltype(batch_score_nms)>();
#if defined(ENABLE_AUTOCAST_VERBOSE)
  verbose::OpNameGuard op_name("batch_score_nms");
#endif
  return op.call(cpu_cached_cast(at::kFloat, dets), cpu_cached_cast(at::kFloat, scores), threshold, max_output);
}

std::tuple<std::vector<at::Tensor>, std::vector<at::Tensor>> rpn_nms(const at::Tensor& batch_dets,
                          const at::Tensor& batch_scores,
                          const std::vector<std::tuple<int64_t, int64_t>>& image_shapes,
                          const int64_t min_size,
                          const double threshold,
                          const int64_t max_output) {
  c10::impl::ExcludeDispatchKeyGuard no_autocastCPU(DispatchKey::AutocastCPU);
  static auto op = torch::Dispatcher::singleton()
    .findSchemaOrThrow("torch_ipex::rpn_nms", "")
    .typed<decltype(rpn_nms)>();
#if defined(ENABLE_AUTOCAST_VERBOSE)
  verbose::OpNameGuard op_name("rpn_nms");
#endif
  return op.call(cpu_cached_cast(at::kFloat, batch_dets), cpu_cached_cast(at::kFloat, batch_scores), image_shapes, min_size, threshold, max_output);
}

std::tuple<std::vector<at::Tensor>, std::vector<at::Tensor>, std::vector<at::Tensor>> box_head_nms(
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
#if defined(ENABLE_AUTOCAST_VERBOSE)
  verbose::OpNameGuard op_name("box_head_nms");
#endif
  return op.call(cpu_cached_cast(at::kFloat, batch_bboxes), cpu_cached_cast(at::kFloat, batch_scores), image_shapes, score_thresh, threshold, detections_per_img, num_classes);
}

std::tuple<at::Tensor, at::Tensor> parallel_scale_back_batch(const at::Tensor& bboxes_in,
                                                                const at::Tensor& scores_in,
                                                                const at::Tensor& dboxes_xywh,
                                                                const double scale_xy,
                                                                const double scale_wh) {
  c10::impl::ExcludeDispatchKeyGuard no_autocastCPU(DispatchKey::AutocastCPU);
  static auto op = torch::Dispatcher::singleton()
    .findSchemaOrThrow("torch_ipex::parallel_scale_back_batch", "")
    .typed<decltype(parallel_scale_back_batch)>();
#if defined(ENABLE_AUTOCAST_VERBOSE)
  verbose::OpNameGuard op_name("parallel_scale_back_batch");
#endif
  return op.call(cpu_cached_cast(at::kFloat, bboxes_in), cpu_cached_cast(at::kFloat, scores_in),
                 cpu_cached_cast(at::kFloat, dboxes_xywh), scale_xy, scale_wh);
}

TORCH_LIBRARY_IMPL(torch_ipex, AutocastCPU, m) {
  m.impl("nms", torch_ipex::autocast::nms);
  m.impl("batch_score_nms", torch_ipex::autocast::batch_score_nms);
  m.impl("rpn_nms", torch_ipex::autocast::rpn_nms);
  m.impl("box_head_nms", torch_ipex::autocast::box_head_nms);
  m.impl("parallel_scale_back_batch", torch_ipex::autocast::parallel_scale_back_batch);
}

} // namespace autocast
} // namespace torch_ipex

