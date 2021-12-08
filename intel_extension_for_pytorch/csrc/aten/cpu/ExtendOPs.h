#pragma once

#include <ATen/Tensor.h>
#include <torch/extension.h>

namespace torch_ipex {

class IPEXROIAlignOp : public torch::autograd::Function<IPEXROIAlignOp> {
 public:
  // forward function without autograd overhead, will go this way when only do
  // forward
  static at::Tensor _forward(
      const at::Tensor& input,
      const at::Tensor& rois,
      double spatial_scale,
      int64_t pooled_height,
      int64_t pooled_width,
      int64_t sampling_ratio,
      bool aligned);

  static at::Tensor forward(
      torch::autograd::AutogradContext* ctx,
      const at::Tensor& input,
      const at::Tensor& rois,
      double spatial_scale,
      int64_t pooled_height,
      int64_t pooled_width,
      int64_t sampling_ratio,
      bool aligned);

  static torch::autograd::variable_list backward(
      torch::autograd::AutogradContext* ctx,
      torch::autograd::variable_list grad_outputs);
};

class AtenIpexTypeExt {
 public:
  static at::Tensor ROIAlign_forward(
      const at::Tensor& input,
      const at::Tensor& rois,
      double spatial_scale,
      int64_t pooled_height,
      int64_t pooled_width,
      int64_t sampling_ratio,
      bool aligned);

  static std::tuple<at::Tensor, at::Tensor, at::Tensor> ipex_lstm(
      const at::Tensor& input,
      std::vector<at::Tensor> hx,
      std::vector<at::Tensor> params,
      bool has_biases,
      int64_t num_layers,
      double dropout_p,
      bool train,
      bool bidirectional,
      bool batch_first);

  /// \brief Perform non-maximum suppression.
  ///
  /// \param dets: predicted loc in ltrb format for one batchsize, size
  /// [number_boxes, 4], for example: [200, 4]. \param scores: predicted score
  /// for one batchsize and one class, size [number_boxes], for example: [200].
  /// \param threshold: IOU threshold(scalar) to suppress bboxs which has the
  /// IOU val larger than the threshold. \param sorted: The score and dets are
  /// already sorted in Descending order.
  ///
  /// \return result is a Tensor of dets' indexs to be keeped.
  static at::Tensor nms(
      const at::Tensor& dets,
      const at::Tensor& scores,
      const double threshold,
      const bool sorted);

  /// \brief Perform batch non-maximum suppression.
  ///
  /// C++ version of Encoder::decode_single.
  /// Refer to
  /// https://github.com/mlcommons/inference/blob/v0.7/others/cloud/single_stage_detector/pytorch/utils.py.
  ///
  /// \param dets: predicted loc in ltrb format, size [BS, number_boxes, 4], for
  /// example: [1, 15130, 4]. \param scores: predicted score, size [BS,
  /// number_boxes, class_number], for example: [1, 15130, 81]. \param
  /// threshold: IOU threshold(scalar) to suppress bboxs which has the IOU val
  /// larger than the threshold. \param max_output: the max number of output
  /// bbox.
  ///
  /// \return result is a list of tensors, each 4 continuous tensors
  /// corresponding the decode results of one image
  ///   bboxes_out_: the selected out bboxes coordinate, size [max_output, 4].
  ///   labels_out_: the label of each selected out bboxes, size [max_output].
  ///   scores_out_: the score of each selected out bboxes, size [max_output].
  ///   length_out_: the number of detection bboxs [1].
  static std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>
  batch_score_nms(
      const at::Tensor& dets,
      const at::Tensor& scores,
      const double threshold,
      const int64_t max_output);

  /// \brief Perform batch non-maximum suppression (NMS) for MaskRCNN RPN part.
  ///
  /// C++ version of batch NMS for MaskRCNN RPN part.
  /// Refer to
  /// https://github.com/facebookresearch/maskrcnn-benchmark/blob/master/maskrcnn_benchmark/modeling/rpn/inference.py#L111.
  ///
  /// \param batch_dets: predicted loc in ltrb format, size [BS, number_boxes,
  /// 4]. \param batch_scores: predicted score, size [BS, number_boxes]. \param
  /// image_shapes: the shapes of images, BS tuples in vector. \param min_size:
  /// the minimum size of bboxs. \param threshold: IOU threshold(scalar) to
  /// suppress bboxs which has the IOU val larger than the threshold. \param
  /// max_output: the maximum number of output bboxs.
  ///
  /// \return result is a tuple. There are 2 vectors of tensors in the tuple:
  ///   bboxes_out_: the selected out bboxes coordinate, BS tensors in vector,
  ///   and the size of each tensor: [selected_box_number, 4]. scores_out_: the
  ///   score of each selected out bboxes, BS tensors in vector, and the size of
  ///   each tensor: [selected_box_number].
  static std::tuple<std::vector<at::Tensor>, std::vector<at::Tensor>> rpn_nms(
      const at::Tensor& batch_dets,
      const at::Tensor& batch_scores,
      const std::vector<std::tuple<int64_t, int64_t>>& image_shapes,
      const int64_t min_size,
      const double threshold,
      const int64_t max_output);

  /// \brief Perform batch non-maximum suppression (NMS) for MaskRCNN box_head
  /// part.
  ///
  /// C++ version of batch NMS for MaskRCNN box_head part.
  /// Refer to
  /// https://github.com/facebookresearch/maskrcnn-benchmark/blob/master/maskrcnn_benchmark/modeling/roi_heads/box_head/inference.py#L79.
  ///
  /// \param batch_bboxes: predicted loc in ltrb format, BS tensors in vector,
  /// and the size of each tensor: [number_boxes, 4]. \param batch_scores:
  /// predicted score, BS tensors in vector, and the size of each tensor:
  /// [number_boxes]. \param image_shapes: the shapes of images, BS tuples in
  /// vector. \param score_thresh: the threshold of score. \param threshold: IOU
  /// threshold(scalar) to suppress bboxs which has the IOU val larger than the
  /// threshold. \param detections_per_img: the max number of detections per
  /// image. \param num_classes: class number of objects.
  ///
  /// \return result is a tuple. There are 3 vectors of tensors in the tuple:
  ///   bboxes_out_: the selected out bboxes coordinate, BS tensors in vector,
  ///   and the size of each tensor: [selected_box_number, 4]. scores_out_: the
  ///   score of each selected out bboxes, BS tensors in vector, and the size of
  ///   each tensor: [selected_box_number]. labels_out_: the label of each
  ///   selected out bboxes, BS tensors in vector, and the size of each tensor:
  ///   [selected_box_number].
  static std::tuple<
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
      const int64_t num_classes);

  static at::Tensor interaction_forward(const std::vector<at::Tensor>& input);
  static std::vector<at::Tensor> interaction_backward(
      const at::Tensor& grad_out,
      const std::vector<at::Tensor>& input);

  static at::Tensor embedding_bag(
      const at::Tensor& weight,
      const at::Tensor& indices,
      const at::Tensor& offsets,
      bool sparse,
      bool include_last_offset);

  static bool embedding_bag_fast_path_sum(
      const at::Tensor weight,
      const c10::optional<at::Tensor> per_sample_weights,
      int64_t mode,
      const c10::optional<int64_t> padding_idx);

  /// \brief Do scale and transform from xywh to ltrb for predicted loc and do
  /// Softmax along the last dim for predicted score.
  ///
  /// C++ version of Encoder::scale_back_batch.
  /// Refer to
  /// https://github.com/mlcommons/inference/blob/v0.7/others/cloud/single_stage_detector/pytorch/utils.py.
  ///
  /// \param bboxes_in: predicted loc in xywh format, size [BS, number_boxes,
  /// 4], for example: [1, 15130, 4]. \param scores_in: predicted score, size
  /// [BS, number_boxes, class_number], for example: [1, 15130, 81]. \param
  /// dboxes_xywh: scale factor for each bbox from predicted loc to true loc,
  /// size [1, number_boxes, 4]. \param scale_xy: scale factor(scalar) of xy
  /// dimention for bboxes_in. \param scale_wh: scale factor(scalar) of wh
  /// dimention for bboxes_in.
  ///
  /// \return tuple<bbox_result, bbox_result>,
  ///   bbox_result: True loc in lrtb format, size [BS, number_boxes, 4], for
  ///   example: [1, 15130, 4]. scores_result: Normalized score, size [BS,
  ///   number_boxes, class_number], for example: [1, 15130, 81].
  static std::tuple<at::Tensor, at::Tensor> parallel_scale_back_batch(
      const at::Tensor& bboxes_in,
      const at::Tensor& scores_in,
      const at::Tensor& dboxes_xywh,
      const double scale_xy,
      const double scale_wh);
  static at::Tensor cumsum(
      at::Tensor& result,
      const at::Tensor& self,
      int64_t dim,
      c10::optional<at::ScalarType> dtype);
};

} // namespace torch_ipex
