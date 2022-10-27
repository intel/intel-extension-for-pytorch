#pragma once

#include <vector>

#include <ATen/Tensor.h>

#include <ideep.hpp>
#include "aten/RNN.h"
#include "ideep/IDeepConversions.h"

namespace torch_ipex {
namespace cpu {

enum LstmDtype : bool {
  Quantized = true,
  Float = false,
};

static ideep::attr_t empty_attr;
struct CommonLstmWeightDesc {
  const at::Tensor& weight_ih_;
  const at::Tensor& weight_hh_;
  int64_t input_size_;
  int64_t num_gates_;
  int64_t hidden_size_;
  const ideep::dims& output_sizes_;
  const ideep::tensor& src_layer_;
  const ideep::tensor& src_iter_;
  const ideep::tensor& src_iter_c_;
  const ideep::tensor& bias_;
  const bool reverse_;
  const float scale_;
  const int32_t zp_;
  const int weights_scale_mask_;
  const std::vector<float>& weights_scales_;

  ideep::tensor w1_src_;
  ideep::tensor w2_src_;

  ideep::attr_t op_attr_;

  ideep::tensor::desc packed_desc_ih_;
  ideep::tensor::desc packed_desc_hh_;

  CommonLstmWeightDesc(
      const at::Tensor& weight_ih,
      const at::Tensor& weight_hh,
      int64_t input_size,
      int64_t num_gates,
      int64_t hidden_size,
      const ideep::dims& output_sizes,
      const ideep::tensor& src_layer,
      const ideep::tensor& src_iter,
      const ideep::tensor& src_iter_c,
      const ideep::tensor& bias,
      const bool reverse,
      const QuantizedLstmParams& quantizedLstmParams)
      : weight_ih_(weight_ih),
        weight_hh_(weight_hh),
        input_size_(input_size),
        num_gates_(num_gates),
        hidden_size_(hidden_size),
        output_sizes_(output_sizes),
        src_layer_(src_layer),
        src_iter_(src_iter),
        src_iter_c_(src_iter_c),
        bias_(bias),
        reverse_(reverse),
        scale_(quantizedLstmParams.scale),
        zp_(quantizedLstmParams.zp),
        weights_scale_mask_(quantizedLstmParams.weights_scale_mask),
        weights_scales_(quantizedLstmParams.weights_scales) {}

  void initialize_weight_src() {
    w1_src_ = itensor_view_from_dense(
        weight_ih_,
        {{1, 1, input_size_, num_gates_, hidden_size_},
         get_mkldnn_dtype(weight_ih_.scalar_type()),
         ideep::format_tag::ldgoi});
    w2_src_ = itensor_view_from_dense(
        weight_hh_,
        {{1, 1, hidden_size_, num_gates_, hidden_size_},
         get_mkldnn_dtype(weight_hh_.scalar_type()),
         ideep::format_tag::ldgoi});
  }

  void initialize_attribute() {
    op_attr_ = empty_attr;
  }

  std::tuple<ideep::tensor, ideep::tensor> get_lstm_public_weight() {
    return std::make_tuple(w1_src_, w2_src_);
  }

  std::tuple<ideep::tensor, ideep::tensor> get_and_save_lstm_packed_weight();
};

template <LstmDtype T>
struct LstmInferenceWeightDesc;

template <>
struct LstmInferenceWeightDesc<LstmDtype::Float> : public CommonLstmWeightDesc {
  LstmInferenceWeightDesc(
      const at::Tensor& weight_ih,
      const at::Tensor& weight_hh,
      int64_t input_size,
      int64_t num_gates,
      int64_t hidden_size,
      const ideep::dims& output_sizes,
      const ideep::tensor& src_layer,
      const ideep::tensor& src_iter,
      const ideep::tensor& src_iter_c,
      const ideep::tensor& bias,
      const bool reverse,
      const QuantizedLstmParams& quantizedLstmParams)
      : CommonLstmWeightDesc(
            weight_ih,
            weight_hh,
            input_size,
            num_gates,
            hidden_size,
            output_sizes,
            src_layer,
            src_iter,
            src_iter_c,
            bias,
            reverse,
            quantizedLstmParams) {}

  void initialize_weight_src() {
    CommonLstmWeightDesc::initialize_weight_src();
  }

  void set_expected_weights_desc();

  void initialize_attribute() {
    CommonLstmWeightDesc::initialize_attribute();
  }

  std::tuple<ideep::tensor, ideep::tensor> get_and_save_lstm_packed_weight() {
    return CommonLstmWeightDesc::get_and_save_lstm_packed_weight();
  }
};

template <>
struct LstmInferenceWeightDesc<LstmDtype::Quantized>
    : public CommonLstmWeightDesc {
  LstmInferenceWeightDesc(
      const at::Tensor& weight_ih,
      const at::Tensor& weight_hh,
      int64_t input_size,
      int64_t num_gates,
      int64_t hidden_size,
      const ideep::dims& output_sizes,
      const ideep::tensor& src_layer,
      const ideep::tensor& src_iter,
      const ideep::tensor& src_iter_c,
      const ideep::tensor& bias,
      const bool reverse,
      const QuantizedLstmParams& quantizedLstmParams)
      : CommonLstmWeightDesc(
            weight_ih,
            weight_hh,
            input_size,
            num_gates,
            hidden_size,
            output_sizes,
            src_layer,
            src_iter,
            src_iter_c,
            bias,
            reverse,
            quantizedLstmParams) {}

  void initialize_weight_src();

  void initialize_attribute() {
    op_attr_.set_rnn_data_qparams(scale_, zp_);
    op_attr_.set_rnn_weights_qparams(weights_scale_mask_, weights_scales_);
  }

  void set_expected_weights_desc();

  std::tuple<ideep::tensor, ideep::tensor> get_and_save_lstm_packed_weight() {
    return CommonLstmWeightDesc::get_and_save_lstm_packed_weight();
  }
};

std::tuple<ideep::tensor, ideep::tensor> get_lstm_packed_weight(
    const at::Tensor& weight_ih,
    const at::Tensor& weight_hh,
    int64_t input_size,
    int64_t num_gates,
    int64_t hidden_size,
    const ideep::dims& output_sizes,
    const ideep::tensor& src_layer,
    const ideep::tensor& src_iter,
    const ideep::tensor& src_iter_c,
    const ideep::tensor& bias,
    const bool reverse,
    const bool train,
    const QuantizedLstmParams& quantizedLstmParams);

bool is_packed(const at::Tensor& weight);

// Get the conv_transpose's expected ideep weight tensor desc.
ideep::tensor::desc get_conv_transpose_expected_weights_desc(
    const ideep::tensor::dims& weights_dims,
    ideep::tensor::data_type w_dtype = ideep::data_type::f32,
    const ideep::tensor::dims& strides = {1, 1, 1},
    const ideep::tensor::dims& padding_l = {0, 0, 0},
    const ideep::tensor::dims& padding_r = {0, 0, 0},
    const ideep::tensor::dims& dilates = {1, 1, 1},
    int groups = 1,
    bool channels_last = false,
    ideep::algorithm aalgorithm = ideep::algorithm::deconvolution_direct,
    ideep::data_type x_dtype = ideep::data_type::f32,
    const ideep::dims& src_dims = ideep::tensor::dims(),
    const ideep::attr_t& attr = ideep::attr_t());

} // namespace cpu
} // namespace torch_ipex
