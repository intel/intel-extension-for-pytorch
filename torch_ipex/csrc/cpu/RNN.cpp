#include "ExtendOPs.h"
#include "WeightPrepack.h"
#include "torch_ipex/csrc/autocast_mode.h"
#include "torch_ipex/csrc/autocast_verbose.h"
#include <ATen/ATen.h>
#include <ATen/Config.h>
#include <ATen/InitialTensorOptions.h>
#include <ATen/MatrixRef.h>
#include <ATen/NativeFunctions.h>
#include <ATen/TensorUtils.h>
#include <c10/util/Exception.h>

namespace torch_ipex {

struct RNNParams {
  ideep::rnn_kind mode;
  int64_t seq_length;
  int64_t mini_batch;
  int64_t input_size;
  int64_t hidden_size;
  int64_t num_directions;
  int64_t num_layers;
  bool batch_first;
  bool train;
  at::IntArrayRef batch_sizes;
  int64_t num_gates;
  int64_t num_bias_gates;

  RNNParams(const at::Tensor &input, at::IntArrayRef batch_sizes_,
            int64_t mode_, int64_t hidden_size_, int64_t num_layers_,
            bool bidirectional, bool batch_first_, bool train_) {
    mode = static_cast<ideep::rnn_kind>(mode_);
    batch_first = batch_first_;
    if (batch_first) {
      seq_length = input.size(1);
      mini_batch = input.size(0);
    } else {
      seq_length = input.size(0);
      mini_batch = input.size(1);
    }
    input_size = input.size(2);
    hidden_size = hidden_size_;
    num_directions = bidirectional ? 2 : 1;
    num_layers = num_layers_;
    train = train_;
    batch_sizes = batch_sizes_;
    if (mode == ideep::rnn_kind::LSTM) {
      num_gates = 4;
      num_bias_gates = 4;
    } else if (mode == ideep::rnn_kind::GRU) {
      num_gates = 3;
      num_bias_gates = 4;
    } else {
      // RNN_RELU; RNN_TANH
      num_gates = 1;
      num_bias_gates = 1;
    }
  }

  bool is_input_packed() const { return batch_sizes.size() != 0; }

  // mkldnn memory descriptors
  using format = ideep::format_tag;
  using desc = ideep::tensor::desc;
  using dtype = ideep::tensor::data_type;
  desc src_layer_desc(int64_t _input_size, dtype dtype) const {
    return {{seq_length, mini_batch, _input_size}, dtype, format::tnc};
  }
  desc src_iter_desc(dtype dtype) const {
    return {{1, 1, mini_batch, hidden_size}, dtype, format::ldnc};
  }
  desc src_iter_c_desc(dtype dtype) const {
    return {{1, 1, mini_batch, hidden_size}, dtype, format::ldnc};
  }
  // logical size described as ldigo
  desc weights_layer_desc(int64_t _input_size, dtype dtype) const {
    return {{1, 1, _input_size, num_gates, hidden_size}, dtype, format::ldgoi};
  }
  desc weights_iter_desc(dtype dtype) const {
    return {{1, 1, hidden_size, num_gates, hidden_size}, dtype, format::ldgoi};
  }
  desc bias_desc(dtype dtype) const {
    return {{1, 1, num_bias_gates, hidden_size}, dtype, format::ldgo};
  }
  desc dst_layer_desc(dtype dtype) const {
    return {{seq_length, mini_batch, hidden_size}, dtype, format::tnc};
  }
  desc dst_iter_desc(dtype dtype) const {
    return {{1, 1, mini_batch, hidden_size}, dtype, format::ldnc};
  }
  desc dst_iter_c_desc(dtype dtype) const {
    return {{1, 1, mini_batch, hidden_size}, dtype, format::ldnc};
  }
};

ideep::tensor::data_type get_mkldnn_dtype(at::ScalarType type) {
  switch (type) {
  case at::ScalarType::Float:
    return ideep::tensor::data_type::f32;
  case at::ScalarType::QInt32:
    return ideep::tensor::data_type::s32;
  case at::ScalarType::QInt8:
    return ideep::tensor::data_type::s8;
  case at::ScalarType::QUInt8:
  case at::ScalarType::Byte:
    return ideep::tensor::data_type::u8;
  case at::ScalarType::BFloat16:
    return ideep::tensor::data_type::bf16;
  default:
    TORCH_CHECK(false, "get_mkldnn_dtype: unsupported data type");
  }
}

std::vector<int64_t> _hidden_size(const RNNParams &rnn) {
  return {rnn.num_layers * rnn.num_directions, rnn.mini_batch, rnn.hidden_size};
}

template <bool is_single_direction>
std::vector<int64_t> _output_size(const RNNParams &rnn) {
  auto output_channels = is_single_direction
                             ? rnn.hidden_size
                             : rnn.hidden_size * rnn.num_directions;
  return {rnn.seq_length, rnn.mini_batch, output_channels};
}

// MKLDNN GRU gate order is different from PyTorch's which requires gates
// shuffle (let rt,zt,nt be reset, update, new gates respectively)
//
//   MKLDNN GRU weight_ih/weight_hh gates order: (zt, rt, nt)
//   PyTorch GRU weight_ih/weight_hh gates order: (rt, zt, nt)
//
// MKLDNN GRU bias has 4 gates instead of 3
//  (PyTorch GRU bias)     (MKLDNN GRU bias)
//
//  bias_ih    bias_hh          bias
//  +-----+    +-----+       +---------+
//  | rt1 |    | rt2 |       | zt1+zt2 |
//  |-----|    |-----|       |---------|
//  | zt1 |    | zt2 |       | rt1+rt2 |
//  |-----|    |-----|       |---------|
//  | nt1 |    | nt2 |       |   nt1   |
//  +-----+    +-----+       |---------|
//                           |   nt2   |
//                           +---------+
//
at::Tensor _shuffle_weight(const at::Tensor &weight, int64_t fn_mode) {
  if (torch_ipex::cpu::is_prepacked(weight))
    return weight;

  auto weight_t = weight.contiguous();
  if (static_cast<ideep::rnn_kind>(fn_mode) == ideep::rnn_kind::GRU) {
    std::vector<at::Tensor> gates = weight_t.chunk(3, /*gates*/ 0);
    return at::cat({gates[1], gates[0], gates[2]}, /*gates*/ 0);
  }
  return weight_t;
};

at::Tensor _shuffle_bias(const at::Tensor &bias_ih, const at::Tensor &bias_hh,
                         int64_t fn_mode) {
  if (static_cast<ideep::rnn_kind>(fn_mode) == ideep::rnn_kind::GRU) {
    std::vector<at::Tensor> b1 = bias_ih.chunk(3, /*output_channels*/ 0);
    std::vector<at::Tensor> b2 = bias_hh.chunk(3, /*output_channels*/ 0);
    return at::cat({b1[1] + b2[1], b1[0] + b2[0], b1[2], b2[2]},
                   /*output_channels*/ 0);
  }
  return bias_ih + bias_hh;
};

at::Tensor mkldnn_rnn_layer(at::Tensor &hy_, at::Tensor &cy_,
                            const at::Tensor &input, at::TensorList weights,
                            const at::Tensor &hx_, const at::Tensor &cx_,
                            bool reverse, const RNNParams &rnn) {
  TORCH_CHECK(weights.size() == 2 || weights.size() == 4);

  auto output_size = _output_size</*is_single_direction*/ true>(rnn);
  auto output = at::empty(output_size, input.options());

  bool has_bias = weights.size() == 4;
  auto weight_ih = _shuffle_weight(weights[0], rnn.mode);
  auto weight_hh = _shuffle_weight(weights[1], rnn.mode);

  auto bias = has_bias ? _shuffle_bias(weights[2], weights[3], rnn.mode)
                       : at::zeros({rnn.num_bias_gates * rnn.hidden_size},
                                   weight_ih.options());

  // per layer input size
  int64_t input_size = input.size(2);
  auto x = torch_ipex::cpu::get_mkldnn_tensor_view(
      input,
      rnn.src_layer_desc(input_size, get_mkldnn_dtype(input.scalar_type())));
  auto hx = torch_ipex::cpu::get_mkldnn_tensor_view(
      hx_, rnn.src_iter_desc(get_mkldnn_dtype(hx_.scalar_type())));
  auto cx = torch_ipex::cpu::get_mkldnn_tensor_view(
      cx_, rnn.src_iter_c_desc(get_mkldnn_dtype(cx_.scalar_type())));
  auto b = torch_ipex::cpu::get_mkldnn_tensor_view(
      bias, rnn.bias_desc(get_mkldnn_dtype(bias.scalar_type())));
  auto y = torch_ipex::cpu::get_mkldnn_tensor_view(
      output, rnn.dst_layer_desc(get_mkldnn_dtype(output.scalar_type())));
  auto hy = torch_ipex::cpu::get_mkldnn_tensor_view(
      hy_, rnn.dst_iter_desc(get_mkldnn_dtype(hy_.scalar_type())));
  auto cy = torch_ipex::cpu::get_mkldnn_tensor_view(
      cy_, rnn.dst_iter_c_desc(get_mkldnn_dtype(cy_.scalar_type())));

  ideep::tensor w1, w2;
  std::tie(w1, w2) = torch_ipex::cpu::get_lstm_prepacked_weight(
      weight_ih, weight_hh, input_size, rnn.num_gates, rnn.hidden_size,
      {output_size.cbegin(), output_size.cend()}, x, hx, cx, b, reverse);

  ideep::lstm_forward::compute(x, hx, cx, w1, w2, b, y, hy, cy, reverse);

  return output;
}

// MKLDNN RNN integration notes:
// I. Memory Formats
//   a. mkldnn will use plain formats for input, hx/cx, output, hy/cy
//      and possibly use blocked formats for weights depending shape info.
//   b. All mkldnn memorys are created (in plain format) as views on ATen
//   tensor,
//      the weight reorder(if any) is handed automatically inside ideep (mkldnn
//      bridge)
//
// II. MKLDNN Primitive Mapping
//   a. mkldnn rnn primitive doesn't support training with dropout or padded
//   input sequence. b. here break a single RNN module into { num_layers *
//   num_directions } mkldnn rnn primitives
//      for future need to cover these feature gaps.
//
// TODO: a. training with dropout
//   b. padded sequence input support
//
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>
mkldnn_rnn(const at::Tensor &input_, at::TensorList weight,
           int64_t weight_stride0, const at::Tensor &hx_, const at::Tensor &cx_,
           int64_t mode, int64_t hidden_size, int64_t num_layers,
           bool batch_first, double dropout_p, bool train, bool bidirectional,
           at::IntArrayRef batch_sizes) {
  TORCH_CHECK(!train || dropout_p == 0.0, "mkldnn_rnn doesn't support dropout");
  TORCH_CHECK(batch_sizes.size() == 0,
              "mkldnn_rnn doesn't support packed input");
  if (static_cast<ideep::rnn_kind>(mode) != ideep::rnn_kind::LSTM) {
    TORCH_CHECK(!cx_.defined(),
                "mkldnn_rnn: illegal defined cx for non-LSTM RNN");
  }

  RNNParams fn(input_, batch_sizes, mode, hidden_size, num_layers,
               bidirectional, batch_first, train);

  auto input = input_;
  if (batch_first && !fn.is_input_packed()) {
    input = input.transpose(0, 1);
  }
  input = input.contiguous();

  auto hx = hx_.contiguous();
  auto cx = cx_.defined() ? cx_.contiguous() : at::Tensor();

  auto hy = at::empty(_hidden_size(fn), hx.options());
  // NB: Not allowed to return undefined tensors
  auto cy = cx.defined() ? at::empty(_hidden_size(fn), cx.options())
                         : at::empty({0}, hx.options());

  at::MatrixRef<at::Tensor> weights{weight,
                                    static_cast<size_t>(weight_stride0)};

  auto num_directions = fn.num_directions;
  auto layer_input = input;
  for (int64_t layer = 0; layer < num_layers; layer++) {
    std::vector<at::Tensor> layer_output(num_directions);
    for (int64_t direction = 0; direction < num_directions; direction++) {
      auto index = layer * num_directions + direction;
      auto layer_weights = weights[index];
      auto layer_hx = hx[index];
      auto layer_hy = hy[index];
      auto layer_cx = cx.defined() ? cx[index] : at::Tensor();
      auto layer_cy =
          cx.defined() ? cy[index] : at::empty({0}, input.options());
      auto reverse = (direction > 0);
      layer_output[direction] =
          mkldnn_rnn_layer(layer_hy, layer_cy, layer_input, layer_weights,
                           layer_hx, layer_cx, reverse, fn);
    }
    layer_input = num_directions == 1
                      ? layer_output[0]
                      : at::cat(layer_output, /*output_channels*/ -1);
  }
  auto output = layer_input;

  if (batch_first && !fn.is_input_packed()) {
    output.transpose_(0, 1);
  }

  return std::make_tuple(output, hy, cy, at::Tensor());
}

namespace {

// Helpers for working with different hidden types.
std::tuple<at::Tensor, at::Tensor> unpack_hidden(const at::Tensor &hidden) {
  return std::make_tuple(hidden, at::Tensor{});
}

std::tuple<at::Tensor, at::Tensor>
unpack_hidden(const std::tuple<at::Tensor, at::Tensor> &hidden) {
  return hidden;
}

template <typename hidden_type>
hidden_type pack_hidden(const at::Tensor &hx, const at::Tensor &cx) {
  static_assert(std::is_same<hidden_type, void>::value,
                "pack_hidden not implemented for this type");
  AT_ERROR("NOT IMPLEMENTED");
}

template <>
at::Tensor pack_hidden<at::Tensor>(const at::Tensor &hx, const at::Tensor &cx) {
  AT_ASSERT(cx.numel() == 0);
  return hx;
}

template <>
std::tuple<at::Tensor, at::Tensor>
pack_hidden<std::tuple<at::Tensor, at::Tensor>>(const at::Tensor &hx,
                                                const at::Tensor &cx) {
  return std::make_tuple(hx, cx);
}

} // anonymous namespace

template <typename hidden_type>
std::pair<at::Tensor, hidden_type>
mkldnn_impl(const at::Tensor &input, const hidden_type &hidden,
            at::TensorList params, bool has_biases, ideep::rnn_kind mode,
            int64_t num_layers, double dropout_p, bool train,
            bool bidirectional, bool batch_first) {

  at::Tensor hx, cx;
  std::tie(hx, cx) = unpack_hidden(hidden);
  int64_t hidden_size = hx.size(2);

  // mkldnn_output = std::tuple<output, hy, cy, workspace>
  auto mkldnn_output =
      mkldnn_rnn(input, params, has_biases ? 4 : 2, hx, cx,
                 static_cast<int>(mode), hidden_size, num_layers, batch_first,
                 dropout_p, train, bidirectional, /*batch_sizes*/ {});

  return {std::get<0>(mkldnn_output),
          pack_hidden<hidden_type>(std::get<1>(mkldnn_output),
                                   std::get<2>(mkldnn_output))};
}

std::tuple<at::Tensor, at::Tensor, at::Tensor>
AtenIpexTypeExt::lstm(const at::Tensor &input, std::vector<at::Tensor> hx,
                      std::vector<at::Tensor> params, bool has_biases,
                      int64_t num_layers, double dropout_p, bool train,
                      bool bidirectional, bool batch_first) {
#if defined(IPEX_DISP_OP)
  printf("IpexExternal::lstm\n");
#endif
#if defined(IPEX_PROFILE_OP)
  RECORD_FUNCTION("IpexExternal::lstm", std::vector<c10::IValue>({}));
#endif
  auto result = mkldnn_impl(input, std::make_tuple(hx[0], hx[1]), params,
                            has_biases, ideep::rnn_kind::LSTM, num_layers,
                            dropout_p, train, bidirectional, batch_first);
  auto output = result.first;
  auto hy = std::get<0>(result.second);
  auto cy = std::get<1>(result.second);

  return std::make_tuple(output, hy, cy);
}

} // namespace torch_ipex

namespace {
static auto dispatch = torch::RegisterOperators().op(
    "torch_ipex::lstm", &torch_ipex::AtenIpexTypeExt::lstm);
}

namespace torch_ipex {
namespace autocast {

std::tuple<at::Tensor, at::Tensor, at::Tensor>
lstm(const at::Tensor &input, std::vector<at::Tensor> hx,
     std::vector<at::Tensor> params, bool has_biases, int64_t num_layers,
     double dropout_p, bool train, bool bidirectional, bool batch_first) {
  c10::impl::ExcludeDispatchKeyGuard no_autocastCPU(DispatchKey::AutocastCPU);
  static auto op = torch::Dispatcher::singleton()
                       .findSchemaOrThrow("torch_ipex::lstm", "")
                       .typed<decltype(lstm)>();
#if defined(ENABLE_AUTOCAST_VERBOSE)
  verbose::OpNameGuard op_name("lstm");
#endif
  auto target_type = get_autocast_dtype();
  // only have bf16 support now, keep fp32 for other target_type
  bool cast_to_bfloat16 =
      !at::GradMode::is_enabled() && at::kBFloat16 == target_type;
  auto casted_input =
      cast_to_bfloat16 ? cpu_cached_cast(at::kBFloat16, input) : input;

  auto casted_hx = cast_to_bfloat16 ? cpu_cached_cast(at::kBFloat16, hx) : hx;
  auto casted_params =
      cast_to_bfloat16 ? cpu_cached_cast(at::kBFloat16, params) : params;
  return op.call(casted_input, casted_hx, casted_params, has_biases, num_layers,
                 dropout_p, train, bidirectional, batch_first);
}

TORCH_LIBRARY_IMPL(torch_ipex, AutocastCPU, m) {
  m.impl("lstm", torch_ipex::autocast::lstm);
}

} // namespace autocast
} // namespace torch_ipex
