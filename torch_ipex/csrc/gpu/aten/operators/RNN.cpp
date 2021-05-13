#include <ATen/ATen.h>
#include <ATen/Config.h>
#include <ATen/NativeFunctions.h>
#include <ATen/native/RNN.h>

#include <core/DPCPPUtils.h>
#include <core/Runtime.h>


using namespace dnnl;
using namespace xpu::dpcpp;

namespace at {
namespace AtenIpexTypeXPU {

std::tuple<Tensor, Tensor, Tensor> _dpcpp_impl(
    const Tensor& input,
    const Tensor& hx_,
    const Tensor& cx_,
    TensorList params,
    bool has_biases,
    int64_t num_layers_,
    double dropout_p,
    bool train,
    bool bidirectional,
    bool batch_first) {
  TORCH_CHECK(!batch_first, "oneDNN does not support batch first input");
  Device curDevice = Device(kXPU, current_device());
  auto engine = GpuEngineManager::Instance().get_engine(curDevice);
  auto strm = GpuStreamManager::Instance().get_stream();

  auto num_layers = num_layers_;
  auto hidden_size = hx_.size(-1);
  auto seq_length = input.size(0);
  auto mini_batch = input.size(1);
  auto input_size = input.size(2);
  auto num_directions = bidirectional ? 2 : 1;
  auto num_gate = 4; // for lstm

  auto layer_hx_0 = hx_.unbind(0);
  auto layer_cx_0 = cx_.unbind(0);
  std::vector<Tensor> layer_hx, layer_cx;

  auto total_layers = layer_hx_0.size();
  layer_hx.reserve(total_layers);
  layer_cx.reserve(total_layers);
  for (int32_t i = 0; i < total_layers; i++) {
    auto tensor_h = at::empty_like(layer_hx_0[0]);
    auto tensor_c = at::empty_like(layer_cx_0[0]);
    tensor_h.copy_(layer_hx_0[i]); // layer_hx_0[i] size {mini_batch, hidden_size}
    tensor_c.copy_(layer_cx_0[i]); // layer_cx_0[i] size {mini_batch, hidden_size}
    layer_hx.push_back(tensor_h);
    layer_cx.push_back(tensor_c);
  }

  std::vector<Tensor> weight_arr_i, weight_arr_h, bias_arr;
  weight_arr_i.reserve(total_layers);
  weight_arr_h.reserve(total_layers);
  bias_arr.reserve(total_layers);
  // params[index] size (layer = 0) {hidden_size*num_gate, input_size}
  //                    (layer > 0) {hidden_size*num_gate,
  //                    num_directions*hidden_size}
  // params[index+1] size {hidden_size*num_gate, hidden_size}
  // params[index+2], params[index+3] size {hidden_size*num_gate}
  for (int32_t i = 0; i < num_layers; i++) {
    for (int32_t j = 0; j < num_directions; j++) {
      auto layer_input_size = (i == 0) ? input_size : hidden_size * num_directions;
      int32_t index = (i * num_directions + j) * (has_biases ? 4 : 2);
      weight_arr_i.push_back(params[index].t().contiguous());
      weight_arr_h.push_back(params[index + 1].t().contiguous());
      if (has_biases) {
        bias_arr.push_back((params[index + 2] + params[index + 3]));
      } else {
        bias_arr.push_back(at::zeros({num_gate * hidden_size}, params[0].options()));
      }
    }
  }

  auto data_t = memory::data_type::f32;
  auto format_any = memory::format_tag::any;
  auto format_tnc = memory::format_tag::tnc;
  auto format_ldnc = memory::format_tag::ldnc;
  auto format_ldigo = memory::format_tag::ldigo;
  auto format_ldgo = memory::format_tag::ldgo;

  memory::dims weights_layer_0_dims = {1, 1, input_size, num_gate, hidden_size}; // layer = 0
  memory::dims weights_layer_1_dims = {1, 1, hidden_size * num_directions, num_gate, hidden_size};
  memory::dims weights_iter_dims = {1, 1, hidden_size, num_gate, hidden_size};
  memory::dims bias_dims = {1, 1, num_gate, hidden_size};
  memory::dims src_layer_0_dims = {seq_length, mini_batch, input_size}; // layer = 0
  memory::dims src_layer_1_dims = {seq_length, mini_batch, hidden_size * num_directions};
  memory::dims src_iter_dims = {1, 1, mini_batch, hidden_size};
  memory::dims src_iter_c_dims = {1, 1, mini_batch, hidden_size};
  memory::dims dst_layer_dims = {seq_length, mini_batch, hidden_size};
  memory::dims dst_iter_dims = {1, 1, mini_batch, hidden_size};
  memory::dims dst_iter_c_dims = {1, 1, mini_batch, hidden_size};

  std::vector<Tensor> hy_arr, cy_arr;
  hy_arr.reserve(total_layers);
  cy_arr.reserve(total_layers);
  for (int32_t i = 0; i < total_layers; i++) {
    hy_arr.push_back(at::empty({dst_iter_dims}, hx_.options()));
    cy_arr.push_back(at::empty({dst_iter_c_dims}, cx_.options()));
  }

  auto layer_x = input;
  for (int32_t layer = 0; layer < num_layers; layer++) {
    std::vector<Tensor> layer_y;
    layer_y.reserve(num_directions);
    for (int32_t direction = 0; direction < num_directions; direction++) {
      layer_y.push_back(at::empty({dst_layer_dims}, input.options()));
      auto index = layer * num_directions + direction;
      auto weights_layer_dims = (layer == 0) ? weights_layer_0_dims : weights_layer_1_dims;
      auto src_layer_dims = (layer == 0) ? src_layer_0_dims : src_layer_1_dims;
      auto dir = (direction > 0) ? rnn_direction::unidirectional_right2left
        : rnn_direction::unidirectional_left2right;

      auto weights_layer_md = memory::desc(weights_layer_dims, data_t, memory::format_tag::any);
      auto weights_iter_md = memory::desc(weights_iter_dims, data_t, memory::format_tag::any);
      auto bias_md = memory::desc(bias_dims, data_t, memory::format_tag::any);
      auto src_layer_md = memory::desc(src_layer_dims, data_t, memory::format_tag::any);
      auto src_iter_md = memory::desc(src_iter_dims, data_t, memory::format_tag::any);
      auto src_iter_c_md = memory::desc(src_iter_c_dims, data_t, memory::format_tag::any);
      auto dst_layer_md = memory::desc(dst_layer_dims, data_t, memory::format_tag::any);
      auto dst_iter_md = memory::desc(dst_iter_dims, data_t, memory::format_tag::any);
      auto dst_iter_c_md = memory::desc(dst_iter_c_dims, data_t, memory::format_tag::any);

      auto lstm_forward_desc =  lstm_forward::desc(
          prop_kind::forward_inference,
          dir,
          src_layer_md,
          src_iter_md,
          src_iter_c_md,
          weights_layer_md,
          weights_iter_md,
          bias_md,
          dst_layer_md,
          dst_iter_md,
          dst_iter_c_md,
          rnn_flags::undef);

      auto lstm_forward_pd = lstm_forward::primitive_desc(lstm_forward_desc, engine);

      auto weights_layer_usr_memory = dpcpp_onednn_memory(
          {weights_layer_dims, data_t, format_ldigo}, engine, weight_arr_i[index].data_ptr());

      auto weights_iter_usr_memory = dpcpp_onednn_memory(
          {weights_iter_dims, data_t, format_ldigo}, engine, weight_arr_h[index].data_ptr());

      auto bias_usr_memory = dpcpp_onednn_memory(
          {bias_dims, data_t, format_ldgo}, engine, bias_arr[index].data_ptr());

      auto src_layer_usr_memory = dpcpp_onednn_memory(
          {src_layer_dims, data_t, format_tnc}, engine, layer_x.data_ptr());

      auto src_iter_usr_memory = dpcpp_onednn_memory(
          {src_iter_dims, data_t, format_ldnc}, engine, layer_hx[index].data_ptr());

      auto src_iter_c_usr_memory = dpcpp_onednn_memory(
          {src_iter_c_dims, data_t, format_ldnc}, engine, layer_cx[index].data_ptr());

      auto dst_layer_usr_memory = dpcpp_onednn_memory(
          {dst_layer_dims, data_t, format_tnc}, engine, layer_y[direction].data_ptr());

      auto dst_iter_usr_memory = dpcpp_onednn_memory(
          {dst_iter_dims, data_t, format_ldnc}, engine, hy_arr[index].data_ptr());

      auto dst_iter_c_usr_memory = dpcpp_onednn_memory(
          {dst_iter_c_dims, data_t, format_ldnc}, engine, cy_arr[index].data_ptr());

      auto expected_weights_layer_md = lstm_forward_pd.weights_layer_desc();
      auto weights_layer_memory = weights_layer_usr_memory;
      if (weights_layer_usr_memory.get_desc() != expected_weights_layer_md) {
        weights_layer_memory = memory(expected_weights_layer_md, engine);
        DPCPP_ONEDNN_EXEC(reorder(weights_layer_usr_memory, weights_layer_memory),
            strm, {{DNNL_ARG_FROM, weights_layer_usr_memory}, {DNNL_ARG_TO, weights_layer_memory}});
      }

      auto expected_weights_iter_md = lstm_forward_pd.weights_iter_desc();
      auto weights_iter_memory = weights_iter_usr_memory;
      if (weights_iter_usr_memory.get_desc() != expected_weights_iter_md) {
        weights_iter_memory = memory(expected_weights_iter_md, engine);
        DPCPP_ONEDNN_EXEC(reorder(weights_iter_usr_memory, weights_iter_memory),
            strm, {{DNNL_ARG_FROM, weights_iter_usr_memory}, {DNNL_ARG_TO, weights_iter_memory}});
      }

      auto expected_bias_md = lstm_forward_pd.bias_desc();
      auto bias_memory = bias_usr_memory;
      if (bias_usr_memory.get_desc() != expected_bias_md) {
        bias_memory = memory(expected_bias_md, engine);
        DPCPP_ONEDNN_EXEC(reorder(bias_usr_memory, bias_memory),
            strm, {{DNNL_ARG_FROM, bias_usr_memory}, {DNNL_ARG_TO, bias_memory}});
      }

      auto expected_src_layer_md = lstm_forward_pd.src_layer_desc();
      auto src_layer_memory = src_layer_usr_memory;
      if (src_layer_usr_memory.get_desc() != expected_src_layer_md) {
        src_layer_memory = memory(expected_src_layer_md, engine);
        DPCPP_ONEDNN_EXEC(reorder(src_layer_usr_memory, src_layer_memory),
            strm, {{DNNL_ARG_FROM, src_layer_usr_memory}, {DNNL_ARG_TO, src_layer_memory}});
      }

      auto expected_src_iter_md = lstm_forward_pd.src_iter_desc();
      auto src_iter_memory = src_iter_usr_memory;
      if (src_iter_usr_memory.get_desc() != expected_src_iter_md) {
        src_iter_memory = memory(expected_src_iter_md, engine);
        DPCPP_ONEDNN_EXEC(reorder(src_iter_usr_memory, src_iter_memory),
            strm, {{DNNL_ARG_FROM, src_iter_usr_memory}, {DNNL_ARG_TO, src_iter_memory}});
      }

      auto expected_src_iter_c_md = lstm_forward_pd.src_iter_c_desc();
      auto src_iter_c_memory = src_iter_c_usr_memory;
      if (src_iter_c_usr_memory.get_desc() != expected_src_iter_c_md) {
        src_iter_c_memory = memory(expected_src_iter_c_md, engine);
        DPCPP_ONEDNN_EXEC(reorder(src_iter_c_usr_memory, src_iter_c_memory),
            strm, {{DNNL_ARG_FROM, src_iter_c_usr_memory}, {DNNL_ARG_TO, src_iter_c_memory}});
      }

      auto expected_dst_layer_md = lstm_forward_pd.dst_layer_desc();
      auto dst_layer_memory = dst_layer_usr_memory;
      if (dst_layer_usr_memory.get_desc() != expected_dst_layer_md) {
        dst_layer_memory = memory(expected_dst_layer_md, engine);
        DPCPP_ONEDNN_EXEC(reorder(dst_layer_usr_memory, dst_layer_memory),
            strm, {{DNNL_ARG_FROM, dst_layer_usr_memory}, {DNNL_ARG_TO, dst_layer_memory}});
      }

      auto expected_dst_iter_md = lstm_forward_pd.dst_iter_desc();
      auto dst_iter_memory = dst_iter_usr_memory;
      if (dst_iter_usr_memory.get_desc() != expected_dst_iter_md) {
        dst_iter_memory = memory(expected_dst_iter_md, engine);
        DPCPP_ONEDNN_EXEC(reorder(dst_iter_usr_memory, dst_iter_memory),
            strm, {{DNNL_ARG_FROM, dst_iter_usr_memory}, {DNNL_ARG_TO, dst_iter_memory}});
      }

      auto expected_dst_iter_c_md = lstm_forward_pd.dst_iter_c_desc();
      auto dst_iter_c_memory = dst_iter_c_usr_memory;
      if (dst_iter_c_usr_memory.get_desc() != expected_dst_iter_c_md) {
        dst_iter_c_memory = memory(expected_dst_iter_c_md, engine);
        DPCPP_ONEDNN_EXEC(reorder(dst_iter_c_usr_memory, dst_iter_c_memory),
            strm, {{DNNL_ARG_FROM, dst_iter_c_usr_memory}, {DNNL_ARG_TO, dst_iter_c_memory}});
      }

      auto lstm1_forward = lstm_forward(lstm_forward_pd);
      DPCPP_ONEDNN_EXEC(lstm1_forward, strm,
          {{DNNL_ARG_SRC_LAYER, src_layer_memory},
           {DNNL_ARG_SRC_ITER, src_iter_memory},
           {DNNL_ARG_SRC_ITER_C, src_iter_c_memory},
           {DNNL_ARG_WEIGHTS_LAYER, weights_layer_memory},
           {DNNL_ARG_WEIGHTS_ITER, weights_iter_memory},
           {DNNL_ARG_BIAS, bias_memory},
           {DNNL_ARG_DST_LAYER, dst_layer_memory},
           {DNNL_ARG_DST_ITER, dst_iter_memory},
           {DNNL_ARG_DST_ITER_C, dst_iter_c_memory}});

      if (dst_layer_memory != dst_layer_usr_memory) {
        DPCPP_ONEDNN_EXEC(reorder(dst_layer_memory, dst_layer_usr_memory),
            strm, {{DNNL_ARG_FROM, dst_layer_memory}, {DNNL_ARG_TO, dst_layer_usr_memory}});
      }

      if (dst_iter_memory != dst_iter_usr_memory) {
        DPCPP_ONEDNN_EXEC(reorder(dst_iter_memory, dst_iter_usr_memory),
            strm, {{DNNL_ARG_FROM, dst_iter_memory}, {DNNL_ARG_TO, dst_iter_usr_memory}});
      }

      if (dst_iter_c_memory != dst_iter_c_usr_memory) {
        DPCPP_ONEDNN_EXEC(reorder(dst_iter_c_memory, dst_iter_c_usr_memory),
            strm, {{DNNL_ARG_FROM, dst_iter_c_memory}, {DNNL_ARG_TO, dst_iter_c_usr_memory}});
      }
    }
    if (num_directions == 1) {
      layer_x = layer_y[0];
    } else {
      layer_x = at::cat(layer_y, 2);
    }
  }
  auto output = layer_x;

  auto hy = at::cat(hy_arr, 0).resize_({num_layers * num_directions, mini_batch, hidden_size});
  auto cy = at::cat(cy_arr, 0).resize_({num_layers * num_directions, mini_batch, hidden_size});

  return std::make_tuple(output, hy, cy);
}

void lstm_dpcpp(
    Tensor& output,
    Tensor& hy,
    Tensor& cy,
    const Tensor& input,
    TensorList hx,
    TensorList params,
    bool has_biases,
    int64_t num_layers,
    double dropout_p,
    bool train,
    bool bidirectional,
    bool batch_first) {
  auto result = _dpcpp_impl(
      input,
      hx[0],
      hx[1],
      params,
      has_biases,
      num_layers,
      dropout_p,
      train,
      bidirectional,
      batch_first);
  std::tie(output, hy, cy) = result;
}
} // namespace AtenIpexTypeXPU
} // namespace at
