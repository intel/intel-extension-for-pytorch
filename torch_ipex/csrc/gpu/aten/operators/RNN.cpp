#include <ATen/native/RNN.h>
#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <ATen/Config.h>

#include <core/DPCPPUtils.h>
#include <core/Runtime.h>


using namespace mkldnn;
using namespace at::dpcpp;

namespace at { namespace AtenIpexTypeDPCPP {

std::tuple<Tensor, Tensor, Tensor> _dpcpp_impl(
      const Tensor& input, const Tensor& hx_, const Tensor& cx_,
      TensorList params, bool has_biases,
      int64_t num_layers_, double dropout_p, bool train, bool bidirectional, bool batch_first) {

  TORCH_CHECK(!batch_first, "_mkldnn_rnn: don't support batch first input");
  Device curDevice = Device(kDPCPP, current_device());
  auto engine = GpuEngineManager::Instance().get_engine(curDevice);
  auto strm = GpuStreamManager::Instance().get_stream();

  int32_t num_layers = num_layers_;
  int32_t hidden_size = hx_.size(-1);
  int32_t seq_length = input.size(0);
  int32_t mini_batch = input.size(1);
  int32_t input_size = input.size(2);
  int32_t num_directions = bidirectional ? 2 : 1;
  int32_t num_gate = 4; //for lstm

  auto layer_hx_0 = hx_.unbind(0);
  auto layer_cx_0 = cx_.unbind(0);
  std::vector<Tensor> layer_hx, layer_cx;

  int32_t total_layers = layer_hx_0.size();
  layer_hx.reserve(total_layers);
  layer_cx.reserve(total_layers);
  for (int32_t i = 0; i < total_layers; i++){
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
  //                    (layer > 0) {hidden_size*num_gate, num_directions*hidden_size}
  // params[index+1] size {hidden_size*num_gate, hidden_size}
  // params[index+2], params[index+3] size {hidden_size*num_gate}
  for (int32_t i = 0; i < num_layers; i++) {
    for (int32_t j = 0; j < num_directions; j++){
      auto layer_input_size = (i == 0) ? input_size : hidden_size * num_directions;
      int32_t index = (i * num_directions + j) * (has_biases ? 4 : 2);
      weight_arr_i.push_back(params[index].t().contiguous());
      weight_arr_h.push_back(params[index+1].t().contiguous());
      if (has_biases) {
        bias_arr.push_back((params[index+2] + params[index+3]));
      } else {
        bias_arr.push_back(at::zeros({num_gate*hidden_size}, params[0].options()));
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
  for (int32_t i = 0; i < total_layers; i++){
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

      auto weights_layer_md = memory::desc(
                {weights_layer_dims}, data_t, memory::format_tag::any);
      auto weights_iter_md = memory::desc(
              {weights_iter_dims}, data_t, memory::format_tag::any);
      auto bias_md
              = memory::desc({bias_dims}, data_t, memory::format_tag::any);
      auto src_layer_md
              = memory::desc({src_layer_dims}, data_t, memory::format_tag::any);
      auto src_iter_md
              = memory::desc({src_iter_dims}, data_t, memory::format_tag::any);
      auto src_iter_c_md = memory::desc(
              {src_iter_c_dims}, data_t, memory::format_tag::any);
      auto dst_layer_md
              = memory::desc({dst_layer_dims}, data_t, memory::format_tag::any);
      auto dst_iter_md
              = memory::desc({dst_iter_dims}, data_t, memory::format_tag::any);
      auto dst_iter_c_md = memory::desc(
              {dst_iter_c_dims}, data_t, memory::format_tag::any);

      std::shared_ptr<lstm_forward::desc> lstm_forward_desc;
      lstm_forward_desc.reset(new lstm_forward::desc(prop_kind::forward_inference,
        dir, src_layer_md, src_iter_md, src_iter_c_md,
        weights_layer_md, weights_iter_md, bias_md,
        dst_layer_md, dst_iter_md, dst_iter_c_md, rnn_flags::undef));

      std::shared_ptr<lstm_forward::primitive_desc> lstm_forward_pd;
      lstm_forward_pd.reset(new lstm_forward::primitive_desc(
        *lstm_forward_desc, engine));

      auto weights_layer_usr_memory = memory({{{weights_layer_dims}, data_t, format_ldigo}, engine});
      dpcpp_set_mkldnn_buffer(weight_arr_i[index].data_ptr(), weights_layer_usr_memory);

      auto weights_iter_usr_memory = memory({{{weights_iter_dims}, data_t, format_ldigo}, engine});
      dpcpp_set_mkldnn_buffer(weight_arr_h[index].data_ptr(), weights_iter_usr_memory);

      auto bias_usr_memory = memory({{{bias_dims}, data_t, format_ldgo}, engine});
      dpcpp_set_mkldnn_buffer(bias_arr[index].data_ptr(), bias_usr_memory);

      auto src_layer_usr_memory = memory({{{src_layer_dims}, data_t, format_tnc}, engine});
      dpcpp_set_mkldnn_buffer(layer_x.data_ptr(), src_layer_usr_memory);

      auto src_iter_usr_memory = memory({{{src_iter_dims}, data_t, format_ldnc}, engine});
      dpcpp_set_mkldnn_buffer(layer_hx[index].data_ptr(), src_iter_usr_memory);

      auto src_iter_c_usr_memory = memory({{{src_iter_c_dims}, data_t, format_ldnc}, engine});
      dpcpp_set_mkldnn_buffer(layer_cx[index].data_ptr(), src_iter_c_usr_memory);

      auto dst_layer_usr_memory = memory({{{dst_layer_dims}, data_t, format_tnc}, engine});
      dpcpp_set_mkldnn_buffer(layer_y[direction].data_ptr(), dst_layer_usr_memory);

      auto dst_iter_usr_memory = memory({{{dst_iter_dims}, data_t, format_ldnc}, engine});
      dpcpp_set_mkldnn_buffer(hy_arr[index].data_ptr(), dst_iter_usr_memory);

      auto dst_iter_c_usr_memory = memory({{{dst_iter_c_dims}, data_t, format_ldnc}, engine});
      dpcpp_set_mkldnn_buffer(cy_arr[index].data_ptr(), dst_iter_c_usr_memory);

      auto expected_weights_layer_md = lstm_forward_pd->weights_layer_desc();
      auto weights_layer_memory = weights_layer_usr_memory;
      if (weights_layer_usr_memory.get_desc() != expected_weights_layer_md) {
        weights_layer_memory = memory(expected_weights_layer_md, engine);
        reorder(weights_layer_usr_memory, weights_layer_memory).
            execute(strm, weights_layer_usr_memory, weights_layer_memory);
      }

      auto expected_weights_iter_md = lstm_forward_pd->weights_iter_desc();
      auto weights_iter_memory = weights_iter_usr_memory;
      if (weights_iter_usr_memory.get_desc() != expected_weights_iter_md) {
        weights_iter_memory = memory(expected_weights_iter_md, engine);
        reorder(weights_iter_usr_memory, weights_iter_memory).
            execute(strm, weights_iter_usr_memory, weights_iter_memory);
      }

      auto expected_bias_md = lstm_forward_pd->bias_desc();
      auto bias_memory = bias_usr_memory;
      if (bias_usr_memory.get_desc() != expected_bias_md) {
        bias_memory = memory(expected_bias_md, engine);
        reorder(bias_usr_memory, bias_memory).
            execute(strm, bias_usr_memory, bias_memory);
      }

      auto expected_src_layer_md = lstm_forward_pd->src_layer_desc();
      auto src_layer_memory = src_layer_usr_memory;
      if (src_layer_usr_memory.get_desc() != expected_src_layer_md) {
        src_layer_memory = memory(expected_src_layer_md, engine);
        reorder(src_layer_usr_memory, src_layer_memory).
            execute(strm, src_layer_usr_memory, src_layer_memory);
      }

      auto expected_src_iter_md = lstm_forward_pd->src_iter_desc();
      auto src_iter_memory = src_iter_usr_memory;
      if (src_iter_usr_memory.get_desc() != expected_src_iter_md) {
        src_iter_memory = memory(expected_src_iter_md, engine);
        reorder(src_iter_usr_memory, src_iter_memory).
            execute(strm, src_iter_usr_memory, src_iter_memory);
      }

      auto expected_src_iter_c_md = lstm_forward_pd->src_iter_c_desc();
      auto src_iter_c_memory = src_iter_c_usr_memory;
      if (src_iter_c_usr_memory.get_desc() != expected_src_iter_c_md) {
        src_iter_c_memory = memory(expected_src_iter_c_md, engine);
        reorder(src_iter_c_usr_memory, src_iter_c_memory).
            execute(strm, src_iter_c_usr_memory, src_iter_c_memory);
      }

      auto expected_dst_layer_md = lstm_forward_pd->dst_layer_desc();
      auto dst_layer_memory = dst_layer_usr_memory;
      if (dst_layer_usr_memory.get_desc() != expected_dst_layer_md) {
        dst_layer_memory = memory(expected_dst_layer_md, engine);
        reorder(dst_layer_usr_memory, dst_layer_memory).
            execute(strm, dst_layer_usr_memory, dst_layer_memory);
      }

      auto expected_dst_iter_md = lstm_forward_pd->dst_iter_desc();
      auto dst_iter_memory = dst_iter_usr_memory;
      if (dst_iter_usr_memory.get_desc() != expected_dst_iter_md) {
        dst_iter_memory = memory(expected_dst_iter_md, engine);
        reorder(dst_iter_usr_memory, dst_iter_memory).
            execute(strm, dst_iter_usr_memory, dst_iter_memory);
      }

      auto expected_dst_iter_c_md = lstm_forward_pd->dst_iter_c_desc();
      auto dst_iter_c_memory = dst_iter_c_usr_memory;
      if (dst_iter_c_usr_memory.get_desc() != expected_dst_iter_c_md) {
        dst_iter_c_memory = memory(expected_dst_iter_c_md, engine);
        reorder(dst_iter_c_usr_memory, dst_iter_c_memory).
            execute(strm, dst_iter_c_usr_memory, dst_iter_c_memory);
      }

      std::shared_ptr<lstm_forward> lstm1_forward;
      lstm1_forward.reset(new lstm_forward(*lstm_forward_pd));
      lstm1_forward->execute(strm, {{DNNL_ARG_SRC_LAYER, src_layer_memory},
                            {DNNL_ARG_SRC_ITER, src_iter_memory},
                            {DNNL_ARG_SRC_ITER_C, src_iter_c_memory},
                            {DNNL_ARG_WEIGHTS_LAYER, weights_layer_memory},
                            {DNNL_ARG_WEIGHTS_ITER, weights_iter_memory},
                            {DNNL_ARG_BIAS, bias_memory},
                            {DNNL_ARG_DST_LAYER, dst_layer_memory},
                            {DNNL_ARG_DST_ITER, dst_iter_memory},
                            {DNNL_ARG_DST_ITER_C, dst_iter_c_memory}});

      if (dst_layer_memory != dst_layer_usr_memory) {
        reorder(dst_layer_memory, dst_layer_usr_memory).
            execute(strm, dst_layer_memory, dst_layer_usr_memory);
      }

      if (dst_iter_memory != dst_iter_usr_memory) {
        reorder(dst_iter_memory, dst_iter_usr_memory).
            execute(strm, dst_iter_memory, dst_iter_usr_memory);
      }

      if (dst_iter_c_memory != dst_iter_c_usr_memory) {
        reorder(dst_iter_c_memory, dst_iter_c_usr_memory).
            execute(strm, dst_iter_c_memory, dst_iter_c_usr_memory);
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

void lstm_dpcpp(Tensor& output, Tensor& hy, Tensor& cy,
      const Tensor& input, TensorList hx,
      TensorList params, bool has_biases,
      int64_t num_layers, double dropout_p, bool train, bool bidirectional, bool batch_first) {
  auto result = _dpcpp_impl(input, hx[0], hx[1], params, has_biases, num_layers, dropout_p, train, bidirectional, batch_first);
  std::tie(output, hy, cy) = result;
}

}} // namespace at::AtenIpexTypeDPCPP
