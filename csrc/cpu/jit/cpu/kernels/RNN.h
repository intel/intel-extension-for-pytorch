#pragma once

#include <ATen/Tensor.h>

#include <c10/core/Scalar.h>
#include <torch/csrc/jit/runtime/custom_operator.h>

#include <ideep.hpp>

namespace torch_ipex {
namespace cpu {

//! function: quantized_lstm
/*!
 *
 * Compute a quantized LSTM for INT8 input, INT8 weight and FP32 initial hidden
 and cell states which
 * returns INT8 ouput along with FP32 final hidden and cell states.
 * \param input: INT8 tensor of shape :math:`(L, N, H_{in})` when
 ``batch_first=False`` or
 *         :math:`(N, L, H_{in})` when ``batch_first=True`` containing the
 features of
 *        the input sequence.
 * \param hx: list of FP32 initial hidden state and cell state:
 * hx[0]: FP32 tensor of shape :math:`(D * \text{num\_layers}, N, H_{out})`
 containing the initial hidden
 *         state for the input sequence batch .
 * hx[1]: FP32 tensor of shape :math:`(D * \text{num\_layers}, N, H_{out})`
 containing the initial cell
 *         state for the input sequence batch .
 * \param weights: List of INT8 weights and FP32 biases.
 * \param has_biases: If ``False``, then the layer does not use bias weights
 `b_ih` and `b_hh`.
 * \param num_layers: the number of layers of LSTM.
 * \param dropout_p: If non-zero, introduces a `Dropout` layer on the outputs of
 each RNN layer except the last layer, with dropout probability equal to
            :attr:`dropout` when the model is in training state.
 * \param train: whether the model is in training state.
 * \param bidirectional: If ``True``, becomes a bidirectional LSTM.
 * \param batch_first: If ``True``, then the input and output tensors are
 provided as `(batch, seq, feature)` instead of `(seq, batch, feature)`. Note
 that this does not apply to hidden or cell states.
 * \param scale: the calibration scale of the output in double.
 * \param zp: the calibration zero point of the output in int64_t.
 * \param dtype: the calibration data type of the output.
 * \return: tuple of output tensors:
 * output[0]: INT8 tensor of shape :math:`(L, N, D * H_{out})` when
 ``batch_first=False`` or :math:`(N, L, D * H_{out})` when ``batch_first=True``
 containing the output features
          `(h_t)` from the last layer of the RNN, for each `t`.
 * output[1]: FP32 tensor of shape :math:`(D * \text{num\_layers}, N, H_{out})`
 containing the final hidden state for each element in the batch.
 * output[2]: FP32 tensor of shape :math:`(D * \text{num\_layers}, N, H_{out})`
 containing the final cell state for each element in the batch.
         where:

        .. math::
            \begin{aligned}
                N ={} & \text{batch size} \\
                L ={} & \text{sequence length} \\
                D ={} & 2 \text{ if bidirectional=True otherwise } 1 \\
                H_{in} ={} & \text{input\_size} \\
                H_{out} ={} & \text{hidden\_size}
            \end{aligned}
 */
std::tuple<at::Tensor, at::Tensor, at::Tensor> quantized_lstm(
    const at::Tensor& input,
    c10::List<at::Tensor> hx,
    c10::List<at::Tensor> weights,
    bool has_biases,
    int64_t num_layers,
    double dropout_p,
    bool train,
    bool bidirectional,
    bool batch_first,
    double scale,
    int64_t zp,
    int64_t dtype);

} // namespace cpu
} // namespace torch_ipex
