import math
import torch
from torch.nn.modules.rnn import RNNBase
from torch.nn.utils.rnn import PackedSequence
from torch import _VF

def gru(input, hx, _flat_weights, bias, num_layers, dropout, training, bidirectional, batch_first):
    if input.device == torch.device('dpcpp') and (dropout == 0 or training == False):
        return torch.ops.torch_ipex.gru(input, hx, _flat_weights, bias, num_layers, dropout, training, bidirectional, batch_first)
    else:
        return _VF.gru(input, hx, _flat_weights, bias, num_layers, dropout, training, bidirectional, batch_first)

class GRU(RNNBase):
    r"""Applies a multi-layer gated recurrent unit (GRU) RNN to an input sequence.


    For each element in the input sequence, each layer computes the following
    function:

    .. math::
        \begin{array}{ll}
            r_t = \sigma(W_{ir} x_t + b_{ir} + W_{hr} h_{(t-1)} + b_{hr}) \\
            z_t = \sigma(W_{iz} x_t + b_{iz} + W_{hz} h_{(t-1)} + b_{hz}) \\
            n_t = \tanh(W_{in} x_t + b_{in} + r_t * (W_{hn} h_{(t-1)}+ b_{hn})) \\
            h_t = (1 - z_t) * n_t + z_t * h_{(t-1)}
        \end{array}

    where :math:`h_t` is the hidden state at time `t`, :math:`x_t` is the input
    at time `t`, :math:`h_{(t-1)}` is the hidden state of the layer
    at time `t-1` or the initial hidden state at time `0`, and :math:`r_t`,
    :math:`z_t`, :math:`n_t` are the reset, update, and new gates, respectively.
    :math:`\sigma` is the sigmoid function, and :math:`*` is the Hadamard product.

    In a multilayer GRU, the input :math:`x^{(l)}_t` of the :math:`l` -th layer
    (:math:`l >= 2`) is the hidden state :math:`h^{(l-1)}_t` of the previous layer multiplied by
    dropout :math:`\delta^{(l-1)}_t` where each :math:`\delta^{(l-1)}_t` is a Bernoulli random
    variable which is :math:`0` with probability :attr:`dropout`.

    Args:
        input_size: The number of expected features in the input `x`
        hidden_size: The number of features in the hidden state `h`
        num_layers: Number of recurrent layers. E.g., setting ``num_layers=2``
            would mean stacking two GRUs together to form a `stacked GRU`,
            with the second GRU taking in outputs of the first GRU and
            computing the final results. Default: 1
        bias: If ``False``, then the layer does not use bias weights `b_ih` and `b_hh`.
            Default: ``True``
        batch_first: If ``True``, then the input and output tensors are provided
            as (batch, seq, feature). Default: ``False``
        dropout: If non-zero, introduces a `Dropout` layer on the outputs of each
            GRU layer except the last layer, with dropout probability equal to
            :attr:`dropout`. Default: 0
        bidirectional: If ``True``, becomes a bidirectional GRU. Default: ``False``

    Inputs: input, h_0
        - **input** of shape `(seq_len, batch, input_size)`: tensor containing the features
          of the input sequence. The input can also be a packed variable length
          sequence. See :func:`torch.nn.utils.rnn.pack_padded_sequence`
          for details.
        - **h_0** of shape `(num_layers * num_directions, batch, hidden_size)`: tensor
          containing the initial hidden state for each element in the batch.
          Defaults to zero if not provided. If the RNN is bidirectional,
          num_directions should be 2, else it should be 1.

    Outputs: output, h_n
        - **output** of shape `(seq_len, batch, num_directions * hidden_size)`: tensor
          containing the output features h_t from the last layer of the GRU,
          for each `t`. If a :class:`torch.nn.utils.rnn.PackedSequence` has been
          given as the input, the output will also be a packed sequence.
          For the unpacked case, the directions can be separated
          using ``output.view(seq_len, batch, num_directions, hidden_size)``,
          with forward and backward being direction `0` and `1` respectively.

          Similarly, the directions can be separated in the packed case.
        - **h_n** of shape `(num_layers * num_directions, batch, hidden_size)`: tensor
          containing the hidden state for `t = seq_len`

          Like *output*, the layers can be separated using
          ``h_n.view(num_layers, num_directions, batch, hidden_size)``.

    Shape:
        - Input1: :math:`(L, N, H_{in})` tensor containing input features where
          :math:`H_{in}=\text{input\_size}` and `L` represents a sequence length.
        - Input2: :math:`(S, N, H_{out})` tensor
          containing the initial hidden state for each element in the batch.
          :math:`H_{out}=\text{hidden\_size}`
          Defaults to zero if not provided. where :math:`S=\text{num\_layers} * \text{num\_directions}`
          If the RNN is bidirectional, num_directions should be 2, else it should be 1.
        - Output1: :math:`(L, N, H_{all})` where :math:`H_{all}=\text{num\_directions} * \text{hidden\_size}`
        - Output2: :math:`(S, N, H_{out})` tensor containing the next hidden state
          for each element in the batch

    Attributes:
        weight_ih_l[k] : the learnable input-hidden weights of the :math:`\text{k}^{th}` layer
            (W_ir|W_iz|W_in), of shape `(3*hidden_size, input_size)` for `k = 0`.
            Otherwise, the shape is `(3*hidden_size, num_directions * hidden_size)`
        weight_hh_l[k] : the learnable hidden-hidden weights of the :math:`\text{k}^{th}` layer
            (W_hr|W_hz|W_hn), of shape `(3*hidden_size, hidden_size)`
        bias_ih_l[k] : the learnable input-hidden bias of the :math:`\text{k}^{th}` layer
            (b_ir|b_iz|b_in), of shape `(3*hidden_size)`
        bias_hh_l[k] : the learnable hidden-hidden bias of the :math:`\text{k}^{th}` layer
            (b_hr|b_hz|b_hn), of shape `(3*hidden_size)`

    .. note::
        All the weights and biases are initialized from :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})`
        where :math:`k = \frac{1}{\text{hidden\_size}}`

    .. include:: cudnn_persistent_rnn.rst

    Examples::

        >>> rnn = nn.GRU(10, 20, 2)
        >>> input = torch.randn(5, 3, 10)
        >>> h0 = torch.randn(2, 3, 20)
        >>> output, hn = rnn(input, h0)
    """
    def __init__(self, *args, **kwargs):
        super(GRU, self).__init__('GRU', *args, **kwargs)

    @torch._jit_internal._overload_method  # noqa: F811
    def forward(self, input, hx=None):  # noqa: F811
        # type: (Tensor, Optional[Tensor]) -> Tuple[Tensor, Tensor]
        pass

    @torch._jit_internal._overload_method  # noqa: F811
    def forward(self, input, hx=None):  # noqa: F811
        # type: (PackedSequence, Optional[Tensor]) -> Tuple[PackedSequence, Tensor]
        pass

    def forward(self, input, hx=None):  # noqa: F811
        orig_input = input
        # xxx: isinstance check needs to be in conditional for TorchScript to compile
        if isinstance(orig_input, PackedSequence):
            input, batch_sizes, sorted_indices, unsorted_indices = input
            max_batch_size = batch_sizes[0]
            max_batch_size = int(max_batch_size)
        else:
            batch_sizes = None
            max_batch_size = input.size(0) if self.batch_first else input.size(1)
            sorted_indices = None
            unsorted_indices = None

        if hx is None:
            num_directions = 2 if self.bidirectional else 1
            hx = torch.zeros(self.num_layers * num_directions,
                             max_batch_size, self.hidden_size,
                             dtype=input.dtype, device=input.device)
        else:
            # Each batch of the hidden state should match the input sequence that
            # the user believes he/she is passing in.
            hx = self.permute_hidden(hx, sorted_indices)

        self.check_forward_args(input, hx, batch_sizes)
        if batch_sizes is None:
            result = gru(input, hx, self._flat_weights, self.bias, self.num_layers,
                             self.dropout, self.training, self.bidirectional, self.batch_first)
        else:
            result = _VF.gru(input, batch_sizes, hx, self._flat_weights, self.bias,
                             self.num_layers, self.dropout, self.training, self.bidirectional)
        output = result[0]
        hidden = result[1]

        # xxx: isinstance check needs to be in conditional for TorchScript to compile
        if isinstance(orig_input, PackedSequence):
            output_packed = PackedSequence(output, batch_sizes, sorted_indices, unsorted_indices)
            return output_packed, self.permute_hidden(hidden, unsorted_indices)
        else:
            return output, self.permute_hidden(hidden, unsorted_indices)

torch.nn.GRU = GRU