import math
import torch
import warnings
import numbers

from torch.nn.modules import Module
from torch.nn.parameter import Parameter
from torch.nn.utils.rnn import PackedSequence
from torch.nn import init
from torch import _VF

def rnn_tanh(input, hx, _flat_weights, bias, num_layers, dropout, training, bidirectional, batch_first):
    if input.device.type == 'xpu' and (dropout == 0 or training == False):
        return torch.ops.torch_ipex.rnn_tanh(input, hx, _flat_weights, bias, num_layers, dropout, training, bidirectional, batch_first)
    else:
        if training and input.device.type == 'xpu':
            raise Exception("IPEX does not support RNN-Tanh training if its dropout is not 0. \
                Please explicity convert the gru module and its tensors to CPU and convert the output tensor back to ipex.DEVICE.")
        return _VF.rnn_tanh(input, hx, _flat_weights, bias, num_layers, dropout, training, bidirectional, batch_first)

def rnn_relu(input, hx, _flat_weights, bias, num_layers, dropout, training, bidirectional, batch_first):
    if input.device.type == 'xpu' and (dropout == 0 or training == False):
        return torch.ops.torch_ipex.rnn_relu(input, hx, _flat_weights, bias, num_layers, dropout, training, bidirectional, batch_first)
    else:
        if training and input.device.type == 'xpu':
            raise Exception("IPEX does not support RNN-ReLU training if its dropout is not 0. \
                Please explicity convert the gru module and its tensors to CPU and convert the output tensor back to ipex.DEVICE.")
        return _VF.rnn_relu(input, hx, _flat_weights, bias, num_layers, dropout, training, bidirectional, batch_first)

_rnn_impls = {
    'RNN_TANH': _VF.rnn_tanh,
    'RNN_RELU': _VF.rnn_relu,
}

ipex_rnn_impls = {
    'RNN_TANH': rnn_tanh,
    'RNN_RELU': rnn_relu,
}


def apply_permutation(tensor, permutation, dim=1):
    # type: (Tensor, Tensor, int) -> Tensor
    return tensor.index_select(dim, permutation)


class RNNBase(Module):
    __constants__ = ['mode', 'input_size', 'hidden_size', 'num_layers', 'bias',
                     'batch_first', 'dropout', 'bidirectional']

    def __init__(self, mode, input_size, hidden_size,
                 num_layers=1, bias=True, batch_first=False,
                 dropout=0., bidirectional=False):
        super(RNNBase, self).__init__()
        self.mode = mode
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.batch_first = batch_first
        self.dropout = float(dropout)
        self.bidirectional = bidirectional
        num_directions = 2 if bidirectional else 1

        if not isinstance(dropout, numbers.Number) or not 0 <= dropout <= 1 or \
                isinstance(dropout, bool):
            raise ValueError("dropout should be a number in range [0, 1] "
                             "representing the probability of an element being "
                             "zeroed")
        if dropout > 0 and num_layers == 1:
            warnings.warn("dropout option adds dropout after all but last "
                          "recurrent layer, so non-zero dropout expects "
                          "num_layers greater than 1, but got dropout={} and "
                          "num_layers={}".format(dropout, num_layers))

        if mode == 'LSTM':
            gate_size = 4 * hidden_size
        elif mode == 'GRU':
            gate_size = 3 * hidden_size
        elif mode == 'RNN_TANH':
            gate_size = hidden_size
        elif mode == 'RNN_RELU':
            gate_size = hidden_size
        else:
            raise ValueError("Unrecognized RNN mode: " + mode)

        self._flat_weights_names = []
        self._all_weights = []
        for layer in range(num_layers):
            for direction in range(num_directions):
                layer_input_size = input_size if layer == 0 else hidden_size * num_directions

                w_ih = Parameter(torch.Tensor(gate_size, layer_input_size))
                w_hh = Parameter(torch.Tensor(gate_size, hidden_size))
                b_ih = Parameter(torch.Tensor(gate_size))
                # Second bias vector included for CuDNN compatibility. Only one
                # bias vector is needed in standard definition.
                b_hh = Parameter(torch.Tensor(gate_size))
                layer_params = (w_ih, w_hh, b_ih, b_hh)

                suffix = '_reverse' if direction == 1 else ''
                param_names = ['weight_ih_l{}{}', 'weight_hh_l{}{}']
                if bias:
                    param_names += ['bias_ih_l{}{}', 'bias_hh_l{}{}']
                param_names = [x.format(layer, suffix) for x in param_names]

                for name, param in zip(param_names, layer_params):
                    setattr(self, name, param)
                self._flat_weights_names.extend(param_names)
                self._all_weights.append(param_names)

        self._flat_weights = [(lambda wn: getattr(self, wn) if hasattr(self, wn) else None)(wn) for wn in self._flat_weights_names]
        self.flatten_parameters()
        self.reset_parameters()

    def __setattr__(self, attr, value):
        if hasattr(self, "_flat_weights_names") and attr in self._flat_weights_names:
            # keep self._flat_weights up to date if you do self.weight = ...
            idx = self._flat_weights_names.index(attr)
            self._flat_weights[idx] = value
        super(RNNBase, self).__setattr__(attr, value)

    def flatten_parameters(self):
        """Resets parameter data pointer so that they can use faster code paths.

        Right now, this works only if the module is on the GPU and cuDNN is enabled.
        Otherwise, it's a no-op.
        """
        # Short-circuits if _flat_weights is only partially instantiated
        if len(self._flat_weights) != len(self._flat_weights_names):
            return

        for w in self._flat_weights:
            if not torch.is_tensor(w):
                return
        # Short-circuits if any tensor in self._flat_weights is not acceptable to cuDNN
        # or the tensors in _flat_weights are of different dtypes

        first_fw = self._flat_weights[0]
        dtype = first_fw.dtype
        for fw in self._flat_weights:
            if (not torch.is_tensor(fw.data) or not (fw.data.dtype == dtype) or
                    not fw.data.is_cuda or
                    not torch.backends.cudnn.is_acceptable(fw.data)):
                return

        # If any parameters alias, we fall back to the slower, copying code path. This is
        # a sufficient check, because overlapping parameter buffers that don't completely
        # alias would break the assumptions of the uniqueness check in
        # Module.named_parameters().
        unique_data_ptrs = set(p.data_ptr() for p in self._flat_weights)
        if len(unique_data_ptrs) != len(self._flat_weights):
            return

        with torch.cuda.device_of(first_fw):
            import torch.backends.cudnn.rnn as rnn

            # Note: no_grad() is necessary since _cudnn_rnn_flatten_weight is
            # an inplace operation on self._flat_weights
            with torch.no_grad():
                if torch._use_cudnn_rnn_flatten_weight():
                    torch._cudnn_rnn_flatten_weight(
                        self._flat_weights, (4 if self.bias else 2),
                        self.input_size, rnn.get_cudnn_mode(self.mode), self.hidden_size, self.num_layers,
                        self.batch_first, bool(self.bidirectional))

    def _apply(self, fn):
        ret = super(RNNBase, self)._apply(fn)

        # Resets _flat_weights
        # Note: be v. careful before removing this, as 3rd party device types
        # likely rely on this behavior to properly .to() modules like LSTM.
        self._flat_weights = [(lambda wn: getattr(self, wn) if hasattr(self, wn) else None)(wn) for wn in self._flat_weights_names]
        # Flattens params (on CUDA)
        self.flatten_parameters()

        return ret

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            init.uniform_(weight, -stdv, stdv)

    def check_input(self, input, batch_sizes):
        # type: (Tensor, Optional[Tensor]) -> None
        expected_input_dim = 2 if batch_sizes is not None else 3
        if input.dim() != expected_input_dim:
            raise RuntimeError(
                'input must have {} dimensions, got {}'.format(
                    expected_input_dim, input.dim()))
        if self.input_size != input.size(-1):
            raise RuntimeError(
                'input.size(-1) must be equal to input_size. Expected {}, got {}'.format(
                    self.input_size, input.size(-1)))

    def get_expected_hidden_size(self, input, batch_sizes):
        # type: (Tensor, Optional[Tensor]) -> Tuple[int, int, int]
        if batch_sizes is not None:
            mini_batch = batch_sizes[0]
            mini_batch = int(mini_batch)
        else:
            mini_batch = input.size(0) if self.batch_first else input.size(1)
        num_directions = 2 if self.bidirectional else 1
        expected_hidden_size = (self.num_layers * num_directions,
                                mini_batch, self.hidden_size)
        return expected_hidden_size

    def check_hidden_size(self, hx, expected_hidden_size, msg='Expected hidden size {}, got {}'):
        # type: (Tensor, Tuple[int, int, int], str) -> None
        if hx.size() != expected_hidden_size:
            raise RuntimeError(msg.format(expected_hidden_size, tuple(hx.size())))

    def check_forward_args(self, input, hidden, batch_sizes):
        # type: (Tensor, Tensor, Optional[Tensor]) -> None
        self.check_input(input, batch_sizes)
        expected_hidden_size = self.get_expected_hidden_size(input, batch_sizes)

        self.check_hidden_size(hidden, expected_hidden_size)

    def permute_hidden(self, hx, permutation):
        # type: (Tensor, Optional[Tensor]) -> Tensor
        if permutation is None:
            return hx
        return apply_permutation(hx, permutation)

    def forward(self, input, hx=None):
        is_packed = isinstance(input, PackedSequence)
        if is_packed:
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
        _impl = _rnn_impls[self.mode]
        ipex_impl = ipex_rnn_impls[self.mode]
        if batch_sizes is None:
            result = ipex_impl(input, hx, self._flat_weights, self.bias, self.num_layers,
                           self.dropout, self.training, self.bidirectional, self.batch_first)
        else:
            result = _impl(input, batch_sizes, hx, self._flat_weights, self.bias,
                           self.num_layers, self.dropout, self.training, self.bidirectional)
        output = result[0]
        hidden = result[1]

        if is_packed:
            output = PackedSequence(output, batch_sizes, sorted_indices, unsorted_indices)
        return output, self.permute_hidden(hidden, unsorted_indices)

    def extra_repr(self):
        s = '{input_size}, {hidden_size}'
        if self.num_layers != 1:
            s += ', num_layers={num_layers}'
        if self.bias is not True:
            s += ', bias={bias}'
        if self.batch_first is not False:
            s += ', batch_first={batch_first}'
        if self.dropout != 0:
            s += ', dropout={dropout}'
        if self.bidirectional is not False:
            s += ', bidirectional={bidirectional}'
        return s.format(**self.__dict__)

    def __setstate__(self, d):
        super(RNNBase, self).__setstate__(d)
        if 'all_weights' in d:
            self._all_weights = d['all_weights']

        if isinstance(self._all_weights[0][0], str):
            return
        num_layers = self.num_layers
        num_directions = 2 if self.bidirectional else 1
        self._flat_weights_names = []
        self._all_weights = []
        for layer in range(num_layers):
            for direction in range(num_directions):
                suffix = '_reverse' if direction == 1 else ''
                weights = ['weight_ih_l{}{}', 'weight_hh_l{}{}', 'bias_ih_l{}{}', 'bias_hh_l{}{}']
                weights = [x.format(layer, suffix) for x in weights]
                if self.bias:
                    self._all_weights += [weights]
                    self._flat_weights_names.extend(weights)
                else:
                    self._all_weights += [weights[:2]]
                    self._flat_weights_names.extend(weights[:2])
        self._flat_weights = [(lambda wn: getattr(self, wn) if hasattr(self, wn) else None)(wn) for wn in self._flat_weights_names]

    @property
    def all_weights(self):
        return [[getattr(self, weight) for weight in weights] for weights in self._all_weights]

    def _replicate_for_data_parallel(self):
        replica = super(RNNBase, self)._replicate_for_data_parallel()
        # Need to copy these caches, otherwise the replica will share the same
        # flat weights list.
        replica._flat_weights = replica._flat_weights[:]
        replica._flat_weights_names = replica._flat_weights_names[:]
        return replica


class RNN(RNNBase):
    r"""Applies a multi-layer Elman RNN with :math:`\tanh` or :math:`\text{ReLU}` non-linearity to an
    input sequence.


    For each element in the input sequence, each layer computes the following
    function:

    .. math::
        h_t = \tanh(W_{ih} x_t + b_{ih} + W_{hh} h_{(t-1)} + b_{hh})

    where :math:`h_t` is the hidden state at time `t`, :math:`x_t` is
    the input at time `t`, and :math:`h_{(t-1)}` is the hidden state of the
    previous layer at time `t-1` or the initial hidden state at time `0`.
    If :attr:`nonlinearity` is ``'relu'``, then :math:`\text{ReLU}` is used instead of :math:`\tanh`.

    Args:
        input_size: The number of expected features in the input `x`
        hidden_size: The number of features in the hidden state `h`
        num_layers: Number of recurrent layers. E.g., setting ``num_layers=2``
            would mean stacking two RNNs together to form a `stacked RNN`,
            with the second RNN taking in outputs of the first RNN and
            computing the final results. Default: 1
        nonlinearity: The non-linearity to use. Can be either ``'tanh'`` or ``'relu'``. Default: ``'tanh'``
        bias: If ``False``, then the layer does not use bias weights `b_ih` and `b_hh`.
            Default: ``True``
        batch_first: If ``True``, then the input and output tensors are provided
            as `(batch, seq, feature)`. Default: ``False``
        dropout: If non-zero, introduces a `Dropout` layer on the outputs of each
            RNN layer except the last layer, with dropout probability equal to
            :attr:`dropout`. Default: 0
        bidirectional: If ``True``, becomes a bidirectional RNN. Default: ``False``

    Inputs: input, h_0
        - **input** of shape `(seq_len, batch, input_size)`: tensor containing the features
          of the input sequence. The input can also be a packed variable length
          sequence. See :func:`torch.nn.utils.rnn.pack_padded_sequence`
          or :func:`torch.nn.utils.rnn.pack_sequence`
          for details.
        - **h_0** of shape `(num_layers * num_directions, batch, hidden_size)`: tensor
          containing the initial hidden state for each element in the batch.
          Defaults to zero if not provided. If the RNN is bidirectional,
          num_directions should be 2, else it should be 1.

    Outputs: output, h_n
        - **output** of shape `(seq_len, batch, num_directions * hidden_size)`: tensor
          containing the output features (`h_t`) from the last layer of the RNN,
          for each `t`.  If a :class:`torch.nn.utils.rnn.PackedSequence` has
          been given as the input, the output will also be a packed sequence.

          For the unpacked case, the directions can be separated
          using ``output.view(seq_len, batch, num_directions, hidden_size)``,
          with forward and backward being direction `0` and `1` respectively.
          Similarly, the directions can be separated in the packed case.
        - **h_n** of shape `(num_layers * num_directions, batch, hidden_size)`: tensor
          containing the hidden state for `t = seq_len`.

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
        weight_ih_l[k]: the learnable input-hidden weights of the k-th layer,
            of shape `(hidden_size, input_size)` for `k = 0`. Otherwise, the shape is
            `(hidden_size, num_directions * hidden_size)`
        weight_hh_l[k]: the learnable hidden-hidden weights of the k-th layer,
            of shape `(hidden_size, hidden_size)`
        bias_ih_l[k]: the learnable input-hidden bias of the k-th layer,
            of shape `(hidden_size)`
        bias_hh_l[k]: the learnable hidden-hidden bias of the k-th layer,
            of shape `(hidden_size)`

    .. note::
        All the weights and biases are initialized from :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})`
        where :math:`k = \frac{1}{\text{hidden\_size}}`

    .. include:: cudnn_persistent_rnn.rst

    Examples::

        >>> rnn = nn.RNN(10, 20, 2)
        >>> input = torch.randn(5, 3, 10)
        >>> h0 = torch.randn(2, 3, 20)
        >>> output, hn = rnn(input, h0)
    """

    def __init__(self, *args, **kwargs):
        self.nonlinearity = kwargs.pop('nonlinearity', 'tanh')
        if self.nonlinearity == 'tanh':
            mode = 'RNN_TANH'
        elif self.nonlinearity == 'relu':
            mode = 'RNN_RELU'
        else:
            raise ValueError("Unknown nonlinearity '{}'".format(self.nonlinearity))
        super(RNN, self).__init__(mode, *args, **kwargs)

torch.nn.RNN = RNN
