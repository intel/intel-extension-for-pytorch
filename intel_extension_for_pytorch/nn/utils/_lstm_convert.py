import torch
import copy
from torch.nn.utils.rnn import PackedSequence


class _LSTM(torch.nn.LSTM):
    # This is a solution to swap the lstm module with the ipex counterpart
    # and will upstream this operator to PyTorch when oneDNN support
    # bias and src_iter_c in bf16 in bf16 inference. Will keep this
    # for better support of blocked-format weight, e.g. for training.
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    # port from torch/nn/modules/rnn.py
    # replace the _VF.lstm with torch.ops.torch_ipex.lstm when the input is not PackedSequence
    def forward(self, input, hx=None):  # noqa: F811
        orig_input = input
        # xxx: isinstance check needs to be in conditional for TorchScript to compile
        if isinstance(orig_input, PackedSequence):
            # fallback to PyTorch LSTM since PackedSequence unsupported in oneDNN
            return super(_LSTM, self).forward(input, hx)
        else:
            batch_sizes = None
            max_batch_size = input.size(0) if self.batch_first else input.size(1)
            sorted_indices = None
            unsorted_indices = None

        if hx is None:
            num_directions = 2 if self.bidirectional else 1
            real_hidden_size = (
                self.proj_size if self.proj_size > 0 else self.hidden_size
            )
            h_zeros = torch.zeros(
                self.num_layers * num_directions,
                max_batch_size,
                real_hidden_size,
                dtype=input.dtype,
                device=input.device,
            )
            c_zeros = torch.zeros(
                self.num_layers * num_directions,
                max_batch_size,
                self.hidden_size,
                dtype=input.dtype,
                device=input.device,
            )
            hx = (h_zeros, c_zeros)
        else:
            # Each batch of the hidden state should match the input sequence that
            # the user believes he/she is passing in.
            hx = self.permute_hidden(hx, sorted_indices)

        self.check_forward_args(input, hx, batch_sizes)
        result = torch.ops.torch_ipex.ipex_lstm(
            input,
            hx,
            self._flat_weights,
            self.bias,
            self.num_layers,
            self.dropout,
            self.training,
            self.bidirectional,
            self.batch_first,
        )
        output = result[0]
        hidden = result[1:]

        return output, self.permute_hidden(hidden, unsorted_indices)


def replace_params_in_optimizer(optimizer, param_dict):
    if optimizer is None:
        return
    for group in optimizer.param_groups:
        for i, p in enumerate(group["params"]):
            if p in param_dict:
                new_param = param_dict[p]
                group["params"][i] = new_param
                if p in optimizer.state:
                    optimizer.state[new_param] = optimizer.state.pop(p)


def replace_lstm_with_ipex_lstm(model, optimizer):
    # replace lstm with ipex lstm during inference
    # does not support the case where model itself is torch.nn.LSTM
    for child_name, child in model.named_children():
        if isinstance(child, torch.nn.LSTM):
            assert hasattr(
                child, "weight_ih_l0"
            ), "torch.nn.LSTM should have weight_ih_l0"
            ipex_lstm = _LSTM(
                child.input_size,
                child.hidden_size,
                child.num_layers,
                child.bias,
                child.batch_first,
                child.dropout,
                child.bidirectional,
                child.proj_size,
                child.weight_ih_l0.device,
                child.weight_ih_l0.dtype,
            )
            ipex_lstm.__dict__ = copy.deepcopy(child.__dict__)
            setattr(model, child_name, ipex_lstm)
            param_dict = {}
            original_params = dict(child.named_parameters())
            for name, para in ipex_lstm.named_parameters():
                param_dict.update({original_params[name]: para})
            replace_params_in_optimizer(optimizer, param_dict)
        else:
            replace_lstm_with_ipex_lstm(child, optimizer)
