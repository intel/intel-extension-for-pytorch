import torch
from torch.nn.utils.rnn import PackedSequence

# This is a solution to swap the lstm module with the ipex counterpart
# and will upstream this operator to PyTorch when oneDNN support 
# bias and src_iter_c in bf16 in bf16 inference. Will keep this
# for better support of blocked-format weight, e.g. for training.


class IpexLSTM(torch.nn.LSTM):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    # port from torch/nn/modules/rnn.py 
    # replace the _VF.lstm with torch.ops.torch_ipex.lstm when the input is not PackedSequence
    def forward(self, input, hx=None):  # noqa: F811
        orig_input = input
        # xxx: isinstance check needs to be in conditional for TorchScript to compile
        if isinstance(orig_input, PackedSequence):
            # fallback to PyTorch LSTM since PackedSequence unsupported in oneDNN
            return super(IpexLSTM, self).forward(input, hx)
        else:
            batch_sizes = None
            max_batch_size = input.size(0) if self.batch_first else input.size(1)
            sorted_indices = None
            unsorted_indices = None

        if hx is None:
            num_directions = 2 if self.bidirectional else 1
            real_hidden_size = self.proj_size if self.proj_size > 0 else self.hidden_size
            h_zeros = torch.zeros(self.num_layers * num_directions,
                                  max_batch_size, real_hidden_size,
                                  dtype=input.dtype, device=input.device)
            c_zeros = torch.zeros(self.num_layers * num_directions,
                                  max_batch_size, self.hidden_size,
                                  dtype=input.dtype, device=input.device)
            hx = (h_zeros, c_zeros)
        else:
            # Each batch of the hidden state should match the input sequence that
            # the user believes he/she is passing in.
            hx = self.permute_hidden(hx, sorted_indices)

        self.check_forward_args(input, hx, batch_sizes)
        result = torch.ops.torch_ipex.ipex_lstm(input, hx, self._flat_weights, self.bias, self.num_layers,
                        self.dropout, self.training, self.bidirectional, self.batch_first)
        output = result[0]
        hidden = result[1:]

        return output, self.permute_hidden(hidden, unsorted_indices)
