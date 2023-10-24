import torch
import intel_extension_for_pytorch as ipex  # noqa


class TransducerLoss(torch.nn.Module):
    """Transducer loss
    Detail of this loss function can be found in: Sequence Transduction with Recurrent Neural
    Networks

    Arguments:
        fuse_softmax_backward (bool, optional) whether to fuse the backward of transducer loss with
            softmax. (default: True)
        opt (int, optional): pick the optimization level in [0, 1]. opt=1 picks a more optimized
            algorithm. In some cases, opt=1 might fall back to opt=0. (default: 1)
        packed_input (bool, optional): whether to pack the output in a compact form with don't-care
        data being removed. (default: False)
    """

    def __init__(self, fuse_softmax_backward=False, opt=1, packed_input=False):
        super(TransducerLoss, self).__init__()
        self.fuse_softmax_backward = fuse_softmax_backward
        self.opt = opt
        self.packed_input = packed_input
        self.dummy_batch_offset = torch.empty(0)

    def forward(
        self,
        x,
        label,
        f_len,
        y_len,
        blank_idx,
        batch_offset=None,
        max_f_len=None,
        debug_list=None,
    ):
        """Forward operation of transducer joint

        Arguments:
            x (tensor): input tensor to the loss function with a shape of (B, T, U, H).
            label (tensor): labels for the input data.
            f_len (tensor): lengths of the inputs in the time dimension for each batch.
            y_len (tensor): lengths of the labels for each batch.
            blank_idx (int): index for the null symbol.
            batch_offset (tensor, optional): tensor containing the offset of each batch
                in the input. For example, batch offset can be obtained from:
                batch_offset = torch.cumsum(f_len*(y_len+1), dim=0)
                This argument is required if packed_input == True, and is ignored if
                packed_input == False. (default: None)
            max_f_len (int, optional): maximum length of the input in the time dimension.
                For example, it can be obtained as
                max_f_len = max(f_len)
                This argument is required if packed_input == True, and is ignored if
                packed_input == False. (default: None)
                (default: None)
            debug_list (list, optional): when an empty list is supplied, Alpha and Beta generated
                in the forward operation will be attached to this list for debug purpose.
                (default: None)
        """
        if self.packed_input:
            if batch_offset is None or max_f_len is None:
                raise Exception(
                    "Please specify batch_offset and max_f_len when packing is \
                                    enabled"
                )
            my_batch_offset = batch_offset
            my_max_f_len = max_f_len
        else:
            my_batch_offset = self.dummy_batch_offset
            my_max_f_len = x.size(1)
        x = torch.nn.functional.log_softmax(x, dim=-1)
        return TransducerLossFunc.apply(
            x,
            label,
            f_len,
            y_len,
            my_batch_offset,
            my_max_f_len,
            blank_idx,
            self.fuse_softmax_backward,
            debug_list,
            self.opt,
            self.packed_input,
        )


class TransducerLossFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x,
        label,
        f_len,
        y_len,
        batch_offset,
        max_f_len,
        blank_idx,
        fuse_softmax_backward,
        debug_list,
        opt,
        packed_input,
    ):
        alpha, beta, loss = torch.ops.torch_ipex.transducer_loss_forward(
            x,
            label,
            f_len,
            y_len,
            batch_offset,
            max_f_len,
            blank_idx,
            opt,
            packed_input,
        )
        if debug_list == []:
            debug_list += [alpha, beta]
        ctx.save_for_backward(x, alpha, beta, f_len, y_len, label, batch_offset)
        ctx.blank_idx = blank_idx
        ctx.fuse_softmax_backward = fuse_softmax_backward
        ctx.opt = opt
        ctx.packed_input = packed_input
        ctx.max_f_len = max_f_len
        return loss

    @staticmethod
    def backward(ctx, loss_grad):
        x, alpha, beta, f_len, y_len, label, batch_offset = ctx.saved_tensors
        x_grad = torch.ops.torch_ipex.transducer_loss_backward(
            x,
            loss_grad,
            alpha,
            beta,
            f_len,
            y_len,
            label,
            batch_offset,
            ctx.max_f_len,
            ctx.blank_idx,
            ctx.opt,
            ctx.fuse_softmax_backward,
            ctx.packed_input,
        )
        return x_grad, None, None, None, None, None, None, None, None, None, None
