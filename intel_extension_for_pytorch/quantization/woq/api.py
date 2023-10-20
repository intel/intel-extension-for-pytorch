import re
import copy
from collections import namedtuple
import torch
import torch.nn as nn
from .gptq import GPTQ
from .utils import find_layers, pack_weight

QUANT_META = namedtuple('QUANT_META', ['scales', 'zeros'])


@torch.no_grad()
def woq(
    model,
    dataset,
    quantized_ckpt,
    wbits=4,
    perchannel=True,
    symmetric=False,
    group_size=-1,
    pack_dtype=torch.uint8,
    mixed_weight=False,
    param_dtype=torch.float,
):
    r"""
      Apply Quantization to the given transformers model (nn.Module) using gptq method.
      Well supported model list: Llama, GPT-J, OPT, Bloom.

      Args:
        model (torch.nn.Module): User model to apply [2, 3, 4] bits quantization.
        dataset (iterable object):  Calib dataset, batch size should be 1.
        quantized_ckpt (str): Quantized checkpoint name.
        wbits (int): Only works for [2, 3, 4], means quantize weight to int2, int3 or int4.
        perchannel (bool): Control quantization granularity. Default value is ``True``.
        symmetric (bool): Control quantization scheme. Default value is ``False``.
        group_size (int): Group as a block along k dimension. Default value is ``-1``.
        pack_dtype (torch.dtype): Pack int2/int4/int3 to the type. Default value is ``torch.uint8``.
        mixed_weight (bool): Whether mix original weight and quantized weight.
        param_dtype (torch.dtype): Determines the other weight's accuracy except quantized weight.

      .. warning::
        We only support HuggingFace transformers model structure. If provide user-defined model,
        there is no guarantee that quantize can run normally.

      Examples:

        >>> from transformers import GPTJForCausalLM
        >>> model_path = ...
        >>> dataset = ...
        >>> model = GPTJForCausalLM.from_pretrained(model_path)
        >>> model.eval()
        >>> ipex.woq(model, dataset, 'quantized_weight.pt', wbits=4)
    """
    print('Starting ...')

    new_model = copy.deepcopy(model)
    new_model = new_model.to(param_dtype)
    model_state = new_model.state_dict()
    nsamples = len(dataset)
    quant_meta = {}

    model.seqlen = 2048
    for sample in dataset:
        model.seqlen = min(model.seqlen, sample[0].shape[-1])

    # Disable kv cache.
    use_cache = model.config.use_cache
    model.config.use_cache = False

    full = find_layers(model)
    sequential = list(full.keys())
    # Valid transformer block name should ends with numeric string.
    valid = [re.search('[0-9]+', s.split('.')[-1]) is not None for s in sequential]
    names = [s for (s, v) in zip(sequential, valid) if v]
    # Sequential calibration need sort transformer block.
    names.sort(key=lambda x: int(x.split('.')[-1]))
    transformer_blocks = {}
    for name in names:
        transformer_blocks[name] = full[name]

    if len(names) == 0:
        raise ValueError("The model may has no transformer block")

    # Find the common prefix of transformer blocks name.
    transformer_name = names[0][0:names[0].rfind('.')]
    # Get transformer blocks module list.
    layers = eval('model.' + transformer_name)
    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros((nsamples, model.seqlen, model.config.hidden_size), dtype=dtype)
    cache = {'i': 0}

    # Catch the input of transformer block.
    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, hidden_states, **kwargs):
            inps[cache['i']] = hidden_states
            cache['i'] += 1
            cache.update(kwargs)
            raise ValueError

    layers[0] = Catcher(layers[0])
    for sample in dataset:
        try:
            model(sample[0][:, :model.seqlen])
        except ValueError:
            pass

    layers[0] = layers[0].module

    outs = torch.zeros_like(inps)
    del cache['i']

    print('Ready.')

    for i in range(len(layers)):
        print(f'Quantizing layer {i+1}/{len(layers)}..')
        print('+------------------+--------------+------------+-----------+-------+')
        print('|       name       | weight_error | fp_inp_SNR | q_inp_SNR | time  |')
        print('+==================+==============+============+===========+=======+')

        layer = layers[i]
        layer_name = names[i]

        subset = {}
        for name in transformer_blocks[layer_name]:
            # Record the map between linear module name and linear module.
            subset[name[len(layer_name) + 1:]] = transformer_blocks[layer_name][name]
        gptq = {}
        for name in subset:
            gptq[name] = GPTQ(subset[name])
            gptq[name].quantizer.configure(wbits, perchannel=perchannel, sym=symmetric, mse=False)

        def add_batch(name):
            def tmp(_, inp, out):
                gptq[name].add_batch(inp[0].data, out.data)
            return tmp

        handles = []
        for name in subset:
            # Hook to obtain input and output of each linear.
            handles.append(subset[name].register_forward_hook(add_batch(name)))
        for j in range(nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), **cache)[0]
        for h in handles:
            h.remove()

        for name in subset:
            scales, zeros, _, _ = gptq[name].fasterquant(groupsize=group_size, name=name)
            quant_meta[transformer_name + '.%d.%s' % (i, name)] = QUANT_META(scales, zeros)
            gptq[name].free()

        # We need re-run this block due to linear's weight has been changed.
        for j in range(nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), **cache)[0]

        del layer
        del gptq

        inps, outs = outs, inps
        print('+------------------+--------------+------------+-----------+-------+')
        print('\n')

    # Quantize lm head.
    print('Quantizing lm head')
    print('+------------------+--------------+------------+-----------+-------+')
    print('|       name       | weight_error | fp_inp_SNR | q_inp_SNR | time  |')
    print('+==================+==============+============+===========+=======+')

    lm_head = model.lm_head
    gptq_lmhead = GPTQ(lm_head)
    gptq_lmhead.quantizer.configure(wbits, perchannel=perchannel, sym=symmetric, mse=False)

    def add_batch():
        def tmp(_, inp, out):
            gptq_lmhead.add_batch(inp[0].data, out.data)
        return tmp

    lm_head.register_forward_hook(add_batch())
    for j in range(nsamples):
        _ = lm_head(inps[j].unsqueeze(0))

    scales, zeros, _, _ = gptq_lmhead.fasterquant(groupsize=group_size, name="lm_head")
    quant_meta['lm_head'] = QUANT_META(scales, zeros)

    print('+------------------+--------------+------------+-----------+-------+')
    print('\n')

    model.config.use_cache = use_cache
    for block in transformer_blocks:
        for name in transformer_blocks[block]:
            module = transformer_blocks[block][name]
            qweight, scales, qzeros = pack_weight(
                module, quant_meta[name].scales,
                quant_meta[name].zeros, wbits=wbits,
                group_size=group_size,
                pack_dtype=pack_dtype)
            if not mixed_weight:
                del model_state[name + '.weight']
            # Set model state.
            model_state[name + '.qweight'] = qweight
            model_state[name + '.scales'] = scales.to(param_dtype)
            model_state[name + '.qzeros'] = qzeros

    qweight, scales, qzeros = pack_weight(
        lm_head, quant_meta['lm_head'].scales,
        quant_meta['lm_head'].zeros, wbits=wbits,
        group_size=group_size, pack_dtype=pack_dtype)
    if not mixed_weight:
        del model_state['lm_head.weight']
    model_state['lm_head.qweight'] = qweight
    model_state['lm_head.scales'] = scales.to(param_dtype)
    model_state['lm_head.qzeros'] = qzeros

    # Save checkpoint.
    torch.save(model_state, quantized_ckpt)
