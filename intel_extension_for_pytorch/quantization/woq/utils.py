import torch
import torch.nn as nn

NUMBER_OF_LINEAR_PER_TRANSFORMER = [6, 7]

DTYPE_BIT_SIZE = {
    torch.uint8: 8,
    torch.int32: 32,
}

def find_layers(module, layers=[nn.Linear], name=''):
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    # We assume the module which contains a specific number of linear is transformer block.
    if len(res) in NUMBER_OF_LINEAR_PER_TRANSFORMER:
        res = {name: res}
    return res

def pack_weight(
    module,
    scales,
    zeros,
    wbits=4,
    group_size=128,
    pack_dtype=torch.uint8,
):
    weight = module.weight.data
    scales = scales.t().contiguous()
    zeros = zeros.t().contiguous()
    weight = weight.t().contiguous()
    scale_zeros = zeros * scales

    intweight = []
    in_features = module.in_features
    out_features = module.out_features
    group_size = in_features if group_size == -1 else group_size
    for idx in range(0, in_features, group_size):
        begin = idx
        end = min(begin + group_size, in_features)
        g_id = begin // group_size
        intweight.append(torch.round(
            (weight[begin:end, :] + scale_zeros[g_id:g_id+1, :]) / scales[g_id:g_id+1, :]).to(torch.int))
    intweight = torch.cat(intweight, dim=0)
  
    pack_bits = DTYPE_BIT_SIZE[pack_dtype]
    pack_factor = pack_bits // wbits
    qweight = torch.zeros(
        [in_features, (out_features + pack_factor - 1) // pack_factor], dtype=pack_dtype)

    zeros -= 1
    zeros = zeros.to(torch.int)
    qzeros = torch.zeros(
        [zeros.shape[0], (out_features + pack_factor - 1) // pack_factor], dtype=pack_dtype)

    i = 0
    col = 0
    while col < qweight.shape[1]:
        j = i
        while j < min(i + pack_factor, intweight.shape[1]):
          qweight[:, col] |= intweight[:, j] << (wbits * (j - i))
          qzeros[:, col] |= zeros[:, j] << (wbits * (j - i))
          j += 1
        i += pack_factor
        col += 1

    return qweight, scales, qzeros