from addict import Dict
import numpy as np


_dtype_bytes = {
    'double': 8, 'float64': 8, 'float': 4, 'float32': 4, 'bfloat16': 2,
    'half': 2, 'float16': 2, 'int64': 8, 'long int': 8, 'int32': 4,
    'int8': 1, 'uint8': 1, 'bool': 1, 'c10::BFloat16': 2, 'c10::Half': 2,
    'long int': 8, 'unsigned char': 1, 'c10::qint8': 1, 'c10::quint8': 1
}


def get_total_bytes(info, mode='inputs'):
    total_size = 0
    for item in info[mode]:
        if not isinstance(item, str):
            continue
        try:
            s = item.find('[')
            dtype = item[:s]
            shape = eval(item[s:])
            total_size += np.prod(shape) * _dtype_bytes[dtype]
        except Exception as e:
            print(str(e))
            pass
    return total_size


def get_bytes(item):
    total_size = 0
    if not isinstance(item, str):
        return 0
    try:
        s = item.find('[')
        dtype = item[:s]
        shape = eval(item[s:])
        return np.prod(shape) * _dtype_bytes[dtype]
    except Exception as e:
        return 0


def get_roofline_of_memboundop(info, spec, time_unit=1.0e6):
    input_bytes = get_total_bytes(info, 'inputs')
    output_bytes = get_total_bytes(info, 'outputs')
    total_bytes = input_bytes + output_bytes
    latency_bytes = spec['latency_bytes']
    peak_bw = spec['peak_bw']
    if total_bytes <= latency_bytes:
        print('Latency bound op is not in consideration now.')
        # info['roofline'] = float('nan')
        # info['efficiency'] = float('nan')
        info['roofline'] = total_bytes / float(peak_bw)
        info['roofline'] *= time_unit  # time unit from s to us
        info['efficiency'] = float(info['roofline']) / (info['time'] + 1e-10) * 100.0
    else:
        info['roofline'] = total_bytes / float(peak_bw)
        info['roofline'] *= time_unit  # time unit from s to us
        info['efficiency'] = float(info['roofline']) / (info['time'] + 1e-10) * 100.0
    info['input_bytes'] = input_bytes
    info['output_bytes'] = output_bytes
    info['bw'] = peak_bw
    return None


def info_set(info, eff_type, opclass):
    info['eff_type'] = eff_type
    info['class'] = opclass
    return None


cfg = Dict()


# Now we need to define mehod to get roofline


def default_roofline_func(info, spec):
    # input_bytes = get_total_bytes(info, 'inputs')
    # output_bytes = get_total_bytes(info, 'outputs')
    # total_bytes = input_bytes + output_bytes
    # peak_bw = spec['peak_bw']
    # info['roofline'] = float('nan')
    # info['efficiency'] = float('nan')
    # info['input_bytes'] = input_bytes
    # info['output_bytes'] = output_bytes
    # info['bw'] = peak_bw
    get_roofline_of_memboundop(info, spec)
    info['eff_type'] = 'other'
    info['class'] = 'other'
    return None


cfg.add.txx = lambda info, spec: (get_roofline_of_memboundop(info, spec), info_set(info, 'mem_bound', 'eltwise'))
cfg.add.ttx = lambda info, spec: (get_roofline_of_memboundop(info, spec), info_set(info, 'mem_bound', 'binary'))
cfg.add_.txx = lambda info, spec: (get_roofline_of_memboundop(info, spec), info_set(info, 'mem_bound', 'eltwise'))
cfg.add_.ttx = lambda info, spec: (get_roofline_of_memboundop(info, spec), info_set(info, 'mem_bound', 'binary'))
cfg.add_out.tttx = lambda info, spec: (get_roofline_of_memboundop(info, spec), info_set(info, 'mem_bound', 'binary'))

cfg.mul.tx = lambda info, spec: (get_roofline_of_memboundop(info, spec), info_set(info, 'mem_bound', 'eltwise'))
cfg.mul.tt = lambda info, spec: (get_roofline_of_memboundop(info, spec), info_set(info, 'mem_bound', 'binary'))
cfg.mul_.tt = lambda info, spec: (get_roofline_of_memboundop(info, spec), info_set(info, 'mem_bound', 'binary'))
cfg.mul_.tx = lambda info, spec: (get_roofline_of_memboundop(info, spec), info_set(info, 'mem_bound', 'eltwise'))
cfg.mul_out.ttt = lambda info, spec: (get_roofline_of_memboundop(info, spec), info_set(info, 'mem_bound', 'binary'))

cfg.div.tt = lambda info, spec: (get_roofline_of_memboundop(info, spec), info_set(info, 'mem_bound', 'binary'))
cfg.div_.tt = lambda info, spec: (get_roofline_of_memboundop(info, spec), info_set(info, 'mem_bound', 'binary'))
cfg.div_out.ttt = lambda info, spec: (get_roofline_of_memboundop(info, spec), info_set(info, 'mem_bound', 'binary'))

cfg.eq.tt = lambda info, spec: (get_roofline_of_memboundop(info, spec), info_set(info, 'mem_bound', 'binary'))
cfg.eq.tx = lambda info, spec: (get_roofline_of_memboundop(info, spec), info_set(info, 'mem_bound', 'eltwise'))
cfg.eq_out.ttx = lambda info, spec: (get_roofline_of_memboundop(info, spec), info_set(info, 'mem_bound', 'eltwise'))
cfg.eq_out.ttt = lambda info, spec: (get_roofline_of_memboundop(info, spec), info_set(info, 'mem_bound', 'binary'))

cfg.tanh.t = lambda info, spec: (get_roofline_of_memboundop(info, spec), info_set(info, 'mem_bound', 'eltwise'))
cfg.tanh_.t = lambda info, spec: (get_roofline_of_memboundop(info, spec), info_set(info, 'mem_bound', 'eltwise'))
cfg.tanh_out.tt = lambda info, spec: (get_roofline_of_memboundop(info, spec), info_set(info, 'mem_bound', 'eltwise'))

cfg._masked_scale.ttx = lambda info, spec: (get_roofline_of_memboundop(
    info, spec), info_set(info, 'mem_bound', 'eltwise'))

cfg.rsub.txx = lambda info, spec: (get_roofline_of_memboundop(info, spec), info_set(info, 'mem_bound', 'eltwise'))
cfg.rsub.ttx = lambda info, spec: (get_roofline_of_memboundop(info, spec), info_set(info, 'mem_bound', 'binary'))

cfg.addcmul.tttx = lambda info, spec: (get_roofline_of_memboundop(info, spec), info_set(info, 'mem_bound', 'binary'))
cfg.addcmul_.tttx = lambda info, spec: (get_roofline_of_memboundop(info, spec), info_set(info, 'mem_bound', 'binary'))
cfg.addcmul_out.ttttx = lambda info, spec: (get_roofline_of_memboundop(
    info, spec), info_set(info, 'mem_bound', 'binary'))


def slice_roofline_func(info, spec):
    return (get_roofline_of_memboundop(info, spec), info_set(info, 'mem_bound', 'slice'))


cfg._embedding_bag.tttxxxxx = slice_roofline_func
cfg._embedding_bag.tttxxxtx = slice_roofline_func

cfg.slice.txxxx = slice_roofline_func

cfg.index_select.txt = slice_roofline_func
cfg.index_select_out.ttxt = slice_roofline_func


def slicebwd_roofline_func(info, spec):
    return (get_roofline_of_memboundop(info, spec), info_set(info, 'mem_bound', 'slicebwd'))


cfg.embedding_dense_backward.ttxxx = slicebwd_roofline_func

cfg.nll_loss_backward.ttttxxt = slicebwd_roofline_func
cfg.nll_loss_backward.tttxxxt = slicebwd_roofline_func
cfg.nll_loss_backward_out.tttttxxt = slicebwd_roofline_func


def memset_roofline_func(info, spec):
    return (get_roofline_of_memboundop(info, spec), info_set(info, 'mem_bound', 'memset'))


cfg.zero_.t = memset_roofline_func

cfg.fill_.tx = memset_roofline_func
cfg.fill_.tt = memset_roofline_func

cfg.bernoulli_.ttx = memset_roofline_func
cfg.bernoulli_.txx = memset_roofline_func

cfg.arange_out.txxx = memset_roofline_func


def reduce_roofline_func(info, spec):
    return (get_roofline_of_memboundop(info, spec), info_set(info, 'mem_bound', 'reduce'))


cfg.sum.tx = reduce_roofline_func
cfg.sum.txxx = reduce_roofline_func
cfg.sum_out.ttxxx = reduce_roofline_func

cfg.norm.txx = reduce_roofline_func
cfg.norm.tx = reduce_roofline_func
cfg.norm.txxxx = reduce_roofline_func
cfg.norm.txxx = reduce_roofline_func
cfg.norm_out.ttxxxx = reduce_roofline_func
cfg.norm_out.ttxxx = reduce_roofline_func

cfg.normal_.txxx = reduce_roofline_func
cfg.normal_out.ttxx = reduce_roofline_func
cfg.normal.txx = reduce_roofline_func
cfg.normal_out.txtx = reduce_roofline_func
cfg.normal.xtx = reduce_roofline_func
cfg.normal_out.tttx = reduce_roofline_func
cfg.normal.ttx = reduce_roofline_func

cfg._softmax.txx = reduce_roofline_func
cfg._log_softmax.txx = reduce_roofline_func
cfg._log_softmax_backward_data.ttxt = reduce_roofline_func
cfg._softmax_backward_data.ttxt = reduce_roofline_func

cfg.nll_loss_forward_out.tttttxx = reduce_roofline_func
cfg.nll_loss_forward.tttxx = reduce_roofline_func
cfg.nll_loss_forward.ttxxx = reduce_roofline_func
cfg.nll_loss_backward_out.tttttxxt = reduce_roofline_func
cfg.nll_loss_backward.ttttxxt = reduce_roofline_func
cfg.nll_loss_backward.tttxxxt = reduce_roofline_func

cfg.sigmoid.t = reduce_roofline_func
cfg.sigmoid_.t = reduce_roofline_func
cfg.sigmoid_out.tt = reduce_roofline_func
cfg.sigmoid_backward_out.ttt = reduce_roofline_func
cfg.sigmoid_backward.tt = reduce_roofline_func

cfg._fused_dropout.txx = lambda info, spec: (get_roofline_of_memboundop(
    info, spec), info_set(info, 'mem_bound', 'dropout'))

cfg.topk_out.tttxxxx = lambda info, spec: (get_roofline_of_memboundop(info, spec), info_set(info, 'mem_bound', 'topk'))
cfg.topk.txxxx = lambda info, spec: (get_roofline_of_memboundop(info, spec), info_set(info, 'mem_bound', 'topk'))

cfg.sort_out.tttxx = lambda info, spec: (get_roofline_of_memboundop(info, spec), info_set(info, 'mem_bound', 'sort'))
cfg.sort.txx = lambda info, spec: (get_roofline_of_memboundop(info, spec), info_set(info, 'mem_bound', 'sort'))
