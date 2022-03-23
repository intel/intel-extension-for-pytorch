import torch
import os
import argparse
import re
import numpy as np
import collections
import math
import microbench


# ================ exclude op set ================

onednn_op = [
    '_local_scalar_dense', 'convolution_overrideable', 'convolution_backward_overrideable',
    'max_pool2d_with_indices', 'max_pool2d_with_indices_backward', 'native_batch_norm',
    'native_batch_norm_backward', 'avg_pool2d', 'avg_pool2d_backward', 'relu_',
    'threshold_backward', 'addmm', 'bmm', 'mm', 'gelu', 'native_layer_norm',
    'gelu_backward', 'native_layer_norm_backward'
]

trivial_op = [
    'as_strided', 'copy_', 'contiguous', 'resize_', 'view', '_unsafe_view',
    'stride', 'as_strided', 'empty_like', 'size', 'is_complex', 'equal',
    'is_same_size', 'set_', 'reshape'
]

# ================ /exclude op set ================


benchdtype_to_torchdtype = {
    'float': torch.float,
    'c10::BFloat16': torch.bfloat16,
    'c10::Half': torch.half,
    'double': torch.double,
    'long int': torch.long,
    'unsigned char': torch.uint8,
    'bool': torch.bool,
    'c10::qint8': torch.qint8,
    'c10::quint8': torch.quint8
}


dtype_transfer = {
    'double': 'double',
    'float64': 'double',
    'fp64': 'double',
    'float': 'float',
    'float32': 'float',
    'fp32': 'float',
    'half': 'c10::Half',
    'float16': 'c10::Half',
    'fp16': 'c10::Half',
    'bfloat16': 'c10::BFloat16',
    'bf16': 'c10::BFloat16',
    'long': 'long int',
    'int64': 'long int',
    'bool': 'bool',
    'qint8': 'c10::qint8',
    'quint8': 'c10::quint8',
    'default': None
}


def identify_bench_pipeline(lines: list, exclude: list) -> list:
    datas = []
    for line in lines:
        op_class_name = re.findall(r'^\[.*?\] ', line)
        op_info = re.findall(r' \{.*?\}$', line)
        if len(op_class_name) < 1 or len(op_info) != 1:
            continue
        op_class_name = op_class_name[0]
        op_info = op_info[0]
        params = line[len(op_class_name):-len(op_info)]
        op_class_name = op_class_name.strip()[1:-1]
        op_info = op_info.strip()
        if op_class_name in exclude:
            continue
        params = params.split(';')
        params = [p.strip() for p in params if len(p) > 0]
        op_info = op_info[1:-1].strip()
        caller = 'microbench.' + op_info
        datas.append({
            'op_class_name': op_class_name,
            'params': params,
            'caller': caller
        })
    return datas


def apply_memory_format(input, memory_format: str):
    if memory_format in ['Contiguous', 'Preserve']:
        return input
    elif memory_format == 'ChannelsLast1d':
        return input.to(memory_format=torch.channels_last_1d)
    elif memory_format == 'ChannelsLast3d':
        return input.to(memory_format=torch.channels_last_3d)
    elif memory_format == 'ChannelsLast':
        return input.to(memory_format=torch.channels_last)


def create_randn_tensor(shape, backend, dtype_str, memory_format):
    dtype = benchdtype_to_torchdtype[dtype_str]
    if 'quint8' in dtype_str or 'qint8' in dtype_str:
        zero_point = 0
        scale_in = 0.4
        if len(shape) != 0:
            t = torch.randn(shape).to(backend)
        else:
            t = torch.tensor(1.0).to(backend)
        output = torch.quantize_per_tensor(t, scale_in, zero_point, dtype)
    else:
        if len(shape) != 0:
            output = torch.randn(shape).to(dtype).to(backend)
        else:
            output = torch.tensor(1).to(dtype).to(backend)
    return apply_memory_format(output, memory_format)


def get_true_inputs(params: list, specific_floating_type=None, backend='xpu') -> list:
    inputs = []
    for agm in params:
        if len(agm) < 1 or agm.endswith('}'):
            continue
        type = re.findall(r'\(.*?\)', agm)[0][1:-1].strip()
        raw_value = agm[agm.rfind('):') + 2:].strip()
        if 'undef' in raw_value or 'unknown' in raw_value:
            inputs.append(None)
        elif 'Generator' in type:
            inputs.append(None)
        elif 'ScalarType' in type:
            inputs.append(None)
        elif 'std::array<bool,3>' in type:
            inputs.append(eval(raw_value))
        elif 'Tensor' in type:
            value = raw_value.strip().replace(
                '[', '^ITNFLAG').replace(']', '^ITNFLAG').split('^ITNFLAG')
            dtype = value[0]
            shape = eval('[' + value[1] + ']')
            memory_format = value[2]
            if specific_floating_type is not None and dtype in ['float', 'c10::BFloat16', 'c10::Half']:
                dtype = specific_floating_type
            inputs.append(create_randn_tensor(
                shape, backend, dtype, memory_format))
        elif 'IntArrayRef' in type:
            inputs.append(eval(raw_value))
        elif 'int64_t' in type or 'double' in type or 'float' in type:
            inputs.append(eval(raw_value))
        elif 'bool' in type:
            if raw_value.strip() == '1':
                inputs.append(True)
            else:
                inputs.append(False)
        elif 'Scalar' in type:
            inputs.append(microbench.bench_scalar_slow(eval(raw_value)))
    return inputs


def get_normalized_time(time_: str, base='us') -> float:
    norm = {'s': 1e9, 'ms': 1e6, 'us': 1e3, 'ns': 1.0}
    time_ = time_.strip()
    t = time_[:-2]
    b = time_[-2:]
    try:
        b_ = float(norm[b]) / norm[base]
    except Exception as e:
        b_ = float(1e9) / norm[base]  # s
    try:
        return float(t) * b_
    except Exception as e:
        return float('nan')


def op_filter(infos):
    def get_id(info):
        return "[{0}]{1}".format(info['op_class_name'], ";".join(info['params']))
    dd_count = collections.defaultdict(float)
    dd_time = collections.defaultdict(float)
    for info in infos:
        id = get_id(info)
        dd_count[id] += 1
        dd_time[id] += info['time']
    info_ = []
    for info in infos:
        id = get_id(info)
        if dd_count[id] > 0:
            info['time'] = dd_time[id] / dd_count[id]
            info_.append(info)
            dd_count[id] = 0
    return info_


def get_time_avg(time_info, record_func_name, time_base, backend):
    backend_ = backend.upper().strip() + ' time avg'
    time_avg = -1.0
    for infoline in time_info.split('\n'):
        if 'Name' in infoline and 'Self CPU %' in infoline:
            titles = infoline.split('  ')
            titles = [t.strip() for t in titles if len(t) > 1]
            itemidx = titles.index(backend_)
        if record_func_name in infoline:
            time_avg = infoline.strip().split(' ')
            time_avg = [t for t in time_avg if len(t) > 1]
            time_avg = time_avg[itemidx]
            break
    return get_normalized_time(time_avg, time_base)


def get_io_info(true_inputs, true_outputs):
    inputs_ = []
    for input in true_inputs:
        if isinstance(input, torch.Tensor):
            str_dtype = str(input.dtype).replace('torch.', '')
            str_shape = str(input.shape).replace('torch.Size(', '')
            str_shape = str_shape[:-1]
            inputs_.append('{0}{1}'.format(str_dtype, str_shape))
        else:
            inputs_.append(input)
    if isinstance(true_outputs, torch.Tensor):
        true_outputs = [true_outputs]
    else:
        true_outputs = true_outputs
    outputs_ = []
    for output in true_outputs:
        if isinstance(output, torch.Tensor):
            str_dtype = str(output.dtype).replace('torch.', '')
            str_shape = str(output.shape).replace('torch.Size(', '')
            str_shape = str_shape[:-1]
            outputs_.append('{0}{1}'.format(str_dtype, str_shape))
        else:
            outputs_.append(output)
    return inputs_, outputs_


def get_prof(inputs, func, backend):
    if backend != 'cpu':
        use_backend = 'use_' + backend
        kwargs = {use_backend: True}
    else:
        kwargs = {}
    time_info = None
    if backend == 'xpu':
        with torch.autograd.profiler_legacy.profile(True, **kwargs) as prof:
            output = func(*inputs)
    else:
        with torch.autograd.profiler.profile(True, **kwargs) as prof:
            output = func(*inputs)
    time_info = str(prof.key_averages().table(sort_by="self_cpu_time_total"))
    if backend != 'cpu':
        eval("torch.{0}.synchronize()".format(backend))
    return output, time_info, prof


flush_tensor = None
def flush_cache(backend, shape=(1024, 1024, 1024)):
    global flush_tensor
    if (flush_tensor is None) or (backend not in str(flush_tensor.device)):
        print('create flush tensor')
        flush_tensor = torch.randn(shape).to(backend)
    flush_tensor += 1


def run_op(filename, bench_type=None, backend='XPU', sample=8, time_base='us', exclude=[]):

    backend_lower = backend.strip().lower()
    bench_type = dtype_transfer[bench_type.strip()]

    # e.g. onednn+trivial
    if isinstance(exclude, str):
        if len(exclude) < 1:
            exclude_ = []
        else:
            exclude = exclude.split('+')
            exclude = [t.strip().lower() + '_op' for t in exclude]
            exclude_ = []
            for item in exclude:
                exclude_ += eval(item)
        exclude = exclude_

    assert sample > 1

    # get bench_pipeline first
    with open(filename, 'r') as f:
        lines = f.readlines()
    lines = [line.strip() for line in lines]
    bench_pipeline = identify_bench_pipeline(lines, exclude=exclude)

    infos = []
    for ct, item in enumerate(bench_pipeline):
        try:

            op_class_name = item['op_class_name']
            params = item['params']
            func_name = item['caller']
            print('processing {0}: {1}'.format(op_class_name, func_name))
            print("----args={}".format(params))

            # get true inputs
            inputs = get_true_inputs(
                params, specific_floating_type=bench_type, backend=backend_lower)
            if '_embedding_bag' in op_class_name:
                inputs[1] = torch.arange(
                    inputs[1].shape[0]).long().to(backend_lower)
                inputs[2] = torch.arange(
                    inputs[2].shape[0]).long().to(backend_lower)

            # dry run
            time_avg_ = []
            func = eval(func_name)
            for i in range(sample):
                flush_cache(backend_lower)
                output, time_info, prof = get_prof(inputs, func, backend_lower)
                time_avg_.append(get_time_avg(
                    time_info, 'aten::' + op_class_name, time_base, backend_lower))
            time_avg_ = sorted(time_avg_)
            time_avg = time_avg_[len(time_avg_) // 2]

            inputs_, outputs_ = get_io_info(inputs, output)
            item = {
                'op_class_name': op_class_name,
                'caller': func_name,
                'inputs': inputs_,
                'time': time_avg,
                'outputs': outputs_,
                'params': params,
                'backend': backend_lower
            }
            print(item)
            print(prof.key_averages().table(sort_by="self_cpu_time_total"))
            infos.append(item)

        except Exception as e:
            print("skipping {0}: {1}".format(op_class_name, e))
            continue

    infos = op_filter(infos)
    print('-------------------- output --------------------')
    for info in infos:
        print(info)
    return infos


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MicroBench for Pytorch')
    parser.add_argument('--log', help='path to log file')
    parser.add_argument('--exclude', help='e.g. onednn+trivial', default='')
    parser.add_argument('--backend', help='backend to run', default='cpu')
    parser.add_argument(
        '--dtype', help='specific floating type', default='default')
    args = parser.parse_args()
    if args.backend.strip().lower() == 'xpu':
        import intel_extension_for_pytorch
    infos = run_op(args.log, bench_type=args.dtype,
                   exclude=args.exclude, backend=args.backend)
