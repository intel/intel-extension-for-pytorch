import torch
import ipex
import os
import argparse
import re
import numpy as np
import collections
import math


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


def identify_call_pipeline(lines: list) -> list:
    while len(lines) > 0:
        line = lines[0]
        if 'XPU total' in line and 'XPU time avg' in line and '# of Calls' in line:
            count_of__ = lines[1].count('-')
            lines = lines[2:]
            ret = []
            while len(lines) > 0:
                item = lines[0]
                if item.count('-') == count_of__:
                    return ret
                item = item.strip().split(' ')
                item = [t for t in item if len(t) > 0]
                if item[10] != '1':
                    break
                else:
                    ret.append(item)
                    lines = lines[1:]
        else:
            lines = lines[1:]
    return []


def get_op_model_time(call_pipeline: list):
    output = collections.defaultdict(str)
    for i, t in enumerate(call_pipeline):
        name = t[0].strip()
        if name.startswith('bench'):
            output[name] = call_pipeline[i - 1][9]
    return output


def identify_bench_pipeline(lines: list) -> list:
    datas = []
    for line in lines:
        op_class_name = re.findall(r'^\[.*?\] ', line)
        op_info = re.findall(r' \{.*?\}$', line)
        if len(op_class_name) < 1 or len(op_info) != 1:
            continue
        op_class_name = op_class_name[0]
        op_info = op_info[0]
        params = line[len(op_class_name):-len(op_info)]
        params = params.split(';')
        params = [p.strip() for p in params if len(p) > 0]
        op_info = op_info[2:-1].split(',')
        caller = 'ipex._C.' + op_info[0].strip()
        bench_id = op_info[1].strip()
        datas.append({
            'op_class_name': op_class_name[1:-2],
            'params': params,
            'caller': caller,
            'bench_id': bench_id
        })
    return datas


def create_randn_tensor(shape, backend, dtype_str):
    dtype = benchdtype_to_torchdtype[dtype_str]
    if 'quint8' in dtype_str or 'qint8' in dtype_str:
        zero_point = 0
        scale_in = 0.4
        if len(shape) != 0:
            t = torch.randn(shape).to(backend)
        else:
            t = torch.tensor(1.0).to(backend)
        return torch.quantize_per_tensor(t, scale_in, zero_point, dtype)
    else:
        if len(shape) != 0:
            return torch.randn(shape).to(dtype).to(backend)
        else:
            return torch.tensor(1).to(dtype).to(backend)


def get_true_inputs(params: list, specific_floating_type=None, backend='xpu') -> list:
    inputs = []
    for agm in params:
        if len(agm) < 1 or agm.endswith('}'):
            continue
        type = re.findall(r'\(.*?\)', agm)[0][1:-1].strip()
        raw_value = agm[agm.rfind('):') + 2:].strip()
        if 'optional<Generator>' in type:
            inputs.append(None)
        elif 'optional<ScalarType>' in type:
            value = raw_value.split(',')
            if value[0].strip() == '0':
                inputs.append(None)
            else:
                raise NotImplementedError('No support for' + type)
        elif 'std::array<bool,3>' in type:
            inputs.append(eval(raw_value))
        elif 'optional<Tensor>' in type:
            if 'nullptr' in agm:
                inputs.append(None)
            else:
                value = raw_value.split(',')[1].strip()
                shape = eval(value[value.find('['):])
                dtype = value[:value.find('[')]
                if specific_floating_type is not None and dtype in ['float', 'c10::BFloat16', 'c10::Half']:
                    dtype = specific_floating_type
                inputs.append(create_randn_tensor(shape, backend, dtype))
        elif 'Tensor' in type:
            shape = eval(raw_value[raw_value.find('['):])
            dtype = raw_value[:raw_value.find('[')]
            if specific_floating_type is not None and dtype in ['float', 'c10::BFloat16', 'c10::Half']:
                dtype = specific_floating_type
            inputs.append(create_randn_tensor(shape, backend, dtype))
        elif 'IntArrayRef' in type:
            inputs.append(eval(raw_value))
        elif 'optional<int64_t>' in type or 'optional<double>' in type or 'optional<float>' in type:
            value = raw_value.split(',')
            if value[0].strip() == '0':
                inputs.append(None)
            else:
                inputs.append(eval(value[1]))
        elif 'int64_t' in type or 'double' in type or 'float' in type:
            inputs.append(eval(raw_value))
        elif 'optional<bool>' in type:
            value = raw_value.split(',')
            if value[0].strip() == '0':
                inputs.append(None)
            else:
                if value[1].strip() == '1':
                    inputs.append(True)
                else:
                    inputs.append(False)
        elif 'bool' in type:
            if raw_value.strip() == '1':
                inputs.append(True)
            else:
                inputs.append(False)
        elif 'optional<Scalar>' in type:
            value = raw_value.split(',')
            if value[0].strip() == '0':
                inputs.append(None)
            else:
                inputs.append(ipex._C.bench_scalar_slow(eval(value[1])))
        elif 'Scalar' in type:
            inputs.append(ipex._C.bench_scalar_slow(eval(raw_value)))
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


def op_filter(infos, model_time_filter=True):
    def get_id(info):
        return "[{0}]{1}".format(info['op_class_name'], ";".join(info['params']))
    dd_count = collections.defaultdict(float)
    dd_time = collections.defaultdict(float)
    dd_model_time = collections.defaultdict(float)
    if model_time_filter:
        infos = [t for t in infos if not math.isnan(t['model_time'])]
    for info in infos:
        id = get_id(info)
        dd_count[id] += 1
        dd_time[id] += info['time']
        dd_model_time[id] += info['model_time']
    info_ = []
    for info in infos:
        id = get_id(info)
        if dd_count[id] > 0:
            info['time'] = dd_time[id] / dd_count[id]
            info['model_time'] = dd_model_time[id] / dd_count[id]
            info_.append(info)
            dd_count[id] = 0
    return info_


def get_time_avg(time_info, func_name, time_base):
    time_avg = -1.0
    for infoline in time_info.split('\n'):
        if func_name[len('ipex._C.'):] in infoline:
            time_avg = infoline.strip().split(' ')
            time_avg = [t for t in time_avg if len(t) > 1]
            time_avg = time_avg[9]
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


def run_op(filename, bench_type=None, dpcpp_only=False, backend='XPU', sample=8, outer=[], inner=None, time_base='us', filter_en=True):

    assert sample > 1

    with open(filename, 'r') as f:
        lines = f.readlines()
    lines = [line.strip() for line in lines]
    call_pipeline = identify_call_pipeline(lines.copy())[::-1]
    model_time_dict = get_op_model_time(call_pipeline)
    bench_pipeline = identify_bench_pipeline(lines)

    infos = []
    for ct, line in enumerate(bench_pipeline):
        try:
            op_class_name = line['op_class_name']
            bench_id = line['bench_id']
            func_name = line['caller']
            try:
                model_time = model_time_dict[bench_id]
            except Exception as e:
                model_time = 'null'
            model_time = get_normalized_time(model_time, time_base)
            params = line['params']
            if op_class_name in outer:
                continue
            if inner is not None and op_class_name not in inner:
                continue
            print('processing {0}: {1}'.format(op_class_name, func_name))
            print("----args={}".format(params))
            func = eval(func_name)
            inputs = get_true_inputs(params, specific_floating_type=bench_type)

            if '_embedding_bag' in op_class_name:
                inputs[1] = torch.arange(inputs[1].shape[0]).long().xpu()
                inputs[2] = torch.arange(inputs[2].shape[0]).long().xpu()

            time_avg_ = []
            continue_flag = False
            for i in range(sample):
                with torch.autograd.profiler.profile(True, use_xpu=True) as prof:
                    output = func(*inputs)
                time_info = str(prof.key_averages().table(sort_by="self_cpu_time_total"))
                if dpcpp_only and 'dnnl_' in time_info:
                    continue_flag = True
                    break
                time_avg_.append(get_time_avg(time_info, func_name, time_base))
                torch.xpu.synchronize()
            if continue_flag:
                continue
            time_avg_ = sorted(time_avg_)
            time_avg = time_avg_[len(time_avg_) // 2]

            inputs_, outputs_ = get_io_info(inputs, output)
            item = {
                'op_class_name': op_class_name,
                'caller': func_name,
                'inputs': inputs_,
                'time': time_avg,
                'model_time': model_time,
                'outputs': outputs_,
                'params': params
            }
            print(item)
            print(prof.key_averages().table(sort_by="self_cpu_time_total"))
            infos.append(item)

        except Exception as e:
            print("skipping {0}: {1}".format(op_class_name, e))
            continue

    infos = op_filter(infos, filter_en)
    print('-------------------- output --------------------')
    for info in infos:
        print(info)
    return infos


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MicroBench for Pytorch')
    parser.add_argument('--log', help='path to log file')
    parser.add_argument('--filter', help='0:disable, 1:enable', default='0')
    args = parser.parse_args()
    if int(args.filter) == 0:
        filter_en = False
    else:
        filter_en = True
    infos = run_op(args.log, filter_en=filter_en)
