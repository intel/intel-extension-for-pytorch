import os
import sys
import argparse
import torch
import csv
from roofline import RooflineManager
from opruntime import run_op


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


oneDNNOp = [
    '_local_scalar_dense', 'convolution_overrideable', 'convolution_backward_overrideable',
    'max_pool2d_with_indices', 'max_pool2d_with_indices_backward', 'native_batch_norm',
    'native_batch_norm_backward', 'avg_pool2d', 'avg_pool2d_backward', 'relu_',
    'threshold_backward', 'addmm', 'bmm', 'mm', 'gelu', 'native_layer_norm',
    'gelu_backward', 'native_layer_norm_backward'
]


def efficiency_workload(filename, spec_file, bench_type, dpcpp_only, outer=[]):
    bench_type = dtype_transfer[bench_type]
    ops_info = run_op(filename, bench_type, dpcpp_only, outer=outer, filter_en=False)
    if len(ops_info) <= 0:
        return None
    roofline_manager = RooflineManager(spec_file)
    output_list = []
    for info in ops_info:
        roofline_manager.get_roofline(info)
        output_list.append(info)
    return output_list


def main():
    parser = argparse.ArgumentParser(description="Roofline Computation")
    parser.add_argument("--spec", default=None, required=True, help="GenSKU file name")
    parser.add_argument("--dtype", type=str, default='default', help="tensor dtype")
    parser.add_argument("--workload", default=None, required=True, help="Workload config txt file")
    parser.add_argument("--dpcpp_only", action="store_true", default=False, help="Bench only for dpcpp kernels")
    parser.add_argument("--ignore", default='none', help="Ignore ops")
    args = parser.parse_args()
    args.dtype = args.dtype.lower()
    outer_str = args.ignore.lower().strip()
    outer = []
    if 'onednn' in outer_str:
        outer = oneDNNOp
    infos = None
    if args.workload:
        filename = args.workload[:args.workload.rfind('.')] + '_' + args.dtype + '_roofline.csv'
        infos = efficiency_workload(args.workload, args.spec, args.dtype, args.dpcpp_only, outer=outer)
    else:
        print("MicroBench only supports workload mode now.")

    def save_csv(filename, infos):
        if isinstance(infos, dict):
            infos = [infos]
        with open(filename, 'w', encoding='utf-8') as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow(["Class", "OP", "Inputs", "Outputs", "InputBytes", "OutputBytes",
                                 "BW", "Type", "ModelTime(us)", "BenchTime(us)", "Roofline[us]", "Efficiency(%)"])
            for info in infos:
                if info["efficiency"]:
                    csv_writer.writerow([str(info["class"]), str(info["name"]), str(info["inputs"]), info["outputs"],
                                         info["input_bytes"], info["output_bytes"], info["bw"], info["eff_type"], info["model_time"],
                                         info["time"], info["roofline"], info["efficiency"]])
    if infos is not None:
        save_csv(filename, infos)


if __name__ == "__main__":
    sys.exit(main())
