import os
import sys
import argparse
import torch
import csv
from roofline import RooflineManager
from runbench import run_op


def efficiency_workload(ops_info, spec_file):
    if len(ops_info) <= 0:
        return None
    roofline_manager = RooflineManager(spec_file)
    output_list = []
    for info in ops_info:
        roofline_manager.get_roofline(info)
        output_list.append(info)
    return output_list


def main():
    parser = argparse.ArgumentParser(description='MicroBench for Pytorch')
    parser.add_argument('--log', help='path to log file')
    parser.add_argument('--exclude', help='e.g. onednn+trivial', default='')
    parser.add_argument('--backend', help='backend to run', default='cpu')
    parser.add_argument('--dtype', help='specific floating type', default='default')
    parser.add_argument("--spec", default=None, required=True, help="GenSKU file name")
    args = parser.parse_args()
    if args.backend.strip().lower() == 'xpu':
        import intel_extension_for_pytorch
    infos = run_op(args.log, bench_type=args.dtype, exclude=args.exclude, backend=args.backend)
    infos = efficiency_workload(infos, args.spec)
    filename = args.log[:args.log.rfind('.')] + '_' + args.dtype + '_report.csv'

    def save_csv(filename, infos):
        if isinstance(infos, dict):
            infos = [infos]
        with open(filename, 'w', encoding='utf-8') as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow(["Class", "OP", "Inputs", "Outputs", "InputBytes", "OutputBytes",
                                 "BW", "Type", "BenchTime(us)", "Roofline[us]", "Efficiency(%)"])
            for info in infos:
                if info["efficiency"]:
                    csv_writer.writerow([str(info["class"]), str(info["name"]), str(info["inputs"]), info["outputs"],
                                         info["input_bytes"], info["output_bytes"], info["bw"], info["eff_type"],
                                         info["time"], info["roofline"], info["efficiency"]])
    if infos is not None:
        save_csv(filename, infos)


if __name__ == "__main__":
    sys.exit(main())
