import os
import argparse
import matplotlib.pyplot as plt
import numpy as np
import csv


def get_info(row, tb):
    inputs = eval(row[tb.index('Inputs')])
    iof32 = row[tb.index('IOBytes(float32)')]
    iof16 = row[tb.index('IOBytes(float16)')]
    iobf16 = row[tb.index('IOBytes(bfloat16)')]
    key = 0
    if len(iof32) > 0:
        key = float(iof32)
    elif len(iof16) > 0:
        key = float(iof16) * 2
    elif len(iobf16) > 0:
        key = float(iobf16) * 2
    numel = key
    eff_float32 = eff_float16 = eff_bfloat16 = -1
    if len(row[tb.index('Efficiency(%)(float32)')]) > 1:
        eff_float32 = float(row[tb.index('Efficiency(%)(float32)')])
    if len(row[tb.index('Efficiency(%)(float16)')]) > 1:
        eff_float16 = float(row[tb.index('Efficiency(%)(float16)')])
    if len(row[tb.index('Efficiency(%)(bfloat16)')]) > 1:
        eff_bfloat16 = float(row[tb.index('Efficiency(%)(bfloat16)')])
    info = {
        'numel': numel,
        'eff_float32': eff_float32,
        'eff_float16': eff_float16,
        'eff_bfloat16': eff_bfloat16
    }
    return info


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MicroBench for Pytorch')
    parser.add_argument('--dir', default='.', help='path to csv dir')
    parser.add_argument('--op', help='op class name', required=True)
    args = parser.parse_args()
    with open(os.path.join(args.dir, 'summary.csv'), 'r') as f:
        reader = csv.reader(f)
        find = False
        infos = []
        for i, row in enumerate(reader):
            if i == 0:
                tb = row
            else:
                if row[1].strip() == args.op.strip():
                    find = True
                    infos.append(get_info(row, tb))
                else:
                    if find:
                        if len(row[1]) > 1:
                            break
                        infos.append(get_info(row, tb))
    if find:
        plt.grid(True)
        plt.xlabel('Numel')
        plt.ylabel('Efficiency(%)')
        plt.title(args.op)
        type_order = ['eff_float32', 'eff_float16', 'eff_bfloat16']
        options = ['rx-', 'bx-', 'gx-']
        for type, option in zip(type_order, options):
            xs = []
            ys = []
            for info in infos:
                eff = info[type]
                numel = info['numel'] / 1024 / 1024
                if eff > 0:
                    xs.append(numel)
                    ys.append(eff)
            plt.plot(xs, ys, option)
        plt.legend(type_order)
        plt.savefig(os.path.join(args.dir, args.op + '.jpg'))
        plt.close()
