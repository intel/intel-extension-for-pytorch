from functools import cmp_to_key
import os
import sys
import csv
import argparse
import collections
import re


class SummaryManager:
    def __init__(self, dirname='.', endswith='roofline.csv') -> None:
        self.dirname = dirname
        self.items = []
        for root, _, files in os.walk(dirname):
            for file in files:
                path = os.path.join(root, file)
                if not path.endswith(endswith):
                    continue
                with open(path, 'r') as f:
                    reader = csv.reader(f)
                    for i, row in enumerate(reader):
                        if i == 0:
                            table = row
                            continue
                        info = {}
                        for col, thing in enumerate(row):
                            info[table[col]] = thing
                        self.items.append(info)
        self.cluster_items_()
        self.merge_items_()
        self.sort_items_()

    def cluster_items_(self):
        items_ = collections.defaultdict(dict)
        for item in self.items:
            op_class_name = item['OP']
            input_str = item['Inputs']
            output_str = item['Outputs']
            dtype = 'any'
            if 'bfloat16' in input_str:
                dtype = 'bfloat16'
            elif 'float16' in input_str:
                dtype = 'float16'
            elif 'float32' in input_str:
                dtype = 'float32'
            elif 'int64' in input_str:
                dtype = 'int64'
            input_str = ''.join(re.split(r'<.*?>,', input_str))
            input_str = ''.join(re.split(r'<.*?>', input_str))
            output_str = ''.join(re.split(r'<.*?>,', output_str))
            output_str = ''.join(re.split(r'<.*?>', output_str))
            inputs = eval(input_str)
            outputs = eval(output_str)
            input_tensor_shapes = []
            output_tensor_shapes = []
            for t in inputs:
                if isinstance(t, str):
                    s = t.find('[')
                    shape = eval(t[s:])
                    input_tensor_shapes.append(shape)
            for t in outputs:
                if isinstance(t, str):
                    s = t.find('[')
                    shape = eval(t[s:])
                    output_tensor_shapes.append(shape)
            item['input_str'] = str(input_str)
            item['output_str'] = str(output_str)
            item['input_tensor_shapes'] = str(input_tensor_shapes)
            item['output_tensor_shapes'] = str(output_tensor_shapes)
            item['total_bytes'] = eval(
                item['InputBytes']) + eval(item['OutputBytes'])
            id = "[{0}]{1}".format(op_class_name, str(input_tensor_shapes))
            items_[id][dtype] = item
        self.items = items_

    def merge_items_(self):
        new_order = ['Class', 'OP', 'input_tensor_shapes',
                     'output_tensor_shapes', 'BW', 'Type', 'ModelTime(us)']
        items_ = []
        for key in self.items:
            infos = self.items[key]
            for ct, dtype in enumerate(infos.keys()):
                info = infos[dtype]
                if ct == 0:
                    item = [info[t] for t in new_order]
                    item += [''] * 4 * 3
                obase = len(new_order)
                if dtype == 'float32':
                    offset = obase
                elif dtype == 'float16':
                    offset = obase + 4
                elif dtype == 'bfloat16':
                    offset = obase + 4 * 2
                t_ = ['total_bytes',
                      'BenchTime(us)', 'Roofline[us]', 'Efficiency(%)']
                for i in range(4):
                    item[offset + i] = info[t_[i]]
            items_.append(item)
        self.items = items_

    def sort_items_(self):
        def get_key(row):
            iof32 = row[7]
            iof16 = row[11]
            iobf16 = row[15]
            key = 0
            if not isinstance(iof32, str):
                key = float(iof32)
            elif not isinstance(iof16, str):
                key = float(iof16) * 2
            elif not isinstance(iobf16, str):
                key = float(iobf16) * 2
            return key
        self.items = sorted(self.items, key=get_key)

        def get_key(row):
            return row[0] + row[1]
        self.items = sorted(self.items, key=get_key)

    def write_to_csv(self, filename):
        line0 = ['Class', 'Operator', 'Inputs', 'Outputs', 'BW', 'Type', 'ModelTime',
                 'IOBytes(float32)', 'BenchTime(float32)', 'Roofline(float32)', 'Efficiency(%)(float32)',
                 'IOBytes(float16)', 'BenchTime(float16)', 'Roofline(float16)', 'Efficiency(%)(float16)',
                 'IOBytes(bfloat16)', 'BenchTime(bfloat16)', 'Roofline(bfloat16)', 'Efficiency(%)(bfloat16)']
        filename = os.path.join(self.dirname, filename)
        with open(filename, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(line0)
            prev_class = ''
            prev_op = ''
            for i, row in enumerate(self.items):
                tmp0 = row[0]
                tmp1 = row[1]
                if tmp0 == prev_class:
                    row[0] = ''
                if tmp1 == prev_op:
                    row[1] = ''
                writer.writerow(row)
                prev_class = tmp0
                prev_op = tmp1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MicroBench for Pytorch')
    parser.add_argument('--dir', default='.', help='path to csv dir')
    parser.add_argument('--out', default='summary.csv', help='path to csv out')
    args = parser.parse_args()
    mng = SummaryManager(args.dir)
    mng.write_to_csv(args.out)
