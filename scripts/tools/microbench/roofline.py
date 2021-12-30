import csv
import argparse
from op_roofline_schema import cfg, default_roofline_func


def get_spec(spec_file):
    spec = {}
    with open(spec_file) as f:
        f_csv = csv.reader(f)
        for row in f_csv:
            if "TOPS" in row[0]:
                if "fmad-fp32" in row[0].lower():
                    spec["int32"] = spec["float32"] = float(row[1]) * 1.0e12
                    spec["half"] = spec["float32"] * 2
                    spec["bfloat16"] = spec["float32"] * 2
                    spec["int64"] = spec["float32"] / 2
                    spec["bool"] = spec["float32"] * 4
                    spec["int8"] = spec["bool"]
            elif "TargetFreq(MHz)" in row[0]:
                spec["freq"] = int(row[1]) * 1e6
            elif "HBMLatency" in row[0]:
                spec["hbm_latency"] = int(row[1])
            elif "MemoryBW" in row[0]:
                spec["peak_bw"] = float(row[1]) * 1024 * 1024 * 1024
            elif "MemBWEffForMLWorkloads" in row[0]:
                spec["bw_eff"] = float(row[1])
        if "bw_eff" in spec:
            spec["peak_bw"] *= spec["bw_eff"]
        # the unit of latency is cycle
        spec["latency_bytes"] = spec["peak_bw"] * spec["hbm_latency"] / spec["freq"]
        # print("------------------Latency bytes: {:.3f}-------------------".format(spec["latency_bytes"]))
    return spec


class RooflineManager:
    def __init__(self, spec_file):
        self.spec = get_spec(spec_file)

    def _get_input_pattern(self, info):
        name = info['op_class_name']
        pattern = ''
        for input in info['inputs']:
            if isinstance(input, str):
                pattern += 't'
            else:
                pattern += 'x'
        return name, pattern

    def get_roofline(self, info):
        name, pattern = self._get_input_pattern(info)
        info['name'] = name
        try:
            f = cfg[name][pattern]
        except Exception as e:
            f = default_roofline_func
        if isinstance(f, dict):
            f = default_roofline_func
        return f(info, self.spec)


if __name__ == '__main__':
    from opruntime import run_op
    parser = argparse.ArgumentParser(description='MicroBench for Pytorch')
    parser.add_argument('--log', help='path to log file')
    parser.add_argument('--spec', help='path to log file')
    args = parser.parse_args()
    infos = run_op(args.log)
    mng = RooflineManager(args.spec)
    infos_out = []
    for info in infos:
        try:
            mng.get_roofline(info)
            infos_out.append(info)
        except Exception as e:
            print('skipping ' + str(info) + ' ' + str(e))
    print('-------------------- roofline out --------------------')
    for info in infos_out:
        print(info)
