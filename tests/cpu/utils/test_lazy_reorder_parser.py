import argparse


def add_args(parser):
    parser.add_argument("--file",
                        help="mkl-dnn verbose log file", default="verbose.log")
    parser.add_argument("--primitive",
                        help="primitive need to analyze", default="all")
    parser.add_argument("--config",
                        action='store_true',
                        help="statistics of each configuration for a specific primitive")

def parse_all(filename='verbose.log', prim='all'):
    primitive_time = {}
    primitive_call = {}

    configurations = {}
    config_call = {}

    with open(filename, 'r') as f:
        for line in f:
            if line.startswith('mkldnn_verbose') or line.startswith('dnnl_verbose'):
                # skip the version line sicne 0.18
                if 'info' in line:
                    continue

                primitive, config, time = parse_line(line)
                if 'backward_data' in line:
                    primitive = primitive + '_backward_data'
                elif 'backward_weights' in line:
                    primitive = primitive + '_backward_weights'
                elif 'backward' in line:
                    primitive = primitive + '_backward'

                if primitive in primitive_time:
                    primitive_time[primitive] += time
                    primitive_call[primitive] += 1
                else:
                    primitive_time[primitive] = time
                    primitive_call[primitive] = 1

                if prim == primitive:
                    if config in configurations:
                        configurations[config] += time
                        config_call[config] += 1
                    else:
                        configurations[config] = time
                        config_call[config] = 1

    return primitive_time, primitive_call, configurations, config_call


def parse_line(line):
    assert len(line) > 1
    if line.startswith('dnnl_verbose'):
        p = line[:-1].split(',')
        primitive = p[3]
        config = p[-2]
        time = float(p[-1])
    else:
        p = line[:-1].split(',')
        primitive = p[2]
        config = p[-2]
        time = float(p[-1])

    return primitive, config, time


def print_title(config=False):
    col1 = 'configuration' if config else 'primtive'
    col2 = 'time (ms)'
    col3 = 'calls'
    print("%-20s    %-20s    %-20s" % (col1, col2, col3))


def print_primitive(primitive, time, call, title=False, config=False):
    if title:
        print_title(config)
    
    print("%-20s    %-20s    %-20s" % (primitive, time, call))


def print_all(time, calls, config=False):
    sorted_pt = sorted(time.items(),
                       key=lambda kv: kv[1], reverse=True)
    print_title(config)
    for p in sorted_pt:
        assert p[0] in calls
        print_primitive(p[0], p[1], calls[p[0]], title=False, config=config)

    tot_time = sum(time.values())
    print("\ntotal time: %s ms for %d items." % (tot_time, len(time)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="mkl-dnn verbose log analysis",             # noqa
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter) # noqa
    add_args(parser)
    args = parser.parse_args()

    pt, pc, config, config_call = parse_all(args.file, prim=args.primitive)
    if args.primitive != 'all' and args.primitive in pt:
        if args.config:
            print_all(config, config_call, config=True)
        else:
            print_primitive(args.primitive, pt[args.primitive], pc[args.primitive], title=True)
    else:
        if args.primitive != 'all':
            import logging
            logging.warning("%s is not called in this log file. Print all primitives as below: \n" % args.primitive)
        print_all(pt, pc)
