# read:  python tools/yaml_utils.py -f dependency_version.yml -d DEPENDENCY -k KEY -v VALUE
# write: python tools/yaml_utils.py -w -f dependency_version.yml -d DEPENDENCY -k KEY -v VALUE

import argparse
import yaml

parser = argparse.ArgumentParser(description='YAML files utils')

parser.add_argument('-f', '--file', default='None', type=str,
                    help='yaml file path')
parser.add_argument('-d', '--dependency', default='None', type=str,
                    help='name of the dependency')
parser.add_argument('-k', '--key', default='None', type=str,
                    help='key of the dependency')
parser.add_argument('-w', '--write', action='store_true', default=False,
                    help='write values to yaml file')
parser.add_argument('-v', '--value', default='None', type=str,
                    help='value of the key')

def main():
    args = parser.parse_args()
    with open(args.file, 'r') as f:
        result = yaml.load(f.read(), Loader=yaml.FullLoader)
    if args.write:
        if 'None' == args.value:
            print("[ERROR] when modifying the yaml file, value should not be 'None'")
            exit(1)
        with open(args.file, 'w') as f:
            result[args.dependency][args.key] = args.value
            yaml.dump(result, f)
    else:
        print(result[args.dependency][args.key])

if __name__ == '__main__':
    main()
