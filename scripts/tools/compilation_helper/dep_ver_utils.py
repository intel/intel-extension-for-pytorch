# read:  python tools/dep_ver_utils.py -f dependency_version.json -k DEPENDENCY:KEY
# write: python tools/dep_ver_utils.py -w -f dependency_version.json -k DEPENDENCY:KEY -v VALUE

import argparse
import json

parser = argparse.ArgumentParser(description='JSON configure files utils')

parser.add_argument('-f', '--file', default='None', type=str,
                    help='JSON file path')
parser.add_argument('-k', '--key', default='None', type=str,
                    help='key of the dependency. hierarchy separated by :.')
parser.add_argument('-w', '--write', action='store_true', default=False,
                    help='write values to yaml file')
parser.add_argument('-v', '--value', default='None', type=str,
                    help='value of the key')

def manipulate_result(result, keys, value='None'):
    ret = None
    if len(keys) == 1:
        if value == 'None':
            ret = result[keys[0]]
        else:
            result[keys[0]] = value
    else:
        key = keys[0]
        if key in result:
            ret = manipulate_result(result[key], keys[1:], value)
    return ret

def main():
    args = parser.parse_args()
    keys = args.key.split(':')
    with open(args.file, 'r') as f:
        result = json.load(f)
    if args.write:
        assert args.value != 'None', "[ERROR] when modifying the json file, value can not be 'None'"
        manipulate_result(result, keys, args.value)
        with open(args.file, 'w') as f:
            json.dump(result, f)
    else:
        print(manipulate_result(result, keys))

if __name__ == '__main__':
    main()
