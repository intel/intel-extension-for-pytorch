# read:  python tools/dep_ver_utils.py -f dependency_version.json -k DEPENDENCY:KEY
# write: python tools/dep_ver_utils.py -w -f dependency_version.json -k DEPENDENCY:KEY -v VALUE

import argparse
import json

def _manipulate_result(result, keys, value='None'):
    ret = None
    if len(keys) == 1:
        if value == 'None':
            ret = result[keys[0]]
        else:
            result[keys[0]] = value
    else:
        key = keys[0]
        if key in result:
            ret = _manipulate_result(result[key], keys[1:], value)
    return ret

def process_file(file, key, value = None, write = False):
    keys = key.split(':')
    with open(file, 'r') as f:
        result = json.load(f)
    if write:
        assert value != 'None', "[ERROR] when modifying the json file, value can not be 'None'"
        _manipulate_result(result, keys, value)
        with open(file, 'w') as f:
            json.dump(result, f)
        return None
    else:
        return _manipulate_result(result, keys)

def main():
    parser = argparse.ArgumentParser(description='JSON configure files utils')

    parser.add_argument('-f', '--file', default='None', type=str,
                        help='JSON file path')
    parser.add_argument('-k', '--key', default='None', type=str,
                        help='key of the dependency. hierarchy separated by :.')
    parser.add_argument('-w', '--write', action='store_true', default=False,
                        help='write values to yaml file')
    parser.add_argument('-v', '--value', default='None', type=str,
                        help='value of the key')
    args = parser.parse_args()
    r = process_file(args.file, args.key, args.value, args.write)
    if not r is None:
        print(r)

if __name__ == '__main__':
    main()
