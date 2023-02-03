import os
import re
import argparse
from ast_convert_tool import convert


def read_file(path):
    with open(path, "r", newline="", encoding="ISO-8859-1") as fin:
        return fin.read()


def write_file(path, content):
    with open(path, "w", encoding="ISO-8859-1") as fin:
        fin.write(content)


def search_and_replace_via_ast(file_list):
    for input_file in file_list:
        convert.run(input_file,
                    execute=False,
                    export=True)


def search_and_replace(from_value, to_value, file_list, in_place, regex_match, verbose):
    change_file_list = []
    change_happen = False
    for input_file in file_list:
        org_content = read_file(input_file)
        dirname, fname = os.path.split(input_file)
        split_fname = list(os.path.splitext(fname))
        new_fname = split_fname[0] + ".xpu" + split_fname[1]
        new_input_file = os.path.join(dirname, new_fname)

        if os.path.exists(new_input_file):
            org_content = read_file(new_input_file)
        if regex_match:
            new_content = re.sub(from_value, to_value, org_content)
        else:
            new_content = org_content.replace(from_value, to_value)
        if org_content != new_content:
            change_happen = True
            change_file_list.append(input_file)
            if verbose:
                print('replace {0} to {1} in file {2}'.format(from_value, to_value, file_list))
            write_file(new_input_file, new_content)

    return change_happen, change_file_list


def main():
    """
    Main function of search and replacement parser.
    """
    parser = argparse.ArgumentParser(description=f"search and replace string parser")
    parser.add_argument('--from-value', '-f', required=True,
                        help='specify the search string')
    parser.add_argument('--file-list', '-l', required=True,
                        help='search the search string from these files, to be parsed in string format: "a,b,c,d,e" NO SPACE in this string!')
    parser.add_argument('--to-value', '-t', required=True,
                        help='specify the replacement string')
    parser.add_argument('--in-place', '-i',
                        action="store_true", help='change files in-place')
    parser.add_argument('--regex-match', '-r',
                        action="store_true", help='use regex matching')
    parser.add_argument('--verbose', '-v',
                        action="store_true", help='turn on verbose mode')
    args = parser.parse_args()
    file_list = args.file_list
    if (args.file_list is not None) and (args.file_list != "Null"):
        file_list = file_list.split(",")
    search_and_replace(args.from_value, args.to_value, file_list,
                       args.in_place, args.regex_match, args.verbose)


if __name__ == "__main__":
    main()
