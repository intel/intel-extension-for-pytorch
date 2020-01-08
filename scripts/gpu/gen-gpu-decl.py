import argparse
import lark
import os
import re
import sys


_GRAMMAR = r"""
    start: type fnname "(" params ")"
    type: CONST? core_type refspec?
    fnname: CNAME
    refspec: REF
           | PTR
    core_type: template
        | TNAME
    template: TNAME "<" typelist ">"
    typelist: type
            | type "," typelist
    REF: "&"
    PTR: "*"
    CONST: "const"
    TNAME: /[a-zA-Z0-9_:]+/
    HEXNUMBER: /0x[0-9a-fA-F]+/
    params: param
          | param "," params
    param: type param_name param_defval?
    param_name: CNAME

    param_defval: "=" init_value
    init_value: "true"
              | "false"
              | "{}"
              | NUMBER
              | SIGNED_NUMBER
              | HEXNUMBER
              | ESCAPED_STRING

    %import common.CNAME -> CNAME
    %import common.NUMBER -> NUMBER
    %import common.SIGNED_NUMBER -> SIGNED_NUMBER
    %import common.ESCAPED_STRING -> ESCAPED_STRING
    %import common.WS
    %ignore WS
    """

_PARSER = lark.Lark(_GRAMMAR, parser='lalr', propagate_positions=True)

_XPARSER = lark.Lark(
    _GRAMMAR, parser='lalr', propagate_positions=True, keep_all_tokens=True)

_DECL_HEADER = """// This file contains all native_functions supported by DPCPP:GPU,
// that can be registered to and the schema string that they should be registered with

{hfuncs}
"""


def gen_output_file(args, name):
  if not args.gpu_decl:
    return sys.stdout
  return open(os.path.join(args.gpu_decl, name), 'w')


def gen_decl_output_file(args):
  return gen_output_file(args, 'RegistrationDeclarations_DPCPP.h')


def is_tensor_api(fndef):
  fndef = fndef.replace('at::', '')
  fndef = fndef.replace('c10::Device', 'Device')
  m = re.search(r'\bTensor\b', fndef)
  return m is not None, fndef


def extract_functions(path):
  functions = []
  errors = []
  for line in open(path, 'r'):
    m = re.match(r'\s*([^\s].*); //\s+(.*)', line)
    if not m:
      continue
    functions.append(line)
    # fndef = m.group(1)
    # try:
    #   _XPARSER.parse(fndef)
    #   functions.append(FuncDef(cpp_sig=fndef, aten_sig=m.group(2)))
    # except Exception as e:
    #   if is_tensor_api(fndef)[0]:
    #     errors.append((fndef, str(e)))
    #     print('Error parsing "{}": {}'.format(fndef, e), file=sys.stderr)
  return functions, errors


def get_mapsig_key(mapsig):
  # PyTorch generates std::tuple<> without space among the tuple types,
  # which would require special understanding in the string rewriter.
  # Since we are using this as simple key, we can just string the spaces.
  return mapsig.replace(' ', '')


def parse_override_keys(path):
  functions = []
  fndef = None
  for line in open(path, 'r'):
    line = line.strip()
    if not fndef:
      m = re.match(r'static\s+(.*);', line)
      if m:
        functions.append(m.group(1))
        continue
      m = re.match(r'static\s+(.*)', line)
      if m:
        fndef = m.group(1)
    else:
      fndef = '{} {}'.format(fndef, line)
      if fndef.endswith(';'):
        functions.append(fndef[:-1])
        fndef = None
  assert fndef is None

  keys = []
  for fndef in functions:
    m = re.search(r'(\s.*)\(', fndef)
    new = m.group(1) + '\('
    if new not in keys:
      keys.append(new)

  return keys


def generate(args):
  fndefs, errors = extract_functions(args.alldecl)
  print(
      'Extracted {} functions ({} errors) from {}'.format(
          len(fndefs), len(errors), args.alldecl),
      file=sys.stderr)
  assert len(errors) == 0

  specific = parse_override_keys(args.ipextype)
  print(
      '{} function dpcpp type in {}'.format(len(specific), args.ipextype),
      file=sys.stderr)

  dedicated = parse_override_keys(args.dedicatedtype)
  print(
      '{} function dedicated overrides in {}'.format(len(dedicated), args.dedicatedtype),
      file=sys.stderr)

  dispatchstub = parse_override_keys(args.dispatchstubtype)
  print(
      '{} function dispatchstub overrides in {}'.format(len(dispatchstub), args.dispatchstubtype),
      file=sys.stderr)

  ordecls = ''
  for fndef in fndefs:
    for override in specific:
      m = re.search(r"{}".format(override), fndef)
      if m:
        ordecls += '{}\n'.format(fndef)
    for override in dedicated:
      m = re.search(r"{}".format(override), fndef)
      if m:
        ordecls += '{}\n'.format(fndef)
    for override in dispatchstub:
      m = re.search(r"{}".format(override), fndef)
      if m:
        ordecls += '{}\n'.format(fndef)

  print(
      _DECL_HEADER.format(gen=os.path.basename(sys.argv[0]), hfuncs=ordecls),
      file=gen_decl_output_file(args))


if __name__ == '__main__':
  arg_parser = argparse.ArgumentParser()
  arg_parser.add_argument('--gpu_decl', type=str)
  arg_parser.add_argument(
      'ipextype',
      type=str,
      metavar='IPEX_TYPE_FILE',
      help='The path to the IPEX ATEN overrides file')
  arg_parser.add_argument(
      'dedicatedtype',
      type=str,
      metavar='DEDICATED_TYPE_FILE',
      help='The path to the DEDICATED ATEN file')
  arg_parser.add_argument(
      'dispatchstubtype',
      type=str,
      metavar='DISPATCH_STUB_TYPE_FILE',
      help='The path to the DISPATCH STUB ATEN file')
  arg_parser.add_argument(
      'alldecl',
      type=str,
      metavar='ALL_PROTOTYPE',
      help='The path to the RegistrationFunctions.h file')
  args, files = arg_parser.parse_known_args()
  for file in files:
    print(file)

  generate(args)
