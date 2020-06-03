# from torch_ipex import _torch_ipex
from sys import version_info
print("ready __init__ ...")
# if version_info >= (2, 6, 0):
#     def swig_import_helper():
#         from os.path import dirname
#         import imp
#         fp = None
#         try:
#             fp, pathname, description = imp.find_module('_torch_ipex', [dirname(__file__)])
#         except ImportError:
#             import _torch_ipex
#             return _torch_ipex
#         if fp is not None:
#             try:
#                 _mod = imp.load_module('_torch_ipex', fp, pathname, description)
#             finally:
#                 fp.close()
#             return _mod
#     _torch_ipex = swig_import_helper()
#     print("import succ !!!")
#     del swig_import_helper
# else:
#     import _torch_ipex

from .lib import torch_ipex
