import torch
import unittest
from torch.testing._internal import expecttest
from functools import wraps
import torch_ipex as ipex

class VerboseTestCase(expecttest.TestCase):
    def __init__(self, method_name='runTest'):
        super(expecttest.TestCase, self).__init__(method_name)

    def is_dnnl_verbose(self, line):
        tokens = line.strip().split(',')
        return tokens[0] == 'dnnl_verbose' and len(tokens) == 11

    def is_dnnl_reorder(self, line):
        assert self.is_dnnl_verbose(line)
        return line.strip().split(',')[3] == 'reorder'

    def get_reorder_info(self, line):
        assert self.is_dnnl_reorder(line)
        tokens = line.split(',')
        src_desc, dst_desc = tokens[6].split(' ')
        src_dtype = src_desc.split('::')[0].split('-')
        src_format = src_desc.split('::')[1]
        dst_dtype = dst_desc.split('::')[0].split('-')
        dst_format = dst_desc.split('::')[1]
        return src_dtype, src_format, dst_dtype, dst_format

    def ReorderForPack(self, line):
        if not self.is_dnnl_reorder(line):
            return False
        src_dtype, src_format, dst_dtype, dst_format = self.get_reorder_info(line)
        return src_dtype == dst_dtype

    def OnlyReorderDtype(self, line):
        if not self.is_dnnl_reorder(line):
            return False
        src_dtype, src_format, dst_dtype, dst_format = self.get_reorder_info(line)
        return src_dtype != dst_dtype and src_format == dst_dtype

    def OnlyReorderFormat(self, line):
        if not self.is_dnnl_reorder(line):
            return False
        src_dtype, src_format, dst_dtype, dst_format = self.get_reorder_info(line)
        return src_dtype == dst_dtype and src_format != dst_dtype

    def assertOnlyReorderDtype(self, line):
        assert OnlyReorderDtype(line), 'the verbose msg shows not only reorder dtype'

    def assertOnlyReorderFormat(self, line):
        assert OnlyReorderFormat(line), 'the verbose msg shows not only reorder format'

    def assertNotReorder(self, line):
        assert not is_dnnl_reorder(line)

TEST_MKL = torch.backends.mkl.is_available()

def skipCUDANonDefaultStreamIf(condition):
    def dec(fn):
        if getattr(fn, '_do_cuda_non_default_stream', True):  # if current True
            fn._do_cuda_non_default_stream = not condition
        return fn
    return dec

def suppress_warnings(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fn(*args, **kwargs)
    return wrapper

def skipIfNoLapack(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        if not torch._C.has_lapack:
            raise unittest.SkipTest('PyTorch compiled without Lapack')
        else:
            fn(*args, **kwargs)
    return wrapper

def int8_calibration(model, data, dir):
    conf = ipex.AmpConf(torch.int8)
    with torch.no_grad():
        for x in data:
            with ipex.AutoMixPrecision(conf, running_mode="calibration"):
                model(x)
    conf.save(dir)
