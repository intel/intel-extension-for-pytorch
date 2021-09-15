import torch
from torch.testing._internal.common_utils import TestCase

import ipex

cpu_device = torch.device("cpu")
dpcpp_device = torch.device("xpu")


def parse_compressed_namedshape(string):
    # This is a metalanguage for describing a shape of a tensor compactly.
    # 'N:3,C:2' -> size = [3, 2], names: ['N', 'C']
    # 'None:3,None:2' -> size = [3, 2], names: ['None', 'None']
    # '3,2' -> size = [3, 2], names=None passed to ctor.
    def parse_name(maybe_name):
        maybe_name = maybe_name.strip()
        if maybe_name == 'None':
            return None
        return maybe_name

    string = string.strip()

    # '' -> size: [], names:None
    if len(string) == 0:
        return None, []

    # '3, 2' -> size = [3, 2], None names.
    if ':' not in string:
        return None, [int(size) for size in string.split(',')]

    dims = string.split(',')
    tuples = [dim.split(':') for dim in dims]
    return zip(*[(parse_name(name), int(size)) for name, size in tuples])


def create(namedshape, factory=torch.randn):
    # namedshape: str
    names, shape = parse_compressed_namedshape(namedshape)
    return factory(shape, names=names)


class TestTorchMethod(TestCase):
    def test_all(self, dtype=torch.float):
        tensor_cpu = create('C:2,N:3,H:5')
        other_cpu = create('N:1,H:1,W:1,C:1')
        output_cpu = tensor_cpu.align_as(other_cpu)

        tensor_xpu = tensor_cpu.to("xpu")
        other_xpu = other_cpu.to("xpu")
        ouput_xpu = tensor_xpu.align_as(other_xpu)

        # ['N', 'H', 'W', 'C']
        self.assertEqual(output_cpu.names, ouput_xpu.names)
        # [3, 5, 1, 2]
        self.assertEqual(output_cpu.shape, ouput_xpu.shape)
