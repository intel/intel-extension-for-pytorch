import torch
from torch.testing._internal.common_utils import TestCase

import intel_extension_for_pytorch  # noqa

import subprocess
from itertools import product
import os
from tempfile import mkstemp
import uuid

cpu_device = torch.device("cpu")
xpu_device = torch.device("xpu")
device_pool = [cpu_device, xpu_device]
dtype_pool = [torch.float64, torch.bfloat16]


class TestLauncherXPU(TestCase):
    launcher_scripts = [
        ["python", "-m", "intel_extension_for_pytorch.xpu.launch"],
        ["ipexrun", "xpu"],
    ]

    def _generate_temp_test_file(self, sv, tv, sd, td):
        script_data = f"""import torch

def _check_helper(sample, tv, td):
    if tv == 'cpu':
        assert 'cpu' in str(sample.device) and sample.dtype == td, \\
            'Error convert: Origin dtype should be kept, but got ' \\
            + str(sample.dtype) + ' on ' + str(sample.device)
    elif tv == 'xpu':
        td = torch.float32 if td == torch.double else td
        assert 'xpu' in str(sample.device) and sample.dtype == td, \\
            'Error convert: target dtype should be ' + str(td) + ', but got ' \\
            + str(sample.dtype) + ' on ' + str(sample.device)

sample = torch.randn(3, dtype={td}, device=torch.device('{tv}'))
_check_helper(sample, '{tv}', {td})

target = torch.randn(3, dtype={td}, device=torch.device('{tv}'))
sample = torch.randn_like(target)
_check_helper(sample, '{tv}', {td})

target = torch.randn(3, dtype={sd}, device=torch.device('{sv}'))
sample = torch.randn_like(target, dtype={td}, device=torch.device('{tv}'))
_check_helper(sample, '{tv}', {td})

from_ = torch.randn(3, dtype={sd}, device=torch.device('{sv}'))
sample = from_.to(device=torch.device('{tv}'), dtype={td})
_check_helper(sample, '{tv}', {td})

from_ = torch.randn(3, dtype={sd}, device=torch.device('{sv}'))
sample = from_.to('{tv}', dtype={td})
_check_helper(sample, '{tv}', {td})

from_ = torch.randn(3, dtype={sd}, device=torch.device('{sv}'))
sample = from_.to(dtype={td})
_check_helper(sample, '{sv}', {td})

from_ = torch.randn(3, dtype={sd}, device=torch.device('{sv}'))
sample = from_.to(device='{tv}')
_check_helper(sample, '{tv}', {sd} if '{sv}' == 'cpu' or {sd} != torch.double else torch.float32)

from_ = torch.randn(3, dtype={sd}, device=torch.device('{sv}'))
sample = from_.to('{tv}', {td})
_check_helper(sample, '{tv}', {td})

from_ = torch.randn(3, dtype={sd}, device=torch.device('{sv}'))
sample = from_.to({td})
_check_helper(sample, '{sv}', {td})

from_ = torch.randn(3, dtype={sd}, device=torch.device('{sv}'))
sample = from_.to('{tv}')
_check_helper(sample, '{tv}', {sd} if '{sv}' == 'cpu' or {sd} != torch.double else torch.float32)

from_ = torch.randn(3, dtype={sd}, device=torch.device('{sv}'))
target = torch.randn(3, dtype={td}, device=torch.device('{tv}'))
sample = from_.to(target)
_check_helper(sample, '{tv}', {td} if '{tv}' == 'cpu' or {td} != torch.double else torch.float32)

print("All check passed.")
"""
        program_absolute_path = os.path.abspath(__file__)
        program_absolute_path_dir = os.path.dirname(program_absolute_path)
        generate_file_suffix = (
            str(hash(program_absolute_path)) + str(uuid.uuid1()) + "_test_launcher"
        )
        _, generate_file = mkstemp(
            suffix=generate_file_suffix, dir=program_absolute_path_dir, text=True
        )
        with open(generate_file, "w") as f:
            f.write(script_data)

        return generate_file

    def test_convert_fp64_to_fp32(self):
        dev_prod = product(device_pool, device_pool)
        dtype_prod = product(dtype_pool, dtype_pool)
        mixed_prod = product(dev_prod, dtype_prod)
        for launcher in self.launcher_scripts:
            for (src_dev, tgt_dev), (src_dtype, tgt_dtype) in mixed_prod:
                generate_file = self._generate_temp_test_file(
                    src_dev, tgt_dev, src_dtype, tgt_dtype
                )
                expected_msg = "All check passed."
                cmd = " ".join(launcher) + " --convert-fp64-to-fp32 " + generate_file
                r = subprocess.run(
                    cmd.split(), stdout=subprocess.PIPE, stderr=subprocess.STDOUT
                )
                try:
                    assert r.returncode == 0
                    assert expected_msg in str(r.stdout, "utf-8")
                finally:
                    os.remove(generate_file)
