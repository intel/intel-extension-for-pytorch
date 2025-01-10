from torch.testing._internal.common_utils import TestCase


class TestTorchMethod(TestCase):
    def test_autoload(self):
        """Validate that intel_extension_for_pytorch is autoloaded."""

        import sys
        import subprocess

        def check_output(script: str) -> str:
            return (
                subprocess.check_output([sys.executable, "-c", script])
                .decode("ascii")
                .strip()
            )

        test_script = """\
import torch
import sys
print('intel_extension_for_pytorch' in sys.modules.keys())
"""
        rc = check_output(test_script)
        self.assertEqual(rc, str(True))
