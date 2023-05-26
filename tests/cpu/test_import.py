import unittest
import subprocess


class TestImport(unittest.TestCase):
    def test_import_ipex_without_warning(self):
        command = 'python -c "import intel_extension_for_pytorch" '
        with subprocess.Popen(
            command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
        ) as p:
            out = p.stdout.readlines()
            print(out)
            assert "warn" not in out


if __name__ == "__main__":
    unittest.main()
