# How to Write and RUN Test Case In Example

## Notice

* These tests are powered by [Pytest](https://docspytest.org/en/stable/).

* Please refer to [Pytest Documents](https://docspytest.org/en/stable/) for more helps.

## Pre-requirements

* PyTorch and Intel GPU Extension for PyTorch have been installed and verified.

* Install Pytest

```bash
python3 -m pip install pytest
```

## Find and Run tests

* All tests are found under the ${PATH_To_Your_Extension_Source_Code}/tests/gpu/ path.
* Or download from the repo with below command.

```bash
git clone --depth=1 https://github.com/intel-innersource/frameworks.ai.pytorch.ipex-gpu.git
```

* Please use pytest to run one test, or all of them if ${Test_Name} is empty.

```bash
pytest ${PATH_To_Your_Extension_Source_Code}/tests/gpu/${Test_Name}
```

## Contribute

1. The file name of each test should start with "test_".
2. The Test Class in file should start with "Test"
3. Test Fuction should start with "test_"
4. Default tolerant value of `assertEqual()` 1e-05, use `assertEqual(True, torch.allclose(y_cpu, y_dpcpp.cpu()), rtol, atol, equal_nan=False)` for different one.

## Example

### General case study:

```python
import torch
import intel_extension_for_pytorch
## import testcase parent class TestCase
from torch.testing._internal.common_utils import TestCase

cpu_device = torch.device("cpu")
dpcpp_device = torch.device("xpu")

## Test Class
## Test Class name should  startswith 'Test'
## TestTorchMethod for torch.xxx
## TestNNMethod for torch.nn.xxx
class TestTorchMethod(TestCase):
    ##  test function
    ##  test function should start  startswith 'test_'
    def test_abs(self, dtype=torch.float):
        data = [[-0.2911, -1.3204,  -2.6425,  -2.4644,  -
                 0.6018, -0.0839, -0.1322, -0.4713, -0.3586, -0.8882]]
        x = torch.tensor(data, device=cpu_device)
        x_dpcpp = x.to(dpcpp_device)
        y = torch.abs(x)
        y_dpcpp = torch.abs(x_dpcpp)
        ## assertion
        self.assertEqual(y, y_dpcpp.to(cpu_device))

    def test_abs2(self, dtype=torch.float):
        data = [[-0.2911, -1.3204,  -2.6425,  -2.4644,  -
                 0.6018, -0.0839, -0.1322, -0.4713, -0.3586, -0.8882]]
        x = torch.tensor(data, device=cpu_device)
        x_dpcpp = x.to(dpcpp_device)
        y = torch.abs(x)
        y_dpcpp = torch.abs(x_dpcpp)
        ## assertion
        self.assertEqual(y, y_dpcpp.to(cpu_device))

class TestTorchMethod2(TestCase):
    def test_abs3(self, dtype=torch.float):
        data = [[-0.2911, -1.3204,  -2.6425,  -2.4644,  -
                 0.6018, -0.0839, -0.1322, -0.4713, -0.3586, -0.8882]]
        x = torch.tensor(data, device=cpu_device)
        x_dpcpp = x.to(dpcpp_device)
        y = torch.abs(x)
        y_dpcpp = torch.abs(x_dpcpp)
        self.assertEqual(y, y_dpcpp.to(cpu_device))
```

### Skip certain case with reason and comments:
```python
import torch
import intel_extension_for_pytorch
import torch.nn as nn
from torch.testing._internal.common_utils import TestCase
import pytest

cpu_device = torch.device("cpu")
dpcpp_device = torch.device("xpu")

## Test Class
## TestTorchMethod for torch.xxx
## TestNNMethod for torch.nn.xxx
class TestNNMethod(TestCase):
    ##  use decorator to skip case
    @pytest.mark.skip(reason='Random Data Generate')
    ##  test function
    def test_random_norm(self, dtype=torch.float):
        x_cpu = torch.tensor([1.111, 2.222, 3.333, 4.444, 5.555,
                              6.666], device=cpu_device, dtype=dtype)
        x_dpcpp = torch.tensor(
            [1.111, 2.222, 3.333, 4.444, 5.555, 6.666], device=dpcpp_device, dtype=dtype)

        print("normal_ cpu", x_cpu.normal_(2.0, 0.5))
        print("normal_ dpcpp", x_dpcpp.normal_(2.0, 0.5).cpu())
        # assertion
        self.assertEqual(x_cpu.normal_(2.0, 0.5),
                         x_dpcpp.normal_(2.0, 0.5).cpu())
```

### skip with judgment

```python
import torch
import intel_extension_for_pytorch
from torch.testing._internal.common_utils import TestCase
import pytest

class TestTorchMethod(TestCase):
    @pytest.mark.skipIf(True, "bla bla")
    def test_skip_with_judgment(self):
        assert 1==1 , "skip with judgment"

```

### repeat for different dtype

```python
import torch
import intel_extension_for_pytorch
from torch.testing._internal.common_utils import TestCase, repeat_test_for_types

cpu_device = torch.device("cpu")
dpcpp_device = torch.device("xpu")


class TestTorchMethod(TestCase):
    @repeat_test_for_types([torch.float, torch.half, torch.bfloat16])
    def test_abs(self, dtype=torch.float):
        data = [[-0.2911, -1.3204,  -2.6425,  -2.4644,  -
                        0.6018, -0.0839, -0.1322, -0.4713, -0.3586, -0.8882, 0.0000, 0.0000, 1.1111, 2.2222, 3.3333]]
        excepted = [[0.2911, 1.3204,  2.6425,  2.4644,
                        0.6018, 0.0839, 0.1322, 0.4713, 0.3586, 0.8882, 0.0000, 0.0000, 1.1111, 2.2222, 3.3333]]
        x_dpcpp = torch.tensor(data, device=dpcpp_device)
        y = torch.tensor(excepted, device=dpcpp_device)
        y_dpcpp = torch.abs(x_dpcpp)
        self.assertEqual(y.to(cpu_device), y_dpcpp.to(cpu_device))
```

# How to RUN and DEBUG Test Cases of Pytorch Test Suite

## RUN the Whole Test Suite

* run the shell script under `tests/gpu/experimental` like:

```
bash tests/gpu/experimental/run_tests.sh
```

* the default log file is under `tests/gpu/experimental/logs`, but you can also save your own log file at where you want:

```
cd tests/gpu/experimental
bash run_tests.sh -L <path-to>/full_logfile.log
```

## RUN ONE Specific Test

* pass necessary flags `-S` and `-F <filename> -V <classname> -K <testcase>` to the shell script like:

```
cd tests/gpu/experimental
bash run_tests.sh -S -F test_unary_ufuncs.py -V TestUnaryUfuncsXPU -K test_abs_zero_xpu_float32
```

## RUN a Serial of a Specific OP or Class

* you can run a specific test class which contains all test cases within it, just leave the `-K` argument as empty

* you can run a serial specific test cases with different dtypes as also, just pass a prefix of the full test name like:
```
bash run_tests.sh -S -F test_unary_ufuncs.py -V TestUnaryUfuncsXPU -K test_abs_zero
```

## SKIP ONE Specific Test

* to add the test name with its class into `common/pytorch_test_base.py` like:

```
DISABLED_TORCH_TESTS_XPU_ONLY = {
    "TestUnaryUfuncsXPU": {
        'test_abs_zero', # reasons
    },
}
```

* some cases which missed an implementation of without XPU support were skipped by default. If you want to run it anyway, please contact the developer Xunsong, Huang <xunsong.huang@intel.com> for details.

## USE the UT Analyzer to get analysis data

* pass `--logfile <path-to>/<filename>` to the UT analyzer under `tests/gpu/experimental/common` like:

```
python ut_analyzer.py --logfile my_logfile.log
```

* this analyzer will output a summary of result like pass rate, fail rate, etc. And also it will dump out detailed log list for each classified field under `tests/gpu/pytorch/logs` as well.

## Known Issues

* the whole test will hang after finishing the TestReductionXPU class. A SIGTERM is necessary for continuing the remained tests.


