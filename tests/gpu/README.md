# How to Write and RUN Test Case

## Notice:

*  These tests are powered by [Pytest](https://docspytest.org/en/stable/).

*  Please refer to [Pytest Documents](https://docspytest.org/en/stable/) for more helps.

## Pre-requirements:

*  PyTorch and Intel GPU Extensio for PyTorch have been installed and verified.

*  Install Pytest
```bash
python3 -m pip install pytest
```

## Find and Run tests

*  All tests are found under the ${PATH_To_Your_Extension_Source_Code}/tests/gpu/ path.
*  Or download from the repo with below command.
```bash
git clone --depth=1 ssh://git@gitlab.devtools.intel.com:29418/intel-pytorch-extension/intel-pytorch-extension.git
```

*  Please use pytest to run one test, or all of them if ${Test_Name} is empty.
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
import torch_ipex
## import testcase parent class TestCaes
from torch.testing._internal.common_utils import TestCase

cpu_device = torch.device("cpu")
dpcpp_device = torch.device("dpcpp")

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
        ## asssert
        self.assertEqual(y, y_dpcpp.to(cpu_device))

    def test_abs2(self, dtype=torch.float):
        data = [[-0.2911, -1.3204,  -2.6425,  -2.4644,  -
                 0.6018, -0.0839, -0.1322, -0.4713, -0.3586, -0.8882]]
        x = torch.tensor(data, device=cpu_device)
        x_dpcpp = x.to(dpcpp_device)
        y = torch.abs(x)
        y_dpcpp = torch.abs(x_dpcpp)
        ## asssert
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
import torch_ipex
import torch.nn as nn
from torch.testing._internal.common_utils import TestCase
import pytest

cpu_device = torch.device("cpu")
dpcpp_device = torch.device("dpcpp")

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
        # assert
        self.assertEqual(x_cpu.normal_(2.0, 0.5),
                         x_dpcpp.normal_(2.0, 0.5).cpu())
```

### skip with judgment

```python 
import torch
import torch_ipex
from torch.testing._internal.common_utils import TestCase
import pytest

class TestTorchMethod(TestCase):
    @pytest.mark.skipIf(True, "bla bla")
    def test_skip_with_judgment(self):
        assert 1==1 , "skip with judgment"

```

### repeat for different  dtype

```python
import torch
import torch_ipex 
from torch.testing._internal.common_utils import TestCase, repeat_test_for_types

cpu_device = torch.device("cpu")
dpcpp_device = torch.device("dpcpp")


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

