# How to Write and RUN Test Case In Example

## Notice

* These tests are powered by [Pytest](https://docs.pytest.org/en/7.3.x/).

* Please refer to [Pytest Documents](https://docs.pytest.org/en/7.3.x/) for more helps.

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
git clone https://github.com/intel/intel-extension-for-pytorch.git
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

### General case study

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

### Skip certain case with reason and comments

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


