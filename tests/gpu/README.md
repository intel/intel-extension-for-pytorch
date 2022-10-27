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
* Please use pytest to run one test, or all of them if ${Test_Name} is empty.

```bash
python3 -m pytest ${PATH_To_Your_Extension_Source_Code}/tests/gpu/${Test_Name}
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



## How to RUN and DEBUG Test Cases of Pytorch Test Suite

### RUN the Whole Test Suite

* run the python script under `tests/gpu/experimental` like:

```bash
python tests/gpu/experimental/run_tests.py [--options opts]
```

* the default log file is under `tests/gpu/experimental/logs/raw_logs`.

### Get the usage of options of runner

* run the python script with `--help` like:

```bash
python tests/gpu/experimental/run_tests.py --help
```

* you may see decriptions of options like:

```bash
usage: run_tests.py [-h] [--logdir logdir] [--spec spectest] [-c count] [-t timeout]
                    [--parallel] [--autoskip] [-q] [--ignore] [--clean]

Main script to run all or specific tests

optional arguments:
  -h, --help            show this help message and exit
  --logdir logdir       the path of logfile to store, it should be a directory
  --spec spectest       the full name of a specific test case. It should be in format:
                        'filename::classname::casename'. In some cases, the case name is
                        optional like 'filename::classname' and all cases in the test class will
                        be triggered.
  -c count, --count count
                        loop times of each test class. Each round of run will be logged into
                        different log files.
  -t timeout, --timeout timeout
                        time limit for each test class in seconds. A zero stand for non-
                        limitation.
  --parallel            run whole test in single process if set
  --autoskip            auto skip core dumped cases and hang cases an re-run corresponding test
                        class if set
  -q, --quiet           don't print out detailed results to screen if set
  --ignore              ignore common failures and errors, and continue to run next test class
                        if set
  --clean               clean raw logs
```

### Suggested command for running

* you can run the whole tests with time threshold, multi-epoches and auto skip core dumped or hang cases like:

```bash
python tests/gpu/experimental/run_tests.py --clean
python tests/gpu/experimental/run_tests.py -c 3 -t 3600 --autoskip --quiet
```

## SKIP ONE Specific Test

* to add the test name with its class into `common/skip_list.json` with explict reasons.

* strongly recommend skip core dumped issues and hang issues by auto skip mechanism

## USE the UT Analyzer to get analysis data

* run python script `ut_analyzer.py` under `tests/gpu/experimental/common` like:

```bash
python tests/gpu/experimental/common/ut_analyzer.py [--options opts]
```

* this analyzer will output a summary of result like pass rate, fail rate, etc. And also it will dump out detailed log list for each classified field under `tests/gpu/experimental/logs/anls_logs` as well.

### Get the usage of options of analyzer

* run the python script with `--help` like:

```bash
python tests/gpu/experimental/common/ut_analyzer.py --help
```

* you may see decriptions of options like:

```bash
usage: ut_analyzer.py [-h] [--logdir logdir] [--saveref] [--compare] [--clean]

Auto script for analysing raw logs

optional arguments:
  -h, --help       show this help message and exit
  --logdir logdir  the path of logfiles stored, it should be a directory and must have
                   'raw_logs' under this path
  --saveref        save reference pass list if no break tests
  --compare        compare current pass list against reference to see if regression occurred
  --clean          clean analysis logs
```
