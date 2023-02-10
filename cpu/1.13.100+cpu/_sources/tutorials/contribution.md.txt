Contribution
============

## Contributing to Intel® Extension for PyTorch\*

Thank you for your interest in contributing to Intel® Extension for PyTorch\*. Before you begin writing code, it is important that you share your intention to contribute with the team, based on the type of contribution:

1. You want to propose a new feature and implement it.
    - Post about your intended feature in a [GitHub issue](https://github.com/intel/intel-extension-for-pytorch/issues), and we shall discuss the design and implementation. Once we agree that the plan looks good, go ahead and implement it.
2. You want to implement a feature or bug-fix for an outstanding issue.
    - Search for your issue in the [GitHub issue list](https://github.com/intel/intel-extension-for-pytorch/issues).
    - Pick an issue and comment that you'd like to work on the feature or bug-fix.
    - If you need more context on a particular issue, ask and we shall provide.

Once you implement and test your feature or bug-fix, submit a Pull Request to https://github.com/intel/intel-extension-for-pytorch.

## Developing Intel® Extension for PyTorch\*

A full set of instructions on installing Intel® Extension for PyTorch\* from source is in the [Installation document](installation.md#install-via-source-compilation).

To develop on your machine, here are some tips:

1. Uninstall all existing Intel® Extension for PyTorch\* installs. You may need to run `pip uninstall intel_extension_for_pytorch` multiple times. You'll know `intel_extension_for_pytorch` is fully uninstalled when you see `WARNING: Skipping intel_extension_for_pytorch as it is not installed`. (You should only have to `pip uninstall` a few times, but you can always `uninstall` with `timeout` or in a loop if you're feeling lazy.)

   ```bash
   yes | pip uninstall intel_extension_for_pytorch
   ```

2. Clone a copy of Intel® Extension for PyTorch\* from source:

   ```bash
   git clone https://github.com/intel/intel-extension-for-pytorch.git
   cd intel-extension-for-pytorch
   ```

   If you already have Intel® Extension for PyTorch\* from source, update it:

   ```bash
   git pull --rebase
   git submodule sync --recursive
   git submodule update --init --recursive --jobs 0
   ```

3. Install Intel® Extension for PyTorch\* in `develop` mode:

   Replace:

   ```bash
   python setup.py install
   ```

   with:

   ```bash
   python setup.py develop
   ```

   This mode will symlink the Python files from the current local source tree into the Python install. After than, if you modify a Python file, you do not need to reinstall PyTorch again. This is especially useful if you are only changing Python files.

   For example:
   - Install local Intel® Extension for PyTorch\* in `develop` mode
   - modify your Python file `intel_extension_for_pytorch/__init__.py` (for example)
   - test functionality

You do not need to repeatedly install after modifying Python files (`.py`). However, you would need to reinstall if you modify a Python interface (`.pyi`, `.pyi.in`) or non-Python files (`.cpp`, `.cc`, `.cu`, `.h`, etc.).

If you want to reinstall, make sure that you uninstall Intel® Extension for PyTorch\* first by running `pip uninstall intel_extension_for_pytorch` until you see `WARNING: Skipping intel_extension_for_pytorch as it is not installed`; next run `python setup.py clean`. After that, you can install in `develop` mode again.

### Tips and Debugging

* Cmake must be installed before installing Intel® Extension for PyTorch\*. If youre developing on MacOS or Linux, We recommend installing Cmake with [Homebrew](https://brew.sh/) with `brew install cmake`.
* Our `setup.py` requires Python >= 3.6
* If you run into errors when running `python setup.py develop`, here are some debugging steps:
  1. Run `printf '#include <stdio.h>\nint main() { printf("Hello World");}'|clang -x c -; ./a.out` to make sure your CMake works and can compile this simple Hello World program without errors.
  2. Remove your `build` directory. The `setup.py` script compiles binaries into the `build` folder and caches many details along the way. This saves time the next time you build. If you're running into issues, you can always `rm -rf build` from the toplevel `pytorch` directory and start over.
  3. If you have made edits to the Intel® Extension for PyTorch\* repo, commit any change you'd like to keep and clean the repo with the following commands (note that clean _really_ removes all untracked files and changes.):
     ```bash
     git submodule deinit -f .
     git clean -xdf
     python setup.py clean
     git submodule update --init --recursive --jobs 0 # very important to sync the submodules
     python setup.py develop                          # then try running the command again
     ```
  4. The main step within `python setup.py develop` is running `make` from the `build` directory. If you want to experiment with some environment variables, you can pass them into the command:
     ```bash
     ENV_KEY1=ENV_VAL1[, ENV_KEY2=ENV_VAL2]* python setup.py develop
     ```

## Unit testing

### Python Unit Testing

All PyTorch test suites are located in the `test` folder and start with `test_`. Run individual test suites using the command `python test/cpu/FILENAME.py`, where `FILENAME` represents the file containing the test suite you wish to run.

For example, to run all the TorchScript JIT tests (located at `test/cpu/test_jit.py`), you would run:

```bash
python test/cpu/test_jit.py
```

You can narrow down what you're testing even further by specifying the name of an individual test with `TESTCLASSNAME.TESTNAME`. Here, `TESTNAME` is the name of the test you want to run, and `TESTCLASSNAME` is the name of the class in which it is defined.

Let's say you want to run `test_Sequential`, which is defined as part of the `TestJit` class in `test/cpu/test_jit.py`. Your command would be:

```bash
python test/test_jit.py TestJit.test_Sequential
```

The `expecttest` and `hypothesis` libraries must be installed to run the tests. `mypy` is an optional dependency, and `pytest` may help run tests more selectively. All these packages can be installed with `conda` or `pip`.

### Better local unit tests with `pytest`

We don't officially support `pytest`, but it works well with our `unittest` tests and offers a number of useful features for local developing. Install it via `pip install pytest`.

If you want to run only tests that contain a specific substring, you can use the `-k` flag:

```bash
pytest test/cpu/test_nn.py -k Loss -v
```

The above is an example of testing a change to all Loss functions: this command runs tests such as `TestNN.test_BCELoss` and `TestNN.test_MSELoss` and can be useful to save keystrokes.

### Local linting

You can run the same linting steps that are used in CI locally via `make`:

```bash
# Lint all files
make lint -j 6  # run lint (using 6 parallel jobs)

# Lint only the files you have changed
make quicklint -j 6
```

These jobs may require extra dependencies that aren't dependencies of Intel® Extension for PyTorch\* itself, so you can install them via this command, which you should only have to run once:

```bash
make setup_lint
```

To run a specific linting step, use one of these targets or see the Makefile for a complete list of options.

```bash
# Check for tabs, trailing newlines, etc.
make quick_checks

make flake8

make mypy

make cmakelint

make clang-tidy
```

To run a lint only on changes, add the `CHANGED_ONLY` option:

```bash
make <name of lint> CHANGED_ONLY=--changed-only
```

### C++ Unit Testing

Intel® Extension for PyTorch\* offers tests located in the `test/cpp` folder. These tests are written in C++ and use the Google Test testing framework. After compiling Intel® Extension for PyTorch\* from source, the test runner binaries will be written to the `build/bin` folder. The command to run one of these tests is `./build/bin/FILENAME --gtest_filter=TESTSUITE.TESTNAME`, where `TESTNAME` is the name of the test you'd like to run and `TESTSUITE` is the suite that test is defined in.

For example, if you wanted to run the test `MayContainAlias`, which is part of the test suite `ContainerAliasingTest` in the file `test/cpp/jit/test_alias_analysis.cpp`, the command would be:

```bash
./build/bin/test_jit --gtest_filter=ContainerAliasingTest.MayContainAlias
```

## Writing documentation

So you want to write some documentation for your code contribution and don't know where to start?

Intel® Extension for PyTorch\* uses [Google style](http://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html) for formatting docstrings. Length of line inside docstrings block must be limited to 80 characters to fit into Jupyter documentation popups.

### Building documentation

To build the documentation:

1. Build and install Intel® Extension for PyTorch\* (as discussed above)

2. Install the prerequisites:

   ```bash
   cd docs
   pip install -r requirements.txt
   ```

3. Generate the documentation HTML files. The generated files will be in `docs/_build/html`.

   ```bash
   make clean
   make html
   ```

#### Tips

The `.rst` source files live in [docs/tutorials](https://github.com/intel/intel-extension-for-pytorch/tree/master/docs/tutorials). Some of the `.rst` files pull in docstrings from Intel® Extension for PyTorch\* Python code (for example, via the `autofunction` or `autoclass` directives). To shorten doc build times, it is helpful to remove the files you are not working on, only keeping the base `index.rst` file and the files you are editing. The Sphinx build will produce missing file warnings but will still complete.
