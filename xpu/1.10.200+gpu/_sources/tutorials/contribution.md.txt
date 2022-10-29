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

## Developing Intel® Extension for PyTorch\* on XPU

A full set of instructions on installing Intel® Extension for PyTorch\* from source is in the [Installation document](installation.md#install-via-source-compilation).

To develop on your machine, here are some tips:

1. Uninstall all existing Intel® Extension for PyTorch\* installs. You may need to run `pip uninstall intel_extension_for_pytorch` multiple times. You'll know `intel_extension_for_pytorch` is fully uninstalled when you see `WARNING: Skipping intel_extension_for_pytorch as it is not installed`. (You should only have to `pip uninstall` a few times, but you can always `uninstall` with `timeout` or in a loop.)

   ```bash
   yes | pip uninstall intel_extension_for_pytorch
   ```

2. Clone a copy of Intel® Extension for PyTorch\* from source:

   ```bash
   git clone https://github.com/intel/intel-extension-for-pytorch.git -b xpu-master
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

   This mode will symlink the Python files from the current local source tree into the Python install. After that, if you modify a Python file, you do not need to reinstall Intel® Extension for PyTorch\* again. This is especially useful if you are only changing Python files.

   For example:
   - Install local Intel® Extension for PyTorch\* in `develop` mode
   - modify your Python file `intel_extension_for_pytorch/__init__.py` (for example)
   - test functionality

You do not need to repeatedly install after modifying Python files (`.py`). However, you would need to reinstall if you modify a Python interface (`.pyi`, `.pyi.in`) or non-Python files (`.cpp`, `.h`, etc.).

If you want to reinstall, make sure that you uninstall Intel® Extension for PyTorch\* first by running `pip uninstall intel_extension_for_pytorch` until you see `WARNING: Skipping intel_extension_for_pytorch as it is not installed`. Then run `python setup.py clean`. After that, you can install in `develop` mode again.

### Tips and Debugging

* Our `setup.py` requires Python >= 3.6
* If you run into errors when running `python setup.py develop`, here are some debugging steps:
  1. Remove your `build` directory. The `setup.py` script compiles binaries into the `build` folder and caches many details along the way. This saves time the next time you build. If you're running into issues, you can always `rm -rf build` from the toplevel directory and start over.
  2. If you have made edits to the Intel® Extension for PyTorch\* repo, commit any change you'd like to keep and clean the repo with the following commands (note that clean _really_ removes all untracked files and changes.):
     ```bash
     git submodule deinit -f .
     git clean -xdf
     python setup.py clean
     git submodule update --init --recursive --jobs 0 # very important to sync the submodules
     python setup.py develop                          # then try running the command again
     ```
  3. The main step within `python setup.py develop` is running `make` from the `build` directory. If you want to experiment with some environment variables, you can pass them into the command:
     ```bash
     ENV_KEY1=ENV_VAL1[, ENV_KEY2=ENV_VAL2]* python setup.py develop
     ```

## Unit testing

All Python test suites are located in the `tests/gpu` folder and start with `test_`. Run individual test suites using the command `python tests/gpu/${Sub_Folder}/FILENAME.py`, where `FILENAME` represents the file containing the test suite you wish to run and `${Sub_Folder}` is one of the following folders:
- examples: unit tests created during op development
- experimental: ported [test suites](https://github.com/pytorch/pytorch/tree/v1.10.0/test) from Stock PyTorch 1.10
- regression: unit tests created during bug fix to avoid future regression

### Better local unit tests with `pytest`

We don't officially support `pytest`, but it works well with our unit tests and offers a number of useful features for local developing. Install it via `pip install pytest`.

For more information about unit tests, please read [README.md](../../tests/gpu/README.md) in the `tests/gpu` folder. 

## Writing documentation

Do you want to write some documentation for your code contribution and don't know where to start?

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

The `.rst` source files live in `docs/tutorials` folder. Some of the `.rst` files pull in docstrings from Intel® Extension for PyTorch\* Python code (for example, via the `autofunction` or `autoclass` directives). To shorten doc build times, it is helpful to remove the files you are not working on, only keeping the base `index.rst` file and the files you are editing. The Sphinx build will produce missing file warnings but will still complete.
