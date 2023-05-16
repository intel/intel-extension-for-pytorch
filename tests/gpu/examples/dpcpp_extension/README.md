# How to Extend Your DPCPP Extension UT

Here is the conventions you MUST follow:
1. Add your package installation file, source cpp file and test python file here.
2. The file name of each package installation file should start with "setup_".
3. The file name of each source file and test python file should start with "test_".

# Notice
Each file name start with "setup_" will be installed automatically via `python setup.py install` in CI test first. Then validate the UT case via `pytest test_{name}.py`.
