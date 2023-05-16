from setuptools import setup, find_packages

setup(
    name='memory_check',
    version='1.0',
    description='memory check plugin',
    packages=['memory_check'],
    install_requires=['torch', 'intel_extension_for_pytorch'],
    package_data={'': ['*']},
    include_package_data=True,
    )

