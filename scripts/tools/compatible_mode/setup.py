from setuptools import setup

setup(
    name="compatible_mode",
    version="1.0",
    description="compatible mode plugin",
    packages=["compatible_mode", "compatible_mode/yaml", "compatible_mode/fake_module"],
    install_requires=["torch", "intel_extension_for_pytorch", "ruamel.yaml"],
    package_data={"": ["*"]},
    include_package_data=True,
)
