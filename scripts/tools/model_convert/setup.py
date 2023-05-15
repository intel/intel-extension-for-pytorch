from setuptools import setup, find_packages

setup(
    name="model_script_convert",
    version="1.0",
    description="model convert plugin",
    packages=["model_convert", "model_convert/yaml", "model_convert/ast_convert_tool"],
    install_requires=["torch", "intel_extension_for_pytorch", "ruamel.yaml"],
    package_data={"": ["*"]},
    include_package_data=True,
)
