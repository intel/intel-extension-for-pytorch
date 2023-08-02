@echo off

set "VER_PYTORCH=v2.0.1"
set "VER_TORCHVISION=v0.15.2"
set "VER_IPEX=xpu-master"

if "%~2"=="" (
    echo Usage: %~nx0 ^<DPCPPROOT^> ^<MKLROOT^> [AOT]
    echo DPCPPROOT and MKLROOT are mandatory, should be absolute or relative path to the root directory of DPC++ compiler and oneMKL respectively.
    echo AOT is optional, should be the text string for environment variable USE_AOT_DEVLIST.
    exit /b 1
)
set "DPCPP_ROOT=%~1"
set "ONEMKL_ROOT=%~2"
set "AOT="
if not "%~3"=="" (
    set "AOT=%~3"
)

rem Check existance of DPCPP and ONEMKL environments
set "DPCPP_ENV=%DPCPP_ROOT%\env\vars.bat"
if NOT EXIST "%DPCPP_ENV%" (
    echo DPC++ compiler environment %DPCPP_ENV% doesn't seem to exist.
    exit /b 2
)
call %DPCPP_ENV%

set "ONEMKL_ENV=%ONEMKL_ROOT%\env\vars.bat"
if NOT EXIST "%ONEMKL_ENV%" (
    echo oneMKL environment %ONEMKL_ENV% doesn't seem to exist.
    exit /b 3
)
call %ONEMKL_ENV%

rem Check existence of required Windows commands
for %%A in (python git cl) do (
    where %%A >nul 2>&1 || (
        echo Error: Command "%%A" not found.
        exit /b 4
    )
)

rem Save current directory path
set "BASEFOLDER=%~dp0"
cd "%BASEFOLDER%"

rem Be verbose now
echo on

rem Install Python dependencies
python -m pip install cmake astunparse numpy ninja pyyaml mkl-static mkl-include setuptools cffi typing_extensions future six requests dataclasses Pillow

rem Checkout individual components
if NOT EXIST pytorch (
    git clone https://github.com/pytorch/pytorch.git
)

if NOT EXIST vision (
    git clone https://github.com/pytorch/vision.git
)

if NOT EXIST intel-extension-for-pytorch (
    git clone https://github.com/intel/intel-extension-for-pytorch.git
)

rem Checkout required branch/commit and update submodules
cd pytorch
if not "%VER_PYTORCH%"=="" (
    git checkout %VER_PYTORCH%
)
git submodule sync
git submodule update --init --recursive
cd ..

cd vision
if not "%VER_TORCHVISION%"=="" (
    git checkout %VER_TORCHVISION%
)
git submodule sync
git submodule update --init --recursive
cd ..

cd intel-extension-for-pytorch
if not "%VER_IPEX%"=="" (
    git checkout %VER_IPEX%
)
git submodule sync
git submodule update --init --recursive

rem Compile individual component

rem PyTorch
cd ..\pytorch
git stash
git clean -f
for %%f in ("..\intel-extension-for-pytorch\torch_patches\*.patch") do git apply "%%f"
python -m pip install -r requirements.txt
call conda install -q --yes -c conda-forge libuv=1.39
rem Ensure cmake can find python packages when using conda or virtualenv
if defined CONDA_PREFIX (
    set "CMAKE_PREFIX_PATH=%CONDA_PREFIX:~1,-1%"
) else if defined VIRTUAL_ENV (
     set "CMAKE_PREFIX_PATH=%VIRTUAL_ENV:~1,-1%"
)
set "USE_STATIC_MKL=1"
set "USE_NUMA=0"
set "USE_CUDA=0"
python setup.py clean
python setup.py bdist_wheel
set "USE_CUDA="
set "USE_NUMA="
set "USE_STATIC_MKL="
set "CMAKE_PREFIX_PATH="
python -m pip uninstall -y mkl-static mkl-include
for %%f in ("dist\*.whl") do python -m pip install --force-reinstall --no-deps "%%f"

rem TorchVision 
cd ..\vision
set "DISTUTILS_USE_SDK=1"
python setup.py clean
python setup.py bdist_wheel
set "DISTUTILS_USE_SDK="
for %%f in ("dist\*.whl") do python -m pip install --force-reinstall --no-deps "%%f"

rem Intel® Extension for PyTorch*
cd ..\intel-extension-for-pytorch
python -m pip install -r requirements.txt
if NOT "%AOT%"=="" (
    set "USE_AOT_DEVLIST=%AOT%"
)
set "BUILD_WITH_CPU=0"
set "USE_MULTI_CONTEXT=1"
set "DISTUTILS_USE_SDK=1"
set "CMAKE_CXX_COMPILER=icx"
python setup.py clean
python setup.py bdist_wheel
set "CMAKE_CXX_COMPILER="
set "DISTUTILS_USE_SDK="
set "USE_MULTI_CONTEXT="
set "BUILD_WITH_CPU="
for %%f in ("dist\*.whl") do python -m pip install --force-reinstall --no-deps "%%f"

rem Sanity Test
cd ..
python -c "import torch; import torchvision; import intel_extension_for_pytorch as ipex; print(f'torch_version:       {torch.__version__}'); print(f'torchvision_version: {torchvision.__version__}'); print(f'ipex_version:        {ipex.__version__}');"