@echo off
setlocal

set "VER_IPEX=xpu-main"

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
if [%DPCPP_ROOT:~-1%]==[\] set "DPCPP_ROOT=%DPCPP_ROOT:~0,-1%"
if [%ONEMKL_ROOT:~-1%]==[\] set "ONEMKL_ROOT=%ONEMKL_ROOT:~0,-1%"

rem Check existance of DPCPP and ONEMKL environments
set "DPCPP_ENV=%DPCPP_ROOT%\env\vars.bat"
if not exist "%DPCPP_ENV%" (
    echo DPC++ compiler environment "%DPCPP_ENV%" doesn't seem to exist.
    exit /b 2
)

set "ONEMKL_ENV=%ONEMKL_ROOT%\env\vars.bat"
if not exist "%ONEMKL_ENV%" (
    echo oneMKL environment "%ONEMKL_ENV%" doesn't seem to exist.
    exit /b 3
)

for %%a in ("%DPCPP_ROOT%") do set "DPCPP_VER=%%~nxa"
set "OCLOC_ENV=%DPCPP_ROOT%\..\..\ocloc\%DPCPP_VER%\env\vars.bat"
if not exist "%OCLOC_ENV%" (
    set "OCLOC_ENV="
)

if "%VCINSTALLDIR%"=="" (
    echo Please activate Visual Studio environment with Visual Studio script "vcvars64.bat" or "vcvarsall.bat x64".
    echo More details can be found at https://learn.microsoft.com/en-us/cpp/build/building-on-the-command-line#developer_command_file_locations
    exit /b 4
)
set index=1
:loop
for /f "tokens=%index% delims=\" %%a in ("%VCINSTALLDIR:~0,-4%") do (
    set /a index=index+1
    if not "%%a"=="Microsoft Visual Studio" goto loop
)
for /f "tokens=%index% delims=\" %%a in ("%VCINSTALLDIR:~0,-4%") do set "VSVER=%%a"
set "VS%VSVER%INSTALLDIR=%VSINSTALLDIR%"

rem Save current directory path
set "BASEFOLDER=%~dp0"
cd "%BASEFOLDER%"


rem Checkout the latest Intel(R) Extension for PyTorch source
cd intel-extension-for-pytorch
if not "%VER_IPEX%"=="" (
    git rm -rf .
    git clean -fxd
    git reset
    git checkout .
    git checkout main
    git pull
    git checkout %VER_IPEX%
)
git submodule sync
git submodule update --init --recursive

python -m pip install pyyaml
for /f %%A in ('python scripts\tools\compilation_helper\yaml_utils.py -f dependency_version.yml -d pytorch -k commit') do set COMMIT_TORCH=%%A
for /f %%A in ('python scripts\tools\compilation_helper\yaml_utils.py -f dependency_version.yml -d torchvision -k commit') do set COMMIT_TORCHVISION=%%A
for /f %%A in ('python scripts\tools\compilation_helper\yaml_utils.py -f dependency_version.yml -d torchaudio -k commit') do set COMMIT_TORCHAUDIO=%%A
python -m pip uninstall -y pyyaml
cd ..

rem Checkout individual components
if not exist pytorch (
    git clone https://github.com/pytorch/pytorch.git
)
if not exist vision (
    git clone https://github.com/pytorch/vision.git
)
if not exist audio (
    git clone https://github.com/pytorch/audio.git
)
if not exist intel-extension-for-pytorch (
    git clone https://github.com/intel/intel-extension-for-pytorch.git
)

rem Checkout required branch/commit and update submodules
cd pytorch
if not "%COMMIT_TORCH%"=="" (
    git rm -rf .
    git clean -fxd
    git reset
    git checkout .
    git checkout main
    git pull
    git checkout %COMMIT_TORCH%
)
git submodule sync
git submodule update --init --recursive
for %%f in ("..\intel-extension-for-pytorch\torch_patches\*.patch") do git apply "%%f"
cd ..

cd vision
if not "%COMMIT_TORCHVISION%"=="" (
    git rm -rf .
    git clean -fxd
    git reset
    git checkout .
    git checkout main
    git pull
    git checkout %COMMIT_TORCHVISION%
)
git submodule sync
git submodule update --init --recursive
cd ..

cd audio
if not "%COMMIT_TORCHAUDIO%"=="" (
    git rm -rf .
    git clean -fxd
    git reset
    git checkout .
    git checkout main
    git pull
    git checkout %COMMIT_TORCHAUDIO%
)
git submodule sync
git submodule update --init --recursive
cd ..

rem Compile individual component
call "%DPCPP_ENV%"
call "%ONEMKL_ENV%"
if not "%OCLOC_ENV%"=="" (
    call "%OCLOC_ENV%"
)

rem Install Python dependencies
python -m pip uninstall -y torch torchvision torchaudio intel-extension-for-pytorch
python -m pip install cmake==3.26.4 make ninja==1.10.2

rem PyTorch
cd pytorch
python -m pip install -r requirements.txt
python -m pip install --force-reinstall mkl-static mkl-include
call conda install -y conda-forge::libuv
rem Ensure cmake can find python packages when using conda or virtualenv
if defined CONDA_PREFIX (
    set "CMAKE_PREFIX_PATH=%CONDA_PREFIX%"
) else if defined VIRTUAL_ENV (
     set "CMAKE_PREFIX_PATH=%VIRTUAL_ENV%"
)
set "CMAKE_INCLUDE_PATH=%CONDA_PREFIX%\Library\include"
set "LIB_BK=%LIB%"
set "LIB=%CONDA_PREFIX%\Library\lib;%LIB%"
set "USE_NUMA=0"
set "USE_CUDA=0"
python setup.py clean
python setup.py bdist_wheel
set "USE_CUDA="
set "USE_NUMA="
set "LIB=%LIB_BK%"
set "CMAKE_INCLUDE_PATH="
set "CMAKE_PREFIX_PATH="
python -m pip uninstall -y mkl-static mkl-include
for %%f in ("dist\*.whl") do python -m pip install "%%f"
cd ..

rem TorchVision 
cd vision
call conda install -y --force-reinstall conda-forge::libpng conda-forge::libjpeg-turbo
set "DISTUTILS_USE_SDK=1"
python setup.py clean
python setup.py bdist_wheel
set "DISTUTILS_USE_SDK="
for %%f in ("dist\*.whl") do python -m pip install "%%f"
cd ..

rem TorchAudio 
cd audio
python -m pip install -r requirements.txt
set "DISTUTILS_USE_SDK=1"
python setup.py clean
python setup.py bdist_wheel
set "DISTUTILS_USE_SDK="
for %%f in ("dist\*.whl") do python -m pip install "%%f"
cd ..

rem Intel® Extension for PyTorch*
cd intel-extension-for-pytorch
python -m pip install -r requirements.txt
if NOT "%AOT%"=="" (
    set "USE_AOT_DEVLIST=%AOT%"
)
set "BUILD_WITH_CPU=0"
set "USE_MULTI_CONTEXT=1"
set "DISTUTILS_USE_SDK=1"
set "ENABLE_ONEAPI_INTEGRATION=1"
python setup.py clean
python setup.py bdist_wheel
set "ENABLE_ONEAPI_INTEGRATION="
set "DISTUTILS_USE_SDK="
set "USE_MULTI_CONTEXT="
set "BUILD_WITH_CPU="
for %%f in ("dist\*.whl") do python -m pip install "%%f"
cd ..

rem Sanity Test
python -c "import torch; import torchvision; import torchaudio; import intel_extension_for_pytorch as ipex; print(f'torch_version:       {torch.__version__}'); print(f'torchvision_version: {torchvision.__version__}'); print(f'torchaudio_version:  {torchaudio.__version__}'); print(f'ipex_version:        {ipex.__version__}');"
endlocal
