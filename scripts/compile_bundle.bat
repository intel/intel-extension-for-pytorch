@echo off
SETLOCAL

set startup_time=%time%
echo startup at:  %startup_time%

set "VER_IPEX=xpu-main"
set "ENABLE_ONEAPI_INTEGRATION=1"

set CMAKE_SHARED_LINKER_FLAGS=/FORCE:MULTIPLE
set CMAKE_MODULE_LINKER_FLAGS=/FORCE:MULTIPLE
set CMAKE_EXE_LINKER_FLAGS=/FORCE:MULTIPLE

set OUTPUT_FOLDER=dist


set argC=0
for %%x in (%*) do Set /A argC+=1

echo "Get %argC% parameters: %*"

if "%~1"=="help" (
    call:help
    exit /b 1
)

if %argC% LSS 2 (
    echo Missed parameters!
    call:help
    exit /b 1
)

rem Check required packages version
if not exist intel-extension-for-pytorch (
    git clone https://github.com/intel/intel-extension-for-pytorch.git intel-extension-for-pytorch
    cd intel-extension-for-pytorch
    if not "%VER_IPEX%"=="" (
        git rm -rf .
        git clean -fxd
        git reset
        git checkout .
        git fetch
        git checkout %VER_IPEX%
    )
    cd ..
)

set "DPCPP_ROOT=NA"
set "ONEMKL_ROOT=NA"

if %argC% EQU 2 (
    rem expect: <ONEAPIROOT> <AOT>
    call:create_compenents_root_by_oneapi_root "%~1" DPCPP_ROOT ONEMKL_ROOT
    set "AOT=%~2"
    set BUILD_IPEX_ONLY=0
)

if %argC% EQU 3 (
    rem expect: <ONEAPIROOT> <AOT> [Target]
    rem expect: <DPCPPROOT> <MKLROOT> <AOT>
    set "ONEAPI_ENV=%~1\setvars.bat"
    if exist "%~1\setvars.bat" (
        call:create_compenents_root_by_oneapi_root "%~1" DPCPP_ROOT ONEMKL_ROOT
        set "AOT=%~2"
        call:get_build_target "%~3" BUILD_IPEX_ONLY
    ) else (
        set "DPCPP_ROOT=%~1"
        set "ONEMKL_ROOT=%~2"
        set "AOT=%~3"
        set BUILD_IPEX_ONLY=0
    )
)

if %argC% EQU 4 (
    rem expect: <DPCPPROOT> <MKLROOT> <AOT> [Target]
    set "DPCPP_ROOT=%~1"
    set "ONEMKL_ROOT=%~2"
    set "AOT=%~3"
    call:get_build_target "%~4" BUILD_IPEX_ONLY
)


echo dpcpp=%DPCPP_ROOT%
echo mkl=%ONEMKL_ROOT%
echo aot=%AOT%
echo build_ipex_only=%BUILD_IPEX_ONLY%

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

rem if %AOT%==none (
rem     set AOT=""
rem )

echo AOT flags:[%AOT%]


if %BUILD_IPEX_ONLY% equ 1 (
    echo Building IPEX only
    set IPEX_ROOT=intel-extension-for-pytorch
    for /f %%A in ('python %IPEX_ROOT%\scripts\tools\compilation_helper\dep_ver_utils.py -f %IPEX_ROOT%\dependency_version.json -k pytorch:version') do set PYTORCH_VER=%%A

    for /f %%A in ('python %IPEX_ROOT%\scripts\tools\compilation_helper\dep_ver_utils.py -f %IPEX_ROOT%\dependency_version.json -k torchaudio:version') do set AUDIO_VER=%%A

    for /f %%A in ('python %IPEX_ROOT%\scripts\tools\compilation_helper\dep_ver_utils.py -f %IPEX_ROOT%\dependency_version.json -k torchvision:version') do set VISION_VER=%%A

) else (
    echo Building torch, torchvision, torchaudio, IPEX
)

if %BUILD_IPEX_ONLY% equ 1 (
    echo dependent on pytorch==%PYTORCH_VER% torchaudio==%AUDIO_VER% torchvision==%VISION_VER%
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
echo Update IPEX submodule
cd intel-extension-for-pytorch
git submodule sync
git submodule update --init --recursive

for /f %%A in ('python scripts\tools\compilation_helper\dep_ver_utils.py -f dependency_version.json -k pytorch:commit') do set COMMIT_TORCH=%%A
for /f %%A in ('python scripts\tools\compilation_helper\dep_ver_utils.py -f dependency_version.json -k torchvision:commit') do set COMMIT_TORCHVISION=%%A
for /f %%A in ('python scripts\tools\compilation_helper\dep_ver_utils.py -f dependency_version.json -k torchaudio:commit') do set COMMIT_TORCHAUDIO=%%A
cd ..

rem Checkout individual components
if %BUILD_IPEX_ONLY% NEQ 1 (
    if not exist pytorch (
        git clone https://github.com/pytorch/pytorch.git
    )

    echo Update torch code and submodules
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
    rem Checkout required branch/commit and update submodules
    for %%f in ("..\intel-extension-for-pytorch\torch_patches\*.patch") do git apply "%%f"
    cd ..

    if not exist vision (
        git clone https://github.com/pytorch/vision.git
    )
    echo Update torchvision code and submodules
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

    if not exist audio (
        git clone https://github.com/pytorch/audio.git
    )
    echo Update torchaudio code and submodules
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
)


rem Compile individual component
call "%DPCPP_ENV%"
call "%ONEMKL_ENV%"
if not "%OCLOC_ENV%"=="" (
    call "%OCLOC_ENV%"
)

echo "BASE_LIB=%LIB%"
echo compiler_root=%CMPLR_ROOT%
echo mkl_root=%MKLROOT%

rem Install Python dependencies
python -m pip install cmake==3.26.4 make ninja==1.10.2
call conda install -y conda-forge::libuv

if not exist %OUTPUT_FOLDER% mkdir %OUTPUT_FOLDER%
echo "Clear folder %OUTPUT_FOLDER%"
del /F/Q "%OUTPUT_FOLDER%\*.*"

if %BUILD_IPEX_ONLY% EQU 1 (
    echo Install packages: torch==%PYTORCH_VER% torchvision==%VISION_VER% torchaudio==%AUDIO_VER%
    python -m pip install torch==%PYTORCH_VER% torchvision==%VISION_VER% torchaudio==%AUDIO_VER% --index-url https://download.pytorch.org/whl/xpu
) else (
    python -m pip uninstall -y torch torchvision torchaudio intel-extension-for-pytorch

    rem remove the packages installed by last buliding & Sanity test. Avoid to impact next building.
    python -m pip uninstall -y mkl mkl-dpcpp mkl-include mkl-static onemkl-sycl-blas onemkl-sycl-datafitting onemkl-sycl-dft onemkl-sycl-lapack onemkl-sycl-rng onemkl-sycl-sparse  onemkl-sycl-stats onemkl-sycl-vm tbb tbb-devel intel-cmplr-lib-rt intel-cmplr-lic-rt intel-opencl-rt intel-sycl-rt intel-openmp

    rem PyTorch
    echo Building torch

    cd pytorch
    python -m pip install -r requirements.txt
    python -m pip install --force-reinstall mkl-static mkl-include

    set "PYTORCH_EXTRA_INSTALL_REQUIREMENTS=intel-cmplr-lib-rt==2025.0.2|intel-cmplr-lib-ur==2025.0.2| intel-cmplr-lic-rt==2025.0.2|intel-sycl-rt==2025.0.2|tcmlib==1.2.0|umf==0.9.1|intel-pti==0.10.0"

    rem Ensure cmake can find python packages when using conda or virtualenvs
    set "CMAKE_PREFIX_PATH_BK=%CMAKE_PREFIX_PATH%"
    if defined CONDA_PREFIX (
        set "CMAKE_PREFIX_PATH=%CONDA_PREFIX%"
    ) else if defined VIRTUAL_ENV (
        set "CMAKE_PREFIX_PATH=%VIRTUAL_ENV%"
    )
    set "CMAKE_INCLUDE_PATH_BK=%CMAKE_INCLUDE_PATH%"
    set "CMAKE_INCLUDE_PATH=%CONDA_PREFIX%\Library\include"
    set "LIB_BK=%LIB%"

    set "LIB=%CONDA_PREFIX%\Library\lib;%LIB%"


    set "USE_NUMA=0"
    set "USE_CUDA=0"
    python setup.py clean
    python setup.py bdist_wheel
    if %ERRORLEVEL% NEQ 0 (
        echo "Error to build torch: %ERRORLEVEL%"
        exit /b 1
    )

    if exist dist  (
        echo "Successfully build torch"
    ) else (
        echo "Fault to build torch"
        exit /b 1
    )

    set "USE_CUDA="
    set "USE_NUMA="

    rem python -m pip uninstall -y mkl-static mkl-include intel-cmplr-lib-ur intel-openmp tbb tbb-devel
    python -m pip uninstall -y mkl-static mkl-include intel-openmp tbb tbb-devel

    for %%f in ("dist\*.whl") do (
        python -m pip install "%%f"
        copy "%%f" ..\%OUTPUT_FOLDER%\
        echo copy "%%f" to %OUTPUT_FOLDER%
    )

    cd ..

    rem TorchVision
    echo Building torchvision
    cd vision

    call conda install -y --force-reinstall conda-forge::libpng conda-forge::libjpeg-turbo
    set "DISTUTILS_USE_SDK=1"
    echo "lib_vision=%LIB%"
    python setup.py clean
    python setup.py bdist_wheel
    if %ERRORLEVEL% NEQ 0 (
        echo "Error to build torchvision: %ERRORLEVEL%"
        exit /b 2
    )
    set "DISTUTILS_USE_SDK="
    set FOUND_WHL=0

    if exist dist  (
        echo "Successfully build torchvision"
    ) else (
        echo "Fault to build torchvision"
        exit /b 2
    )

    for %%f in ("dist\*.whl") do (
        python -m pip install "%%f"
        copy "%%f" ..\%OUTPUT_FOLDER%\
        echo copy "%%f" to %OUTPUT_FOLDER%
    )

    cd ..

    rem TorchAudio
    echo Building torchaudio
    cd audio

    python -m pip install -r requirements.txt
    set "DISTUTILS_USE_SDK=1"
    python setup.py clean
    python setup.py bdist_wheel
    if %ERRORLEVEL% NEQ 0 (
        echo "Error to build torchaudio: %ERRORLEVEL%"
        exit /b 3
    )

    if exist dist  (
        echo "Successfully build torchaudio"
    ) else (
        echo "Fault to build torchaudio"
        exit /b 3
    )

    set "DISTUTILS_USE_SDK="

    for %%f in ("dist\*.whl") do (
        python -m pip install "%%f"
        copy "%%f" ..\%OUTPUT_FOLDER%\
        echo copy "%%f" to %OUTPUT_FOLDER%
    )

    cd ..
)
rem Intel Extension for PyTorch*
echo "Uninstall Packages for IPEX build"
python -m pip uninstall -y intel-extension-for-pytorch
cd intel-extension-for-pytorch
echo "Install Packages for IPEX build"
python -m pip install -r requirements.txt

if NOT "%AOT%"=="" (
    set "USE_AOT_DEVLIST=%AOT%"
)
set "BUILD_WITH_CPU=0"
set "USE_MULTI_CONTEXT=1"
set "DISTUTILS_USE_SDK=1"
python setup.py clean
python setup.py bdist_wheel

if %ERRORLEVEL% NEQ 0 (
    echo "Error to build IPEX: %ERRORLEVEL%"
    exit /b 4
)

set "ENABLE_ONEAPI_INTEGRATION="
set "DISTUTILS_USE_SDK="
set "USE_MULTI_CONTEXT="
set "BUILD_WITH_CPU="

if exist dist  (
        echo "Successfully build IPEX"
) else (
        echo "Fault to build IPEX"
        exit /b 4
)

for %%f in ("dist\*.whl") do (
    python -m pip install "%%f"
    copy "%%f" ..\%OUTPUT_FOLDER%\
    echo copy "%%f" to %OUTPUT_FOLDER%
)

cd ..

call:calcu_time %startup_time%

rem show built whl files
echo All created WHL files are saved to folder: %OUTPUT_FOLDER%
dir %OUTPUT_FOLDER%

pip list | findstr torch

set PYTHON_FILE=test_build.bat
echo Create Test Script: %PYTHON_FILE%

call:create_test_py_file %PYTHON_FILE%
echo "Sanity Test"
python %PYTHON_FILE%
echo Remove %PYTHON_FILE%
del %PYTHON_FILE%

endlocal
exit /b 0

:create_test_py_file
SETLOCAL ENABLEDELAYEDEXPANSION
    set "PYTHON_FILE=%~1"
    echo try: >%PYTHON_FILE%
    echo ^    ^import torch >>%PYTHON_FILE%
    echo ^    ^print(f^'torch version:       {torch.__version__}^') >>%PYTHON_FILE%
    echo except: >>%PYTHON_FILE%
    echo ^    ^print(f^'torch version:       Can\^'t import torch^') >>%PYTHON_FILE%
    echo try: >>%PYTHON_FILE%
    echo ^    ^import torchvision >>%PYTHON_FILE%
    echo ^    ^print(f^'torchvision version: {torchvision.__version__}^') >>%PYTHON_FILE%
    echo except: >>%PYTHON_FILE%
    echo ^    ^print(f^'torchvision version: Can\^'t import torchvision^') >>%PYTHON_FILE%
    echo try: >>%PYTHON_FILE%
    echo ^    ^import torchaudio >>%PYTHON_FILE%
    echo ^    ^print(f^'torchaudio version:  {torchaudio.__version__}^') >>%PYTHON_FILE%
    echo except: >>%PYTHON_FILE%
    echo ^    ^print(f^'torchaudio version:  Can\^'t import torchaudio^') >>%PYTHON_FILE%
    echo try: >>%PYTHON_FILE%
    echo ^    ^import intel_extension_for_pytorch as ipex >>%PYTHON_FILE%
    echo ^    ^print(f'ipex version:        {ipex.__version__}') >>%PYTHON_FILE%
    echo ^    ^print(f'ipex_aot:            {ipex.__build_aot__}') >>%PYTHON_FILE%
    echo except: >>%PYTHON_FILE%
    echo ^    ^print(f^'ipex version:        Can\^'t import intel_extension_for_pytorch^') >>%PYTHON_FILE%
ENDLOCAL
goto:eof

:calcu_time
SETLOCAL ENABLEDELAYEDEXPANSION
    set "startup_time=%~1"
    rem calculate the running time
    set end_time=%time%
    echo stop at %end_time%
    set options="tokens=1-4 delims=:.,"
    for /f %options% %%a in ("%startup_time%") do set start_h=%%a&set /a start_m=100%%b %% 100&set /a start_s=100%%c %% 100&set /a start_ms=100%%d %% 100
    for /f %options% %%a in ("%end_time%") do set end_h=%%a&set /a end_m=100%%b %% 100&set /a end_s=100%%c %% 100&set /a end_ms=100%%d %% 100
    ::
    set /a hours=%end_h%-%start_h%
    set /a mins=%end_m%-%start_m%
    set /a secs=%end_s%-%start_s%
    set /a ms=%end_ms%-%start_ms%
    if %ms% lss 0 set /a secs = %secs% - 1 & set /a ms = 100%ms%
    if %secs% lss 0 set /a mins = %mins% - 1 & set /a secs = 60%secs%
    if %mins% lss 0 set /a hours = %hours% - 1 & set /a mins = 60%mins%
    if %hours% lss 0 set /a hours = 24%hours%
    if 1%ms% lss 100 set ms=0%ms%

    set /a totalsecs = %hours%*3600 + %mins%*60 + %secs%

    echo Build from %startup_time% to %end_time%
    echo Build took %hours%:%mins%:%secs%.%ms% (%totalsecs%.%ms%s total)
ENDLOCAL
goto:eof

:help
    SETLOCAL ENABLEDELAYEDEXPANSION
        echo.
        echo Help:
        echo This script is used to build Intel Extension for PyTorch package with AOT or not. In same time, support to build torch, torchvision, torchaudio.
        echo.
        echo Usage:
        echo.
        echo compile_bundle.bat ^<ONEAPI_ROOT^> ^<AOT^> [Target]
        echo.
        echo ^  ^ONEAPI_ROOT: The root folder of DPCPP compiler. Like "C:\Program Files (x86)\Intel\oneAPI".
        echo ^               ^The versions of oneAPI components used are defined in dependency_version.json.
        rem echo ^  ^AOT:       One of more flags of AOT (Ahead-of-Time) - [^"^"^|none^|flag^|flags]. Like none or dg2 or ^"dg2,mtl^".
        echo ^  ^AOT:         One of more flags of AOT (Ahead-of-Time). [flag^|flags]. Like dg2 or ^"dg2,mtl^".
        rem echo ^             ^"^":    no AOT.
        rem echo ^             ^none:  no AOT.
        echo ^               ^flag:  one AOT flag. Like dg2
        echo ^               ^flags: more AOT flags with ^"^". Like ^"dg2,mtl^"
        echo ^               ^Note, more flags will reduce the delay of first startup on more hardware, increase the package's size.
        echo ^  ^Target:      The target of building (Optional). [ipex^|all].
        echo ^               ^all:   Build torch, torchvision, torchaudio and Intel Extension for PyTorch^* with same python version. Default value.
        echo ^               ^ipex:  Build Intel Extension for PyTorch^* only.

        echo ^                    ^Prepare:
        echo ^                    ^  Install the pre-built versions of torch, torchvision, torchaudio by pip from https://pytorch-extension.intel.com.
        echo ^                    ^Note, it's possible to use different AOT for Intel Extension for PyTorch^* from torch.
    ENDLOCAL
goto:eof


:get_build_target
    if "%~1"=="ipex" (
        set %~2=1
    ) else (
        if "%~1"=="all" (
            set %~2=0
        )  else (
            if "%~1"=="" (
                set %~2=0
            ) else (
                echo Target value: [%~1] is wrong
                call:help
                exit /b 1
            )
        )
    )
goto:eof

:create_compenents_root_by_oneapi_root
    rem echo %~1 %~2 %~3 %~4

    set "INPUT_TMP_ONEAPI_ROOT=%~1"

    set IPEX_ROOT=intel-extension-for-pytorch

    for /f %%A in ('python %IPEX_ROOT%\scripts\tools\compilation_helper\dep_ver_utils.py -f %IPEX_ROOT%\dependency_version.json -k basekit:dpcpp-cpp-rt:version') do set DPCPP_VER=%%A
    set DPCPP_VER=%DPCPP_VER:~0,6%

    for /f %%A in ('python %IPEX_ROOT%\scripts\tools\compilation_helper\dep_ver_utils.py -f %IPEX_ROOT%\dependency_version.json -k basekit:mkl-dpcpp:version') do set MKL_VER=%%A
    set MKL_VER=%MKL_VER:~0,6%

    echo %MKL_VER%

    set "%~2=%INPUT_TMP_ONEAPI_ROOT%\compiler\%DPCPP_VER%"
    set "%~3=%INPUT_TMP_ONEAPI_ROOT%\mkl\%MKL_VER%"

goto:eof
