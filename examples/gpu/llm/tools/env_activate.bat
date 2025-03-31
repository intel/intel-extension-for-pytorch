@echo off
setlocal enabledelayedexpansion

:: Usage message
set "MSG_USAGE=Usage: call %~nx0 [inference|fine-tuning|bitsandbytes|training]"

:: Check if an argument is provided
if "%~1"=="" (
    echo !MSG_USAGE!
    exit /b 1
)

:: Get the mode from the first argument
set MODE=%1

:: Validate MODE
if not "%MODE%"=="inference" if not "%MODE%"=="fine-tuning" if not "%MODE%"=="bitsandbytes" if not "%MODE%"=="training" (
    echo !MSG_USAGE!
    exit /b 2
)

:: Get the script's directory (equivalent to BASEFOLDER in Bash)
set BASEFOLDER=%~dp0

:: Call the Python script with the given mode
python "%BASEFOLDER%\env_activate.py" %MODE%
endlocal & set "GBASEFOLDER=%BASEFOLDER%" & set "GMODE=%MODE%"
cd /d "%GBASEFOLDER%\..\%GMODE%"
