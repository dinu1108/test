@echo off
setlocal
echo [Setup] Attempting to find Python 3.10...

:: Try 'py -3.10' launcher first (Standard Windows)
py -3.10 --version >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo [Setup] Found Python 3.10 via 'py' launcher.
    set PYTHON_CMD=py -3.10
    goto :create_venv
)

:: Try just 'python' (maybe it IS 3.10?)
python --version 2>&1 | findstr "3.10" >nul
if %ERRORLEVEL% EQU 0 (
    echo [Setup] default 'python' is 3.10.
    set PYTHON_CMD=python
    goto :create_venv
)

echo [Error] Could not find Python 3.10 automatically.
echo Please ensure Python 3.10 is installed and available via 'py -3.10' or 'python'.
pause
exit /b 1

:create_venv
echo [Setup] Creating Virtual Environment 'venv_310' using %PYTHON_CMD%...
%PYTHON_CMD% -m venv venv_310

echo [Setup] Activating venv_310...
call venv_310\Scripts\activate

echo [Setup] Upgrading pip...
python -m pip install --upgrade pip

echo [Setup] Installing PyTorch (CUDA 12.1 Stable for 3.10)...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

echo [Setup] Installing requirements...
pip install -r requirements.txt

echo.
echo ========================================================
echo [Setup] DONE!
echo ========================================================
echo [Usage] To run the program, use start.bat (I will create it for you).
echo.
pause
