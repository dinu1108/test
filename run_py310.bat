@echo off
echo [Setup] Connecting to existing environment 'py310'...

:: Try to find Conda Activate script
set "CONDA_ACTIVATE="
set "POSSIBLE_PATHS=%UserProfile%\anaconda3\Scripts\activate.bat;%UserProfile%\miniconda3\Scripts\activate.bat;C:\ProgramData\Anaconda3\Scripts\activate.bat;C:\ProgramData\miniconda3\Scripts\activate.bat;C:\Anaconda3\Scripts\activate.bat"

for %%P in ("%POSSIBLE_PATHS:;=" "%") do (
    if exist %%P (
        set "CONDA_ACTIVATE=%%~P"
        goto :FoundConda
    )
)

:FoundConda
if defined CONDA_ACTIVATE (
    echo [Setup] Found Conda at: %CONDA_ACTIVATE%
    call "%CONDA_ACTIVATE%"
    call conda activate py310
) else (
    echo [Error] Could not find Anaconda/Miniconda installation automatically.
    echo Please open "Anaconda Prompt" manually and navigate to this folder, then run:
    echo     conda activate py310
    echo     pip install -r requirements.txt
    echo     python main.py
    pause
    exit /b 1
)

:: 2. Install Project Requirements (into py310)
:: We assume PyTorch is already there as per user statement.
echo [Setup] Installing project dependencies into py310...
pip install -r requirements.txt

echo.
echo [Setup] Dependencies installed.
echo.

:: 3. Run Main
echo [Start] Running Auto Highlight Extractor...
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
python main.py %1
pause
