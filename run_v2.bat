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
    echo     python hybrid_agent_v2/main_v2.py
    pause
    exit /b 1
)

:: 2. Upgrade Deps
echo [Setup] Installing Hybrid Agent dependencies...
pip install -r hybrid_agent_v2/requirements.txt

echo.
echo [Start] Running Hybrid Agent V2...
python main_v2.py %1
pause
