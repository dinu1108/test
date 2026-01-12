@echo off
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
    call "%CONDA_ACTIVATE%"
    call conda activate py310
    if errorlevel 1 (
        echo Failed to activate py310 environment.
        pause
    ) else (
        echo Activated py310 environment.
        cmd /k
    )
) else (
    echo Could not find Anaconda/Miniconda installation automatically.
    echo Please activate it manually.
    pause
)
