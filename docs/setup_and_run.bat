@echo off
REM Set the code page to UTF-8
chcp 65001

REM Set the PYTHONIOENCODING environment variable to utf-8
set PYTHONIOENCODING=utf-8

setlocal


:: Check if Python is installed
echo Checking if Python is installed...
where python >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo Python not found. Attempting to install Python...

    REM Download Python installer
    powershell -Command "Invoke-WebRequest -Uri 'https://www.python.org/ftp/python/3.9.13/python-3.9.13-amd64.exe' -OutFile 'python-installer.exe'" >nul 2>&1
    if %ERRORLEVEL% neq 0 (
        echo Failed to download Python installer.
        pause
        exit /b 1
    )
    
    REM Install Python silently
    start /wait python-installer.exe /quiet InstallAllUsers=1 PrependPath=1 >nul 2>&1
    if %ERRORLEVEL% neq 0 (
        echo Python installer failed. Please install Python from the Microsoft Store or manually from the official website.
        pause
        exit /b 1
    )

    REM Check if Python installation was successful
    where python >nul 2>&1
    if %ERRORLEVEL% neq 0 (
        echo Failed to install Python. Please check the installation process.
        pause
        exit /b 1
    )
) else (
    echo Python is already installed.
)

:: Check if pip is installed
echo Checking if pip is installed...
where pip >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo Pip not found. Installing pip...
    python -m ensurepip >nul 2>&1
    if %ERRORLEVEL% neq 0 (
        echo Failed to install pip.
        pause
        exit /b 1
    )
) else (
    echo Pip is already installed.
)

:: Check if requirements.txt exists
if exist "requirements.txt" (
    echo requirements.txt found. Installing dependencies from requirements.txt...
    pip install -r requirements.txt
    if %ERRORLEVEL% neq 0 (
        echo Failed to install dependencies from requirements.txt.
        pause
        exit /b 1
    )
) else (
    echo requirements.txt not found.
    pause
    exit /b 1
)

:: Run the interface script
echo Running the interface script...
python interface.py
if %ERRORLEVEL% neq 0 (
    echo Failed to run the interface script.
    pause
    exit /b 1
)

endlocal

echo Script completed successfully.
pause
