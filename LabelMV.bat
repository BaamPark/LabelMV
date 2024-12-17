@echo off
REM Check if opencv-python and ultralytics are installed

echo Checking if OpenCV and Ultralytics are installed...

REM Check for opencv-python
pip list | findstr "opencv-python" >nul
IF %ERRORLEVEL%==0 (
    echo OpenCV is already installed.
) ELSE (
    echo OpenCV is not installed. Installing now...
    pip install opencv-python
)

REM Check for ultralytics
pip list | findstr "ultralytics" >nul
IF %ERRORLEVEL%==0 (
    echo Ultralytics is already installed.
) ELSE (
    echo Ultralytics is not installed. Installing now...
    pip install ultralytics
)

python launcher.py