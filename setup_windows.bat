@echo off
cd /d %~dp0
py -m pip install -r requirements.txt
if errorlevel 1 goto error
py byte.py status
pause
exit /b 0
:error
echo Setup failed. Check that Python is installed and available as 'py'.
pause
exit /b 1
