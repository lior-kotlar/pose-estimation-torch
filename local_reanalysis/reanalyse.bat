@echo off
rem Re-analyse, collect and upload. Drag a folder onto this file, or double-click and paste one.
rem See LOCAL_REANALYSIS.md.
setlocal
set "ROOT=%~dp0.."
set "PY=%ROOT%\venv\Scripts\python.exe"
if not exist "%PY%" (
    echo This PC is not set up yet: run local_reanalysis\setup.bat first.
    pause
    exit /b 1
)
"%PY%" -X utf8 "%ROOT%\code\local_reanalysis.py" run %*
pause
