@echo off
rem Say what predict.bat would send for a folder, and stop: it reads only this PC (the first
rem time it also works out the experiment's prep.json). Nothing is uploaded. See LOCAL_PREDICT.md.
setlocal
set "ROOT=%~dp0.."
set "PY=%ROOT%\venv\Scripts\python.exe"
if not exist "%PY%" (
    echo This PC is not set up yet: run local_reanalysis\setup.bat first.
    pause
    exit /b 1
)
"%PY%" -X utf8 "%ROOT%\code\local_reanalysis.py" predict --check %*
pause
