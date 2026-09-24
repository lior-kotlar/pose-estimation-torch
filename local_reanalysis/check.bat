@echo off
rem Say what each movie in a folder would need, and stop. Nothing is uploaded and nothing is
rem changed. Safe to run at any time, on any PC. See LOCAL_REANALYSIS.md.
setlocal
set "ROOT=%~dp0.."
set "PY=%ROOT%\venv\Scripts\python.exe"
if not exist "%PY%" (
    echo This PC is not set up yet: run local_reanalysis\setup.bat first.
    pause
    exit /b 1
)
"%PY%" -X utf8 "%ROOT%\code\local_reanalysis.py" run --check %*
pause
