@echo off
rem Download the latest committed code from the server. See LOCAL_REANALYSIS.md.
setlocal
set "ROOT=%~dp0.."
set "PY=%ROOT%\venv\Scripts\python.exe"
if not exist "%PY%" (
    echo This PC is not set up yet: run local_reanalysis\setup.bat first.
    pause
    exit /b 1
)
rem one line on purpose: the update replaces this very file, and cmd has already read the whole line
"%PY%" -X utf8 "%ROOT%\code\local_reanalysis.py" update & pause & exit /b
