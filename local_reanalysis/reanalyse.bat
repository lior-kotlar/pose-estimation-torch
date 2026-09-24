@echo off
rem Work out what each movie in a folder still needs -- realigning, re-analysing, rendering its
rem video -- and do only that, then collect and upload. Drag a folder onto this file, or
rem double-click and paste one. See LOCAL_REANALYSIS.md.
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
