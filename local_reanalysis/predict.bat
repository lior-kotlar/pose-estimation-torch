@echo off
rem Prep and predict, on the lab cluster, the raw movies in a folder on this PC, and bring the
rem results home. Drag an experiment folder (or a folder of experiments) onto this file, or
rem double-click and paste one. The first time it asks only what the movies cannot tell it.
rem Closing the window is safe: run it again and it carries on. See LOCAL_PREDICT.md.
setlocal
set "ROOT=%~dp0.."
set "PY=%ROOT%\venv\Scripts\python.exe"
if not exist "%PY%" (
    echo This PC is not set up yet: run local_reanalysis\setup.bat first.
    pause
    exit /b 1
)
"%PY%" -X utf8 "%ROOT%\code\local_reanalysis.py" predict %*
pause
