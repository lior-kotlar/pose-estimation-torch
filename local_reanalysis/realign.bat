@echo off
rem Only repair ensembles that put both wings on one wing; leave the rest alone. The everyday
rem command is reanalyse.bat, which does this when a movie needs it. See LOCAL_REANALYSIS.md.
setlocal
set "ROOT=%~dp0.."
set "PY=%ROOT%\venv\Scripts\python.exe"
if not exist "%PY%" (
    echo This PC is not set up yet: run local_reanalysis\setup.bat first.
    pause
    exit /b 1
)
"%PY%" -X utf8 "%ROOT%\code\local_reanalysis.py" run --only realign %*
pause
