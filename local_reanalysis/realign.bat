@echo off
rem Re-run the ensemble of movies whose wings are labelled the other way round in one of the
rem models, on the lab cluster, put the new 3D points in place here, then re-analyse them.
rem On a PC that may not write to the cluster it only reports which movies are affected.
rem A round takes hours. You can close this window and run it again to pick it up. See
rem LOCAL_REANALYSIS.md.
setlocal
set "ROOT=%~dp0.."
set "PY=%ROOT%\venv\Scripts\python.exe"
if not exist "%PY%" (
    echo This PC is not set up yet: run local_reanalysis\setup.bat first.
    pause
    exit /b 1
)
"%PY%" -X utf8 "%ROOT%\code\local_reanalysis.py" realign %*
pause
