@echo off
rem Same as reanalyse.bat, but ALSO collects and uploads movies from bad_signal / bad_wings
rem folders. They go under their experiment's own bad_signal\ / bad_wings\ subfolder, never
rem among the experiment's usable movies. See LOCAL_REANALYSIS.md.
setlocal
set "ROOT=%~dp0.."
set "PY=%ROOT%\venv\Scripts\python.exe"
if not exist "%PY%" (
    echo This PC is not set up yet: run local_reanalysis\setup.bat first.
    pause
    exit /b 1
)
"%PY%" -X utf8 "%ROOT%\code\local_reanalysis.py" run --include-bad %*
pause
