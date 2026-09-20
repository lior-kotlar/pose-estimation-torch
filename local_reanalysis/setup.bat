@echo off
rem One-time setup of this PC for re-analysing predicted movies. See LOCAL_REANALYSIS.md.
setlocal
set "ROOT=%~dp0.."
set "PY=%ROOT%\venv\Scripts\python.exe"
if exist "%PY%" goto packages
echo Creating the Python environment in %ROOT%\venv ...
py -3.11 -m venv "%ROOT%\venv"
if errorlevel 1 goto nopython
:packages
echo.
echo Installing the Python packages. The first time this downloads about 1 GB and takes a few minutes.
"%PY%" -m pip install --upgrade pip --disable-pip-version-check
"%PY%" -m pip install -r "%ROOT%\requirements-analysis.txt" --disable-pip-version-check
if errorlevel 1 goto failed
echo.
"%PY%" -X utf8 "%ROOT%\code\local_reanalysis.py" setup
goto end
:nopython
echo.
echo Python 3.11 was not found. Install "Windows installer 64-bit" from
echo https://www.python.org/downloads/release/python-3119/ and run setup.bat again.
goto end
:failed
echo.
echo Installing the packages failed; see the messages above.
:end
pause
