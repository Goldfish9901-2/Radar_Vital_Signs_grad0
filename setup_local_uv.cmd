@echo off
setlocal
set "ANALYSIS_PYTHON=%~1"
if not defined ANALYSIS_PYTHON set "ANALYSIS_PYTHON=%APPDATA%\uv\python\cpython-3.13.14-windows-x86_64-none\python.exe"
if not exist "%ANALYSIS_PYTHON%" (
  echo Supply the real versioned Python executable path as the first argument.
  exit /b 1
)
"%ANALYSIS_PYTHON%" -m venv --copies --without-pip .venv
if errorlevel 1 exit /b 1
if not exist tmp mkdir tmp
uv export --quiet --locked --format requirements-txt --no-emit-project --output-file tmp\analysis-requirements.txt
if errorlevel 1 exit /b 1
uv pip sync --python .venv\Scripts\python.exe --link-mode copy --require-hashes tmp\analysis-requirements.txt
if errorlevel 1 exit /b 1
uv run --locked --no-sync python --version
if errorlevel 1 exit /b 1
echo Copy environment ready. Use uv run --locked --no-sync python ...
