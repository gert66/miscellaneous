@echo off
setlocal EnableDelayedExpansion

:: ============================================================
:: run_enricher_queue_generic.bat
:: mYngle Lead Prioritizer — generic batch runner
::
:: Usage (from CMD):
::   run_enricher_queue_generic.bat [dry|test|full] [max_rows]
::
:: Modes:
::   dry   — path/config check only, no API calls, no enrichment
::   test  — enrich first MAX_ROWS_TEST rows (default: 5)
::   full  — enrich all rows (default)
::
:: Config:
::   Edit the CONFIG section below to point to your input files,
::   output folders, and Python script path.
::
:: Output:
::   - Live Python [enricher] progress lines visible in this CMD window
::   - All output also written to LOG_FILE
:: ============================================================

:: ── CONFIG ───────────────────────────────────────────────────────────────────

:: Path to enrich_clients_claude.py (absolute or relative to this bat file)
set "SCRIPT_FILE=%~dp0enrich_clients_claude.py"

:: Input file — override on command line or edit here
set "INPUT_FILE=%~dp0input\batch_input.xlsx"

:: Output directory for enrichedResults_*.xlsx
set "OUTPUT_DIR=%~dp0output"

:: Log directory and file
set "LOG_DIR=%~dp0logs"
set "LOG_FILE=%LOG_DIR%\enricher_%DATE:~10,4%%DATE:~4,2%%DATE:~7,2%_%TIME:~0,2%%TIME:~3,2%%TIME:~6,2%.log"

:: Rows to process in test mode
set "MAX_ROWS_TEST=5"

:: Stop queue if any batch returns non-zero exit code (1=yes, 0=no)
set "STOP_ON_ERROR=1"

:: API keys — leave empty to load from .streamlit/secrets.toml or environment
set "ANTHROPIC_KEY="
set "SERPER_KEY="

:: ── END CONFIG ───────────────────────────────────────────────────────────────

:: Parse mode argument (dry / test / full)
set "MODE=full"
if /I "%~1"=="dry"  set "MODE=dry"
if /I "%~1"=="test" set "MODE=test"
if /I "%~1"=="full" set "MODE=full"

:: Parse optional max_rows override
set "MAX_ROWS_OVERRIDE="
if not "%~2"=="" set "MAX_ROWS_OVERRIDE=%~2"

:: Sanitise log timestamp (colons in %TIME% break filenames on some systems)
set "LOG_FILE=%LOG_DIR%\enricher_%DATE:~10,4%%DATE:~4,2%%DATE:~7,2%.log"

:: Create required directories
if not exist "%OUTPUT_DIR%" mkdir "%OUTPUT_DIR%"
if not exist "%LOG_DIR%"    mkdir "%LOG_DIR%"

:: ── Build Python arg list ─────────────────────────────────────────────────────
set "PY_ARGS=--input "%INPUT_FILE%" --output-dir "%OUTPUT_DIR%""

if not "%ANTHROPIC_KEY%"=="" set "PY_ARGS=%PY_ARGS% --anthropic-key "%ANTHROPIC_KEY%""
if not "%SERPER_KEY%"==""    set "PY_ARGS=%PY_ARGS% --serper-key "%SERPER_KEY%""

if "%MODE%"=="dry" (
    set "PY_ARGS=%PY_ARGS% --dry-run-paths"
) else if "%MODE%"=="test" (
    if not "%MAX_ROWS_OVERRIDE%"=="" (
        set "PY_ARGS=%PY_ARGS% --max-rows %MAX_ROWS_OVERRIDE%"
    ) else (
        set "PY_ARGS=%PY_ARGS% --max-rows %MAX_ROWS_TEST%"
    )
) else (
    if not "%MAX_ROWS_OVERRIDE%"=="" set "PY_ARGS=%PY_ARGS% --max-rows %MAX_ROWS_OVERRIDE%"
)

:: ── Header ───────────────────────────────────────────────────────────────────
echo.
echo ============================================================
echo  START ENRICHMENT BATCH
echo  Mode:        %MODE%
echo  Input:       %INPUT_FILE%
echo  Output dir:  %OUTPUT_DIR%
echo  Log file:    %LOG_FILE%
echo  LIVE PYTHON OUTPUT: enabled
echo ============================================================
echo.

:: Write header to log file
(
echo ============================================================
echo  START ENRICHMENT BATCH
echo  Mode:        %MODE%
echo  Date/time:   %DATE% %TIME%
echo  Input:       %INPUT_FILE%
echo  Output dir:  %OUTPUT_DIR%
echo  Log file:    %LOG_FILE%
echo  Script:      %SCRIPT_FILE%
echo ============================================================
echo.
) >> "%LOG_FILE%"

:: ── Run Python with live tee via PowerShell ───────────────────────────────────
::
:: python -u    = unbuffered stdout/stderr (line-by-line live output)
:: Tee-Object   = writes each line to console AND appends to LOG_FILE
:: $LASTEXITCODE passed back via exit so errorlevel is preserved correctly
::
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$env:PYTHONUNBUFFERED='1'; " ^
  "& python -u '%SCRIPT_FILE%' %PY_ARGS% 2>&1 | Tee-Object -FilePath '%LOG_FILE%' -Append; " ^
  "exit $LASTEXITCODE"

set "EXIT_CODE=%ERRORLEVEL%"

:: ── Completion ────────────────────────────────────────────────────────────────
echo.
if "%EXIT_CODE%"=="0" (
    echo OK: batch completed successfully.
    echo OK: batch completed successfully. >> "%LOG_FILE%"
) else (
    echo ERROR: Python exited with code %EXIT_CODE%.
    echo ERROR: Python exited with code %EXIT_CODE%. >> "%LOG_FILE%"
    if "%STOP_ON_ERROR%"=="1" (
        echo STOP_ON_ERROR=1 -- halting queue.
        echo STOP_ON_ERROR=1 -- halting queue. >> "%LOG_FILE%"
        exit /b %EXIT_CODE%
    )
)

echo Log saved: %LOG_FILE%
echo.

endlocal
exit /b %EXIT_CODE%
