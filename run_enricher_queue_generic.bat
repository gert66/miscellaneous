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
::
:: Live output implementation:
::   Paths are passed to PowerShell via environment variables, not via
::   %PY_ARGS% string substitution.  This avoids CMD/PowerShell quote-nesting
::   conflicts that silently break the command when paths contain spaces or
::   when the inner "" of --input "..." close the outer PS string early.
::   python -u  +  PYTHONUNBUFFERED=1  ensure unbuffered line-by-line output.
::   Tee-Object writes every line to both CMD window and LOG_FILE.
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

:: ── Verify SCRIPT_FILE exists before attempting to run ───────────────────────
if not exist "%SCRIPT_FILE%" (
    echo.
    echo ERROR: SCRIPT_FILE not found.
    echo        Expected: %SCRIPT_FILE%
    echo        Edit the CONFIG section in this bat file to set the correct path.
    echo.
    exit /b 1
)

:: ── Header (console) ─────────────────────────────────────────────────────────
echo.
echo ============================================================
echo  START ENRICHMENT BATCH
echo  Mode:              %MODE%
echo  Input:             %INPUT_FILE%
echo  Output dir:        %OUTPUT_DIR%
echo  Log file:          %LOG_FILE%
echo  Script:            %SCRIPT_FILE%
echo  LIVE PYTHON OUTPUT: enabled
echo  PYTHON UNBUFFERED:  enabled (python -u + PYTHONUNBUFFERED=1)
echo ============================================================
echo.

:: ── Header (log file) ────────────────────────────────────────────────────────
(
echo ============================================================
echo  START ENRICHMENT BATCH
echo  Mode:          %MODE%
echo  Date/time:     %DATE% %TIME%
echo  Input:         %INPUT_FILE%
echo  Output dir:    %OUTPUT_DIR%
echo  Log file:      %LOG_FILE%
echo  Script:        %SCRIPT_FILE%
echo ============================================================
echo.
) >> "%LOG_FILE%"

:: ── Python / script identity check ───────────────────────────────────────────
echo  Python version:
python --version
echo.
echo  Script identity:
python -c "import pathlib; p=pathlib.Path(r'%SCRIPT_FILE%'); print('  SCRIPT_EXISTS:', p.exists()); print('  SCRIPT_MTIME: ', int(p.stat().st_mtime) if p.exists() else 'MISSING')"
echo.

:: ── Resolve mode flags into separate env vars ─────────────────────────────────
:: Passing a single PY_ARGS string with embedded "" into PowerShell -Command
:: breaks the PS string parser.  Instead, pass everything via env vars and
:: reconstruct the arg array cleanly inside PowerShell.

set "PYTHONUNBUFFERED=1"

:: Paths
set "_RUN_SCRIPT=%SCRIPT_FILE%"
set "_RUN_INPUT=%INPUT_FILE%"
set "_RUN_OUTPUT=%OUTPUT_DIR%"
set "_RUN_LOG=%LOG_FILE%"

:: Keys (passed to PS but never printed to console)
set "_RUN_ANTHROPIC=%ANTHROPIC_KEY%"
set "_RUN_SERPER=%SERPER_KEY%"

:: Mode flags
set "_RUN_DRY="
set "_RUN_MAXROWS="

if "%MODE%"=="dry" (
    set "_RUN_DRY=1"
) else if "%MODE%"=="test" (
    if not "%MAX_ROWS_OVERRIDE%"=="" (
        set "_RUN_MAXROWS=%MAX_ROWS_OVERRIDE%"
    ) else (
        set "_RUN_MAXROWS=%MAX_ROWS_TEST%"
    )
) else (
    if not "%MAX_ROWS_OVERRIDE%"=="" set "_RUN_MAXROWS=%MAX_ROWS_OVERRIDE%"
)

:: ── Run Python — live output to CMD window + log via Tee-Object ──────────────
::
:: All paths come from $env:_RUN_* variables, so no quote nesting in -Command.
:: @xargs is a PowerShell array built element-by-element — handles spaces in paths.
:: python -u = unbuffered; 2>&1 merges stderr into stdout for tee.
:: $LASTEXITCODE after the pipe reflects python's exit (Tee-Object is a PS cmdlet
:: and does not overwrite $LASTEXITCODE set by the native python process).
::
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$s=$env:_RUN_SCRIPT; $i=$env:_RUN_INPUT; $o=$env:_RUN_OUTPUT; $l=$env:_RUN_LOG;" ^
  "$ak=$env:_RUN_ANTHROPIC; $sk=$env:_RUN_SERPER;" ^
  "$mr=$env:_RUN_MAXROWS; $dry=$env:_RUN_DRY;" ^
  "$xargs = @('--input', $i, '--output-dir', $o);" ^
  "if ($ak) { $xargs += '--anthropic-key'; $xargs += $ak };" ^
  "if ($sk) { $xargs += '--serper-key';    $xargs += $sk };" ^
  "if ($dry -eq '1') { $xargs += '--dry-run-paths' }" ^
  "elseif ($mr)      { $xargs += '--max-rows'; $xargs += $mr };" ^
  "& python -u `"$s`" @xargs 2>&1 | Tee-Object -FilePath `"$l`" -Append;" ^
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
