@echo off
setlocal EnableDelayedExpansion

:: ============================================================
:: run_enricher_queue_generic.bat
:: mYngle Lead Prioritizer — generic batch runner v4
::
:: Usage (from CMD):
::   run_enricher_queue_generic.bat [dry|test|full] [max_rows]
::
:: Modes:
::   dry   — path/config check only, no API calls, no enrichment
::   test  — enrich first MAX_ROWS_TEST rows (default: 5)
::   full  — enrich all rows (default)
::
:: v4 fix notes:
::   - Replaces fragile powershell -Command "^..." block with a temp .ps1
::     file run via powershell -File.  This eliminates all CMD/PS
::     quote-nesting issues that broke paths with spaces
::     (e.g. OneDrive - UMC Utrecht).
::   - Locale-safe timestamp via PowerShell Get-Date (fixes Dutch/EU
::     date format breaking log filename: enricher_0264-6-.log).
::   - Debug block prints Python exe, script path, input, output dir,
::     log file, and args (keys excluded) before Python starts.
::   - All existing behaviour preserved: modes dry/test/full,
::     max_rows, Tee-Object logging, PYTHONUNBUFFERED, STOP_ON_ERROR.
:: ============================================================

:: ── CONFIG ───────────────────────────────────────────────────────────────────

:: Path to enrich_clients_claude.py (absolute, relative to this bat)
set "SCRIPT_FILE=%~dp0enrich_clients_claude.py"

:: Input file
set "INPUT_FILE=%~dp0input\batch_input.xlsx"

:: Output directory for enrichedResults_*.xlsx
set "OUTPUT_DIR=%~dp0output"

:: Log directory
set "LOG_DIR=%~dp0logs"

:: Rows to process in test mode (override with 2nd CLI argument)
set "MAX_ROWS_TEST=5"

:: Halt queue on non-zero exit from Python (1=yes, 0=no)
set "STOP_ON_ERROR=1"

:: API keys — leave empty to load from .streamlit/secrets.toml or env
set "ANTHROPIC_KEY="
set "SERPER_KEY="

:: ── END CONFIG ───────────────────────────────────────────────────────────────

:: ── Parse mode (dry / test / full) ───────────────────────────────────────────
set "MODE=full"
if /I "%~1"=="dry"  set "MODE=dry"
if /I "%~1"=="test" set "MODE=test"
if /I "%~1"=="full" set "MODE=full"

:: ── Optional max_rows override ────────────────────────────────────────────────
set "MAX_ROWS_OVERRIDE="
if not "%~2"=="" set "MAX_ROWS_OVERRIDE=%~2"

:: ── Locale-safe timestamp (avoids Dutch/EU %DATE% format bugs) ───────────────
:: %DATE% on Dutch Windows looks like "zo 14-06-2026" which breaks substring
:: extraction.  PowerShell Get-Date always returns yyyyMMdd_HHmmss.
for /f "usebackq delims=" %%T in (`powershell -NoProfile -Command "Get-Date -Format yyyyMMdd_HHmmss"`) do set "LOG_STAMP=%%T"
set "LOG_FILE=%LOG_DIR%\enricher_%LOG_STAMP%.log"

:: ── Create directories ────────────────────────────────────────────────────────
if not exist "%OUTPUT_DIR%" mkdir "%OUTPUT_DIR%"
if not exist "%LOG_DIR%"    mkdir "%LOG_DIR%"

:: ── Verify enrich_clients_claude.py exists before anything else ───────────────
if not exist "%SCRIPT_FILE%" (
    echo.
    echo ERROR: SCRIPT_FILE not found.
    echo        Expected: %SCRIPT_FILE%
    echo        Edit the CONFIG section in this bat file to set the correct path.
    echo.
    exit /b 1
)

:: ── Console header ────────────────────────────────────────────────────────────
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

:: ── Log file header ───────────────────────────────────────────────────────────
echo ============================================================>>"%LOG_FILE%"
echo  START ENRICHMENT BATCH>>"%LOG_FILE%"
echo  Mode:          %MODE%>>"%LOG_FILE%"
echo  Date/time:     %LOG_STAMP%>>"%LOG_FILE%"
echo  Input:         %INPUT_FILE%>>"%LOG_FILE%"
echo  Output dir:    %OUTPUT_DIR%>>"%LOG_FILE%"
echo  Log file:      %LOG_FILE%>>"%LOG_FILE%"
echo  Script:        %SCRIPT_FILE%>>"%LOG_FILE%"
echo ============================================================>>"%LOG_FILE%"

:: ── Python version and script identity check ──────────────────────────────────
echo  Python version:
python --version
echo.
echo  Script identity:
python -c "import pathlib; p=pathlib.Path(r'%SCRIPT_FILE%'); print('  SCRIPT_EXISTS:', p.exists()); print('  SCRIPT_MTIME: ', int(p.stat().st_mtime) if p.exists() else 'MISSING')"
echo.

:: ── Pass all config into env vars for the PS1 to pick up ─────────────────────
:: Keys are passed via env (never echoed to console).
:: Mode flags are broken into _RUN_DRY and _RUN_MAXROWS strings.
set "PYTHONUNBUFFERED=1"
set "_RUN_SCRIPT=%SCRIPT_FILE%"
set "_RUN_INPUT=%INPUT_FILE%"
set "_RUN_OUTPUT=%OUTPUT_DIR%"
set "_RUN_LOG=%LOG_FILE%"
set "_RUN_ANTHROPIC=%ANTHROPIC_KEY%"
set "_RUN_SERPER=%SERPER_KEY%"
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

:: ── Write temp PowerShell runner script ──────────────────────────────────────
:: Strategy: write the PS1 with individual  echo line>>file  statements.
:: These are NOT inside a CMD ( ) block, so ) in PS code is just a literal
:: character and does not close any CMD group.
:: CMD special chars in PS code are escaped with ^:
::   &  ->  ^&      |  ->  ^|      >  ->  ^>
:: The PS1 reads all config from $env:_RUN_* so no path quoting is needed
:: inside the PS script itself.
:: powershell -File avoids -Command quote-nesting entirely.
set "TEMP_PS1=%TEMP%\enricher_run_%RANDOM%.ps1"

echo $pythonExe  = 'python'>>"%TEMP_PS1%"
echo $scriptPath = $env:_RUN_SCRIPT>>"%TEMP_PS1%"
echo $inputFile  = $env:_RUN_INPUT>>"%TEMP_PS1%"
echo $outputDir  = $env:_RUN_OUTPUT>>"%TEMP_PS1%"
echo $logFile    = $env:_RUN_LOG>>"%TEMP_PS1%"
echo $antKey     = $env:_RUN_ANTHROPIC>>"%TEMP_PS1%"
echo $serperKey  = $env:_RUN_SERPER>>"%TEMP_PS1%"
echo $maxRows    = $env:_RUN_MAXROWS>>"%TEMP_PS1%"
echo $isDry      = $env:_RUN_DRY>>"%TEMP_PS1%"
echo # --- debug block (paths only, keys excluded) --->>"%TEMP_PS1%"
echo Write-Host "[runner] Python:     $pythonExe">>"%TEMP_PS1%"
echo Write-Host "[runner] Script:     $scriptPath">>"%TEMP_PS1%"
echo Write-Host "[runner] Input:      $inputFile">>"%TEMP_PS1%"
echo Write-Host "[runner] Output dir: $outputDir">>"%TEMP_PS1%"
echo Write-Host "[runner] Log file:   $logFile">>"%TEMP_PS1%"
echo if ($maxRows)        { Write-Host "[runner] Max rows:   $maxRows" }>>"%TEMP_PS1%"
echo if ($isDry -eq '1') { Write-Host "[runner] Mode:       DRY RUN (no API calls)" }>>"%TEMP_PS1%"
echo if ($antKey)         { Write-Host "[runner] Anthropic key: present (not shown)" }>>"%TEMP_PS1%"
echo if ($serperKey)      { Write-Host "[runner] Serper key:    present (not shown)" }>>"%TEMP_PS1%"
echo Write-Host "">>"%TEMP_PS1%"
echo # --- build arg array --->>"%TEMP_PS1%"
echo $xargs = @('--input', $inputFile, '--output-dir', $outputDir)>>"%TEMP_PS1%"
echo if ($antKey)    { $xargs += '--anthropic-key'; $xargs += $antKey }>>"%TEMP_PS1%"
echo if ($serperKey) { $xargs += '--serper-key';    $xargs += $serperKey }>>"%TEMP_PS1%"
echo if ($isDry -eq '1') {>>"%TEMP_PS1%"
echo     $xargs += '--dry-run-paths'>>"%TEMP_PS1%"
echo } elseif ($maxRows) {>>"%TEMP_PS1%"
echo     $xargs += '--max-rows'>>"%TEMP_PS1%"
echo     $xargs += $maxRows>>"%TEMP_PS1%"
echo }>>"%TEMP_PS1%"
echo # --- run Python with live output tee'd to log --->>"%TEMP_PS1%"
echo ^& $pythonExe -u $scriptPath @xargs 2^>^&1 ^| Tee-Object -FilePath $logFile -Append>>"%TEMP_PS1%"
echo exit $LASTEXITCODE>>"%TEMP_PS1%"

:: ── Execute the temp PS1 (no -Command, no quote-nesting issues) ───────────────
powershell -NoProfile -ExecutionPolicy Bypass -File "%TEMP_PS1%"
set "EXIT_CODE=%ERRORLEVEL%"

:: ── Clean up temp PS1 ─────────────────────────────────────────────────────────
del "%TEMP_PS1%" 2>nul

:: ── Completion ────────────────────────────────────────────────────────────────
echo.
if "%EXIT_CODE%"=="0" (
    echo OK: batch completed successfully.
    echo OK: batch completed successfully.>>"%LOG_FILE%"
) else (
    echo ERROR: Python exited with code %EXIT_CODE%.
    echo ERROR: Python exited with code %EXIT_CODE%.>>"%LOG_FILE%"
    if "%STOP_ON_ERROR%"=="1" (
        echo STOP_ON_ERROR=1 -- halting queue.
        echo STOP_ON_ERROR=1 -- halting queue.>>"%LOG_FILE%"
        exit /b %EXIT_CODE%
    )
)

echo Log saved: %LOG_FILE%
echo.

endlocal
exit /b %EXIT_CODE%
