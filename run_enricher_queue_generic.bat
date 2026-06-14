@echo off
setlocal EnableDelayedExpansion

:: ============================================================
:: run_enricher_queue_generic.bat
:: mYngle Lead Prioritizer — queue-aware batch runner v5
::
:: Usage:
::   run_enricher_queue_generic.bat <queue> "<batches>" <mode> [max_rows]
::
:: Arguments:
::   queue     - Queue name or numeric alias: Italy100, Italy200, Germany
::               Aliases: 1=Italy100, 2=Italy200
::   batches   - Batch number(s) in quotes: "1" or "1 2 3" or "11 12 13"
::   mode      - dry | test | full
::   max_rows  - (optional) override test row limit
::
:: Examples:
::   run_enricher_queue_generic.bat Italy100 "1" dry
::   run_enricher_queue_generic.bat Italy100 "1" test
::   run_enricher_queue_generic.bat Italy100 "1 2 3" full
::   run_enricher_queue_generic.bat Italy200 "11 12 13" test
::   run_enricher_queue_generic.bat Germany "1" test
::   run_enricher_queue_generic.bat 1 "1" test    (alias: 1=Italy100)
::
:: Input file discovery (tries in order for each batch):
::   1. {PROJECT_ROOT}\{Queue}\01_cleaned_domains\{Queue}_{N}_*_cleaned_*.xlsx
::   2. {PROJECT_ROOT}\{Queue}\01_cleaned_domains\{Queue}_{N}_*.xlsx
::   3. {BAT_DIR}\input\{Queue}_{N}*_cleaned_*.xlsx
::   4. {BAT_DIR}\input\{Queue}_{N}*.xlsx
::   Most-recently-modified file wins when multiple files match.
::
:: Output goes to: {BAT_DIR}\output\{Queue}\batch_{N}\
:: Logs go to:     {BAT_DIR}\logs\enricher_{Queue}_{N}_{timestamp}.log
::
:: Implementation: writes a temp .ps1 per batch and runs it with
::   powershell -File  to avoid all CMD/PS quote-nesting issues.
::   Tee-Object streams live Python output to console + log.
:: ============================================================

:: ── CONFIG ───────────────────────────────────────────────────────────────────

:: Path to enrich_clients_claude.py
set "SCRIPT_FILE=%~dp0enrich_clients_claude.py"

:: Project root: parent folder that contains Italy100\, Italy200\, Germany\ etc.
:: Default: one level above this repo (i.e. the GitHub / Myngle root folder).
:: Override this if your pipeline data is elsewhere.
set "PROJECT_ROOT=%~dp0.."

:: Fallback flat input folder (used if no file found in pipeline structure)
set "FLAT_INPUT_DIR=%~dp0input"

:: Base output and log dirs (relative to this bat file)
set "BASE_OUTPUT_DIR=%~dp0output"
set "LOG_DIR=%~dp0logs"

:: Rows to process in test mode (overridable with 4th arg)
set "MAX_ROWS_TEST=5"

:: Halt entire queue run on first non-zero Python exit (1=yes, 0=no)
set "STOP_ON_ERROR=1"

:: API keys — leave empty to load from .streamlit/secrets.toml or environment
set "ANTHROPIC_KEY="
set "SERPER_KEY="

:: ── END CONFIG ───────────────────────────────────────────────────────────────

:: ── Show usage if no args ─────────────────────────────────────────────────────
if "%~1"=="" goto :show_usage

:: ── Detect legacy single-arg format: run_enricher_queue_generic.bat dry|test|full ──
:: If the first arg is a mode keyword (and no queue was intended), fall back to the
:: original single-INPUT_FILE behaviour using input\batch_input.xlsx.
set "_ARG1=%~1"
if /I "!_ARG1!"=="dry"  goto :legacy_mode
if /I "!_ARG1!"=="test" goto :legacy_mode
if /I "!_ARG1!"=="full" goto :legacy_mode

:: ── New queue-based argument parsing ─────────────────────────────────────────
set "QUEUE_RAW=%~1"
set "BATCH_NUMBERS=%~2"
set "MODE_RAW=%~3"
set "MAX_ROWS_OVERRIDE=%~4"

:: Apply numeric aliases
set "QUEUE_NAME=%QUEUE_RAW%"
if "%QUEUE_RAW%"=="1" set "QUEUE_NAME=Italy100"
if "%QUEUE_RAW%"=="2" set "QUEUE_NAME=Italy200"

:: Parse mode (default: full if not recognised)
set "MODE=full"
if /I "%MODE_RAW%"=="dry"  set "MODE=dry"
if /I "%MODE_RAW%"=="test" set "MODE=test"
if /I "%MODE_RAW%"=="full" set "MODE=full"

:: Validate required args
if "%QUEUE_NAME%"=="" (
    echo ERROR: queue name is required.
    goto :show_usage
)
if "%BATCH_NUMBERS%"=="" (
    echo ERROR: batch number^(s^) required.  Example: "1" or "1 2 3"
    goto :show_usage
)
if "%MODE_RAW%"=="" (
    echo ERROR: mode required: dry / test / full
    goto :show_usage
)

:: ── Print raw args and parsed values (makes mis-parses immediately visible) ──
echo.
echo [runner] Raw args:
echo   arg1 (queue):    %~1
echo   arg2 (batches):  %~2
echo   arg3 (mode):     %~3
if not "%~4"=="" echo   arg4 (max_rows): %~4
echo.
echo [runner] Parsed:
echo   queue:           %QUEUE_NAME%
echo   batch numbers:   %BATCH_NUMBERS%
echo   mode:            %MODE%
if not "%MAX_ROWS_OVERRIDE%"=="" echo   max rows:        %MAX_ROWS_OVERRIDE% ^(override^)
if     "%MAX_ROWS_OVERRIDE%"=="" if "%MODE%"=="test" echo   max rows:        %MAX_ROWS_TEST% ^(default test limit^)
echo.

:: ── Locale-safe timestamp ─────────────────────────────────────────────────────
for /f "usebackq delims=" %%T in (`powershell -NoProfile -Command "Get-Date -Format yyyyMMdd_HHmmss"`) do set "LOG_STAMP=%%T"

:: ── Create base directories ───────────────────────────────────────────────────
if not exist "%BASE_OUTPUT_DIR%" mkdir "%BASE_OUTPUT_DIR%"
if not exist "%LOG_DIR%"         mkdir "%LOG_DIR%"

:: ── Verify script exists ──────────────────────────────────────────────────────
if not exist "%SCRIPT_FILE%" (
    echo ERROR: enrich_clients_claude.py not found.
    echo        Expected: %SCRIPT_FILE%
    exit /b 1
)

:: ── Loop over each batch number ───────────────────────────────────────────────
set "QUEUE_EXIT=0"
for %%B in (%BATCH_NUMBERS%) do (
    call :run_one_batch "%%B"
    set "QUEUE_EXIT=!ERRORLEVEL!"
    if !QUEUE_EXIT! neq 0 (
        if "!STOP_ON_ERROR!"=="1" (
            echo [runner] STOP_ON_ERROR=1 — halting queue after batch %%B.
            exit /b !QUEUE_EXIT!
        )
    )
)
echo [runner] All batches done. Final exit code: %QUEUE_EXIT%
exit /b %QUEUE_EXIT%

:: ═══════════════════════════════════════════════════════════════════════════════
:run_one_batch
:: Process a single batch number.  Called with the batch number as %~1.
:: Sets _RUN_* env vars and delegates to :write_and_run_ps1.
:: ═══════════════════════════════════════════════════════════════════════════════
set "BATCH_NUM=%~1"
echo.
echo ============================================================
echo  QUEUE: %QUEUE_NAME%  BATCH: %BATCH_NUM%  MODE: %MODE%
echo ============================================================

:: Resolve input file
call :find_input "%QUEUE_NAME%" "%BATCH_NUM%"

if "!RESOLVED_INPUT!"=="" (
    echo.
    echo ERROR: No input file found for %QUEUE_NAME% batch %BATCH_NUM%.
    echo.
    echo   Searched ^(in order^):
    echo     1. %PROJECT_ROOT%\%QUEUE_NAME%\01_cleaned_domains\%QUEUE_NAME%_%BATCH_NUM%_*_cleaned_*.xlsx
    echo     2. %PROJECT_ROOT%\%QUEUE_NAME%\01_cleaned_domains\%QUEUE_NAME%_%BATCH_NUM%_*.xlsx
    echo     3. %FLAT_INPUT_DIR%\%QUEUE_NAME%_%BATCH_NUM%*_cleaned_*.xlsx
    echo     4. %FLAT_INPUT_DIR%\%QUEUE_NAME%_%BATCH_NUM%*.xlsx
    echo.
    echo   Candidate files found anywhere matching %QUEUE_NAME%*:
    powershell -NoProfile -Command "$r='%PROJECT_ROOT%'; $q='%QUEUE_NAME%'; $flat='%FLAT_INPUT_DIR%'; $found=@(); foreach($d in @(\"$r\$q\01_cleaned_domains\",$flat,$r)) { try { $found += Get-ChildItem $d -Filter \"$q*.xlsx\" -EA 0 | Select -First 5 } catch {} }; if ($found) { $found | ForEach { Write-Host \"    $_\" } } else { Write-Host '    (none found)' }"
    echo.
    exit /b 1
)

:: Resolve output dir and log file (per batch)
set "BATCH_OUTPUT_DIR=%BASE_OUTPUT_DIR%\%QUEUE_NAME%\batch_%BATCH_NUM%"
set "BATCH_LOG_FILE=%LOG_DIR%\enricher_%QUEUE_NAME%_%BATCH_NUM%_%LOG_STAMP%.log"
if not exist "!BATCH_OUTPUT_DIR!" mkdir "!BATCH_OUTPUT_DIR!"

:: ── Parsed + resolved debug block ────────────────────────────────────────────
echo [runner] Resolved input file: !RESOLVED_INPUT!
echo [runner] Output dir:          !BATCH_OUTPUT_DIR!
echo [runner] Log file:            !BATCH_LOG_FILE!
echo.

:: Write log header
echo ============================================================>>"%BATCH_LOG_FILE%"
echo  QUEUE: %QUEUE_NAME%  BATCH: %BATCH_NUM%  MODE: %MODE%>>"%BATCH_LOG_FILE%"
echo  Date/time: %LOG_STAMP%>>"%BATCH_LOG_FILE%"
echo  Input:     !RESOLVED_INPUT!>>"%BATCH_LOG_FILE%"
echo  Output:    !BATCH_OUTPUT_DIR!>>"%BATCH_LOG_FILE%"
echo ============================================================>>"%BATCH_LOG_FILE%"

:: Python version and script identity
echo  Python version:
python --version
echo.
echo  Script identity:
python -c "import pathlib; p=pathlib.Path(r'%SCRIPT_FILE%'); print('  SCRIPT_EXISTS:', p.exists()); print('  SCRIPT_MTIME: ', int(p.stat().st_mtime) if p.exists() else 'MISSING')"
echo.

:: Set _RUN_* env vars consumed by the PS1
set "PYTHONUNBUFFERED=1"
set "_RUN_SCRIPT=%SCRIPT_FILE%"
set "_RUN_INPUT=!RESOLVED_INPUT!"
set "_RUN_OUTPUT=!BATCH_OUTPUT_DIR!"
set "_RUN_LOG=!BATCH_LOG_FILE!"
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

call :write_and_run_ps1
exit /b %ERRORLEVEL%


:: ═══════════════════════════════════════════════════════════════════════════════
:find_input
:: Find the most-recently-modified input file for a given queue + batch number.
:: Sets RESOLVED_INPUT to the full path, or empty string if not found.
:: Args: %1=queue name, %2=batch number
:: ═══════════════════════════════════════════════════════════════════════════════
set "RESOLVED_INPUT="
set "_FI_QUEUE=%~1"
set "_FI_BATCH=%~2"

:: Candidates (in priority order, passed as separate env vars to avoid quoting hell)
:: 1. Pipeline cleaned dir — with _cleaned_ infix (most specific)
set "_SEARCH_DIR=%PROJECT_ROOT%\%_FI_QUEUE%\01_cleaned_domains"
set "_SEARCH_PAT=%_FI_QUEUE%_%_FI_BATCH%_*_cleaned_*.xlsx"
for /f "usebackq delims=" %%F in (`powershell -NoProfile -Command "$d=$env:_SEARCH_DIR; $p=$env:_SEARCH_PAT; $f=Get-ChildItem -Path $d -Filter $p -EA 0 | Sort-Object LastWriteTime -Descending | Select-Object -First 1; if ($f){$f.FullName}"`) do set "RESOLVED_INPUT=%%F"
if not "!RESOLVED_INPUT!"=="" exit /b 0

:: 2. Pipeline cleaned dir — any batch file (no _cleaned_ infix)
set "_SEARCH_PAT=%_FI_QUEUE%_%_FI_BATCH%_*.xlsx"
for /f "usebackq delims=" %%F in (`powershell -NoProfile -Command "$d=$env:_SEARCH_DIR; $p=$env:_SEARCH_PAT; $f=Get-ChildItem -Path $d -Filter $p -EA 0 | Sort-Object LastWriteTime -Descending | Select-Object -First 1; if ($f){$f.FullName}"`) do set "RESOLVED_INPUT=%%F"
if not "!RESOLVED_INPUT!"=="" exit /b 0

:: 3. Flat input dir — with _cleaned_ infix
set "_SEARCH_DIR=%FLAT_INPUT_DIR%"
set "_SEARCH_PAT=%_FI_QUEUE%_%_FI_BATCH%*_cleaned_*.xlsx"
for /f "usebackq delims=" %%F in (`powershell -NoProfile -Command "$d=$env:_SEARCH_DIR; $p=$env:_SEARCH_PAT; $f=Get-ChildItem -Path $d -Filter $p -EA 0 | Sort-Object LastWriteTime -Descending | Select-Object -First 1; if ($f){$f.FullName}"`) do set "RESOLVED_INPUT=%%F"
if not "!RESOLVED_INPUT!"=="" exit /b 0

:: 4. Flat input dir — any file starting with queue_batch
set "_SEARCH_PAT=%_FI_QUEUE%_%_FI_BATCH%*.xlsx"
for /f "usebackq delims=" %%F in (`powershell -NoProfile -Command "$d=$env:_SEARCH_DIR; $p=$env:_SEARCH_PAT; $f=Get-ChildItem -Path $d -Filter $p -EA 0 | Sort-Object LastWriteTime -Descending | Select-Object -First 1; if ($f){$f.FullName}"`) do set "RESOLVED_INPUT=%%F"
exit /b 0


:: ═══════════════════════════════════════════════════════════════════════════════
:write_and_run_ps1
:: Write a temp .ps1 from _RUN_* env vars and execute it with powershell -File.
:: Avoids all CMD/PS quote-nesting issues for paths with spaces.
:: ═══════════════════════════════════════════════════════════════════════════════
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
echo # --- debug: paths only, keys excluded --->>"%TEMP_PS1%"
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

powershell -NoProfile -ExecutionPolicy Bypass -File "%TEMP_PS1%"
set "PS1_EXIT=%ERRORLEVEL%"
del "%TEMP_PS1%" 2>nul

echo.
if "%PS1_EXIT%"=="0" (
    echo OK: batch completed successfully.
    echo OK: batch completed successfully.>>"%BATCH_LOG_FILE%"
) else (
    echo ERROR: Python exited with code %PS1_EXIT%.
    echo ERROR: Python exited with code %PS1_EXIT%.>>"%BATCH_LOG_FILE%"
)
echo Log saved: %BATCH_LOG_FILE%
exit /b %PS1_EXIT%


:: ═══════════════════════════════════════════════════════════════════════════════
:legacy_mode
:: Original single-file mode: run_enricher_queue_generic.bat [dry|test|full] [max_rows]
:: Input file: input\batch_input.xlsx  (edit CONFIG or drop a file there)
:: ═══════════════════════════════════════════════════════════════════════════════
set "QUEUE_NAME=legacy"
set "BATCH_NUM=0"
set "MODE=full"
if /I "%~1"=="dry"  set "MODE=dry"
if /I "%~1"=="test" set "MODE=test"
if /I "%~1"=="full" set "MODE=full"
set "MAX_ROWS_OVERRIDE=%~2"

set "RESOLVED_INPUT=%~dp0input\batch_input.xlsx"
set "BATCH_OUTPUT_DIR=%BASE_OUTPUT_DIR%"
set "BATCH_LOG_FILE=%LOG_DIR%\enricher_legacy_%LOG_STAMP%.log"

for /f "usebackq delims=" %%T in (`powershell -NoProfile -Command "Get-Date -Format yyyyMMdd_HHmmss"`) do set "LOG_STAMP=%%T"
set "BATCH_LOG_FILE=%LOG_DIR%\enricher_legacy_%LOG_STAMP%.log"

if not exist "%BASE_OUTPUT_DIR%" mkdir "%BASE_OUTPUT_DIR%"
if not exist "%LOG_DIR%"         mkdir "%LOG_DIR%"
if not exist "%SCRIPT_FILE%" (
    echo ERROR: enrich_clients_claude.py not found: %SCRIPT_FILE%
    exit /b 1
)
if not exist "%RESOLVED_INPUT%" (
    echo ERROR: legacy input file not found: %RESOLVED_INPUT%
    echo        Place your input file at  input\batch_input.xlsx  or use:
    echo        run_enricher_queue_generic.bat ^<queue^> "^<batches^>" ^<mode^>
    exit /b 1
)

echo [runner] Legacy mode: %MODE%, input: %RESOLVED_INPUT%

set "PYTHONUNBUFFERED=1"
set "_RUN_SCRIPT=%SCRIPT_FILE%"
set "_RUN_INPUT=%RESOLVED_INPUT%"
set "_RUN_OUTPUT=%BATCH_OUTPUT_DIR%"
set "_RUN_LOG=%BATCH_LOG_FILE%"
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
call :write_and_run_ps1
exit /b %ERRORLEVEL%


:: ═══════════════════════════════════════════════════════════════════════════════
:show_usage
:: ═══════════════════════════════════════════════════════════════════════════════
echo.
echo  USAGE:
echo    run_enricher_queue_generic.bat ^<queue^> "^<batches^>" ^<mode^> [max_rows]
echo.
echo  QUEUE:    Italy100 ^| Italy200 ^| Germany ^| 1 ^| 2
echo  BATCHES:  "1" ^| "1 2 3" ^| "11 12 13"   (in quotes^)
echo  MODE:     dry  = path check only, no API calls
echo            test = first %MAX_ROWS_TEST% rows only
echo            full = all rows
echo.
echo  EXAMPLES:
echo    run_enricher_queue_generic.bat Italy100 "1" dry
echo    run_enricher_queue_generic.bat Italy100 "1" test
echo    run_enricher_queue_generic.bat Italy100 "1 2 3" full
echo    run_enricher_queue_generic.bat 1 "1" test
echo.
echo  INPUT FILE DISCOVERY:
echo    Looks for cleaned output files in:
echo      {PROJECT_ROOT}\{Queue}\01_cleaned_domains\{Queue}_{N}_*_cleaned_*.xlsx
echo    PROJECT_ROOT defaults to: %PROJECT_ROOT%
echo    Edit CONFIG section in this bat to override PROJECT_ROOT.
echo.
exit /b 1
