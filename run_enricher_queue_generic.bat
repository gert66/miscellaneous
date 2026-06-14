@echo off
setlocal EnableDelayedExpansion

:: ============================================================
:: run_enricher_queue_generic.bat
:: mYngle Lead Prioritizer -- queue-aware batch runner v6
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
:: PROJECT_ROOT resolution (in priority order):
::   1. MYNGLE_DATA_ROOT environment variable (if set)
::   2. Parent of this repo folder  (%~dp0..)
::   3. This repo folder            (%~dp0)
::   4. %~dp0data subfolder
::   5. %~dp0input subfolder
::   The first candidate that contains any .xlsx file wins.
::
:: Input file discovery for queue Italy100 batch 1:
::   Searches these patterns (underscore-bounded, safe against batch 10/11):
::     {root}\Italy100\01_cleaned_domains\Italy100_1_*_cleaned_*.xlsx
::     {root}\Italy100\01_cleaned_domains\Italy100_1_*.xlsx
::     {root}\Italy100\01_cleaned_domains\Italy100_01_*.xlsx
::     {root}\Italy100\Italy100_1_*.xlsx
::     {root}\Italy100\Italy100_01_*.xlsx
::     {root}\Italy100\Italy100_batch_1_*.xlsx
::     {root}\Italy100\Italy100*R0001*.xlsx
::     {bat_dir}\input\Italy100_1_*.xlsx
::     {bat_dir}\input\Italy100_01_*.xlsx
::     {bat_dir}\Italy100_1_*.xlsx
::   Multiple matches with same filename stem -> most recent wins.
::   Multiple different stems -> prints all candidates and stops.
::
:: Output goes to: {BAT_DIR}\output\{Queue}\batch_{N}\
:: Logs go to:     {BAT_DIR}\logs\enricher_{Queue}_{N}_{timestamp}.log
::
:: Implementation: writes a temp .ps1 per batch and runs it with
::   powershell -File  to avoid all CMD/PS quote-nesting issues.
::   Tee-Object streams live Python output to console + log.
:: ============================================================

:: -- CONFIG -----------------------------------------------------------

:: Path to enrich_clients_claude.py
set "SCRIPT_FILE=%~dp0enrich_clients_claude.py"

:: BAT_DIR = folder containing this bat file (no trailing backslash)
set "BAT_DIR=%~dp0"
if "!BAT_DIR:~-1!"=="\" set "BAT_DIR=!BAT_DIR:~0,-1!"

:: PROJECT_ROOT: resolved below after parsing args (needs queue name for heuristic)
:: Preset to the default; will be overridden if MYNGLE_DATA_ROOT is set.
set "PROJECT_ROOT=%~dp0.."

:: Flat input dir fallback
set "FLAT_INPUT_DIR=%~dp0input"

:: Base output and log dirs (relative to this bat file)
set "BASE_OUTPUT_DIR=%~dp0output"
set "LOG_DIR=%~dp0logs"

:: Rows to process in test mode (overridable with 4th arg)
set "MAX_ROWS_TEST=5"

:: Halt entire queue run on first non-zero Python exit (1=yes, 0=no)
set "STOP_ON_ERROR=1"

:: API keys -- leave empty to load from .streamlit/secrets.toml or environment
set "ANTHROPIC_KEY="
set "SERPER_KEY="

:: -- END CONFIG -------------------------------------------------------

:: -- Show usage if no args --------------------------------------------
if "%~1"=="" goto :show_usage

:: -- Detect legacy single-arg format: run_enricher_queue_generic.bat dry|test|full --
set "_ARG1=%~1"
if /I "!_ARG1!"=="dry"  goto :legacy_mode
if /I "!_ARG1!"=="test" goto :legacy_mode
if /I "!_ARG1!"=="full" goto :legacy_mode

:: -- New queue-based argument parsing ---------------------------------
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

:: -- Resolve PROJECT_ROOT (MYNGLE_DATA_ROOT override or auto-detect) --
call :resolve_project_root

:: -- Print raw args and parsed values (makes mis-parses immediately visible) --
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
echo [runner] PROJECT_ROOT:   %PROJECT_ROOT%
echo [runner] FLAT_INPUT_DIR: %FLAT_INPUT_DIR%
echo [runner] BAT_DIR:        %BAT_DIR%
echo.

:: -- Locale-safe timestamp --------------------------------------------
for /f "usebackq delims=" %%T in (`powershell -NoProfile -Command "Get-Date -Format yyyyMMdd_HHmmss"`) do set "LOG_STAMP=%%T"

:: -- Create base directories ------------------------------------------
if not exist "%BASE_OUTPUT_DIR%" mkdir "%BASE_OUTPUT_DIR%"
if not exist "%LOG_DIR%"         mkdir "%LOG_DIR%"

:: -- Verify script exists ---------------------------------------------
if not exist "%SCRIPT_FILE%" (
    echo ERROR: enrich_clients_claude.py not found.
    echo        Expected: %SCRIPT_FILE%
    exit /b 1
)

:: -- Loop over each batch number --------------------------------------
set "QUEUE_EXIT=0"
for %%B in (%BATCH_NUMBERS%) do (
    call :run_one_batch "%%B"
    set "QUEUE_EXIT=!ERRORLEVEL!"
    if !QUEUE_EXIT! neq 0 (
        if "!STOP_ON_ERROR!"=="1" (
            echo [runner] STOP_ON_ERROR=1 -- halting queue after batch %%B.
            exit /b !QUEUE_EXIT!
        )
    )
)
echo [runner] All batches done. Final exit code: %QUEUE_EXIT%
exit /b %QUEUE_EXIT%


:: =======================================================================
:resolve_project_root
:: Resolve PROJECT_ROOT from MYNGLE_DATA_ROOT env var or auto-detect.
:: Sets PROJECT_ROOT and prints which source was used.
:: =======================================================================
if defined MYNGLE_DATA_ROOT (
    set "PROJECT_ROOT=%MYNGLE_DATA_ROOT%"
    echo [runner] PROJECT_ROOT: using MYNGLE_DATA_ROOT=%MYNGLE_DATA_ROOT%
    exit /b 0
)

:: Auto-detect: check each candidate; first one that contains a .xlsx file wins.
set "_PR_FOUND="
call :_try_project_root "%~dp0.."
if not defined _PR_FOUND call :_try_project_root "%~dp0"
if not defined _PR_FOUND call :_try_project_root "%~dp0data"
if not defined _PR_FOUND call :_try_project_root "%~dp0input"
if defined _PR_FOUND (
    echo [runner] PROJECT_ROOT: auto-detected as %PROJECT_ROOT%
) else (
    set "PROJECT_ROOT=%~dp0.."
    echo [runner] PROJECT_ROOT: no .xlsx files found in any candidate -- defaulting to %PROJECT_ROOT%
)
exit /b 0

:_try_project_root
if defined _PR_FOUND exit /b 0
set "_PR_CAND=%~1"
for /f "usebackq delims=" %%X in (`powershell -NoProfile -Command "if (Get-ChildItem '%_PR_CAND%' -Filter '*.xlsx' -Recurse -EA 0 | Select-Object -First 1) { 'yes' }" 2^>nul`) do (
    if "%%X"=="yes" (
        set "PROJECT_ROOT=%_PR_CAND%"
        set "_PR_FOUND=1"
    )
)
exit /b 0


:: =======================================================================
:run_one_batch
:: Process a single batch number.  Called with the batch number as %~1.
:: =======================================================================
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
    echo   Diagnostics:
    echo     Current dir:      !CD!
    echo     BAT dir:          %BAT_DIR%
    echo     PROJECT_ROOT:     %PROJECT_ROOT%
    echo     Queue:            %QUEUE_NAME%
    echo     Batch:            %BATCH_NUM%
    echo.
    echo   Patterns searched ^(in order^):
    echo     1.  %PROJECT_ROOT%\%QUEUE_NAME%\01_cleaned_domains\%QUEUE_NAME%_%BATCH_NUM%_*_cleaned_*.xlsx
    echo     2.  %PROJECT_ROOT%\%QUEUE_NAME%\01_cleaned_domains\%QUEUE_NAME%_%BATCH_NUM%_*.xlsx
    echo     3.  %PROJECT_ROOT%\%QUEUE_NAME%\01_cleaned_domains\%QUEUE_NAME%_0%BATCH_NUM%_*.xlsx
    echo     4.  %PROJECT_ROOT%\%QUEUE_NAME%\%QUEUE_NAME%_%BATCH_NUM%_*.xlsx
    echo     5.  %PROJECT_ROOT%\%QUEUE_NAME%\%QUEUE_NAME%_0%BATCH_NUM%_*.xlsx
    echo     6.  %PROJECT_ROOT%\%QUEUE_NAME%\%QUEUE_NAME%_batch_%BATCH_NUM%_*.xlsx
    echo     7.  %PROJECT_ROOT%\%QUEUE_NAME%\%QUEUE_NAME%*R[zero-padded batch]*.xlsx
    echo     8.  %FLAT_INPUT_DIR%\%QUEUE_NAME%_%BATCH_NUM%_*.xlsx
    echo     9.  %FLAT_INPUT_DIR%\%QUEUE_NAME%_0%BATCH_NUM%_*.xlsx
    echo     10. %BAT_DIR%\%QUEUE_NAME%_%BATCH_NUM%_*.xlsx
    echo.
    echo   .xlsx files found under PROJECT_ROOT ^(max 50^):
    powershell -NoProfile -Command "$r='%PROJECT_ROOT%'; $hits = Get-ChildItem $r -Filter '*.xlsx' -Recurse -EA 0 | Select-Object -First 50; if ($hits) { $hits | ForEach { Write-Host \"    $_\" } } else { Write-Host '    (none found)' }"
    echo.
    echo   .xlsx files found under BAT_DIR ^(max 50^):
    powershell -NoProfile -Command "$r='%BAT_DIR%'; $hits = Get-ChildItem $r -Filter '*.xlsx' -Recurse -EA 0 | Select-Object -First 50; if ($hits) { $hits | ForEach { Write-Host \"    $_\" } } else { Write-Host '    (none found)' }"
    echo.
    echo   Files matching queue name '%QUEUE_NAME%' anywhere under PROJECT_ROOT ^(max 50^):
    powershell -NoProfile -Command "$r='%PROJECT_ROOT%'; $q='%QUEUE_NAME%'; $hits = Get-ChildItem $r -Recurse -EA 0 | Where-Object { $_.Name -like \"*$q*\" } | Select-Object -First 50; if ($hits) { $hits | ForEach { Write-Host \"    $_\" } } else { Write-Host '    (none found)' }"
    echo.
    exit /b 1
)

:: Resolve output dir and log file (per batch)
set "BATCH_OUTPUT_DIR=%BASE_OUTPUT_DIR%\%QUEUE_NAME%\batch_%BATCH_NUM%"
set "BATCH_LOG_FILE=%LOG_DIR%\enricher_%QUEUE_NAME%_%BATCH_NUM%_%LOG_STAMP%.log"
if not exist "!BATCH_OUTPUT_DIR!" mkdir "!BATCH_OUTPUT_DIR!"

:: -- Parsed + resolved debug block ------------------------------------
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


:: =======================================================================
:find_input
:: Find the most-recently-modified input file for a given queue + batch.
:: Uses PowerShell with regex matching so batch 1 never matches batch 10/11.
:: Sets RESOLVED_INPUT to the full path, or empty string if not found.
:: Args: %1=queue name, %2=batch number
:: =======================================================================
set "RESOLVED_INPUT="
set "_FI_QUEUE=%~1"
set "_FI_BATCH=%~2"

:: Build zero-padded batch variants: 1->01, 1->0001  (used in R-number patterns)
for /f "usebackq delims=" %%P in (`powershell -NoProfile -Command "$b=[int]'%_FI_BATCH%'; '{0:D2}' -f $b"`) do set "_FI_BATCH_2D=%%P"
for /f "usebackq delims=" %%P in (`powershell -NoProfile -Command "$b=[int]'%_FI_BATCH%'; 'R{0:D4}' -f $b"`) do set "_FI_RNUM=%%P"

:: All search roots (in priority order)
set "_FI_ROOT1=%PROJECT_ROOT%\%_FI_QUEUE%\01_cleaned_domains"
set "_FI_ROOT2=%PROJECT_ROOT%\%_FI_QUEUE%"
set "_FI_ROOT3=%PROJECT_ROOT%"
set "_FI_ROOT4=%FLAT_INPUT_DIR%"
set "_FI_ROOT5=%BAT_DIR%"

:: Patterns (all underscore-bounded so batch 1 does not match batch 10/11)
:: Pattern group A: exact batch number with underscore separator
::   {Q}_{N}_*.xlsx   -- e.g. Italy100_1_...xlsx
::   {Q}_0{N}_*.xlsx  -- zero-padded   Italy100_01_...xlsx
:: Pattern group B: "batch" keyword
::   {Q}_batch_{N}_*.xlsx
:: Pattern group C: R-number (range-start notation)
::   {Q}*{R0001}*.xlsx  -- e.g. Italy100_R0001_0500_cleaned.xlsx

:: Write a temp PS1 so we avoid all CMD/PS quoting issues with paths containing spaces.
set "_FIND_PS1=%TEMP%\enricher_find_%RANDOM%.ps1"
echo $q  = $env:_FI_QUEUE>>"%_FIND_PS1%"
echo $n  = $env:_FI_BATCH>>"%_FIND_PS1%"
echo $rn = $env:_FI_RNUM>>"%_FIND_PS1%"
echo $roots = @($env:_FI_ROOT1,$env:_FI_ROOT2,$env:_FI_ROOT3,$env:_FI_ROOT4,$env:_FI_ROOT5)>>"%_FIND_PS1%"
echo $regex  = '^' + [regex]::Escape($q) + '[_-]0*' + [regex]::Escape($n) + '[_.]'>>"%_FIND_PS1%"
echo $rregex = [regex]::Escape($rn)>>"%_FIND_PS1%"
echo $cands = @()>>"%_FIND_PS1%"
echo foreach ($root in $roots) {>>"%_FIND_PS1%"
echo   if (-not $root -or -not (Test-Path $root)) { continue }>>"%_FIND_PS1%"
echo   $files = Get-ChildItem $root -Filter '*.xlsx' -EA 0>>"%_FIND_PS1%"
echo   foreach ($f in $files) {>>"%_FIND_PS1%"
echo     if ($f.Name -match $regex -or $f.Name -match $rregex) { $cands += $f }>>"%_FIND_PS1%"
echo   }>>"%_FIND_PS1%"
echo }>>"%_FIND_PS1%"
echo if ($cands.Count -eq 0) { exit 0 }>>"%_FIND_PS1%"
echo $unique = $cands ^| Sort-Object FullName -Unique>>"%_FIND_PS1%"
echo if ($unique.Count -eq 1) { Write-Output $unique[0].FullName; exit 0 }>>"%_FIND_PS1%"
echo $stems = $unique ^| ForEach { $_.BaseName -replace '_[0-9]{8}.*$','' } ^| Sort-Object -Unique>>"%_FIND_PS1%"
echo if ($stems.Count -eq 1) {>>"%_FIND_PS1%"
echo   $best = $unique ^| Sort-Object LastWriteTime -Descending ^| Select-Object -First 1>>"%_FIND_PS1%"
echo   Write-Output $best.FullName; exit 0>>"%_FIND_PS1%"
echo }>>"%_FIND_PS1%"
echo Write-Host '[runner] WARNING: Multiple distinct input candidates -- cannot auto-select:'>>"%_FIND_PS1%"
echo foreach ($f in $unique) { Write-Host "  $($f.FullName)  [modified $($f.LastWriteTime)]" }>>"%_FIND_PS1%"
echo Write-Host '[runner] Set MYNGLE_DATA_ROOT or move files to disambiguate.'>>"%_FIND_PS1%"
echo exit 2>>"%_FIND_PS1%"

for /f "usebackq delims=" %%F in (`powershell -NoProfile -ExecutionPolicy Bypass -File "%_FIND_PS1%"`) do (
    set "RESOLVED_INPUT=%%F"
)
del "%_FIND_PS1%" 2>nul
exit /b 0


:: =======================================================================
:write_and_run_ps1
:: Write a temp .ps1 from _RUN_* env vars and execute with powershell -File.
:: Avoids all CMD/PS quote-nesting issues for paths with spaces.
:: =======================================================================
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


:: =======================================================================
:legacy_mode
:: Original single-file mode: run_enricher_queue_generic.bat [dry|test|full] [max_rows]
:: Input file: input\batch_input.xlsx  (edit CONFIG or drop a file there)
:: =======================================================================
set "QUEUE_NAME=legacy"
set "BATCH_NUM=0"
set "MODE=full"
if /I "%~1"=="dry"  set "MODE=dry"
if /I "%~1"=="test" set "MODE=test"
if /I "%~1"=="full" set "MODE=full"
set "MAX_ROWS_OVERRIDE=%~2"

set "RESOLVED_INPUT=%~dp0input\batch_input.xlsx"
set "BATCH_OUTPUT_DIR=%BASE_OUTPUT_DIR%"

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


:: =======================================================================
:show_usage
:: =======================================================================
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
echo    Looks for cleaned output files under PROJECT_ROOT.
echo    Set MYNGLE_DATA_ROOT environment variable to override PROJECT_ROOT.
echo    Otherwise PROJECT_ROOT is auto-detected as the first ancestor folder
echo    that contains any .xlsx file (default: parent of this repo folder).
echo.
exit /b 1
