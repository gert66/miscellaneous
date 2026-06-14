@echo off
:: ============================================================
:: run_enricher_queue_generic_v3.bat
:: Compatibility wrapper — forwards all arguments to the real runner.
::
:: Usage:
::   run_enricher_queue_generic_v3.bat [dry|test|full] [max_rows]
::
:: This wrapper exists so existing shortcuts, scripts, or documentation
:: that reference v3 continue to work after the runner was consolidated
:: into run_enricher_queue_generic.bat.
:: ============================================================
call "%~dp0run_enricher_queue_generic.bat" %*
exit /b %ERRORLEVEL%
