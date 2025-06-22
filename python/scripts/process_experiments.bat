@echo off
REM Process and Compress Experiment PKL Files

echo ========================================
echo RivalAI Experiment PKL File Processor
echo ========================================
echo.

REM Change to the RivalAI root directory
cd /d "%~dp0..\.."

echo Current directory: %CD%
echo.

if "%1"=="--dry-run" (
    echo === DRY RUN MODE ===
    echo This will show what files would be processed without actually processing them.
    echo.
    python python/scripts/process_and_compress_experiments.py --dry-run
    goto end
)

if "%1"=="--help" (
    echo Usage:
    echo   process_experiments.bat              - Process all pkl files
    echo   process_experiments.bat --dry-run    - Show what would be processed
    echo   process_experiments.bat --fast       - Use more workers for faster processing
    echo   process_experiments.bat --cleanup    - Process and cleanup original files
    echo   process_experiments.bat --help       - Show this help
    echo.
    echo The script will:
    echo   1. Find all .pkl files in python/experiments
    echo   2. Convert them to JSON training format
    echo   3. Compress them with gzip
    echo   4. Store them in python/compressed_experiments
    echo   5. Optionally cleanup original files
    goto end
)

if "%1"=="--fast" (
    echo === FAST MODE - Using 8 workers ===
    python python/scripts/process_and_compress_experiments.py --max-workers 8
    goto end
)

if "%1"=="--cleanup" (
    echo === PROCESSING WITH CLEANUP ===
    echo WARNING: This will DELETE original pkl files after processing!
    echo Press Ctrl+C now if you want to cancel...
    timeout /t 10
    python python/scripts/process_and_compress_experiments.py --cleanup --confirm-cleanup
    goto end
)

REM Default processing
echo === STANDARD PROCESSING ===
echo Processing all pkl files with compression...
echo Original files will be preserved.
echo.
python python/scripts/process_and_compress_experiments.py

:end
echo.
echo Processing complete!
pause 