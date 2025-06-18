@echo off
echo Starting RivalAI Server (Safe Mode)
echo ===================================

REM Activate virtual environment
call venvEngine\Scripts\activate.bat

REM Set environment variables for better CUDA handling
set CUDA_VISIBLE_DEVICES=0
set PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

REM Add PyTorch library path to PATH
set PATH=venvEngine\Lib\site-packages\torch\lib;%PATH%

echo Environment setup complete.
echo.

REM Check if checkpoint file exists
if not exist "checkpoints\rival_ai\checkpoint_20250615_042203_epoch_159.pt" (
    echo WARNING: Checkpoint file not found!
    echo The server will run without neural network model.
    echo.
)

REM Run the server
echo Starting server...
echo If you encounter CUDA errors, the server will automatically fall back to CPU mode.
echo.
cargo run --bin server

REM If the server crashes, show helpful message
if errorlevel 1 (
    echo.
    echo Server crashed or encountered an error.
    echo.
    echo Troubleshooting steps:
    echo 1. Run the diagnostic script: python fix_cuda_windows.py
    echo 2. Try running without checkpoint: cargo run --bin server
    echo 3. Check if all dependencies are installed
    echo.
    pause
) 