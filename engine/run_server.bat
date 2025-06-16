@echo off
echo Starting RivalAI Server...

REM Activate the virtual environment
call venvEngine\Scripts\activate.bat

REM Set Python path to include the engine's venv
set PYTHONPATH=%CD%\venvEngine\Lib\site-packages;%CD%\..\python\src;%PYTHONPATH%

REM Run the server
cargo run --bin server

REM Deactivate when done
call venvEngine\Scripts\deactivate.bat 