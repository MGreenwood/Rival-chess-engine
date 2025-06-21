Write-Host "Starting RivalAI Server..." -ForegroundColor Green

# Change to the engine directory
Set-Location $PSScriptRoot

# Activate the Python virtual environment
.\venvEngine\Scripts\Activate.ps1

# Set Python path to include the engine's venv and Python source
$env:PYTHONPATH = "$PWD\venvEngine\Lib\site-packages;$PWD\..\python\src;$env:PYTHONPATH"

Write-Host "Python environment set up:" -ForegroundColor Yellow
Write-Host "  PYTHONPATH: $env:PYTHONPATH" -ForegroundColor Gray

# Run the server with the specified model path
cargo run --bin server -- --model-path "..\python\experiments\rival_ai_v1_Alice\run_20250618_134810\checkpoints\checkpoint_epoch_5.pt"

# Change back to the original directory
Set-Location ..

# Deactivate when done
deactivate 