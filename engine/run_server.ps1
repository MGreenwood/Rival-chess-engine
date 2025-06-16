Write-Host "Starting RivalAI Server..." -ForegroundColor Green

# Activate the virtual environment
& ".\venvEngine\Scripts\Activate.ps1"

# Set Python path to include the engine's venv and Python source
$env:PYTHONPATH = "$PWD\venvEngine\Lib\site-packages;$PWD\..\python\src;$env:PYTHONPATH"

Write-Host "Python environment set up:" -ForegroundColor Yellow
Write-Host "  PYTHONPATH: $env:PYTHONPATH" -ForegroundColor Gray

# Run the server
Write-Host "Building and running server..." -ForegroundColor Green
cargo run --bin server

# Deactivate when done
deactivate 