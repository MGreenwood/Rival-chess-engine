Write-Host "Starting RivalAI Server..." -ForegroundColor Green

# Activate the Python virtual environment from the python directory
Push-Location ..\python
.\venv\Scripts\Activate.ps1
Pop-Location

# Set Python path to include the engine's venv and Python source
$env:PYTHONPATH = "$PWD\venvEngine\Lib\site-packages;$PWD\..\python\src;$env:PYTHONPATH"

Write-Host "Python environment set up:" -ForegroundColor Yellow
Write-Host "  PYTHONPATH: $env:PYTHONPATH" -ForegroundColor Gray

# Run the server
Write-Host "Building and running server..." -ForegroundColor Green
cargo run --bin server -- --games-dir ../python/training_games

# Deactivate when done
deactivate 