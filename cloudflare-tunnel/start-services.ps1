# Start RivalAI Services
# Run this script to start both frontend and backend

Write-Host "Starting RivalAI Services..." -ForegroundColor Green

# Function to start a service in a new PowerShell window
function Start-ServiceInNewWindow {
    param(
        [string]$Title,
        [string]$Command,
        [string]$WorkingDirectory
    )
    
    Write-Host "Starting $Title..." -ForegroundColor Yellow
    Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$WorkingDirectory'; $Command" -WindowStyle Normal
}

# Get the script directory (now in cloudflare-tunnel subdirectory)
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectRoot = Split-Path -Parent $scriptDir

# Start Backend (Rust server)
$backendDir = Join-Path $projectRoot "engine"
Start-ServiceInNewWindow -Title "RivalAI Backend" -Command "cargo run --bin server" -WorkingDirectory $backendDir

# Wait a moment for backend to start
Start-Sleep -Seconds 3

# Start Frontend (React/Vite)
$frontendDir = Join-Path $projectRoot "engine\web"
Start-ServiceInNewWindow -Title "RivalAI Frontend" -Command "npm run dev" -WorkingDirectory $frontendDir

Write-Host ""
Write-Host "Services are starting..." -ForegroundColor Green
Write-Host "Backend (Rust): http://localhost:3000" -ForegroundColor Cyan
Write-Host "Frontend (React): http://localhost:5173" -ForegroundColor Cyan
Write-Host ""
Write-Host "Once both services are running, you can start the Cloudflare tunnel." -ForegroundColor Yellow
Write-Host "Press any key to exit..." -ForegroundColor Gray
Read-Host 