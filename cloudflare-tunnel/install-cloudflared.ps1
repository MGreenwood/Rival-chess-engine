# Install Cloudflared on Windows
# Run this script as Administrator

Write-Host "Installing Cloudflared..." -ForegroundColor Green

# Download the latest cloudflared for Windows
$url = "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-windows-amd64.exe"
$output = "$env:ProgramFiles\Cloudflare\cloudflared.exe"

# Create directory if it doesn't exist
$dir = Split-Path $output -Parent
if (!(Test-Path $dir)) {
    New-Item -ItemType Directory -Path $dir -Force
}

# Download cloudflared
Write-Host "Downloading cloudflared..." -ForegroundColor Yellow
Invoke-WebRequest -Uri $url -OutFile $output

# Add to PATH if not already there
$currentPath = [Environment]::GetEnvironmentVariable("PATH", "Machine")
$cloudflareDir = Split-Path $output -Parent

if ($currentPath -notlike "*$cloudflareDir*") {
    Write-Host "Adding Cloudflared to PATH..." -ForegroundColor Yellow
    [Environment]::SetEnvironmentVariable("PATH", "$currentPath;$cloudflareDir", "Machine")
    $env:PATH = "$env:PATH;$cloudflareDir"
}

Write-Host "Cloudflared installed successfully!" -ForegroundColor Green
Write-Host "You may need to restart your terminal to use 'cloudflared' command." -ForegroundColor Yellow

# Test installation
& $output --version 