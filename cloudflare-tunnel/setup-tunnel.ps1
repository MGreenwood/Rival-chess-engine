# Setup Cloudflare Tunnel for RivalAI
# Run this after you have:
# 1. A domain added to Cloudflare
# 2. Cloudflared installed
# 3. Your services running

Write-Host "=== Cloudflare Tunnel Setup for RivalAI ===" -ForegroundColor Green
Write-Host ""

# Check if cloudflared is installed
try {
    $version = cloudflared --version
    Write-Host "✓ Cloudflared is installed: $version" -ForegroundColor Green
} catch {
    Write-Host "✗ Cloudflared is not installed or not in PATH" -ForegroundColor Red
    Write-Host "Please run install-cloudflared.ps1 first" -ForegroundColor Yellow
    exit 1
}

Write-Host ""

# Get domain from user
$domain = Read-Host "Enter your domain name (e.g., yoursite.com)"
if ([string]::IsNullOrWhiteSpace($domain)) {
    Write-Host "Domain is required!" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "Step 1: Login to Cloudflare" -ForegroundColor Yellow
Write-Host "This will open a browser window for authentication..."
cloudflared tunnel login

Write-Host ""
Write-Host "Step 2: Create a tunnel" -ForegroundColor Yellow
$tunnelName = "rivalai-$($env:USERNAME)"
Write-Host "Creating tunnel: $tunnelName"
cloudflared tunnel create $tunnelName

Write-Host ""
Write-Host "Step 3: Get tunnel information" -ForegroundColor Yellow
$tunnelList = cloudflared tunnel list
Write-Host $tunnelList

# Extract tunnel ID (this is a simple approach, might need adjustment)
$tunnelId = (cloudflared tunnel list | Select-String $tunnelName | ForEach-Object { ($_ -split '\s+')[0] })

if ([string]::IsNullOrWhiteSpace($tunnelId)) {
    Write-Host "Could not automatically detect tunnel ID." -ForegroundColor Red
    $tunnelId = Read-Host "Please enter your tunnel ID manually"
}

Write-Host "Using tunnel ID: $tunnelId" -ForegroundColor Cyan

Write-Host ""
Write-Host "Step 4: Create DNS record" -ForegroundColor Yellow
Write-Host "Creating CNAME record for $domain..."
cloudflared tunnel route dns $tunnelId $domain

Write-Host ""
Write-Host "Step 5: Update tunnel configuration" -ForegroundColor Yellow

# Update the config file with actual values
$configContent = Get-Content "tunnel-config.yml" -Raw
$configContent = $configContent -replace "YOUR_TUNNEL_ID_HERE", $tunnelId
$configContent = $configContent -replace "YOUR_DOMAIN_HERE", $domain

# Find credentials file
$credentialsPath = "$env:USERPROFILE\.cloudflared\$tunnelId.json"
if (Test-Path $credentialsPath) {
    $configContent = $configContent -replace "YOUR_CREDENTIALS_FILE_PATH_HERE", $credentialsPath
} else {
    Write-Host "Warning: Could not find credentials file at $credentialsPath" -ForegroundColor Yellow
    $credentialsPath = Read-Host "Please enter the full path to your credentials file"
    $configContent = $configContent -replace "YOUR_CREDENTIALS_FILE_PATH_HERE", $credentialsPath
}

# Save updated config
$configContent | Out-File "tunnel-config.yml" -Encoding UTF8

Write-Host "✓ Configuration updated!" -ForegroundColor Green

Write-Host ""
Write-Host "Step 6: Test the tunnel" -ForegroundColor Yellow
Write-Host "Make sure your services are running first!" -ForegroundColor Red
Write-Host "Backend should be on http://localhost:3000" -ForegroundColor Cyan
Write-Host "Frontend should be on http://localhost:5173" -ForegroundColor Cyan
Write-Host ""

$startTunnel = Read-Host "Start the tunnel now? (y/n)"
if ($startTunnel -eq 'y' -or $startTunnel -eq 'Y') {
    Write-Host ""
    Write-Host "Starting tunnel..." -ForegroundColor Green
    Write-Host "Your site will be available at: https://$domain" -ForegroundColor Green
    Write-Host ""
    Write-Host "Press Ctrl+C to stop the tunnel" -ForegroundColor Yellow
    cloudflared tunnel --config tunnel-config.yml run $tunnelId
} else {
    Write-Host ""
    Write-Host "To start the tunnel later, run:" -ForegroundColor Yellow
    Write-Host "cloudflared tunnel --config tunnel-config.yml run $tunnelId" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Your site will be available at: https://$domain" -ForegroundColor Green
} 