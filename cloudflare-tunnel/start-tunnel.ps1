# Start Cloudflare Tunnel for RivalAI
# Run this script to start your configured tunnel

Write-Host "Starting RivalAI Cloudflare Tunnel..." -ForegroundColor Green

# Check if config file exists
if (!(Test-Path "tunnel-config.yml")) {
    Write-Host "❌ tunnel-config.yml not found!" -ForegroundColor Red
    Write-Host "Please run setup-tunnel.ps1 first to configure your tunnel." -ForegroundColor Yellow
    exit 1
}

# Read the tunnel ID from config
try {
    $configContent = Get-Content "tunnel-config.yml" -Raw
    $tunnelId = ($configContent | Select-String "tunnel: (.+)" | ForEach-Object { $_.Matches[0].Groups[1].Value }).Trim()
    
    if ([string]::IsNullOrWhiteSpace($tunnelId)) {
        throw "Tunnel ID not found in config"
    }
    
    Write-Host "✓ Found tunnel ID: $tunnelId" -ForegroundColor Green
} catch {
    Write-Host "❌ Could not read tunnel configuration!" -ForegroundColor Red
    Write-Host "Please run setup-tunnel.ps1 to reconfigure your tunnel." -ForegroundColor Yellow
    exit 1
}

# Check if services are running
Write-Host ""
Write-Host "Checking if your services are running..." -ForegroundColor Yellow

try {
    $frontendCheck = Test-NetConnection -ComputerName localhost -Port 5173 -InformationLevel Quiet -WarningAction SilentlyContinue
    $backendCheck = Test-NetConnection -ComputerName localhost -Port 3000 -InformationLevel Quiet -WarningAction SilentlyContinue
    
    if ($frontendCheck) {
        Write-Host "✓ Frontend service (port 5173) is running" -ForegroundColor Green
    } else {
        Write-Host "⚠️  Frontend service (port 5173) is not running" -ForegroundColor Yellow
    }
    
    if ($backendCheck) {
        Write-Host "✓ Backend service (port 3000) is running" -ForegroundColor Green
    } else {
        Write-Host "⚠️  Backend service (port 3000) is not running" -ForegroundColor Yellow
    }
    
    if (!$frontendCheck -or !$backendCheck) {
        Write-Host ""
        Write-Host "Warning: Some services are not running!" -ForegroundColor Red
        Write-Host "Run start-services.ps1 first to start your RivalAI services." -ForegroundColor Yellow
        
        $continue = Read-Host "Continue anyway? (y/n)"
        if ($continue -ne 'y' -and $continue -ne 'Y') {
            exit 1
        }
    }
} catch {
    Write-Host "⚠️  Could not check service status (this is OK)" -ForegroundColor Yellow
}

# Extract domain from config for display
try {
    $domain = ($configContent | Select-String "hostname: (.+)" | Select-Object -First 1 | ForEach-Object { $_.Matches[0].Groups[1].Value }).Trim()
} catch {
    $domain = "your-domain.com"
}

Write-Host ""
Write-Host "🚀 Starting tunnel..." -ForegroundColor Green
Write-Host "Your site will be available at: https://$domain" -ForegroundColor Cyan
Write-Host ""
Write-Host "Press Ctrl+C to stop the tunnel" -ForegroundColor Yellow
Write-Host "========================================" -ForegroundColor Gray

# Start the tunnel
cloudflared tunnel --config tunnel-config.yml run $tunnelId 