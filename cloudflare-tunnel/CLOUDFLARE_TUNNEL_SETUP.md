# Cloudflare Tunnel Setup for RivalAI

This guide will help you set up a Cloudflare tunnel to make your RivalAI chess application accessible from the internet with HTTPS, even without port forwarding or a static IP address.

## What You'll Get

- 🌐 **Public Access**: Your RivalAI site accessible from anywhere
- 🔒 **HTTPS**: Automatic SSL/TLS encryption
- 🚀 **CDN**: Cloudflare's global network for fast loading
- 🛡️ **DDoS Protection**: Built-in security features
- 📱 **Mobile Friendly**: Works on all devices

## Prerequisites

- Windows 10/11 with PowerShell
- Your RivalAI project (this repository)
- A domain name (we'll help you get one)

## Step 1: Get a Domain

Choose one of these options:

### Option A: Buy through Cloudflare (Recommended)
1. Go to [Cloudflare Registrar](https://www.cloudflare.com/products/registrar/)
2. Search for available domains
3. Register your domain (they have competitive pricing)
4. Domain automatically added to your Cloudflare account

### Option B: Free Subdomain Services
- **FreeDNS**: https://freedns.afraid.org/
- **DuckDNS**: https://www.duckdns.org/
- **NoIP**: https://www.noip.com/

### Option C: Existing Domain
If you have a domain from another registrar:
1. Go to [Cloudflare](https://cloudflare.com)
2. Add your site to Cloudflare
3. Update your domain's nameservers to Cloudflare's

## Step 2: Install Cloudflared

Navigate to the cloudflare-tunnel directory and run PowerShell as Administrator:

```powershell
cd cloudflare-tunnel
.\install-cloudflared.ps1
```

This script will:
- Download the latest cloudflared
- Install it to Program Files
- Add it to your system PATH

## Step 3: Start Your Services

Before setting up the tunnel, make sure your RivalAI services are running:

```powershell
.\start-services.ps1
```

This will start:
- **Backend** (Rust server) on `http://localhost:3000`
- **Frontend** (React app) on `http://localhost:5173`

Keep these running in separate windows.

## Step 4: Set Up the Tunnel

Run the tunnel setup script:

```powershell
.\setup-tunnel.ps1
```

This script will:
1. **Login to Cloudflare** - Opens browser for authentication
2. **Create a tunnel** - Creates a secure tunnel named `rivalai-{username}`
3. **Configure DNS** - Automatically creates DNS records
4. **Update configuration** - Sets up routing rules
5. **Start the tunnel** - Makes your site live!

### What the Script Does Automatically

The tunnel configuration routes traffic as follows:
- `yoursite.com/api/*` → Backend (port 3000)
- `yoursite.com/ws/*` → WebSocket connections (port 3000)  
- `yoursite.com/community/*` → Community features (port 3000)
- `yoursite.com/*` → Frontend (port 5173)

## Step 5: Access Your Site

Once the tunnel is running, your RivalAI application will be available at:
- **Your Domain**: `https://yourdomain.com`
- **Features Available**:
  - Play chess against AI
  - Community chess games
  - Real-time updates via WebSocket
  - Model training interface
  - Game history and statistics

## Managing Your Tunnel

### Start the Tunnel
```powershell
cd cloudflare-tunnel
.\start-tunnel.ps1
```

Or manually:
```powershell
cloudflared tunnel --config tunnel-config.yml run {tunnel-id}
```

### Stop the Tunnel
Press `Ctrl+C` in the tunnel window

### Check Tunnel Status
```powershell
cloudflared tunnel list
```

### View Tunnel Logs
```powershell
cloudflared tunnel --config tunnel-config.yml run {tunnel-id} --loglevel debug
```

## Troubleshooting

### Common Issues

**1. "Tunnel not found" error**
- Make sure you're logged into the correct Cloudflare account
- Verify the tunnel ID in `tunnel-config.yml`

**2. "Service unavailable" (502/503 errors)**
- Ensure your backend service is running on port 3000
- Ensure your frontend service is running on port 5173
- Check that services are accessible via `http://localhost:3000` and `http://localhost:5173`

**3. WebSocket connections failing**
- Verify the `/ws/*` route in `tunnel-config.yml`
- Check that your backend WebSocket handler is working locally

**4. CSS/JS not loading**
- Clear browser cache
- Check browser developer tools for 404 errors
- Verify the frontend service is running

### Debug Steps

1. **Test locally first**:
   ```powershell
   # Test backend
   curl http://localhost:3000/api/health
   
   # Test frontend
   curl http://localhost:5173
   ```

2. **Check tunnel status**:
   ```powershell
   cloudflared tunnel list
   cloudflared tunnel info {tunnel-id}
   ```

3. **Verify DNS**:
   ```powershell
   nslookup yourdomain.com
   ```

## Advanced Configuration

### Custom Subdomains
You can set up subdomains for different services:

```yaml
ingress:
  - hostname: api.yourdomain.com
    service: http://localhost:3000
  - hostname: yourdomain.com
    service: http://localhost:5173
```

### Multiple Environments
Create separate tunnels for development/production:

```powershell
cloudflared tunnel create rivalai-dev
cloudflared tunnel create rivalai-prod
```

### Automatic Startup
To start your tunnel automatically on boot:

1. Create a Windows service:
   ```powershell
   cloudflared service install
   ```

2. Or use Task Scheduler to run your scripts on startup

## File Structure

After setup, you'll have these files in the `cloudflare-tunnel/` directory:
```
cloudflare-tunnel/
├── install-cloudflared.ps1    # Cloudflared installer
├── start-services.ps1         # Service startup script  
├── setup-tunnel.ps1          # Tunnel setup script
├── start-tunnel.ps1          # Quick tunnel starter
├── tunnel-config.yml         # Tunnel configuration
└── CLOUDFLARE_TUNNEL_SETUP.md # This guide
```

## Security Considerations

- **HTTPS Only**: Cloudflare enforces HTTPS by default
- **Rate Limiting**: Built-in DDoS protection
- **Access Control**: You can add Cloudflare Access for authentication
- **Firewall Rules**: Configure custom firewall rules in Cloudflare dashboard

## Next Steps

1. **Custom Domain**: Consider getting a memorable domain name
2. **Monitoring**: Set up Cloudflare Analytics to monitor traffic
3. **Performance**: Use Cloudflare's optimization features
4. **Security**: Enable additional security features in Cloudflare dashboard

## Support

- **Cloudflare Docs**: https://developers.cloudflare.com/cloudflare-one/connections/connect-apps/
- **RivalAI Issues**: Open an issue in this repository
- **Cloudflare Community**: https://community.cloudflare.com/

---

🎉 **Congratulations!** Your RivalAI chess application is now live on the internet with professional-grade infrastructure! 