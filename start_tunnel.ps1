<#
.SYNOPSIS
    Starts an on-demand quick Cloudflare Tunnel pointing to local serving gateway (Port 80)
.DESCRIPTION
    Uses Docker to spin up cloudflare/cloudflared or a local cloudflared binary, giving
    a public *.trycloudflare.com URL immediately for testing without needing account credentials.
#>

param(
    [int]$LocalPort = 80
)

Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "Starting AutoMLOps Cloudflare Quick Tunnel on Port $LocalPort..." -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan

# Check if Docker is available
docker run --rm --net=host cloudflare/cloudflared:latest tunnel --url http://localhost:$LocalPort
