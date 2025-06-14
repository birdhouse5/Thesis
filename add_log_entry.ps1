# add_log_entry.ps1 - Add a timestamped entry to today's log

$today = Get-Date -Format "yyyy-MM-dd"
$timestamp = Get-Date -Format "HH:mm"
$filename = "docs\logs\$today.md"

# Create today's log if it doesn't exist
if (-not (Test-Path $filename)) {
    & .\new_log.ps1
    Start-Sleep -Seconds 2
}

# Add timestamped section
$entry = @"

---
### $timestamp Session

## What I Did
- 

## Key Observations
- 

"@

Add-Content $filename $entry -Encoding UTF8

# Open at the end of file
Start-Process $filename

Write-Host "Added entry to: $filename" -ForegroundColor Green
Start-Sleep -Seconds 2