# new_log.ps1 - Create daily log with proper date formatting

$today = Get-Date -Format "yyyy-MM-dd"
$filename = "docs\logs\$today.md"

# Create directories if they don't exist
if (-not (Test-Path "docs\logs")) {
    New-Item -ItemType Directory -Path "docs\logs" -Force | Out-Null
}
if (-not (Test-Path "docs\templates")) {
    New-Item -ItemType Directory -Path "docs\templates" -Force | Out-Null
}

# Create template if it doesn't exist
$templatePath = "docs\templates\daily_log.md"
if (-not (Test-Path $templatePath)) {
    $template = @"
# DATE - Trading RL Log

## What I Did
- 

## Key Observations
- 

## Next Steps
- [ ] 

## Quick Notes


Time spent: X hours
"@
    $template | Set-Content $templatePath -Encoding UTF8
    Write-Host "Created template: $templatePath" -ForegroundColor Cyan
}

if (-not (Test-Path $filename)) {
    # Copy template
    Copy-Item $templatePath $filename
    
    # Replace DATE placeholder with actual date
    (Get-Content $filename) -replace 'DATE', $today | Set-Content $filename -Encoding UTF8
    
    Write-Host "Created: $filename" -ForegroundColor Green
    
    # Small delay to ensure file is written
    Start-Sleep -Milliseconds 500
} else {
    Write-Host "Log already exists: $filename" -ForegroundColor Yellow
}

# Open in default text editor (usually Notepad or VS Code)
Start-Process $filename

# Keep console open for 2 seconds to see the message
Start-Sleep -Seconds 2