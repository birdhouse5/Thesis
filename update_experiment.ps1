# update_experiment.ps1 - Quick update experiment results

# Get all experiments sorted by most recent first
$experiments = Get-ChildItem "docs\experiments\EXP*.md" | Sort-Object LastWriteTime -Descending

if ($experiments.Count -eq 0) {
    Write-Host "No experiments found!" -ForegroundColor Red
    Start-Sleep -Seconds 2
    exit
}

# Display experiments with numbers for selection
Write-Host "`nSelect experiment to update:" -ForegroundColor Cyan
Write-Host "----------------------------" -ForegroundColor Gray

$i = 1
$experimentMap = @{}
foreach ($exp in $experiments) {
    # Read status from file
    $content = Get-Content $exp.FullName -Raw
    $status = "🏃"
    if ($content -match '✅ Complete') { $status = "✅" }
    elseif ($content -match '❌ Complete') { $status = "❌" }
    
    # Extract experiment title
    $title = $exp.BaseName
    if ($content -match '# (EXP\d+_[^\n]+)') {
        $title = $matches[1]
    }
    
    # Format last modified time
    $lastModified = $exp.LastWriteTime.ToString("yyyy-MM-dd HH:mm")
    
    Write-Host "$i. $status $title" -ForegroundColor White
    Write-Host "   Last modified: $lastModified" -ForegroundColor Gray
    
    $experimentMap[$i] = $exp
    $i++
    
    # Show only first 10 for readability
    if ($i -gt 10) {
        Write-Host "`n   ... and $($experiments.Count - 10) more" -ForegroundColor Gray
        Write-Host "   (Enter 0 to see all)" -ForegroundColor Yellow
        break
    }
}

# Get selection
Write-Host ""
$selection = Read-Host "Enter number (or 0 to see all)"

# Handle "see all" option
if ($selection -eq "0") {
    Clear-Host
    Write-Host "All experiments:" -ForegroundColor Cyan
    Write-Host "----------------" -ForegroundColor Gray
    
    $i = 1
    $experimentMap = @{}
    foreach ($exp in $experiments) {
        $content = Get-Content $exp.FullName -Raw
        $status = "🏃"
        if ($content -match '✅ Complete') { $status = "✅" }
        elseif ($content -match '❌ Complete') { $status = "❌" }
        
        Write-Host "$i. $status $($exp.BaseName)" -ForegroundColor White
        $experimentMap[$i] = $exp
        $i++
    }
    
    Write-Host ""
    $selection = Read-Host "Enter number"
}

# Validate selection
if (-not $experimentMap.ContainsKey([int]$selection)) {
    Write-Host "Invalid selection!" -ForegroundColor Red
    Start-Sleep -Seconds 2
    exit
}

$selectedExp = $experimentMap[[int]$selection]
$filename = $selectedExp.FullName
$expName = $selectedExp.BaseName

Write-Host "`nUpdating: $expName" -ForegroundColor Green

# Quick results entry
Write-Host "`nQuick Results Entry" -ForegroundColor Yellow
Write-Host "-------------------" -ForegroundColor Gray
$trainSharpe = Read-Host "Train Sharpe"
$testSharpe = Read-Host "Test Sharpe"
$success = Read-Host "Success? (y/n)"

$timestamp = Get-Date -Format "yyyy-MM-dd HH:mm"

# Read current content
$content = Get-Content $filename -Raw

# Update status
if ($success -eq 'y') {
    $content = $content -replace '🏃 Running', '✅ Complete'
    $content = $content -replace '❓ \(Change to ✅ or ❌\)', '✅'
} else {
    $content = $content -replace '🏃 Running', '❌ Complete'
    $content = $content -replace '❓ \(Change to ✅ or ❌\)', '❌'
}

# Add results note
$resultsNote = @"

## Update: $timestamp
- Train Sharpe: $trainSharpe
- Test Sharpe: $testSharpe
- Status: $(if ($success -eq 'y') { 'Success' } else { 'Failed' })
"@

# Append to file
$content + $resultsNote | Set-Content $filename -Encoding UTF8

Write-Host "`nUpdated: $filename" -ForegroundColor Green

# Also update daily log
$today = Get-Date -Format "yyyy-MM-dd"
$logEntry = @"

### Experiment Updated: $expName
- Train Sharpe: $trainSharpe, Test Sharpe: $testSharpe
- Result: $(if ($success -eq 'y') { '✅ Success' } else { '❌ Failed' })
"@

$todayLog = "docs\logs\$today.md"
if (Test-Path $todayLog) {
    Add-Content $todayLog $logEntry -Encoding UTF8
    Write-Host "Added to daily log" -ForegroundColor Cyan
}

# Open to review
Start-Process $filename
Start-Sleep -Seconds 2