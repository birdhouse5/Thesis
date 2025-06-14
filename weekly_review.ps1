# weekly_review.ps1 - Generate weekly summary

$endDate = Get-Date
$startDate = $endDate.AddDays(-7)

Write-Host "📊 Weekly Review" -ForegroundColor Cyan
Write-Host "Period: $($startDate.ToString('yyyy-MM-dd')) to $($endDate.ToString('yyyy-MM-dd'))"

# Collect daily logs
$logs = @()
for ($i = 0; $i -lt 7; $i++) {
    $date = $endDate.AddDays(-$i)
    $logFile = "docs\logs\$($date.ToString('yyyy-MM-dd')).md"
    if (Test-Path $logFile) {
        $logs += @{
            Date = $date
            Content = Get-Content $logFile -Raw
        }
    }
}

# Collect experiments
$experiments = Get-ChildItem "docs\experiments\EXP*.md" | 
    Where-Object { $_.LastWriteTime -ge $startDate -and $_.LastWriteTime -le $endDate }

# Generate summary
$summary = @"
# Weekly Review: $($startDate.ToString('yyyy-MM-dd')) to $($endDate.ToString('yyyy-MM-dd'))

## Activity Summary
- Days logged: $($logs.Count) / 7
- Experiments run: $($experiments.Count)

## Key Activities
"@

# Extract key observations from logs
$allObservations = @()
foreach ($log in $logs) {
    if ($log.Content -match '## Key Observations\s*\n((?:- .*\n?)+)') {
        $allObservations += $matches[1].Trim()
    }
}

if ($allObservations.Count -gt 0) {
    $summary += "`n### Observations This Week`n"
    $summary += ($allObservations -join "`n") + "`n"
}

# List experiments
if ($experiments.Count -gt 0) {
    $summary += "`n## Experiments Run`n"
    foreach ($exp in $experiments) {
        $content = Get-Content $exp.FullName -Raw
        $status = "🏃"
        if ($content -match '✅') { $status = "✅" }
        elseif ($content -match '❌') { $status = "❌" }
        
        $summary += "- $status [$($exp.BaseName)]($($exp.FullName))`n"
    }
}

# Add knowledge gained section
$summary += @"

## Knowledge Gained
[Review and fill in key learnings]
- 
- 

## Next Week's Focus
[Based on this week's learnings]
- 
- 

## Questions to Explore
- 
"@

# Save weekly review
$reviewFile = "docs\logs\WEEK_$($startDate.ToString('yyyy-MM-dd')).md"
$summary | Set-Content $reviewFile -Encoding UTF8

Write-Host "`nWeekly review saved to: $reviewFile" -ForegroundColor Green
Start-Process $reviewFile

Start-Sleep -Seconds 3