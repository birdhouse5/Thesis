# new_experiment.ps1 - Create new experiment documentation

param(
    [string]$Name = (Read-Host "Experiment name (e.g., temporal_split)")
)

# Create directories if needed
if (-not (Test-Path "docs\experiments")) {
    New-Item -ItemType Directory -Path "docs\experiments" -Force | Out-Null
}

# Find next experiment number
$existingExperiments = Get-ChildItem "docs\experiments\EXP*.md" -ErrorAction SilentlyContinue
$nextNumber = 1
if ($existingExperiments) {
    $numbers = $existingExperiments | ForEach-Object {
        if ($_.Name -match 'EXP(\d+)_') {
            [int]$matches[1]
        }
    }
    $nextNumber = ($numbers | Measure-Object -Maximum).Maximum + 1
}

$expId = "EXP{0:D3}_{1}" -f $nextNumber, $Name
$filename = "docs\experiments\$expId.md"
$today = Get-Date -Format "yyyy-MM-dd"
$time = Get-Date -Format "HH:mm"

# Create experiment template
$template = @"
# $expId

**Date**: $today  
**Time Started**: $time  
**Status**: 🏃 Running

## Hypothesis
[What do I think will happen?]

## Setup
\`\`\`python
# Key configuration
data_split = {
    'method': 'temporal',
    'train': ['2018-2021'],
    'test': ['2022-2023']
}

model_config = {
    'latent_dim': 32,
    'learning_rate': 3e-4
}
\`\`\`

## Results
| Metric | Train | Test | Baseline |
|--------|-------|------|----------|
| Sharpe |       |      |          |
| Return |       |      |          |
| Max DD |       |      |          |

## Observations
- 

## Conclusion
**Success?** ❓ (Change to ✅ or ❌)

**Key Learning**: 

## Artifacts
- Config: \`configs/$expId.yaml\`
- Model: \`models/$expId.pt\`
- Plots: \`results/$expId/\`

## Link to Daily Log
- [Today's log](../logs/$today.md)
"@

$template | Set-Content $filename -Encoding UTF8

Write-Host "Created: $filename" -ForegroundColor Green

# Also add entry to today's log
$logEntry = @"

---
### Experiment Started: $expId
- See: [docs/experiments/$expId.md](../experiments/$expId.md)
"@

$todayLog = "docs\logs\$today.md"
if (Test-Path $todayLog) {
    Add-Content $todayLog $logEntry -Encoding UTF8
    Write-Host "Added reference to daily log" -ForegroundColor Cyan
}

# Open the experiment file
Start-Process $filename

# Keep console open
Start-Sleep -Seconds 2