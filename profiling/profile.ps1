# Create a timestamped subfolder in /profiling
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$outputDir = "$timestamp"
New-Item -ItemType Directory -Path $outputDir -Force | Out-Null

# Resolve absolute path to script
$repoRoot = Resolve-Path ".."
$scriptToProfile = Join-Path $repoRoot "src/ecg_reference_demo.py"

# Launch scalene process with arguments
$proc = Start-Process scalene `
    -ArgumentList "$scriptToProfile", "--profile-interval", "0.1", "--output-dir", $outputDir `
    -NoNewWindow -PassThru

Write-Host "Started scalene (PID=$($proc.Id)) - will wait up to 6 hours."

# Wait for 6 hours = 21600000 milliseconds
$proc.WaitForExit(21600000)

# Kill if still running
if (-not $proc.HasExited) {
    Write-Host "Scalene still running after 6 hours. Terminating..."
    $proc.Kill()
} else {
    Write-Host "Scalene completed before timeout."
}

Write-Host "Profiling results saved in: $outputDir"
