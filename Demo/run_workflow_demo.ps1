$VenvName = "venv"
$OutputFolder = "output"
$LogFile = "demo_run.log"

# Utility: Log with timestamp
function Log {
    param([string]$Message)
    $timestamp = Get-Date -Format "HH:mm:ss"
    Write-Host "[$timestamp] $Message"
    Add-Content -Path $LogFile -Value "[$timestamp] $Message"
}

# Utility: Run command with timing and logging
function Run-Step {
    param([string]$Title, [ScriptBlock]$Action)
    Log "== $Title =="
    $start = Get-Date
    try {
        & $Action 2>&1 | Tee-Object -FilePath $LogFile -Append
    } catch {
        Log "Error: $_"
        exit 1
    }
    $end = Get-Date
    $elapsed = ($end - $start).TotalSeconds
    Log "Completed in $elapsed seconds`n"
}

# Step 1: Move to project root
Run-Step "Changing to project root" {
    Set-Location ..
}

# Step 2: Create virtual environment
Run-Step "Creating virtual environment" {
    if (Test-Path $VenvName) {
        Write-Host "Virtual environment '$VenvName' already exists. Deleting it..." -ForegroundColor Yellow
        Remove-Item -Recurse -Force $VenvName
    }
    python -m venv $VenvName
}

# Step 3: Activate environment (Windows)
Run-Step "Activating environment" {
    & "$VenvName\Scripts\Activate.ps1"
}

# Step 4: Install dependencies
Run-Step "Installing dependencies" {
    python -m pip install --upgrade pip
    pip install -r requirements.txt
    pip install jupyter nbconvert
}

# Step 5: Recreate output folder
Run-Step "Recreating output folder" {
    if (Test-Path $OutputFolder) {
        Write-Host "Output folder '$OutputFolder' already exists. Deleting it..." -ForegroundColor Yellow
        Remove-Item -Recurse -Force $OutputFolder
    }
    New-Item -ItemType Directory -Path $OutputFolder | Out-Null
}

# Step 6: Run notebook and stream output, disabling timeout
Run-Step "Executing notebook and logging to console" {
    jupyter nbconvert `
        --execute ECG_BO_demo.ipynb `
        --to notebook `
        --output "$OutputFolder/demo_result.ipynb" `
        --ExecutePreprocessor.timeout=-1 `
        --ExecutePreprocessor.kernel_name=python3
}

# Step 7: Cleanup environment
Run-Step "Deleting virtual environment" {
    Set-Location ..
    Remove-Item -Recurse -Force $VenvName
    Set-Location Demo
}

# Done
Log "Notebook execution complete. Output saved to $OutputFolder/demo_result.ipynb"
