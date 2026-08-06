[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"

if (-not $IsWindows) {
    throw "Anomalib Trainer currently supports Windows setup only."
}

$python = Get-Command py -ErrorAction SilentlyContinue
if ($python) {
    $pythonArgs = @("-3.11")
} else {
    $python = Get-Command python -ErrorAction SilentlyContinue
    if (-not $python) {
        throw "Python 3.11 was not found. Install Python 3.11 and rerun setup."
    }
    $pythonArgs = @()
}

if ($python) {
    & $python.Source @pythonArgs -m venv .venv
}
if (-not (Test-Path .venv)) {
    & python -m venv .venv
}

. .\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
$choice = Read-Host "Install CPU or CUDA dependencies? [cpu/cuda]"
switch ($choice.ToLowerInvariant()) {
    "cuda" { python -m pip install -r requirements/cuda.txt }
    default { python -m pip install -r requirements/cpu.txt }
}
python scripts/verify_installation.py
Write-Host ""
Write-Host "Setup complete. Start the application with:"
Write-Host "  powershell -ExecutionPolicy Bypass -File scripts/run.ps1"
