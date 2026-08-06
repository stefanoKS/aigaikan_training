[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"
. .\.venv\Scripts\Activate.ps1

Remove-Item build, dist, release -Recurse -Force -ErrorAction SilentlyContinue
python -m pytest tests
python scripts/download_weights.py
python -m PyInstaller --noconfirm AnomalibTrainer.spec

$exe = Join-Path $PWD "dist\\AnomalibTrainer\\AnomalibTrainer.exe"
if (Test-Path $exe) {
    $process = Start-Process -FilePath $exe -PassThru
    Start-Sleep -Seconds 3
    if (-not $process.HasExited) {
        Stop-Process -Id $process.Id -Force
    }
}

New-Item -ItemType Directory -Force -Path release | Out-Null
Copy-Item dist\AnomalibTrainer -Destination release -Recurse

$iscc = Get-Command iscc -ErrorAction SilentlyContinue
if ($iscc) {
    & $iscc installer\AnomalibTrainer.iss
}
