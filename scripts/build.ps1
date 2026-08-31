[CmdletBinding()]
param(
    [string]$EnvironmentName = "anomalib-trainer"
)

$ErrorActionPreference = "Stop"
$conda = Get-Command conda -ErrorAction SilentlyContinue
if (-not $conda) {
    throw "Conda was not found. Run scripts/setup.ps1 after installing Miniconda or Anaconda."
}

Remove-Item build, dist, release -Recurse -Force -ErrorAction SilentlyContinue
& $conda.Source run --name $EnvironmentName python -m pytest tests
if ($LASTEXITCODE -ne 0) {
    throw "Tests failed. Build cancelled."
}
& $conda.Source run --name $EnvironmentName python scripts/download_weights.py
if ($LASTEXITCODE -ne 0) {
    throw "PatchCore weight download failed. Build cancelled."
}
& $conda.Source run --name $EnvironmentName python -m PyInstaller --noconfirm AnomalibTrainer.spec
if ($LASTEXITCODE -ne 0) {
    throw "PyInstaller build failed."
}

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
