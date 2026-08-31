[CmdletBinding()]
param(
    [ValidateSet("cpu", "cuda")]
    [string]$Backend = "cpu",
    [string]$EnvironmentName = "anomalib-trainer",
    [switch]$Recreate
)

$ErrorActionPreference = "Stop"

$conda = Get-Command conda -ErrorAction SilentlyContinue
if (-not $conda) {
    throw "Conda was not found. Install Miniconda or Anaconda, then rerun setup."
}

function Test-CondaEnvironment {
    $environmentList = & $conda.Source env list --json | ConvertFrom-Json
    if ($LASTEXITCODE -ne 0) {
        throw "Conda could not list available environments."
    }

    return $null -ne ($environmentList.envs | Where-Object {
        (Split-Path $_ -Leaf) -ieq $EnvironmentName
    })
}

function Get-CudaRequirementsFile {
    try {
        $nvidiaNames = Get-CimInstance Win32_VideoController -ErrorAction Stop |
            Where-Object { $_.Name -match "NVIDIA" } |
            Select-Object -ExpandProperty Name
    }
    catch {
        Write-Warning "Could not detect the NVIDIA GPU. Installing the standard CUDA 12.6 build."
        return "requirements/cuda.txt"
    }

    if ($nvidiaNames | Where-Object { $_ -match "RTX\s+50\d{2}" }) {
        Write-Host "Detected Blackwell GPU: $($nvidiaNames -join ', ')"
        Write-Host "Installing PyTorch 2.7.1 with CUDA 12.8 support."
        return "requirements/cuda-blackwell.txt"
    }

    Write-Host "Detected NVIDIA GPU: $($nvidiaNames -join ', ')"
    Write-Host "Installing the standard CUDA 12.6 PyTorch build."
    return "requirements/cuda.txt"
}

$environmentExists = Test-CondaEnvironment
if ($Recreate -and $environmentExists) {
    & $conda.Source env remove --name $EnvironmentName --yes
    if ($LASTEXITCODE -ne 0) {
        throw "Conda could not remove environment '$EnvironmentName'."
    }
    $environmentExists = $false
}

if (-not $environmentExists) {
    & $conda.Source env create --file environment.yml --name $EnvironmentName
    if ($LASTEXITCODE -ne 0) {
        throw "Conda could not create environment '$EnvironmentName'."
    }
}

$requirementsFile = "requirements/$Backend.txt"
if ($Backend -eq "cuda") {
    $requirementsFile = Get-CudaRequirementsFile
}

& $conda.Source run --name $EnvironmentName python -m pip install --upgrade pip
& $conda.Source run --name $EnvironmentName python -m pip install -r $requirementsFile
if ($LASTEXITCODE -ne 0) {
    throw "Dependency installation failed for the $Backend backend."
}
$verificationArguments = @("scripts/verify_installation.py")
if ($Backend -eq "cuda") {
    $verificationArguments += "--require-cuda"
}
& $conda.Source run --name $EnvironmentName python @verificationArguments
if ($LASTEXITCODE -ne 0) {
    throw "Installation verification failed."
}
Write-Host ""
Write-Host "Setup complete. Start the application with:"
Write-Host "  powershell -ExecutionPolicy Bypass -File scripts/run.ps1 -EnvironmentName $EnvironmentName"
