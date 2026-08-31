[CmdletBinding()]
param(
	[string]$EnvironmentName = "anomalib-trainer"
)

$ErrorActionPreference = "Stop"
$conda = Get-Command conda -ErrorAction SilentlyContinue
if (-not $conda) {
	throw "Conda was not found. Run scripts/setup.ps1 after installing Miniconda or Anaconda."
}
& $conda.Source run --live-stream --name $EnvironmentName python -m app.main

