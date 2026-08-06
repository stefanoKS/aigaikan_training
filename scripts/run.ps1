[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"
. .\.venv\Scripts\Activate.ps1
python -m app.main

