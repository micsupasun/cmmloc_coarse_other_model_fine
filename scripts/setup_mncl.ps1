param(
    [string]$Destination = (
        Join-Path (Split-Path -Parent $PSScriptRoot) "third_party\MNCL"
    )
)

$ErrorActionPreference = "Stop"
$requiredSource = Join-Path $Destination "models\cross_matcher.py"

if (Test-Path -LiteralPath $requiredSource) {
    Write-Host "MNCL source is already available at $Destination"
    exit 0
}

if (Test-Path -LiteralPath $Destination) {
    throw "Destination exists but is not a valid MNCL checkout: $Destination"
}

$parent = Split-Path -Parent $Destination
New-Item -ItemType Directory -Force -Path $parent | Out-Null
git clone --depth 1 https://github.com/dqliua/MNCL.git $Destination

if (-not (Test-Path -LiteralPath $requiredSource)) {
    throw "MNCL clone completed but the expected source file is missing."
}

Write-Host "MNCL source installed at $Destination"
