param(
    [string]$Destination = (
        Join-Path (Split-Path -Parent $PSScriptRoot) "third_party\MNCL"
    )
)

$ErrorActionPreference = "Stop"
$requiredSource = Join-Path $Destination "models\cross_matcher.py"
$mnclCommit = "11ea10e1658b38e53b2127f4ee55f9d4236d9f50"

if (Test-Path -LiteralPath $requiredSource) {
    $currentCommit = git -C $Destination rev-parse HEAD
    if ($LASTEXITCODE -ne 0 -or $currentCommit.Trim() -ne $mnclCommit) {
        throw (
            "MNCL checkout is not the verified revision $mnclCommit. " +
            "Current revision: $currentCommit. Move third_party\MNCL aside, " +
            "then run this script again."
        )
    }
    Write-Host "Verified MNCL source $mnclCommit at $Destination"
    exit 0
}

if (Test-Path -LiteralPath $Destination) {
    throw "Destination exists but is not a valid MNCL checkout: $Destination"
}

$parent = Split-Path -Parent $Destination
New-Item -ItemType Directory -Force -Path $parent | Out-Null
git clone https://github.com/dqliua/MNCL.git $Destination
git -C $Destination checkout --detach $mnclCommit

if ($LASTEXITCODE -ne 0 -or -not (Test-Path -LiteralPath $requiredSource)) {
    throw "MNCL clone completed but the expected source file is missing."
}

Write-Host "MNCL source $mnclCommit installed at $Destination"
