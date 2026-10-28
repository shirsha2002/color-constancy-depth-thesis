param(
    [string]$Python = 'python',
    [switch]$UseExistingEnvironment,
    [switch]$AuditOnly
)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$manifestPath = Join-Path $projectRoot 'phase2\data\vari\manifest.json'
$requirementsPath = Join-Path $PSScriptRoot 'requirements-acquisition.txt'

if ($UseExistingEnvironment) {
    $acquisitionPython = $Python
} else {
    $environmentDir = Join-Path $PSScriptRoot '.venv-acquire'
    $acquisitionPython = Join-Path $environmentDir 'Scripts\python.exe'
    if (-not (Test-Path -LiteralPath $acquisitionPython)) {
        & $Python -c 'import sys; assert sys.version_info >= (3,11)'
        if ($LASTEXITCODE -ne 0) { throw 'A Python version of at least3.11 is required.' }
        & $Python -m venv $environmentDir
        if ($LASTEXITCODE -ne 0) { throw 'Could not create the separate acquisition environment.' }
    }
    & $acquisitionPython -m pip install -r $requirementsPath
    if ($LASTEXITCODE -ne 0) { throw 'Could not install acquisition dependencies.' }
}

& $acquisitionPython -c 'import requests,numpy,PIL,tifffile'
if ($LASTEXITCODE -ne 0) { throw 'The acquisition environment lacks requests,NumPy,Pillow,or tifffile.' }

if (Test-Path -LiteralPath $manifestPath) {
    Write-Host 'A frozen selected manifest already exists. Running its read-only audit.'
} elseif ($AuditOnly) {
    throw 'No selected manifest exists to audit.'
} else {
    & $acquisitionPython (Join-Path $PSScriptRoot 'acquire_vari_subset.py')
    if ($LASTEXITCODE -ne 0) { throw 'Selected Vari acquisition failed. Re-run to resume verified original files.' }
}

& $acquisitionPython (Join-Path $PSScriptRoot 'audit_frozen_vari.py')
if ($LASTEXITCODE -ne 0) { throw 'Selected Vari integrity audit failed.' }
Write-Host 'Selected Vari subset is complete and matches the recorded source snapshot.'
Write-Host $manifestPath
