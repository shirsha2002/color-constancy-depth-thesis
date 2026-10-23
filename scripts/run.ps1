$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot
& '.\.venv\Scripts\python.exe' scripts/run_experiment.py --manifest data/nyu_pilot/manifest.json --input-size 518 --threads 8
if ($LASTEXITCODE -ne 0) { throw 'Experiment stopped. See the error above; completed predictions remain cached.' }
