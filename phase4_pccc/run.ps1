$ErrorActionPreference = 'Stop'
$taskRoot = Split-Path -Parent $PSScriptRoot
$taskPython = Join-Path $taskRoot '.venv\Scripts\python.exe'
Push-Location -LiteralPath $taskRoot
try {
    if (-not (Test-Path -LiteralPath $taskPython)) { throw 'Create the isolated Python environment described in phase4_pccc/README.md first.' }
    foreach ($taskScript in @('prepare.py', 'checks.py')) {
        & $taskPython (Join-Path $PSScriptRoot $taskScript)
        if ($LASTEXITCODE -ne 0) { throw "$taskScript failed" }
    }
    & $taskPython -u (Join-Path $PSScriptRoot 'pilot.py') train --epochs 100 --threads 8
    if ($LASTEXITCODE -ne 0) { throw 'Training failed' }
    & $taskPython -u (Join-Path $PSScriptRoot 'finish.py')
    if ($LASTEXITCODE -ne 0) { throw 'Evaluation or audit failed' }
}
finally { Pop-Location }
