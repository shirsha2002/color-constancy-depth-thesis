param([string]$ResultDirectory = 'phase4_pccc/results/cpu_v1')
$ErrorActionPreference = 'Stop'
$taskRoot = Split-Path -Parent $PSScriptRoot
$taskPython = Join-Path $taskRoot '.venv\Scripts\python.exe'
Push-Location -LiteralPath $taskRoot
try {
    if (-not (Test-Path -LiteralPath $taskPython)) { throw 'Create the isolated Python environment described in phase4_pccc/README.md first.' }
    $taskFingerprintPath = Join-Path $ResultDirectory 'evaluation_sources.json'
    if (Test-Path -LiteralPath $taskFingerprintPath) {
        $taskIdentity = Get-Content -LiteralPath $taskFingerprintPath -Raw | ConvertFrom-Json
        foreach ($taskModelName in $taskIdentity.checkpoint_sha256.PSObject.Properties.Name) {
            $taskModelPath = Join-Path (Join-Path $ResultDirectory 'models') $taskModelName
            if (-not (Test-Path -LiteralPath $taskModelPath)) {
                throw 'This frozen result directory has missing reference weights. Use -ResultDirectory phase4_pccc/results/reproduction_01 for a fresh reproduction.'
            }
        }
    }
    foreach ($taskScript in @('prepare.py', 'checks.py')) {
        & $taskPython (Join-Path $PSScriptRoot $taskScript)
        if ($LASTEXITCODE -ne 0) { throw "$taskScript failed" }
    }
    & $taskPython -u (Join-Path $PSScriptRoot 'pilot.py') train --epochs 100 --threads 8 --out $ResultDirectory
    if ($LASTEXITCODE -ne 0) { throw 'Training failed' }
    & $taskPython -u (Join-Path $PSScriptRoot 'finish.py') --out $ResultDirectory
    if ($LASTEXITCODE -ne 0) { throw 'Evaluation or audit failed' }
}
finally { Pop-Location }
