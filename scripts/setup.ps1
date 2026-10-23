param([switch]$SkipData)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot
if (-not (Test-Path -LiteralPath '.venv\Scripts\python.exe')) {
    python -m venv .venv
    if ($LASTEXITCODE -ne 0) { throw 'Python environment creation failed.' }
}
$projectPython = Join-Path $projectRoot '.venv\Scripts\python.exe'
& $projectPython -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) { throw 'Python dependencies could not be installed.' }
if (-not (Test-Path -LiteralPath 'external\Depth-Anything-V2\.git')) {
    git clone https://github.com/DepthAnything/Depth-Anything-V2.git external/Depth-Anything-V2
    if ($LASTEXITCODE -ne 0) { throw 'Upstream model download failed.' }
}
git -C external/Depth-Anything-V2 checkout a561b849ebae10a6f5ef49e26c83cbbcd36c71bf
if ($LASTEXITCODE -ne 0) { throw 'Pinned model revision could not be selected.' }
& $projectPython scripts/download_model.py
if ($LASTEXITCODE -ne 0) { throw 'Model checkpoint download failed.' }
if (-not $SkipData) {
    & $projectPython scripts/download_data.py
    if ($LASTEXITCODE -ne 0) { throw 'Dataset acquisition failed.' }
}
& $projectPython scripts/check_environment.py
& $projectPython -m unittest discover -s tests -v
if ($LASTEXITCODE -ne 0) { throw 'Scientific verification failed.' }
Write-Host 'Setup complete. Open ColorConstancyDepth.code-workspace in VS Code.'
