$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot
$artifactRuntime = Join-Path $env:USERPROFILE '.cache\codex-runtimes\codex-primary-runtime'
$artifactNode = Join-Path $artifactRuntime 'dependencies\node\bin\node.exe'
$artifactPackages = Join-Path $artifactRuntime 'dependencies\node\node_modules'
if (-not (Test-Path -LiteralPath $artifactNode)) { throw 'The Codex presentation runtime is required to rebuild the PPTX.' }
$presentationPackages = Join-Path $PSScriptRoot 'node_modules'
if (-not (Test-Path -LiteralPath $presentationPackages)) {
    New-Item -ItemType Junction -Path $presentationPackages -Target $artifactPackages | Out-Null
}
$env:RUNTIME_NODE_MODULES = $artifactPackages
$buildStartedUtc = [DateTime]::UtcNow
& $artifactNode presentation/build_deck.mjs
$builderExit = $LASTEXITCODE
$buildReceiptPath = Join-Path $PSScriptRoot '.build\latest_build.json'
if (-not (Test-Path -LiteralPath $buildReceiptPath)) { throw "Editable presentation generation failed (exit $builderExit)." }
$receiptFile = Get-Item -LiteralPath $buildReceiptPath
if ($receiptFile.LastWriteTimeUtc -lt $buildStartedUtc) { throw "No completed presentation receipt from this build (exit $builderExit)." }
$buildReceipt = Get-Content -LiteralPath $buildReceiptPath -Raw -Encoding UTF8 | ConvertFrom-Json
if (-not (Test-Path -LiteralPath $buildReceipt.finalPath)) { throw 'Validated PowerPoint file is missing.' }
$deliveredHash = (Get-FileHash -LiteralPath $buildReceipt.finalPath -Algorithm SHA256).Hash.ToLowerInvariant()
if ($deliveredHash -ne $buildReceipt.sha256) { throw 'Delivered PowerPoint does not match the validated file.' }
$validation = Get-Content -LiteralPath $buildReceipt.validationReceipt -Raw -Encoding UTF8 | ConvertFrom-Json
if ($validation.packageIntegrity.status -ne 'pass' -or -not $validation.firstPartyImport.passed -or -not $validation.nativeChartValidation.passed) { throw 'Presentation validation did not pass.' }
# The supplied exporter may return 1 after a successful build. A fresh completed
# receipt and matching final hash establish success without hiding build errors.
& $artifactNode presentation/export_deck_pdf.mjs
if ($LASTEXITCODE -ne 0) { throw 'Presentation PDF copy generation failed.' }
Write-Host 'Presentation generated. Inspect rendered slides under presentation/.build/renders.'
