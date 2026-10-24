$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot
$projectPython = Join-Path $projectRoot '.venv\Scripts\python.exe'
foreach ($reportScript in @('report/make_figures.py','report/prepare_payload.py')) {
    & $projectPython $reportScript
    if ($LASTEXITCODE -ne 0) { throw "Report stage failed: $reportScript" }
}
& $projectPython report/build_report.py --payload report/report_payload.json
if ($LASTEXITCODE -ne 0) { throw 'PDF report generation failed.' }
& $projectPython report/render_report.py output/pdf/Shirsha_Mitra_Thesis_Progress_Report.pdf
if ($LASTEXITCODE -ne 0) { throw 'PDF report render checks failed.' }
& $projectPython scripts/make_qq_update.py
Write-Host 'Report generated. Inspect rendered pages under report/tmp/pdfs.'
