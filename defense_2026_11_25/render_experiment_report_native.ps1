$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Drawing
$taskRoot = Split-Path -Parent $PSScriptRoot
$docPath = Join-Path $taskRoot 'output\defense_2026_11_25\Shirsha_Mitra_Proposal_Report_With_Experiments.docx'
$renderPath = Join-Path $PSScriptRoot '.build\experiment_report_native'
[void][System.IO.Directory]::CreateDirectory($renderPath)
$app = $null
$doc = $null
try {
    $app = New-Object -ComObject Word.Application
    $app.Visible = $false
    $app.DisplayAlerts = 0
    $doc = $app.Documents.Open($docPath, $false, $true)
    $doc.ActiveWindow.View.Type = 3
    $doc.ActiveWindow.View.DisplayPageBoundaries = $true
    $doc.ActiveWindow.View.Zoom.Percentage = 100
    $doc.Repaginate()
    $count = $doc.ComputeStatistics(2)
    Write-Output ('Rendering ' + $count + ' native Word pages')
    $pages = @()
    for ($number = 1; $number -le $count; $number++) {
        $pageRange = $doc.GoTo(1, 1, $number)
        $pageRange.Select()
        $doc.ActiveWindow.ScrollIntoView($pageRange, $true)
        $app.ScreenRefresh()
        [byte[]]$bits = $doc.ActiveWindow.Panes.Item(1).Pages.Item($number).EnhMetaFileBits
        $emf = Join-Path $renderPath ('page-' + $number + '.emf')
        $png = Join-Path $renderPath ('page-' + $number + '.png')
        [System.IO.File]::WriteAllBytes($emf, $bits)
        $stream = New-Object System.IO.MemoryStream(,$bits)
        $pageImage = [System.Drawing.Image]::FromStream($stream)
        $bitmap = New-Object System.Drawing.Bitmap(2480, 3508)
        $bitmap.SetResolution(300, 300)
        $graphics = [System.Drawing.Graphics]::FromImage($bitmap)
        $graphics.Clear([System.Drawing.Color]::White)
        $graphics.InterpolationMode = [System.Drawing.Drawing2D.InterpolationMode]::HighQualityBicubic
        $graphics.DrawImage($pageImage, 0, 0, 2480, 3508)
        $bitmap.Save($png, [System.Drawing.Imaging.ImageFormat]::Png)
        $graphics.Dispose()
        $bitmap.Dispose()
        $pageImage.Dispose()
        $stream.Dispose()
        $pages += [ordered]@{ page = $number; path = $png; emf = $emf }
        Write-Output ('Rendered page ' + $number)
    }
    [ordered]@{
        status = 'PASS'
        renderer = 'Microsoft Word native page Enhanced Metafile rendering'
        pages = $count
        word_version = $app.Version
        source = $docPath
        source_sha256 = (Get-FileHash -LiteralPath $docPath -Algorithm SHA256).Hash.ToLowerInvariant()
        images = $pages
        fallback_reason = 'LibreOffice unavailable and Word PDF export stalled; used native page rendering'
    } | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath (Join-Path $PSScriptRoot 'experiment_report_native_render.json') -Encoding UTF8
}
finally {
    if ($null -ne $doc) {
        $doc.Close(0)
        [void][System.Runtime.InteropServices.Marshal]::ReleaseComObject($doc)
    }
    if ($null -ne $app) {
        if ($app.Documents.Count -eq 0) { $app.Quit() }
        [void][System.Runtime.InteropServices.Marshal]::ReleaseComObject($app)
    }
}
