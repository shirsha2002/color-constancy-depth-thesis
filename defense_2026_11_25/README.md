# Current defense materials

Updated 24 November 2026. The current report is
`output/defense_2026_11_25/Shirsha_Mitra_Proposal_Report_With_Experiments.docx`
and the matching PDF. Both have four pages, four experimental figures and six
tables. Student and supervisor fields are completed from the user's details.
The earlier six-page report is retained as a previous version.

The existing seven-slide presentation is unchanged by this report update.
The private ZIP now contains the illustrated report in place of the earlier
report. Its meeting details and the ZIP must remain private and gitignored.

## Evidence

All 21 ETH3D pilot models completed earlier. This update checked the 42 saved
checkpoint hashes and ran fresh inference on the local Intel Arc 140T GPU.
975 predictions from the learned, mask and PCCC fold-zero checkpoints across
all five frozen input conditions reproduced exactly. No new training or Modal
job was launched. `experiment_report_evidence.json` records the data, source
hashes, histories, selected inputs and predictions. Example IDs are the first
and last fold-zero test IDs alphabetically, without selection on their errors.

Images are real sparse-cloud renderings, not original full photographs.
Training curves use saved logs without smoothing. All charts are grayscale;
report text and table borders are black. The negative/inconclusive pilot result
is preserved. Longer schedules, repeated seeds and rendered-scene work remain
future studies, as stated in the report.

## Build and verify

Use `.venv-xpu/Scripts/python.exe prepare_report_evidence.py` for research
inference, and `.venv/Scripts/python.exe plot_experiment_report.py` for scientific
plots, running these scripts with the `defense_2026_11_25/` prefix from the repo
root. Use the bundled Codex Python runtime for `build_report_with_experiments.py`
and `export_experiment_report_pdf.py`; they use python-docx and ReportLab.

The canonical `render_docx.py` was attempted, but LibreOffice is absent. Native
Word page Enhanced Metafile rendering was used to inspect all four DOCX pages
with `render_experiment_report_native.ps1`. The searchable PDF is exported from
the same DOCX content with ReportLab and was inspected separately via Poppler.
Its paragraphs and table cells match the DOCX exactly. QA renders stay in
`.build/`. `experiment_report_native_render.json`,
`experiment_report_pdf_export.json` and `experiment_report_delivery.json` record
the checks. `verify_experiment_report.py` verifies the current report and ZIP;
the existing `verify_and_package.py` dispatches to it when this report exists.
