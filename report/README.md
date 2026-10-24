# Evidence-backed thesis report

`build_report.py` generates the eight-page English scientific progress report for Shirsha Mitra (student ID 2023521460130). `report_payload.json` is created only after the experiment outputs are checked; it contains the measured values and the interpretation used in the final report.

From the project directory:

```powershell
python report/make_figures.py
python report/prepare_payload.py
python report/build_report.py --payload report/report_payload.json
python report/render_report.py output/pdf/Shirsha_Mitra_Thesis_Progress_Report.pdf
```

The report uses ReportLab and Windows Arial fonts. The QA script uses PyMuPDF and Pillow. Its output is a PNG for every page, a contact sheet, and text-bound checks under `report/tmp/pdfs/`. Every rendered page must be inspected before delivery.

## Evidence policy

- Numerical statements must trace to saved metrics from the actual run.
- Dataset subset size and selection, model checkpoint, input size, validity masks, depth units, and scaling policy must be disclosed.
- Synthetic color casts must be identified as synthetic. They do not prove performance under real lighting changes.
- A pretrained inference experiment must not be described as training a new model or reproducing the published benchmark.
- Helpful and harmful correction cases must both be shown.

## Repository publication

Version the experiment code, configuration, dependency list, documentation, metrics CSV/JSON, and a small set of permitted result figures. Do not version the virtual environment, downloaded model weights, full datasets, caches, or upstream source trees. Link to official dataset/model download sources instead. A local run establishes the result; a GitHub commit records the files and does not independently validate the science.
