# Original PCCC auxiliary assets

Source: [Point Cloud Color Constancy, CVPR 2022](https://github.com/xyxingx/Point-Cloud-Color-Constancy),
by Xiaoyan Xing, Yanlin Qian and colleagues. Public assets were acquired from
the Drive folder linked by that author repository. File identities, lengths,
archive members and SHA-256 values are recorded in `acquisition_manifest.json`.

This acquisition includes original illumination labels, fold tables and sampled
XYZRGB point-cloud archives for the NYU/DIODE, ETH3D and DepthAWB groups. It does
not include the full native dense RGB/depth image archives. An XYZRGB cloud is
not a replacement for a dense aligned metric depth reference, and its coordinate
units/preprocessing must be audited before any geometry scoring.

NYU/DIODE labels use reannotated camera-processed, linearized RGB; ETH3D and
DepthAWB involve RAW-linear data. Neutral-area annotations and color-checker
calibration are different forms of reference information. Label-file names,
source frame indexing, image linearization and official folds need explicit
alignment before training an absolute-illuminant model.

The current frozen phase-three depth experiment does not use these assets as
extra supervised training data and does not claim a PCCC baseline reproduction.
The published code has execution/configuration mismatches described in the
[literature review](../../research/LITERATURE_REVIEW.md). Any future adapted
baseline must document its fixes, validation split and source differences.

The authors retain source/data rights. Their upstream code is not copied into
this project's MIT-licensed source. The small downloader and attribution are
provided; bulky original archives and clouds remain local.

For a fresh acquisition, install the optional archive dependency, then run:

```powershell
& .\.venv\Scripts\python.exe -m pip install -r phase3/data_tools/requirements-pccc.txt
& .\.venv\Scripts\python.exe phase3/data_tools/acquire_pccc.py
```

An existing complete acquisition can be verified without rewriting its frozen
manifest using `phase3/data_tools/acquire_pccc.py --audit-only`.
