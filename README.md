# Color-constancy-assisted monocular depth estimation

Undergraduate thesis project by **Shirsha Mitra**, student ID **2023521460130**,
Sichuan University.

The first experiment measures whether controlled color casts change a frozen
monocular depth model's accuracy, and whether fixed color-constancy baselines
help. It uses genuine Kinect-derived NYU reference depth and saves numerical
predictions, paired metrics, configurations and source identifiers.

## Open and run

Open `ColorConstancyDepth.code-workspace` in VS Code. The workspace selects the
project's isolated `.venv` Python interpreter. The existing laptop setup is ready.

```powershell
# From D:\thesis work\thesis code
& .\.venv\Scripts\python.exe scripts/check_environment.py
& .\.venv\Scripts\python.exe -m unittest discover -s tests -v
& .\.venv\Scripts\python.exe scripts/run_experiment.py --manifest data/nyu_pilot/manifest.json --input-size 518 --threads 8
```

The experiment is resumable: completed predictions remain in the local cache.
The program checks the model, data-manifest, transformation-code and input-size
fingerprint before reusing a prediction. Change the configuration only in a
separate experiment directory or after moving the old results to an archive.

For a new checkout, run `scripts/setup.ps1`. It installs the pinned requirements,
clones the official model revision and downloads the checkpoint and selected
dataset samples. No downloaded code or weights are included in this repository.

## Experimental design

- Model: Depth Anything V2 Metric Hypersim Small,24.8M parameters, indoor maximum20m.
- Data:8 development scenes from official NYU train;32 evaluation scenes from
  official NYU test; one frame per scene; no scene overlap.
- Fixed seed:20261023. Scene selection occurs before predictions or evaluation.
- Conditions: original RGB plus warm,cool,green casts at mild and strong settings.
- Processing: none,Gray-World,White-Patch and fixed half-strength Gray-World.
- Color gains operate in linear RGB, preserving mean luminance before clipping.
- Evaluation: per-image AbsRel,RMSE(m),MAE(m),delta1; crop rows45:471,columns41:601;
  valid reference depth strictly between0.1 and10m; prediction clipping[0.1,10]m.
- No prediction-specific scale or shift fitting, and no parameter selection on
  evaluation reference depth. Model weights remain frozen.
- Confidence intervals resample whole evaluation images, retaining all six
  casts together. One evaluation image comes from each distinct scene.

This pilot does not reproduce the complete654-image NYU benchmark and does not
establish performance under real changes in lighting. Reference depth includes
the authors' missing-value filling. Classical correction baselines and a fixed
strength ablation are not a new algorithm contribution.

## Files

| Location | Purpose |
|---|---|
| `scripts/science.py` | Color transforms and depth metrics |
| `scripts/run_experiment.py` | Frozen-model inference, caching, paired evaluation |
| `scripts/download_data.py` | Byte-range extraction from the official NYU source |
| `data/nyu_pilot/manifest.json` | Scene IDs, official split, hashes and sources |
| `results/metrics.csv` | One row per image,condition and method after completion |
| `results/summary.json` | Measured aggregate scores and exact protocol |
| `report/` | Scientific figures and PDF generation |
| `presentation/` | Editable PPTX and slide PDF generation |
| `output/` | Final professor-facing report,slides and English message |

## Predict an image

```powershell
& .\.venv\Scripts\python.exe scripts/predict_image.py --image path\to\room.jpg --method gray_world_half
```

This saves a numerical depth estimate in meters and an image comparison under
`results/demo/`. Without reference depth, this demonstration does not measure
accuracy.

## Sources and ownership

See [THIRD_PARTY.md](THIRD_PARTY.md) and
[data/DATA_SOURCES.md](data/DATA_SOURCES.md). Original project code uses the MIT
license. Model and dataset rights remain with their authors. Environments,
credentials,upstream source,checkpoints,raw RGB/depth data and large prediction
caches are excluded from GitHub. Generated research figures include attributed
dataset examples.
