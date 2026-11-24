# Color-constancy-assisted monocular depth estimation

Undergraduate thesis project by **Shirsha Mitra**, student ID **2023521460130**,
Sichuan University.

## Download the latest thesis and presentation files

Updated 24 November 2026. Start with the [friend's presentation guide](FRIEND_START_HERE.md).

- [Download the public presentation materials ZIP](output/defense_2026_11_25/Presentation_Materials_Public.zip)
- [Editable PowerPoint](output/defense_2026_11_25/Shirsha_Mitra_Proposal_Defense_2026.pptx)
- [Illustrated proposal PDF](output/defense_2026_11_25/Shirsha_Mitra_Proposal_Report_With_Experiments.pdf)
- [Editable Word report](output/defense_2026_11_25/Shirsha_Mitra_Proposal_Report_With_Experiments.docx)
- [Official Tables 1 to 3 unsigned draft](<output/proposal_submission/2023521460130_Shirsha Mitra_Proposal Report Table1_3.pdf>)
- [Experimental figures](output/defense_2026_11_25/figures) and [slide previews](output/defense_2026_11_25/slide_previews)
- [Training checkpoints and records](https://github.com/shirsha2002/color-constancy-depth-thesis/releases/tag/thesis-handoff-2026-11-24)

The written report has four pages, four experimental figures and six tables.
The official form retains seven template pages and appends three pages of
experimental evidence. It needs supervisor completion, actual dates and signatures.
The research has not established a reliability-gating improvement. The registered
title remains unchanged pending supervisor confirmation of the current scope.

## Audited fusion pilot and defense preparation

23 November 2026. The [ETH3D fusion pilot](phase9_reliability_fusion/pilot_v1/RESULTS.md)
completed 21 local-Intel-GPU models at 100 epochs. All 42 checkpoints and 13,650
prediction rows passed audit and independent verification. Both primary intervals
include zero; the gate did not outperform the matched PCCC baseline. No new
Modal training was used. This is a reduced development pilot, not full-budget
or independent scene-level validation.

The [defense folder](output/defense_2026_11_25/START_HERE.txt) contains the
five-main-slide presentation, two backups, written proposal, speaking script
and English/Bengali questions for 25 November. Student and supervisor details have
been filled from the student's information; formal title/scope alignment and the
supervisor-owned form sections still require supervisor confirmation.

Current continuation: [ETH3D scene-scale inputs](phase8_eth3d_scale/README.md)
are prepared and independently verified for all 195 captures. The protocol is
frozen; no new training or accuracy result is claimed. The latest completed
full-budget representation experiment remains the camera-frame study below.

## Earlier camera-frame and depth-scale study, 16 November

**Completed and independently audited.** Nine new PCCC models trained for 10,000
epochs on the official NYU-v2 + DIODE folds, reusing six native-reference models.
Mean final-epoch angular errors are 2.6135° for native geometry, 2.5471° for
canonical camera axes, 2.7192° after median-depth normalization, and 2.8164° for
the matched canonical flat control. **Neither primary comparison establishes
a normalization benefit**; both paired 95% intervals include zero.

Read the [16 November report](output/pdf/Shirsha_Mitra_PCCC_Depth_Format_Update_2026_11_16.pdf),
[all results and limits](phase7_depth_format/RESULTS.md),
[execution and recovery notes](phase7_depth_format/EXECUTION_NOTES.md), and
[independent audit](phase7_depth_format/results/nyu_frame_v1/audit.json).
The audit checked 3,240 saved predictions and re-inferred all 15 final checkpoints.
This is a representation diagnostic with one seed per fold and prior development
exposure. That study did not implement reliability-aware fusion; the new local
prototype is linked above. The cloud application
is stopped; no extra training or automatic retry is pending. The cancelled
laptop study remains stopped.

## Earlier: full PCCC on all three datasets with a cloud GPU, 10 November

PCCC was trained with the paper's full budget (**10,000 epochs**, official 3-fold splits) on
**DepthAWB, ETH3D and NYU-v2 + DIODE** on rented Modal NVIDIA T4 GPUs: 48 models, about
22.5 GPU-hours. Real 3D clearly helps on ETH3D (0.98 vs 1.31 degrees without depth) but not on
DepthAWB (1.77 vs 1.86) or NYU-v2 + DIODE (2.61 vs 2.51). A depth-quality test shows that damaged
depth hurts a normally trained model. Read [phase6_modal/README.md](phase6_modal/README.md) and
[the report](output/pdf/Shirsha_Mitra_PCCC_GPU_Update_2026_11_10.pdf). This supersedes the
cancelled laptop run below; that run's partial checkpoints stay local and unused.

## Paper-budget reproduction cancelled, 9 November

**Cancelled at the user's request because of runtime. Both CPU and GPU workers
are stopped; queued jobs and automatic completion publication will not run.**
Partial checkpoints are preserved locally (GPU epoch6200, CPU epoch1550).
The earlier completed pilot remains available below. See [cancellation status](phase5_pccc/STATUS.md).

The cancelled experiment was planned to run PCCC at **10000 epochs**, all three official DepthAWB
folds and 256-point input. The **Intel Arc 140T GPU** passed actual training and
exact restart checks and is the main backend. A separate full-budget CPU
reference was launched as well. This is the DepthAWB scope, not all experiments
and datasets in the original paper.

**At launch these new results were not complete.** Read [the current status](phase5_pccc/STATUS.md)
and [the runnable guide](phase5_pccc/README.md). The completed 100-epoch pilot
below remains a separate earlier experiment. This cancelled run has no pending
completion or publication step.

## Earlier: small PCCC reproduction, 9 November

This earlier follow-up directly follows **Point Cloud Color Constancy (CVPR 2022)**:
RGB plus depth-derived colored points are used to estimate illumination. This
answers the supervisor's request for PCCC data and CC results; the earlier
monocular-depth experiments remain separately preserved.

The laptop pilot uses all **185 real DepthAWB samples**, 256 XYZRGB points each,
with **98 training / 25 inner validation / 62 held-out test** samples. Four
predeclared model variants are trained with three seeds each. Read
[the PCCC reproduction guide](phase4_pccc/README.md),
[measured results](phase4_pccc/RESULTS.md) and
[the independent audit](phase4_pccc/results/cpu_v1/independent_audit.json).
The 100-epoch, single-outer-fold CPU study is an adapted reproduction pilot;
it does not reproduce the complete paper's published benchmark.

The released-head real-geometry model has mean angular error **3.74 +/- 0.33
degrees** versus **3.42 +/- 0.06** for flat geometry (three seeds). The restored
weighted family also does not establish a geometry benefit in this 100-epoch
pilot. All **868** held-out scores and twelve checkpoint predictions passed
independent verification. See the full tables and uncertainty before making
a claim about depth helping CC.

## Earlier completed depth studies, through 7 November

The default `main` branch also preserves the three earlier research phases. Read
the [7 November weekly report](output/pdf/Shirsha_Mitra_Weekly_Update_2026_11_07.pdf),
the [phase-three measured results](phase3/RESULTS.md) and the
[current project status](PROJECT_STATUS.md).

The 7 November study acquires **735 captured real-light ReMID images** and
**368 new NYU RGB-D scenes** (200 training / 40 validation / 128 test).
It compares **15 trained variants plus the frozen model**, with 26,224 unique
evaluated predictions and a separate 640-prediction floating-point color/exposure
probe. Read [phase three](phase3/README.md) for the internal architecture, losses,
matched controls, immutable inputs and independently recomputed metrics.

This is a controlled CPU research study at a 224-pixel model transform, not a
full benchmark or a demonstrated novel CC algorithm. Improvements over the
frozen model must be distinguished from a color-specific contribution.

| Completed study | Evidence and outcome |
|---|---|
| Real-light and explicit chromatic controls, 7 November | ReMID external evaluation, new scene-separated NYU data, global/local internal adapters and matched loss ablations. [All methods and claim limits](phase3/RESULTS.md). |
| Initial NYU preprocessing pilot | 40 images from separate development/evaluation scenes; 1,120 predictions; fixed color-cast and correction comparisons. [Results](results/summary.json). |
| Integrated monocular depth prototype | Vari: 528 images, 48 camera groups, 2 rooms; six methods evaluated on 264 images. The CC-inspired module has AbsRel 0.4277 versus 0.3988 for ordinary task adaptation; a CC-specific advantage is unproven. [Methods and results](phase2/README.md). |
| Calibrated real-light stereo diagnostic | DTU: 42 view/light cases and 210 method scores. Held-out raw matching MAE rises from 13.19 to 54.21 mm with changed lighting; internal scalar-gain compensation gives 29.90 mm. [Verified results](phase2/results/stereo/main_v2/RESULTS.md). |

### Reports and presentations

| Material | Scope |
|---|---|
| [16 November depth-format report](output/pdf/Shirsha_Mitra_PCCC_Depth_Format_Update_2026_11_16.pdf) | Latest camera-frame and depth-scale diagnostic; both primary contrasts remain inconclusive |
| [Latest measured PCCC results](phase7_depth_format/RESULTS.md) / [audit](phase7_depth_format/results/nyu_frame_v1/audit.json) | Nine new 10,000-epoch models, six reused references and independent prediction/checkpoint checks |
| [Latest English QQ draft](phase7_depth_format/QQ_Update_English.txt) | Short message based on the audited 16 November study |
| [Earlier PCCC pilot results](phase4_pccc/RESULTS.md) | Small CPU reproduction on genuine DepthAWB, source/paper distinctions and geometry controls |
| [7 November research report](output/pdf/Shirsha_Mitra_Weekly_Update_2026_11_07.pdf) | Earlier ReMID/NYU depth study, simulated controls, internal methods and loss ablations |
| [Previous phase-two report](output/pdf/Shirsha_Mitra_Phase2_Research_Update.pdf) | Earlier Vari prototype and real-light stereo diagnostic |
| [Original PowerPoint](output/Shirsha_Mitra_Thesis_Progress.pptx) / [slide PDF](output/Shirsha_Mitra_Thesis_Progress_Slides.pdf) | Initial NYU preprocessing pilot; this earlier deck does not contain the follow-up experiments |
| [Original progress report](output/pdf/Shirsha_Mitra_Thesis_Progress_Report.pdf) | Detailed initial NYU experiment |
| [Presentation notes](output/Shirsha_Mitra_Presentation_Notes.txt) | Notes for the original PowerPoint |
| [Small 3D point-cloud previews](phase2/results/stereo/main_v2/pointclouds_inference/) | Deterministic 30,000-point previews of actual stereo outputs, including outliers; not benchmark-quality meshes |

Source code, protocols, selected data manifests, all measured metric tables,
audit evidence, nineteen trained adapters across phases two and three, four first-seed PCCC checkpoints, and final materials are versioned here.
Raw datasets, pretrained base weights, installed environments and large inference
caches are retained locally and reproduced through the documented setup and
acquisition commands. See [what is included](PROJECT_STATUS.md#repository-scope).

The previous follow-up study is documented in [phase2/README.md](phase2/README.md).
It acquires paired-light Vari and calibrated DTU data, runs a task-integrated
feature-conditioning prototype, and compares strong adaptation and radiometric
controls. Lighting sensitivity is observed; the current chromatic module has
not demonstrated an advantage over ordinary task adaptation. The original
NYU preprocessing study below is preserved separately.

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

- Model: Depth Anything V2 Metric Hypersim Small, 24.8M parameters, indoor maximum depth 20 m.
- Data: 8 development scenes from official NYU train; 32 evaluation scenes from
  official NYU test; one frame per scene; no scene overlap.
- Fixed seed: 20261023. Scene selection occurs before predictions or evaluation.
- Conditions: original RGB plus warm, cool and green casts at mild and strong settings.
- Processing: none, Gray-World, White-Patch and fixed half-strength Gray-World.
- Color gains operate in linear RGB, preserving mean luminance before clipping.
- Evaluation: per-image AbsRel, RMSE(m), MAE(m), delta1; crop rows 45:471, columns 41:601;
  valid reference depth strictly between 0.1 and 10 m; prediction clipping [0.1,10] m.
- No prediction-specific scale or shift fitting, and no parameter selection on
  evaluation reference depth. Model weights remain frozen.
- Confidence intervals resample whole evaluation images, retaining all six
  casts together. One evaluation image comes from each distinct scene.

This pilot does not reproduce the complete 654-image NYU benchmark and does not
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
credentials, upstream source, checkpoints, raw RGB/depth data and large prediction
caches are excluded from GitHub. Generated research figures include attributed
dataset examples.
