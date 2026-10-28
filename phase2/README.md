# Phase 2: lighting, integrated features, and geometry

Shirsha Mitra, 2023521460130, Sichuan University. Completed pilot recorded on
28-29 October 2026. The earlier NYU preprocessing experiment remains unchanged.

The supervisor requested relevant datasets, evidence that lighting affects
geometry, and a method that incorporates color handling within the geometry task.
This phase implements that sequence. **The CC-inspired monocular module runs,
but it has not established an advantage over ordinary task adaptation.**

## Acquired data

| Dataset | Selected data | Split and purpose |
|---|---|---|
| Vari, official released test archive | 48 fixed-camera groups, 11 rendered lights per group, 528 RGBs, 48 shared depth references | Room 212: 20 training groups / 220 RGBs, 4 validation groups / 44 RGBs. Room 485: 24 evaluation groups / 264 RGBs. |
| DTU original SampleSet | Scenes 1 and 6, views 25/26/35, all 7 captured lights: 42 RGBs; original cameras and structured-light point clouds; 130 native files | Scene 1 development, scene 6 held-out diagnostic. Fixed diffuse reference light 3, other views at lights 0-6. |

Vari's public archive contains only two original rooms. This is a new exploratory
repartition of released **test** assets, not an official benchmark training split.
The authors still list the official training release as forthcoming. An unseen
room supplies 24 correlated view groups, not 24 independent scenes. Light names
do not provide calibrated illuminant RGB ground truth.

The primary Vari reference preserves the native 16-bit TIFF values using
`(1 - uint16/65535) * 10` meters. The authors' public PIL loader reduces precision
to 8 bits; both conventions are retained. This explicit decoder refinement means
these scores must not be compared directly with published benchmark scores.
The main method ranking is unchanged when independently rescored with the legacy
8-bit convention.

DTU supplies real lighting differences, primarily direction, shading and
reflections. It does not isolate chromatic illumination. Its point-cloud units
are millimeters; its reference depth maps are derived by camera projection with
missing pixels retained. These low-resolution depth scores are not the official
full DTU 3D reconstruction benchmark.

Sources, acquisition details and limitations:
[Vari](vari_research/README.md),
[DTU](data/dtu/DATA_SOURCES.md),
[next dataset steps](research/DATASET_NEXT_STEPS.md).

## Integrated monocular experiment

The original Depth Anything V2 Metric Hypersim Small encoder and decoder stay
frozen. A 101,504-parameter, identity-initialized feature modulation module sits
**between the encoder and depth decoder**. It uses relative log-channel means as
a color cue, predicts bounded feature gains/shifts, and is trained with depth
supervision and paired-lighting consistency. It does not produce a corrected
input image. The cue is a Gray-World-inspired proxy, not a calibrated illuminant
estimate; feature modulation itself is established prior art.

Four learned controls share module capacity, 150 update steps, the same training
pairs and the same frozen base. Validation checkpoint selection uses only room
212 views. Input size is 224 with preserved aspect ratio, seed 20261028, eight CPU
threads. There is no fitted prediction scale or shift. Read the premeasurement
[protocol](PROTOCOL.md) for losses, masks and the decoder amendment.

Results on room 485, 264 images / 24 fixed-camera groups:

| Method | AbsRel down | RMSE (m) down | delta1 up | Lighting MAD down |
|---|---:|---:|---:|---:|
| Frozen model | 1.749718 | 2.981941 | 0.002284 | 0.102475 |
| Gray-World preprocessing | 1.782727 | 3.049256 | 0.002129 | 0.117123 |
| Ordinary task adapter | **0.398801** | **0.706782** | 0.255943 | **0.061085** |
| Achromatic cue + consistency | 0.435963 | 0.782173 | 0.240302 | 0.099080 |
| Chromatic cue, supervision only | 0.434747 | 0.770324 | 0.291804 | 0.163718 |
| Chromatic cue + consistency | 0.427686 | 0.758361 | **0.297049** | 0.158748 |

Lighting MAD is the average absolute deviation across the 11 predictions at one
fixed camera, normalized by reference depth, then averaged over groups. It is
dimensionless; it is not MSE, and stable predictions can still be inaccurate.

The full CC-inspired model improves over the frozen model, but the ordinary task
adapter has lower AbsRel, RMSE and lighting MAD. CC + consistency has **7.24% worse
AbsRel and 2.60 times the lighting MAD** of that control, with a different delta1
tradeoff. Improvements over the frozen baseline cannot therefore be attributed
to color constancy. Domain adaptation and depth-scale bias are major confounds.
The current global color cue also mixes object reflectance with illumination.

The [independent audit](results/mono/independent_audit.json) recomputed every
metric from all 1,584 saved predictions, all 144 lighting-group scores, all source
and data hashes, the four adapter checkpoints, and their 44-image validation
scores. Numerical metrics agree with saved results; legacy-decoder rescoring
preserves the ranking. Numerical reproducibility does not remove the two-room
generalization limitation.

## Calibrated stereo diagnostic

The five costs are RGB SSD, bounded scalar gain, bounded per-channel gains,
grayscale ZNCC and per-channel ZNCC. Gains are optimized within each depth
hypothesis, so radiometric handling participates in geometry estimation.
Scalar and normalized-correlation controls distinguish exposure/shading robustness
from a color-specific benefit. These are implementations of known matching ideas,
not claims of a novel stereo algorithm.

Use only [main_v2 results](results/stereo/main_v2/summary.csv). Earlier attempts
used float32 photometric moments; numerical validation detected variance
cancellation in saturated patches. Final moments are float64 and were checked
against independent formulas. The final batch has 42 view/light cases and five
methods per case. Consult [the stereo protocol and commands](stereo/README.md)
and [design audit](research/stereo_design_audit.md) before interpreting its scores.

## Reproduce on this laptop

Open `ColorConstancyDepth.code-workspace` in VS Code and use the existing isolated
`.venv` interpreter. It is a CPU run; a CUDA GPU is not required. Run from the
project root, `D:\thesis work\thesis code`.

```powershell
# Download only selected Vari members, or audit the existing frozen manifest.
powershell -NoProfile -ExecutionPolicy Bypass -File .\phase2\vari_research\acquire_selected_vari.ps1

# Download selected DTU originals; verify the mirror's members against the official ZIP.
& .\.venv\Scripts\python.exe phase2/scripts/download_dtu.py --stage full --transport verified-mirror

# Check frozen inputs without inference, then audit the saved predictions.
& .\.venv\Scripts\python.exe phase2/audit/reproduce_guard.py
& .\.venv\Scripts\python.exe phase2/audit/audit_mono.py
& .\.venv\Scripts\python.exe phase2/analyze_mono.py

# Full fresh reproduction: destination must not already exist.
& .\.venv\Scripts\python.exe phase2/audit/reproduce_guard.py --run-fresh --out phase2/results/mono_reproduction_01

# Final stereo configuration and analysis.
& .\.venv\Scripts\python.exe phase2/stereo/run_experiment.py --scenes 1 --refs 25 --sources 26 --lights 3,0 --out calibration_pair_v2
& .\.venv\Scripts\python.exe phase2/stereo/validate.py --pair-run calibration_pair_v2
& .\.venv\Scripts\python.exe phase2/stereo/run_experiment.py --scenes 1,6 --out main_v2
& .\.venv\Scripts\python.exe phase2/stereo/analyze.py --run main_v2
& .\.venv\Scripts\python.exe phase2/stereo/finalize.py
```

For a fresh checkout, the root `scripts/setup.ps1` installs the pinned model
environment and acquires the original baseline assets. Vari's wrapper uses a
separate pinned acquisition environment. Base model weights and upstream code
must match `results/mono/source_snapshot.json`; results use the official model
revision `a561b849ebae10a6f5ef49e26c83cbbcd36c71bf` and the checkpoint digest in
`results/mono/config.json`.

Keep this numerical snapshot unchanged. The original monocular runner has a
configuration fingerprint, but its cache reuse does not cover every possible
source/image mutation. Use the [reproduction guard](audit/reproducibility_review.md)
and a new destination for a fresh run; do not combine modified code/data with
these saved predictions. The guard verifies the frozen sources and inputs and
refuses existing output directories. The independent audit is read-only with respect to
data, sources and predictions and writes only audit evidence. Re-auditing a fresh
training run additionally requires its local prediction/feature caches; those
large arrays are intentionally not versioned.

## Repository contents and next decision

Code, frozen protocols, public asset manifests, numerical tables, audit evidence,
figures and the four small learned adapters are retained. Raw data, large feature
and prediction caches, environments and base weights remain local. Source asset
rights remain with their authors.

The next research decision is to separate chromatic effects from achromatic
effects on more independent scenes, then improve the internal cue or objective
only if those controls support a CC-specific problem. Keep all baseline results,
including harms, and select settings on development scenes before a new held-out
evaluation. The [dataset plan](research/DATASET_NEXT_STEPS.md) states concrete
acceptance checks and which candidate data are actually available.
