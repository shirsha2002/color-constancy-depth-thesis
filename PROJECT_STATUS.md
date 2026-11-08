# Thesis project status and repository scope

Shirsha Mitra, student ID 2023521460130, Sichuan University.
Status checked on 9 November 2026.

Thesis: **Design and Implementation of a Color-Constancy-Assisted Monocular
Depth Estimation System**.

## Completed work

### Latest direction: genuine PCCC color-constancy reproduction

The supervisor's latest direction centers **RGB-D / point-cloud information
improving illumination estimation**. It is a different endpoint from the earlier
depth-adaptation study, whose results are preserved below.

The small CPU pilot uses the authors' pinned PCCC model and genuine DepthAWB
inputs: 185 released samples with 256 XYZRGB points and illuminant labels.
It fixes official fold 0 as 62-sample test and reserves 25 inner-validation
samples from the remaining data, leaving 98 for training. Four variants and
three seeds test native versus flat geometry under the released sum head and
an explicitly restored paper-weighted head. Each run uses 100 epochs.

The exact execution fixes, source/paper differences, frozen configuration and
reduced-budget limits are in [the PCCC protocol](phase4_pccc/PROTOCOL.md).
Read [the measured results](phase4_pccc/RESULTS.md) and
[the independent audit](phase4_pccc/results/cpu_v1/independent_audit.json).
The original published accuracy is not claimed to be reproduced.

### 1. Reproducible laptop environment and original pilot

- Isolated Python environment, official pinned Depth Anything V2 metric source
  and checkpoint, VS Code workspace, setup commands and environment checks.
- NYU Depth V2 subset: 8 development and 32 evaluation scenes, with source
  identities, official split membership, reference depth and content hashes.
- Seven image conditions and four correction choices: 1,120 numerical depth
  predictions across development and evaluation; paired metric tables, analysis,
  limitations and a fixed correction-strength ablation.
- Original progress report, editable PowerPoint, slide PDF, presentation notes
  and English QQ update. These cover the initial preprocessing study.

### 2. Follow-up to supervisor feedback

- Acquired and verified 528 Vari images in 48 fixed-camera groups across eleven
  lights and two rooms, plus shared reference depths. The released TEST archive
  is repartitioned for an explicitly exploratory study; it is not the official
  training benchmark.
- Acquired 42 DTU images from two scenes, three views and seven real lights,
  original calibrated cameras and structured-light references. Selected native
  files and official/mirror archive member identities were verified.
- Implemented a 101,504-parameter module between the frozen encoder and depth
  decoder. Four learned variants test ordinary adaptation, achromatic cues,
  chromatic cues and consistency supervision; two frozen-model baselines are
  retained.
- Compared six methods on 264 evaluation images, saving 1,584 predictions and
  144 camera-group lighting-stability scores. All prediction metrics were
  independently recomputed. Data, source hashes, adapter weights and checkpoint
  selection were checked; the legacy depth decoder preserves the main ranking.
- Ran calibrated stereo for 42 view/light cases with five costs, giving 210
  independently checked score rows. Numerical cost validation, geometry checks,
  depth/error plots and small 3D previews are retained.
- Wrote the follow-up research report, next-dataset plan and short English
  professor reply. Added a guard for fresh reproductions without stale caches.

### 3. Real-light data and explicit chromatic controls

- Reviewed the supervisor-mentioned **Point Cloud Color Constancy (CVPR 2022)**
  and close prior work **LumiDepth (ECCV 2026)**. PCCC estimates illumination from
  measured RGB-D; our model remains RGB-only at inference.
- Acquired the complete public ReMID release: 735 captured real-light RGBs,
  50 fixed-capture groups and aligned measured depth in millimeters. All ReMID
  images are external to training and checkpoint selection. Twenty-five
  conservative labels do not prove independent physical rooms.
- Acquired 368 new NYU RGB-D scenes, frozen as 200 training / 40 validation /
  128 evaluation, excluding all forty original pilot scenes. Digital color and
  exposure changes are simulated illumination on real photographs.
- Acquired PCCC labels, original folds and sampled XYZRGB clouds for future
  auxiliary baselines. They are not used as dense-depth training or evaluation.
- Trained seven global and eight local variants with common source data,
  schedules and validation rules. The local adapter has 42,036 trainable
  parameters; predicted first-pass depth guides internal relative-chroma fields.
- Evaluated sixteen unique methods across 640 NYU input conditions, 735 ReMID
  images and 264 previously inspected rendered Vari images: 26,224 unique rows,
  plus a separate 640-prediction floating-point intervention probe. All saved
  metrics are independently checked against native prediction arrays.
- Recorded the validation-selected recipe before reading final-test scores.
  Both the fixed full-loss hypothesis and the validation-selected no-equivariance
  recipe remain visible, alongside all matched controls and failure cases.
- Prepared the weekly PDF, measured results, source manifests, small trained
  adapters, analysis figures and a short English professor reply. Read
  [the results](phase3/RESULTS.md) before making any performance claim.

## What the earlier results establish

Lighting can change geometry estimates in the tested real DTU scene. Basic RGB
matching changes from 13.19 mm to 54.21 mm mean absolute depth error under changed
lighting. Internal scalar-gain compensation reduces changed-light error to
29.90 mm; separate channel gains do not beat that control.

On the unseen Vari room, the chromatic consistency module has AbsRel 0.4277
versus 0.3988 for ordinary task adaptation, with worse lighting stability.
**The present experiments do not establish a CC-specific benefit or a novel
algorithm contribution.** They provide working implementations, data and
controlled evidence to guide the next experiment.

The paired-light learning study has only two rendered rooms, with one evaluation
room. The real stereo diagnostic has one held-out scene, and its lights mainly
change direction/shading. Neither supports broad generalization or a claim of
physically accurate illuminant recovery.

## Repository scope

| Included on GitHub | Location |
|---|---|
| Original experiment code and checks | `scripts/`, `tests/`, `.vscode/` |
| Small PCCC CPU reproduction, source wrapper, data identities and CC results | `phase4_pccc/` |
| Follow-up code, frozen protocols and audit tools | `phase2/mono/`, `phase2/stereo/`, `phase2/audit/`, `phase2/PROTOCOL.md` |
| Expanded real-data study, protocols, learning and independent analysis | `phase3/README.md`, `phase3/RESULTS.md`, `phase3/learning/`, `phase3/data_tools/`, `phase3/analyze.py` |
| Setup, source revisions, package versions and data acquisition | `scripts/setup.ps1`, `requirements-lock.txt`, `phase2/scripts/`, `phase2/vari_research/` |
| Selected data identities, splits, hashes and provenance | `data/nyu_pilot/manifest.json`, `phase2/data/vari/manifest.json`, `phase2/data/dtu/manifest.json` and linked documentation |
| Numerical metric tables, training histories and independent audit records | `results/`, `phase2/results/mono/`, `phase2/results/stereo/main_v2/` |
| Four small trained feature adapters | `phase2/results/mono/models/` |
| Fifteen small phase-three adapters and complete metrics/audits | `phase3/results/study_v1/models/`, `phase3/results/spatial_v1/models/`, `phase3/results/spatial_controls/models/`, `phase3/results/combined/` |
| Real-light / new NYU / auxiliary PCCC identities and provenance | `phase3/data/remid/`, `phase3/data/nyu_rgbd/`, `phase3/data/pccc/` (manifests and documentation only) |
| Research figures and deterministic small 3D output previews | `report/`, `phase2/results/mono/figures/`, `phase2/results/stereo/main_v2/figures/`, `phase2/results/stereo/main_v2/pointclouds_inference/` |
| Final PDF reports, original PPTX/slides, notes and QQ text | `output/` |
| Authoring source for the reports and original slides | `report/`, `presentation/`, `phase2/build_update.py` |
| Data limitations, attribution and next research decision | `THIRD_PARTY.md`, `phase2/README.md`, `phase2/research/DATASET_NEXT_STEPS.md` |

The following remain on the laptop rather than in Git history: downloaded raw
datasets and archives, the pretrained base checkpoint, upstream source clones,
installed environments and packages, large feature/prediction arrays, full
point clouds, render/build intermediates and superseded diagnostic caches.
Credentials and temporary signed download responses are excluded.

These excluded files are not represented as uploaded. Raw inputs can be
reacquired with the checked-in setup/acquisition scripts; frozen model and data
identities are recorded in manifests. Predictions and full point clouds can be
regenerated using the documented experiment commands. This GitHub repository is
a reproducible research project, not a byte-for-byte backup of the laptop.

University reference templates in the parent workspace are source materials,
not authored research outputs, and are not added to this public code repository.

## Read or reproduce

Start with [the PCCC results](phase4_pccc/RESULTS.md) and [the output guide](output/START_HERE.md).
The [7 November report](output/pdf/Shirsha_Mitra_Weekly_Update_2026_11_07.pdf) covers the earlier depth study.
For exact commands and claim limits, read [the PCCC reproduction guide](phase4_pccc/README.md),
[the real-light depth study](phase3/README.md), [the prior follow-up README](phase2/README.md),
[the reproduction guard notes](phase2/audit/reproducibility_review.md), and
[the stereo instructions](phase2/stereo/README.md).

The original study remains separately documented in the root README. An existing
inference cache is not an independent rerun; use a new output directory with the
reproduction guard for a fresh run of the follow-up monocular study.
