# Thesis project status and repository scope

Shirsha Mitra, student ID 2023521460130, Sichuan University.
Status checked on 24 November 2026.

The latest public files are indexed in [FRIEND_START_HERE.md](FRIEND_START_HERE.md).
The illustrated report now has four pages and four experimental figures. Official
Tables 1-3 have prefilled student sections and three pages of experimental outputs;
supervisor sections, actual dates and signatures remain outstanding. Fresh local
Intel-GPU inference reproduced 975 pilot predictions exactly. No new training or
Modal job was launched for the report update.

Thesis: **Design and Implementation of a Color-Constancy-Assisted Monocular
Depth Estimation System**.

## Completed work

### Audited fusion pilot and proposal defense preparation 23 November

All 21 ETH3D fusion-pilot models completed 100 epochs on the local Intel GPU.
The [results](phase9_reliability_fusion/pilot_v1/RESULTS.md) retain all seven
methods, five test conditions, three folds and both checkpoint rules. All 42
checkpoints were re-inferred; 13,650 predictions, 70 summary rows, 210 fold
summaries and both bootstrap intervals passed independent verification.
The [audit recovery note](phase9_reliability_fusion/AUDIT_RECOVERY.md) records
the contiguous-input-layout fix without retraining or altered numerical inputs.

Clean learned-gate mean is 2.5740 degrees, mask 2.5881 and matched PCCC 2.3177.
Both learned-minus-mask primary intervals include zero. Improvement, novelty,
full-budget repeated-seed evidence and independent new-scene validation remain
unestablished. No new cloud job was allocated.

The 25 November [defense materials](output/defense_2026_11_25/START_HERE.txt)
include a three-minute main talk, two backup slides, written proposal, bilingual
script/Q&A and a private meeting note excluded from Git. The proposal preserves
the registered title and explicitly identifies current RGB-D illuminant
estimation as a scope question for supervisor confirmation. It is prepared for
review; unconfirmed administrative and approval fields are not invented.

### Initial reliability-aware fusion implementation 16 November

The [phase-nine candidate](phase9_reliability_fusion/README.md) now implements
separate RGB/geometry branches, a learned input-derived gate, ordinary/mask/RGB
controls and a pinned original-PCCC adapter. The 34,020-parameter candidate
passed gradient, missing-depth fallback, permutation and checkpoint checks.
All 704 released input identities and split membership were checked. A synthetic
worker test passed optimizer/test separation and saved-model re-inference.
No full fusion benchmark or paid run has been launched. Performance improvement,
novelty and new data validation remained outstanding at this stage. This stage
addresses the teacher's proposed algorithm directly; phase eight remains a
separate scale-only preparation.

Hardware follow-up: [local Intel GPU checks](phase9_reliability_fusion/HARDWARE.md)
passed for all four new fusion modes and the complete synthetic worker. Local
GPU is preferred for modest prototype work under the user's latest instruction.
Original PCCC initially failed a strict CPU/GPU gradient comparison. Later
same-XPU repeatability checks passed; all pilot methods then used XPU while
retaining the unresolved CPU-gradient difference. No canceled job was restarted.

### Current continuation: ETH3D scale inputs prepared (16 November)

The [ETH3D follow-up](phase8_eth3d_scale/README.md) now has a separate frozen
protocol, 195 locally prepared median-depth-normalized clouds and an independent
input audit. RGB, labels, native axes and official folds are preserved. The
existing loader accepted every capture and all six reusable native/flat final
checkpoint hashes passed. Tiny subthreshold residuals were preserved explicitly.
This is input preparation, not a new trained experiment or accuracy result.
Three new models are planned; a bounded runner and new runtime/spending allowance
remain necessary before paid execution. No training or cloud job was launched.

### Latest: camera-frame and depth-scale diagnostic (16 November)

Completed nine new released-head PCCC models at 10,000 epochs on the official
NYU-v2 + DIODE folds. Six phase-six native/native-flat references were reused.
The independent audit passed: 3,240 saved predictions checked and all 15 final
checkpoints re-inferred. Read the [report](output/pdf/Shirsha_Mitra_PCCC_Depth_Format_Update_2026_11_16.pdf),
[measured results](phase7_depth_format/RESULTS.md),
[execution and recovery notes](phase7_depth_format/EXECUTION_NOTES.md), and
[audit](phase7_depth_format/results/nyu_frame_v1/audit.json).

Native mean angular error is 2.6135°; canonical axes give 2.5471°, canonical axes
plus per-capture median-depth normalization 2.7192°, and the matched canonical
flat control 2.8164°. Both primary paired intervals include zero, so neither
normalization nor relative-depth benefit is established by these comparisons.
One seed per fold and prior development exposure limit the evidence.
That diagnostic did not implement a reliability gate or new fusion architecture;
the subsequent local prototype is recorded above.

New training consumed 5.758 GPU-hours. The training-only resource planning
allowance is $4.67, excluding smoke/build/startup overhead; it is not an invoice.
The original approved cap was $8. The application stopped after completion and
reported zero tasks. No extra training or automatic retry is pending, and the
cancelled phase-five laptop jobs remain stopped.

### Full PCCC on all three datasets with a cloud GPU (10 November)

Completed on Modal NVIDIA T4 GPUs: 48 models at 10,000 epochs (about 22.5 GPU-hours) on DepthAWB,
ETH3D and NYU-v2 + DIODE, real 3D vs flat geometry, plus a DepthAWB depth-quality test. Real 3D
helps clearly only on ETH3D. See [phase6_modal/README.md](phase6_modal/README.md) and the
[report](output/pdf/Shirsha_Mitra_PCCC_GPU_Update_2026_11_10.pdf). One seed per fold; the
reliability-aware fusion in the report is a proposal, not an implemented result.

### Cancelled follow-up: paper-budget PCCC reproduction

Cancelled on9November at the user's explicit request because it takes too much
time. Both workers are stopped; nothing is scheduled to restart. Partial GPU
epoch6200 and CPU epoch1550 checkpoints are preserved. No completed 10,000-epoch
benchmark is claimed for this cancelled laptop study. The launch details below
are historical; the separately completed cloud studies are described above.

The DepthAWB256point, three-fold,10000epoch study was launched on9November.
Actual PCCC training, CPU source equivalence and exact restart checks passed on
both CPU and Intel Arc140T XPU. The GPU is the main suite backend; a separate
CPU fold0 reference was launched too. Check [live-run instructions](phase5_pccc/README.md)
and [cancellation status](phase5_pccc/STATUS.md). Launching is not completing
the experiment. No completion or publication worker remains pending for it.
Earlier completed studies remain below.

### Earlier PCCC color-constancy reproduction pilot

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
| Latest camera-frame/depth-scale protocol, code, compact results and audit | `phase7_depth_format/` (large checkpoints, clouds and full histories remain local) |
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

Start with [the latest PCCC results](phase7_depth_format/RESULTS.md),
[execution notes](phase7_depth_format/EXECUTION_NOTES.md),
[independent audit](phase7_depth_format/results/nyu_frame_v1/audit.json), and
[the output guide](output/START_HERE.md).
The [7 November report](output/pdf/Shirsha_Mitra_Weekly_Update_2026_11_07.pdf) covers the earlier depth study.
For exact commands and claim limits, read [the PCCC reproduction guide](phase4_pccc/README.md),
[the real-light depth study](phase3/README.md), [the prior follow-up README](phase2/README.md),
[the reproduction guard notes](phase2/audit/reproducibility_review.md), and
[the stereo instructions](phase2/stereo/README.md).

The original study remains separately documented in the root README. An existing
inference cache is not an independent rerun; use a new output directory with the
reproduction guard for a fresh run of the follow-up monocular study.
