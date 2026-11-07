# Paper-oriented research plan

7 November 2026. This is a research plan, not a claim that a publishable
contribution or state-of-the-art result has already been obtained.

## Working question

Can geometry predicted from one RGB image guide an explicit chromatic-invariance
mechanism that improves depth more than ordinary appearance-robust adaptation?

The prototype couples a frozen first-pass depth estimate, normalized virtual
point geometry, a relative-chromatic cue and bounded internal token residuals.
Its objectives separate depth accuracy, paired depth agreement and known-gain
chromatic equivariance. Ground-truth geometry is supervision/evaluation only.

## Evidence required for the proposed claims

| Candidate claim | Evidence needed |
|---|---|
| Color changes matter to geometry | Luminance-matched chromatic interventions on128new NYU scenes, alongside exposure-only changes; scene-wise effects and failures |
| Explicit chromatic handling helps beyond generic robustness | Equal data, training schedule and common depth losses; ordinary augmented, content and achromatic controls; original-image accuracy preserved |
| Predicted geometry adds something useful | Full versus flat-geometry comparison at the same active architecture and losses; no measured-depth input at inference |
| The chromatic loss contributes | Chromatic-with/without-equivariance comparison; measure the known relative-gain prediction error as well as depth |
| The result transfers to captured lighting | ReMID735real images as an external evaluation, with capture/group weighting and valid measured depth |
| Simulation is informative but not sufficient | Keep rendered Vari and digital NYU changes separate from captured real-light ReMID evidence |

## Prior-art distinction to establish

PCCC uses measured RGB-D/point clouds to estimate illumination. The proposed
student task remains RGB-only depth inference; its geometric context comes from
its own first-pass prediction. This different direction is useful motivation,
but does not automatically prove a novel method.

LumiDepth already addresses multi-illumination monocular depth using
disagreement-calibrated pseudo supervision and frequency-aware consistency.
FiLM, low-rank residuals, PointNet pooling, illuminant equivariance and
augmentation consistency also have established prior art. A paper must explain
why the particular geometry/chroma coupling is necessary, identify a method
distinction and substantiate it experimentally. Merely assembling these blocks
or renaming a known loss is insufficient.

The current run does not use calibrated absolute-illuminant supervision. Its
estimated log-ratios are a relative chromatic cue, not demonstrated physical
illuminant recovery. PCCC assets are acquired for a future carefully aligned
CC baseline; their source indexing, radiometry and sparse geometry must be
verified before combining them with dense depth.

## Before calling the project publication-ready

1. Finish and independently audit the locked run. Report every control and harm,
   including any result that contradicts the proposed coupling.
2. Explain any observed benefit over the strongest conditional control, not only
   the weaker frozen/ordinary baselines. Distinguish accuracy, stability and
   chromatic-specific effects.
3. Compare competitive depth/illumination methods using matched resolution,
   supervision, compute and splits. The224-pixel CPU adapter experiment is not
   directly comparable with full-resolution published benchmark scores.
4. Repeat the chosen development recipe with at least three independent training
   seeds and report training variation. Current paired scene/group intervals
   describe evaluation sampling, not seed uncertainty.
5. If the method changes after reading ReMID scores, treat those data as
   development evidence. Reserve another genuinely independent real dataset
   (for example appropriately grouped DIODE/ETH3D captures) for confirmation.
6. Verify a stronger real-data grouping where possible. Fifty ReMID captures and
   twenty-five conservative labels do not establish fifty independent rooms.
7. Obtain the full closest-paper methods/checkpoints through legitimate public
   releases or author/institutional access; do not claim a reproduction from an
   abstract alone. No author has been contacted automatically.

## Suggested paper structure

Introduction: the precise chromatic nuisance and why geometry may help.
Related work: CC, RGB-D illumination, robust monocular depth and invariance.
Method: RGB-only forward graph, geometry/chroma coupling and objective.
Experiments: scene-separated source training, controlled color/exposure probes,
rendered data and captured real-light evaluation.
Ablations: cue, geometry, loss, active capacity, resolution and compute.
Limitations: physical illumination labels, scene grouping, camera/FOV shift and
negative results. Conclude only what the measured controls support.
