# ETH3D fusion accuracy pilot

Predeclared 16 November 2026 before new accuracy training or scoring. This is
a 100-epoch development pilot, not the full paper-budget experiment. No cloud
allocation is planned. Use the existing Intel Arc GPU in the isolated XPU
environment; do not resume canceled phase-five jobs.

## Question and fixed scope

Does learned geometry gating improve illuminant recovery over otherwise
matched mask and ordinary fusion controls on the released ETH3D captures?

195 labeled 256-point XYZRGB clouds, official three outer folds, one seed per
fold (20261116+fold). Each outer fold has 130 development / 65 held-out captures.
The existing hash-based inner rule reserves 26 development captures for
validation, leaving 104 for optimizer updates. Preserve all IDs, native scale,
axes, RGB, labels and file identities. These captures were previously inspected
and are development evidence. Related scene/view identities are not established.

Seven conditions, three folds each, 21 new models:

| Condition | Architecture | Depth damage training | Consistency weight |
|---|---|---|---:|
| rgb | RGB branch control | yes | 0.1 |
| ordinary | ordinary fusion | yes | 0.1 |
| mask | validity-mask fusion | yes | 0.1 |
| learned | input-derived geometry gate | yes | 0.1 |
| learned_clean | same learned gate | no | 0.1 |
| learned_no_consistency | same learned gate | yes | 0 |
| pccc | pinned released PCCC sum head | yes | 0.1 |

Use 100 epochs, batch 16, Adam LR 0.0003, float32, deterministic algorithms,
and the same sample-order/damage RNG streams per fold. No RGB gain or XYZ
rotation augmentation. Primary comparisons isolate gating within the new
architecture; PCCC has different capacity and output architecture and must be
identified as such. Its earlier 10,000-epoch results are historical context and
must not be treated as this pilot's matched controls.

The original PCCC failed a strict CPU/XPU gradient check. Keep that failure.
Separate training-only evidence verifies ten finite updates and two exactly
repeated XPU histories/final states, with initial CPU/XPU prediction disagreement
below 0.02 degrees. All pilot methods therefore use the same XPU backend;
this is not a claim of CPU-identical gradients. Require the recorded model and
repeatability-checker hashes before a PCCC run. Record this limitation.

## Selection and fixed test conditions

Fixed final epoch 100 is primary. Best inner-validation mean checkpoint is
secondary, earliest tie. No outer-test labels in optimizer, model selection,
loss-weight selection, corruption selection or stopping decisions.

Prepare five test inputs per capture before training: clean, positive lognormal
depth noise sigma 0.1, positive lognormal noise sigma 0.3, removal of half valid
geometry, and XYZ shuffling. Use data.py's exact transformations. Seed each
capture/condition with default_rng([20261117, crc32('ETH3D/id/condition')]).
Freeze byte hashes of all derived test arrays. RGB and labels never change.
These are synthetic depth interventions on real captures, not measured sensor
failures or a rendered-scene simulation study. No test transform search.

## Endpoints and claim rules

Recovery angular error, degrees, float64 scoring with cosine clamp +/-0.999999.
Pool each capture's out-of-fold prediction once per condition/checkpoint rule.
Report mean, median and worst-quarter mean for every method and test condition,
both checkpoint rules, and per-fold results. Retain all prediction/label vectors.

Two primary fixed-final contrasts:

1. learned minus mask on clean inputs;
2. learned minus mask on each capture's average of the four damaged-input errors.

Negative favors learned gating. Use 10,000 paired capture-bootstrap replicates,
NumPy default_rng(20261118), in the stated order, percentile 2.5/97.5 intervals.
No multiplicity adjustment. These intervals are conditional on the trained
models/captures and do not measure seed or room uncertainty. Report both
regardless of outcome. Ordinary/PCCC/RGB and both ablations are secondary
comparisons; show their complete tables and harms. No post-test tuning or
extra runs to obtain a favorable result. Gate weights are not calibrated
reliability probabilities by construction.

Audit all 21 completion/config/checkpoint/source identities and all 100-epoch
histories, selection epochs and sample IDs. Re-infer saved final and validation
checkpoints and independently recompute angles with atan2. No accuracy claim
before audit passes. Preserve earlier results and raw inputs.

## Execution bounds and follow-up

Run serially, one local GPU worker at a time, maximum 900 seconds per model and
5,400 seconds for the suite. No automatic retry, hyperparameter sweep, larger
model or cloud fallback. Reuse only already complete jobs whose identities
verify; an incomplete existing directory requires review. A failure stops the
suite. Runtime bounds are safeguards, not completion promises.

This pilot can provide preliminary findings for a proposal. It cannot complete
a publishable novelty/generalization claim. Full-budget schedules, repeated
seeds, new real data and genuine simulated-scene validation remain future
research even if the pilot improves. After recording the outcomes, prepare the
proposal using the teacher's direction and the university's written form.
