# PCCC camera-frame and depth-scale experiment

Shirsha Mitra, 2023521460130. Predeclared 16 November 2026, before training.

## Question and scope

Does consistent camera orientation and per-capture relative-depth scaling improve
global illuminant estimation on the released NYU-v2 + DIODE point clouds?
This tests a data-representation hypothesis, not a new fusion algorithm.

All 324 real released captures, 256 XYZRGB points, official three folds. Original
phase-six results remain immutable and are reused as native and native-flat
references. No physical camera calibration, units, or dense-image reconstruction
is inferred from sparse coordinates. Prior test results motivated this study;
these folds are development/benchmark evidence, not new independent confirmation.

## Frozen transformations (RGB and illumination labels never change)

The 16x16 sampling order supplies image columns and rows. On valid points,
measure the sign of median Z and the correlations of X/Z with column and Y/Z
with row. Apply the corresponding diagonal axis rotation so canonical X points
right, Y down and Z forward. Verify a proper rotation (determinant +1). In the
current release this leaves all 258 NYU clouds unchanged and rotates all 66
DIODE clouds by diag(1,-1,-1). This is an observed convention standardization,
not proof that either source's original coordinates were erroneous.

Three newly trained conditions:

1. **camera_real:** canonical camera axes, original per-capture scale.
2. **camera_unitdepth:** canonical axes, divide valid XYZ by that capture's
   median absolute valid Z. Relative depth, rays, RGB and invalid-XYZ mask remain.
   This intentionally discards absolute scene-scale cues and is not metric depth.
3. **camera_flat:** canonical axes, divide each valid XYZ by its own positive Z.
   This retains canonical image rays and missingness but removes relative depth.
   It is a ray-and-RGB control, not a pure RGB-only architecture.

Zero geometry remains zero. Orientation and scale use only input geometry/grid,
never illumination labels or prediction errors. All samples and transformations
are frozen with byte hashes before cloud training. No post-test transform search.

## Model and training

Use the frozen phase-five trainer/model and pinned author source revision
ddc8770e04ef5509a621c5a7ee04e330b9204a93. **Released three-channel sum head only**,
original scalar augmentation SD0.20, Adam LR0.0003, batch16, float32,
10000epochs, deterministic CUDA, PyTorch2.5.1, T4. Same seed20261109+fold and
sample schedule as the corresponding phase-six baseline. Nine new models.
The weighted head and a new gate are outside this experiment.

Same runtime-only NumPy RNG serialization accommodation as phase six. Record
runner hashes and actual threads separately. No change to frozen numerical source.
No CPU/XPU partial study is resumed.

Primary checkpoint: fixed epoch10000. An author-style best-test-fold checkpoint
is retained only as a separately labelled optimistic reference, not used to
select conditions or the primary claim. Test labels never enter the optimizer.
No automatic hyperparameter or budget adaptation based on held-out outcomes.

## Endpoints and interpretation

Recovery angular degrees, lower better; preserve original cosine clamp0.999999.
Pool each capture's out-of-fold prediction exactly once. Retain all vectors,
labels, selected epochs and per-capture errors; independently recompute with
atan2 and verify checkpoint identities/prediction provenance on saved models.

Primary paired contrasts: camera_unitdepth minus native and camera_unitdepth
minus camera_flat. Secondary: camera_real minus native, camera_unitdepth minus
camera_real, and camera_flat minus native_flat. Report all, with mean/median,
quartiles and conditional capture-bootstrap95% intervals. Report NYU and DIODE
subgroups descriptively. Intervals do not measure training-seed or room-level
uncertainty; two primary comparisons are not multiplicity-adjusted. Use one
predeclared seed/fold; additional seeds are future work.

An improvement would support this particular representation change. It would
not prove which sensor convention caused prior failure, guarantee a reliability
gate benefit, establish novelty, or demonstrate generalization to new rooms.

## Budget and execution

User explicitly approved **US$8 maximum for this new experiment**, including a
small timing check. Use T4 only; no expensive GPU fallback, subscription change,
automatic retries, parameter sweeps or additional seeds. Reuse old baselines.

Current public Modal list rates checked16November2026: T4 $0.000164/sec,
CPU $0.0000131/physical-core/sec, RAM $0.00000222/GiB/sec.
Source: https://modal.com/pricing . Up to4CPU cores and4GiB are bounded per
container, timeout3000seconds/model, nine model invocations, one300second timing
invocation. Compute upper planning allowance: $6.15014; reserve$1.50 for build,
startup, idle, storage/network and uncertainty, total$7.65014. This is a resource
planning bound, not a provider invoice or an account-wide spending cap.

Require explicit --confirm-budget 8 for cloud entrypoint. Smoke test first using
training data only; stop if it fails or projected budget does not fit. No
automatic rerun of failed/timeout/preempted training. Record all launch/complete
metadata. Local audit and reports do not allocate cloud resources. Keep raw
data, transformed clouds, checkpoints and full histories out of GitHub.

## Files

prepare.py: immutable inputs and geometric checks; run_modal.py: bounded T4
execution; analyze.py: complete-run scoring/audit; inputs/*/manifest.json:
per-sample hashes; preflight.json and budget.json: preparation evidence.

