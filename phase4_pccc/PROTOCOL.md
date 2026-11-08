# Small CPU PCCC reproduction protocol

Client date: 9 November 2026. Shirsha Mitra, 2023521460130.

## Scope fixed before training and test scoring

Use the original authors' source and genuine released DepthAWB colored clouds
and illumination labels. This is an **adapted reproduction pilot**, not a claim
to reproduce the complete CVPR paper benchmark or its published accuracy.

- Upstream: https://github.com/xyxingx/Point-Cloud-Color-Constancy
- Pinned revision: `ddc8770e04ef5509a621c5a7ee04e330b9204a93`.
- All 185 released samples, each with 256 XYZRGB points (the 16x16 setting).
- Official fold 0 is the final 62-sample test set. Of the remaining 123 samples,
  25 deterministic samples form inner validation and 98 are training.
- Physical room identities are unknown. Official fold membership does not prove
  room-separated generalization. No exact duplicate cloud occurs in the data.
- Native normalized negative-Z geometry and zero-geometry points are retained.
  Do not label these coordinates as measured metric distances. The supplied
  colored clouds already pair XYZ/RGB; full sensor alignment is not recreated.
- Final test labels are read for integrity checks, then excluded from training,
  validation, checkpoint selection and development decisions.

## Author source and necessary distinctions

The original clone remains unmodified and is not redistributed as project code.
The wrapper fixes a two-versus-three-value return mismatch in memory. Author
PointNet, STN, point-wise head, angular training loss and RotateZ are executed
directly. Hard-coded file paths, unconditional CUDA and ignored dataset argument
in the original training entry point are replaced by an explicit CPU runner.

The released code has a **three-channel head and sums point outputs**. Paper
Section 3.4 / Eq.4 describes **four channels: three illumination values plus a
certainty logit**, with softmax over points and a weighted sum. These are not
equivalent. We run both named families; no silent architecture correction.

| Predeclared variant | Head | Geometry | Intensity scalar SD |
|---|---|---|---:|
| released_real | Author three-channel sum | Native XYZ | 0.20, released loader |
| released_flat | Author three-channel sum | Constant-range control | 0.20 |
| weighted_real | Paper four-channel weighted restoration | Native XYZ | 0.15, paper Eq.6 |
| weighted_flat | Paper four-channel weighted restoration | Constant-range control | 0.15 |

Head families also differ in intensity augmentation. Their difference is **not
an isolated head ablation**. Within each family, geometry is the only intended
change. The restored weighted variant reuses the author's backbone and head
layers; its final convolution and aggregation are explicitly changed. This
does not certify equivalence to the authors' unpublished training implementation.

For the flat control divide valid native XYZ by abs(Z), yielding Z=-1 and the
same camera rays. Keep missing XYZ=0 and retain RGB bitwise. Apply augmentation
after this transformation. This preserves two-dimensional layout/missingness;
it must not be called a pure RGB-only baseline or exact original-depth rebuild.

## Training and selection

- CPU, 8 intra-op threads, one inter-op thread. A discarded training-only timing
  probe compared 2/4/8 threads before fixing the runtime setting.
- Adam, learning rate 0.0003, original defaults, batch 16, 100 epochs.
- Seeds 20261109 / 20261110 / 20261111 for every variant: 12 trained models.
- Same initial seed and sample schedule within each real/flat pair.
- Original RotateZ with Gaussian 30-degree perturbation; only training is
  augmented. No synthetic illumination is added to the real test set.
- Select each checkpoint by lowest mean **inner validation** angular loss,
  checking every epoch. Record all validation histories. No outer-test tuning.
- All twelve models must finish before running the separate test stage.

The paper states 10,000 epochs and reports broader evaluation. A 100-epoch,
single-outer-fold CPU pilot cannot be compared as a matched published benchmark.
The original entry point repeatedly selects on its held-out fold; our separate
inner validation intentionally prevents final-test checkpoint selection.

## Measurements and reporting

Recovery angular error in degrees, lower is better. Retain the original
cosine clamp +/-0.999999 (minimum about0.081 degrees). Report mean, median,
trimean, best/worst25% and95th percentile. Save every test prediction and label.
Independently recompute angular scores using an atan2 formulation, checking the
same clamp. Re-run saved models to verify prediction provenance.

Include Gray-World on the original sampled linear RGB, and a constant illuminant
estimated from training labels only. Report every variant and seed; do not
choose a final winner from test scores. Pair real/flat differences by test
capture, with training-seed variation shown separately. Capture intervals do
not establish room independence or eliminate all uncertainty.

Future longer runs or algorithm changes require a new result directory and an
explicit development protocol. Do not adjust this frozen run after seeing test
outcomes. This pilot estimates global illumination; it is not a multi-light
local-illumination benchmark or a novel algorithm claim.
