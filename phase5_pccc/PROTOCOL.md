# DepthAWB paper-budget reproduction, CPU and Intel GPU

Shirsha Mitra / 2023521460130 / 9 November 2026.

## Scope

Reproduce the PCCC DepthAWB experiment at 256 points, all three official folds
and **10000 epochs per model**. This is one benchmark and resolution, not the
entire paper's datasets, 4096-point experiments or every competing algorithm.
The earlier 100-epoch pilot stays unchanged in `phase4_pccc/`.

All 185 genuine released captures are used. For each official fold, train on
the other two folds: train/test counts 123/62, 123/62 and 124/61. Full-resolution
sensor alignment is inherited from the released clouds, not reconstructed.
All folds had earlier development exposure through the pilot; this is a
benchmark reproduction rather than fresh independent confirmation.

## Architecture and controls

Pinned author revision: `ddc8770e04ef5509a621c5a7ee04e330b9204a93` at
https://github.com/xyxingx/Point-Cloud-Color-Constancy.

The author source tree remains unmodified. Execute its PointNet/STN and angular
training loss, fixing return unpacking and moving STN identity tensors to the
input device (needed for Intel XPU). Verify CPU outputs are bitwise unchanged.

The released AdaptiveAvgPool1d(256) receives exactly256points and is therefore
an identity. XPU has no deterministic implementation for its generic backward;
replace this redundant operation with Identity only in this256point setting.
Verify CPU outputs and gradients against the original pool, then verify exact
interrupted/resumed XPU training with deterministic algorithms enabled. The
earlier non-deterministic restart mismatch is an engineering finding, not a
new experimental result or a silently relaxed numerical tolerance.

Primary paper restoration: four output channels, point RGB plus a softmax
certainty channel, aggregated over points (paper Eq.4). Intensity SD0.15
follows paper Eq.6. Also retain the released three-channel sum / SD0.20 path
as a separately labelled source-code reference. This distinction is necessary:
the released head does not contain the described fourth certainty channel.
Comparing families is not an isolated head ablation because augmentation differs.

Within each family compare real native XYZ against the predeclared flat-range
control: divide valid XYZ by abs(Z), preserve RGB, camera rays and missing XYZ0.
This is not pure RGB-only and is not claimed to be an exact reconstruction of
the paper's full RGB-D no-depth preprocessing. Native normalized negative Z
is retained without inventing metric units.

## Training and checkpoint interpretation

- Adam, LR0.0003, original defaults, batch16, no new loss or LR schedule.
- Float32, no mixed precision. Seed20261109+fold, same initialization/schedule
  within real/flat pairs. Author Gaussian30-degree pose rotation.
- Fixed budget10000epochs, with full source-fold training data and no early stop.
- Every50epochs store optimizer, model, best-reference state and Python/NumPy/
  torch/order-generator RNG. Verify interrupted/resumed training matches an
  uninterrupted run on training-only preflight data.
- **Primary result:** fixed final epoch, independently scored from saved logits.
- **Author-code reference:** the released training entry point repeatedly scores
  its held-out fold and retains its best checkpoint. Reproduce this separately
  and label it **test-selected / optimistic**, never as unbiased performance.
- Held-out labels never enter optimizer updates. No architecture, budget or
  hyperparameter choice will be based on the accumulating held-out scores.
  Console progress shows training loss; held-out selection is logged for the
  predeclared source-reference diagnostic only.

The fixed-final and source-selected results must both be reported, including
failures. No claim that the original published accuracy has been obtained until
the completed protocol and metrics justify it. These checkpoint conventions,
device/runtime differences and restoration ambiguities limit exact equivalence.

## CPU and GPU

Hardware: Intel Core Ultra7 255H and Intel Arc140T. The original CPU environment
is preserved. Install matching PyTorch2.14.1+xpu in a separate environment;
do not replace the original study environment or alter the graphics driver
without first establishing a concrete need.

Benchmark both heads on the same original fold0 training captures, with warmup,
synchronization, gradient and real Adam-update checks. Compare initialized CPU
and GPU outputs within explicit tolerance, recording actual discrepancies.
Choose the main suite backend by successful execution and throughput, not
evaluation accuracy. CPU reference training can use a separate result folder;
do not silently resume a CPU optimizer on XPU or mix fold devices in a reported
out-of-fold series. A failed GPU check must be reported, with CPU work continuing.

The full suite has four variants x three folds =12models, one seed per fold.
On completion aggregate out-of-fold predictions, angular mean/median/trimean,
best/worst quartiles and95th percentile. Include sampled-RGB Gray-World and
training-fold-only mean-illuminant baselines. Independently check all scores,
source/data/checkpoint identities and saved-model prediction provenance.
