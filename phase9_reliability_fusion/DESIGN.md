# Candidate depth-reliability-aware fusion

16 November 2026. Implemented research prototype, not a validated improvement
or a novelty claim. No paid training or paper result is established here.

## Task and architecture

Input: a released 256-point XYZRGB cloud. Output: one global illuminant RGB
vector. This is RGB-D-assisted illumination estimation, not depth-map prediction.

Two independent pointwise branches map RGB and XYZ to 64 channels each. For
each point, a learned scalar sigmoid gate scales the geometry feature before
adding it to the color feature. Global mean/max context and local fused features
feed a shared illumination head; positive pointwise predictions are averaged.

The geometry gate receives four input-only cues: validity abs(Z)>1e-8, capture
valid fraction, mean absolute neighboring log-depth difference, and RMS of that
difference. Use eight nearest valid camera rays, excluding self and including
all ties at the eighth distance to avoid point-order-dependent neighborhoods. These cues
need camera-coordinate Z but no image grid ordering. Detach cue construction
from XYZ gradients. Native XYZ scale and axes remain in the geometry branch;
the prior normalization experiment did not justify mandatory normalization.

Invalid geometry is zeroed only inside this new model; source files are never
edited. The learned gate is forced to zero on invalid points. If all points
lack valid geometry, its output equals this model's RGB branch with the same
weights. This is an architectural property, not proof of robust accuracy.

No illumination labels, test errors, corruption identifiers, clean reference
depth or ground-truth reliability enter the forward call. During training,
illumination labels supervise predictions in the loss, as usual. A large depth
discontinuity is not necessarily unreliable depth. Gate scores are learned
mixing weights, not calibrated uncertainty or ground-truth reliability.

## Implemented controls

`rgb`: color branch only. `ordinary`: ungated color + geometry. `mask`: explicit
validity-mask fusion. `learned`: input-derived gate times validity mask.
All four allocate identical modules and report active/allocated parameter counts;
inactive geometry/gate parameters are not a capacity match by themselves.
`pccc`: pinned authors' released-sum-head model through the existing portable
wrapper, using this worker's same data/splits/updates/loss for a matched rerun.
This has a different architecture/capacity, which must be reported.

The worker also supports clean-only training (`--damage-training` absent), depth
damage training (flag present), and consistency removal (`--consistency 0`).
No-damage and no-consistency ablations are separate training conditions, not
different inference inputs or new architectures.

## Objective and corruptions

L = (angular(clean prediction,label) + angular(damaged prediction,label))/2
    + 0.1 * squared L2(normalize(clean prediction)-normalize(damaged prediction)).

Angular loss is in degrees with the source cosine clamp +/-0.999999. The
consistency weight is an initial prototype setting, not a tuned best value.
Every matched control uses the same weight and paired supervision unless it is
the declared no-consistency ablation. A gate-specific benefit must exceed the
mask/ordinary controls with the same damage training and objective.

Training damage samples uniformly among clean, positive lognormal depth noise
(sigma 0.1 or 0.3, mean multiplier 1), removal of half the valid XYZ points, and
XYZ/RGB correspondence shuffle. RGB and labels are untouched. This is synthetic
depth damage, not a measured sensor-failure dataset. Lognormal noise differs
from phase-six Gaussian noise: do not merge those accuracy tables or present
this as exact reproduction of phase-six corruption training. The worker applies
no RGB gain or XYZ rotation augmentation in this first prototype; matched
original PCCC must be rerun under this same policy for a direct comparison.

## Splits and evaluation before a funded experiment

Use the official outer folds. Select an inner validation fifth from the official
training IDs by sorting SHA256('20261116/'+id). The remaining examples train.
This is capture-level splitting: room/view independence is not established.
Every released benchmark has previous development exposure. Do not describe a
new checkpoint on these same folds as independent confirmation.

Fixed-final checkpoint is primary; inner-validation-selected checkpoint is
secondary. Outer-test scoring occurs only after training and selection finish.
Separate order/damage RNG streams keep schedules matched across modes. Record
all sample IDs, source hashes, capacity, optimizer settings and checkpoint IDs.
An existing output directory is refused rather than silently resumed.

The local worker is implemented, but the full experimental protocol is not yet
frozen: model/epoch/seed counts, primary paired contrasts, corruption test
manifests, independent angle/checkpoint audit and timing/cost bounds must be
specified before paid execution. Report all fixed clean/damaged tests, paired
conditional uncertainty and training-seed variation. Preserve the historical
PCCC results; they are context until the matched rerun is complete.

The smoke test uses only official training captures and checks implementation,
not angular-error improvement. New training needs a separate explicit budget.
The earlier $8 was for the completed phase-seven experiment.

Teacher alignment: this implements a first fusion candidate on available sparse
real captures. It does not complete the requested performance evidence, simulation
extension, independent real-data validation, or thesis proposal. Relevant CC and
multimodal fusion literature still needs review before any novelty statement.
