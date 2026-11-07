# Independent review: real-data splits and evaluation

Reviewed 7 November 2026, before inspecting any phase-three model score. This is
a review brief for the implementation owner, not a claim that every proposed
experiment has run. The executed configuration and hashed manifests must state
the final numeric choices before test scoring.

## Split decision

**Use the entire acquired ReMID corpus as an external real-data test of the
NYU-trained method.** Select architecture, optimizer, loss weights, checkpoint,
clipping rules and thresholds using NYU training/validation only. Do not tune on
ReMID scores and continue calling ReMID an untouched test.

ReMID's `split_1.zip` through `split_5.zip` are upload shards. The author README
explicitly says so; they are not experimental train/test partitions. An observed
capture path is
`split_N/category/scene_000006/{meta.json,rgb/*.png,depth/depth_raw_mm.npy}`.
Category/numeric scene identifiers restart between shards. The data audit found
different arrangements under the same such identifier, so string equality is
not proof of the same physical scene. Conversely, different identifiers do not
prove independent rooms. No verified physical-room key is currently available.

Before any result computation:

1. Assign a unique **capture ID** from the full shard/category/scene path. All
   lighting images, their depth, metadata and derivative images stay together.
2. Record original relative paths, camera calibration, RGB/depth SHA-256 hashes
   and source archive/member identities. Reject conflicting file identities.
3. Create conservative **audit clusters** merging the same category/numeric
   scene ID across shards. Also join clusters linked by exact RGB-plus-reference
   duplicate evidence. Preserve the evidence for each merge; this is a cautious
   grouping rule, not a recovered room annotation.
4. Count captures, audit clusters, categories, unique RGBs and light variants
   separately. Deduplicate identical RGB/depth pairs by a fixed hash rule before
   scoring, not by model performance. Flag identical RGBs with conflicting
   depth or calibration rather than silently treating them as independent data.
5. Keep the source manifest immutable once scoring starts. Publish its hash and
   retain the original asset inventory and full duplicate audit.

If ReMID training is needed later, first establish stronger physical scene/camera
groups and create a new recorded study. Random light-image or archive-shard
splits are unacceptable. Results from the current external test must then be
acknowledged as development evidence, with a new independent final evaluation.

## NYU and rendered-data arms

The planned new NYU selection is 200 training, 40 validation and 128 evaluation
scenes, one frame per scene, seed 20261107. Preserve official train/test
membership and exclude all 40 scenes from the original pilot across every
partition. Independently verify unique scene IDs, disjoint splits and no exact
RGB/reference duplicates. Selection must precede prediction or score inspection.

Report NYU original images as real RGB-D photographs. Digitally transformed NYU
images are **simulated color/exposure changes on real content**, not physically
captured multi-light pairs. Vari supplies the separately rendered multi-light
arm. Its two rooms and previously inspected phase-two evaluation room remain
limitations: a repeated Vari test is exploratory reuse, not fresh independent
confirmation of a new algorithm. Additional views cannot establish scene scale.

## Real reference-depth contract

The acquisition agent inspected ReMID metadata declaring float32 depth in
millimeters, zero invalid, and alignment to RGB via `rs.align(color)`. Native RGB
and aligned depth are 1280 x 720; preserve the provided intrinsics/distortion
metadata. This contract must be checked for **every** capture, not inferred for
the entire release from a single example.

Use the raw NPY array, not the colored depth-visualization image. Convert
`d_m = d_mm / 1000` once. The predeclared primary mask is:

```text
M = finite(d_mm) AND (d_mm > 100) AND (d_mm < 10000)
```

Thus zeros, negative values, NaNs and positive sentinel-like 65535 mm readings
are excluded by a reference-only rule fixed before model scoring. Record each
reason's pixel count and the excluded fraction per capture. Retain the original
raw values; do not rewrite sentinel values in the source data. A 10 m cap defines
this exploratory scoring domain; it is not an undocumented official ReMID rule.

Use one identical reference mask for every method and every paired lighting
condition within a capture. Confirm that capture geometry and reference
registration are shared across its lights. If a future source supplies different
per-light references, compare calibration and geometry first and use their
reference-only common support, or exclude the group from paired-light analysis
with an explicit reason. Do not infer physical alignment merely because image
dimensions match.

Require at least 1,000 valid native-resolution reference pixels for the primary
metric table; report every excluded capture and reason. This minimum is a
recommended pre-test rule and must be recorded in the executed configuration if
adopted. Resize sparse reference depth only with validity-aware nearest sampling
for training. For evaluation, bilinearly resize each predicted depth to the native
reference shape, using the same image-coordinate convention for every method.
Do not interpolate reference zeros into valid foreground distances.

Clip predictions to [0.1, 10] meters for the declared primary depth scores.
Nonfinite predictions on valid reference support are a method/run failure;
never discard their pixels to improve that method's score. No per-image
median/scale/shift fitting, test-depth threshold fitting, or normalization by
the prediction's own extrema is permitted. If a separate scale-invariant score
is later reported, label its alignment explicitly and keep it out of the metric
depth table.

## Exact primary and secondary endpoints

For valid pixels, let `p` be the clipped prediction in meters and `d` the reference.

```text
AbsRel = mean(|p - d| / d)
RMSE_m = sqrt(mean((p - d)^2))
delta1 = mean(max(p / d, d / p) < 1.25)
```

Calculate these per image. Average image scores over a capture's light variants,
then average over captures within an audit cluster, then average audit clusters.
This prevents a capture with more lights or a repeated identifier from receiving
extra primary weight. Publish the simpler per-image and per-capture means as
secondary descriptive summaries so aggregation is transparent.

The primary real-data comparison is full proposed method minus the strongest
predeclared capacity/training-matched non-CC robustness control in mean AbsRel.
Choose the control and checkpoints on NYU validation, not by whichever real test
score makes the proposal look better. Also publish all control scores, per-cluster
paired differences, RMSE and delta1. Improvement over the frozen backbone alone
does not isolate a CC contribution.

For a capture with `L` aligned lighting predictions, use:

```text
p_bar(x) = mean_l(p_l(x))
LightMAD = mean_{l,x in M}(|p_l(x) - p_bar(x)| / d(x))
WorstLightAbsRel = max_l AbsRel_l
```

Both quantities are dimensionless; lower is better. Aggregate them over captures
and audit clusters as above. For a one-light capture, paired-light stability is
undefined; report it only in the ordinary accuracy table. Constant wrong depth
can produce perfect stability, so accept no stability-only performance claim.

For NYU, one selected image corresponds to one scene, so scene-level paired
means and scene-bootstrap intervals are appropriate descriptive uncertainty
summaries. For ReMID, audit clusters are not established independent physical
rooms. Give cluster-bootstrap intervals only as explicitly conditional
sensitivity summaries; do not claim broad population significance or count
light variants as independent scenes. Multiple training seeds and stronger
scene metadata are required before a final paper-level inference.

## Chromatic versus brightness intervention

Use the same original RGB, depth and reference mask across conditions. Convert
processed sRGB to a declared approximate linear-light representation before
applying channel gains. This is a digital corruption model; it does not undo a
camera ISP or recreate a physical spectral relighting experiment.

A reproducible design uses two orthogonal log-chroma directions
`b1 = (1,-1,0)/sqrt(2)` and `b2 = (1,1,-2)/sqrt(6)`, with positive gains
`g = exp(sign * strength * b)`. A suggested frozen grid is strengths 0.2 and 0.4,
both signs, with a separate scalar direction `(1,1,1)/sqrt(3)` and an identity
arm. The implementation owner may freeze another grid using development data
before scoring; the exact directions and strengths must then be reported.

For the chromatic arm, choose a single image-wide scalar `alpha` so mean linear
luminance with weights [0.2126, 0.7152, 0.0722] is preserved before clipping.
Record **effective gains** `g_eff = alpha * g`, because these are the actual
intervention. Scalar exposure arms change brightness without channel ratios.
Report achieved luminance drift, clipping and quantization by condition. These
different interventions are not automatically equal in perceptual severity.

Compare each method's paired AbsRel change from identity in chromatic and scalar
arms, and whether the full method improves the chromatic arm over the matched
generic robustness control while maintaining clean-image accuracy. Include
achromatic/content conditioning, no-CC-loss and depth-only chromatic ablations.
Known-gain equivariance supervision describes relative synthetic illumination;
it does not supply a real absolute illuminant label.

## Exact-inverse diagnostic and leakage boundary

An oracle inverse of a known digital intervention may divide by its stored
`g_eff` **without consulting depth**. This is a pipeline sanity check with known
synthetic nuisance, not a learned method. Exact recovery is impossible after
clipping; use a predeclared no-clipping diagnostic arm or publish results on
the intersection of original/transformed unsaturated support alongside clipping
counts. sRGB re-encoding or uint8 quantization introduces additional tolerance.
Never claim exact bitwise inversion unless the data path actually preserves it.

Real PCCC illumination labels, if later matched by immutable source identifiers
to the correct NYU color space, can support a separately labelled illuminant
oracle. Do not assume the label applies to an already white-balanced original
NYU image. Do not locate, select or scale a CC correction by the resulting test
depth error. Real illuminant-oracle outputs and exact-gain inverse outputs are
diagnostics and must never be pooled with RGB-only deployable model scores.

The forward interface for every monocular method receives only RGB and the
same explicitly declared metadata. Reference depth, reference normals and test
illumination labels are training/evaluation targets, not inference inputs.

## Release and verification checklist

Before evaluation, save manifest/config/source/checkpoint hashes, exact scene
IDs, duplicate clustering and exclusion rules. Log a split-lock timestamp before
any prediction is scored. After the run, independently recompute metrics from
saved numerical predictions, confirm common support, inspect color-space/depth
examples and publish all planned comparisons, including harms and failed runs.

Keep real original, real captured multi-light, rendered, and digitally transformed
real-image rows separate. Do not compare these exploratory scores with official
ReMID or other paper tables without matching their complete protocol. Expanding
image count is useful engineering progress; it does not resolve unknown room
identity, simulated lighting assumptions, or an unproven method improvement.
