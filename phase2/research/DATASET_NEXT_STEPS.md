# Data needed for the next CC-specific test

Checked 29 October 2026. This document distinguishes acquired data from a proposed
extension. It does not report new experiments or claim access to an unacquired
dataset.

## Current evidence and gap

Vari provides fixed-camera, fixed-geometry images under 11 rendered lighting
conditions, but the released archive has only two rooms. The current split is an
exploratory repartition of that released test archive. The
[official authors' repository](https://github.com/zzt76/dca) still says the
training split is forthcoming. Camera views do not replace independent rooms.

DTU supplies calibrated real images and structured-light geometry, but its light
conditions primarily change direction and shading. It supports a geometry
robustness diagnostic, not an isolated test of illuminant color. Neither acquired
archive supplies calibrated ground-truth illuminant RGB vectors.

The current learned CC cue does not beat ordinary task adaptation. The next
dataset must distinguish **chromatic change**, **exposure/shading change** and
**scene-content change**. More copies of the same two rooms will not solve that
problem.

## Candidate: Structured3D, subject to access and projection checks

The [official repository](https://github.com/bertjiazheng/Structured3D) describes
3,500 house designs and splits by design: 3,000 train, 250 validation and 250 test.
It requires an agreement form before the authors provide access. The code's MIT
license must not be substituted for the dataset terms. No agreement has been
submitted and no access has been claimed in this phase.

The [official data organization](https://github.com/bertjiazheng/Structured3D/blob/master/data_organization.md)
places `rgb_coldlight.png`, `rgb_rawlight.png`, `rgb_warmlight.png` beside a shared
depth map in **panorama** directories. The perspective directories list only
`rgb_rawlight.png`. Consequently, downloading perspective RGB-D alone would not
supply the desired paired warm/cold experiment. Depth is uint16 millimeters;
zero denotes no reading.

The laptop's earlier probes of legacy published panorama ZIP links returned 404
(see `paired_candidate_access.json`). Public mirror metadata was inspected in
`structured3d_mirror_metadata.json`, but a mirror file listing or license label
does not establish that the required paired-light assets are present, intact or
appropriately licensed. No mirror data have been used for these results.

If access is obtained, first inspect a small number of complete panorama groups.
Confirm all three lighting images align, share geometry and match the original
scene IDs. For the pinhole monocular model, project all RGB variants through the
same panorama-to-perspective mapping. Independently establish whether source
depth is radial distance or camera-axis depth, and transform it into the correct
target camera-Z convention. Validate backprojection against structure/camera
metadata. Do not feed a panorama directly into the pinhole model and interpret
its metric output as a calibrated result.

This makes Structured3D a plausible next candidate, not an already working
replacement. No large download is necessary until access, asset completeness
and projection conventions pass those checks.

## Controlled diagnosis possible before new data arrive

Use existing RGB-D scenes to create a deliberately limited *digital corruption*
test. Apply bounded positive diagonal gains in linear RGB, with geometry and
reference depth unchanged. Hold luminance constant for the chromatic arm; use
scalar gains for the exposure arm; retain an identity arm. Record saturation and
exclude uninformative severely clipped settings by a predeclared rule. Fit no
gain to evaluation depth. This isolates a response to simulated channel changes,
but must never be called a physical relighting experiment or a calibrated
illuminant-recovery benchmark.

Retain scene-disjoint training, validation and evaluation. The previous NYU test
is already inspected, so it is development evidence for a new algorithm. Reserve
new untouched scenes, lock their IDs before running, and never promote the old
test to a fresh independent confirmation. For a geometry-driven extension,
consider a view-global relative chromatic gain with local scalar shading instead
of unconstrained three-channel corrections at each location. This is a hypothesis
and needs literature comparison, implementation and evaluation before claiming
any gain.

## Minimum experiment acceptance checks

1. Track original scene IDs, camera groups, lighting variants and all source
   hashes. Split by independent scene before any training or parameter choice.
2. Require unchanged geometry and registered, independently sourced reference
   depth. Check units, invalid-depth masks, camera model and visibility.
3. Compare the same base model and training budget with ordinary adaptation,
   an achromatic control, a chromatic module, and its consistency-loss ablation.
   Keep the robust stereo correlation/scalar controls in any geometry diagnostic.
4. Score depth accuracy and paired-light instability together. Record failures,
   clipping and scene-wise outcomes; stable but wrong depth is not success.
5. Select settings only on development scenes, then evaluate once on untouched
   scenes. A CC-specific claim requires added value over the ordinary and
   achromatic controls, not merely a gain over an unadapted base model.

These checks directly address the supervisor's requested sequence: establish the
task's lighting/color sensitivity, then design and test an integrated remedy.
At present, the integrated prototype is implemented and audited; a specific
benefit from its chromatic cue remains unproven.
