# How to explain this week's work

Shirsha Mitra, 2023521460130. Read the measured result tables and weekly PDF
alongside this guide before discussing them with your supervisor.

## The question in plain English

The camera and objects can stay still while the light changes. The true depth
should stay the same. We ask whether color changes disturb a depth model, and
whether an internal color mechanism can reduce that error beyond ordinary
training and brightness robustness.

## Which data are real

- NYU contains real camera images and Kinect-derived reference depths. The new
  selection has200training,40validation and128test scenes, excluding the original
  forty-scene pilot. Artificial channel/exposure changes on these images are
  simulated illumination interventions.
- ReMID contains735physically captured lighting images and measured aligned
  depth. Its creators are the LumiDepth authors. Our work acquires, verifies and
  maps their public release into a controlled external test. Fifty capture groups
  do not establish fifty independent rooms; location identities are unknown.
- Vari contains rendered indoor lighting variants. The selected evaluation has
 264images from one room and remains an exploratory simulation diagnostic.
- PCCC assets include illumination labels, fold tables and sampled colored point
  clouds. They support literature/baseline development. They are not silently
  used as dense reference depth in the current model.

## What the network receives

At inference it receives one RGB image. A frozen model predicts initial depth.
That prediction provides approximate virtual3D structure. The new internal
module combines local color and semantic features, and uses depth boundaries
to guide a relative chromatic field. This field changes latent feature tokens
before depth decoding. Reference depth is used only for training or evaluation.

The virtual geometry is derived from a prediction and normalized image rays.
It is not a measured calibrated world point cloud. The chromatic field is not
calibrated physical illumination merely because its variable name says so.

## Why so many controls

An ablation removes one component to test its contribution. Ordinary adaptation
tests whether generic domain fitting explains a gain. A content control tests
generic conditional capacity. A brightness control tests an achromatic cue.
Flat geometry keeps architecture and losses while replacing predicted structure
with a constant-depth reference. Removing each added loss tests whether that
objective helps or hurts.

The relative-chroma loss did not win the initial validation comparison. We kept
that negative result and added loss-matched controls. The best validation recipe
is recorded before final-score inspection; it can be a control rather than the
originally proposed full method. Every recipe remains visible in the tables.

## Reading the numbers

AbsRel averages absolute distance error divided by reference distance. Lower is
better. For example, predicting2.2m for a2.0m reference gives0.1relative error.
An AbsRel of0.16 is not a classification accuracy of84%.

RMSE is a distance error in meters and emphasizes large mistakes. Delta1 is the
fraction of valid pixels within a1.25prediction/reference ratio. Higher is better.

Lighting instability measures variation across light images of one fixed
capture. A constant wrong map can be stable, so accuracy and stability must be
read together. Paired intervals retain whole scenes or conservative capture
labels. They describe this evaluation and one training seed, not all future
scenes or training randomness.

## What makes a paper contribution

Working code and a better score over a frozen model are useful progress. A
CC-specific claim additionally needs improvement over strong capacity/loss
controls and a coherent explanation of the chromatic mechanism. A novelty claim
also needs comparison with close prior work such as PCCC and LumiDepth. The
current study is a prototype and a controlled test, not proof of novelty.

The eight-bit main pipeline has measured residual brightness drift below0.63%
in the intended chromatic controls. A separate floating-point frozen-model probe
checks the intervention under tighter input brightness control. Both protocols
are reported, with no trained model changed to improve the probe result.

## What to show your professor

1. The weekly PDF, especially the real-data provenance, matched controls and
   complete tables.
2. The latest GitHub main page and the phase-three protocol/code links.
3. The exact best-validation recipe and its final test outcomes, with the failed
   loss and all competing controls visible.
4. The next method question from PAPER_PLAN.md, rather than claiming the paper
   is already ready for submission.
