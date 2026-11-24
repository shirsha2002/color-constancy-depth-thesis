# Completed pilot audit on 23 November 2026

All 21 existing 100-epoch training jobs were complete when inspected. No model
was retrained and no new Modal compute was allocated for the final review.

The frozen original analyzer failed an exact re-inference assertion on small
float32 differences, with a maximum observed component difference of
5.96046448e-8. The training loader used `x.T.copy()` before stacking its batches;
the original analyzer stacked `x.T` without that contiguous copy.

The original analyzer remains byte-for-byte preserved and still matches the
pretraining frozen-run hash. `analyze_pilot_v2.py` uses the same contiguous input
layout as the training loader and records its own source hash, the frozen
original analyzer hash and this execution correction. Model source, training
settings, checkpoints, source inputs, fixed test corruptions and contrasts were
not changed. No tolerance was relaxed: all 42 clean checkpoint prediction arrays
then reproduced exactly. All 13,650 saved prediction scores passed a separate
scalar-acos recomputation in `verify_pilot_v2.py`.

The v2 verifier checks the executed analyzer path/hash and the original frozen
analyzer hash. It independently reproduces 70 method/condition summary rows,
210 fold summaries and both seeded bootstrap intervals. The maximum scalar
angle discrepancy is approximately 4.26e-12 degrees. Final evidence is in
`pilot_v1/audit.json` and `pilot_v1/independent_verification.json`.

The result remains inconclusive for the proposed gate. Clean learned-minus-mask
is -0.0141 degrees with interval [-0.1171,+0.0849]. The per-capture average of
four damaged-input contrasts is -0.0903 with interval [-0.1828,+0.0022].
The matched original PCCC baseline performs better. This is preliminary
development evidence, not a successful full-budget reproduction of a new
method, a novelty claim or independent scene-level confirmation.
