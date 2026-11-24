# ETH3D reliability fusion pilot results

21 new models completed 100 epochs on the local Intel GPU. Three official folds; one seed per fold. Development pilot, not full paper-budget training.

| Method | Clean | Noise 10 | Noise 30 | Drop 50 | Shuffle |
|---|---:|---:|---:|---:|---:|
| rgb | 2.6819 | 2.6819 | 2.6819 | 2.6819 | 2.6819 |
| ordinary | 2.5608 | 2.5940 | 2.8748 | 2.5638 | 2.6065 |
| mask | 2.5881 | 2.6060 | 2.7452 | 2.9542 | 2.6381 |
| learned | 2.5740 | 2.5741 | 2.6488 | 2.7603 | 2.5989 |
| learned_clean | 2.8128 | 2.8074 | 2.8968 | 3.3154 | 2.8233 |
| learned_no_consistency | 2.5944 | 2.5938 | 2.6690 | 2.7776 | 2.6158 |
| pccc | 2.3177 | 2.3056 | 2.6299 | 2.3319 | 2.3978 |

All entries are mean recovery angular error in degrees; lower is better. Fixed-final checkpoint results above. Both checkpoint rules and all folds remain in the tables.

## Primary paired comparisons

- clean: learned minus mask -0.0141 degrees, conditional 95% interval [-0.1171, +0.0849].
- average_damaged: learned minus mask -0.0903 degrees, conditional 95% interval [-0.1828, +0.0022].

This pilot does not establish an across-endpoint gating benefit. Report the two endpoints separately and retain all harms.

Intervals are conditional on these trained captures/models, not room or repeated-seed uncertainty; the two primary comparisons are not multiplicity adjusted. Synthetic depth damage is not measured sensor failure. Gate values are not calibrated reliability. No novelty or broad generalization is established.

Audit PASS: all 42 checkpoints re-inferred, 13,650 prediction rows checked, and all histories/validation selection/source/input/checkpoint identities verified. The earlier CPU/XPU gradient mismatch for original PCCC remains recorded; same-XPU repeatability passed, and all 21 models used XPU.
