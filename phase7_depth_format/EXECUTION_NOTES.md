# Execution, recovery and claim limits

16 November 2026. Nine newly trained PCCC models completed10000epochs; their
complete records, fixed-final and test-selected checkpoints and predictions are
downloaded locally. All15 final checkpoints (nine new plus six reused) were
re-inferred, and3240 saved prediction rows independently checked. Audit: PASS.

## Post-save metadata error, recovered without retraining

The first cloud wrapper returned `dict(condition=condition,fold=fold,**result)`.
The frozen trainer's completed result already contains `fold`, so Python raised
`TypeError: dict() got multiple values for keyword argument 'fold'`. This happens
**after** `train_job` writes its completed checkpoints/predictions and after
`volume.commit()` returns. Consequently the nine Modal function calls display
an error although their numerical training outputs are complete.

No full model was retrained to handle this error. The completion worker permits
only that exact known post-save error, then requires all nine complete.json
records,10000epoch histories, input/config/source/checkpoint hashes, independent
angular recomputation and saved-model inference. Unknown errors are not ignored.

`run_modal.py` now merges the dictionary without duplicating the fold field.
The actual source used for this run is retained byte-for-byte as
`snapshots/run_modal_v1.py` (SHA256
`5fe1a80ec7e60d4140f3ff819a0747ae4ca43aa0ed31fde4cdc3506476e14a58`).
The auditor checks this exact snapshot against each original cloud configuration;
it does not pretend the corrected wrapper produced the existing checkpoints.
Do not execute the snapshot as a new launcher. The model, optimizer, epochs,
input transforms and numerical training source have not changed during recovery.

The timing stage also needed a plain string instead of a returned TorchVersion
object for the lightweight local client. That short training-only timing check
was repeated; the overhead is part of the budget reserve. No held-out outcome
was used to make either execution fix.

## What the results mean

- Native mean error:2.6135degrees; canonical axes:2.5471; canonical plus per-scene
  median-depth scaling:2.7192; matched canonical flat:2.8164.
- Both primary paired intervals include zero. This experiment **does not
  establish a normalization benefit**, nor a geometry benefit under the new
  normalized representation. The small axes-only mean gain is also inconclusive.
- The canonical flat control worsens versus the original flat representation;
  its conditional difference is+0.3103degrees, interval[+0.1037,+0.5165]. This is
  a secondary result, not proof of a particular mechanism.
- DIODE subgroup error increases from2.6175(native) to3.3268(normalized). This is
  descriptive evidence to retain, not a basis for selecting another transform
  on these same test captures. NYU subgroup error is2.6125 versus2.5637.
- Axis convention and depth-scale differences are real input observations.
  Their existence does not prove dataset corruption, incorrect physical units,
  or that format mismatch caused the earlier geometry failure.
- Coordinate standardization is not a novel reliability-aware fusion algorithm.
  One seed per fold and prior development exposure remain material limitations.

## Cost and shutdown

New training time:5.757657GPU-hours. Using public T4 pricing and the maximum
allocated CPU/RAM rate, training compute allowance is approximately$4.67.
Smoke, startup, image build, idle and transfer overhead are additional; the
original allowance reserved$1.50 for these. This is an estimate, not a provider
invoice. User-approved new-experiment cap was$8.

The training application stopped at11:17:33Asia/Shanghai on16November2026,
and the Modal API reported zero tasks. No model retry, extra seed, expensive GPU
fallback or cancelled laptop run was launched. Remaining budget was not used
to search for a better test result.
