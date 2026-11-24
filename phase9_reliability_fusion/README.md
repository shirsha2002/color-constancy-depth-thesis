# Depth reliability aware fusion and audited ETH3D pilot

16 November 2026. This implements the supervisor-supported fusion direction
using the released PCCC XYZRGB inputs. It estimates one global illuminant.
**Update 23 November: the 21-model 100-epoch ETH3D pilot is complete and audited.**
Read [all results](pilot_v1/RESULTS.md), [independent verification](pilot_v1/independent_verification.json)
and [audit recovery](AUDIT_RECOVERY.md). Both primary intervals include zero;
the learned gate does not outperform the matched original PCCC baseline.
The study is preliminary development evidence; a full 10,000-epoch fusion study,
repeated seeds and independent scenes remain outstanding. No new paid compute
was allocated.

The [hardware check](HARDWARE.md) confirms the Intel GPU runs the new fusion
model and controls, including a complete synthetic training/checkpoint test.
Local GPU is the first choice for a modest pilot. The CPU was faster for these
small batches; original PCCC still has an unresolved strict XPU gradient check,
so CPU-identical gradients are not established. Subsequent same-XPU repeatability
checks passed and permitted all 21 pilot models on that backend. No Modal job was needed.

The [model](model.py) has RGB and geometry branches and a learned per-point gate
that controls how much geometry enters fusion. The gate uses input validity
and neighboring depth-disagreement cues. It gets no illumination labels, clean
reference depth, test errors or corruption IDs. Labels enter the training loss.

The candidate has 34,020 active parameters. Controls are color-only, ordinary
fusion, explicit-mask fusion, no-damage training, and no-consistency loss.
The pinned original PCCC also runs through the same worker for a future matched
rerun. Capacity and training differences must remain visible. Read
[DESIGN.md](DESIGN.md) for exact equations, choices, controls and limitations.

## Verified locally

[Implementation checks](implementation_checks.json) cover all 704 source
capture identities across three datasets, disjoint capture-level splits, finite
predictions, gradients reaching both branches and the gate, an optimizer update
for every model/control, all-missing-depth fallback, point-order invariance,
fixed-seed damage reproducibility and exact checkpoint save/load.

An initial permutation check exposed ambiguous nearest-neighbor ties on the
regular ray grid. The cue builder now includes all points tied at the kth
distance and computes distances directly. The corrected implementation passes
the invariance check. No accuracy outcomes were used to choose this fix.

The [worker test](worker_checks.json) uses only generated fixtures and checks
training-only optimizer access, outer-test access after training finishes,
checkpoint hashes, both selection rules and exact saved-model re-inference.
These are software checks, not evidence that the algorithm improves CC.

```powershell
& .\.venv\Scripts\python.exe phase9_reliability_fusion/smoke.py
& .\.venv\Scripts\python.exe phase9_reliability_fusion/test_worker.py
& .\.venv\Scripts\python.exe phase9_reliability_fusion/train.py --dataset ETH3D --mode learned --fold 0 --damage-training --dry-run
& .\.venv-xpu\Scripts\python.exe phase9_reliability_fusion/test_worker.py --device xpu
```

Smoke uses four official ETH3D **training** captures for the optimizer checks;
no benchmark test scores are produced. The synthetic worker test is separate.
No Modal API calls or paid compute are made by these checks.

The worker requires `--execute-training` and a new output directory for an
actual training job; it does not silently start, resume, rent a GPU or retry.
No full paper-budget fusion run has been launched. The completed pilot used the
separately frozen [pilot protocol](PILOT_PROTOCOL.md). Before any new paid execution, freeze model
counts, epochs, seeds, primary comparisons, corruption-test manifests, full
evaluation/audit and cost limits. A new cloud budget must be authorized; the
earlier $8 was for phase seven. The canceled laptop experiment remains stopped.

Synthetic noise/drop/shuffle does not establish real sensor robustness. The
gate is not calibrated uncertainty, and adding branches/gating is not by itself
a novelty claim. Independent scene-level confirmation remains outstanding.
Proposal defense materials are in [the defense folder](../output/defense_2026_11_25/START_HERE.txt).
The separate ETH3D scale-only preparation is preserved under
[phase eight](../phase8_eth3d_scale/README.md); it is not this fusion algorithm.

Final audit commands, allocating no cloud resources:

```powershell
& .\.venv-xpu\Scripts\python.exe phase9_reliability_fusion/analyze_pilot_v2.py
& .\.venv\Scripts\python.exe phase9_reliability_fusion/verify_pilot_v2.py
& .\.venv\Scripts\python.exe phase9_reliability_fusion/plot_pilot.py
```

The original frozen analyzer is preserved; use the v2 auditor for the documented
training-loader memory-layout correction. Do not rerun training to regenerate
an already complete experiment.
