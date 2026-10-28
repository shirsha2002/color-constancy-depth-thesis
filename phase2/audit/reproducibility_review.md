# Reproducibility review

The completed mono run passed an independent numerical audit. All 1,584 saved
evaluation predictions reproduced their per-image metrics exactly; all 144
lighting-instability scores agreed within floating-point precision. Input files,
the original 16-bit depth decoding, the room-separated split, the adapter-only
checkpoints and validation checkpoint selection were checked independently.
The corresponding evidence is `mono_independent_audit.json` and
`../results/mono/independent_audit.json`.

This does not establish a CC-specific advantage. The task-only adapter has lower
AbsRel, RMSE and lighting instability than the CC-conditioned adapter. The CC
adapter has a different delta1 tradeoff. There is one evaluation room and many
correlated views, so 264 images must not be described as 264 independent scenes.

## Remaining engineering risks in the preserved numeric runner

- Its configuration hash includes `cc_model.py`, but not `run_study.py` or the
  complete upstream source tree. The separate source snapshot records the files
  used during the completed run, and the upstream revision is pinned.
- Existing feature caches check input size, manifest and checkpoint hashes but
  do not compare their stored RGB hash to the current RGB file.
- Existing prediction files are reused by filename alone. They do not prove
  their generating adapter or source revision.
- Official `image2tensor` can auto-select CUDA/MPS while this prototype retains
  its model on CPU. The completed run used the laptop's CPU-only environment.

These are future rerun risks, not observed invalidations of the audited completed
run. The numeric source is retained unchanged so its recorded SHA-256 snapshot
remains meaningful.

## Safe commands from the project root

Read-only input-integrity check (no inference or result mutation):

```powershell
& .\.venv\Scripts\python.exe phase2/audit/reproduce_guard.py
```

The guard checks all frozen source snapshots, the base checkpoint, manifest,
528 RGB files, 48 decoded depth arrays and 48 original depth TIFFs. It also
checks the upstream git revision and refuses modified tracked upstream code or
untracked Python source. It checks the original audited summary hash. It is an
integrity safeguard, not a cryptographic signature or a new numerical audit.

For a full rerun, choose a result directory that does not yet exist:

```powershell
& .\.venv\Scripts\python.exe phase2/audit/reproduce_guard.py --run-fresh --out phase2/results/mono_reproduction_01
```

The wrapper uses the frozen 224-pixel/150-step/8-thread settings, requires a
CPU-only process, and redirects only the imported runner's output root. It
refuses every existing output directory, including an interrupted rerun, so no
cached features, predictions or adapters can be silently reused. Use a new
directory after an interruption. The completed reference output is preserved.
The wrapper does not claim bitwise equality across different PyTorch/CPU builds.

To independently audit the original saved outputs without retraining:

```powershell
& .\.venv\Scripts\python.exe phase2/audit/audit_mono.py
```

That auditor currently targets the completed reference directory; it must be
explicitly adapted and reviewed before auditing a new result directory.

## Git artifact scope

Raw datasets, extracted arrays, inference feature/prediction caches, full point
clouds, partial downloads and acquisition environments are local evidence. The
stereo-specific `.gitignore` already excludes nested `arrays/`, evaluation-only
`pointclouds/`, full inference PLYs and superseded float32 diagnostic runs while
allowing small named preview PLYs. The acquisition-specific `.gitignore` excludes
OneDrive response files, signed-link probes and archive tails. Stage selected
source metadata/manifests deliberately; do not force-add entire data folders.
Before publishing, inspect staged paths and sizes and search staged text for
credentials, temporary signed URLs and absolute local account paths.
