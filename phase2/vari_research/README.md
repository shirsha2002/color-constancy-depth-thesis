# Vari acquisition, provenance, and frozen-data audit

The selected dataset is complete and ready for the exploratory depth experiment.
It contains **48 camera/view groups and 528 RGB images**, with the same reference
depth shared by all 11 lighting variants in each group. The independent audit
passed on 28 October 2026 without changing the dataset.

## Frozen snapshot

Use `phase2/data/vari/manifest.json`. Paths in the manifest are relative to the
thesis project root, `D:/thesis work/thesis code`.

| Split | Source room | Camera/view groups | RGB images |
|---|---|---:|---:|
| training | 212 | 20 | 220 |
| validation | 212 | 4 | 44 |
| evaluation | 485 | 24 | 264 |

The split uses NumPy PCG64 with seed `20261028`. We sort complete groups by
`view_group`, permute room 212, take its first 20 groups for training and next 4
for validation, then permute room 485 and take its first 24 groups for evaluation.
Each selected group retains all 11 lighting conditions. No camera/view group
crosses a partition. Evaluation uses a different room from training and validation.

The manifest SHA-256 at the completed audit is:

```text
8afcc4ecf1704620655f14b7c15b82f6529e02d1c28cb87c5bf7aba44e1ed33a
```

`frozen_vari_integrity_audit.json` records the independent checks. The audit
verified all 576 selected original files by SHA-256 and ZIP CRC32, then verified
all 96 primary and legacy depth arrays by SHA-256 and exact reconstruction of
their decoding formulas. It also reproduced the seeded split and confirmed
identical reference paths and hashes across the 11 variants of every group.
`selected_asset_hashes.json` records the public asset identities and hashes in
the code repository, so a future download must match the original source
snapshot rather than merely trusting a newly generated manifest.

## Actual source access

The [official DCA repository](https://github.com/zzt76/dca) links the released test
archive through this public OneDrive share:

<https://1drv.ms/u/s!Al8Z5hpFSN2xgfYrToE2XkNuATXN9A?e=XnlU87>

Appending the ordinary `&download=1` parameter returned a ZIP of **824,263,108
bytes**. Anonymous HTTP byte ranges returned status 206 with matching
`Content-Range` headers. We used those ranges to read the central directory and
retrieve selected members, checking each extracted member against the source
ZIP's CRC32. No account or credentials were used. The public Shares API returned
401, so acquisition used the working public web download route.

The actual central directory contains 4,514 files and 1,183,853,068 uncompressed
bytes. Its 2,992 labeled lighting RGBs form **272 complete view groups from only
two rooms**: 81 groups in room 212 and 191 in room 485. This agrees with the
official test path list. The selected original files occupy 178,232,433 bytes.

The full ZIP has not been assembled. A 122 MiB prefix and 83 of 84 remaining byte
range parts are retained locally. The missing part does not affect any selected
file: the training subset is complete and independently audited. No acquisition
process was running at the final audit.

The official source code is pinned in `dca_reference` at commit:

```text
2c8fde439bc94b56783b267198b3fe7e00b23fcf
```

The [pinned README](https://github.com/zzt76/dca/blob/2c8fde439bc94b56783b267198b3fe7e00b23fcf/README.md)
still lists the training split as forthcoming. Training and validation text lists
exist in the repository, but those lists are file names, not the underlying
training assets. Their combined 50,798 entries cover 45 rooms. We did not obtain
the official training split and do not claim to reproduce its benchmark.

## Depth encoding and the explicit decoder refinement

The original depth TIFFs are **16-bit RGB**. All three channels are bit-identical
in every selected file. Their selected stored values range from 0 to 63,552.
Pillow exposes these files as RGB uint8, which discards precision before the
[official loader](https://github.com/zzt76/dca/blob/2c8fde439bc94b56783b267198b3fe7e00b23fcf/dca/dataloader.py)
converts them to grayscale and inverts their normalized value.

We preserve the raw TIFF and provide both decoders:

| Manifest field | File folder | Formula | Nominal step |
|---|---|---|---:|
| `depth` | `depth_m16` | `(1 - raw_uint16 / 65535) * 10` | 0.00015259 m |
| `legacy_depth` | `depth_m_author8` | `(1 - PIL_gray_uint8 / 255) * 10` | 0.03921569 m |

The primary reference uses the preserved 16-bit values. This is an explicit
exploratory decoder refinement and differs from the authors' public evaluation
loader. Both use the authors' inverted normalized 10 m convention. We have not
independently calibrated the original renderer's physical distance settings.

Across the 48 groups, the mean absolute difference between these two references
is **0.013124721 m** and the maximum is **0.038910866 m**. These are reference
differences, not model errors. The full per-group checks are in
`phase2/data/vari/depth_decode_audit.json`.

All RGBs have resolution 640 x 480 and mode RGB. Depth arrays have shape
`(480, 640)`, dtype float32, and values in meters under the stated convention.
One complete group for benchmarking is described in
`phase2/data/vari/first_group_manifest.json`.

## What this dataset can establish

The images provide an actual rendered change of illumination at a fixed camera
view with unchanged scene geometry. They support a paired test of depth error
and prediction consistency across lighting. This is a stronger geometry control
than applying only synthetic channel gains to an unrelated RGB photograph.

The public subset has important limits:

- It contains two original rooms. Different views and lighting variants are
  correlated, so 528 images do not represent 528 independent scenes.
- Training and validation share room 212. Validation holds out views, while
  evaluation holds out one room. One unseen room cannot establish broad scene
  generalization.
- These are rendered images, not a new collection of real photographs.
- Lighting labels combine sunlight, indoor lights, and environment lighting.
  They vary shadows, brightness, and reflections as well as color. A depth effect
  does not by itself isolate color constancy as its cause.
- The released image archive contains no calibrated illuminant RGB vectors.
  Lighting names alone cannot supply an illuminant angular-error reference.
- BRDF-related maps are image files. Their physical units and color calibration
  are not documented sufficiently to treat their pixel values as calibrated
  numerical BRDF parameters.
- This custom split uses assets released as the official **test** split.
  Results must be labeled exploratory rather than official benchmark results.
- The primary depth decoder differs from the official loader. Scores cannot be
  compared directly with published DCA benchmark numbers without a matching
  legacy-decoder evaluation and other matching settings.

The [original paper](https://arxiv.org/html/2112.08006v1) describes the dataset's
rendering process and geometry maps. The repository and downloaded file contents
determine what is actually available in this experiment.

## One command from a fresh checkout

Install Python 3.11 or newer, then run this command from the project root:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\phase2\vari_research\acquire_selected_vari.ps1
```

The wrapper creates a separate `.venv-acquire` environment inside this folder,
installs the exact versions in `requirements-acquisition.txt`, dynamically
resolves the original public share, reads its ZIP central directory, and obtains
only the selected original members. It then creates both depth references and
audits the result against the recorded asset hashes. It does not require the
full ZIP, a DCA code checkout, a saved signed URL, an account, or any probe cache.

The dependencies reproduce the acquisition environment used for this snapshot:
requests 2.32.5, NumPy 2.3.5, Pillow 12.0.0, and tifffile 2025.10.4. The initial
acquisition ran with Python 3.13.9. The wrapper does not alter the model-training
environment.

For an existing environment containing those dependencies:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\phase2\vari_research\acquire_selected_vari.ps1 -Python python -UseExistingEnvironment
```

When `manifest.json` already exists, the wrapper runs only the read-only audit.
That path was executed successfully against the frozen snapshot, including the
recorded public asset hashes. It leaves the manifest and all dataset images and
arrays unchanged.

## Individual script order

The current manifest and images are frozen for the ongoing experiment. The audit
command below is read-only. The acquisition commands recreate data and should
run before a new experiment begins.

The individual acquisition commands, after installing the pinned dependencies,
are:

```powershell
python phase2/vari_research/acquire_vari_subset.py
python phase2/vari_research/audit_frozen_vari.py
```

`acquire_vari_subset.py` reads the remote ZIP central directory, selects the
declared groups, retrieves original files, verifies CRCs, and calls
`prepare_vari_manifest.py` to save the two depth conventions and manifest.
Existing original files must pass their ZIP CRC before the acquisition command
reuses them. It resumes verified downloads after transient network errors and
keeps TLS verification enabled. The central-directory metadata is regenerated
from the public source; no local probe is needed.

`probe_public_access.py` and `inspect_official_lists.py` are optional research
utilities. The latter requires an independently cloned copy of the pinned DCA
repository under `dca_reference`; neither is part of the fresh-checkout workflow.
`snapshot_selected_assets.py` records a new public hash reference only after an
intentional, independently audited source snapshot change. It is not an
acquisition step, because replacing the reference automatically would defeat
the reproducibility check.

The local Git ignore rules exclude the cloned repository, acquisition virtual
environment, OneDrive response/probe files, signed redirect URLs in those
responses, and binary cache. Dataset originals and arrays are excluded by the
project's `phase2/data/` rule. The versioned source and hash reference contain
only the original public share URL and public file identities.

`complete_vari_archive.py` can finish the optional full archive and extract its
remaining original members after the current experiment. It is not needed to
train on the frozen selected subset. All extraction paths must resolve under
`phase2/data/vari/raw`.

The code repository has an Apache-2.0 license. That does not establish a separate
license for the downloaded scene assets. Retain source attribution and keep the
large original assets out of the code repository.
