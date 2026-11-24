# Completed study downloads

Download the training archives from the
[24 November thesis handoff release](https://github.com/shirsha2002/color-constancy-depth-thesis/releases/tag/thesis-handoff-2026-11-24).
The studies use small, independently extractable ZIP parts so an interrupted
connection does not require downloading a whole study again. Download every
part of the study you need and extract them into the same repository root.
No concatenation is needed. Each part preserves repository-relative paths and
includes its own file-hash inventory.

| Archive prefix (download all numbered parts) | Completed models | Study |
|---|---:|---|
| Phase4_CPU_Pilot_Training_part | 12 | Earlier 100-epoch DepthAWB CPU pilot |
| Phase6_DepthAWB_Training_part | 12 | Full-budget DepthAWB reproduction |
| Phase6_AllDatasets_Training_part | 36 | Other datasets and depth-damage conditions |
| Phase7_DepthFormat_Training_part | 9 | Full-budget camera-frame/depth-scale diagnostic |
| Phase9_ETH3D_Fusion_Pilot_Training_part | 21 | New 100-epoch ETH3D fusion pilot |

The phase-six and phase-seven full-budget models use 10,000 epochs. The newer
fusion pilot remains a reduced experiment and does not establish a gating gain.
These archives include training/resume states as well as final checkpoints.
The JSON download manifest records archive sizes and SHA-256 hashes.

To download and verify all checkpoint ZIPs (about 2.14 GiB) with Python 3:

```powershell
python scripts/download_study_checkpoints.py
```

Files are saved in `downloads/checkpoints`. Re-running the command skips
verified downloads. Use `--list` to see the studies, or select one, for example:

```powershell
python scripts/download_study_checkpoints.py --study Phase9_ETH3D_Fusion_Pilot_Training
```

The presentation ZIP is much smaller and does not require these training files.

Original datasets and external pretrained weights remain available from their
attributed sources and setup scripts. They are not mirrored in these archives.
Local environments, credentials, private meeting details, temporary renders,
smoke checks and cancelled-training checkpoints are not part of this handoff.
