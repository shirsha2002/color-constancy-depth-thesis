# Completed study downloads

Download the training archives from the
[24 November thesis handoff release](https://github.com/shirsha2002/color-constancy-depth-thesis/releases/tag/thesis-handoff-2026-11-24).
Each archive preserves repository-relative paths, saved checkpoints, training
records and a `Checkpoint_File_Hashes.json` inventory. Extract it at the repo
root to restore the original paths.

| Archive | Completed models | Study |
|---|---:|---|
| Phase4_CPU_Pilot_Training.zip | 12 | Earlier 100-epoch DepthAWB CPU pilot |
| Phase6_DepthAWB_Training.zip | 12 | Full-budget DepthAWB reproduction |
| Phase6_AllDatasets_Training.zip | 36 | Other datasets and depth-damage conditions |
| Phase7_DepthFormat_Training.zip | 9 | Full-budget camera-frame/depth-scale diagnostic |
| Phase9_ETH3D_Fusion_Pilot_Training.zip | 21 | New 100-epoch ETH3D fusion pilot |

The phase-six and phase-seven full-budget models use 10,000 epochs. The newer
fusion pilot remains a reduced experiment and does not establish a gating gain.
These archives include training/resume states as well as final checkpoints.
The JSON download manifest records archive sizes and SHA-256 hashes.

Original datasets and external pretrained weights remain available from their
attributed sources and setup scripts. They are not mirrored in these archives.
Local environments, credentials, private meeting details, temporary renders,
smoke checks and cancelled-training checkpoints are not part of this handoff.
