# Thesis presentation materials

Updated 24 November 2026. This guide is for preparing Shirsha Mitra's proposal
presentation on 25 November. The main talk is three minutes, followed by about
two minutes of questions.

## Download

- [Presentation materials ZIP](output/defense_2026_11_25/Presentation_Materials_Public.zip)
- [Entire repository ZIP](https://github.com/shirsha2002/color-constancy-depth-thesis/archive/refs/heads/main.zip)
- [Study checkpoints and training records](https://github.com/shirsha2002/color-constancy-depth-thesis/releases/tag/thesis-handoff-2026-11-24)

## Open these files first

1. [Current PowerPoint](output/defense_2026_11_25/Shirsha_Mitra_Proposal_Defense_2026.pptx)
2. [Illustrated four-page proposal PDF](output/defense_2026_11_25/Shirsha_Mitra_Proposal_Report_With_Experiments.pdf)
3. [Editable proposal Word file](output/defense_2026_11_25/Shirsha_Mitra_Proposal_Report_With_Experiments.docx)
4. [Timed script with Bengali explanation](output/defense_2026_11_25/Three_Minute_Script_English_and_Bengali.txt)
5. [English and Bengali defense questions](output/defense_2026_11_25/Defense_QA_English_and_Bengali.txt)
6. [Four scientific figures](output/defense_2026_11_25/figures)
7. [Slide previews](output/defense_2026_11_25/slide_previews)
8. [Official Tables 1 to 3](<output/proposal_submission/2023521460130_Shirsha Mitra_Proposal Report Table1_3.pdf>)

The official form is unsigned. Table 2, meeting records, supervisor assessment,
actual dates and signatures still require the appropriate people. Do not treat
the form as an approved or submitted document.

## Presentation structure

Use five main slides: question, prior evidence, proposed method, current pilot
results, and next experiments. Slides 6 and 7 are backups. The existing script
allocates 20, 35, 40, 55 and 30 seconds. Rehearse aloud with a timer.

Keep the registered title until the supervisor approves a change:
**Design and Implementation of a Color-Constancy-Assisted Monocular Depth
Estimation System**. The current research module estimates global illumination
from RGB-D point clouds; its connection to monocular depth remains a scope
question for the supervisor.

## Results that must stay accurate

The ETH3D development pilot completed 21 models: seven methods, three official
folds, 100 epochs and one seed per fold. It used the local Intel GPU. All 42
saved checkpoints and 13,650 prediction errors passed audit. A later local
inference check reproduced 975 predictions exactly. It was not a new training
run and did not use Modal.

| Method | Clean mean angular error in degrees |
|---|---:|
| Original PCCC | 2.318 |
| Learned gate | 2.574 |
| Mask fusion | 2.588 |
| Ordinary fusion | 2.561 |
| RGB only | 2.682 |

Lower is better. The learned-minus-mask conditional 95% intervals are
[-0.1171, +0.0849] for clean inputs and [-0.1828, +0.0022] for the per-capture
average of four damaged inputs. Both include zero. A gating improvement has
not been established, and the gate did not beat PCCC in this pilot. PCCC has
different capacity; the primary gate comparison is with mask fusion.

The earlier 48-model Modal study used 10,000 epochs and three datasets. Keep
its numbers separate from this new pilot. The camera-frame/scale diagnostic
is also a separate study. Read the linked phase reports before combining results.

The input illustrations show released 256-point clouds, not original full
photographs. Depth noise/removal/shuffling are synthetic interventions on real
captures. Rendered-scene experiments and geometry foundation model experiments
have not yet been completed.

## Code and evidence

- [Fusion implementation and protocol](phase9_reliability_fusion/README.md)
- [All seven methods and five test conditions](phase9_reliability_fusion/pilot_v1/RESULTS.md)
- [CSV results](phase9_reliability_fusion/pilot_v1/summary.csv)
- [Per-capture predictions](phase9_reliability_fusion/pilot_v1/predictions.json)
- [Auditor layout correction](phase9_reliability_fusion/AUDIT_RECOVERY.md)
- [Earlier Modal reproduction](phase6_modal/README.md)
- [Camera-frame and depth-scale study](phase7_depth_format/RESULTS.md)

The public package excludes credentials, meeting access details, private notes,
local environments, temporary renders and downloaded third-party datasets or
pretrained models. See [source attribution](THIRD_PARTY.md) for dataset access.
Completed study checkpoints are separate release assets; smoke-test and cancelled
run checkpoints are not presented as research results.
