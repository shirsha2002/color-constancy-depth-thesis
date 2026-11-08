"""Write a readable account only after complete test evaluation and audit."""
import csv
import json
from pathlib import Path
from upstream import ROOT

HOME=ROOT/'phase4_pccc';OUT=HOME/'results/cpu_v1'
NAMES={'released_real':'Released code / real geometry','released_flat':'Released code / flat geometry',
       'weighted_real':'Restored paper-weighted / real geometry','weighted_flat':'Restored paper-weighted / flat geometry',
       'gray_world_points':'Gray-World on sampled RGB','training_mean_illuminant':'Training-mean illuminant'}

def main():
    result=json.loads((OUT/'summary.json').read_text());assert result['audit']=='PASS'
    audit=json.loads((OUT/'independent_audit.json').read_text());assert audit['audit']=='PASS'
    table=result['table'];lookup={r['method']:r for r in table};effects=result['geometry_effects']
    lines=['# Small PCCC reproduction: measured results','',
        'Shirsha Mitra / 2023521460130 / 9 November 2026.','',
        '**Completed: an adapted, reduced-budget CPU reproduction on genuine PCCC DepthAWB data.** '
        'This is not a reproduction of the complete published benchmark or its accuracy.','',
        '## What actually ran','',
        '- All 185 original released DepthAWB samples, each with 256 XYZRGB points and its illumination label.',
        '- 98 training / 25 inner validation / 62 final test samples. Official fold 0 is final test; it was never used to select a checkpoint.',
        '- Four predeclared variants, three training seeds each: 12 trained models, 100 epochs/model, 8 CPU threads.',
        '- The original authors\' PointNet, STN, point-wise network and angular loss are executed from a pinned, unmodified clone.',
        '- Both the released three-channel sum head and an explicit restoration of the paper\'s four-channel weighted head are reported.',
        '- 868 held-out predictions across the 12 trained models and 2 non-neural baselines. All angular scores were independently recomputed; all saved models were re-evaluated to verify their predictions.','',
        'The [protocol](PROTOCOL.md), [source wrapper](upstream.py), [data identities](manifest.json), '
        '[preflight](preflight.json) and [independent audit](results/cpu_v1/independent_audit.json) specify the exact scope.','',
        '## Global illumination estimation','',
        'Recovery angular error is in degrees; lower is better. Learned rows average the three separately selected seeds. '
        'The +/- value is sample standard deviation across training seeds, not an interval over independent scenes.','',
        '| Method | Mean +/- seed SD | Median | Trimean | Worst 25% |',
        '|---|---:|---:|---:|---:|']
    for r in table:
        spread=f" +/- {r['mean_deg_seed_std']:.3f}" if r['training_seeds']>1 else ''
        lines.append(f"| {NAMES[r['method']]} | {r['mean_deg']:.3f}{spread} | {r['median_deg']:.3f} | {r['trimean_deg']:.3f} | {r['worst25_deg']:.3f} |")
    lines+=['','See [every individual seed](results/cpu_v1/per_seed_metrics.csv) and '
        '[every prediction](results/cpu_v1/predictions.csv). The author cosine clamp is retained; '
        'its minimum angular error is approximately 0.081 degrees. Best/worst quartiles use the original integer-slice convention.','',
        '![All methods](results/cpu_v1/figures/angular_error_comparison.png)','',
        '## Does genuine geometry help in this pilot?','',
        'The flat control retains the same RGB values, camera rays and invalid-geometry mask. It removes valid relative ranges '
        'by setting native Z to -1, then applies the same training augmentation. It is not a pure RGB-only model.','',
        '| Family | Real minus flat mean error | Conditional capture-bootstrap interval |',
        '|---|---:|---:|']
    for family,effect in effects.items():
        lo,hi=effect['conditional_capture_bootstrap_95_interval']
        lines.append(f"| {family} | {effect['mean_real_minus_flat_deg']:+.3f} deg | {lo:+.3f} to {hi:+.3f} deg |")
    lines+=['','Negative differences favor native geometry. Intervals resample 62 test captures after averaging the three seed-wise '
        'paired differences. They are conditional on these trained seeds and captures. Unknown physical room identities prevent '
        'claims about room-independent population generalization. Read the [per-seed differences](results/cpu_v1/geometry_effects.json), '
        'including reversals.','',
        '## Comparison with the original paper','',
        'The original CVPR 2022 Table 2 reports **1.05 degree mean / 0.37 degree median** for DepthAWB at 256 points, '
        'and 2.39 degree mean for its no-depth 256-point variant. These are reference values, not matched results for this pilot. '
        'The paper states 10000 training epochs and broader evaluation; this experiment uses 100 epochs, one outer fold and inner-validation '
        'checkpoint selection. Its native-coordinate flat control also preserves missingness rather than rebuilding the complete RGB-D input. '
        '**The published numerical accuracy has not been reproduced.**','',
        'The released model and training entry point do not implement the paper\'s explicit fourth-channel certainty head. '
        'The restored weighted variant documents this change rather than silently labelling the released sum as the paper algorithm. '
        'The families additionally differ in intensity augmentation SD 0.20 (release) versus 0.15 (paper), so their comparison '
        'cannot isolate weighting alone.','',
        '## Actual source-data example','',
        '![First fixed test example](results/cpu_v1/figures/real_example.png)','',
        'The example is the first numeric final-test ID, using the first predeclared seed. No sample or seed was selected for low error. '
        'The 16x16 thumbnails reshape the released sampled colors; they are not full-resolution photographs. Reference labels are global '
        'illuminants, so these images do not establish spatial multi-illumination accuracy.','',
        '![Validation histories](results/cpu_v1/figures/validation_curves.png)','',
        '## Laptop use and retained evidence','',
        f"The twelve training runs took {result['total_training_seconds']/60:.1f} minutes in total, excluding setup, timing, audit and analysis. "
        'Each model has about 1.68 million trainable parameters. The measured timing probe and environment are in '
        '[benchmark.json](benchmark.json) and [runtime.json](runtime.json). The runtime memory value is a single process-RSS snapshot, not peak RAM.','',
        'Code, split IDs, hashes, histories, test predictions, audit and figures are suitable for publication. '
        'Original data and the unmodified upstream clone remain local. The first predeclared seed checkpoint of each variant is published; '
        'the other eight checkpoints remain local to keep the repository lightweight. All twelve SHA identities and metrics are retained.','',
        '## What this establishes and what comes next','',
        'The original released PCCC pipeline now runs on this CPU laptop with genuine illumination labels. '
        'It produces an auditable CC result, rather than substituting the previous monocular-depth experiment. '
        'The real/flat comparisons describe the contribution of native geometry only under this reduced protocol.','',
        'Next, discuss the results with the supervisor, then predeclare a longer-budget/all-fold study. '
        'If this test result guides a method change, treat it as development evidence and reserve a fresh outer fold or independent real dataset '
        'for confirmation. Multi-light work requires suitable local illuminant labels and a stronger prior-art comparison. '
        'No novel method or state-of-the-art result is claimed.','',
        'Primary references: [author code/data](https://github.com/xyxingx/Point-Cloud-Color-Constancy), '
        '[paper manuscript](https://arxiv.org/abs/2111.11280).','']
    (HOME/'RESULTS.md').write_text('\n'.join(lines),encoding='utf-8',newline='\n')
    released=lookup['released_real'];flat=lookup['released_flat'];weighted=lookup['weighted_real']
    text=("Professor, I have now completed a small CPU PCCC reproduction pilot using the authors' source and all185released DepthAWB samples "
        "(256XYZRGBpoints each). I used98training,25inner-validation and62held-out samples from official fold0.\n\n"
        "I trained real-geometry and flat-geometry variants with three seeds each, and separately tested a restored paper-weighted head. "
        f"The released-code real-geometry model achieved mean angular error {released['mean_deg']:.2f} +/- {released['mean_deg_seed_std']:.2f} degrees, "
        f"versus {flat['mean_deg']:.2f} +/- {flat['mean_deg_seed_std']:.2f} with flat geometry. "
        f"The restored weighted real-geometry model achieved {weighted['mean_deg']:.2f} +/- {weighted['mean_deg_seed_std']:.2f} degrees.\n\n"
        "All868test prediction scores passed independent recomputation, and all12checkpoints were checked. "
        "This100-epoch, single-fold pilot does not reproduce the paper's full10000-epoch benchmark. "
        "The source/paper differences and all seeds are documented. Next I plan to discuss the geometry comparison and define a longer all-fold study.\n\n"
        "Code and complete results: https://github.com/shirsha2002/color-constancy-depth-thesis/tree/main/phase4_pccc\n")
    # Keep prose readable while leaving the numerical source untouched.
    for a,b in [('all185','all 185'),('256XYZRGBpoints','256 XYZRGB points'),('used98training,25inner-validation and62held-out','used 98 training, 25 inner-validation and 62 held-out'),
        ('fold0','fold 0'),('All868test','All 868 test'),('all12checkpoints','all 12 checkpoints'),('This100-epoch','This 100-epoch'),('full10000-epoch','full 10000-epoch')]:text=text.replace(a,b)
    final=ROOT/'output/QQ_PCCC_CPU_Pilot_English.txt';final.write_text(text,encoding='utf-8',newline='\n')
    print(HOME/'RESULTS.md');print(final)

if __name__=='__main__':main()
