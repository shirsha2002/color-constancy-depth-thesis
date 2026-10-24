"""Adapt actual experiment evidence to the premium scientific report layout.

No performance values are embedded in this file. The adapter requires a
completed, consistent CSV/JSON result pair and the real generated figures.
"""
from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
NAMES={'none':'No correction','gray_world':'Gray-World','white_patch':'White-Patch',
       'gray_world_half':'Gray-World (0.5)'}
EVAL={'evaluation','test','val','validation'}


def percent(a,b):
    return 100*(a-b)/a


def main():
    s=json.loads((ROOT/'results/summary.json').read_text(encoding='utf-8'))
    f=json.loads((ROOT/'results/figures/figure_manifest.json').read_text(encoding='utf-8'))
    with (ROOT/'results/metrics.csv').open(encoding='utf-8-sig',newline='') as stream:
        allrows=list(csv.DictReader(stream))
    rows=[r for r in allrows if r['split'] in EVAL]
    ids=sorted({r['image_id'] for r in rows})
    dev_ids={r['image_id'] for r in allrows if r['split'] not in EVAL}
    pr=s['protocol'];rt=s['runtime'];results={r['method']:r for r in s['results']}
    environment_path=ROOT/'results/environment.json'
    environment=json.loads(environment_path.read_text(encoding='utf-8')) if environment_path.exists() else {}
    versions=''
    if environment:
        versions=f'Python {environment["python"].split()[0]}; PyTorch {environment.get("packages",{}).get("torch","unrecorded")}. '
    base=results['none'];gw=results['gray_world'];half=results['gray_world_half']
    best=min((r for m,r in results.items() if m!='none'),key=lambda r:r['corrupted_abs_rel'])
    boot=next(b for b in s['bootstrap'] if b['method']=='gray_world')
    n=len(ids);dev=len(dev_ids);ncond=len(pr['casts']);nmethods=len(pr['methods'])
    if len(rows)!=n*ncond*nmethods:raise ValueError('Evaluation CSV is incomplete')
    if n!=rt['evaluation_images']:raise ValueError('Evaluation count disagrees with summary')
    for method,r in results.items():
        selected=[float(x['abs_rel']) for x in rows if x['method']==method and x['condition']!='clean']
        if not np.isclose(np.mean(selected),r['corrupted_abs_rel'],rtol=1e-9,atol=1e-12):
            raise ValueError(f'Metric mismatch for {method}')
    for key in ['baseline','condition','method','best','failure']:
        if not (ROOT/f[key]).exists():raise FileNotFoundError(f[key])
    bestgain=percent(base['corrupted_abs_rel'],best['corrupted_abs_rel'])
    gwgain=percent(base['corrupted_abs_rel'],gw['corrupted_abs_rel'])
    cleancost=100*(gw['clean_abs_rel']/base['clean_abs_rel']-1)
    halfcost=100*(half['clean_abs_rel']/base['clean_abs_rel']-1)
    castcost=100*(base['corrupted_abs_rel']/base['clean_abs_rel']-1)
    completion=datetime.fromisoformat(s['completed_at'])
    selected=f['selection']['failure']
    displaydiff=selected['displayed_absrel_reduction']
    effect='lowered' if bestgain>=0 else 'increased'
    change_clean='increased' if cleancost>=0 else 'reduced'
    figurecap=('Selected by the lowest mean six-cast Gray-World benefit; the displayed cast is its least favorable cast. '
               'This extreme is not a random or typical example.')
    improvement='benefit' if displaydiff>=0 else 'harm'
    clean_text=f'Full Gray-World {change_clean} clean-image AbsRel by {abs(cleancost):.1f}%.'
    if boot['ci95_low']>0:
        evidence_title='Average Gray-World benefit is supported in this pilot'
        evidence_body=(f'Mean paired AbsRel reduction is {boot["mean"]:.4f}, with a 95% bootstrap interval '
                       f'[{boot["ci95_low"]:.4f}, {boot["ci95_high"]:.4f}]. {clean_text}')
    elif boot['ci95_high']<0:
        evidence_title='Average Gray-World harm is observed in this pilot'
        evidence_body=(f'Mean paired AbsRel reduction is {boot["mean"]:.4f}; its 95% bootstrap interval '
                       f'[{boot["ci95_low"]:.4f}, {boot["ci95_high"]:.4f}] lies below zero. {clean_text}')
    else:
        evidence_title='No clear average Gray-World gain is established'
        evidence_body=(f'Cast-average error changes by {-gwgain:+.1f}%. The paired 95% interval '
                       f'[{boot["ci95_low"]:.4f}, {boot["ci95_high"]:.4f}] includes zero. '
                       'Selected helpful cases do not establish an average improvement.')
    executive=(f'{NAMES[best["method"]]} {effect} cast-average AbsRel by {abs(bestgain):.1f}%. '
               'The Gray-World paired confidence interval includes zero; this pilot does not establish an average benefit.'
               if boot['ci95_low']<=0<=boot['ci95_high'] else
               f'{NAMES[best["method"]]} {effect} mean six-cast AbsRel by {abs(bestgain):.1f}% versus no correction. {clean_text}')
    table=[['Clean / no correction',f'{base["clean_abs_rel"]:.4f}',f'{base["clean_rmse_m"]:.3f}',f'{100*base["clean_delta1"]:.1f}%']]
    for method in pr['methods']:
        r=results[method]
        table.append([NAMES[method],f'{r["corrupted_abs_rel"]:.4f}',f'{r["corrupted_rmse_m"]:.3f}',f'{100*r["corrupted_delta1"]:.1f}%'])
    crop=pr['crop']['at_480x640']
    payload={
      'evidence_verified':True,
      'thesis_title':s['thesis_title'],
      'date_label':completion.strftime('%d %B %Y').upper(),
      'phase_label':'FIRST CONTROLLED PILOT / REAL RGB-D DATA',
      'cover_figure':{'path':f.get('cover_pair',f['baseline']),
          'caption':'Actual NYU Depth V2 input and saved model prediction; shared metric-depth scale in the evidence figures.'},
      'cover_stats':[{'value':str(n),'label':'held-out evaluation scenes'},
          {'value':str(ncond),'label':'input color conditions'},
          {'value':f'{len(allrows):,}','label':f'predictions including {dev} dev scenes'}],
      'executive_takeaway':executive,
      'hypothesis':'Can simple illumination-color correction reduce the depth error of a frozen pretrained model when the same scene receives controlled color casts?',
      'implemented_methods':[
          f'Pretrained baseline: {pr["model"]}, {pr["parameters"]} parameters. The network weights remain fixed; this pilot performs inference and evaluation.',
          'Classical corrections: Gray-World estimates illumination from channel means; White-Patch uses channel maxima. Both apply global channel gains in linear RGB.',
          'Gentler correction: Gray-World (0.5) uses the square root of full correction gains. Mean luminance is preserved before clipping; strength was fixed before evaluation.'],
      'depth_explanation':'Each output pixel estimates distance in meters. Colors visualize the saved numerical prediction; every reference and prediction panel uses the same 0.1-10 m color scale.',
      'protocol_rows':[
          ['Dataset and split',f'{n} evaluation scenes + {dev} development scenes; one RGB-D frame per scene, using the official NYU test/train splits.'],
          ['Model and input',f'{pr["model"]}; input-size parameter {pr["input_size"]}; {pr["parameters"]} parameters; output maximum {pr["model_max_depth_m"]} m.'],
          ['Color conditions',f'Clean plus warm, cool, and green casts at two strengths: {ncond} conditions x {nmethods} processing methods. Linear-RGB gains; luminance normalization.'],
          ['Scored pixels',f'Zero-based rows {crop[0]}:{crop[1]}, columns {crop[2]}:{crop[3]} at 480x640; finite reference depth in ({pr["evaluation_min_depth_m"]}, {pr["evaluation_max_depth_m"]}) m.'],
          ['Prediction handling',f'Predictions clipped to [{pr["prediction_clipping_m"][0]}, {pr["prediction_clipping_m"][1]}] m. No fitted depth scale or shift; identical mask for every variant.'],
          ['Aggregation',f'Mean of per-image metrics; six casts equally weighted. Image-level paired bootstrap: {boot["resamples"]:,} resamples.']],
      'protocol_caution':'Development scenes support setup; reported scores use evaluation scenes only. Reference depth is the original NYU filled-depth field. This paired robustness pilot does not reproduce the full published benchmark.',
      'results_title':('Measured error across clean and color-cast inputs' if bestgain>0 else
                       'Classical correction did not improve mean accuracy'),
      'results_subtitle':f'Evaluation only: {n} held-out scenes. The four processing rows average the six synthetic casts; the first row is the clean baseline.',
      'results_headers':['Input / method','AbsRel','RMSE (m)','Delta-1'],
      'results_widths':[208,94,100,107.276],
      'results_rows':table,
      'highlight_row':pr['methods'].index(best['method'])+1,
      'results_figure':{'path':f['method']},
      'results_takeaway':f'Lowest cast-average AbsRel among tested corrections: {best["corrupted_abs_rel"]:.4f} ({NAMES[best["method"]]}), versus {base["corrupted_abs_rel"]:.4f} uncorrected. Delta-1 is the fraction within a 1.25x depth ratio.',
      'stress_title':'Robustness varies across casts and images',
      'stress_subtitle':'The same clean result anchors each cast family. Error can respond differently to warm, cool, and green changes.',
      'stress_figure':{'path':f['condition']},
      'stress_findings':[
          f'Without correction, mean AbsRel is {base["clean_abs_rel"]:.4f} on clean inputs and {base["corrupted_abs_rel"]:.4f} across the six casts ({castcost:+.1f}%).',
          f'Gray-World improves the six-cast mean for {boot["improved_images"]}/{n} images. Mean paired AbsRel reduction (uncorrected minus corrected): {boot["mean"]:+.4f}; 95% bootstrap interval [{boot["ci95_low"]:.4f}, {boot["ci95_high"]:.4f}].',
          f'Clean-image AbsRel changes by {cleancost:+.1f}% with full Gray-World and {halfcost:+.1f}% with half strength. These measurements identify the clean-input cost of each fixed strength.'],
      'statistics_reading_note':'Paired compares the same image under each method. Bootstrap resamples whole images to estimate uncertainty within this pilot. An interval above zero supports an average reduction in AbsRel on these sampled scenes.',
      'qualitative_subtitle':f'Observed {improvement}: {selected["image_id"]}, {selected.get("scene","unknown")}, {selected["condition"].replace("_"," ")}.',
      'qualitative_figure':{'path':f.get('failure_report',f['failure']),'caption':figurecap},
      'qualitative_stats':[
          {'title':'No correction','value':f'{selected["abs_rel_none"]:.4f}','label':'AbsRel on the displayed cast'},
          {'title':'Gray-World','value':f'{selected["abs_rel_gray_world"]:.4f}','label':'AbsRel on the same pixels'},
          {'title':'Paired change','value':f'{-displaydiff:+.4f}','label':'Corrected minus uncorrected AbsRel'}],
      'qualitative_note':'Gray-World assumes the scene average should be neutral. Colored objects and clipping can violate that assumption. Error maps expose where correction changes depth; a visually neutral image is not guaranteed to yield more accurate geometry.',
      'findings':[
          {'title':'A working, auditable system is established',
           'body':f'The laptop produced {len(allrows):,} saved metric-depth predictions over {n+dev} real RGB-D scenes, with raw metrics for {ncond} conditions and {nmethods} processing methods.'},
          {'title':evidence_title,'body':evidence_body},
          {'title':('Fixed half strength did not outperform bypass' if half['corrupted_abs_rel']>=base['corrupted_abs_rel'] else
                    'Fixed half strength is a baseline worth extending'),
           'body':f'Cast-average AbsRel is {half["corrupted_abs_rel"]:.4f} at half strength, {gw["corrupted_abs_rel"]:.4f} at full strength, and {base["corrupted_abs_rel"]:.4f} uncorrected. The next study should decide when correction is appropriate, using development scenes.'}],
      'limitations':[
          f'{n} evaluation scenes form a small deterministic subset. Bootstrap intervals describe these sampled images and do not establish generalization to all scenes.',
          'Synthetic global color gains do not recreate mixed lighting, shadows, camera processing, or all real illumination changes.',
          'NYU filled depth contains sensor and interpolation limitations; the reference is not error-free physical ground truth.',
          'One pretrained model was tested. No new network was trained, and this pilot does not establish a novel algorithm contribution.'],
      'next_step':'Next: select correction strength using development scenes, freeze the rule, then evaluate on a larger held-out NYU subset and genuinely different lighting data.',
      'environment_summary':versions+f'Inference device: {rt["device"]}; {rt["cpu"].replace("Ultra7255H","Ultra 7 255H")}; {rt["ram_gb"]} GB RAM; {rt["threads"]} inference threads. Mean saved inference time: {rt["mean_inference_seconds"]:.2f} s/image. Input-size parameter: {pr["input_size"]}.',
      'commands':[
          r'.\.venv\Scripts\python.exe scripts\check_environment.py',
          rf'.\.venv\Scripts\python.exe scripts\run_experiment.py --manifest data\nyu_pilot\manifest.json --input-size {pr["input_size"]} --threads {rt["threads"]}',
          r'python report\make_figures.py'],
      'reproducibility_note':f'Checkpoint SHA-256 prefix: {pr["checkpoint_sha256"][:20]}. Upstream revision: {(pr["upstream_git_revision"] or "unavailable")[:12]}. Full settings, hashes and numerical evidence are saved in results/summary.json and results/metrics.csv. Report build commands are in report/README.md.',
      'sources':[
          {'title':'Depth Anything V2: official metric-depth implementation and models',
           'url':'https://github.com/DepthAnything/Depth-Anything-V2/tree/main/metric_depth'},
          {'title':'NYU Depth V2: original RGB-D dataset and reference-depth description',
           'url':'https://cs.nyu.edu/~fergus/datasets/nyu_depth_v2.html'},
          {'title':'ZoeDepth: primary-source NYU crop, depth clipping and metric code',
           'url':'https://github.com/isl-org/ZoeDepth/blob/main/zoedepth/utils/misc.py'}]
    }
    target=ROOT/'report/report_payload.json'
    target.write_text(json.dumps(payload,indent=2),encoding='utf-8')
    print(target)


if __name__=='__main__':main()
