"""Black and white scientific figures from checked experimental records."""
from pathlib import Path
import json
import hashlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

HOME = Path(__file__).resolve().parent
OUT = HOME / '.build/experiment_report_figures'
OUT.mkdir(parents=True, exist_ok=True)
D = json.loads((HOME / 'experiment_report_evidence.json').read_text())
plt.rcParams.update({'font.family': 'Arial', 'font.size': 10, 'text.color': 'black',
                     'axes.labelcolor': 'black', 'axes.edgecolor': 'black',
                     'xtick.color': 'black', 'ytick.color': 'black',
                     'figure.facecolor': 'white', 'axes.facecolor': 'white',
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'savefig.facecolor': 'white', 'legend.frameon': False})
manifest = {}


def save(fig, name, source):
    path = OUT / f'{name}.png'
    fig.savefig(path, dpi=260, bbox_inches='tight', pad_inches=0.08)
    plt.close(fig)
    # Enforce genuine monochrome RGB pixels, including anti-aliased text.
    with Image.open(path) as im:
        im.convert('L').save(path, dpi=(260, 260))
    manifest[name] = {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                      'source': source}


def value(method, condition):
    return next(r['mean'] for r in D['summary'] if r['method'] == method and r['selection'] == 'fixed_final' and r['condition'] == condition)


# One fixed input chosen by sample ID, with its exact pre-frozen corruptions.
example = D['examples'][0]
kinds = ['clean', 'noise30', 'drop50']
fig = plt.figure(figsize=(8.0, 2.8))
valid_clouds = []
for kind in kinds:
    a = np.array(example['clouds'][kind]['xyzrgb'])
    valid_clouds.append(a[np.abs(a[:, 2]) > 1e-8, :3])
combined = np.concatenate(valid_clouds)
lo, hi = combined.min(0), combined.max(0)
pad = np.maximum((hi - lo) * .04, 1e-5)
for i, kind in enumerate(kinds):
    ax = fig.add_subplot(1, 3, i + 1, projection='3d')
    a = np.array(example['clouds'][kind]['xyzrgb'])
    valid = np.abs(a[:, 2]) > 1e-8
    luminance = np.clip(a[valid, 3:] @ np.array([.2126, .7152, .0722]), 0, 1)
    ax.scatter(*a[valid, :3].T, c=luminance, cmap='gray', vmin=0, vmax=1,
               s=13, edgecolors='black', linewidths=.18, depthshade=False)
    label = ['Raw input', 'Log noise SD 0.30', 'Half valid depth removed'][i]
    ax.set_title(f'{label}\n{valid.sum()} of 256 points have depth', fontsize=10, pad=2)
    for axis, lim0, lim1 in zip(['x', 'y', 'z'], lo - pad, hi + pad):
        getattr(ax, f'set_{axis}lim')(lim0, lim1)
    ax.set_xlabel('X', labelpad=-4, fontsize=9)
    ax.set_ylabel('Y', labelpad=-4, fontsize=9)
    ax.set_zlabel('Z', labelpad=-3, fontsize=9)
    ax.tick_params(labelsize=7, pad=-1)
    ax.view_init(elev=17, azim=-65)
    for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
        axis.set_pane_color((1, 1, 1, 1))
        axis._axinfo['grid']['color'] = (0, 0, 0, .15)
fig.subplots_adjust(left=.01, right=.99, top=.86, bottom=.0, wspace=.08)
save(fig, 'input_clouds', {'sample_id': example['id'], 'conditions': kinds,
                           'data': 'Exact frozen XYZRGB clouds; RGB rendered as luminance; native XYZ units'})

# Unsmoothed means of three independently saved fold histories.
styles = {'pccc': ('Original PCCC', '-', 'o'), 'mask': ('Mask fusion', '--', 's'),
          'learned': ('Learned gate', ':', '^')}
fig, axes = plt.subplots(1, 2, figsize=(7.7, 2.65))
epoch = np.arange(1, 101)
for method, (label, linestyle, marker) in styles.items():
    for ax, key in zip(axes, ['training_loss', 'validation_mean_deg']):
        values = np.array([[r[key] for r in history] for history in D['histories'][method]])
        ax.plot(epoch, values.mean(0), color='black', linestyle=linestyle, linewidth=1.2,
                marker=marker, markersize=3.2, markevery=[0, 19, 39, 59, 79, 99], label=label)
for ax in axes:
    ax.set_xlabel('Epoch')
    ax.set_xlim(1, 100)
    ax.set_xticks([1, 25, 50, 75, 100])
    ax.grid(axis='y', color='black', alpha=.12, linewidth=.5)
axes[0].set_ylabel('Training objective (degrees + consistency)')
axes[0].set_title('Training', fontsize=11)
axes[1].set_ylabel('Mean angular error (degrees)')
axes[1].set_title('Inner validation', fontsize=11)
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=3, bbox_to_anchor=(.5, -.01), fontsize=9)
fig.subplots_adjust(left=.09, right=.98, bottom=.27, top=.87, wspace=.36)
save(fig, 'training_curves', {'folds': [0, 1, 2], 'methods': list(styles), 'aggregation': 'Mean across saved fold histories; no smoothing'})

# Continuous noise levels belong on a line plot; removal and shuffle remain in the table.
fig, ax = plt.subplots(figsize=(7.35, 2.55))
for method, (label, linestyle, marker) in {'rgb': ('RGB only', '-.', 'D'), **styles}.items():
    y = [value(method, condition) for condition in ['clean', 'noise10', 'noise30']]
    ax.plot([0, .1, .3], y, color='black', linestyle=linestyle, linewidth=1.35,
            marker=marker, markersize=4.8, label=label)
ax.set_xticks([0, .1, .3], ['0 clean', '0.10', '0.30'])
ax.set_xlabel('Log space standard deviation of multiplicative depth noise')
ax.set_ylabel('Mean angular error (degrees)')
ax.grid(axis='y', color='black', alpha=.12, linewidth=.5)
ax.legend(ncol=2, fontsize=9, loc='upper left')
ax.set_ylim(2.18, 2.88)
fig.subplots_adjust(left=.12, right=.98, top=.98, bottom=.23)
save(fig, 'noise_results', {'selection': 'fixed_final', 'captures': 195,
                            'conditions': ['clean', 'noise10', 'noise30'], 'error_bars': 'None; descriptive pilot means'})

# Actual output is a three-component illuminant, shown without colored patches.
fig, axes = plt.subplots(1, 2, figsize=(7.7, 2.65))
series = [('Reference', None, 'white', '///'), ('Learned gate', 'learned', 'black', None),
          ('Original PCCC', 'pccc', 'white', 'xx')]
for ax, example in zip(axes, D['examples']):
    p = example['predictions']
    target = np.array(p['learned']['clean']['label'])
    vectors = [target] + [np.array(p[m]['clean']['prediction']) for m in ['learned', 'pccc']]
    x = np.arange(3)
    for j, ((label, method, face, hatch), vector) in enumerate(zip(series, vectors)):
        normalized = vector / vector.sum()
        ax.bar(x + (j - 1) * .24, normalized, width=.22, color=face, edgecolor='black',
               linewidth=.8, hatch=hatch, label=label)
    ax.set_xticks(x, ['R', 'G', 'B'])
    ax.set_ylim(0, .65)
    ax.set_ylabel('Component divided by R + G + B')
    ax.set_title(example['id'], fontsize=11)
    ax.grid(axis='y', color='black', alpha=.12, linewidth=.5)
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=3, bbox_to_anchor=(.5, -.005), fontsize=9)
fig.subplots_adjust(left=.09, right=.98, top=.87, bottom=.25, wspace=.33)
save(fig, 'illuminant_outputs', {'sample_ids': [e['id'] for e in D['examples']], 'condition': 'clean',
                                'selection': D['example_selection'], 'normalization': 'Vector divided by its component sum'})
(OUT / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8', newline='\n')
print(json.dumps({'status': 'PASS', 'figures': len(manifest), 'all_images_grayscale': True, 'output': str(OUT)}, indent=2))
