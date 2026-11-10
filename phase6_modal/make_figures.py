r"""Figures and GIFs for the phase-6 PCCC update, drawn only from saved results.

Run:  .venv\Scripts\python.exe phase6_modal/make_figures.py
PNG figures go to phase6_modal/report/figures/, GIFs to output/gif/.
Example scenes are fixed by ID (report_data.example_scene), never by error.
"""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image
import report_data as D

FIG = D.ROOT/'phase6_modal/report/figures'; GIF = D.ROOT/'output/gif'
INK, INK2, MUTED, GRID, BASE = '#0b0b0b', '#52514e', '#898781', '#e1e0d9', '#c3c2b7'
REAL, FLAT = '#2a78d6', '#eb6834'            # categorical slots 1-2 (validated)
RETRAINED, CLEAN_TRAINED = '#6250d6', '#1baf7a'  # validated pair; aqua needs labels/table
DEPTH_CMAP = LinearSegmentedColormap.from_list('depth', ['#cde2fb', '#0d366b'])
plt.rcParams.update({'font.family': 'Arial', 'font.size': 8.5, 'axes.edgecolor': BASE, 'axes.labelcolor': INK2,
                     'xtick.color': INK2, 'ytick.color': INK2, 'axes.linewidth': 0.8, 'axes.titlesize': 9,
                     'axes.titlecolor': INK, 'legend.frameon': False, 'savefig.facecolor': 'white'})


def clean_axes(ax, grid_axis='y'):
    for s in ['top', 'right']:
        ax.spines[s].set_visible(False)
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.6); ax.set_axisbelow(True)
    ax.tick_params(length=0)


def save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG/name, dpi=220, bbox_inches='tight'); plt.close(fig)


def fig_main():
    fig, ax = plt.subplots(figsize=(6.6, 3.1))
    clean_axes(ax); w = 0.3
    for i, ds in enumerate(D.DATASETS):
        for off, geo, col in [(-0.17, 'real', REAL), (0.17, 'flat', FLAT)]:
            v = D.pccc(ds, geo)['mean']
            ax.bar(i + off, v, w, color=col, label={'real': 'PCCC, real 3D', 'flat': 'PCCC, flat (no depth)'}[geo] if i == 0 else None)
            # Value inside the bar base (white on the saturated fill); paper markers are all >0.8,
            # so they can never cover it.
            ax.text(i + off, 0.08, f'{v:.2f}', ha='center', va='bottom', fontsize=8, color='white', fontweight='bold')
            ax.plot(i + off, D.paper(ds, geo)['mean'], marker='D', ms=5.5, mfc='white', mec=INK, mew=1.1, ls='none',
                    label='Paper, Table 2 (256 points)' if (i == 0 and geo == 'real') else None)
        d = D.real_minus_flat(ds)
        note = (f'3D better by {-d["mean"]:.2f}°\n(95% CI {-d["hi"]:.2f} to {-d["lo"]:.2f})' if d['hi'] < 0 else
                'no clear difference')
        ax.text(i, 3.3, note, ha='center', va='bottom', fontsize=8, color=INK)
    ax.set_xticks(range(3), [f'{D.NAME[d]}\n({int(D.pccc(d, "real")["n"])} scenes)' for d in D.DATASETS])
    ax.set_ylim(0, 3.75); ax.set_ylabel('Mean angular error (°), lower is better')
    h, l = ax.get_legend_handles_labels(); order = [l.index(x) for x in ['PCCC, real 3D', 'PCCC, flat (no depth)', 'Paper, Table 2 (256 points)']]
    ax.legend([h[k] for k in order], [l[k] for k in order], loc='upper left', bbox_to_anchor=(0, -0.2), ncol=3, fontsize=8, handlelength=1.2)
    save(fig, 'fig1_real_vs_flat.png')


def fig_scatter():
    fig, axes = plt.subplots(1, 3, figsize=(6.6, 2.35))
    for ax, ds in zip(axes, D.DATASETS):
        r, f = D.per_capture(ds, 'real'), D.per_capture(ds, 'flat'); ids = sorted(r)
        x = np.array([f[i] for i in ids]); y = np.array([r[i] for i in ids])
        ax.plot([0.03, 40], [0.03, 40], color=BASE, lw=0.8, zorder=1)
        ax.scatter(x, y, s=8, color=REAL, alpha=0.75, edgecolors='white', linewidths=0.3, zorder=2)
        ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlim(0.05, 30); ax.set_ylim(0.05, 30)
        ax.set_xticks([0.1, 1, 10], ['0.1', '1', '10']); ax.set_yticks([0.1, 1, 10], ['0.1', '1', '10'])
        ax.minorticks_off(); clean_axes(ax, 'both'); ax.set_aspect('equal')
        d = D.real_minus_flat(ds)
        ax.set_title(f'{D.NAME[ds]}\n3D better on {d["real_better"]} of {d["n"]} scenes', loc='left', fontsize=8.5)
        ax.set_xlabel('Flat (no depth) error, °')
    axes[0].set_ylabel('Real 3D error, °')
    fig.text(0.5, -0.08, 'Each dot is one test scene (final-epoch models, released output layer). Below the grey line = real 3D was better.',
             ha='center', fontsize=7.5, color=INK2)
    fig.subplots_adjust(wspace=0.35)
    save(fig, 'fig2_per_scene.png')


def fig_depth_quality():
    cats = ['clean'] + D.DEGRADED
    labels = ['Clean depth', '10% noise', '30% noise', '50% removed', 'Shuffled']
    clean = D.pccc('DepthAWB', 'real')['mean']; flat = D.pccc('DepthAWB', 'flat')['mean']
    re = [clean] + [D.pccc('DepthAWB', c)['mean'] for c in D.DEGRADED]
    ct = [clean] + [D.clean_trained_on(c)['mean'] for c in D.DEGRADED]
    fig, ax = plt.subplots(figsize=(6.6, 2.7)); clean_axes(ax)
    ax.axhline(flat, color=MUTED, lw=0.8)
    ax.text(4.42, flat + 0.012, f'No depth (flat): {flat:.2f}°', ha='right', va='bottom', fontsize=7.5, color=INK2)
    x = np.arange(5)
    ax.scatter(x - 0.09, re, s=46, color=RETRAINED, edgecolors='white', linewidths=1.2, zorder=3,
               label='Trained and tested on this depth')
    ax.scatter(x + 0.09, ct, s=46, color=CLEAN_TRAINED, edgecolors='white', linewidths=1.2, zorder=3,
               label='Trained on clean depth, tested on this depth')
    for xi, v in [(0, clean)]:
        ax.text(xi, v - 0.06, f'{v:.2f}°', ha='center', va='top', fontsize=7.5, color=INK2)
    ax.text(2 - 0.09 - 0.1, re[2], f'{re[2]:.2f}°', ha='right', va='center', fontsize=7.5, color=INK2)
    ax.text(4 + 0.09 + 0.1, ct[4], f'{ct[4]:.2f}°', ha='left', va='center', fontsize=7.5, color=INK2)
    ax.set_xticks(x, labels); ax.set_xlim(-0.5, 4.6); ax.set_ylim(1.45, 2.5)
    ax.set_ylabel('Mean angular error (°)')
    ax.legend(loc='upper left', fontsize=7.5, handletextpad=0.3)
    save(fig, 'fig3_depth_quality.png')


def display(rgb, scale=None):
    img = np.clip(rgb, 0, None); scale = scale or np.percentile(img.max(-1), 97)
    return np.clip(img/scale, 0, 1) ** (1/2.2), scale


def balance(rgb, illum):
    illum = np.asarray(illum, dtype=np.float64); return rgb * (illum[1]/illum)


def fig_examples():
    rows = D.DATASETS
    fig, axes = plt.subplots(len(rows), 5, figsize=(6.6, 4.35))
    titles = ['Input colours', 'Depth (light = near)', 'Corrected with\nground truth', 'Corrected with\nPCCC real 3D', 'Corrected with\nPCCC flat']
    for r, ds in enumerate(rows):
        sid = D.example_scene(ds); c, label = D.cloud(ds, sid); rgb = c[:, 3:].reshape(16, 16, 3)
        z = np.abs(c[:, 2]).reshape(16, 16); z = np.where(z > 1e-8, z, np.nan)
        pr, er, fold = D.prediction(ds, 'real', sid); pf, ef, _ = D.prediction(ds, 'flat', sid)
        gt_img, s = display(balance(rgb, label))
        panels = [display(rgb)[0], None, gt_img, display(balance(rgb, pr), s)[0], display(balance(rgb, pf), s)[0]]
        notes = ['', '', '0° (reference)', f'error {er:.2f}°', f'error {ef:.2f}°']
        for k, ax in enumerate(axes[r]):
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
            if k == 1:
                cmap = DEPTH_CMAP.copy(); cmap.set_bad('#f0efec')
                ax.imshow(z, cmap=cmap, interpolation='nearest')
            else:
                ax.imshow(panels[k], interpolation='nearest')
            if r == 0:
                ax.set_title(titles[k], fontsize=7.5, color=INK)
            if notes[k]:
                ax.text(0.5, -0.06, notes[k], transform=ax.transAxes, ha='center', va='top', fontsize=7.5, color=INK)
        axes[r][0].set_ylabel(f'{D.NAME[ds]}\n{sid}\n(fold {fold} test)', fontsize=7, color=INK2, rotation=0, ha='right', va='center', labelpad=6)
    fig.subplots_adjust(wspace=0.08, hspace=0.42)
    save(fig, 'fig4_examples.png')


def fig_curves():
    fig, ax = plt.subplots(figsize=(6.6, 2.4)); clean_axes(ax)
    for geo, col, name in [('real', REAL, 'Real 3D'), ('flat', FLAT, 'Flat (no depth)')]:
        h = json.loads((D.BC/f'ETH3D_{geo}'/'released_real_fold0'/'history.json').read_text())
        e = np.array([row['author_reference_heldout_loss_deg'] for row in h]); k = 100
        smooth = np.convolve(e, np.ones(k)/k, mode='valid'); x = np.arange(k, len(e) + 1)
        ax.plot(np.arange(1, len(e) + 1), e, color=col, lw=0.4, alpha=0.18)
        ax.plot(x, smooth, color=col, lw=2, label=name)
    ax.set_xlim(0, 10000); ax.set_ylim(0, 4); ax.set_xlabel('Epoch'); ax.set_ylabel('Test-fold error (°)')
    ax.legend(loc='upper right', fontsize=7.5)
    save(fig, 'figA_eth3d_curves.png')


def orient(c):
    """Map a cloud to (right, forward, up) using its own 16x16 pixel order to fix axis signs."""
    v = np.abs(c[:, 2]) > 1e-8; idx = np.arange(len(c)); row, col = idx // 16, idx % 16
    sx = 1 if np.corrcoef(c[v, 0], col[v])[0, 1] >= 0 else -1
    sy = -1 if np.corrcoef(c[v, 1], row[v])[0, 1] >= 0 else 1
    return np.stack([sx*c[:, 0], np.abs(c[:, 2]), sy*c[:, 1]], 1), v


def cloud_colours(rgb):
    return display(rgb.reshape(16, 16, 3))[0].reshape(-1, 3)


def draw_cloud(ax, c, colours, title, note):
    p, v = orient(c)
    ax.scatter(p[v, 0], p[v, 1], p[v, 2], c=colours[v], s=14, depthshade=False, edgecolors='none')
    ax.set_title(title, fontsize=9, color=INK, pad=0)
    ax.text2D(0.5, 0.02, note, transform=ax.transAxes, ha='center', fontsize=8.5, color=INK)
    ax.set_xlabel('right', fontsize=7, color=MUTED, labelpad=-12); ax.set_ylabel('depth', fontsize=7, color=MUTED, labelpad=-12)
    ax.set_zlabel('up', fontsize=7, color=MUTED, labelpad=-12)
    ax.set_xticklabels([]); ax.set_yticklabels([]); ax.set_zticklabels([])
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_pane_color((0.98, 0.98, 0.97, 1)); axis._axinfo['grid']['color'] = GRID
    ax.set_box_aspect((1, 1.3, 0.8))


def animate(fig, axes, path, frames=48):
    imgs = []
    for t in range(frames):
        az = -62 + 38*np.sin(2*np.pi*t/frames)
        for ax in axes:
            ax.view_init(elev=14, azim=az)
        fig.canvas.draw(); imgs.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[..., :3]))
    pal = imgs[0].quantize(colors=255, method=Image.Quantize.MEDIANCUT)
    frames_p = [im.quantize(palette=pal, dither=Image.Dither.NONE) for im in imgs]
    GIF.mkdir(parents=True, exist_ok=True)
    frames_p[0].save(path, save_all=True, append_images=frames_p[1:], duration=90, loop=0, optimize=True)
    for ax in axes:  # sharper still of the first frame, for the PDF
        ax.view_init(elev=14, azim=-62)
    fig.savefig(FIG/(Path(path).stem + '_still.png'), dpi=200, facecolor='white')


def gif_eth3d():
    ds = 'ETH3D'; sid = D.example_scene(ds); c, _ = D.cloud(ds, sid); colours = cloud_colours(c[:, 3:])
    flat = c.copy(); v = np.abs(flat[:, 2]) > 1e-8; flat[v, :3] /= np.abs(flat[v, 2:3])
    _, er, fold = D.prediction(ds, 'real', sid); _, ef, _ = D.prediction(ds, 'flat', sid)
    mr, mf = D.pccc(ds, 'real')['mean'], D.pccc(ds, 'flat')['mean']; n = int(D.pccc(ds, 'real')['n'])
    fig = plt.figure(figsize=(9, 4.6), dpi=100); fig.patch.set_facecolor('white')
    a1 = fig.add_subplot(1, 2, 1, projection='3d'); a2 = fig.add_subplot(1, 2, 2, projection='3d')
    draw_cloud(a1, c, colours, 'Real 3D (depth as released)', f'PCCC error on this scene: {er:.2f}°\n(mean over all {n} scenes: {mr:.2f}°)')
    draw_cloud(a2, flat, colours, 'Flat: depth removed, same pixel rays', f'PCCC error on this scene: {ef:.2f}°\n(mean over all {n} scenes: {mf:.2f}°)')
    fig.suptitle(f'ETH3D scene {sid} (fold {fold} test scene): the same 256 coloured points, with and without depth',
                 fontsize=10, color=INK, y=0.98)
    fig.text(0.5, 0.015, f'Colours are the raw input (display gamma 2.2). {int((~v).sum())} of 256 points have no depth and are not drawn.',
             ha='center', fontsize=8, color=INK2)
    fig.subplots_adjust(left=0, right=1, top=0.9, bottom=0.06, wspace=0)
    animate(fig, [a1, a2], GIF/'pccc_eth3d_real_vs_flat.gif'); plt.close(fig)


def gif_depthawb():
    ds = 'DepthAWB'; sid = D.complete_depth_scene(ds); clean, _ = D.cloud(ds, sid); colours = cloud_colours(clean[:, 3:])
    versions = [('Clean depth (as released)', clean, D.per_capture('DepthAWB', 'real')[sid], D.pccc('DepthAWB', 'real')['mean'])]
    for cond, name in [('DepthAWB_noise30', '30% depth noise'), ('DepthAWB_drop50', '50% of depth removed'),
                       ('DepthAWB_shuffle', '3D positions shuffled between points')]:
        c, _ = D.cloud(cond, sid)
        assert np.array_equal(c[:, 3:], clean[:, 3:])
        err = D.ERR[f'{cond}:PCCC released, trained on clean depth:inference_only_fixed_final'][sid]
        versions.append((name, c, err, D.clean_trained_on(cond)['mean']))
    fold = next(f for f in '012' if sid in D.manifest(ds)['folds'][f]['test_ids'])
    fig = plt.figure(figsize=(8.4, 7.2), dpi=100); fig.patch.set_facecolor('white'); axes = []
    for k, (name, c, err, mean) in enumerate(versions):
        ax = fig.add_subplot(2, 2, k + 1, projection='3d'); axes.append(ax)
        draw_cloud(ax, c, colours, name, f'error on this scene: {err:.2f}°\n(mean over all 185 scenes: {mean:.2f}°)')
    fig.suptitle(f'DepthAWB scene {sid} (fold {fold} test): one model trained on clean depth, tested on damaged depth',
                 fontsize=10, color=INK, y=0.985)
    fig.text(0.5, 0.01, 'Same colours in every panel; only the depth changes. Points without depth are not drawn.',
             ha='center', fontsize=8, color=INK2)
    fig.subplots_adjust(left=0, right=1, top=0.93, bottom=0.04, wspace=0, hspace=0.08)
    animate(fig, axes, GIF/'pccc_depthawb_depth_damage.gif'); plt.close(fig)


if __name__ == '__main__':
    fig_main(); fig_scatter(); fig_depth_quality(); fig_examples(); fig_curves(); gif_eth3d(); gif_depthawb()
    for p in sorted(FIG.glob('*.png')) + sorted(GIF.glob('pccc_*.gif')):
        print(p.relative_to(D.ROOT), f'{p.stat().st_size/1e6:.2f} MB')
