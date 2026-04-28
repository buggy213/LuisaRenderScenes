import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

df = pd.read_csv('benchmark_results.csv')

scenes = sorted(df['scene'].unique())  # 10 scenes
spectra = ['sRGB', 'Hero']

mega_configs = ['ser=off', 'ser hints=none', 'ser hint=hit', 'ser hint=material', 'ser hint=rr', 'ser hints=all']
bar_order   = ['WavePath'] + mega_configs
bar_labels  = ['WavePath', 'ser=off', 'hints=∅', 'hint=hit', 'hint=mat', 'hint=rr', 'hints=all']
n_bars = len(bar_order)

cmap   = plt.colormaps['tab10']
colors = [cmap(i) for i in range(n_bars)]

# 4 rows × 5 cols: rows 0-1 = sRGB, rows 2-3 = Hero
SCENES_PER_ROW = 5
N_SPEC_ROWS    = 2   # ceil(10 / 5)
N_ROWS         = N_SPEC_ROWS * len(spectra)  # 4

fig, axes = plt.subplots(N_ROWS, SCENES_PER_ROW,
                         figsize=(20, 11),
                         facecolor='white',
                         constrained_layout=True)
fig.patch.set_facecolor('white')

for spec_idx, spectrum in enumerate(spectra):
    for scene_idx, scene in enumerate(scenes):
        row = spec_idx * N_SPEC_ROWS + scene_idx // SCENES_PER_ROW
        col = scene_idx % SCENES_PER_ROW
        ax  = axes[row, col]
        ax.set_facecolor('white')

        # MegaPath ser=off baseline
        base_q = df[(df['scene']      == scene)    &
                    (df['integrator'] == 'MegaPath') &
                    (df['spectrum']   == spectrum)  &
                    (df['ser_config'] == 'ser=off')]
        if base_q.empty:
            ax.set_visible(False)
            continue
        base_ms = base_q['wall_ms'].values[0]

        heights  = []
        abs_vals = []

        # WavePath bar (leftmost)
        wave_q = df[(df['scene']      == scene)     &
                    (df['integrator'] == 'WavePath') &
                    (df['spectrum']   == spectrum)]
        if not wave_q.empty:
            w = wave_q['wall_ms'].values[0]
            heights.append(w / base_ms)
            abs_vals.append(w)
        else:
            heights.append(np.nan)
            abs_vals.append(None)

        # MegaPath SER configs
        for cfg in mega_configs:
            cfg_q = df[(df['scene']      == scene)    &
                       (df['integrator'] == 'MegaPath') &
                       (df['spectrum']   == spectrum)  &
                       (df['ser_config'] == cfg)]
            if not cfg_q.empty:
                ms = cfg_q['wall_ms'].values[0]
                heights.append(ms / base_ms)
                abs_vals.append(ms)
            else:
                heights.append(np.nan)
                abs_vals.append(None)

        xs = np.arange(n_bars)
        valid_heights = [h for h in heights if not np.isnan(h)]

        for i, (h, a) in enumerate(zip(heights, abs_vals)):
            if np.isnan(h):
                continue
            ax.bar(xs[i], h, 0.72, color=colors[i], zorder=2)
            # Annotate WavePath and ser=off with absolute wall time
            if i in (0, 1) and a is not None:
                ax.text(xs[i], h + 0.012, f'{a/1000:.1f}s',
                        ha='center', va='bottom', fontsize=5.5, color='#222')

        ax.axhline(1.0, color='#999', linestyle='--', linewidth=0.8, zorder=1)
        ax.set_xticks([])
        ax.set_title(scene, fontsize=7.5, pad=3)

        ymax = max(valid_heights)
        ax.set_ylim(0, ymax + max((ymax - min(valid_heights)) * 0.25, 0.12) + 0.1)

        ax.yaxis.grid(True, alpha=0.25, linewidth=0.5, zorder=0)
        ax.set_axisbelow(True)
        for spine in ax.spines.values():
            spine.set_linewidth(0.5)
        ax.tick_params(axis='y', labelsize=6)

        if col == 0:
            ax.set_ylabel('norm. wall time', fontsize=6)

# Section labels (sRGB / Hero) as rotated text on the right margin
# Using figure coordinates — place after constrained_layout has run
fig.canvas.draw()   # needed to resolve constrained_layout positions

for spec_idx, spectrum in enumerate(spectra):
    # Average y of the two rows belonging to this spectrum
    row0 = spec_idx * N_SPEC_ROWS
    row1 = row0 + N_SPEC_ROWS - 1
    bbox0 = axes[row0, -1].get_position()
    bbox1 = axes[row1, -1].get_position()
    y_mid = (bbox0.y0 + bbox1.y1) / 2
    x_right = bbox0.x1 + 0.01
    fig.text(x_right + 0.005, y_mid, spectrum,
             fontsize=11, fontweight='bold', va='center', ha='left',
             rotation=270, color='#333',
             transform=fig.transFigure)

# Shared legend at the bottom
legend_handles = [mpatches.Patch(color=colors[i], label=bar_labels[i])
                  for i in range(n_bars)]
fig.legend(handles=legend_handles,
           loc='lower center',
           ncol=n_bars,
           fontsize=8,
           bbox_to_anchor=(0.5, -0.04),
           frameon=True, edgecolor='#ccc')

fig.suptitle('Normalized Wall Time relative to MegaPath ser=off  '
             '(lower = faster)\n'
             'Absolute times annotated on WavePath and ser=off bars',
             fontsize=10, fontweight='bold')

plt.savefig('benchmark_visualization.png', dpi=150,
            bbox_inches='tight', facecolor='white')
print("Saved benchmark_visualization.png")
