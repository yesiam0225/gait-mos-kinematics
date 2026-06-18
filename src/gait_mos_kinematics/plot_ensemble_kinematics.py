"""
plot_ensemble_kinematics.py — Mean ± SD ensemble plots from wide-format CSVs.

Produces one publication-style figure per ensemble file: solid mean lines and
shaded ±1 SD ribbons, grouped by condition (default: board × time).
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

N_PCT = 101
PCT_COLS = [f'pct_{i}' for i in range(N_PCT)]
X_PCT = np.linspace(0, 100, N_PCT)

GROUP_COLORS = [
    ('#2ca02c', '#98df8a'),  # green
    ('#d62728', '#ff9896'),  # red
    ('#1f77b4', '#aec7e8'),  # blue
    ('#ff7f0e', '#ffbb78'),  # orange
    ('#9467bd', '#c5b0d5'),  # purple
    ('#8c564b', '#c49c94'),  # brown
]

SIGNAL_YLABEL = {
    'angle': '{joint} Angle (degrees)',
    'velocity': '{joint} Angular Velocity (deg/s)',
    'acceleration': '{joint} Angular Acceleration (deg/s²)',
}

JOINT_TITLE = {'hip': 'Hip', 'knee': 'Knee', 'ankle': 'Ankle'}
PHASE_TITLE = {
    'approach': 'Approach',
    'recovery': 'Recovery',
    'crossing_lead': 'Crossing (lead)',
    'crossing_trail': 'Crossing (trail)',
}


def parse_ensemble_name(path: Path) -> tuple[str, str, str] | None:
    m = re.match(r'ensemble_(.+)_(hip|knee|ankle)_(angle|velocity|acceleration)\.csv$', path.name)
    if not m:
        return None
    return m.group(1), m.group(2), m.group(3)


def group_label(row: pd.Series, group_cols: list[str]) -> str:
    return ' '.join(str(row[c]) for c in group_cols)


def subject_curves(df: pd.DataFrame, group_cols: list[str],
                   collapse_side: bool) -> pd.DataFrame:
    """One curve per subject per condition group (optionally collapse L/R)."""
    keys = ['subject_id'] + group_cols
    if collapse_side:
        return df.groupby(keys, as_index=False)[PCT_COLS].mean()
    keys.append('side')
    return df.groupby(keys, as_index=False)[PCT_COLS].mean()


def group_mean_sd(curves: pd.DataFrame, label: str) -> tuple[np.ndarray, np.ndarray, int]:
    arr = curves[PCT_COLS].to_numpy(dtype=float)
    return arr.mean(axis=0), arr.std(axis=0, ddof=1), len(arr)


def plot_ensemble_csv(csv_path: Path, output_dir: Path,
                      group_cols: list[str], collapse_side: bool,
                      dpi: int = 150, filename_suffix: str = '') -> Path | None:
    parsed = parse_ensemble_name(csv_path)
    if parsed is None:
        return None
    phase, joint, signal = parsed

    df = pd.read_csv(csv_path)
    if not set(PCT_COLS).issubset(df.columns):
        return None

    subj_curves = subject_curves(df, group_cols, collapse_side)
    subj_curves = subj_curves.copy()
    subj_curves['_label'] = subj_curves.apply(
        lambda r: group_label(r, group_cols), axis=1
    )
    labels = sorted(subj_curves['_label'].unique())

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for i, label in enumerate(labels):
        grp = subj_curves[subj_curves['_label'] == label]
        if grp.empty:
            continue
        mean, sd, n = group_mean_sd(grp, label)
        color, fill = GROUP_COLORS[i % len(GROUP_COLORS)]
        ax.plot(X_PCT, mean, color=color, linewidth=2.0, label=f'{label} Mean (n={n})')
        ax.fill_between(X_PCT, mean - sd, mean + sd, color=fill, alpha=0.45,
                        label=f'{label} SD')

    ax.set_xlim(0, 100)
    ax.set_xlabel('Gait Cycle (%)', fontsize=11)
    ylabel = SIGNAL_YLABEL[signal].format(joint=JOINT_TITLE[joint])
    ax.set_ylabel(ylabel, fontsize=11)
    side_note = 'L/R collapsed' if collapse_side else 'per side'
    ax.set_title(
        f"{PHASE_TITLE.get(phase, phase)} — {JOINT_TITLE[joint]} ({signal})\n"
        f"Subject mean ± SD; {side_note}",
        fontsize=12,
    )
    ax.legend(loc='best', fontsize=8, framealpha=0.9)
    ax.grid(True, alpha=0.25, linestyle='--')
    fig.tight_layout()

    out = output_dir / f"{csv_path.stem}{filename_suffix}.png"
    fig.savefig(out, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    return out


def plot_all(ensemble_dir: Path, output_dir: Path,
             group_cols: list[str], collapse_side: bool,
             by_side_plots: bool, dpi: int, verbose: bool) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(ensemble_dir.glob('ensemble_*_*_*.csv'))
    n = 0
    for f in files:
        out = plot_ensemble_csv(f, output_dir, group_cols, collapse_side, dpi=dpi)
        if out:
            n += 1
            if verbose:
                print(f"  {out.name}")
        if by_side_plots:
            out2 = plot_ensemble_csv(
                f, output_dir, group_cols + ['side'],
                collapse_side=False, dpi=dpi, filename_suffix='_by_side',
            )
            if out2:
                n += 1
                if verbose:
                    print(f"  {out2.name}")
    return n


def main():
    p = argparse.ArgumentParser(description="Plot ensemble kinematics mean ± SD")
    p.add_argument('--ensemble-dir', type=Path, required=True,
                   help='Directory with ensemble_*.csv wide files')
    p.add_argument('--output-dir', type=Path, required=True,
                   help='Directory for PNG figures')
    p.add_argument('--group-cols', type=str, default='board,time',
                   help='Comma-separated columns defining plot groups (default: board,time)')
    p.add_argument('--no-collapse-side', action='store_true',
                   help='Average only within same side (default: collapse left/right per subject)')
    p.add_argument('--by-side-plots', action='store_true',
                   help='Also write separate figures grouped by side')
    p.add_argument('--dpi', type=int, default=150)
    p.add_argument('--quiet', action='store_true')
    args = p.parse_args()

    group_cols = [c.strip() for c in args.group_cols.split(',') if c.strip()]
    print(f"ensemble_dir: {args.ensemble_dir}")
    print(f"output_dir:   {args.output_dir}")
    print(f"group_cols:   {group_cols}\n")

    n = plot_all(
        args.ensemble_dir, args.output_dir, group_cols,
        collapse_side=not args.no_collapse_side,
        by_side_plots=args.by_side_plots,
        dpi=args.dpi, verbose=not args.quiet,
    )
    print(f"\nWrote {n} figure(s) to {args.output_dir}")


if __name__ == '__main__':
    main()
