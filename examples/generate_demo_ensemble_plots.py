#!/usr/bin/env python3
"""Generate ensemble kinematics demo PNGs (synthetic Group 1 / Group 2 data)."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
ASSETS = REPO / "docs" / "assets"
DEMO_DIR = REPO / "examples" / "demo_data"
N_PCT = 101
X_PCT = np.linspace(0, 100, N_PCT)
PCT_COLS = [f"pct_{i}" for i in range(N_PCT)]

JOINT_PARAMS = {
    "hip": {"g1_amp": 32.0, "g2_amp": 44.0, "baseline": 15.0},
    "knee": {"g1_amp": 55.0, "g2_amp": 68.0, "baseline": 10.0},
    "ankle": {"g1_amp": 18.0, "g2_amp": 24.0, "baseline": 5.0},
}
JOINT_TITLE = {"hip": "Hip", "knee": "Knee", "ankle": "Ankle"}
GROUP_COLORS = [
    ("#2ca02c", "#98df8a"),
    ("#d62728", "#ff9896"),
]


def _synthetic_curve(amp: float, baseline: float, phase_shift: float = 0.0) -> np.ndarray:
    x = np.linspace(0, 1, N_PCT)
    curve = baseline + amp * np.sin(np.pi * x + phase_shift)
    curve += 0.08 * amp * np.sin(4 * np.pi * x)
    return curve


def build_demo_ensemble_df(joint: str) -> pd.DataFrame:
    params = JOINT_PARAMS[joint]
    rows = []
    for i, (subj, group) in enumerate(
        [
            ("DEMO_G1A", "Group 1"),
            ("DEMO_G1B", "Group 1"),
            ("DEMO_G1C", "Group 1"),
            ("DEMO_G2A", "Group 2"),
            ("DEMO_G2B", "Group 2"),
            ("DEMO_G2C", "Group 2"),
        ]
    ):
        amp = params["g1_amp"] if group == "Group 1" else params["g2_amp"]
        curve = _synthetic_curve(amp, params["baseline"], phase_shift=0.15 * i)
        row = {
            "subject_id": subj,
            "group": group,
            "board": "RB",
            "time": "pre",
            "side": "left",
            "phase": "approach",
            "n_strides_avg": 4,
            "n_strides_kept": 4,
            "n_strides_rejected": 0,
        }
        for j, val in enumerate(curve):
            row[f"pct_{j}"] = val
        rows.append(row)
    return pd.DataFrame(rows)


def plot_joint_demo(df: pd.DataFrame, joint: str, out_path: Path, dpi: int = 150) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for i, group in enumerate(["Group 1", "Group 2"]):
        grp = df[df["group"] == group]
        arr = grp[PCT_COLS].to_numpy(dtype=float)
        mean = arr.mean(axis=0)
        sd = arr.std(axis=0, ddof=1)
        n = len(grp)
        line, fill = GROUP_COLORS[i % len(GROUP_COLORS)]
        ax.plot(X_PCT, mean, color=line, linewidth=2.0, label=f"{group} mean (n={n})")
        ax.fill_between(X_PCT, mean - sd, mean + sd, color=fill, alpha=0.45, label=f"{group} SD")

    ax.set_xlim(0, 100)
    ax.set_xlabel("Gait cycle (%)", fontsize=11)
    ax.set_ylabel(f"{JOINT_TITLE[joint]} angle (degrees)", fontsize=11)
    ax.set_title(
        f"Approach — {JOINT_TITLE[joint]} angle (demo)\n"
        "Subject mean ± SD; illustrative synthetic data",
        fontsize=12,
    )
    ax.legend(loc="best", fontsize=8, framealpha=0.9)
    ax.grid(True, alpha=0.25, linestyle="--")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=ASSETS)
    parser.add_argument("--dpi", type=int, default=150)
    args = parser.parse_args()

    DEMO_DIR.mkdir(parents=True, exist_ok=True)
    for joint in ("hip", "knee", "ankle"):
        df = build_demo_ensemble_df(joint)
        csv_path = DEMO_DIR / f"ensemble_approach_{joint}_angle.csv"
        df.to_csv(csv_path, index=False)
        png_path = args.output_dir / f"ensemble_{joint}_angle_demo.png"
        plot_joint_demo(df, joint, png_path, dpi=args.dpi)
        print(f"Wrote {png_path}")

    print(f"Demo CSVs under {DEMO_DIR} (optional local regen; do not commit if policy requires)")


if __name__ == "__main__":
    main()
