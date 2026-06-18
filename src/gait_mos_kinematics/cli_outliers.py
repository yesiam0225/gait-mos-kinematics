"""Shared CLI arguments and JSON reporting for outlier handling."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def add_tier1_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        '--spike-threshold', type=float, default=100.0,
        help='Max mm/frame for marker spike rejection (default 100; use 1e9 to disable)',
    )
    parser.add_argument(
        '--filter-cutoff', type=float, default=6.0,
        help='Butterworth low-pass cutoff Hz (default 6; <=0 disables)',
    )
    parser.add_argument(
        '--no-filter', action='store_true',
        help='Disable Butterworth filter entirely',
    )


def add_tier2_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        '--mahalanobis', action='store_true',
        help='Enable Tier-2 Mahalanobis stride rejection during ensemble averaging',
    )
    parser.add_argument(
        '--mahalanobis-alpha', type=float, default=0.001,
        help='Chi-square significance for Mahalanobis cutoff (default 0.001)',
    )
    parser.add_argument(
        '--mahalanobis-components', type=int, default=5,
        help='PCA components for Mahalanobis space (default 5)',
    )


def tier1_config_from_args(args: argparse.Namespace) -> dict[str, Any]:
    cutoff = None if args.no_filter else args.filter_cutoff
    if cutoff is not None and cutoff <= 0:
        cutoff = None
    return {
        'spike_threshold_mm_per_frame': args.spike_threshold,
        'filter_cutoff_hz': cutoff,
    }


def tier2_config_from_args(args: argparse.Namespace) -> dict[str, Any]:
    return {
        'enabled': bool(getattr(args, 'mahalanobis', False)),
        'alpha': args.mahalanobis_alpha,
        'n_components': args.mahalanobis_components,
    }


def new_outlier_report(tier1: dict[str, Any], tier2: dict[str, Any]) -> dict[str, Any]:
    return {
        'tier1': {
            'spike_threshold_mm_per_frame': tier1['spike_threshold_mm_per_frame'],
            'filter_cutoff_hz': tier1['filter_cutoff_hz'],
            'trials': {},
        },
        'tier2': {
            'enabled': tier2['enabled'],
            'alpha': tier2['alpha'],
            'n_components': tier2['n_components'],
            'cells': [],
        },
    }


def record_trial_spikes(report: dict[str, Any], subject_id: str, trial: int,
                        spike_report: dict[str, int]) -> None:
    nonzero = {k: v for k, v in spike_report.items() if v > 0}
    if nonzero:
        key = f'{subject_id}_T{trial}'
        report['tier1']['trials'][key] = nonzero


def save_outlier_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)


def print_outlier_summary(report: dict[str, Any]) -> None:
    tier1_trials = report['tier1']['trials']
    total_spikes = sum(sum(v.values()) for v in tier1_trials.values())
    print(f"\nOutlier summary (Tier 1): {total_spikes} spike pair(s) across "
          f"{len(tier1_trials)} trial(s)")
    if tier1_trials:
        for trial_key, markers in sorted(tier1_trials.items()):
            parts = ', '.join(f'{m}:{n}' for m, n in markers.items())
            print(f"  {trial_key}: {parts}")

    if report['tier2']['enabled']:
        cells = report['tier2']['cells']
        total_rejected = sum(c.get('n_rejected', 0) for c in cells)
        n_cells_with_rejects = sum(1 for c in cells if c.get('n_rejected', 0) > 0)
        print(f"Outlier summary (Tier 2): {total_rejected} stride(s) rejected in "
              f"{n_cells_with_rejects} cell(s)")
