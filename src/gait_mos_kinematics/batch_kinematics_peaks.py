"""
batch_kinematics_peaks.py — Export per-stride joint angle peaks and timings.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from . import gait_kinematics as gk
from .batch_kinematics_ensemble import resolve_trial_csv
from .cli_outliers import (
    add_tier1_args, tier1_config_from_args, new_outlier_report,
    record_trial_spikes, save_outlier_report, print_outlier_summary,
)

META_COLS = [
    'subject_id', 'group', 'board', 'time', 'trial',
    'side', 'phase', 'stride_idx_in_trial',
    'hs_start_frame', 'hs_end_frame', 'leg_length_mm',
]

ANGLE_PEAK_COLS = []
for joint in ('hip', 'knee', 'ankle'):
    ANGLE_PEAK_COLS.extend([
        f'{joint}_peak_flexion',
        f'{joint}_peak_flexion_pct',
        f'{joint}_peak_extension',
        f'{joint}_peak_extension_pct',
        f'{joint}_rom',
    ])


def batch_process(obs_csv: Path, ps_csv: Path, trial_dir: Path,
                  output_dir: Path, verbose: bool = True,
                  filter_trials: set | None = None,
                  tier1: dict | None = None) -> dict:
    if tier1 is None:
        tier1 = {'spike_threshold_mm_per_frame': 100.0, 'filter_cutoff_hz': 6.0}

    obs = pd.read_csv(obs_csv)
    ps = pd.read_csv(ps_csv)
    output_dir.mkdir(parents=True, exist_ok=True)
    tier2 = {'enabled': False, 'alpha': 0.001, 'n_components': 5}
    outlier_report = new_outlier_report(tier1, tier2)

    if filter_trials is not None:
        obs = obs[obs.apply(
            lambda r: (r['subject_id'], int(r['trial'])) in filter_trials, axis=1
        )].copy()
        if verbose:
            print(f"Filtered to {len(obs)} trials")

    all_strides: list[pd.DataFrame] = []
    n_found = n_missing = n_errors = 0

    for i, trial_row in obs.iterrows():
        if verbose and (i % 50 == 0 or len(obs) <= 20):
            print(f"  [{i + 1}/{len(obs)}] {trial_row['subject_id']} "
                  f"T{trial_row['trial']}")

        trial_csv = resolve_trial_csv(trial_dir, trial_row['csv_path'])
        if trial_csv is None:
            n_missing += 1
            continue
        n_found += 1

        try:
            summary, _, spike_report = gk.process_trial(
                str(trial_csv), ps,
                trial_row['subject_id'], int(trial_row['trial']),
                leg_length_mm=float(trial_row['leg_length_mm']),
                spike_threshold_mm_per_frame=tier1['spike_threshold_mm_per_frame'],
                filter_cutoff_hz=tier1['filter_cutoff_hz'],
            )
            record_trial_spikes(
                outlier_report, trial_row['subject_id'], int(trial_row['trial']),
                spike_report)
            if summary.empty:
                continue
            summary = summary[summary['phase'] != 'unknown'].copy()
            summary['group'] = trial_row['group']
            summary['board'] = trial_row['board']
            summary['time'] = trial_row['time']
            all_strides.append(summary)
        except Exception as e:
            n_errors += 1
            if verbose:
                print(f"    Error: {e}")

    if verbose:
        print(f"\nTrials found:   {n_found}/{len(obs)}")
        print(f"Trials missing: {n_missing}")
        print(f"Processing errors: {n_errors}")

    if not all_strides:
        save_outlier_report(output_dir / 'outlier_report.json', outlier_report)
        return {'n_strides': 0}

    long_df = pd.concat(all_strides, ignore_index=True)
    peak_cols = [c for c in ANGLE_PEAK_COLS if c in long_df.columns]
    long_df = long_df[[c for c in META_COLS if c in long_df.columns] + peak_cols]

    long_path = output_dir / 'kinematics_all_strides.csv'
    long_df.to_csv(long_path, index=False)
    if verbose:
        print(f"\nSaved long-format CSV: {long_path}")
        print(f"  Rows: {len(long_df)}")

    agg = (
        long_df.groupby(['subject_id', 'group', 'board', 'time', 'phase'])[peak_cols]
        .mean()
        .reset_index()
    )
    agg['n_strides'] = (
        long_df.groupby(['subject_id', 'group', 'board', 'time', 'phase'])
        .size()
        .values
    )

    agg_path = output_dir / 'kinematics_subject_condition.csv'
    agg.to_csv(agg_path, index=False)
    if verbose:
        print(f"Saved aggregated CSV: {agg_path}")
        print(f"  Rows: {len(agg)}")

    save_outlier_report(output_dir / 'outlier_report.json', outlier_report)
    print_outlier_summary(outlier_report)

    return {
        'n_trials_processed': n_found,
        'n_strides_total': len(long_df),
        'n_aggregated_cells': len(agg),
        'outlier_report': outlier_report,
    }


def main():
    p = argparse.ArgumentParser(description="Batch joint angle peak export")
    p.add_argument('--obs-csv', type=Path, required=True)
    p.add_argument('--ps-csv', type=Path, required=True)
    p.add_argument('--trial-dir', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--filter-trials', type=str, default=None)
    p.add_argument('--quiet', action='store_true')
    add_tier1_args(p)
    args = p.parse_args()

    tier1 = tier1_config_from_args(args)
    filter_set = None
    if args.filter_trials:
        filter_set = set()
        for item in args.filter_trials.split(','):
            s, t = item.strip().split(':')
            filter_set.add((s.strip(), int(t)))

    result = batch_process(
        args.obs_csv, args.ps_csv, args.trial_dir, args.output_dir,
        verbose=not args.quiet, filter_trials=filter_set, tier1=tier1,
    )
    print(f"\n{'=' * 60}")
    print("BATCH KINEMATICS PEAKS COMPLETE")
    print(f"{'=' * 60}")
    for k, v in result.items():
        if k != 'outlier_report':
            print(f"  {k}: {v}")


if __name__ == '__main__':
    main()
