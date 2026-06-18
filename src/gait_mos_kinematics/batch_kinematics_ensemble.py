"""
batch_kinematics_ensemble.py — Batch processing for joint kinematics with
subject × condition × phase ensemble averaging for SPM analysis.

Pipeline:
  1. Loop over all trials in obs_trials.csv
  2. For each trial, run gait_kinematics.process_trial → time-normalized curves
  3. Group strides by (subject_id, board, time, phase, side, joint, signal_type)
  4. Average curves across all trials × strides within each group
  5. Output 36 wide-format CSVs (4 phases × 3 joints × 3 signals)
  6. Output peak summaries to ../peaks/ (peaks_per_stride.csv, peaks_subject_condition.csv)

Each CSV row = one (subject, condition) cell.
Each column = one time point (pct_0 through pct_100).

Requires obs_trials.csv, per_stride_data.csv, and corrected marker trial CSVs.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from collections import defaultdict

from . import gait_kinematics as gk
from .cli_outliers import (
    add_tier1_args, add_tier2_args, tier1_config_from_args, tier2_config_from_args,
    new_outlier_report, record_trial_spikes, save_outlier_report, print_outlier_summary,
)
from .outlier_rejection import mahalanobis_reject

PHASES = ['approach', 'recovery', 'crossing_lead', 'crossing_trail']
JOINTS = ['hip', 'knee', 'ankle']
SIGNALS = ['angle', 'velocity', 'acceleration']
N_PCT_POINTS = 101

PEAK_META_COLS = [
    'subject_id', 'group', 'board', 'time', 'trial',
    'side', 'phase', 'stride_idx_in_trial',
    'hs_start_frame', 'hs_end_frame', 'leg_length_mm',
]
PEAK_COLS = []
for _joint in JOINTS:
    PEAK_COLS.extend([
        f'{_joint}_peak_flexion',
        f'{_joint}_peak_flexion_pct',
        f'{_joint}_peak_extension',
        f'{_joint}_peak_extension_pct',
        f'{_joint}_rom',
        f'{_joint}_peak_velocity',
        f'{_joint}_peak_velocity_pct',
        f'{_joint}_peak_acceleration',
        f'{_joint}_peak_acceleration_pct',
])


def curve_key(joint, signal):
    sig_map = {'angle': 'angle', 'velocity': 'vel', 'acceleration': 'acc'}
    return f'{joint}_{sig_map[signal]}_norm'


def resolve_trial_csv(trial_dir: Path, csv_path_field: str) -> Path | None:
    candidates = [
        trial_dir / csv_path_field,
        trial_dir / csv_path_field.replace('corrected/', ''),
        trial_dir / csv_path_field.replace(' ', '_'),
        trial_dir / csv_path_field.replace(' ', '_').replace('corrected/', ''),
        trial_dir / Path(csv_path_field).name,
        trial_dir / Path(csv_path_field).name.replace(' ', '_'),
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def process_one_trial(trial_csv: Path, trial_row: pd.Series,
                       stride_records: pd.DataFrame,
                       tier1: dict) -> tuple[list[dict], pd.DataFrame, dict[str, int]]:
    summary, curves, spike_report = gk.process_trial(
        str(trial_csv), stride_records,
        trial_row['subject_id'], int(trial_row['trial']),
        leg_length_mm=float(trial_row['leg_length_mm']),
        spike_threshold_mm_per_frame=tier1['spike_threshold_mm_per_frame'],
        filter_cutoff_hz=tier1['filter_cutoff_hz'],
    )

    out = []
    peak_rows = []
    for _, r in summary.iterrows():
        if r['phase'] == 'unknown':
            continue
        stride_id = (trial_row['subject_id'], int(trial_row['trial']),
                     r['side'], int(r['stride_idx_in_trial']))
        if stride_id not in curves:
            continue
        out.append({
            'subject_id': trial_row['subject_id'],
            'group': trial_row['group'],
            'board': trial_row['board'],
            'time': trial_row['time'],
            'trial': int(trial_row['trial']),
            'side': r['side'],
            'phase': r['phase'],
            'stride_idx_in_trial': int(r['stride_idx_in_trial']),
            'curves': curves[stride_id],
        })
        peak_row = {
            'subject_id': trial_row['subject_id'],
            'group': trial_row['group'],
            'board': trial_row['board'],
            'time': trial_row['time'],
            'trial': int(trial_row['trial']),
            'side': r['side'],
            'phase': r['phase'],
            'stride_idx_in_trial': int(r['stride_idx_in_trial']),
            'hs_start_frame': int(r['hs_start_frame']),
            'hs_end_frame': int(r['hs_end_frame']),
            'leg_length_mm': float(trial_row['leg_length_mm']),
        }
        for col in PEAK_COLS:
            peak_row[col] = r.get(col, np.nan)
        peak_rows.append(peak_row)

    peaks_df = pd.DataFrame(peak_rows) if peak_rows else pd.DataFrame(columns=PEAK_META_COLS + PEAK_COLS)
    return out, peaks_df, spike_report


def write_peak_summaries(all_peaks: list[pd.DataFrame], peaks_dir: Path,
                         verbose: bool = True) -> dict:
    """Write peaks_per_stride.csv and peaks_subject_condition.csv."""
    peaks_dir.mkdir(parents=True, exist_ok=True)
    if not all_peaks:
        return {'n_peaks_strides': 0, 'n_peaks_cells': 0}

    per_stride = pd.concat(all_peaks, ignore_index=True)
    peak_cols = [c for c in PEAK_COLS if c in per_stride.columns]
    per_stride = per_stride[PEAK_META_COLS + peak_cols]
    per_stride = per_stride.sort_values(
        ['subject_id', 'board', 'time', 'phase', 'side', 'trial', 'stride_idx_in_trial']
    ).reset_index(drop=True)

    per_stride_path = peaks_dir / 'peaks_per_stride.csv'
    per_stride.to_csv(per_stride_path, index=False)

    group_cols = ['subject_id', 'group', 'board', 'time', 'side', 'phase']
    subject_condition = (
        per_stride.groupby(group_cols, as_index=False)[peak_cols].mean()
    )
    subject_condition['n_strides'] = (
        per_stride.groupby(group_cols).size().values
    )
    subject_condition = subject_condition.sort_values(
        ['subject_id', 'board', 'time', 'phase', 'side']
    ).reset_index(drop=True)

    subject_path = peaks_dir / 'peaks_subject_condition.csv'
    subject_condition.to_csv(subject_path, index=False)

    if verbose:
        print(f"\nWriting peak summaries to {peaks_dir}...")
        print(f"  peaks_per_stride.csv: {len(per_stride)} rows")
        print(f"  peaks_subject_condition.csv: {len(subject_condition)} rows")

    return {
        'n_peaks_strides': len(per_stride),
        'n_peaks_cells': len(subject_condition),
    }


def batch_process(obs_csv: Path, ps_csv: Path, trial_dir: Path,
                  output_dir: Path, verbose: bool = True,
                  tier1: dict | None = None,
                  tier2: dict | None = None) -> dict:
    if tier1 is None:
        tier1 = {'spike_threshold_mm_per_frame': 100.0, 'filter_cutoff_hz': 6.0}
    if tier2 is None:
        tier2 = {'enabled': False, 'alpha': 0.001, 'n_components': 5}

    obs = pd.read_csv(obs_csv)
    ps = pd.read_csv(ps_csv)
    output_dir.mkdir(parents=True, exist_ok=True)
    outlier_report = new_outlier_report(tier1, tier2)

    all_strides = []
    all_peaks: list[pd.DataFrame] = []
    n_found = n_missing = n_errors = 0
    missing_trials = []

    for i, trial_row in obs.iterrows():
        if verbose and i % 50 == 0:
            print(f"  Processing trial {i+1}/{len(obs)}: "
                  f"{trial_row['subject_id']} T{trial_row['trial']}")

        trial_csv = resolve_trial_csv(trial_dir, trial_row['csv_path'])
        if trial_csv is None:
            n_missing += 1
            missing_trials.append((trial_row['subject_id'], int(trial_row['trial'])))
            continue
        n_found += 1

        try:
            stride_dicts, peaks_df, spike_report = process_one_trial(
                trial_csv, trial_row, ps, tier1)
            record_trial_spikes(
                outlier_report, trial_row['subject_id'], int(trial_row['trial']),
                spike_report)
            all_strides.extend(stride_dicts)
            if not peaks_df.empty:
                all_peaks.append(peaks_df)
        except Exception as e:
            n_errors += 1
            if verbose:
                print(f"    Error on {trial_row['subject_id']} "
                      f"T{trial_row['trial']}: {e}")

    if verbose:
        print(f"\nTrials found:   {n_found}/{len(obs)}")
        print(f"Trials missing: {n_missing}")
        print(f"Processing errors: {n_errors}")
        print(f"Total strides collected: {len(all_strides)}")

    groups = defaultdict(list)
    for s in all_strides:
        key = (s['subject_id'], s['group'], s['board'], s['time'], s['side'], s['phase'])
        groups[key].append(s)

    if verbose:
        print(f"\nUnique (subject × condition × side × phase) cells: {len(groups)}")
        from collections import Counter
        phase_counts = Counter(s['phase'] for s in all_strides)
        print(f"Strides per phase: {dict(phase_counts)}")

    pct_cols = [f'pct_{i}' for i in range(N_PCT_POINTS)]
    rows_by_phase_joint_signal = defaultdict(list)

    for (subj, grp, board, time_, side, phase), strides in groups.items():
        for joint in JOINTS:
            for signal in SIGNALS:
                key_curve = curve_key(joint, signal)
                curves = np.array([s['curves'][key_curve] for s in strides])
                n_total = len(curves)

                if tier2['enabled']:
                    keep_mask = mahalanobis_reject(
                        curves, alpha=tier2['alpha'],
                        n_components=tier2['n_components'])
                    curves_kept = curves[keep_mask]
                    n_rejected = int((~keep_mask).sum())
                else:
                    curves_kept = curves
                    n_rejected = 0

                n_kept = len(curves_kept)
                with np.errstate(all='ignore'):
                    mean_curve = np.nanmean(curves_kept, axis=0) if n_kept else np.full(
                        N_PCT_POINTS, np.nan)

                row = {
                    'subject_id': subj,
                    'group': grp,
                    'board': board,
                    'time': time_,
                    'side': side,
                    'phase': phase,
                    'n_strides_avg': n_kept,
                    'n_strides_kept': n_kept,
                    'n_strides_rejected': n_rejected,
                }
                for i, val in enumerate(mean_curve):
                    row[f'pct_{i}'] = val
                rows_by_phase_joint_signal[(phase, joint, signal)].append(row)

                if tier2['enabled']:
                    outlier_report['tier2']['cells'].append({
                        'subject_id': subj,
                        'condition': f'{board}_{time_}',
                        'phase': phase,
                        'side': side,
                        'joint': joint,
                        'signal': signal,
                        'n_strides_total': n_total,
                        'n_rejected': n_rejected,
                    })

    if verbose:
        print(f"\nWriting wide-format CSVs to {output_dir}...")
    n_written = 0
    meta_cols = [
        'subject_id', 'group', 'board', 'time', 'side', 'phase',
        'n_strides_avg', 'n_strides_kept', 'n_strides_rejected',
    ]
    for (phase, joint, signal), rows in rows_by_phase_joint_signal.items():
        df = pd.DataFrame(rows)
        df = df[meta_cols + pct_cols]
        df = df.sort_values(['subject_id', 'board', 'time', 'side']).reset_index(drop=True)
        fname = f'ensemble_{phase}_{joint}_{signal}.csv'
        df.to_csv(output_dir / fname, index=False)
        n_written += 1
        if verbose:
            print(f"  {fname}: {len(df)} rows")

    peaks_dir = output_dir.parent / 'peaks'
    peaks_result = write_peak_summaries(all_peaks, peaks_dir, verbose=verbose)

    report_path = output_dir / 'outlier_report.json'
    save_outlier_report(report_path, outlier_report)
    if verbose:
        print(f"\nSaved outlier report: {report_path}")
        print_outlier_summary(outlier_report)

    return {
        'n_trials_processed': n_found,
        'n_strides': len(all_strides),
        'n_cells': len(groups),
        'n_csvs': n_written,
        'missing_trials': missing_trials,
        'outlier_report': outlier_report,
        **peaks_result,
    }


def main():
    p = argparse.ArgumentParser(description="Batch ensemble kinematics for SPM")
    p.add_argument('--obs-csv', type=Path, required=True)
    p.add_argument('--ps-csv', type=Path, required=True)
    p.add_argument('--trial-dir', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--filter-trials', type=str, default=None)
    p.add_argument('--quiet', action='store_true')
    add_tier1_args(p)
    add_tier2_args(p)
    args = p.parse_args()

    tier1 = tier1_config_from_args(args)
    tier2 = tier2_config_from_args(args)

    print(f"obs_trials:    {args.obs_csv}")
    print(f"per_stride:    {args.ps_csv}")
    print(f"trial_dir:     {args.trial_dir}")
    print(f"output_dir:    {args.output_dir}")
    print(f"Tier 1: spike={tier1['spike_threshold_mm_per_frame']} mm/fr, "
          f"filter={tier1['filter_cutoff_hz']} Hz")
    print(f"Tier 2: mahalanobis={'on' if tier2['enabled'] else 'off'}")

    obs = pd.read_csv(args.obs_csv)
    if args.filter_trials:
        wanted = set()
        for item in args.filter_trials.split(','):
            s, t = item.strip().split(':')
            wanted.add((s.strip(), int(t)))
        before = len(obs)
        obs = obs[obs.apply(
            lambda r: (r['subject_id'], int(r['trial'])) in wanted, axis=1)].copy()
        print(f"Filtered: {before} → {len(obs)} trials")
    print()

    filtered_obs_path = args.output_dir.parent / '_filtered_obs.csv'
    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    obs.to_csv(filtered_obs_path, index=False)

    result = batch_process(
        filtered_obs_path, args.ps_csv, args.trial_dir, args.output_dir,
        verbose=not args.quiet, tier1=tier1, tier2=tier2)

    print(f"\n{'='*60}")
    print("BATCH PROCESSING COMPLETE")
    print(f"{'='*60}")
    print(f"  Trials processed: {result['n_trials_processed']}")
    print(f"  Total strides:    {result['n_strides']}")
    print(f"  Ensemble cells:   {result['n_cells']}")
    print(f"  CSVs written:     {result['n_csvs']}")
    if 'n_peaks_strides' in result:
        print(f"  Peak strides:     {result['n_peaks_strides']}")
        print(f"  Peak cells:       {result['n_peaks_cells']}")
    if result['missing_trials']:
        print(f"  Missing trials:   {len(result['missing_trials'])}")


if __name__ == '__main__':
    main()
