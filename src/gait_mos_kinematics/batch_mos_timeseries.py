"""
batch_mos_timeseries.py — Batch processing for MoS time-series with subject ×
condition × phase ensemble averaging for SPM analysis.
"""
from __future__ import annotations
import argparse
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd

from . import gait_kinematics as gk
from . import gait_mos as gm
from .cli_outliers import (
    add_tier1_args, add_tier2_args, tier1_config_from_args, tier2_config_from_args,
    new_outlier_report, record_trial_spikes, save_outlier_report, print_outlier_summary,
)
from .outlier_rejection import mahalanobis_reject

PHASES = ['approach', 'recovery', 'crossing_lead', 'crossing_trail']
N_PCT_POINTS = 101


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


def compute_mos_timeseries(df: pd.DataFrame, com: np.ndarray, xcom: np.ndarray,
                            stance_side: str, hs_start: int, hs_end: int,
                            n_pct: int = N_PCT_POINTS) -> dict:
    stoe_x = df[f'{stance_side}TOE_x'].to_numpy()
    shee_y = df[f'{stance_side}HEE_y'].to_numpy()
    mos_ap = xcom[:, 0] - stoe_x
    if stance_side == 'L':
        mos_ml = shee_y - xcom[:, 1]
    else:
        mos_ml = xcom[:, 1] - shee_y
    mos_ap_n = gk.time_normalize(mos_ap, hs_start, hs_end, n_pct)
    mos_ml_n = gk.time_normalize(mos_ml, hs_start, hs_end, n_pct)
    return dict(mos_ap_norm=mos_ap_n, mos_ml_norm=mos_ml_n)


def process_one_trial(csv_path: Path, trial_row: pd.Series,
                       stride_records: pd.DataFrame,
                       tier1: dict, fs: float = 100.0
                       ) -> tuple[list[dict], dict[str, int]]:
    df = gk.load_marker_csv(str(csv_path))
    df = gk.normalize_walking_direction(df)
    df, spike_report = gk.preprocess_markers(
        df, gm.ALL_BODY_MARKERS,
        spike_threshold_mm_per_frame=tier1['spike_threshold_mm_per_frame'],
        filter_cutoff_hz=tier1['filter_cutoff_hz'], fs=fs, max_gap=100)

    leg = float(trial_row['leg_length_mm'])
    height = float(trial_row['height_mm'])
    com = gm.compute_whole_body_com(df, leg)
    v = gm.compute_com_velocity(com, fs)
    xcom = gm.compute_xcom(com, v, leg)

    out = []
    sub = stride_records[
        (stride_records['subject_id'] == trial_row['subject_id']) &
        (stride_records['trial'] == int(trial_row['trial']))
    ]
    for _, r in sub.iterrows():
        if r['phase'] == 'unknown':
            continue
        stance_side = 'L' if r['side'] == 'left' else 'R'
        curves = compute_mos_timeseries(
            df, com, xcom, stance_side,
            int(r['hs_start_frame']), int(r['hs_end_frame']))
        out.append({
            'subject_id': trial_row['subject_id'],
            'group': trial_row['group'],
            'board': trial_row['board'],
            'time': trial_row['time'],
            'trial': int(trial_row['trial']),
            'side': r['side'],
            'phase': r['phase'],
            'stride_idx_in_trial': int(r['stride_idx_in_trial']),
            'leg_length_mm': leg,
            'height_mm': height,
            'mos_ap_norm': curves['mos_ap_norm'],
            'mos_ml_norm': curves['mos_ml_norm'],
            'mos_ap_norm_height': curves['mos_ap_norm'] / height,
            'mos_ml_norm_height': curves['mos_ml_norm'] / height,
        })
    return out, spike_report


def batch_process(obs_csv: Path, ps_csv: Path, trial_dir: Path,
                  output_dir: Path, verbose: bool = True,
                  filter_trials: set | None = None,
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

    if filter_trials is not None:
        obs = obs[obs.apply(
            lambda r: (r['subject_id'], int(r['trial'])) in filter_trials, axis=1)].copy()

    all_strides = []
    n_found = n_missing = n_errors = 0

    for i, trial_row in obs.iterrows():
        if verbose and (i % 50 == 0 or len(obs) <= 20):
            print(f"  [{i+1}/{len(obs)}] {trial_row['subject_id']} "
                  f"T{trial_row['trial']}")
        trial_csv = resolve_trial_csv(trial_dir, trial_row['csv_path'])
        if trial_csv is None:
            n_missing += 1
            continue
        n_found += 1
        try:
            strides, spike_report = process_one_trial(
                trial_csv, trial_row, ps, tier1)
            record_trial_spikes(
                outlier_report, trial_row['subject_id'], int(trial_row['trial']),
                spike_report)
            all_strides.extend(strides)
        except Exception as e:
            n_errors += 1
            if verbose:
                print(f"    Error: {e}")

    if verbose:
        print(f"\nTrials found: {n_found}/{len(obs)}, missing: {n_missing}, "
              f"errors: {n_errors}")
        print(f"Total strides: {len(all_strides)}")

    if not all_strides:
        save_outlier_report(output_dir / 'outlier_report.json', outlier_report)
        return {'n_strides': 0}

    groups = defaultdict(list)
    for s in all_strides:
        key = (s['subject_id'], s['group'], s['board'], s['time'], s['side'], s['phase'])
        groups[key].append(s)

    pct_cols = [f'pct_{i}' for i in range(N_PCT_POINTS)]
    meta_cols = [
        'subject_id', 'group', 'board', 'time', 'side', 'phase',
        'n_strides_avg', 'n_strides_kept', 'n_strides_rejected',
        'leg_length_mm', 'height_mm',
    ]
    curve_keys_to_export = [
        ('mos_ap_norm', 'ap', 'raw'),
        ('mos_ml_norm', 'ml', 'raw'),
        ('mos_ap_norm_height', 'ap', 'normheight'),
        ('mos_ml_norm_height', 'ml', 'normheight'),
    ]

    n_written = 0
    for phase in PHASES:
        for ck, direction, suffix in curve_keys_to_export:
            rows = []
            for (subj, grp, board, time_, side, ph), strides in groups.items():
                if ph != phase:
                    continue
                curves = np.array([s[ck] for s in strides])
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
                    'subject_id': subj, 'group': grp, 'board': board,
                    'time': time_, 'side': side, 'phase': ph,
                    'n_strides_avg': n_kept,
                    'n_strides_kept': n_kept,
                    'n_strides_rejected': n_rejected,
                    'leg_length_mm': strides[0]['leg_length_mm'],
                    'height_mm': strides[0]['height_mm'],
                }
                for i, val in enumerate(mean_curve):
                    row[f'pct_{i}'] = val
                rows.append(row)

                if tier2['enabled']:
                    outlier_report['tier2']['cells'].append({
                        'subject_id': subj,
                        'condition': f'{board}_{time_}',
                        'phase': ph,
                        'side': side,
                        'curve_key': ck,
                        'direction': direction,
                        'normalization': suffix,
                        'n_strides_total': n_total,
                        'n_rejected': n_rejected,
                    })

            if not rows:
                continue
            df_out = pd.DataFrame(rows)
            df_out = df_out[meta_cols + pct_cols]
            df_out = df_out.sort_values(
                ['subject_id', 'board', 'time', 'side']).reset_index(drop=True)
            fname = f'ensemble_mos_{phase}_{direction}_{suffix}.csv'
            df_out.to_csv(output_dir / fname, index=False)
            n_written += 1
            if verbose:
                print(f"  {fname}: {len(df_out)} rows")

    report_path = output_dir / 'outlier_report.json'
    save_outlier_report(report_path, outlier_report)
    if verbose:
        print(f"\nSaved outlier report: {report_path}")
        print_outlier_summary(outlier_report)

    return {
        'n_trials_processed': n_found,
        'n_strides_total': len(all_strides),
        'n_cells': len(groups),
        'n_csvs': n_written,
        'outlier_report': outlier_report,
    }


def main():
    p = argparse.ArgumentParser()
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

    filter_set = None
    if args.filter_trials:
        filter_set = set()
        for item in args.filter_trials.split(','):
            s, t = item.strip().split(':')
            filter_set.add((s.strip(), int(t)))

    print(f"obs:        {args.obs_csv}")
    print(f"trial_dir:  {args.trial_dir}")
    print(f"output_dir: {args.output_dir}")
    print(f"Tier 1: spike={tier1['spike_threshold_mm_per_frame']} mm/fr, "
          f"filter={tier1['filter_cutoff_hz']} Hz")
    print(f"Tier 2: mahalanobis={'on' if tier2['enabled'] else 'off'}\n")

    result = batch_process(
        args.obs_csv, args.ps_csv, args.trial_dir, args.output_dir,
        verbose=not args.quiet, filter_trials=filter_set,
        tier1=tier1, tier2=tier2)

    print(f"\n{'='*60}")
    print("MoS TIME-SERIES ENSEMBLE COMPLETE")
    print(f"{'='*60}")
    for k, v in result.items():
        if k != 'outlier_report':
            print(f"  {k}: {v}")


if __name__ == '__main__':
    main()
