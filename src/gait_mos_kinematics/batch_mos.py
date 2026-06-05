"""
batch_mos.py — Batch MoS analysis with subject × condition aggregation.

Output:
  1. mos_all_strides.csv — long format, every stride a row (raw data)
  2. mos_subject_condition.csv — subject × condition × phase aggregated means
     (used for downstream ANOVA / mixed model analysis)

For SPM analysis on MoS time series (rather than discrete event values),
use batch_mos_timeseries.py.
"""

from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from . import gait_mos as gm


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


def batch_process(obs_csv: Path, ps_csv: Path, trial_dir: Path,
                  output_dir: Path, verbose: bool = True,
                  filter_trials: set = None) -> dict:
    """Process all (or filtered) trials, output per-stride and aggregated CSVs."""
    obs = pd.read_csv(obs_csv)
    ps  = pd.read_csv(ps_csv)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if filter_trials is not None:
        obs = obs[obs.apply(lambda r: (r['subject_id'], int(r['trial'])) in filter_trials, axis=1)].copy()
        if verbose:
            print(f"Filtered to {len(obs)} trials")
    
    all_strides = []
    n_found = 0
    n_missing = 0
    n_errors = 0
    
    for i, trial_row in obs.iterrows():
        if verbose and (i % 50 == 0 or len(obs) <= 20):
            print(f"  [{i+1}/{len(obs)}] {trial_row['subject_id']} T{trial_row['trial']}")
        
        trial_csv = resolve_trial_csv(trial_dir, trial_row['csv_path'])
        if trial_csv is None:
            n_missing += 1
            continue
        n_found += 1
        
        try:
            mos_df = gm.process_trial_mos(
                str(trial_csv), ps,
                trial_row['subject_id'], int(trial_row['trial']),
                float(trial_row['leg_length_mm']),
                float(trial_row['height_mm'])
            )
            # Add condition metadata
            mos_df['group'] = trial_row['group']
            mos_df['board'] = trial_row['board']
            mos_df['time']  = trial_row['time']
            all_strides.append(mos_df)
        except Exception as e:
            n_errors += 1
            if verbose:
                print(f"    Error: {e}")
    
    if verbose:
        print(f"\nTrials found:   {n_found}/{len(obs)}")
        print(f"Trials missing: {n_missing}")
        print(f"Processing errors: {n_errors}")
    
    if not all_strides:
        print("No strides processed.")
        return {'n_strides': 0}
    
    long_df = pd.concat(all_strides, ignore_index=True)
    
    # Reorder columns: metadata first, then MoS values
    meta_cols = ['subject_id', 'group', 'board', 'time', 'trial',
                 'side', 'phase', 'stride_idx_in_trial', 'stance_side',
                 'leg_length_mm', 'height_mm', 'step_length_mm', 'step_width_mm']
    mos_cols = [c for c in long_df.columns if c not in meta_cols]
    long_df = long_df[[c for c in meta_cols if c in long_df.columns] + mos_cols]
    
    long_path = output_dir / 'mos_all_strides.csv'
    long_df.to_csv(long_path, index=False)
    if verbose:
        print(f"\nSaved long-format CSV: {long_path}")
        print(f"  Rows: {len(long_df)}")
    
    # ----- Subject × condition × phase aggregation -----
    # Average all numeric MoS columns within each (subject, group, board, time, phase) cell
    # Side is collapsed (both L and R contribute to the average for asymmetric conditions
    # like approach/recovery; for crossing, lead and trail are kept separate via phase).
    numeric_cols = [c for c in long_df.columns 
                    if c.startswith('mos_') or c.startswith('ap_') or c.startswith('ml_')
                    or 'clearance' in c or c.startswith('cross_')]
    
    agg = long_df.groupby(['subject_id','group','board','time','phase'])[numeric_cols].mean().reset_index()
    # Also add stride count per cell
    agg['n_strides'] = long_df.groupby(['subject_id','group','board','time','phase']).size().values
    
    agg_path = output_dir / 'mos_subject_condition.csv'
    agg.to_csv(agg_path, index=False)
    if verbose:
        print(f"Saved aggregated CSV: {agg_path}")
        print(f"  Rows: {len(agg)} (= subject × condition × phase cells)")
    
    return {
        'n_trials_processed': n_found,
        'n_strides_total': len(long_df),
        'n_aggregated_cells': len(agg),
    }


def main():
    p = argparse.ArgumentParser(description="Batch MoS analysis")
    p.add_argument('--obs-csv', type=Path, required=True,
                   help='Trial manifest (obs_trials.csv)')
    p.add_argument('--ps-csv', type=Path, required=True,
                   help='Per-stride metadata (per_stride_data.csv)')
    p.add_argument('--trial-dir', type=Path, required=True,
                   help='Directory containing corrected marker trial CSVs')
    p.add_argument('--output-dir', type=Path, required=True,
                   help='Output directory for MoS CSVs')
    p.add_argument('--filter-trials', type=str, default=None)
    p.add_argument('--quiet', action='store_true')
    args = p.parse_args()
    
    filter_set = None
    if args.filter_trials:
        filter_set = set()
        for item in args.filter_trials.split(','):
            s, t = item.strip().split(':')
            filter_set.add((s.strip(), int(t)))
    
    print(f"obs:        {args.obs_csv}")
    print(f"ps:         {args.ps_csv}")
    print(f"trial_dir:  {args.trial_dir}")
    print(f"output_dir: {args.output_dir}\n")
    
    result = batch_process(args.obs_csv, args.ps_csv, args.trial_dir,
                            args.output_dir, verbose=not args.quiet,
                            filter_trials=filter_set)
    
    print(f"\n{'='*60}")
    print("BATCH MOS PROCESSING COMPLETE")
    print(f"{'='*60}")
    for k, v in result.items():
        print(f"  {k}: {v}")


if __name__ == '__main__':
    main()
