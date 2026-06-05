"""
batch_mos_timeseries.py — Batch processing for MoS time-series with subject ×
condition × phase ensemble averaging for SPM analysis.

For each stride, computes MoS_AP and MoS_ML at every frame, then time-
normalizes to 0-100% (101 points). Averages across all strides within each
(subject × condition × phase × side) cell, then writes 8 wide-format CSVs:
  ensemble_mos_<phase>_<direction>.csv  for each phase × {ap, ml}

Each CSV row = one (subject × board × time × side × phase) cell.
Each column = pct_0, pct_1, ..., pct_100.
"""
from __future__ import annotations
import argparse
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd

from . import gait_kinematics as gk
from . import gait_mos as gm

PHASES = ['approach', 'recovery', 'crossing_lead', 'crossing_trail']
DIRECTIONS = ['ap', 'ml']
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
    """
    Compute MoS_AP and MoS_ML at every frame in [hs_start, hs_end], then
    time-normalize to 0-100% (n_pct points).
    
    Returns dict with 'mos_ap_norm' and 'mos_ml_norm' arrays (length n_pct).
    """
    # Vectorized MoS computation across frames
    stoe_x = df[f'{stance_side}TOE_x'].to_numpy()
    shee_y = df[f'{stance_side}HEE_y'].to_numpy()
    
    # AP: positive when XCOM is anterior to stance toe
    mos_ap = xcom[:, 0] - stoe_x
    
    # ML: positive when XCOM is medial to stance heel
    if stance_side == 'L':
        mos_ml = shee_y - xcom[:, 1]
    else:  # R
        mos_ml = xcom[:, 1] - shee_y
    
    # Time-normalize each
    mos_ap_n = gk.time_normalize(mos_ap, hs_start, hs_end, n_pct)
    mos_ml_n = gk.time_normalize(mos_ml, hs_start, hs_end, n_pct)
    
    return dict(mos_ap_norm=mos_ap_n, mos_ml_norm=mos_ml_n)


def process_one_trial(csv_path: Path, trial_row: pd.Series,
                       stride_records: pd.DataFrame,
                       fs: float = 100.0) -> list[dict]:
    """Process one trial; return list of dicts containing per-stride normalized MoS curves."""
    df = gk.load_marker_csv(str(csv_path))
    df = gk.normalize_walking_direction(df)
    df = gk.fill_gaps(df, gm.ALL_BODY_MARKERS, max_gap=100)
    df = gk.reconstruct_pelvis_markers(df)
    
    leg = float(trial_row['leg_length_mm'])
    height = float(trial_row['height_mm'])
    
    com = gm.compute_whole_body_com(df, leg)
    v   = gm.compute_com_velocity(com, fs)
    xcom = gm.compute_xcom(com, v, leg)
    
    out = []
    sub = stride_records[(stride_records['subject_id'] == trial_row['subject_id']) &
                         (stride_records['trial'] == int(trial_row['trial']))]
    
    for _, r in sub.iterrows():
        if r['phase'] == 'unknown':
            continue
        stance_side = 'L' if r['side'] == 'left' else 'R'
        
        curves = compute_mos_timeseries(
            df, com, xcom, stance_side,
            int(r['hs_start_frame']), int(r['hs_end_frame'])
        )
        
        out.append({
            'subject_id': trial_row['subject_id'],
            'group': trial_row['group'],
            'board': trial_row['board'],
            'time':  trial_row['time'],
            'trial': int(trial_row['trial']),
            'side':  r['side'],
            'phase': r['phase'],
            'stride_idx_in_trial': int(r['stride_idx_in_trial']),
            'leg_length_mm': leg,
            'height_mm': height,
            'mos_ap_norm': curves['mos_ap_norm'],
            'mos_ml_norm': curves['mos_ml_norm'],
            # Also store height-normalized versions
            'mos_ap_norm_height': curves['mos_ap_norm'] / height,
            'mos_ml_norm_height': curves['mos_ml_norm'] / height,
        })
    return out


def batch_process(obs_csv: Path, ps_csv: Path, trial_dir: Path,
                  output_dir: Path, verbose: bool = True,
                  filter_trials: set = None,
                  normalize_by_height: bool = True) -> dict:
    """Run batch and write 8 wide-format ensemble CSVs."""
    obs = pd.read_csv(obs_csv)
    ps  = pd.read_csv(ps_csv)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if filter_trials is not None:
        obs = obs[obs.apply(lambda r: (r['subject_id'], int(r['trial'])) in filter_trials, axis=1)].copy()
    
    all_strides = []
    n_found = 0; n_missing = 0; n_errors = 0
    
    for i, trial_row in obs.iterrows():
        if verbose and (i % 50 == 0 or len(obs) <= 20):
            print(f"  [{i+1}/{len(obs)}] {trial_row['subject_id']} T{trial_row['trial']}")
        trial_csv = resolve_trial_csv(trial_dir, trial_row['csv_path'])
        if trial_csv is None:
            n_missing += 1
            continue
        n_found += 1
        try:
            strides = process_one_trial(trial_csv, trial_row, ps)
            all_strides.extend(strides)
        except Exception as e:
            n_errors += 1
            if verbose:
                print(f"    Error: {e}")
    
    if verbose:
        print(f"\nTrials found: {n_found}/{len(obs)}, missing: {n_missing}, errors: {n_errors}")
        print(f"Total strides: {len(all_strides)}")
    
    if not all_strides:
        return {'n_strides': 0}
    
    # Group by (subject × condition × side × phase) — same as kinematics ensemble
    groups = defaultdict(list)
    for s in all_strides:
        key = (s['subject_id'], s['group'], s['board'], s['time'], s['side'], s['phase'])
        groups[key].append(s)
    
    if verbose:
        print(f"Unique (subject × condition × side × phase) cells: {len(groups)}")
    
    # Build wide-format CSVs: one per (phase, direction)
    pct_cols = [f'pct_{i}' for i in range(N_PCT_POINTS)]
    meta_cols = ['subject_id', 'group', 'board', 'time', 'side', 'phase',
                 'n_strides_avg', 'leg_length_mm', 'height_mm']
    
    # Determine which key to average ('mos_ap_norm' or 'mos_ap_norm_height')
    curve_keys_to_export = ['mos_ap_norm', 'mos_ml_norm',
                             'mos_ap_norm_height', 'mos_ml_norm_height']
    
    n_written = 0
    for phase in PHASES:
        for ck in curve_keys_to_export:
            direction = 'ap' if 'ap' in ck else 'ml'
            is_norm = 'height' in ck
            suffix = '_normheight' if is_norm else '_raw'
            
            rows = []
            for (subj, grp, board, time_, side, ph), strides in groups.items():
                if ph != phase: continue
                stack = np.array([s[ck] for s in strides])
                with np.errstate(all='ignore'):
                    mean_curve = np.nanmean(stack, axis=0)
                row = {
                    'subject_id': subj, 'group': grp, 'board': board, 
                    'time': time_, 'side': side, 'phase': ph,
                    'n_strides_avg': len(strides),
                    'leg_length_mm': strides[0]['leg_length_mm'],
                    'height_mm': strides[0]['height_mm'],
                }
                for i, val in enumerate(mean_curve):
                    row[f'pct_{i}'] = val
                rows.append(row)
            
            if not rows:
                continue
            df_out = pd.DataFrame(rows)
            df_out = df_out[meta_cols + pct_cols]
            df_out = df_out.sort_values(['subject_id','board','time','side']).reset_index(drop=True)
            
            fname = f'ensemble_mos_{phase}_{direction}{suffix}.csv'
            df_out.to_csv(output_dir / fname, index=False)
            n_written += 1
            if verbose:
                print(f"  {fname}: {len(df_out)} rows")
    
    return {'n_trials_processed': n_found, 'n_strides_total': len(all_strides),
            'n_cells': len(groups), 'n_csvs': n_written}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--obs-csv', type=Path, required=True,
                   help='Trial manifest (obs_trials.csv)')
    p.add_argument('--ps-csv', type=Path, required=True,
                   help='Per-stride metadata (per_stride_data.csv)')
    p.add_argument('--trial-dir', type=Path, required=True,
                   help='Directory containing corrected marker trial CSVs')
    p.add_argument('--output-dir', type=Path, required=True,
                   help='Output directory for MoS time-series ensemble CSVs')
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
    print(f"trial_dir:  {args.trial_dir}")
    print(f"output_dir: {args.output_dir}\n")
    
    result = batch_process(args.obs_csv, args.ps_csv, args.trial_dir,
                            args.output_dir, verbose=not args.quiet,
                            filter_trials=filter_set)
    
    print(f"\n{'='*60}")
    print("MoS TIME-SERIES ENSEMBLE COMPLETE")
    print(f"{'='*60}")
    for k, v in result.items():
        print(f"  {k}: {v}")


if __name__ == '__main__':
    main()
