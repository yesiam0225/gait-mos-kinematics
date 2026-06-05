"""
batch_kinematics_ensemble.py — Batch processing for joint kinematics with
subject × condition × phase ensemble averaging for SPM analysis.

Pipeline:
  1. Loop over all trials in obs_trials.csv
  2. For each trial, run gait_kinematics.process_trial → time-normalized curves
  3. Group strides by (subject_id, board, time, phase, side, joint, signal_type)
  4. Average curves across all trials × strides within each group
  5. Output 36 wide-format CSVs (4 phases × 3 joints × 3 signals)
  6. Bonus: 1 long-format CSV with all data

Each CSV row = one (subject, condition) cell.
Each column = one time point (pct_0 through pct_100).

Usage (local):
    python batch_kinematics_ensemble.py \\
        --obs-csv path/to/obs_trials.csv \\
        --ps-csv path/to/per_stride_data.csv \\
        --trial-dir path/to/corrected/ \\
        --output-dir path/to/output/

Requires obs_trials.csv, per_stride_data.csv, and corrected marker trial CSVs.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from collections import defaultdict

from . import gait_kinematics as gk

# ----------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------

PHASES = ['approach', 'recovery', 'crossing_lead', 'crossing_trail']
JOINTS = ['hip', 'knee', 'ankle']
SIGNALS = ['angle', 'velocity', 'acceleration']
N_PCT_POINTS = 101  # 0% to 100% in 1% steps

# Curve key for the curves dict from process_trial
def curve_key(joint, signal):
    """Map (joint, signal) → key in curves dict, e.g. 'hip_angle_norm'."""
    sig_map = {'angle': 'angle', 'velocity': 'vel', 'acceleration': 'acc'}
    return f'{joint}_{sig_map[signal]}_norm'


# ----------------------------------------------------------------------------
# Trial path resolution
# ----------------------------------------------------------------------------

def resolve_trial_csv(trial_dir: Path, csv_path_field: str) -> Path | None:
    """
    Resolve trial CSV path. The obs_trials.csv csv_path field uses
    'corrected/BBA01 Trial 05_corrected.csv' (with space), but actual files
    may be 'BBA01_Trial_05_corrected.csv' (with underscores) at trial_dir root.
    Tries multiple naming conventions.
    """
    candidates = [
        trial_dir / csv_path_field,                                # exact
        trial_dir / csv_path_field.replace('corrected/', ''),      # no subdir
        trial_dir / csv_path_field.replace(' ', '_'),              # spaces → _
        trial_dir / csv_path_field.replace(' ', '_').replace('corrected/', ''),
        trial_dir / Path(csv_path_field).name,                     # basename only
        trial_dir / Path(csv_path_field).name.replace(' ', '_'),   # basename, _
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


# ----------------------------------------------------------------------------
# Per-trial processing → strides with phase, side, condition, curves
# ----------------------------------------------------------------------------

def process_one_trial(trial_csv: Path, trial_row: pd.Series,
                       stride_records: pd.DataFrame) -> list[dict]:
    """
    Process one trial. Returns list of stride dicts:
      [{condition: {...}, phase, side, joint, signal: curve_array, ...}, ...]
    Each dict represents one stride's contribution to ensemble pools.
    """
    summary, curves = gk.process_trial(
        str(trial_csv), stride_records,
        trial_row['subject_id'], int(trial_row['trial']),
        leg_length_mm=float(trial_row['leg_length_mm'])
    )
    
    out = []
    for _, r in summary.iterrows():
        # Skip phase == 'unknown' (obstacle detection failure)
        if r['phase'] == 'unknown':
            continue
        
        stride_id = (trial_row['subject_id'], int(trial_row['trial']),
                     r['side'], int(r['stride_idx_in_trial']))
        if stride_id not in curves:
            continue
        curve_dict = curves[stride_id]
        
        out.append({
            'subject_id': trial_row['subject_id'],
            'group': trial_row['group'],
            'board': trial_row['board'],
            'time': trial_row['time'],
            'trial': int(trial_row['trial']),
            'side': r['side'],
            'phase': r['phase'],
            'stride_idx_in_trial': int(r['stride_idx_in_trial']),
            'curves': curve_dict,  # {'hip_angle_norm': array, 'hip_vel_norm': array, ...}
        })
    return out


# ----------------------------------------------------------------------------
# Main batch processing
# ----------------------------------------------------------------------------

def batch_process(obs_csv: Path, ps_csv: Path, trial_dir: Path,
                  output_dir: Path, verbose: bool = True) -> dict:
    """
    Process all trials. Returns ensemble curves and writes 36 wide CSVs.
    """
    obs = pd.read_csv(obs_csv)
    ps  = pd.read_csv(ps_csv)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Collect per-stride data across all trials
    all_strides = []
    n_found = 0
    n_missing = 0
    missing_trials = []
    n_errors = 0
    
    for i, trial_row in obs.iterrows():
        if verbose and i % 50 == 0:
            print(f"  Processing trial {i+1}/{len(obs)}: {trial_row['subject_id']} T{trial_row['trial']}")
        
        trial_csv = resolve_trial_csv(trial_dir, trial_row['csv_path'])
        if trial_csv is None:
            n_missing += 1
            missing_trials.append((trial_row['subject_id'], int(trial_row['trial'])))
            continue
        n_found += 1
        
        try:
            stride_dicts = process_one_trial(trial_csv, trial_row, ps)
            all_strides.extend(stride_dicts)
        except Exception as e:
            n_errors += 1
            if verbose:
                print(f"    Error on {trial_row['subject_id']} T{trial_row['trial']}: {e}")
    
    if verbose:
        print(f"\nTrials found:   {n_found}/{len(obs)}")
        print(f"Trials missing: {n_missing}")
        print(f"Processing errors: {n_errors}")
        print(f"Total strides collected: {len(all_strides)}")
    
    # ---- Ensemble averaging: group by (subject × condition × side × phase) ----
    # For crossing phases, 'side' isn't meaningful as a group identifier the same
    # way (it just tells which leg led/trailed). But we still average per side.
    # Group key: (subject, group, board, time, side, phase)
    
    groups = defaultdict(list)
    for s in all_strides:
        key = (s['subject_id'], s['group'], s['board'], s['time'], s['side'], s['phase'])
        groups[key].append(s)
    
    if verbose:
        print(f"\nUnique (subject × condition × side × phase) cells: {len(groups)}")
        from collections import Counter
        phase_counts = Counter(s['phase'] for s in all_strides)
        print(f"Strides per phase: {dict(phase_counts)}")
    
    # ---- Build output CSVs ----
    # For each (joint, signal) → one CSV with phase rows
    pct_cols = [f'pct_{i}' for i in range(N_PCT_POINTS)]
    
    rows_by_phase_joint_signal = defaultdict(list)  # (phase, joint, signal) → [row_dicts]
    
    for (subj, grp, board, time_, side, phase), strides in groups.items():
        for joint in JOINTS:
            for signal in SIGNALS:
                key_curve = curve_key(joint, signal)
                # Stack all curves for this group, average across strides
                stack = np.array([s['curves'][key_curve] for s in strides])
                # Handle NaNs row-wise
                with np.errstate(all='ignore'):
                    mean_curve = np.nanmean(stack, axis=0)
                
                row = {
                    'subject_id': subj,
                    'group': grp,
                    'board': board,
                    'time': time_,
                    'side': side,
                    'phase': phase,
                    'n_strides_avg': len(strides),
                }
                for i, val in enumerate(mean_curve):
                    row[f'pct_{i}'] = val
                
                rows_by_phase_joint_signal[(phase, joint, signal)].append(row)
    
    # Write 36 wide-format CSVs
    if verbose:
        print(f"\nWriting wide-format CSVs to {output_dir}...")
    n_written = 0
    for (phase, joint, signal), rows in rows_by_phase_joint_signal.items():
        df = pd.DataFrame(rows)
        # Reorder columns
        meta_cols = ['subject_id', 'group', 'board', 'time', 'side', 'phase', 'n_strides_avg']
        df = df[meta_cols + pct_cols]
        df = df.sort_values(['subject_id', 'board', 'time', 'side']).reset_index(drop=True)
        
        fname = f'ensemble_{phase}_{joint}_{signal}.csv'
        df.to_csv(output_dir / fname, index=False)
        n_written += 1
        if verbose:
            print(f"  {fname}: {len(df)} rows")
    
    if verbose:
        print(f"\nTotal CSVs written: {n_written}")
    
    return {
        'n_trials_processed': n_found,
        'n_strides': len(all_strides),
        'n_cells': len(groups),
        'n_csvs': n_written,
        'missing_trials': missing_trials,
    }


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Batch ensemble kinematics for SPM")
    p.add_argument('--obs-csv', type=Path, required=True,
                   help='Trial manifest (obs_trials.csv)')
    p.add_argument('--ps-csv', type=Path, required=True,
                   help='Per-stride metadata (per_stride_data.csv)')
    p.add_argument('--trial-dir', type=Path, required=True,
                   help='Directory containing corrected marker trial CSVs')
    p.add_argument('--output-dir', type=Path, required=True,
                   help='Output directory for ensemble wide-format CSVs')
    p.add_argument('--filter-trials', type=str, default=None,
                   help="Comma-separated 'subject:trial' pairs to process, "
                        "e.g. 'BBA01:5,BBA01:23,BBA02:5'. If omitted, processes all.")
    p.add_argument('--quiet', action='store_true')
    args = p.parse_args()
    
    print(f"obs_trials:    {args.obs_csv}")
    print(f"per_stride:    {args.ps_csv}")
    print(f"trial_dir:     {args.trial_dir}")
    print(f"output_dir:    {args.output_dir}")
    
    # Load obs and optionally filter
    obs = pd.read_csv(args.obs_csv)
    if args.filter_trials:
        wanted = set()
        for item in args.filter_trials.split(','):
            s, t = item.strip().split(':')
            wanted.add((s.strip(), int(t)))
        before = len(obs)
        obs = obs[obs.apply(lambda r: (r['subject_id'], int(r['trial'])) in wanted, axis=1)].copy()
        print(f"Filtered: {before} → {len(obs)} trials")
    print()
    
    # Save filtered obs to temp for batch_process
    filtered_obs_path = args.output_dir.parent / '_filtered_obs.csv'
    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    obs.to_csv(filtered_obs_path, index=False)
    
    result = batch_process(filtered_obs_path, args.ps_csv, args.trial_dir,
                            args.output_dir, verbose=not args.quiet)
    
    print(f"\n{'='*60}")
    print("BATCH PROCESSING COMPLETE")
    print(f"{'='*60}")
    print(f"  Trials processed: {result['n_trials_processed']}")
    print(f"  Total strides:    {result['n_strides']}")
    print(f"  Ensemble cells:   {result['n_cells']}")
    print(f"  CSVs written:     {result['n_csvs']}")
    if result['missing_trials']:
        print(f"  Missing trials:   {len(result['missing_trials'])}")

if __name__ == '__main__':
    main()
