"""
gait_mos.py — Margin of Stability (MoS) analysis for obstacle-walking gait.

Implements:
  - Whole-body COM via Dempster anthropometric segment masses (14 segments)
  - COM velocity via central differences
  - XCOM = COM + v_COM / omega_0
  - omega_0 = sqrt(g / leg_length_m)
  - MoS = XCOM - BoS (AP and ML)
    * BoS AP edge = stance leg toe
    * BoS ML edge = stance leg heel
  - 3 gait events per stride: heel strike, mid-swing, foot-off
    * Mid-swing = swing toe AP position passes stance toe AP position
  - MoS normalized by subject height
  - AP/ML clearance = MoS_HS - step_length / step_width

Sign convention:
  - Positive AP MoS: XCOM is anterior to BoS edge (more forward than stance toe)
  - Positive ML MoS: XCOM is medial to BoS edge (toward midline from stance heel)
"""

from __future__ import annotations
import numpy as np
import pandas as pd

# Reuse loading helpers from kinematics module
from .gait_kinematics import (
    load_marker_csv, fill_gaps, get_marker_3d, normalize_walking_direction,
    reconstruct_pelvis_markers, compute_hjc, PIG_LOWER_MARKERS
)

G = 9.81  # m/s^2

# ============================================================================
# Dempster segment masses and COM-from-proximal ratios
# ============================================================================
# Mass fractions sum to ~0.99 (close enough for whole-body COM).
# COM ratio = fraction of segment length from PROXIMAL end where the segment
# COM is located. (e.g., thigh COM is 0.433 from hip toward knee.)

DEMPSTER = {
    'head':       dict(mass=0.0810, com_ratio=0.500),  # head & neck
    'trunk':      dict(mass=0.4970, com_ratio=0.500),  # thorax + abdomen + pelvis
    'upper_arm':  dict(mass=0.0280, com_ratio=0.436),  # each side
    'forearm':    dict(mass=0.0160, com_ratio=0.430),
    'hand':       dict(mass=0.0060, com_ratio=0.506),
    'thigh':      dict(mass=0.1000, com_ratio=0.433),
    'shank':      dict(mass=0.0465, com_ratio=0.433),
    'foot':       dict(mass=0.0145, com_ratio=0.500),
}

UPPER_BODY_MARKERS = ['C7','CLAV','T10','STRN','RBAK',
                      'LFHD','RFHD','LBHD','RBHD',
                      'LSHO','RSHO','LELB','RELB','LWRA','RWRA','LWRB','RWRB',
                      'LFIN','RFIN','LUPA','RUPA','LFRM','RFRM']

ALL_BODY_MARKERS = PIG_LOWER_MARKERS + UPPER_BODY_MARKERS


# ============================================================================
# Whole-body COM via segment summation
# ============================================================================

def compute_segment_com(proximal: np.ndarray, distal: np.ndarray,
                         com_ratio: float) -> np.ndarray:
    """COM = proximal + ratio * (distal - proximal)"""
    return proximal + com_ratio * (distal - proximal)


def compute_whole_body_com(df: pd.DataFrame, leg_length_mm: float) -> np.ndarray:
    """
    Compute whole-body COM in lab coordinates (Nx3) via Dempster segment masses.
    Requires full PiG upper + lower body markers.
    
    Segments:
      Head, Trunk, L/R Upper-arm, L/R Forearm, L/R Hand,
      L/R Thigh, L/R Shank, L/R Foot
    """
    # Lower body — HJC for hip joint centers, KNE/ANK/HEE/TOE for distal endpoints
    hjc_L = compute_hjc(df, 'L', leg_length_mm)
    hjc_R = compute_hjc(df, 'R', leg_length_mm)
    LKNE = get_marker_3d(df, 'LKNE'); RKNE = get_marker_3d(df, 'RKNE')
    LANK = get_marker_3d(df, 'LANK'); RANK = get_marker_3d(df, 'RANK')
    LHEE = get_marker_3d(df, 'LHEE'); RHEE = get_marker_3d(df, 'RHEE')
    LTOE = get_marker_3d(df, 'LTOE'); RTOE = get_marker_3d(df, 'RTOE')
    
    # Upper body — joint center approximations
    # Head center: average of 4 head markers
    head_center = (get_marker_3d(df, 'LFHD') + get_marker_3d(df, 'RFHD') +
                   get_marker_3d(df, 'LBHD') + get_marker_3d(df, 'RBHD')) / 4.0
    # Trunk: bounded by C7 (top) and pelvis center (bottom)
    C7 = get_marker_3d(df, 'C7')
    pelv_top = C7  # use C7 as trunk proximal
    asis_mid = (get_marker_3d(df, 'LASI') + get_marker_3d(df, 'RASI')) / 2
    psi_mid  = (get_marker_3d(df, 'LPSI') + get_marker_3d(df, 'RPSI')) / 2
    pelv_center = (asis_mid + psi_mid) / 2
    
    # Shoulders, elbows, wrists, hands
    LSHO = get_marker_3d(df, 'LSHO'); RSHO = get_marker_3d(df, 'RSHO')
    LELB = get_marker_3d(df, 'LELB'); RELB = get_marker_3d(df, 'RELB')
    # Wrist center: midpoint of WRA and WRB
    LWR = (get_marker_3d(df, 'LWRA') + get_marker_3d(df, 'LWRB')) / 2
    RWR = (get_marker_3d(df, 'RWRA') + get_marker_3d(df, 'RWRB')) / 2
    LFIN = get_marker_3d(df, 'LFIN'); RFIN = get_marker_3d(df, 'RFIN')
    
    segments = {
        'head':         (head_center, pelv_center),  # use head→pelvis as "head segment" 
                                                     # but Dempster head COM is ~at head center
        # Actually for head we should use head center alone since its COM is right there.
        # Re-define properly:
    }
    
    # Compute each segment's COM
    coms = {}
    
    # Head — COM is roughly at head center (ratio 0.5 means midpoint of proximal-distal,
    # but for head we use head center directly as the COM)
    coms['head'] = head_center
    
    # Trunk — proximal = C7, distal = pelvis center
    coms['trunk'] = compute_segment_com(C7, pelv_center, DEMPSTER['trunk']['com_ratio'])
    
    # Upper arms
    coms['L_upper_arm'] = compute_segment_com(LSHO, LELB, DEMPSTER['upper_arm']['com_ratio'])
    coms['R_upper_arm'] = compute_segment_com(RSHO, RELB, DEMPSTER['upper_arm']['com_ratio'])
    
    # Forearms
    coms['L_forearm'] = compute_segment_com(LELB, LWR, DEMPSTER['forearm']['com_ratio'])
    coms['R_forearm'] = compute_segment_com(RELB, RWR, DEMPSTER['forearm']['com_ratio'])
    
    # Hands
    coms['L_hand'] = compute_segment_com(LWR, LFIN, DEMPSTER['hand']['com_ratio'])
    coms['R_hand'] = compute_segment_com(RWR, RFIN, DEMPSTER['hand']['com_ratio'])
    
    # Thighs
    coms['L_thigh'] = compute_segment_com(hjc_L, LKNE, DEMPSTER['thigh']['com_ratio'])
    coms['R_thigh'] = compute_segment_com(hjc_R, RKNE, DEMPSTER['thigh']['com_ratio'])
    
    # Shanks
    coms['L_shank'] = compute_segment_com(LKNE, LANK, DEMPSTER['shank']['com_ratio'])
    coms['R_shank'] = compute_segment_com(RKNE, RANK, DEMPSTER['shank']['com_ratio'])
    
    # Feet — proximal = HEE, distal = TOE
    coms['L_foot'] = compute_segment_com(LHEE, LTOE, DEMPSTER['foot']['com_ratio'])
    coms['R_foot'] = compute_segment_com(RHEE, RTOE, DEMPSTER['foot']['com_ratio'])
    
    # Masses (sum should be ~0.99-1.00)
    masses = {
        'head':         DEMPSTER['head']['mass'],
        'trunk':        DEMPSTER['trunk']['mass'],
        'L_upper_arm':  DEMPSTER['upper_arm']['mass'],
        'R_upper_arm':  DEMPSTER['upper_arm']['mass'],
        'L_forearm':    DEMPSTER['forearm']['mass'],
        'R_forearm':    DEMPSTER['forearm']['mass'],
        'L_hand':       DEMPSTER['hand']['mass'],
        'R_hand':       DEMPSTER['hand']['mass'],
        'L_thigh':      DEMPSTER['thigh']['mass'],
        'R_thigh':      DEMPSTER['thigh']['mass'],
        'L_shank':      DEMPSTER['shank']['mass'],
        'R_shank':      DEMPSTER['shank']['mass'],
        'L_foot':       DEMPSTER['foot']['mass'],
        'R_foot':       DEMPSTER['foot']['mass'],
    }
    total_mass = sum(masses.values())
    
    # Weighted sum
    com = np.zeros_like(coms['head'])
    for seg_name, seg_com in coms.items():
        com += masses[seg_name] * seg_com
    com /= total_mass
    
    return com


def compute_com_velocity(com: np.ndarray, fs: float = 100.0) -> np.ndarray:
    """Velocity via central differences. Returns Nx3 (mm/s)."""
    n = len(com)
    v = np.full_like(com, np.nan, dtype=float)
    dt = 1.0 / fs
    v[1:-1] = (com[2:] - com[:-2]) / (2 * dt)
    v[0]  = (com[1] - com[0]) / dt
    v[-1] = (com[-1] - com[-2]) / dt
    return v


# ============================================================================
# XCOM and MoS
# ============================================================================

def compute_xcom(com: np.ndarray, v_com: np.ndarray, leg_length_mm: float
                  ) -> np.ndarray:
    """
    XCOM = COM + v_COM / omega_0,  where omega_0 = sqrt(g/l)
    com/v_com in mm or mm/s, leg_length_mm in mm.
    omega_0 has units 1/s, so v/omega_0 has units mm.
    """
    l_m = leg_length_mm / 1000.0  # convert to meters for omega_0
    omega_0 = np.sqrt(G / l_m)
    return com + v_com / omega_0


def compute_mos(xcom_ap: float, xcom_ml: float,
                 stance_toe_ap: float, stance_heel_ml: float,
                 stance_side: str) -> dict:
    """
    Compute MoS in AP and ML directions for a single time point.
    
    Sign convention (per spec):
      - MoS_AP positive: XCOM is anterior to (in front of) stance toe.
        mos_ap = xcom_ap - stance_toe_ap
      - MoS_ML positive: XCOM is medial to stance heel.
        For LEFT stance (lateral edge at +Y in our coord): 
          medial direction = -Y, so XCOM medial means XCOM_Y < heel_Y
          mos_ml = stance_heel_y - xcom_y
        For RIGHT stance (lateral edge at -Y): 
          medial direction = +Y, so XCOM medial means XCOM_Y > heel_Y
          mos_ml = xcom_y - stance_heel_y
    
    Returns dict with keys: mos_ap, mos_ml.
    """
    # AP: positive when XCOM is anterior to (forward of) stance toe
    mos_ap = xcom_ap - stance_toe_ap
    
    # ML: positive when XCOM is medial to stance heel
    if stance_side == 'L':
        mos_ml = stance_heel_ml - xcom_ml
    else:  # R stance
        mos_ml = xcom_ml - stance_heel_ml
    
    return dict(mos_ap=mos_ap, mos_ml=mos_ml)


# ============================================================================
# Event detection — mid-swing
# ============================================================================

def find_mid_swing_frame(df: pd.DataFrame, swing_side: str, stance_side: str,
                          hs_start_frame: int, hs_end_frame: int) -> int | None:
    """
    Find the frame within [hs_start_frame, hs_end_frame] where the swing leg
    toe's AP coordinate (X) equals the stance leg toe's AP coordinate.
    
    Method: find sign change in (swing_toe_x - stance_toe_x).
    """
    swing_toe_x = df[f'{swing_side}TOE_x'].to_numpy()
    stance_toe_x = df[f'{stance_side}TOE_x'].to_numpy()
    diff = swing_toe_x - stance_toe_x
    
    # In normal walking direction (+X), swing leg starts BEHIND stance (diff < 0)
    # and ends in FRONT (diff > 0). Mid-swing = sign change from negative to positive.
    seg = diff[hs_start_frame:hs_end_frame + 1]
    sign_changes = np.where(np.diff(np.sign(seg)))[0]
    if len(sign_changes) == 0:
        return None
    # Pick the first sign change (or middle one if multiple)
    return hs_start_frame + sign_changes[0]


# ============================================================================
# Whole-body crossing speed (COM speed at obstacle crossing)
# ============================================================================

def find_obstacle_x(df: pd.DataFrame) -> float | None:
    """
    Return the obstacle's AP (X) position. Uses mean of OBSTACLE_L and 
    OBSTACLE_R markers across all frames where present. Returns None if 
    obstacle markers absent.
    
    Note: this is called AFTER normalize_walking_direction, so the obstacle
    X is in the normalized (+X forward) coordinate.
    """
    cols_L = [f'OBSTACLE_L_x']
    cols_R = [f'OBSTACLE_R_x']
    if not all(c in df.columns for c in cols_L + cols_R):
        return None
    obs_x = pd.concat([df['OBSTACLE_L_x'], df['OBSTACLE_R_x']]).dropna()
    if len(obs_x) == 0:
        return None
    return float(obs_x.mean())


def find_com_above_obstacle_frame(com: np.ndarray, obstacle_x: float,
                                    hs_start: int, hs_end: int) -> int | None:
    """
    Find the frame within [hs_start, hs_end] where COM_AP (X) crosses 
    obstacle_x. Detects sign change in (com_x - obstacle_x).
    """
    seg = com[hs_start:hs_end + 1, 0] - obstacle_x
    valid = ~np.isnan(seg)
    if valid.sum() < 2: return None
    # Find first sign change from negative to positive (subject approaches and crosses)
    sign_changes = np.where(np.diff(np.sign(seg)))[0]
    if len(sign_changes) == 0:
        return None
    return hs_start + sign_changes[0]


def compute_crossing_speed(com: np.ndarray, v_com: np.ndarray,
                            obstacle_x: float,
                            hs_start: int, hs_end: int) -> dict:
    """
    Find the frame where COM is directly above the obstacle, then return:
      - cross_frame: the frame index
      - cross_speed_3d: 3D speed of COM at that frame (mm/s)
      - cross_vx, cross_vy, cross_vz: component velocities
    Returns dict with NaN values if COM doesn't cross obstacle in this stride.
    """
    if obstacle_x is None:
        return dict(cross_frame=np.nan, cross_speed_3d=np.nan,
                    cross_vx=np.nan, cross_vy=np.nan, cross_vz=np.nan)
    fr = find_com_above_obstacle_frame(com, obstacle_x, hs_start, hs_end)
    if fr is None:
        return dict(cross_frame=np.nan, cross_speed_3d=np.nan,
                    cross_vx=np.nan, cross_vy=np.nan, cross_vz=np.nan)
    vx, vy, vz = v_com[fr]
    speed_3d = float(np.sqrt(vx**2 + vy**2 + vz**2))
    return dict(cross_frame=int(fr),
                cross_speed_3d=speed_3d,
                cross_vx=float(vx), cross_vy=float(vy), cross_vz=float(vz))


# ============================================================================
# Per-stride MoS extraction
# ============================================================================

def process_stride_mos(df: pd.DataFrame, com: np.ndarray, xcom: np.ndarray,
                        stance_side: str, swing_side: str,
                        hs_start: int, foot_off: int, hs_end: int,
                        height_mm: float) -> dict:
    """
    Compute MoS at 3 gait events for one stride.
    
    Stance side: foot on the ground during this stride's stance phase.
    Swing side: opposite foot (swinging).
    
    Events:
      - HS (heel strike): hs_start_frame
      - Mid-swing: when swing toe passes stance toe (AP)
      - Foot-off (toe-off): to_frame (when stance side toe leaves ground)
    
    Returns dict with MoS_AP and MoS_ML at each event, plus clearances.
    """
    # Validate frame indices
    n = len(com)
    if any(f < 0 or f >= n for f in [hs_start, foot_off, hs_end] if not np.isnan(f)):
        return None
    
    # Find mid-swing
    ms = find_mid_swing_frame(df, swing_side, stance_side, hs_start, hs_end)
    if ms is None:
        ms = (hs_start + foot_off) // 2  # fallback
    
    result = {}
    events = {'HS': hs_start, 'midswing': ms, 'footoff': foot_off}
    
    for event_name, frame in events.items():
        if frame is None or np.isnan(frame):
            result[f'mos_ap_{event_name}'] = np.nan
            result[f'mos_ml_{event_name}'] = np.nan
            continue
        frame = int(frame)
        
        # XCOM at this frame
        xcom_ap = xcom[frame, 0]
        xcom_ml = xcom[frame, 1]
        
        # BoS edges from stance leg
        stance_toe_ap   = df[f'{stance_side}TOE_x'].iloc[frame]
        stance_heel_ml  = df[f'{stance_side}HEE_y'].iloc[frame]
        
        mos = compute_mos(xcom_ap, xcom_ml, stance_toe_ap, stance_heel_ml, stance_side)
        result[f'mos_ap_{event_name}']  = mos['mos_ap']
        result[f'mos_ml_{event_name}']  = mos['mos_ml']
        result[f'frame_{event_name}']   = frame
    
    # Height-normalized versions
    for event_name in events:
        for direction in ['ap', 'ml']:
            raw = result.get(f'mos_{direction}_{event_name}', np.nan)
            result[f'mos_{direction}_{event_name}_norm'] = raw / height_mm if not np.isnan(raw) else np.nan
    
    return result


# ============================================================================
# Trial-level processing
# ============================================================================

def process_trial_mos(csv_path: str, stride_records: pd.DataFrame,
                       subject_id: str, trial: int,
                       leg_length_mm: float, height_mm: float,
                       fs: float = 100.0) -> pd.DataFrame:
    """
    Process all strides in one trial for MoS analysis.
    Returns a DataFrame with one row per stride, containing MoS values at
    HS, mid-swing, foot-off, in both AP and ML directions, raw and normalized.
    
    Also computes AP/ML clearance (per Beerse et al., 2024):
      AP clearance = step_length - AP MoS  (positive = foot anterior to MoS)
      ML clearance = step_width - ML MoS   (positive = foot lateral to MoS)
    Note: per_stride_data must contain 'step_length_mm' and 'step_width_mm' columns.
    """
    df = load_marker_csv(csv_path)
    df = normalize_walking_direction(df)
    df = fill_gaps(df, ALL_BODY_MARKERS, max_gap=100)
    df = reconstruct_pelvis_markers(df)
    
    com  = compute_whole_body_com(df, leg_length_mm)
    v    = compute_com_velocity(com, fs)
    xcom = compute_xcom(com, v, leg_length_mm)
    obstacle_x = find_obstacle_x(df)
    
    summary = []
    sub = stride_records[(stride_records['subject_id'] == subject_id) &
                         (stride_records['trial'] == trial)]
    
    for _, r in sub.iterrows():
        if r['phase'] == 'unknown':
            continue
        stance_side = 'L' if r['side'] == 'left' else 'R'
        swing_side  = 'R' if stance_side == 'L' else 'L'
        
        res = process_stride_mos(
            df, com, xcom, stance_side, swing_side,
            int(r['hs_start_frame']), int(r['to_frame']), int(r['hs_end_frame']),
            height_mm
        )
        if res is None:
            continue
        
        # Whole-body crossing speed (only reported for crossing_trail strides)
        # COM typically crosses obstacle during the trail-leg stride.
        if r['phase'] == 'crossing_trail':
            cs = compute_crossing_speed(com, v, obstacle_x,
                                         int(r['hs_start_frame']), int(r['hs_end_frame']))
        else:
            cs = dict(cross_frame=np.nan, cross_speed_3d=np.nan,
                      cross_vx=np.nan, cross_vy=np.nan, cross_vz=np.nan)
        res.update(cs)
        res['cross_speed_3d_norm'] = (cs['cross_speed_3d'] / height_mm 
                                       if not np.isnan(cs['cross_speed_3d']) else np.nan)
        
        # AP/ML clearance — per Beerse et al. (2024):
        #   AP clearance = step_length - AP MoS  (positive = foot anterior to MoS)
        #   ML clearance = step_width - ML MoS   (positive = foot lateral to MoS)
        # Computed in two forms:
        #   *_clearance_raw : raw mm (step_length_mm - mos_ap_HS_mm)
        #   *_clearance     : normalized form using leg-length-normalized step
        #                     parameters from per_stride_data and height-normalized
        #                     MoS. The two normalizations differ but follow the 
        #                     per_stride_data and Beerse 2024 conventions respectively.
        step_len_mm   = r.get('step_length_mm', np.nan)
        step_wid_mm   = r.get('step_width_mm', np.nan)
        step_len_norm = r.get('step_length_norm', np.nan)   # leg-length-normalized
        step_wid_norm = r.get('step_width_norm', np.nan)    # leg-length-normalized
        
        # Raw mm clearance
        ap_clearance_raw = step_len_mm - res.get('mos_ap_HS', np.nan)
        ml_clearance_raw = step_wid_mm - res.get('mos_ml_HS', np.nan)
        res['ap_clearance_raw'] = ap_clearance_raw
        res['ml_clearance_raw'] = ml_clearance_raw
        
        # Normalized clearance: direct subtraction of normalized step parameters
        # (leg-length-normalized) and height-normalized MoS — NOT re-normalized.
        res['ap_clearance'] = step_len_norm - res.get('mos_ap_HS_norm', np.nan)
        res['ml_clearance'] = step_wid_norm - res.get('mos_ml_HS_norm', np.nan)
        
        row = {
            'subject_id': subject_id, 'trial': trial,
            'side': r['side'], 'phase': r['phase'],
            'stride_idx_in_trial': int(r['stride_idx_in_trial']),
            'leg_length_mm': leg_length_mm, 'height_mm': height_mm,
            'stance_side': stance_side,
            'step_length_mm': step_len_mm, 'step_width_mm': step_wid_mm,
            'step_length_norm': step_len_norm, 'step_width_norm': step_wid_norm,
        }
        row.update(res)
        summary.append(row)
    
    return pd.DataFrame(summary)
