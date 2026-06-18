"""
gait_kinematics.py — Sagittal-plane joint kinematics for obstacle-walking gait.

Uses Newington-Gage Hip Joint Center (HJC) computed from pelvis markers and
subject leg length (PiG-style anatomical landmark estimation), then sagittal
projection (X-Z, where X=AP forward and Z=vertical up) for joint angles.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt

# ============================================================================
# Loading and gap-filling
# ============================================================================

def load_marker_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip().replace(' ', '') for c in df.columns]
    return df

def fill_gaps(df, marker_names, max_gap=20):
    out = df.copy()
    for m in marker_names:
        for axis in ['x','y','z']:
            col = f'{m}_{axis}'
            if col not in out.columns: continue
            out[col] = out[col].interpolate(method='linear', limit=max_gap,
                                             limit_direction='both')
    return out


def _continuous_segments(valid: np.ndarray) -> list[tuple[int, int]]:
    """Return (start, end_exclusive) index pairs for contiguous True runs."""
    segments: list[tuple[int, int]] = []
    start: int | None = None
    for i, ok in enumerate(valid):
        if ok and start is None:
            start = i
        elif not ok and start is not None:
            segments.append((start, i))
            start = None
    if start is not None:
        segments.append((start, len(valid)))
    return segments


def reject_marker_spikes(df: pd.DataFrame, markers: list[str],
                         threshold_mm_per_frame: float = 100.0
                         ) -> tuple[pd.DataFrame, dict[str, int]]:
    """
    Detect frame-to-frame 3D jumps exceeding threshold_mm_per_frame.
    NaN both frames of each spike pair so fill_gaps can re-interpolate.
    """
    out = df.copy()
    report = {m: 0 for m in markers}
    any_removed = False

    for m in markers:
        cols = [f'{m}_x', f'{m}_y', f'{m}_z']
        if not all(c in out.columns for c in cols):
            continue
        pos = out[cols].to_numpy(dtype=float)
        n = len(pos)
        for i in range(n - 1):
            if np.any(np.isnan(pos[i])) or np.any(np.isnan(pos[i + 1])):
                continue
            d = float(np.linalg.norm(pos[i + 1] - pos[i]))
            if d > threshold_mm_per_frame:
                out.loc[out.index[i], cols] = np.nan
                out.loc[out.index[i + 1], cols] = np.nan
                pos[i] = np.nan
                pos[i + 1] = np.nan
                report[m] += 1
                any_removed = True

    if any_removed:
        total = sum(report.values())
        parts = ', '.join(f'{k}:{v}' for k, v in report.items() if v > 0)
        print(f"reject_marker_spikes: removed {total} spike(s) ({parts})")

    return out, report


def butterworth_filter(df: pd.DataFrame, markers: list[str],
                       cutoff_hz: float = 6.0, order: int = 4,
                       fs: float = 100.0) -> pd.DataFrame:
    """
    Zero-lag 4th-order Butterworth low-pass (filtfilt) per marker axis.
    Filters each continuous non-NaN segment separately.
    """
    out = df.copy()
    nyq = fs / 2.0
    wn = cutoff_hz / nyq
    if wn <= 0 or wn >= 1.0:
        return out

    b, a = butter(order, wn, btype='low')
    padlen = 3 * max(len(a), len(b))

    for m in markers:
        for axis in ('x', 'y', 'z'):
            col = f'{m}_{axis}'
            if col not in out.columns:
                continue
            arr = out[col].to_numpy(dtype=float)
            valid = ~np.isnan(arr)
            for start, end in _continuous_segments(valid):
                seg = arr[start:end]
                if len(seg) <= padlen:
                    continue
                try:
                    arr[start:end] = filtfilt(b, a, seg)
                except ValueError:
                    pass
            out[col] = arr
    return out


def preprocess_markers(df: pd.DataFrame, markers: list[str], *,
                       spike_threshold_mm_per_frame: float = 100.0,
                       filter_cutoff_hz: float | None = 6.0,
                       fs: float = 100.0, max_gap: int = 100,
                       reconstruct_pelvis: bool = True
                       ) -> tuple[pd.DataFrame, dict[str, int]]:
    """
    Tier-1 preprocessing: spike rejection → gap fill → pelvis recon → Butterworth.
    """
    out, spike_report = reject_marker_spikes(
        df, markers, threshold_mm_per_frame=spike_threshold_mm_per_frame)
    out = fill_gaps(out, markers, max_gap=max_gap)
    if reconstruct_pelvis:
        out = reconstruct_pelvis_markers(out)
    if filter_cutoff_hz is not None and filter_cutoff_hz > 0:
        out = butterworth_filter(out, markers, cutoff_hz=filter_cutoff_hz,
                                 order=4, fs=fs)
    return out, spike_report


def normalize_walking_direction(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize trial so subject walks in +X direction.
    Detects walking direction from heel marker progression. If walking is in
    -X, flips both X and Y axes (Y flip preserves left/right anatomy when X 
    is reversed — both left and right sides reflect symmetrically).
    
    All marker columns *_x and *_y are negated; *_z unchanged.
    """
    out = df.copy()
    # Detect direction: mean of first 10% vs last 10% of frames using LHEE_x
    n = len(out)
    n_seg = max(10, n // 10)
    start_x = out['LHEE_x'].iloc[:n_seg].mean()
    end_x = out['LHEE_x'].iloc[-n_seg:].mean()
    
    if end_x < start_x:
        # Walking in -X direction → flip X and Y
        x_cols = [c for c in out.columns if c.endswith('_x')]
        y_cols = [c for c in out.columns if c.endswith('_y')]
        for c in x_cols:
            out[c] = -out[c]
        for c in y_cols:
            out[c] = -out[c]
    return out


def reconstruct_pelvis_markers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reconstruct missing pelvis markers (LASI, RASI, LPSI, RPSI) using rigid
    body assumption: when 3 of the 4 markers are available, the 4th can be
    reconstructed from the rigid pelvis frame.
    
    Strategy: for frames where one marker is missing but the other three are
    present, compute its position in the pelvis local frame (using a reference
    frame where all 4 are available), then transform back to lab coordinates.
    
    Returns a copy of df with reconstructed pelvis marker columns.
    """
    out = df.copy()
    markers = ['LASI', 'RASI', 'LPSI', 'RPSI']
    
    # Build availability mask: True if all 3 axes present for that marker
    avail = {m: ~out[[f'{m}_x', f'{m}_y', f'{m}_z']].isna().any(axis=1)
             for m in markers}
    all_4 = avail['LASI'] & avail['RASI'] & avail['LPSI'] & avail['RPSI']
    
    if not all_4.any():
        return out  # Can't reconstruct without any complete reference frame
    
    # Pick reference frames (where all 4 are present) — use first ~30 such frames
    ref_idx = np.where(all_4)[0][:30]
    
    # For each marker, compute its mean position in pelvis local frame
    # Local frame: origin = mean(ASIS midpoint, PSI midpoint)
    #              Y axis = RASI → LASI (normalized)
    #              X axis = perpendicular to Y, in horizontal-ish direction
    #              Z axis = X × Y
    
    def build_pelvis_frame(frame_idx):
        """Build pelvis local frame at one frame. Returns (origin, R) where
        R is 3x3 rotation matrix (columns = X, Y, Z axes)."""
        lasi = out[['LASI_x','LASI_y','LASI_z']].iloc[frame_idx].to_numpy()
        rasi = out[['RASI_x','RASI_y','RASI_z']].iloc[frame_idx].to_numpy()
        lpsi = out[['LPSI_x','LPSI_y','LPSI_z']].iloc[frame_idx].to_numpy()
        rpsi = out[['RPSI_x','RPSI_y','RPSI_z']].iloc[frame_idx].to_numpy()
        
        asis_mid = (lasi + rasi) / 2
        psi_mid  = (lpsi + rpsi) / 2
        origin = (asis_mid + psi_mid) / 2
        
        y_axis = lasi - rasi
        y_axis = y_axis / np.linalg.norm(y_axis)
        x_prov = asis_mid - psi_mid
        z_axis = np.cross(x_prov, y_axis)
        z_axis = z_axis / np.linalg.norm(z_axis)
        x_axis = np.cross(y_axis, z_axis)
        
        R = np.column_stack([x_axis, y_axis, z_axis])
        return origin, R
    
    # Compute local-frame positions of each marker, averaged over reference frames
    local_positions = {m: [] for m in markers}
    for fi in ref_idx:
        origin, R = build_pelvis_frame(fi)
        for m in markers:
            lab_pos = out[[f'{m}_x', f'{m}_y', f'{m}_z']].iloc[fi].to_numpy()
            local_pos = R.T @ (lab_pos - origin)
            local_positions[m].append(local_pos)
    local_mean = {m: np.mean(local_positions[m], axis=0) for m in markers}
    
    # Reconstruct missing markers
    for target in markers:
        miss_mask = ~avail[target]
        if not miss_mask.any():
            continue
        # Determine which 3 are present in each missing frame
        for idx in np.where(miss_mask)[0]:
            present = [m for m in markers if m != target and avail[m].iloc[idx]]
            if len(present) < 3:
                continue  # Need all 3 others to build frame
            
            # Build pelvis frame using the 3 present + assume target at local_mean[target]
            # Trick: we can build frame from any 3 markers, then place target via local mean.
            # Simpler approach: use the same frame-building logic, but substitute target
            # marker's lab position with a guess (e.g., from previous frame), iterate.
            # 
            # Best: use the 3 present markers to construct a pelvis frame directly.
            # For simplicity here: average pairwise vectors among the 3 present, then
            # solve for target = origin + R @ local_mean[target].
            
            # We use a known geometry: the 3 present markers form a triangle in pelvis frame.
            # Compute the rigid transform from pelvis local frame (reference geometry) to
            # current frame using these 3 markers.
            
            # Reference local positions for present markers
            local_ref = np.array([local_mean[m] for m in present])
            # Current lab positions for present markers
            lab_curr = np.array([
                out[[f'{m}_x', f'{m}_y', f'{m}_z']].iloc[idx].to_numpy()
                for m in present
            ])
            
            # Find rigid transform (R, t) such that R @ local_ref.T + t = lab_curr.T
            # Use Kabsch algorithm
            centroid_ref = local_ref.mean(axis=0)
            centroid_curr = lab_curr.mean(axis=0)
            centered_ref = local_ref - centroid_ref
            centered_curr = lab_curr - centroid_curr
            H = centered_ref.T @ centered_curr
            U, _, Vt = np.linalg.svd(H)
            d = np.sign(np.linalg.det(Vt.T @ U.T))
            D = np.diag([1, 1, d])
            R_fit = Vt.T @ D @ U.T
            t_fit = centroid_curr - R_fit @ centroid_ref
            
            # Reconstruct target
            target_recon = R_fit @ local_mean[target] + t_fit
            
            out.at[idx, f'{target}_x'] = target_recon[0]
            out.at[idx, f'{target}_y'] = target_recon[1]
            out.at[idx, f'{target}_z'] = target_recon[2]
    
    return out

def get_marker_3d(df, name):
    return df[[f'{name}_x', f'{name}_y', f'{name}_z']].to_numpy()

def get_sagittal(df, name):
    """Return Nx2 (X=AP, Z=vertical)."""
    return df[[f'{name}_x', f'{name}_z']].to_numpy()

# ============================================================================
# Vector helpers
# ============================================================================

def _unit(v):
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    n = np.where(n < 1e-9, np.nan, n)
    return v / n

def _signed_angle_2d(v_from, v_to):
    cross = v_from[:,0]*v_to[:,1] - v_from[:,1]*v_to[:,0]
    dot   = v_from[:,0]*v_to[:,0] + v_from[:,1]*v_to[:,1]
    return np.degrees(np.arctan2(cross, dot))

# ============================================================================
# Newington-Gage Hip Joint Center (3D)
# ============================================================================
# Reference: Davis et al. (1991), PiG manual.
# Computes HJC location in lab coordinates using pelvis markers and leg length.

def compute_hjc(df, side: str, leg_length_mm: float) -> np.ndarray:
    """
    Compute Hip Joint Center for the given side.
    
    Pelvis local coordinate system:
      - Origin: SACR (midpoint of LPSI, RPSI)
      - Y axis: RASI → LASI (mediolateral, +Y to subject's left)
      - Provisional X: SACR → ASIS_midpoint
      - Z axis: X × Y (vertical-ish, perpendicular to pelvis plane)
      - Final X: Y × Z (forces orthogonality)
    
    Newington-Gage HJC offset in pelvis local frame:
      C = LegLength × 0.115 - 15.3 mm
      asis_troch_dist = 0.1288 * LegLength - 48.56
      θ = 28.4°, β = 18° (defaults)
      
      X_HJC = C × cos(θ) × sin(β) - asis_troch × cos(β)
      Y_HJC = ±(C × sin(θ) - 0.5 × inter_asis_dist)  [+ for left, - for right]
      Z_HJC = -C × cos(θ) × cos(β) - asis_troch × sin(β)
    
    HJC is then expressed relative to ASIS_midpoint (origin shift), 
    then transformed to lab coordinates using pelvis basis vectors.
    
    Returns:
      hjc: Nx3 array in lab coordinates (mm)
    """
    assert side in ('L', 'R')
    LASI = get_marker_3d(df, 'LASI')
    RASI = get_marker_3d(df, 'RASI')
    LPSI = get_marker_3d(df, 'LPSI')
    RPSI = get_marker_3d(df, 'RPSI')
    
    asis_mid = (LASI + RASI) / 2.0
    psi_mid  = (LPSI + RPSI) / 2.0
    
    # Pelvis basis vectors (each Nx3, per-frame)
    # Y axis: RASI → LASI
    y_axis = LASI - RASI
    y_axis_u = _unit(y_axis)
    # Provisional X: ASIS_mid - SACR (i.e., posterior → anterior)
    x_prov = asis_mid - psi_mid
    # Z axis: x_prov × y_axis (right-hand rule)
    z_axis = np.cross(x_prov, y_axis_u)
    z_axis_u = _unit(z_axis)
    # Final X: y × z (force orthogonality)
    x_axis_u = np.cross(y_axis_u, z_axis_u)
    
    # Leg length and derived constants
    L = float(leg_length_mm)
    C = L * 0.115 - 15.3
    asis_troch = 0.1288 * L - 48.56
    inter_asis = np.linalg.norm(LASI - RASI, axis=1).mean()  # average across frames
    theta = np.deg2rad(28.4)
    beta  = np.deg2rad(18.0)
    
    # HJC in pelvis local frame, expressed relative to ASIS_midpoint
    # (Davis et al. 1991 / PiG convention)
    x_local = C * np.cos(theta) * np.sin(beta) - asis_troch * np.cos(beta)
    z_local = -C * np.cos(theta) * np.cos(beta) - asis_troch * np.sin(beta)
    if side == 'L':
        y_local =  (C * np.sin(theta) - 0.5 * inter_asis)
    else:
        y_local = -(C * np.sin(theta) - 0.5 * inter_asis)
    
    # Transform to lab coordinates: 
    # hjc_lab = asis_mid + x_local * x_axis + y_local * y_axis + z_local * z_axis
    hjc = (asis_mid
           + x_local * x_axis_u
           + y_local * y_axis_u
           + z_local * z_axis_u)
    return hjc

def compute_hjc_sagittal(df, side, leg_length_mm):
    """Return HJC in (X, Z) sagittal projection."""
    hjc_3d = compute_hjc(df, side, leg_length_mm)
    return hjc_3d[:, [0, 2]]  # X, Z

# ============================================================================
# Pelvis reference vertical (for hip angle reference)
# ============================================================================

def pelvis_vertical_sagittal(df):
    """
    Return unit vector perpendicular to ASIS-PSIS line in sagittal plane,
    pointing upward. This is the 'pelvis-up' reference for hip angle.
    """
    LASI = get_sagittal(df, 'LASI'); RASI = get_sagittal(df, 'RASI')
    LPSI = get_sagittal(df, 'LPSI'); RPSI = get_sagittal(df, 'RPSI')
    asis_mid = (LASI + RASI) / 2.0
    psi_mid  = (LPSI + RPSI) / 2.0
    pelvis_vec = psi_mid - asis_mid
    pelv_perp = np.stack([-pelvis_vec[:,1], pelvis_vec[:,0]], axis=1)
    flip = pelv_perp[:,1] < 0
    pelv_perp[flip] = -pelv_perp[flip]
    return _unit(pelv_perp)

# ============================================================================
# Joint angles (sagittal plane)
# ============================================================================

def hip_angle_sagittal(df, side, leg_length_mm):
    """
    Hip flexion+/extension-, in degrees.
    Reference: pelvis perpendicular vertical (from ASIS-PSIS).
    Thigh long axis: HJC (Newington-Gage) → KNE.
    """
    pelv_vert = pelvis_vertical_sagittal(df)
    HJC = compute_hjc_sagittal(df, side, leg_length_mm)
    KNE = get_sagittal(df, f'{side}KNE')
    # Thigh-up vector: from knee toward HJC (upward along thigh)
    thigh_up = HJC - KNE
    thigh_up_u = _unit(thigh_up)
    # Signed angle from pelvis-vertical to thigh-up
    # Hip flexion: thigh swings forward → KNE moves forward of HJC → thigh-up
    # vector tilts backward (−X component). Pelvis-vertical points slightly
    # forward (+X). The signed angle (CCW positive in X-Z view from +Y)
    # is positive when thigh-up rotates CCW from pelv_vert. CCW rotation in 
    # this view corresponds to thigh tilting backward (extension would seem positive)
    # — but empirical verification at frame 85 (known HS with flexed hip) showed
    # the raw signed angle is +26.6° while subject is clearly flexed at HS.
    # So raw signed angle IS flexion-positive. No negation needed.
    return _signed_angle_2d(pelv_vert, thigh_up_u)

def knee_angle_sagittal(df, side, leg_length_mm):
    """Knee flexion+/hyperextension-, in degrees."""
    HJC = compute_hjc_sagittal(df, side, leg_length_mm)
    KNE = get_sagittal(df, f'{side}KNE')
    ANK = get_sagittal(df, f'{side}ANK')
    # Thigh extension direction: HJC → KNE, continuing downward
    thigh_ext = KNE - HJC
    thigh_ext_u = _unit(thigh_ext)
    shank = ANK - KNE
    shank_u = _unit(shank)
    return -_signed_angle_2d(thigh_ext_u, shank_u)

def ankle_angle_sagittal(df, side):
    """Ankle dorsiflexion+/plantarflexion-, in degrees. Does not need leg length.
    
    Reference: when foot is flat on ground and shank is vertical (neutral standing),
    raw angle between shank (KNE→ANK direction, points down) and foot (HEE→TOE,
    points forward) is 90°.
    
    Dorsiflexion: shank rotates forward over fixed foot → raw angle INCREASES > 90°
    Plantarflexion: foot rotates down while shank vertical → raw angle DECREASES < 90°
    
    So dorsiflexion = raw - 90 (positive), plantarflexion = raw - 90 (negative).
    """
    KNE = get_sagittal(df, f'{side}KNE')
    ANK = get_sagittal(df, f'{side}ANK')
    HEE = get_sagittal(df, f'{side}HEE')
    TOE = get_sagittal(df, f'{side}TOE')
    shank = ANK - KNE
    foot  = TOE - HEE
    shank_u = _unit(shank); foot_u = _unit(foot)
    cos_a = np.clip(np.sum(shank_u * foot_u, axis=1), -1.0, 1.0)
    raw = np.degrees(np.arccos(cos_a))
    return raw - 90.0

# ============================================================================
# Time normalization, derivatives, peaks
# ============================================================================

def time_normalize(signal, start_idx, end_idx, n_points=101):
    seg = signal[start_idx:end_idx+1]
    if len(seg) < 2: return np.full(n_points, np.nan)
    valid = ~np.isnan(seg)
    if valid.sum() < 2: return np.full(n_points, np.nan)
    x_old = np.linspace(0, 100, len(seg))
    x_new = np.linspace(0, 100, n_points)
    seg_clean = np.interp(x_old, x_old[valid], seg[valid])
    return np.interp(x_new, x_old, seg_clean)

def angular_velocity(angle_deg, fs=100.0):
    n = len(angle_deg)
    v = np.full(n, np.nan)
    dt = 1.0/fs
    v[1:-1] = (angle_deg[2:] - angle_deg[:-2]) / (2*dt)
    v[0]  = (angle_deg[1] - angle_deg[0]) / dt
    v[-1] = (angle_deg[-1] - angle_deg[-2]) / dt
    return v

def angular_acceleration(angle_deg, fs=100.0):
    return angular_velocity(angular_velocity(angle_deg, fs), fs)

def peak_stats(signal_norm):
    if np.all(np.isnan(signal_norm)):
        return dict(max_value=np.nan, max_pct=np.nan,
                    min_value=np.nan, min_pct=np.nan, rom=np.nan)
    n = len(signal_norm)
    pct_axis = np.linspace(0, 100, n)
    imax = int(np.nanargmax(signal_norm))
    imin = int(np.nanargmin(signal_norm))
    return dict(max_value=float(signal_norm[imax]),
                max_pct=float(pct_axis[imax]),
                min_value=float(signal_norm[imin]),
                min_pct=float(pct_axis[imin]),
                rom=float(signal_norm[imax] - signal_norm[imin]))

# ============================================================================
# Per-stride and per-trial processing
# ============================================================================

PIG_LOWER_MARKERS = ['LASI','RASI','LPSI','RPSI',
                     'LTHI','LKNE','LTIB','LANK','LHEE','LTOE',
                     'RTHI','RKNE','RTIB','RANK','RHEE','RTOE']

def process_stride(df, side, leg_length_mm, hs_start, hs_end,
                   fs=100.0, n_norm_points=101):
    hip   = hip_angle_sagittal(df, side, leg_length_mm)
    knee  = knee_angle_sagittal(df, side, leg_length_mm)
    ankle = ankle_angle_sagittal(df, side)
    hip_v, knee_v, ank_v = (angular_velocity(s, fs) for s in (hip, knee, ankle))
    hip_a, knee_a, ank_a = (angular_acceleration(s, fs) for s in (hip, knee, ankle))
    def tn(s): return time_normalize(s, hs_start, hs_end, n_norm_points)
    hip_n, knee_n, ank_n = tn(hip), tn(knee), tn(ankle)
    hipv_n, knev_n, ankv_n = tn(hip_v), tn(knee_v), tn(ank_v)
    hipa_n, knea_n, anka_n = tn(hip_a), tn(knee_a), tn(ank_a)
    result = {'hip_angle_norm': hip_n, 'knee_angle_norm': knee_n, 'ankle_angle_norm': ank_n,
              'hip_vel_norm': hipv_n,  'knee_vel_norm': knev_n,  'ankle_vel_norm': ankv_n,
              'hip_acc_norm': hipa_n,  'knee_acc_norm': knea_n,  'ankle_acc_norm': anka_n}
    for j, sig in [('hip', hip_n), ('knee', knee_n), ('ankle', ank_n)]:
        ps = peak_stats(sig)
        result[f'{j}_peak_flexion']       = ps['max_value']
        result[f'{j}_peak_flexion_pct']   = ps['max_pct']
        result[f'{j}_peak_extension']     = ps['min_value']
        result[f'{j}_peak_extension_pct'] = ps['min_pct']
        result[f'{j}_rom']                = ps['rom']
    pct_axis = np.linspace(0, 100, n_norm_points)
    for j, vsig, asig in [('hip', hipv_n, hipa_n),
                          ('knee', knev_n, knea_n),
                          ('ankle', ankv_n, anka_n)]:
        if not np.all(np.isnan(vsig)):
            i = int(np.nanargmax(np.abs(vsig)))
            result[f'{j}_peak_velocity']     = float(vsig[i])
            result[f'{j}_peak_velocity_pct'] = float(pct_axis[i])
        else:
            result[f'{j}_peak_velocity'] = np.nan
            result[f'{j}_peak_velocity_pct'] = np.nan
        if not np.all(np.isnan(asig)):
            i = int(np.nanargmax(np.abs(asig)))
            result[f'{j}_peak_acceleration']     = float(asig[i])
            result[f'{j}_peak_acceleration_pct'] = float(pct_axis[i])
        else:
            result[f'{j}_peak_acceleration'] = np.nan
            result[f'{j}_peak_acceleration_pct'] = np.nan
    return result

def process_trial(csv_path, stride_records, subject_id, trial,
                  leg_length_mm, fs=100.0,
                  spike_threshold_mm_per_frame: float = 100.0,
                  filter_cutoff_hz: float | None = 6.0):
    df = load_marker_csv(csv_path)
    df = normalize_walking_direction(df)
    df, spike_report = preprocess_markers(
        df, PIG_LOWER_MARKERS,
        spike_threshold_mm_per_frame=spike_threshold_mm_per_frame,
        filter_cutoff_hz=filter_cutoff_hz, fs=fs, max_gap=100)
    summary_rows = []
    curves = {}
    sub = stride_records[(stride_records['subject_id']==subject_id) &
                         (stride_records['trial']==trial)]
    for _, r in sub.iterrows():
        side = 'L' if r['side']=='left' else 'R'
        hs_start = int(r['hs_start_frame'])
        hs_end   = int(r['hs_end_frame'])
        res = process_stride(df, side, leg_length_mm, hs_start, hs_end, fs)
        stride_id = (subject_id, trial, r['side'], int(r['stride_idx_in_trial']))
        curves[stride_id] = {k: res[k] for k in res if k.endswith('_norm')}
        row = {'subject_id': subject_id, 'trial': trial, 'side': r['side'],
               'phase': r['phase'], 'stride_idx_in_trial': int(r['stride_idx_in_trial']),
               'hs_start_frame': hs_start, 'hs_end_frame': hs_end,
               'leg_length_mm': leg_length_mm}
        row.update({k: v for k, v in res.items() if not k.endswith('_norm')})
        summary_rows.append(row)
    return pd.DataFrame(summary_rows), curves, spike_report
