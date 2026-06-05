# Gait MoS & Kinematics

Python tools for **sagittal-plane joint kinematics** and **margin of stability (MoS)** analysis during obstacle-crossing gait. Designed for full-body Plug-in Gait marker data and downstream SPM ensemble averaging.

## Installation

```bash
git clone git@github.com:yesiam0225/gait-mos-kinematics.git
cd gait-mos-kinematics
pip install -e .
```

This installs the `gait_mos_kinematics` package and three batch CLI commands.

## Requirements

- Python 3.10+
- NumPy, pandas

## Input data

Batch scripts expect:

| File | Description |
|------|-------------|
| `obs_trials.csv` | Trial manifest: `subject_id`, `trial`, `group`, `board`, `time`, `csv_path`, `leg_length_mm`, `height_mm`, … |
| `per_stride_data.csv` | Per-stride events from spatiotemporal pipeline: `hs_start_frame`, `hs_end_frame`, `to_frame`, `phase`, `side`, `step_length_mm`, … |
| Corrected marker CSVs | Full-body PiG marker trials (mm, 100 Hz) referenced by `csv_path` |

Trial CSV paths in the manifest may use spaces (`BBA01 Trial 05_corrected.csv`) or underscores; the batch scripts resolve multiple naming conventions automatically.

## Command-line tools

### Joint kinematics ensemble (SPM)

Time-normalizes hip/knee/ankle angle, velocity, and acceleration per stride, then averages within each subject × condition × phase × side cell. Writes 36 wide-format CSVs (`ensemble_<phase>_<joint>_<signal>.csv`).

```bash
batch-kinematics-ensemble \
  --obs-csv path/to/obs_trials.csv \
  --ps-csv path/to/per_stride_data.csv \
  --trial-dir path/to/corrected/ \
  --output-dir path/to/output/ensemble_curves/
```

Optional: `--filter-trials BBA01:5,BBA01:23` to process a subset.

### MoS at gait events

Computes whole-body COM, XCOM, and MoS at heel strike, mid-swing, and foot-off. Outputs:

- `mos_all_strides.csv` — one row per stride
- `mos_subject_condition.csv` — subject × condition × phase means

```bash
batch-mos \
  --obs-csv path/to/obs_trials.csv \
  --ps-csv path/to/per_stride_data.csv \
  --trial-dir path/to/corrected/ \
  --output-dir path/to/output/mos/
```

### MoS time-series ensemble (SPM)

Frame-wise MoS_AP and MoS_ML curves time-normalized to 0–100%, ensemble-averaged per cell. Writes `ensemble_mos_<phase>_<direction>[_normheight].csv`.

```bash
batch-mos-timeseries \
  --obs-csv path/to/obs_trials.csv \
  --ps-csv path/to/per_stride_data.csv \
  --trial-dir path/to/corrected/ \
  --output-dir path/to/output/mos_timeseries/
```

## Python API

```python
from gait_mos_kinematics import process_trial, process_trial_mos
import pandas as pd

strides = pd.read_csv("per_stride_data.csv")

summary, curves = process_trial(
    "BBA01_Trial_05_corrected.csv",
    strides,
    subject_id="BBA01",
    trial=5,
    leg_length_mm=850.0,
)

mos_df = process_trial_mos(
    "BBA01_Trial_05_corrected.csv",
    strides,
    subject_id="BBA01",
    trial=5,
    leg_length_mm=850.0,
    height_mm=1700.0,
)
```

## Modules

| Module | Role |
|--------|------|
| `gait_kinematics.py` | Newington-Gage HJC, sagittal joint angles/derivatives, time normalization |
| `gait_mos.py` | Dempster whole-body COM, XCOM, MoS, clearance, crossing speed |
| `batch_kinematics_ensemble.py` | Batch kinematics → SPM ensemble CSVs |
| `batch_mos.py` | Batch discrete-event MoS |
| `batch_mos_timeseries.py` | Batch MoS time-series → SPM ensemble CSVs |

## Related projects

- [gait-spatiotemporal](https://github.com/yesiam0225/gait-spatiotemporal) — gait event detection and spatiotemporal parameters (produces `per_stride_data.csv`)
- [marker-label](https://github.com/yesiam0225/marker-label) — marker labeling and corrected trial CSVs

## License

MIT — see [LICENSE](LICENSE).
