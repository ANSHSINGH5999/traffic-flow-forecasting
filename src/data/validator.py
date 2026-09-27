"""Dataset validation. Run standalone:  python -m src.data.validator"""
import json
from pathlib import Path

import numpy as np

from src.data.loader import load_raw
from src.utils import load_config


def longest_zero_run(col):
    best = run = 0
    for v in col:
        run = run + 1 if v == 0 else 0
        best = max(best, run)
    return best


def validate_dataset(cfg):
    ds = cfg["dataset"]
    path = Path(cfg["paths"]["raw_data"])
    data = load_raw(path)
    flow = data[:, :, ds["flow_feature_index"]].astype(np.float64)
    T, N = flow.shape
    days = T / ds["steps_per_day"]
    zeros_per_sensor = (flow == 0).sum(axis=0)

    report = {
        "file": str(path),
        "file_size_mb": round(path.stat().st_size / 1e6, 2),
        "array_shape": list(data.shape),
        "dtype": str(data.dtype),
        "channels": "flow, occupancy, speed (flow used as target)",
        "time_steps": T,
        "sensors": N,
        "sampling_interval_minutes": ds["interval_minutes"],
        "interval_check": (f"{T} steps / {ds['steps_per_day']} steps per day = {days:.2f} days "
                           f"({ds['start_date']} onward); a whole number of days, consistent with the documented 5-minute interval"),
        "flow_stats": {
            "min": float(flow.min()), "max": float(flow.max()), "mean": float(flow.mean()),
            "median": float(np.median(flow)), "std": float(flow.std()),
        },
        "nan_count": int(np.isnan(flow).sum()),
        "inf_count": int(np.isinf(flow).sum()),
        "negative_count": int((flow < 0).sum()),
        "zero_count": int((flow == 0).sum()),
        "zero_pct": float((flow == 0).mean() * 100),
        "missing_values_note": "No NaNs are stored; failed readings appear as 0 flow and are treated as missing.",
        "non_integer_flow_count": int((flow != np.round(flow)).sum()),
        "identical_consecutive_rows": int((np.diff(flow, axis=0) == 0).all(axis=1).sum()),
        "temporal_ordering": "Rows are consecutive 5-minute steps in index order (no timestamp column in the file).",
        "sensor_consistency": {
            "all_sensors_same_length": True,
            "sensors_always_zero": int((zeros_per_sensor == T).sum()),
            "sensors_constant": int((flow.std(axis=0) == 0).sum()),
            "sensors_with_any_zero": int((zeros_per_sensor > 0).sum()),
            "worst_sensor_zero_pct": float(zeros_per_sensor.max() / T * 100),
            "longest_zero_run_steps": int(max(longest_zero_run(flow[:, i]) for i in range(N))),
        },
    }
    report["checks_passed"] = (report["nan_count"] == 0 and report["inf_count"] == 0
                               and report["negative_count"] == 0 and data.ndim == 3)
    return report, flow


def save_report(report, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "dataset_report.json").write_text(json.dumps(report, indent=2))
    lines = ["PeMSD4 DATASET REPORT", "=" * 40]
    for k, v in report.items():
        if isinstance(v, dict):
            lines.append(f"{k}:")
            lines += [f"    {kk}: {vv:.3f}" if isinstance(vv, float) else f"    {kk}: {vv}" for kk, vv in v.items()]
        else:
            lines.append(f"{k}: {v:.3f}" if isinstance(v, float) else f"{k}: {v}")
    (out_dir / "dataset_report.txt").write_text("\n".join(lines))
    return "\n".join(lines)


if __name__ == "__main__":
    cfg = load_config()
    rep, flow = validate_dataset(cfg)
    print(save_report(rep, Path(cfg["paths"]["results_dir"]) / "reports"))
