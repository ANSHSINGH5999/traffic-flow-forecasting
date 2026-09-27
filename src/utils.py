"""Shared helpers: config, seeds, logging, device, environment info."""
import datetime as dt
import logging
import platform
import random
import sys
from importlib.metadata import version
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_config(path=ROOT / "config.yaml"):
    with open(path) as f:
        cfg = yaml.safe_load(f)
    ds = cfg["dataset"]
    ds["steps_per_day"] = 24 * 60 // ds["interval_minutes"]
    ds["start_weekday"] = dt.date.fromisoformat(ds["start_date"]).weekday()   # 0 = Monday
    return cfg


def set_seed(seed):
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device():
    import torch
    if torch.backends.mps.is_available():
        return "mps"
    return "cuda" if torch.cuda.is_available() else "cpu"


def get_logger(log_file=None):
    log = logging.getLogger("traffic")
    log.setLevel(logging.INFO)
    log.handlers.clear()
    fmt = logging.Formatter("%(asctime)s  %(message)s", "%H:%M:%S")
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, mode="w"))
    for h in handlers:
        h.setFormatter(fmt)
        log.addHandler(h)
    return log


def environment_info():
    libs = ["numpy", "pandas", "scikit-learn", "xgboost", "torch", "matplotlib", "streamlit", "pyyaml"]
    return {
        "python": sys.version.split()[0],
        "os": f"{platform.system()} {platform.release()} ({platform.machine()})",
        "libraries": {lib: _version(lib) for lib in libs},
    }


def _version(lib):
    try:
        return version(lib)
    except Exception:
        return None
