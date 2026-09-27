"""Load the PeMSD4 traffic-flow matrix from data/raw/pems04.npz."""
from pathlib import Path

import numpy as np

DATASET_HELP = (
    "PeMSD4 not found. Expected file: data/raw/pems04.npz (NumPy archive with key 'data', "
    "shape [time_steps, 307 sensors, 3 channels: flow, occupancy, speed]). "
    "It is distributed with the ASTGCN paper's code, e.g.\n"
    "  curl -L -o data/raw/pems04.npz https://github.com/Davidham3/ASTGCN/raw/master/data/PEMS04/pems04.npz"
)


def load_raw(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(DATASET_HELP)
    with np.load(path) as archive:
        if "data" not in archive.files:
            raise ValueError(f"{path} has keys {archive.files}; expected 'data'")
        return archive["data"]


def load_flow(path, feature_index=0, max_sensors=None):
    """Return flow as float32 array [time_steps, sensors]."""
    data = load_raw(path)
    if data.ndim != 3:
        raise ValueError(f"Expected a 3-D array [time, sensors, channels], got shape {data.shape}")
    flow = data[:, :, feature_index].astype(np.float32)
    return flow[:, :max_sensors] if max_sensors else flow
