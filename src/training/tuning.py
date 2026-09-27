"""Small grid search helpers. Candidates are scored on the VALIDATION set only - never on test."""
import itertools

import numpy as np


def grid(search):
    keys = list(search)
    return [dict(zip(keys, values)) for values in itertools.product(*search.values())]


def sensor_subset(split, max_sensors):
    """Samples from the first `max_sensors` sensors (keeps the search affordable on 3.6M training rows)."""
    keep = split["sensor"] < max_sensors
    return {k: v[keep] for k, v in split.items()}


def val_mae(pred, y):
    return float(np.abs(pred - y).mean())
