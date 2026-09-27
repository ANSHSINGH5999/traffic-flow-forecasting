"""Model 2 - Random Forest on engineered features."""
import joblib
from sklearn.ensemble import RandomForestRegressor


def build(params, seed):
    return RandomForestRegressor(random_state=seed, **params)


def save(model, path):
    joblib.dump(model, path, compress=3)


def load(path):
    return joblib.load(path)
