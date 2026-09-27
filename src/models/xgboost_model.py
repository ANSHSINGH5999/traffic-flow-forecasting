"""Model 3 - XGBoost on engineered features (also the final stage of the hybrid)."""
from xgboost import XGBRegressor


def build(params, seed):
    return XGBRegressor(random_state=seed, n_jobs=-1, **params)


def fit(params, seed, F_train, y_train, F_val, y_val):
    """Early stopping is monitored on the validation set."""
    model = build(params, seed)
    model.fit(F_train, y_train, eval_set=[(F_val, y_val)], verbose=False)
    return model


def save(model, path):
    model.save_model(path)


def load(path):
    model = XGBRegressor()
    model.load_model(path)
    return model
