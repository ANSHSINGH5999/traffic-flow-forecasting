"""Models 4 & 5 - LSTM and GRU. Same code, only the recurrent cell differs (keeps the comparison fair).

    12-step sequence -> LSTM/GRU -> last hidden state -> Linear -> flow at t+3
"""
import copy
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


class RNNForecaster(nn.Module):
    def __init__(self, cell="lstm", hidden_size=64, num_layers=1, dropout=0.0):
        super().__init__()
        self.cell = cell
        rnn = nn.LSTM if cell == "lstm" else nn.GRU
        self.rnn = rnn(input_size=1, hidden_size=hidden_size, num_layers=num_layers, batch_first=True,
                       dropout=dropout if num_layers > 1 else 0.0)
        self.head = nn.Linear(hidden_size, 1)

    def encode(self, x):                       # [B, 12] -> [B, hidden]: the learned temporal representation
        out, _ = self.rnn(x.unsqueeze(-1))
        return out[:, -1]

    def forward(self, x):
        return self.head(self.encode(x)).squeeze(-1)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


@torch.no_grad()
def run_batched(model, X, fn=None, device="cpu", batch_size=8192):
    model.eval()
    fn = fn or model
    out = [fn(torch.from_numpy(X[i:i + batch_size]).to(device)).cpu().numpy()
           for i in range(0, len(X), batch_size)]
    return np.concatenate(out)


def train(cell, arch, train_params, X_tr, y_tr, X_va, y_va_raw, scaler, device, seed, max_epochs, log=print):
    """X_* scaled, y_tr scaled, y_va_raw in vehicles. Early stopping on validation MAE. Returns model, history."""
    torch.manual_seed(seed)
    model = RNNForecaster(cell, **arch).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=train_params["learning_rate"])
    loss_fn = nn.MSELoss()
    rng = np.random.default_rng(seed)
    bs = train_params["batch_size"]

    history, best, best_state, bad = [], np.inf, None, 0
    for epoch in range(1, max_epochs + 1):
        model.train()
        order, total = rng.permutation(len(X_tr)), 0.0
        for i in range(0, len(order), bs):
            idx = order[i:i + bs]
            xb, yb = torch.from_numpy(X_tr[idx]).to(device), torch.from_numpy(y_tr[idx]).to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        val_mae = float(np.abs(scaler.inverse(run_batched(model, X_va, device=device)) - y_va_raw).mean())
        history.append({"epoch": epoch, "train_mse_scaled": total / len(X_tr), "val_mae": val_mae})
        log(f"      {cell.upper()} epoch {epoch:2d}  train MSE {total / len(X_tr):.4f}  val MAE {val_mae:.3f}")
        if val_mae < best:
            best, best_state, bad = val_mae, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= train_params["patience"]:
                break
    model.load_state_dict(best_state)
    return model, history


def save(model, arch, extra, directory, filename="model.pt"):
    d = Path(directory)
    d.mkdir(parents=True, exist_ok=True)
    torch.save({k: v.cpu() for k, v in model.state_dict().items()}, d / filename)
    (d / "config.json").write_text(json.dumps({"cell": model.cell, "architecture": arch, **extra}, indent=2))


def load(directory, filename="model.pt", device="cpu"):
    d = Path(directory)
    cfg = json.loads((d / "config.json").read_text())
    model = RNNForecaster(cfg["cell"], **cfg["architecture"])
    model.load_state_dict(torch.load(d / filename, map_location="cpu"))
    return model.to(device).eval(), cfg
