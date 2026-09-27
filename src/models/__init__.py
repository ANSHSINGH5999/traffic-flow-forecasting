# XGBoost must be imported before torch: both bundle OpenMP, and the reverse order segfaults on macOS.
import xgboost  # noqa: F401
import torch

# With both OpenMP runtimes loaded, torch CPU thread pools deadlock XGBoost; RNNs train on the GPU anyway.
torch.set_num_threads(1)
