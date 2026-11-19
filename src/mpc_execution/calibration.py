import numpy as np
import pandas as pd

def brier(y, p) -> float:
    """Mean squared error of the prediction: mean((p - y)^2). 0 is perfect; a constant p = mean(y) gives var(y)."""
    assert len(p) == len(y)
    y, p = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    return float(np.mean((p - y) ** 2))

def log_loss(y, p, eps: float = 1e-12) -> float:
    """Cross-entropy -mean(y log p + (1 - y) log(1 - p)), with p clipped to [eps, 1 - eps]. Valid for fractional y."""
    assert len(p) == len(y)
    y, p = np.asarray(y, dtype=float), np.clip(np.asarray(p, dtype=float), eps, 1 - eps)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))

def reliability_table(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    """Calibration by equal-count bins of p. One row per bin, in increasing order of p.
    Columns: predicted (mean p), observed (mean y), sem (standard error of observed), count."""
    assert len(p) == len(y)
    y, p = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    order = np.argsort(p)                            # indices that sort p, lowest first
    rows = []
    for idx in np.array_split(order, n_bins):        # equal-count groups of those indices
        rows.append({"predicted": p[idx].mean(),
                     "observed": y[idx].mean(),
                     "sem": y[idx].std(ddof=1) / np.sqrt(len(idx)),
                     "count": len(idx)})
    return pd.DataFrame(rows)