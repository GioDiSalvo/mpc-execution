import numpy as np

def twap(Q: int, n_steps: int) -> np.ndarray:
    """s[k] = round(Q * k / n). Equal slices."""
    return np.round(Q * np.arange(n_steps + 1) / n_steps).astype(int)


def almgren_chriss(Q: int, n_steps: int, psi: float) -> np.ndarray:
    """s[k] = round(Q * (1 - sinh(psi * (n - k)) / sinh(psi * n))). Front-loaded for psi > 0;
    psi is per step. Tends to TWAP as psi -> 0."""
    k = np.arange(n_steps + 1)
    s = Q * (1 - np.sinh(psi * (n_steps - k)) / np.sinh(psi * n_steps))
    return np.round(s).astype(int)

def from_volume_profile(Q: int, cum_volume: np.ndarray) -> np.ndarray:
    """VWAP-style: s[k] = round(Q * cum_volume[k] / cum_volume[-1]). cum_volume has length n+1,
    is non-decreasing and starts at 0. With a flat profile this is TWAP."""
    cum_volume = np.asarray(cum_volume, dtype=float)
    assert cum_volume[0] == 0 and np.all(np.diff(cum_volume) >= 0) and cum_volume[-1] > 0
    return np.round(Q * cum_volume / cum_volume[-1]).astype(int)