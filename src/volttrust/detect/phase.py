"""Which phase each home is on, when STEG's map is missing.

With a substation reading: for each home and phase,
    score = corr(change in home voltage minus the 3-phase mean, change in that phase's voltage minus
                 the 3-phase mean)
          + corr(change in home power, change in that phase's power at the substation)
over a quiet learning window. Homes on the same wire dip together, and a home's load steps show up
in its own phase's power. The best phase wins; margin = best minus second-best score.

Without a substation reading, homes on each feeder are grouped by how their voltage changes
correlate (k-means into 3 groups). Group labels are arbitrary, so they stand in for phases only within the
detector.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans


def _corr(x: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Pearson correlation of x (n,) with each column of Y (n, k), on rows where all are finite."""
    ok = np.isfinite(x) & np.isfinite(Y).all(1)
    if ok.sum() < 10:
        return np.full(Y.shape[1], np.nan)
    x = x[ok] - x[ok].mean()
    Y = Y[ok] - Y[ok].mean(0)
    den = np.sqrt((x ** 2).sum() * (Y ** 2).sum(0))
    return (x @ Y) / np.where(den > 0, den, np.nan)


def infer(V: np.ndarray, P: np.ndarray, feeder: np.ndarray, head_v: np.ndarray | None,
          head_p: np.ndarray | None, window: slice, use_changes: bool = True,
          use_power: bool = True) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (phase (H,), margin (H,), scores (H, 3)). V, P: (T, H); head_v, head_p: (T, F, 3)."""
    H = V.shape[1]
    d = (lambda a: np.diff(a, axis=0)) if use_changes else (lambda a: a[1:])
    Vw = np.where(V[window] > 50, V[window], np.nan)
    # quiet rows only: a phase out (outage, phase loss) would make every home on the feeder move
    # together and drown the per-phase signal, so those hours and the 4 h after are left out
    F = int(feeder.max()) + 1
    quiet = np.ones((Vw.shape[0], F), bool)
    for f in range(F):
        if head_v is not None:
            dead = (head_v[window, f, :] < 0.9 * np.nanmedian(head_v[window, f, :])).any(1)
        else:
            dead = (V[window][:, feeder == f] < 50).mean(1) > 0.2
        dead = np.convolve(dead.astype(float), np.ones(17), "full")[:len(dead)] > 0
        quiet[:, f] = ~dead
    Vw = np.where(quiet[:, feeder], Vw, np.nan)
    phase = np.zeros(H, int)
    margin = np.zeros(H)
    scores = np.full((H, 3), np.nan)
    if head_v is not None:
        for j in range(H):
            f = feeder[j]
            hv = np.where(quiet[:, f, None], head_v[window, f, :], np.nan)
            mean = hv.mean(1)
            sc = _corr(d(Vw[:, j] - mean), d(hv - mean[:, None]))
            if use_power and head_p is not None:
                pw = np.where(quiet[:, f], P[window, j], np.nan)
                hp = np.where(quiet[:, f, None], head_p[window, f, :], np.nan)
                sc = sc + np.nan_to_num(_corr(np.diff(pw), np.diff(hp, axis=0)))
            scores[j] = sc
        o = np.sort(np.nan_to_num(scores, nan=-9), axis=1)
        phase = np.nanargmax(np.nan_to_num(scores, nan=-9), axis=1)
        margin = o[:, -1] - o[:, -2]
        return phase, margin, scores

    # no substation reading: group homes on each feeder by voltage co-movement
    for f in np.unique(feeder):
        sel = np.where(feeder == f)[0]
        X = Vw[:, sel] - np.nanmean(Vw[:, sel], axis=1, keepdims=True)
        X = d(X)
        C = np.nan_to_num(pd.DataFrame(X).corr(min_periods=10).to_numpy())
        lab = KMeans(3, n_init=20, random_state=0).fit_predict(C)
        cent = np.stack([C[:, lab == g].mean(1) if (lab == g).any() else np.full(len(sel), -1.0)
                         for g in range(3)], axis=1)
        o = np.sort(cent, axis=1)
        phase[sel] = lab
        margin[sel] = o[:, -1] - o[:, -2]
        scores[sel] = cent
    return phase, margin, scores
