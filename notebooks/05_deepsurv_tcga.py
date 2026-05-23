"""
05_deepsurv_tcga.py
====================
DeepSurv on a simulated TCGA-style dataset.
Corresponds to: Ch14 (Survival Prediction).

Requirements
------------
- pycox lifelines torchtuples torch numpy pandas

Note: this script generates synthetic data so you can run it without
TCGA access. Replace `simulate_tcga()` with your real data loader.
"""
import numpy as np
import pandas as pd
import torch
import torchtuples as tt
from pycox.models import CoxPH
from pycox.evaluation import EvalSurv


def simulate_tcga(n: int = 1500, p: int = 50, seed: int = 7):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, p)).astype("float32")
    beta = rng.standard_normal(p) * 0.3
    risk = X @ beta
    # Weibull-like times
    t = np.minimum(np.random.exponential(np.exp(-risk - 0.5), size=n) * 12, 60)
    e = (t < np.random.uniform(5, 60, n)).astype(int)
    dur = t.astype("float32")
    return X, dur, e


def main():
    X, dur, evt = simulate_tcga()
    n = len(X)
    idx = np.arange(n)
    np.random.default_rng(0).shuffle(idx)
    train, val, test = idx[: int(.7 * n)], idx[int(.7 * n): int(.85 * n)], idx[int(.85 * n):]

    Xtr, durtr, evttr = X[train], dur[train], evt[train]
    Xva, durva, evtva = X[val], dur[val], evt[val]
    Xte, durte, evtte = X[test], dur[test], evt[test]

    net = tt.practical.MLPVanilla(
        in_features=X.shape[1],
        num_nodes=[128, 64],
        out_features=1,
        batch_norm=True,
        dropout=0.3,
        output_bias=False,
    )
    model = CoxPH(net, tt.optim.Adam(1e-3))

    bs = 128
    log = model.fit(
        Xtr, (durtr, evttr),
        batch_size=bs, epochs=80,
        val_data=(Xva, (durva, evtva)),
        verbose=False,
    )
    model.compute_baseline_hazards()
    surv = model.predict_surv_df(Xte)
    ev = EvalSurv(surv, durte, evtte, censor_surv="km")
    print("Test C-index :", round(ev.concordance_td(), 4))
    print("Test IBS     :", round(ev.integrated_brier_score(
        np.linspace(0, durte.max(), 100)), 4))


if __name__ == "__main__":
    main()
