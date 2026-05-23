"""
06_calibration_decision_curve.py
================================
Calibration analysis (reliability diagram, Brier, ECE) + Decision Curve.
Corresponds to: Ch17 (Evaluation, Calibration & Fairness).

Requirements
------------
- scikit-learn matplotlib numpy pandas dcurves (optional)

Usage
-----
$ python 06_calibration_decision_curve.py
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from sklearn.metrics import brier_score_loss, roc_auc_score


def expected_calibration_error(y, p, n_bins: int = 10):
    bins = np.linspace(0, 1, n_bins + 1)
    binids = np.digitize(p, bins) - 1
    ece, n = 0.0, len(y)
    for b in range(n_bins):
        mask = binids == b
        if mask.any():
            ece += (mask.sum() / n) * abs(y[mask].mean() - p[mask].mean())
    return ece


def main():
    X, y = make_classification(
        n_samples=4000, n_features=24, n_informative=8, weights=[0.85, 0.15],
        random_state=7,
    )
    Xtr, Xte = X[:3000], X[3000:]
    ytr, yte = y[:3000], y[3000:]

    raw = GradientBoostingClassifier(n_estimators=200, random_state=7).fit(Xtr, ytr)
    p_raw = raw.predict_proba(Xte)[:, 1]

    # Calibration via isotonic
    cal = CalibratedClassifierCV(GradientBoostingClassifier(n_estimators=200, random_state=7),
                                 method="isotonic", cv=5).fit(Xtr, ytr)
    p_cal = cal.predict_proba(Xte)[:, 1]

    print("                AUROC    Brier    ECE")
    for name, p in [("raw", p_raw), ("isotonic", p_cal)]:
        print(f"{name:>14}  {roc_auc_score(yte, p):.4f}  {brier_score_loss(yte, p):.4f}  "
              f"{expected_calibration_error(yte, p):.4f}")

    # Reliability diagram
    fig, ax = plt.subplots(figsize=(5.5, 5))
    for name, p in [("raw", p_raw), ("isotonic", p_cal)]:
        prob_true, prob_pred = calibration_curve(yte, p, n_bins=10, strategy="quantile")
        ax.plot(prob_pred, prob_true, marker="o", label=name)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="ideal")
    ax.set(xlabel="Predicted probability", ylabel="Observed frequency",
           title="Reliability diagram")
    ax.legend()
    fig.tight_layout()
    fig.savefig("reliability.png", dpi=150)
    print("\nSaved reliability.png")

    # Decision Curve (manual; net benefit)
    thresholds = np.arange(0.01, 0.6, 0.02)
    nb_model, nb_all = [], []
    prev = yte.mean()
    for t in thresholds:
        pred = (p_cal >= t).astype(int)
        tp = ((pred == 1) & (yte == 1)).mean()
        fp = ((pred == 1) & (yte == 0)).mean()
        nb = tp - fp * (t / (1 - t))
        nb_model.append(nb)
        nb_all.append(prev - (1 - prev) * (t / (1 - t)))
    plt.figure(figsize=(5.5, 4))
    plt.plot(thresholds, nb_model, label="Model (isotonic)")
    plt.plot(thresholds, nb_all, label="Treat all", linestyle="--")
    plt.axhline(0, color="k", linestyle=":", label="Treat none")
    plt.xlabel("Decision threshold"); plt.ylabel("Net benefit")
    plt.title("Decision Curve Analysis")
    plt.legend(); plt.tight_layout(); plt.savefig("dca.png", dpi=150)
    print("Saved dca.png")


if __name__ == "__main__":
    main()
