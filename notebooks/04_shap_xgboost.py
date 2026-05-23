"""
04_shap_xgboost.py
==================
Train XGBoost on a tabular medical dataset, then explain with SHAP.
Corresponds to: Ch18 (Explainable AI) and Ch03 (ML/DL foundations).

Demo uses sklearn's `load_breast_cancer` as a stand-in.

Requirements
------------
- xgboost
- shap
- scikit-learn matplotlib
"""
import numpy as np
import xgboost as xgb
import shap
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss


def main():
    data = load_breast_cancer()
    X, y = data.data, data.target
    feat_names = data.feature_names

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, stratify=y, random_state=7, test_size=0.25
    )

    clf = xgb.XGBClassifier(
        n_estimators=400, max_depth=4, learning_rate=0.05,
        subsample=0.9, colsample_bytree=0.9, eval_metric="logloss",
        random_state=7, n_jobs=4,
    )
    clf.fit(X_tr, y_tr)
    p = clf.predict_proba(X_te)[:, 1]

    print("AUROC :", round(roc_auc_score(y_te, p), 4))
    print("AUPRC :", round(average_precision_score(y_te, p), 4))
    print("Brier :", round(brier_score_loss(y_te, p), 4))

    # SHAP explanations
    expl = shap.TreeExplainer(clf)
    sv = expl.shap_values(X_te)
    print("\nTop 10 features by mean |SHAP|:")
    imp = np.abs(sv).mean(axis=0)
    order = np.argsort(imp)[::-1][:10]
    for i in order:
        print(f"  {feat_names[i]:<35} {imp[i]:.4f}")

    # Save a summary plot if matplotlib is available
    try:
        import matplotlib.pyplot as plt
        shap.summary_plot(sv, X_te, feature_names=feat_names, show=False)
        plt.tight_layout()
        plt.savefig("shap_summary.png", dpi=150)
        print("\nSaved shap_summary.png")
    except Exception as e:
        print(f"(plotting skipped: {e})")


if __name__ == "__main__":
    main()
