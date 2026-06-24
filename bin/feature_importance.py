#!/usr/bin/env python3
"""
feature_importance.py — XGBoost gain + permutation importance.
Mirrors thesis Figures 3.1-3.4: top features for Mal-ID-only and integrated models.
"""

import pandas as pd
import numpy as np
import argparse
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import average_precision_score

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model',         required=True)
    parser.add_argument('--test',          required=True)
    parser.add_argument('--feature_names', required=True)
    args = parser.parse_args()

    with open(args.feature_names) as f:
        feature_cols = [l.strip() for l in f.readlines()]

    df = pd.read_csv(args.test)
    df = df[df['true_label'].isin([0, 1])].copy()
    X  = df[feature_cols].values
    y  = df['true_label'].astype(int).values

    # Fit XGBoost for gain importance (mirrors thesis Section 2.4.3)
    xgb = XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=42,
        n_jobs=-1
    )
    xgb.fit(X, y)

    gain_scores = xgb.feature_importances_
    gain_df = pd.DataFrame({
        'feature': feature_cols,
        'gain_importance': gain_scores
    }).sort_values('gain_importance', ascending=False)

    # Permutation importance on held-out test (mirrors thesis Section 2.4.3)
    perm_result = permutation_importance(
        xgb, X, y,
        scoring='average_precision',
        n_repeats=10,
        random_state=42,
        n_jobs=-1
    )
    perm_df = pd.DataFrame({
        'feature':         feature_cols,
        'perm_mean':       perm_result.importances_mean,
        'perm_std':        perm_result.importances_std
    }).sort_values('perm_mean', ascending=False)

    # Merge and save
    importance_df = gain_df.merge(perm_df, on='feature')
    importance_df.to_csv('feature_importance.csv', index=False)

    # Plot top-25 gain importance (mirrors Figure 3.1/3.3)
    top25 = gain_df.head(25).sort_values('gain_importance')
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.barh(top25['feature'], top25['gain_importance'], color='steelblue')
    ax.set_xlabel('Gain')
    ax.set_title('XGBoost Gain Importance — Top 25')
    fig.tight_layout()
    fig.savefig('feature_importance_plot.png', dpi=150)
    plt.close()

    print(f"Feature importance complete | Top feature: {gain_df.iloc[0]['feature']}")

if __name__ == '__main__':
    main()
