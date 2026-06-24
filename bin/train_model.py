#!/usr/bin/env python3
"""
train_model.py — Train Elastic Net logistic regression on TCR features.
Mirrors thesis methodology: stratified k-fold CV, AUC metric, feature normalization.
"""

import pandas as pd
import numpy as np
import argparse
import joblib
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import roc_auc_score, average_precision_score

FEATURE_COLS = [
    'length', 'hydro', 'charge',
    'p_hc', 'p_ms', 'p_sle', 'p_as',
    'v_tirp', 'm_tirp', 'j_tirp',
    'p107_score', 'p113_score'
] + [f'aa_freq_{aa}' for aa in list("ACDEFGHIKLMNPQRSTVWY")]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train',    required=True)
    parser.add_argument('--cv_folds', type=int, default=5)
    parser.add_argument('--seed',     type=int, default=42)
    args = parser.parse_args()

    df = pd.read_csv(args.train)

    # Use only labelled sequences (exclude lab panel with true_label == -1)
    df = df[df['true_label'].isin([0, 1])].copy()
    # Binary label: autoreactive (MS or SLE) = 1, HC = 0
    df['binary_label'] = df['true_label'].astype(int)

    X = df[FEATURE_COLS].values
    y = df['binary_label'].values

    print(f"Training on {len(df)} sequences | Positive: {y.sum()} | Negative: {(y==0).sum()}")

    # Elastic Net logistic regression with CV (mirrors thesis)
    cv = StratifiedKFold(n_splits=args.cv_folds, shuffle=True, random_state=args.seed)

    model = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegressionCV(
            cv=cv,
            solver='saga',
            l1_ratios=[0.1, 0.5, 0.9],
            Cs=10,
            max_iter=2000,
            scoring='roc_auc',
            random_state=args.seed,
            n_jobs=-1
        ))
    ])

    # Cross-validate to get CV metrics
    cv_results = cross_validate(
        model, X, y, cv=cv,
        scoring=['roc_auc', 'average_precision', 'balanced_accuracy'],
        return_train_score=True
    )

    # Fit final model on all training data
    model.fit(X, y)

    # Save
    joblib.dump(model, 'model_elasticnet.pkl')

    # Save feature names
    with open('feature_names.txt', 'w') as f:
        f.write('\n'.join(FEATURE_COLS))

    # CV results
    cv_df = pd.DataFrame({
        'fold':              range(1, args.cv_folds + 1),
        'val_roc_auc':       cv_results['test_roc_auc'],
        'val_avg_precision': cv_results['test_average_precision'],
        'val_bal_accuracy':  cv_results['test_balanced_accuracy'],
        'train_roc_auc':     cv_results['train_roc_auc'],
    })
    cv_df.to_csv('cv_results.csv', index=False)

    with open('training_report.txt', 'w') as f:
        f.write(f"Training sequences: {len(df)}\n")
        f.write(f"Features:           {len(FEATURE_COLS)}\n")
        f.write(f"CV folds:           {args.cv_folds}\n")
        f.write(f"\nCV Results (mean ± std):\n")
        f.write(f"  AUC:              {cv_results['test_roc_auc'].mean():.4f} ± {cv_results['test_roc_auc'].std():.4f}\n")
        f.write(f"  Avg Precision:    {cv_results['test_average_precision'].mean():.4f} ± {cv_results['test_average_precision'].std():.4f}\n")
        f.write(f"  Balanced Acc:     {cv_results['test_balanced_accuracy'].mean():.4f} ± {cv_results['test_balanced_accuracy'].std():.4f}\n")
        clf = model.named_steps['clf']
        f.write(f"\nBest l1_ratio:      {clf.l1_ratio_[0]:.2f}\n")
        f.write(f"Best C:             {clf.C_[0]:.4f}\n")

    print("Training complete.")
    print(f"  CV AUC: {cv_results['test_roc_auc'].mean():.4f}")

if __name__ == '__main__':
    main()
