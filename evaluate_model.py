#!/usr/bin/env python3
"""
evaluate_model.py — Evaluate trained Elastic Net on held-out test set.
Outputs: metrics CSV, predictions CSV, ROC/PR curve plots.
"""

import pandas as pd
import numpy as np
import argparse
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score, average_precision_score, balanced_accuracy_score,
    roc_curve, precision_recall_curve, confusion_matrix
)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model',         required=True)
    parser.add_argument('--test',          required=True)
    parser.add_argument('--feature_names', required=True)
    args = parser.parse_args()

    model = joblib.load(args.model)
    df    = pd.read_csv(args.test)

    with open(args.feature_names) as f:
        feature_cols = [l.strip() for l in f.readlines()]

    df = df[df['true_label'].isin([0, 1])].copy()
    X  = df[feature_cols].values
    y  = df['true_label'].astype(int).values

    y_prob = model.predict_proba(X)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    # Metrics
    auc    = roc_auc_score(y, y_prob)
    ap     = average_precision_score(y, y_prob)
    bal_ac = balanced_accuracy_score(y, y_pred)

    # MS-specific recall (mirrors thesis Table focus)
    ms_mask   = df['disease_label'] == 'MS'
    ms_recall = y_pred[ms_mask].sum() / ms_mask.sum() if ms_mask.sum() > 0 else 0.0

    metrics = pd.DataFrame([{
        'metric':  m,
        'value':   v
    } for m, v in [
        ('roc_auc',       round(auc,    4)),
        ('avg_precision', round(ap,     4)),
        ('bal_accuracy',  round(bal_ac, 4)),
        ('ms_recall',     round(float(ms_recall), 4)),
        ('n_test',        len(df)),
        ('n_positive',    int(y.sum())),
        ('n_negative',    int((y == 0).sum())),
    ]])
    metrics.to_csv('evaluation_metrics.csv', index=False)

    # Predictions
    df['predicted_prob'] = y_prob
    df['predicted_label'] = y_pred
    df.to_csv('predictions_test.csv', index=False)

    # ROC curve
    fpr, tpr, _ = roc_curve(y, y_prob)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, lw=2, color='steelblue', label=f'AUC = {auc:.3f}')
    ax.plot([0, 1], [0, 1], '--', color='grey', lw=1)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve — Autoreactivity Classifier')
    ax.legend(loc='lower right')
    fig.tight_layout()
    fig.savefig('roc_curve.png', dpi=150)
    plt.close()

    # PR curve
    prec, rec, _ = precision_recall_curve(y, y_prob)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(rec, prec, lw=2, color='tomato', label=f'AP = {ap:.3f}')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curve — Autoreactivity Classifier')
    ax.legend(loc='upper right')
    fig.tight_layout()
    fig.savefig('pr_curve.png', dpi=150)
    plt.close()

    print(f"Evaluation complete | AUC: {auc:.4f} | AP: {ap:.4f} | MS Recall: {ms_recall:.4f}")

if __name__ == '__main__':
    main()
