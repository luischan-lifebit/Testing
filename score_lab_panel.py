#!/usr/bin/env python3
"""
score_lab_panel.py — Apply trained model to independent lab panel.
Mirrors thesis Section 2.4.4: qualitative scoring of Rodriguez Martinez lab sequences.
"""

import pandas as pd
import numpy as np
import argparse
import joblib

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model',         required=True)
    parser.add_argument('--lab_panel',     required=True)
    parser.add_argument('--feature_names', required=True)
    args = parser.parse_args()

    model = joblib.load(args.model)
    df    = pd.read_csv(args.lab_panel)

    with open(args.feature_names) as f:
        feature_cols = [l.strip() for l in f.readlines()]

    X = df[feature_cols].values
    probs = model.predict_proba(X)[:, 1]

    df['autoreactivity_score'] = probs
    df['autoreactivity_rank']  = df['autoreactivity_score'].rank(ascending=False).astype(int)
    df['candidate'] = df['autoreactivity_score'] > 0.5

    df_sorted = df.sort_values('autoreactivity_rank')
    df_sorted.to_csv('lab_panel_scores.csv', index=False)

    n_candidates = df['candidate'].sum()
    print(f"Lab panel scored | {len(df)} sequences | {n_candidates} candidates (score > 0.5)")
    print(df_sorted[['cdr3_aa', 'v_gene', 'j_gene', 'autoreactivity_score', 'candidate']].head(10).to_string(index=False))

if __name__ == '__main__':
    main()
