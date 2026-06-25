#!/usr/bin/env python3
"""
preprocess.py — Split synthetic TCR dataset into train/test/lab_panel.
Leakage prevention: LabPanel sequences are fully held out (never in train/test).
Subject-level split prevents same subject appearing in both train and test.
"""

import pandas as pd
import numpy as np
import argparse
from sklearn.model_selection import train_test_split

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input',     required=True)
    parser.add_argument('--test_size', type=float, default=0.2)
    parser.add_argument('--seed',      type=int,   default=42)
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    print(f"Loaded {len(df)} sequences")

    # Separate lab panel (always held out — mirrors thesis leakage prevention)
    lab_panel = df[df['source'] == 'LabPanel'].copy()
    modelling = df[df['source'] != 'LabPanel'].copy()

    print(f"  Modelling set: {len(modelling)} | Lab panel: {len(lab_panel)}")

    # Subject-level split to avoid data leakage
    subjects = modelling['subject_id'].unique()
    train_subjects, test_subjects = train_test_split(
        subjects, test_size=args.test_size, random_state=args.seed
    )

    train_df = modelling[modelling['subject_id'].isin(train_subjects)]
    test_df  = modelling[modelling['subject_id'].isin(test_subjects)]

    print(f"  Train: {len(train_df)} | Test: {len(test_df)}")

    train_df.to_csv('train_data.csv',   index=False)
    test_df.to_csv('test_data.csv',    index=False)
    lab_panel.to_csv('lab_panel.csv',  index=False)

    with open('preprocess_report.txt', 'w') as f:
        f.write(f"Total sequences:  {len(df)}\n")
        f.write(f"Train sequences:  {len(train_df)}\n")
        f.write(f"Test sequences:   {len(test_df)}\n")
        f.write(f"Lab panel:        {len(lab_panel)}\n")
        f.write(f"\nTrain label distribution:\n{train_df['disease_label'].value_counts().to_string()}\n")
        f.write(f"\nTest label distribution:\n{test_df['disease_label'].value_counts().to_string()}\n")

    print("Preprocessing complete.")

if __name__ == '__main__':
    main()
