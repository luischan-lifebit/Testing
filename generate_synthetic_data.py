#!/usr/bin/env python3
"""
Generate a synthetic TCR beta-chain CDR3 dataset based on:
  Luis Chan MS Thesis - Interpretable ML for Autoreactive TCR Discovery (Yale CBB, 2026)

Dataset mimics the 4-source structure:
  - McPAS-TCR  (positive/autoreactive: MS, SLE)
  - TCRdb 2.0  (background healthy control)
  - Mal-ID     (disease probability features)
  - Lab Panel  (independent test, Rodriguez Martinez lab)

CDR3 length constrained to 12-17 aa (TiRP compatibility).
All Mal-ID and TiRP values are simulated from thesis-reported distributions.

Usage:
  python generate_synthetic_data.py --out synthetic_tcr_dataset.csv
"""

import numpy as np
import pandas as pd
import random
import argparse

random.seed(42)
np.random.seed(42)

AA = list("ACDEFGHIKLMNPQRSTVWY")

KD_HYDRO = {
    'A': 1.8,  'C': 2.5,  'D': -3.5, 'E': -3.5, 'F': 2.8,
    'G': -0.4, 'H': -3.2, 'I': 4.5,  'K': -3.9, 'L': 3.8,
    'M': 1.9,  'N': -3.5, 'P': -1.6, 'Q': -3.5, 'R': -4.5,
    'S': -0.8, 'T': -0.7, 'V': 4.2,  'W': -0.9, 'Y': -1.3
}
AA_CHARGE = {'K': 1, 'R': 1, 'H': 0.1, 'D': -1, 'E': -1}

TRBV_MS   = ['TRBV11-3', 'TRBV25-1', 'TRBV6-4',  'TRBV13-1']
TRBV_SLE  = ['TRBV20-1', 'TRBV7-2',  'TRBV12-3', 'TRBV29-1']
TRBV_BG   = ['TRBV2',    'TRBV3-1',  'TRBV4-1',  'TRBV5-1',
             'TRBV9',    'TRBV10-3', 'TRBV18',   'TRBV28']
TRBJ_AUTO = ['TRBJ2-3', 'TRBJ1-5', 'TRBJ2-1']
TRBJ_BG   = ['TRBJ1-1', 'TRBJ1-2', 'TRBJ2-2', 'TRBJ2-4', 'TRBJ2-7']


def random_cdr3(length, bias_hydrophobic=False, bias_charged=False):
    hydrophobic_aa = [a for a in AA if KD_HYDRO.get(a, 0) > 1.5]
    charged_aa     = [a for a in AA if abs(AA_CHARGE.get(a, 0)) > 0]
    seq = ['C']
    for _ in range(1, length - 1):
        if bias_hydrophobic and random.random() < 0.4:
            seq.append(random.choice(hydrophobic_aa))
        elif bias_charged and random.random() < 0.3:
            seq.append(random.choice(charged_aa))
        else:
            seq.append(random.choice(AA))
    seq.append('F')
    return ''.join(seq)


def compute_features(cdr3, v_gene, j_gene):
    length = len(cdr3)
    hydro  = float(np.mean([KD_HYDRO.get(aa, 0) for aa in cdr3]))
    charge = float(sum(AA_CHARGE.get(aa, 0) for aa in cdr3))
    aa_freq = {aa: cdr3.count(aa) / length for aa in AA}

    is_ms_gene  = v_gene in TRBV_MS  and j_gene in TRBJ_AUTO
    is_sle_gene = v_gene in TRBV_SLE and j_gene in TRBJ_AUTO

    base_ms  = np.random.beta(3, 5) if is_ms_gene  else np.random.beta(1, 6)
    base_sle = np.random.beta(3, 5) if is_sle_gene else np.random.beta(1, 6)
    base_as  = np.random.beta(1, 8)
    p_hc     = max(0.01, 1.0 - base_ms - base_sle - base_as + np.random.normal(0, 0.05))
    total    = base_ms + base_sle + base_as + p_hc
    p_ms, p_sle, p_as, p_hc = (base_ms/total, base_sle/total, base_as/total, p_hc/total)

    v_tirp = float(np.random.normal(0.42 if is_ms_gene else 0.25, 0.08))
    m_tirp = float(np.random.normal(-0.10 if is_ms_gene else 0.00, 0.12))
    j_tirp = float(np.random.normal(0.30 if j_gene in TRBJ_AUTO else 0.15, 0.07))
    p107   = float(np.random.normal(0.12 if is_ms_gene else 0.06, 0.03))
    p113   = float(np.random.normal(0.09 if is_ms_gene else 0.05, 0.02))

    return {
        'length':      length,
        'hydro':       round(hydro,  4),
        'charge':      round(charge, 2),
        'p_hc':        round(p_hc,   4),
        'p_ms':        round(p_ms,   4),
        'p_sle':       round(p_sle,  4),
        'p_as':        round(p_as,   4),
        'v_tirp':      round(v_tirp, 4),
        'm_tirp':      round(m_tirp, 4),
        'j_tirp':      round(j_tirp, 4),
        'p107_score':  round(p107,   4),
        'p113_score':  round(p113,   4),
        **{f'aa_freq_{aa}': round(aa_freq[aa], 4) for aa in AA}
    }


def generate_dataset(n_ms=300, n_sle=150, n_hc=600, n_lab=50):
    records = []

    for i in range(n_ms):
        v, j = random.choice(TRBV_MS), random.choice(TRBJ_AUTO)
        cdr3 = random_cdr3(random.randint(12, 17), bias_hydrophobic=True)
        records.append({'cdr3_aa': cdr3, 'v_gene': v, 'j_gene': j,
                        'source': 'McPAS', 'disease_label': 'MS', 'true_label': 1,
                        'subject_id': f'MS_{i:04d}', **compute_features(cdr3, v, j)})

    for i in range(n_sle):
        v, j = random.choice(TRBV_SLE), random.choice(TRBJ_AUTO)
        cdr3 = random_cdr3(random.randint(12, 17), bias_charged=True)
        records.append({'cdr3_aa': cdr3, 'v_gene': v, 'j_gene': j,
                        'source': 'McPAS', 'disease_label': 'SLE', 'true_label': 1,
                        'subject_id': f'SLE_{i:04d}', **compute_features(cdr3, v, j)})

    for i in range(n_hc):
        v, j = random.choice(TRBV_BG), random.choice(TRBJ_BG)
        cdr3 = random_cdr3(random.randint(12, 17))
        records.append({'cdr3_aa': cdr3, 'v_gene': v, 'j_gene': j,
                        'source': 'TCRdb', 'disease_label': 'HC', 'true_label': 0,
                        'subject_id': f'HC_{i:04d}', **compute_features(cdr3, v, j)})

    for i in range(n_lab):
        v = random.choice(TRBV_MS + TRBV_SLE + TRBV_BG)
        j = random.choice(TRBJ_AUTO + TRBJ_BG)
        cdr3 = random_cdr3(random.randint(12, 17))
        records.append({'cdr3_aa': cdr3, 'v_gene': v, 'j_gene': j,
                        'source': 'LabPanel', 'disease_label': 'Unknown', 'true_label': -1,
                        'subject_id': f'LAB_{i:04d}', **compute_features(cdr3, v, j)})

    return pd.DataFrame(records)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate synthetic TCR dataset')
    parser.add_argument('--n_ms',  type=int, default=300, help='MS autoreactive sequences')
    parser.add_argument('--n_sle', type=int, default=150, help='SLE autoreactive sequences')
    parser.add_argument('--n_hc',  type=int, default=600, help='Healthy control sequences')
    parser.add_argument('--n_lab', type=int, default=50,  help='Lab panel sequences')
    parser.add_argument('--out',   type=str, default='synthetic_tcr_dataset.csv')
    args = parser.parse_args()

    df = generate_dataset(args.n_ms, args.n_sle, args.n_hc, args.n_lab)
    df.to_csv(args.out, index=False)
    print(f"Generated {len(df)} sequences -> {args.out}")
    print(df['disease_label'].value_counts().to_string())
