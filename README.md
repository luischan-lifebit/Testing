# TCR Autoreactivity Pipeline

**Interpretable ML for Autoreactive TCR Discovery — Lifebit CloudOS**

Based on: *Luis Chan MS Thesis, Yale CBB 2026*

---

## Overview

This Nextflow pipeline implements the computational framework from the thesis:
- Generates (or accepts) a synthetic TCR β-chain CDR3 dataset
- Preprocesses and splits data with leakage prevention
- Trains an **Elastic Net logistic regression** classifier
- Evaluates with AUC, average precision, and MS-specific recall
- Computes **XGBoost gain + permutation feature importance**
- Scores an independent lab panel
- Produces an interactive **R Shiny dashboard**

---

## Repository Structure

```
.
├── main.nf                        # Nextflow DSL2 pipeline
├── nextflow.config                # Config: params, Docker, profiles
├── Dockerfile                     # Container definition
├── requirements.txt               # Python dependencies
├── data/
│   └── synthetic_tcr_dataset.csv # Pre-generated dataset (1100 sequences)
└── bin/
    ├── generate_synthetic_data.py # Synthetic dataset generator
    ├── preprocess.py              # Train/test/lab split
    ├── train_model.py             # Elastic Net training
    ├── evaluate_model.py          # Metrics + ROC/PR plots
    ├── feature_importance.py      # XGBoost + permutation importance
    ├── score_lab_panel.py         # Independent panel scoring
    └── generate_shiny_report.R    # R Shiny dashboard
```

---

## Quick Start

### 1. Build and push Docker image

```bash
docker build -t YOUR_DOCKERHUB_USERNAME/tcr-autoreactivity:latest .
docker push YOUR_DOCKERHUB_USERNAME/tcr-autoreactivity:latest
```

Then update `nextflow.config`:
```groovy
process {
    container = 'YOUR_DOCKERHUB_USERNAME/tcr-autoreactivity:latest'
}
```

### 2. Run locally

```bash
nextflow run main.nf -profile local
```

### 3. Run with your own CSV

```bash
nextflow run main.nf --input_csv /path/to/your_data.csv -profile local
```

### 4. Run on Lifebit CloudOS

1. Import this GitHub repo as a pipeline in CloudOS (Advanced Analytics → Pipelines → Import)
2. Set parameters in the CloudOS UI:
   - `input_csv`: upload your CSV or leave as `GENERATE`
   - `outdir`: e.g. `s3://your-bucket/tcr-results`
3. Submit the job

---

## Running on Lifebit CloudOS

**You do NOT need to use Lifebit's own repos.** CloudOS imports directly from your GitHub:

> Advanced Analytics → Pipelines → New Pipeline → paste your GitHub URL

The pipeline is then versioned by your GitHub branches/tags.

### Requirements for CloudOS:
- Docker image pushed to a public or accessible registry (DockerHub, ECR, etc.)
- `nextflow.config` with `docker.enabled = true`
- Pipeline parameters exposed via `params {}` in `nextflow.config`

---

## Synthetic Dataset

The synthetic dataset (`data/synthetic_tcr_dataset.csv`) contains **1,100 sequences** across 4 sources:

| Source   | Label   | N   | Description                          |
|----------|---------|-----|--------------------------------------|
| McPAS    | MS      | 300 | MS-associated CDR3 (TRBV11-3, etc.)  |
| McPAS    | SLE     | 150 | SLE-associated CDR3 (TRBV20-1, etc.) |
| TCRdb    | HC      | 600 | Healthy control background           |
| LabPanel | Unknown | 50  | Independent test panel (held out)    |

Features per sequence:
- CDR3 length, hydrophobicity (Kyte-Doolittle), net charge
- Simulated Mal-ID disease probabilities: `p_hc`, `p_ms`, `p_sle`, `p_as`
- Simulated TiRP components: `v_tirp`, `m_tirp`, `j_tirp`, `p107_score`, `p113_score`
- Per-amino-acid frequencies (20 features)

**Total: 32 features per sequence**

---

## Pipeline Parameters

| Parameter   | Default    | Description                              |
|-------------|------------|------------------------------------------|
| `input_csv` | `GENERATE` | Input CSV path, or `GENERATE` for synthetic |
| `outdir`    | `results`  | Output directory                         |
| `n_ms`      | 300        | MS sequences to generate                 |
| `n_sle`     | 150        | SLE sequences to generate                |
| `n_hc`      | 600        | Healthy control sequences                |
| `n_lab`     | 50         | Lab panel sequences                      |
| `test_size` | 0.2        | Train/test split fraction                |
| `cv_folds`  | 5          | Stratified k-fold CV folds               |
| `seed`      | 42         | Random seed                              |

---

## Outputs

```
results/
├── data/
│   ├── synthetic_tcr_dataset.csv
│   ├── train_data.csv
│   ├── test_data.csv
│   └── lab_panel.csv
├── model/
│   ├── model_elasticnet.pkl
│   ├── cv_results.csv
│   └── training_report.txt
├── results/
│   ├── evaluation_metrics.csv
│   ├── predictions_test.csv
│   ├── roc_curve.png
│   ├── pr_curve.png
│   ├── feature_importance.csv
│   ├── feature_importance_plot.png
│   └── lab_panel_scores.csv
├── report/
│   ├── app.R               ← R Shiny dashboard
│   └── report_data.rds
└── pipeline_info/
    ├── timeline.html
    ├── report.html
    └── trace.txt
```

---

## References

- Chan L. *Interpretable Machine Learning for Autoreactive T-Cell Receptor Discovery*. Yale CBB MS Thesis, 2026.
- Lagattuta et al., *Nature Immunology*, 2022 (TiRP)
- Zaslavsky et al., *Science*, 2025 (Mal-ID)
- Wu et al., *PMLR*, 2024 (TCR-BERT)
