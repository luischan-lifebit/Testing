#!/usr/bin/env nextflow

nextflow.enable.dsl = 2

log.info """
    TCR Autoreactivity ML Pipeline | Lifebit CloudOS | Nextflow DSL2
    input_csv : ${params.input_csv}
    outdir    : ${params.outdir}
    n_ms      : ${params.n_ms}
    n_sle     : ${params.n_sle}
    n_hc      : ${params.n_hc}
    n_lab     : ${params.n_lab}
    """.stripIndent()

process GENERATE_SYNTHETIC_DATA {
    tag "generate"
    publishDir "${params.outdir}/data", mode: 'copy'

    output:
    path "synthetic_tcr_dataset.csv", emit: dataset

    script:
    """
    generate_synthetic_data.py \
        --n_ms  ${params.n_ms} \
        --n_sle ${params.n_sle} \
        --n_hc  ${params.n_hc} \
        --n_lab ${params.n_lab} \
        --out   synthetic_tcr_dataset.csv
    """
}

process PREPROCESS {
    tag "preprocess"
    publishDir "${params.outdir}/data", mode: 'copy'

    input:
    path dataset

    output:
    path "train_data.csv",        emit: train
    path "test_data.csv",         emit: test
    path "lab_panel.csv",         emit: lab_panel
    path "preprocess_report.txt", emit: report

    script:
    """
    preprocess.py \
        --input     ${dataset} \
        --test_size ${params.test_size} \
        --seed      ${params.seed}
    """
}

process TRAIN_MODEL {
    tag "train"
    publishDir "${params.outdir}/model", mode: 'copy'

    input:
    path train_data

    output:
    path "model_elasticnet.pkl", emit: model
    path "cv_results.csv",       emit: cv_results
    path "feature_names.txt",    emit: feature_names
    path "training_report.txt",  emit: report

    script:
    """
    train_model.py \
        --train    ${train_data} \
        --cv_folds ${params.cv_folds} \
        --seed     ${params.seed}
    """
}

process EVALUATE_MODEL {
    tag "evaluate"
    publishDir "${params.outdir}/results", mode: 'copy'

    input:
    path model
    path test_data
    path feature_names

    output:
    path "evaluation_metrics.csv", emit: metrics
    path "predictions_test.csv",   emit: predictions
    path "roc_curve.png",          emit: roc_plot
    path "pr_curve.png",           emit: pr_plot

    script:
    """
    evaluate_model.py \
        --model         ${model} \
        --test          ${test_data} \
        --feature_names ${feature_names}
    """
}

process FEATURE_IMPORTANCE {
    tag "feature_importance"
    publishDir "${params.outdir}/results", mode: 'copy'

    input:
    path model
    path test_data
    path feature_names

    output:
    path "feature_importance.csv",      emit: importance
    path "feature_importance_plot.png", emit: importance_plot

    script:
    """
    feature_importance.py \
        --model         ${model} \
        --test          ${test_data} \
        --feature_names ${feature_names}
    """
}

process SCORE_LAB_PANEL {
    tag "score_lab_panel"
    publishDir "${params.outdir}/results", mode: 'copy'

    input:
    path model
    path lab_panel
    path feature_names

    output:
    path "lab_panel_scores.csv", emit: scores

    script:
    """
    score_lab_panel.py \
        --model         ${model} \
        --lab_panel     ${lab_panel} \
        --feature_names ${feature_names}
    """
}

process SHINY_REPORT {
    tag "shiny_report"
    publishDir "${params.outdir}/report", mode: 'copy'

    input:
    path metrics
    path predictions
    path importance
    path lab_scores

    output:
    path "app.R",           emit: shiny_app
    path "report_data.rds", emit: report_data

    script:
    """
    generate_shiny_report.R \
        --metrics     ${metrics} \
        --predictions ${predictions} \
        --importance  ${importance} \
        --lab_scores  ${lab_scores}
    """
}

workflow {

    if (params.input_csv != 'GENERATE') {
        dataset_ch = Channel.fromPath(params.input_csv)
    } else {
        GENERATE_SYNTHETIC_DATA()
        dataset_ch = GENERATE_SYNTHETIC_DATA.out.dataset
    }

    PREPROCESS(dataset_ch)
    TRAIN_MODEL(PREPROCESS.out.train)

    EVALUATE_MODEL(
        TRAIN_MODEL.out.model,
        PREPROCESS.out.test,
        TRAIN_MODEL.out.feature_names
    )

    FEATURE_IMPORTANCE(
        TRAIN_MODEL.out.model,
        PREPROCESS.out.test,
        TRAIN_MODEL.out.feature_names
    )

    SCORE_LAB_PANEL(
        TRAIN_MODEL.out.model,
        PREPROCESS.out.lab_panel,
        TRAIN_MODEL.out.feature_names
    )

    SHINY_REPORT(
        EVALUATE_MODEL.out.metrics,
        EVALUATE_MODEL.out.predictions,
        FEATURE_IMPORTANCE.out.importance,
        SCORE_LAB_PANEL.out.scores
    )
}
