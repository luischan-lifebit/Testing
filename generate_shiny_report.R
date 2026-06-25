#!/usr/bin/env Rscript
# generate_shiny_report.R
# Bundles results into an RDS file and writes a self-contained Shiny app.
# Mirrors thesis results: metrics, feature importance, lab panel scores.

suppressPackageStartupMessages({
  library(optparse)
  library(readr)
  library(dplyr)
})

option_list <- list(
  make_option("--metrics",     type="character"),
  make_option("--predictions", type="character"),
  make_option("--importance",  type="character"),
  make_option("--lab_scores",  type="character")
)
opt <- parse_args(OptionParser(option_list=option_list))

metrics     <- read_csv(opt$metrics,     show_col_types=FALSE)
predictions <- read_csv(opt$predictions, show_col_types=FALSE)
importance  <- read_csv(opt$importance,  show_col_types=FALSE)
lab_scores  <- read_csv(opt$lab_scores,  show_col_types=FALSE)

# Save bundled RDS for Shiny to load
saveRDS(
  list(
    metrics     = metrics,
    predictions = predictions,
    importance  = importance,
    lab_scores  = lab_scores
  ),
  file = "report_data.rds"
)

# Write app.R
shiny_code <- '
library(shiny)
library(shinydashboard)
library(ggplot2)
library(dplyr)
library(DT)
library(plotly)

data <- readRDS("report_data.rds")

ui <- dashboardPage(
  dashboardHeader(title = "TCR Autoreactivity Pipeline"),
  dashboardSidebar(
    sidebarMenu(
      menuItem("Overview",          tabName = "overview",    icon = icon("tachometer-alt")),
      menuItem("Feature Importance",tabName = "importance",  icon = icon("chart-bar")),
      menuItem("Lab Panel Scores",  tabName = "labpanel",    icon = icon("flask")),
      menuItem("Predictions",       tabName = "predictions", icon = icon("table"))
    )
  ),
  dashboardBody(
    tabItems(

      # Overview tab
      tabItem(tabName = "overview",
        fluidRow(
          valueBoxOutput("auc_box"),
          valueBoxOutput("ap_box"),
          valueBoxOutput("ms_recall_box")
        ),
        fluidRow(
          box(title="Evaluation Metrics", width=12,
              DTOutput("metrics_table"))
        )
      ),

      # Feature importance tab
      tabItem(tabName = "importance",
        fluidRow(
          box(title="Top 25 Features by Gain Importance", width=8,
              plotlyOutput("gain_plot", height="500px")),
          box(title="Feature Table", width=4,
              DTOutput("importance_table"))
        )
      ),

      # Lab panel tab
      tabItem(tabName = "labpanel",
        fluidRow(
          box(title="Autoreactivity Score Distribution", width=6,
              plotlyOutput("lab_hist")),
          box(title="Top Candidate Sequences", width=6,
              DTOutput("lab_table"))
        )
      ),

      # Predictions tab
      tabItem(tabName = "predictions",
        fluidRow(
          box(title="Predicted Probability by Disease Label", width=8,
              plotlyOutput("pred_box")),
          box(title="Predictions Table", width=12,
              DTOutput("pred_table"))
        )
      )
    )
  )
)

server <- function(input, output, session) {

  # Value boxes
  output$auc_box <- renderValueBox({
    val <- data$metrics %>% filter(metric == "roc_auc") %>% pull(value)
    valueBox(round(val, 3), "ROC AUC", icon=icon("chart-line"), color="blue")
  })
  output$ap_box <- renderValueBox({
    val <- data$metrics %>% filter(metric == "avg_precision") %>% pull(value)
    valueBox(round(val, 3), "Avg Precision", icon=icon("bullseye"), color="green")
  })
  output$ms_recall_box <- renderValueBox({
    val <- data$metrics %>% filter(metric == "ms_recall") %>% pull(value)
    valueBox(round(val, 3), "MS-Specific Recall", icon=icon("dna"), color="red")
  })

  output$metrics_table <- renderDT({
    datatable(data$metrics, options=list(pageLength=10))
  })

  # Feature importance
  output$gain_plot <- renderPlotly({
    top25 <- data$importance %>%
      arrange(desc(gain_importance)) %>%
      head(25) %>%
      arrange(gain_importance)
    p <- ggplot(top25, aes(x=gain_importance, y=reorder(feature, gain_importance))) +
      geom_bar(stat="identity", fill="steelblue") +
      labs(x="Gain", y="Feature", title="XGBoost Gain Importance — Top 25") +
      theme_minimal()
    ggplotly(p)
  })

  output$importance_table <- renderDT({
    datatable(data$importance %>% arrange(desc(gain_importance)),
              options=list(pageLength=10))
  })

  # Lab panel
  output$lab_hist <- renderPlotly({
    p <- ggplot(data$lab_scores, aes(x=autoreactivity_score, fill=candidate)) +
      geom_histogram(bins=20, alpha=0.8) +
      scale_fill_manual(values=c("FALSE"="steelblue","TRUE"="tomato")) +
      labs(x="Autoreactivity Score", y="Count", title="Score Distribution") +
      theme_minimal()
    ggplotly(p)
  })

  output$lab_table <- renderDT({
    data$lab_scores %>%
      select(cdr3_aa, v_gene, j_gene, autoreactivity_score, candidate) %>%
      arrange(desc(autoreactivity_score)) %>%
      datatable(options=list(pageLength=10))
  })

  # Predictions
  output$pred_box <- renderPlotly({
    df <- data$predictions %>% filter(true_label %in% c(0, 1))
    p <- ggplot(df, aes(x=disease_label, y=predicted_prob, fill=disease_label)) +
      geom_boxplot(alpha=0.7) +
      labs(x="Disease Label", y="Predicted Autoreactivity Probability") +
      theme_minimal() +
      theme(legend.position="none")
    ggplotly(p)
  })

  output$pred_table <- renderDT({
    data$predictions %>%
      select(cdr3_aa, v_gene, j_gene, disease_label, predicted_prob, predicted_label) %>%
      datatable(options=list(pageLength=15))
  })
}

shinyApp(ui, server)
'
writeLines(shiny_code, "app.R")
cat("Shiny report generated: app.R + report_data.rds\n")
