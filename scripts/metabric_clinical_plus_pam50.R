# ============================================================================
# METABRIC: does PAM50 gene expression add prognostic value over clinical data?
# ----------------------------------------------------------------------------
# 5-year survival classification. Three models, all trained/tested on the SAME
# patients (the subset with expression data), using one shared train/test split:
#
#   1. XGBoost  — clinicopathologic features only
#   2. Logistic — NPI only            (simple baseline anchor)
#   3. XGBoost  — clinical + PAM50 gene expression
#
# Key comparison: model 3 vs model 1 (same model class, nested feature sets)
# -> a clean test of whether expression adds anything beyond clinical variables.
#
# NOTE: expression-derived classifiers (CLAUDIN_SUBTYPE, THREEGENE, INTCLUST)
# are deliberately EXCLUDED from the clinical set, so "clinical" is genuinely
# clinicopathologic and the genomic comparison isn't confounded.
#
# Install once:
#   install.packages(c("xgboost", "shapviz", "pROC", "ggplot2"))
# ============================================================================

set.seed(42)

library(xgboost)
library(shapviz)
library(pROC)
library(ggplot2)
library(data.table)

data_dir <- "P:/metabric_cancer/data"
out_dir  <- "P:/metabric_cancer/outputs/clinical_pam50"
dir.create(out_dir, showWarnings = FALSE)

# ---------------------------------------------------------------------------
# 1. LOAD CLINICAL
# ---------------------------------------------------------------------------
read_cbio <- function(path) {
  read.delim(path, skip = 4, stringsAsFactors = FALSE, check.names = TRUE,
             na.strings = c("", "NA", "NaN"))
}
patient <- read_cbio(file.path(data_dir, "data_clinical_patient.txt"))
sample  <- read_cbio(file.path(data_dir, "data_clinical_sample.txt"))
clin    <- merge(patient, sample, by = "PATIENT_ID")
cat("Clinical rows:", nrow(clin), "\n")

# ---------------------------------------------------------------------------
# 2. LOAD PAM50 EXPRESSION
#    check.names = FALSE so sample IDs like "MB-0000" are NOT mangled to "MB.0000".
# ---------------------------------------------------------------------------
pam50 <- c(
  "ACTR3B","ANLN","BAG1","BCL2","BIRC5","BLVRA","CCNB1","CCNE1","CDC20","CDC6",
  "CDH3","CENPF","CEP55","CXXC5","EGFR","ERBB2","ESR1","EXO1","FGFR4","FOXA1",
  "FOXC1","GPR160","GRB7","KIF2C","KRT14","KRT17","KRT5","MAPT","MDM2","MELK",
  "MIA","MKI67","MLPH","MMP11","MYBL2","MYC","NAT1","NDC80","NUF2","ORC6L",
  "PGR","PHGDH","PTTG1","RRM2","SFRP1","SLC39A6","TMEM45B","TYMS","UBE2C","UBE2T"
)

# expr_raw <- read.delim(file.path(data_dir, "data_mrna_illumina_microarray"),
#                        stringsAsFactors = FALSE, check.names = FALSE,
#                        na.strings = c("", "NA", "NaN"))

expr_raw <- fread("P:/metabric_cancer/data/data_mrna_illumina_microarray.txt")
expr_raw <- as.data.frame(expr_raw)

meta_cols   <- c("Hugo_Symbol", "Entrez_Gene_Id")
sample_cols <- setdiff(names(expr_raw), meta_cols)

expr_mat <- as.matrix(expr_raw[, sample_cols])   # genes x samples
storage.mode(expr_mat) <- "numeric"
rownames(expr_mat) <- expr_raw$Hugo_Symbol

pam_present <- intersect(pam50, rownames(expr_mat))
cat("PAM50 genes found:", length(pam_present), "of 50\n")
if (length(setdiff(pam50, pam_present)))
  cat("  (missing:", paste(setdiff(pam50, pam_present), collapse = ", "), ")\n")

expr_pam <- t(expr_mat[pam_present, , drop = FALSE])   # samples x genes
# rownames(expr_pam) are SAMPLE_IDs

# ---------------------------------------------------------------------------
# 3. BINARY TARGET (5-year survival, honest censoring handling)
# ---------------------------------------------------------------------------
os_months <- as.numeric(clin$OS_MONTHS)
deceased  <- grepl("DECEASED", clin$OS_STATUS, ignore.case = TRUE)

horizon <- 60
label <- rep(NA_integer_, nrow(clin))
label[deceased  & os_months <= horizon] <- 1L
label[deceased  & os_months >  horizon] <- 0L
label[!deceased & os_months >= horizon] <- 0L

# ---------------------------------------------------------------------------
# 4. SHARED COHORT: known 5-year status AND expression available
# ---------------------------------------------------------------------------
gi         <- match(clin$SAMPLE_ID, rownames(expr_pam))   # NA if no expression
keep       <- !is.na(label) & !is.na(os_months) & !is.na(gi)

clin_k <- clin[keep, ]
y      <- label[keep]
expr_k <- expr_pam[gi[keep], , drop = FALSE]   # rows aligned to clin_k

cat("\nShared cohort (clinical + expression):", nrow(clin_k), "patients\n")
cat("Class balance (0 = survived, 1 = died <5y):\n"); print(table(y))

# ---------------------------------------------------------------------------
# 5. CLINICAL FEATURE MATRIX  (clinicopathologic only — no molecular subtypes)
# ---------------------------------------------------------------------------
clin_features <- c(
  "AGE_AT_DIAGNOSIS", "TUMOR_SIZE", "LYMPH_NODES_EXAMINED_POSITIVE",
  "NPI", "GRADE", "TUMOR_STAGE",
  "CELLULARITY", "ER_STATUS", "PR_STATUS", "HER2_STATUS", "ER_IHC", "HER2_SNP6",
  "HORMONE_THERAPY", "RADIO_THERAPY", "CHEMOTHERAPY",
  "INFERRED_MENOPAUSAL_STATE", "LATERALITY", "BREAST_SURGERY",
  "TYPE_OF_BREAST_SURGERY", "HISTOLOGICAL_SUBTYPE",
  "ONCOTREE_CODE", "CANCER_TYPE_DETAILED"
  # excluded on purpose: CLAUDIN_SUBTYPE, THREEGENE, INTCLUST (expression-derived)
)
num_like <- c("AGE_AT_DIAGNOSIS", "TUMOR_SIZE", "LYMPH_NODES_EXAMINED_POSITIVE",
              "NPI", "GRADE", "TUMOR_STAGE")

feat <- intersect(clin_features, names(clin_k))
df   <- clin_k[, feat, drop = FALSE]
for (nm in intersect(num_like, names(df)))
  df[[nm]] <- suppressWarnings(as.numeric(df[[nm]]))
for (nm in setdiff(names(df), intersect(num_like, names(df))))
  df[[nm]] <- factor(df[[nm]])

one_hot <- function(d) {
  out <- list()
  for (nm in names(d)) {
    col <- d[[nm]]
    if (is.numeric(col)) { out[[nm]] <- col } else {
      for (lv in levels(col)) {
        v <- as.integer(col == lv); v[is.na(col)] <- NA_integer_
        out[[paste0(nm, "_", lv)]] <- v
      }
    }
  }
  as.matrix(as.data.frame(out, check.names = FALSE))
}
X_clin <- one_hot(df)
X_both <- cbind(X_clin, expr_k)                 # clinical + PAM50
npi    <- suppressWarnings(as.numeric(clin_k$NPI))

cat(sprintf("\nClinical matrix: %d x %d | +PAM50: %d x %d\n",
            nrow(X_clin), ncol(X_clin), nrow(X_both), ncol(X_both)))

# ---------------------------------------------------------------------------
# 6. ONE SHARED TRAIN / TEST SPLIT (stratified on outcome) — reused by all models
# ---------------------------------------------------------------------------
strat_split <- function(g, p = 0.8) {
  idx <- integer(0)
  for (cls in unique(g)) {
    rows <- which(g == cls)
    idx  <- c(idx, sample(rows, floor(p * length(rows))))
  }
  sort(idx)
}
tr  <- strat_split(y, 0.8)
spw <- sum(y[tr] == 0) / sum(y[tr] == 1)

# ---------------------------------------------------------------------------
# 7. FIT THE THREE MODELS
# ---------------------------------------------------------------------------
fit_xgb <- function(Xtr, ytr) {
  dtr <- xgb.DMatrix(Xtr, label = ytr)
  params <- list(objective = "binary:logistic", eval_metric = "auc",
                 max_depth = 4, eta = 0.05, subsample = 0.8,
                 colsample_bytree = 0.8, min_child_weight = 5,
                 scale_pos_weight = spw)
  cv <- xgb.cv(params, dtr, nrounds = 1000, nfold = 5,
               early_stopping_rounds = 30, verbose = 0)
  nb <- which.max(cv$evaluation_log$test_auc_mean)
  xgb.train(params, dtr, nrounds = nb)
}

m_clin <- fit_xgb(X_clin[tr, ], y[tr])
m_both <- fit_xgb(X_both[tr, ], y[tr])
gl_npi <- glm(yy ~ NPI, data = data.frame(yy = y[tr], NPI = npi[tr]),
              family = binomial)

# predictions on the held-out test rows
p_clin <- predict(m_clin, X_clin[-tr, ])
p_both <- predict(m_both, X_both[-tr, ])
p_npi  <- predict(gl_npi, newdata = data.frame(NPI = npi[-tr]), type = "response")

# ---------------------------------------------------------------------------
# 8. EVALUATE — all on the SAME test patients (those with a usable NPI)
# ---------------------------------------------------------------------------
yt     <- y[-tr]
common <- !is.na(p_npi)
yt <- yt[common]

r_npi  <- roc(yt, p_npi[common],  quiet = TRUE)
r_clin <- roc(yt, p_clin[common], quiet = TRUE)
r_both <- roc(yt, p_both[common], quiet = TRUE)

show_auc <- function(name, r)
  cat(sprintf("%-22s AUROC %.3f  (95%% CI %.3f-%.3f)\n",
              name, auc(r), ci.auc(r)[1], ci.auc(r)[3]))
cat("\n"); show_auc("NPI only (logistic)", r_npi)
show_auc("Clinical (XGBoost)",       r_clin)
show_auc("Clinical + PAM50 (XGBoost)", r_both)

cat("\nDeLong tests:\n")
cat("  Clinical+PAM50 vs Clinical  (does expression add value?):\n")
print(roc.test(r_both, r_clin, method = "delong"))
cat("  Clinical vs NPI:\n")
print(roc.test(r_clin, r_npi, method = "delong"))
cat("  Clinical+PAM50 vs NPI:\n")
print(roc.test(r_both, r_npi, method = "delong"))

# ---------------------------------------------------------------------------
# 9. THREE-CURVE ROC PLOT (publication-ready)
# ---------------------------------------------------------------------------
roc_to_df <- function(r, lab) {
  d <- data.frame(fpr = 1 - r$specificities, tpr = r$sensitivities, model = lab)
  d[order(d$fpr, d$tpr), ]
}
lab <- function(name, r)
  sprintf("%s — AUC %.3f (%.3f-%.3f)", name, auc(r), ci.auc(r)[1], ci.auc(r)[3])
L_npi  <- lab("NPI only",          r_npi)
L_clin <- lab("Clinical",          r_clin)
L_both <- lab("Clinical + PAM50",  r_both)

roc_data <- rbind(roc_to_df(r_npi, L_npi),
                  roc_to_df(r_clin, L_clin),
                  roc_to_df(r_both, L_both))
roc_data$model <- factor(roc_data$model, levels = c(L_both, L_clin, L_npi))
pal <- setNames(c("#009E73", "#0072B2", "#E69F00"), c(L_both, L_clin, L_npi))

p <- ggplot(roc_data, aes(fpr, tpr, colour = model)) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed",
              colour = "grey60", linewidth = 0.4) +
  geom_line(linewidth = 1.1) +
  scale_colour_manual(values = pal, name = NULL) +
  scale_x_continuous("1 − Specificity (false positive rate)",
                     limits = c(0, 1), expand = c(0, 0), breaks = seq(0, 1, .25)) +
  scale_y_continuous("Sensitivity (true positive rate)",
                     limits = c(0, 1), expand = c(0, 0), breaks = seq(0, 1, .25)) +
  coord_equal() +
  labs(title = "5-year survival: clinical vs clinical + PAM50 expression",
       subtitle = "METABRIC · shared expression cohort · held-out test set") +
  theme_minimal(base_size = 13) +
  theme(
    plot.title       = element_text(face = "bold", size = 13.5),
    plot.subtitle    = element_text(colour = "grey35", margin = margin(b = 8)),
    panel.grid.minor = element_blank(),
    panel.grid.major = element_line(colour = "grey92", linewidth = 0.3),
    panel.border     = element_rect(colour = "grey80", fill = NA, linewidth = 0.5),
    legend.position  = c(0.98, 0.02), legend.justification = c(1, 0),
    legend.background = element_rect(fill = alpha("white", 0.85), colour = NA),
    legend.key       = element_blank(),
    plot.margin      = margin(12, 16, 12, 12)
  )
print(p)
ggsave(file.path(out_dir, "roc_clinical_vs_pam50.png"), p,
       width = 6.5, height = 6.5, dpi = 300, bg = "white")
ggsave(file.path(out_dir, "roc_clinical_vs_pam50.pdf"), p,
       width = 6.5, height = 6.5, bg = "white")

# ---------------------------------------------------------------------------
# 10. SHAP ON THE CLINICAL + PAM50 MODEL
#     Shows whether any genes rise to the top alongside clinical drivers.
# ---------------------------------------------------------------------------
X_both_test <- X_both[-tr, ]
sv <- shapviz(m_both, X_pred = X_both_test, X = as.data.frame(X_both_test))

ggsave(file.path(out_dir, "shap_both_beeswarm.png"),
       sv_importance(sv, kind = "beeswarm", max_display = 20) +
         ggtitle("SHAP — clinical + PAM50 (top 20)"),
       width = 9, height = 7, dpi = 150, bg = "white")

ggsave(file.path(out_dir, "shap_both_importance.png"),
       sv_importance(sv, kind = "bar", max_display = 20) +
         ggtitle("Mean |SHAP| — clinical + PAM50 (top 20)"),
       width = 9, height = 7, dpi = 150, bg = "white")

message("Done. Metrics printed above; figures written to ", out_dir)
