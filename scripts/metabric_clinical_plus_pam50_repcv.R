# ============================================================================
# METABRIC: does PAM50 expression add value over clinical data? — REPEATED CV
# ----------------------------------------------------------------------------
# Same three models as before, but instead of ONE 80/20 split we repeat a
# stratified split many times (Monte-Carlo CV) and look at the DISTRIBUTION of
# performance. This tells us whether the ~0.02 AUROC edge for PAM50 is a stable
# signal or an artifact of one lucky partition.
#
#   1. XGBoost  — clinicopathologic features only
#   2. Logistic — NPI only
#   3. XGBoost  — clinical + PAM50 gene expression
#
# For each repeat we record each model's test AUROC (on the same patients) and
# the paired differences. We report mean AUROC, a 2.5-97.5% interval, and the
# fraction of repeats where each model beats another ("win rate").
#
# STATS NOTE: the repeats share overlapping training data, so they are NOT
# independent — do not run a t-test on the differences and call it a p-value.
# Repeated CV stabilises the ESTIMATE; the formal significance test remains the
# single-split DeLong test (in the previous script), which respects the paired
# per-patient structure. Here we read significance off whether the ΔAUROC
# interval excludes 0 and how lopsided the win rate is.
#
# Install once:
#   install.packages(c("xgboost", "shapviz", "pROC", "ggplot2", "data.table"))
# ============================================================================

set.seed(42)

library(xgboost)
library(shapviz)
library(pROC)
library(ggplot2)
library(data.table)

data_dir <- "P:/metabric_cancer/data"
out_dir  <- "P:/metabric_cancer/outputs/clinical_pam50_cv"
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

N_REPEATS <- 25          # raise to 50-100 for smoother estimates (slower)

# ---------------------------------------------------------------------------
# 1-5. DATA PREP (identical to the single-split script)
# ---------------------------------------------------------------------------
read_cbio <- function(path)
  read.delim(path, skip = 4, stringsAsFactors = FALSE, check.names = TRUE,
             na.strings = c("", "NA", "NaN"))

patient <- read_cbio(file.path(data_dir, "data_clinical_patient.txt"))
sample  <- read_cbio(file.path(data_dir, "data_clinical_sample.txt"))
clin    <- merge(patient, sample, by = "PATIENT_ID")

pam50 <- c(
  "ACTR3B","ANLN","BAG1","BCL2","BIRC5","BLVRA","CCNB1","CCNE1","CDC20","CDC6",
  "CDH3","CENPF","CEP55","CXXC5","EGFR","ERBB2","ESR1","EXO1","FGFR4","FOXA1",
  "FOXC1","GPR160","GRB7","KIF2C","KRT14","KRT17","KRT5","MAPT","MDM2","MELK",
  "MIA","MKI67","MLPH","MMP11","MYBL2","MYC","NAT1","NDC80","NUF2","ORC6L",
  "PGR","PHGDH","PTTG1","RRM2","SFRP1","SLC39A6","TMEM45B","TYMS","UBE2C","UBE2T"
)

expr_raw <- as.data.frame(fread(file.path(data_dir, "data_mrna_illumina_microarray.txt")))
meta_cols   <- c("Hugo_Symbol", "Entrez_Gene_Id")
sample_cols <- setdiff(names(expr_raw), meta_cols)
expr_mat <- as.matrix(expr_raw[, sample_cols]); storage.mode(expr_mat) <- "numeric"
rownames(expr_mat) <- expr_raw$Hugo_Symbol
pam_present <- intersect(pam50, rownames(expr_mat))
cat("PAM50 genes found:", length(pam_present), "of 50\n")
expr_pam <- t(expr_mat[pam_present, , drop = FALSE])

os_months <- as.numeric(clin$OS_MONTHS)
deceased  <- grepl("DECEASED", clin$OS_STATUS, ignore.case = TRUE)
horizon <- 60
label <- rep(NA_integer_, nrow(clin))
label[deceased  & os_months <= horizon] <- 1L
label[deceased  & os_months >  horizon] <- 0L
label[!deceased & os_months >= horizon] <- 0L

gi   <- match(clin$SAMPLE_ID, rownames(expr_pam))
keep <- !is.na(label) & !is.na(os_months) & !is.na(gi)
clin_k <- clin[keep, ]; y <- label[keep]; expr_k <- expr_pam[gi[keep], , drop = FALSE]
cat("Shared cohort:", nrow(clin_k), "patients | deaths <5y:", sum(y), "\n")

clin_features <- c(
  "AGE_AT_DIAGNOSIS", "TUMOR_SIZE", "LYMPH_NODES_EXAMINED_POSITIVE",
  "NPI", "GRADE", "TUMOR_STAGE",
  "CELLULARITY", "ER_STATUS", "PR_STATUS", "HER2_STATUS", "ER_IHC", "HER2_SNP6",
  "HORMONE_THERAPY", "RADIO_THERAPY", "CHEMOTHERAPY",
  "INFERRED_MENOPAUSAL_STATE", "LATERALITY", "BREAST_SURGERY",
  "TYPE_OF_BREAST_SURGERY", "HISTOLOGICAL_SUBTYPE",
  "ONCOTREE_CODE", "CANCER_TYPE_DETAILED"
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
X_both <- cbind(X_clin, expr_k)
npi    <- suppressWarnings(as.numeric(clin_k$NPI))

strat_split <- function(g, p = 0.8) {
  idx <- integer(0)
  for (cls in unique(g)) {
    rows <- which(g == cls); idx <- c(idx, sample(rows, floor(p * length(rows))))
  }
  sort(idx)
}

# ---------------------------------------------------------------------------
# 6. PICK nrounds ONCE PER FEATURE SET (kept fixed across repeats for speed;
#    applied equally to both models, so the comparison stays fair)
# ---------------------------------------------------------------------------
base_params <- function(spw)
  list(objective = "binary:logistic", eval_metric = "auc",
       max_depth = 4, eta = 0.05, subsample = 0.8, colsample_bytree = 0.8,
       min_child_weight = 5, scale_pos_weight = spw)

pick_nrounds <- function(X, yv) {
  spw <- sum(yv == 0) / sum(yv == 1)
  cv  <- xgb.cv(base_params(spw), xgb.DMatrix(X, label = yv),
                nrounds = 1000, nfold = 5, early_stopping_rounds = 30, verbose = 0)
  which.max(cv$evaluation_log$test_auc_mean)
}
set.seed(7)
nb_clin <- pick_nrounds(X_clin, y)
nb_both <- pick_nrounds(X_both, y)
cat(sprintf("nrounds: clinical = %d | clinical+PAM50 = %d\n", nb_clin, nb_both))

train_fixed <- function(Xtr, ytr, nb) {
  spw <- sum(ytr == 0) / sum(ytr == 1)
  xgb.train(base_params(spw), xgb.DMatrix(Xtr, label = ytr), nrounds = nb, verbose = 0)
}

# ---------------------------------------------------------------------------
# 7. REPEATED STRATIFIED HOLD-OUT
# ---------------------------------------------------------------------------
M <- matrix(NA_real_, N_REPEATS, 3, dimnames = list(NULL, c("npi","clin","both")))

cat("Repeats: ")
for (i in seq_len(N_REPEATS)) {
  set.seed(1000 + i)                 # different split each repeat, reproducible
  tri <- strat_split(y, 0.8)

  m_clin <- train_fixed(X_clin[tri, ], y[tri], nb_clin)
  m_both <- train_fixed(X_both[tri, ], y[tri], nb_both)
  gl_npi <- glm(yy ~ NPI, data = data.frame(yy = y[tri], NPI = npi[tri]),
                family = binomial)

  pc <- predict(m_clin, X_clin[-tri, ])
  pb <- predict(m_both, X_both[-tri, ])
  pn <- predict(gl_npi, newdata = data.frame(NPI = npi[-tri]), type = "response")

  yte <- y[-tri]; cm <- !is.na(pn); yy <- yte[cm]   # same test patients for all 3
  M[i, "npi"]  <- as.numeric(auc(roc(yy, pn[cm], quiet = TRUE)))
  M[i, "clin"] <- as.numeric(auc(roc(yy, pc[cm], quiet = TRUE)))
  M[i, "both"] <- as.numeric(auc(roc(yy, pb[cm], quiet = TRUE)))
  cat(i, "")
}
cat("\n")

# ---------------------------------------------------------------------------
# 8. SUMMARISE
# ---------------------------------------------------------------------------
summ <- function(x) sprintf("%.3f  (2.5-97.5%%: %.3f to %.3f)",
                            mean(x), quantile(x, .025), quantile(x, .975))
cat("\nMean test AUROC over", N_REPEATS, "repeats\n")
cat("  NPI only        :", summ(M[, "npi"]),  "\n")
cat("  Clinical        :", summ(M[, "clin"]), "\n")
cat("  Clinical + PAM50:", summ(M[, "both"]), "\n")

d_bc <- M[, "both"] - M[, "clin"]     # the key contrast
d_cn <- M[, "clin"] - M[, "npi"]
d_bn <- M[, "both"] - M[, "npi"]

diff_line <- function(name, d)
  cat(sprintf("  %-28s mean %+.3f  (2.5-97.5%%: %+.3f to %+.3f)  win-rate %.0f%%\n",
              name, mean(d), quantile(d, .025), quantile(d, .975), 100 * mean(d > 0)))
cat("\nPaired ΔAUROC (positive = first model better)\n")
diff_line("PAM50+clinical vs clinical", d_bc)
diff_line("clinical vs NPI",            d_cn)
diff_line("PAM50+clinical vs NPI",      d_bn)

# ---------------------------------------------------------------------------
# 9a. AUROC DISTRIBUTIONS PER MODEL (boxplot)
# ---------------------------------------------------------------------------
long <- data.frame(
  AUROC = c(M[, "both"], M[, "clin"], M[, "npi"]),
  model = factor(rep(c("Clinical + PAM50", "Clinical", "NPI only"), each = N_REPEATS),
                 levels = c("NPI only", "Clinical", "Clinical + PAM50")))
pal <- c("NPI only" = "#E69F00", "Clinical" = "#0072B2", "Clinical + PAM50" = "#009E73")

pbox <- ggplot(long, aes(model, AUROC, fill = model)) +
  geom_boxplot(width = 0.55, outlier.shape = NA, alpha = 0.85, colour = "grey30") +
  geom_jitter(width = 0.12, size = 1.3, alpha = 0.5, colour = "grey20") +
  scale_fill_manual(values = pal, guide = "none") +
  labs(title = "Test AUROC across repeated stratified splits",
       subtitle = sprintf("METABRIC · shared expression cohort · %d repeats", N_REPEATS),
       x = NULL, y = "Test AUROC") +
  theme_minimal(base_size = 13) +
  theme(plot.title = element_text(face = "bold", size = 13.5),
        plot.subtitle = element_text(colour = "grey35", margin = margin(b = 8)),
        panel.grid.major.x = element_blank(),
        panel.grid.minor = element_blank())
ggsave(file.path(out_dir, "auroc_boxplot_repcv.png"), pbox,
       width = 7, height = 5.5, dpi = 300, bg = "white")
ggsave(file.path(out_dir, "auroc_boxplot_repcv.pdf"), pbox,
       width = 7, height = 5.5, bg = "white")

# ---------------------------------------------------------------------------
# 9b. DISTRIBUTION OF THE KEY ΔAUROC (PAM50 gain over clinical)
# ---------------------------------------------------------------------------
dd <- data.frame(d = d_bc)
pdiff <- ggplot(dd, aes(d)) +
  geom_histogram(bins = 15, fill = "#009E73", colour = "white", alpha = 0.9) +
  geom_vline(xintercept = 0, linetype = "dashed", colour = "grey40") +
  geom_vline(xintercept = mean(d_bc), colour = "#009E73", linewidth = 1) +
  labs(title = "Does PAM50 add discrimination over clinical?",
       subtitle = sprintf("ΔAUROC per split · mean %+.3f · positive in %.0f%% of splits",
                           mean(d_bc), 100 * mean(d_bc > 0)),
       x = "ΔAUROC  (clinical + PAM50)  −  (clinical)", y = "Number of splits") +
  theme_minimal(base_size = 13) +
  theme(plot.title = element_text(face = "bold", size = 13.5),
        plot.subtitle = element_text(colour = "grey35", margin = margin(b = 8)),
        panel.grid.minor = element_blank())
ggsave(file.path(out_dir, "delta_auroc_pam50_repcv.png"), pdiff,
       width = 7, height = 5, dpi = 300, bg = "white")

# ---------------------------------------------------------------------------
# 10. SHAP on a full-cohort clinical+PAM50 fit (interpretation, not evaluation)
# ---------------------------------------------------------------------------
m_full <- train_fixed(X_both, y, nb_both)
sv <- shapviz(m_full, X_pred = X_both, X = as.data.frame(X_both))
ggsave(file.path(out_dir, "shap_both_beeswarm.png"),
       sv_importance(sv, kind = "beeswarm", max_display = 20) +
         ggtitle("SHAP — clinical + PAM50 (full cohort, top 20)"),
       width = 9, height = 7, dpi = 150, bg = "white")

message("Done. Summary printed above; figures in ", out_dir)
