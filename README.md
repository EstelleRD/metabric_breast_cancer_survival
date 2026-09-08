# Predicting 5-Year Breast-Cancer Survival: Does Gene Expression Add Value Over Clinical Data?

A reproducible machine-learning study on the **METABRIC** breast-cancer cohort, asking a question that matters in practice: once you already have a patient's standard clinical variables, **does adding PAM50 gene-expression data actually improve prognosis prediction?** The project is built around gradient-boosted trees (XGBoost), rigorous validation, SHAP interpretability, and honest reporting — including a robustness check that changes the conclusion.

> **Headline finding.** On a held-out test set, a clinicopathologic XGBoost model reached **AUROC 0.778** versus **0.754** for the Nottingham Prognostic Index (NPI) alone; adding the 50-gene PAM50 panel raised it to **0.798**. Repeated cross-validation (25 splits) then showed that the **clinical model robustly beats NPI** (mean ΔAUROC **+0.036**, better in **100%** of splits) — but **PAM50 adds no robust improvement over clinical data** (mean ΔAUROC **+0.006**, 95% interval spanning zero, better in only **68%** of splits). In this cohort, a rich clinical panel improves on the classic index, while gene expression is largely redundant with variables clinicians already record.

---

## Why this project

- **Clinically motivated question.** "Incremental value of genomics over clinical data" is a recurring question in precision oncology, and one where the literature is biased toward positive results. A clean, honest answer either way is genuinely useful.
- **As an ML portfolio piece**, it demonstrates: a leakage-aware pipeline, a proper baseline, correct handling of censored survival data, threshold-independent evaluation with confidence intervals, model interpretability with SHAP, a robustness analysis with repeated CV, and careful statistical reasoning about what the single-split and repeated-CV results each can and cannot say.

---

## The question, in three steps

1. **Baseline vs model.** Can XGBoost on the full clinicopathologic feature set beat the single established index (NPI)?
2. **Does genomics help?** Does adding PAM50 gene expression to the clinical model improve discrimination? *(the core test — same model class, nested feature sets)*
3. **Is any gain robust?** Does the effect survive repeated resampling, or is it an artifact of one lucky train/test split?

---

## Data

- **Cohort:** METABRIC (Molecular Taxonomy of Breast Cancer International Consortium) — 2,509 clinically annotated tumours in this release, with gene expression, CNA and mutation data.
- **Source:** [cBioPortal — `brca_metabric`](https://www.cbioportal.org/study/summary?id=brca_metabric).
- **Files used:** `data_clinical_patient.txt`, `data_clinical_sample.txt`, `data_mrna_illumina_microarray.txt` (Illumina HT-12 microarray expression).
- **Analysis cohort:** patients with a determinable 5-year survival status **and** available expression data → **1,916 patients** (1,489 survived, 427 died within 5 years). All models are trained and evaluated on this **same** cohort so comparisons are fair.

**Licensing / ethics.** The processed METABRIC data on cBioPortal (clinical tables and expression matrices) is publicly available; only the raw sequencing reads are under controlled access. No patient-level data is redistributed in this repository — see [`data/README.md`](data/README.md) for how to download it yourself.

---

## Methods

**Outcome.** Binary 5-year overall survival. Deaths within 60 months → `1`; patients known to be alive at ≥60 months (or who died later) → `0`. Patients **censored before 5 years** (alive but with <60 months follow-up) have unknown status and are **dropped rather than guessed** — a common and consequential source of label noise if handled naively.

**Features.**
- *Clinicopathologic:* age at diagnosis, tumour size, positive lymph nodes, grade, stage, NPI, cellularity, ER/PR/HER2 status, menopausal state, treatments (hormone/radio/chemo), surgery type, histology. → 57 columns after one-hot encoding.
- *Genomic:* the PAM50 genes (Parker et al. 2009) from the expression matrix. **49 of 50** matched the array annotation (the 50th, `ORC6L`, is an alias of `ORC6` and simply absent under that symbol — not missing data). → 106 columns with expression added.
- **Deliberately excluded from "clinical":** `CLAUDIN_SUBTYPE`, `THREEGENE`, `INTCLUST` — these are themselves **derived from expression/CNA**, so including them would smuggle genomic information into the "clinical" model and confound the very comparison we are making.

**Models.**

| # | Name | Model | Features |
|---|------|-------|----------|
| 1 | NPI only | Logistic regression | NPI (1 variable) |
| 2 | Clinical | XGBoost | clinicopathologic (57) |
| 3 | Clinical + PAM50 | XGBoost | clinicopathologic + 49 genes (106) |

**Validation & metrics.** Stratified 80/20 split; XGBoost rounds chosen by 5-fold CV with early stopping; class imbalance handled via `scale_pos_weight`. Reported: **AUROC with 95% CI** (threshold-independent), **DeLong's test** for paired AUROC differences on identical patients, and **SHAP** for interpretability. A **repeated-CV** analysis (25 stratified splits) then tests stability.

**Guardrails against common mistakes.** Identifiers and outcome-derived fields (`OS_MONTHS`, `OS_STATUS`, `VITAL_STATUS`) are never used as features (leakage); all three models are compared on the **same test patients**; sample IDs are read with `check.names = FALSE` so `MB-0000`-style keys aren't silently mangled.

---

## Results

### 1–2. Single held-out test set (n = 1,916 shared cohort)

| Model | Features | AUROC (95% CI) |
|-------|----------|----------------|
| NPI only (logistic) | 1 | **0.754** (0.694–0.813) |
| XGBoost | clinicopathologic | **0.778** (0.722–0.833) |
| XGBoost | clinical + PAM50 | **0.798** (0.745–0.851) |

DeLong tests (paired, same patients):

| Comparison | ΔAUROC | *p* |
|------------|--------|-----|
| Clinical vs NPI | +0.024 | 0.300 |
| **Clinical + PAM50 vs Clinical** *(key test)* | +0.021 | 0.195 |
| Clinical + PAM50 vs NPI | +0.045 | **0.040** |

![ROC — three models](clinical_pam50/roc_clinical_vs_pam50.png)

On this single split, only the *cumulative* clinical+PAM50 model significantly beats the bare NPI index (p = 0.040); neither incremental step (clinical over NPI, or PAM50 over clinical) is individually significant, and the ROC curves for the two XGBoost models are nearly superimposed. A single split is noisy, though — see the repeated-CV analysis below, which resolves the ambiguity.

**SHAP (clinical + PAM50 model).** Standard clinical drivers (age, tumour size/nodes, receptor status) dominate; PAM50 genes contribute modestly — consistent with expression being largely correlated with, rather than additive to, clinical variables.

![SHAP summary](clinical_pam50/shap_both_beeswarm.png)

### 3. Repeated cross-validation (25 stratified splits) — the robustness check

| Model | Mean AUROC (2.5–97.5%) |
|-------|------------------------|
| NPI only | 0.723 (0.656–0.774) |
| Clinical | 0.759 (0.708–0.814) |
| Clinical + PAM50 | 0.765 (0.717–0.807) |

Paired ΔAUROC across splits (positive = first model better):

| Comparison | mean ΔAUROC | 2.5–97.5% | win-rate |
|------------|-------------|-----------|----------|
| **Clinical vs NPI** | **+0.036** | +0.007 to +0.066 | **100%** |
| PAM50 + clinical vs clinical | +0.006 | −0.034 to +0.033 | 68% |
| PAM50 + clinical vs NPI | +0.042 | +0.006 to +0.085 | 100% |

![AUROC across repeats](clinical_pam50_cv/auroc_boxplot_repcv.png)
![ΔAUROC distribution](clinical_pam50_cv/delta_auroc_pam50_repcv.png)

The repeated analysis sharpens the conclusion:

- **Clinical robustly beats NPI.** The mean gain is +0.036, the interval excludes zero, and the clinical model wins in **every** split. The single-split DeLong (p = 0.30) simply lacked power on one partition.
- **PAM50 does not robustly help.** Its edge over the clinical model averages just +0.006, the interval straddles zero, and it wins only 68% of the time — indistinguishable from noise.

> **Statistical note.** The repeats share overlapping training data and are **not independent**, so no *p*-value is derived from them — repeated CV characterises the **stability** of the estimate, while the single-split DeLong test provides the formal paired significance test. The two are complementary: here, repeated CV reveals that the clinical-over-NPI gain (ambiguous on one split) is in fact robust, while the PAM50-over-clinical gain is not. (Mean AUROCs are also slightly lower than the single split, indicating that particular seed was a mildly favourable partition — another reason to trust the averaged estimate.)

---

## Interpretation

Two clear messages emerge. First, **a full clinicopathologic model reliably outperforms the single NPI index** — a modest but consistent gain (≈ +0.03–0.04 AUROC) that XGBoost extracts from the richer feature set. Second, **PAM50 gene expression provides no robust incremental discrimination beyond those clinical variables** for 5-year survival in METABRIC. This is consistent with the well-documented observation that much of the prognostic signal in gene-expression signatures is correlated with grade, size, nodal status and receptor status — information clinicians already have.

## Limitations

- **Power.** Even at n ≈ 1,900, resolving AUROC differences of ~0.02 is hard; "not detected" for PAM50 is not "proven absent."
- **PAM50 is a fixed 49/50-gene panel**, so this tests that specific signature — not "expression in general." A top-variance or full-transcriptome model could behave differently.
- **Single cohort.** No external validation (e.g. TCGA-BRCA); generalisation is untested.
- **Dichotomising survival at 5 years** discards time-to-event information a Cox/AFT model would use.

---

## Reproducing this analysis

```r
# 1. Install dependencies (once)
install.packages(c("xgboost", "shapviz", "pROC", "ggplot2", "data.table"))

# 2. Download METABRIC into data/  (see data/README.md)

# 3. Run
source("R/01_clinical_vs_npi_vs_pam50.R")   # analyses 1 & 2 + ROC + SHAP
source("R/02_repeated_cv.R")                # analysis 3 (robustness)
```

Figures and metrics are written to `outputs/`. A fixed `set.seed(42)` and per-repeat seeds make results reproducible. For fully pinned package versions, initialise [`renv`](https://rstudio.github.io/renv/) (`renv::init()`) and commit the resulting `renv.lock`.

## Repository structure

```
metabric-survival-ml/
├── README.md
├── LICENSE
├── .gitignore
├── data/
│   └── README.md          # how to obtain METABRIC (data is NOT committed)
├── R/
│   ├── 01_clinical_vs_npi_vs_pam50.R
│   └── 02_repeated_cv.R
└── outputs/               # generated figures (committed so the README renders)
    ├── roc_clinical_vs_pam50.png
    ├── shap_both_beeswarm.png
    ├── auroc_boxplot_repcv.png
    └── delta_auroc_pam50_repcv.png
```
*(Map your existing script filenames to `R/01_…` and `R/02_…`. Copy the four figures into `outputs/` — the repeated-CV PNGs are in your `…/clinical_pam50_cv/` folder.)*

## Requirements

- R ≥ 4.1
- `xgboost`, `shapviz`, `pROC`, `ggplot2`, `data.table`

## References

1. Curtis C, et al. *The genomic and transcriptomic architecture of 2,000 breast tumours reveals novel subgroups.* Nature. 2012;486:346–352.
2. Pereira B, et al. *The somatic mutation profiles of 2,433 breast cancers refine their genomic and transcriptomic landscapes.* Nat Commun. 2016;7:11479.
3. Parker JS, et al. *Supervised risk predictor of breast cancer based on intrinsic subtypes (PAM50).* J Clin Oncol. 2009;27:1160–1167.
4. Galea MH, et al. *The Nottingham Prognostic Index in primary breast cancer.* Breast Cancer Res Treat. 1992;22:207–219.
5. Cerami E, et al. *The cBioPortal for Cancer Genomics.* Cancer Discov. 2012;2:401–404.
6. Gao J, et al. *Integrative analysis of complex cancer genomics and clinical profiles using the cBioPortal.* Sci Signal. 2013;6:pl1.
7. Chen T, Guestrin C. *XGBoost: A Scalable Tree Boosting System.* KDD 2016.
8. Lundberg SM, Lee S-I. *A Unified Approach to Interpreting Model Predictions (SHAP).* NeurIPS 2017.

## License

Code released under the MIT License (see [`LICENSE`](LICENSE)). METABRIC data is subject to its own terms via cBioPortal and is not redistributed here.

## Author

‹YOUR NAME› — ‹contact / GitHub / LinkedIn›

*This project is for research and educational purposes and is not a medical device or clinical decision tool.*
