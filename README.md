# Can Machine Learning Predict Breast-Cancer Survival — and Do Genes Help?

A personal machine-learning project on a real, public breast-cancer dataset. It has two layers: a plain-language walkthrough of the question and findings (below), and a full **technical section** further down covering the pipeline, modelling and statistics. Nothing here is a clinical tool — it's a portfolio project about doing applied ML carefully and reporting it honestly.

---

# Part 1 — The project in plain language

## The story in one paragraph

When someone is diagnosed with breast cancer, doctors estimate how the disease is likely to progress. For decades they've used simple scores based on tumour size, how aggressive the cells look, and whether the cancer has spread to lymph nodes. More recently it's become possible to also read the tumour's **gene activity**. The natural question: **does adding genetic information actually help predict who will survive — or do the standard clinical measurements already tell you most of what you need to know?** That's what this project tests.

## The data

The project uses **METABRIC**, a well-known public breast-cancer dataset (~2,500 patients from the UK and Canada). For each patient it includes standard **clinical information** (age, tumour size, grade, stage, hormone-receptor status, treatments), **gene-activity ("expression") data** (a molecular readout of how active thousands of genes are), and the **outcome** (whether and when the patient died). It's free to download from [cBioPortal](https://www.cbioportal.org/study/summary?id=brca_metabric); only the raw sequencing files are restricted, and no patient data is stored in this repo.

**Who's in the analysis:** the **1,916 patients** who had both gene-activity data and a clear 5-year outcome (1,489 alive at five years, 427 died).

## The aim

Predict whether a patient would **survive five years**, and compare three approaches:

1. **NPI only** — the classic clinical score (Nottingham Prognostic Index) as a single number. The simple baseline.
2. **Clinical** — a machine-learning model using the *full* set of standard clinical measurements.
3. **Clinical + genes** — the same model, also given a well-known 50-gene signature (PAM50).

The key question is whether #3 beats #2 — whether genes add anything on top of clinical data.

## Two ideas you'll need

**AUROC** — a score from 0.5 to 1.0. Pick a patient who died and one who survived at random; a good model gives the one who died the higher risk score. AUROC is how often it gets that ranking right. **0.5** = guessing, **1.0** = perfect, **~0.75–0.80** (where this lands) = useful but imperfect, roughly "right 4 times out of 5."

**The p-value** — when one model scores a bit higher, is that real or just luck from which patients landed in the test set? A **small p-value (< 0.05)** means the difference is probably real; a large one means it could be chance.

## What I found

**First look (one split):**

| Contestant | AUROC |
|-----------|-------|
| NPI only | 0.754 |
| Clinical | 0.778 |
| Clinical + genes | 0.798 |

![The three models compared](clinical_pam50/roc_clinical_vs_pam50.png)

Each step adds a little, but the gaps are small. This chart shows what the model leaned on — the usual clinical factors (age, tumour size, nodes, receptor status) dominate; genes play a minor role.

![What the model paid attention to](clinical_pam50/shap_both_beeswarm.png)

**Second look (repeating 25 times, to avoid being fooled by one lucky split):**

- **Full clinical data beat the old NPI score in all 25 of 25 runs** — small but consistent and real.
- **Adding genes beat clinical-alone in only 68% of runs**, by an amount indistinguishable from noise.

![Scores across 25 repeats](clinical_pam50_cv/auroc_boxplot_repcv.png)
![How much the genes helped (centred near zero)](clinical_pam50_cv/delta_auroc_pam50_repcv.png)

## The bottom line

A fuller set of clinical measurements reliably beats the classic single-number score — while the PAM50 gene signature did **not** meaningfully improve predictions once clinical data was included. That matches a lot of published research: much of what genes "say" about prognosis is already reflected in what doctors measure. Slightly anticlimactic — but reporting it honestly, rather than over-hyping a tiny difference, is the point.

---

# Part 2 — Technical write-up

## Problem framing

Right-censored survival is reframed as **binary classification at a 5-year horizon**: died within 60 months → `1`; known alive at ≥60 months (or died later) → `0`. Patients **censored before 60 months** (alive but with shorter follow-up) have unknown 5-year status and are **excluded rather than imputed** — labelling them as survivors would inject systematic label noise. This is the most consequential modelling decision in the pipeline. (A time-to-event model such as Cox or XGBoost-AFT would use the discarded follow-up time; that's noted as a limitation.)

## Data pipeline

- **Sources:** cBioPortal `brca_metabric` — `data_clinical_patient.txt`, `data_clinical_sample.txt`, `data_mrna_illumina_microarray.txt`.
- **Join:** patient and sample clinical tables merged on `PATIENT_ID`; the expression matrix (genes × samples) is transposed and linked to patients via `SAMPLE_ID`.
- **Gotcha handled:** expression columns are sample IDs like `MB-0000`. Reading with `check.names = FALSE` prevents R from silently rewriting them to `MB.0000`, which would break the join and yield an empty cohort.
- **Shared cohort:** all models run on the **same 1,916 patients** — those with a determinable 5-year label *and* available expression — so feature-set comparisons aren't confounded by different populations.

## Feature engineering

- **Clinical (57 columns one-hot encoded):** age, tumour size, positive nodes, grade, stage, NPI, cellularity, ER/PR/HER2 status, menopausal state, treatments, surgery type, histology.
- **Genomic:** the PAM50 signature; **49/50 genes matched** the array annotation (`ORC6L` is an alias of `ORC6`, absent under that symbol — not missing data). Combined matrix = **106 columns**.
- **Custom one-hot encoder** preserves `NA` as `NA` (not a spurious category), letting XGBoost use its native missing-value handling rather than forcing imputation.
- **Leakage control:** identifiers and outcome-derived fields (`OS_MONTHS`, `OS_STATUS`, `VITAL_STATUS`) are never features.
- **Confound control:** the expression-derived classifiers `CLAUDIN_SUBTYPE`, `THREEGENE` and `INTCLUST` are **deliberately excluded** from the "clinical" set — otherwise "clinical" would already contain genomic information, invalidating the clinical-vs-genomic comparison.

## Models

| Model | Algorithm | Feature set |
|-------|-----------|-------------|
| NPI only | Logistic regression | NPI (1 predictor) |
| Clinical | XGBoost | clinicopathologic (57) |
| Clinical + PAM50 | XGBoost | clinical + genes (106) |

**XGBoost configuration:** `objective = binary:logistic`, `eval_metric = auc`, `max_depth = 4`, `eta = 0.05`, `subsample = 0.8`, `colsample_bytree = 0.8`, `min_child_weight = 5`. Shallow trees + low learning rate + row/column subsampling are chosen to limit overfitting on a modest-sized cohort.

**Class imbalance** (~3.5:1 survivors:deaths) handled with `scale_pos_weight` set to the training-set negative/positive ratio, so the minority "death" class isn't ignored.

**Model selection:** number of boosting rounds chosen by **5-fold cross-validation with early stopping** (up to 1000 rounds, stop after 30 without AUC improvement) — no manually fixed `nrounds`.

## Evaluation

- **Metric:** AUROC (threshold-independent — appropriate under class imbalance), reported with **95% confidence intervals** (`pROC::ci.auc`, DeLong variance).
- **Significance:** **DeLong's paired test** for correlated ROC curves, always computed on the **same held-out patients** for both models so the comparison is truly paired.
- **Held-out discipline:** stratified 80/20 split; all metrics on patients unseen during training.

Single-split results:

| Comparison | ΔAUROC | DeLong *p* |
|------------|--------|-----------|
| Clinical vs NPI | +0.024 | 0.300 |
| Clinical + PAM50 vs Clinical *(key test)* | +0.021 | 0.195 |
| Clinical + PAM50 vs NPI | +0.045 | **0.040** |

## Interpretability

**SHAP** (`shapviz`) on the clinical+PAM50 model quantifies each feature's contribution to individual predictions. Standard clinical variables dominate the ranking; PAM50 genes contribute marginally — mechanistic support for the quantitative finding that expression is largely redundant with clinical data here.

## Robustness: repeated cross-validation

A single 80/20 split is high-variance, so the experiment is repeated over **25 stratified splits** (reproducible per-repeat seeds). For speed, `nrounds` is tuned once per feature set (clinical = 59, +PAM50 = 56) and held fixed across repeats — applied equally to both XGBoost models, so the *comparison* stays fair.

| Model | Mean AUROC (2.5–97.5%) |
|-------|------------------------|
| NPI only | 0.723 (0.656–0.774) |
| Clinical | 0.759 (0.708–0.814) |
| Clinical + PAM50 | 0.765 (0.717–0.807) |

Paired ΔAUROC across splits:

| Comparison | mean ΔAUROC | 2.5–97.5% | win-rate |
|------------|-------------|-----------|----------|
| **Clinical vs NPI** | **+0.036** | +0.007 to +0.066 | **100%** |
| PAM50 + clinical vs clinical | +0.006 | −0.034 to +0.033 | 68% |
| PAM50 + clinical vs NPI | +0.042 | +0.006 to +0.085 | 100% |

**Statistical reasoning that matters here:** the repeats share overlapping training data and are **not independent**, so I deliberately do **not** compute a p-value from the 25 differences (a paired t-test would badly overstate significance). Repeated CV is used to assess the **stability** of the estimate; the DeLong test remains the formal per-patient significance test. The two are complementary — and here they resolve the single split's ambiguity: clinical-over-NPI (only p = 0.30 on one split) turns out robust across all 25 repeats, while PAM50-over-clinical stays within noise. The repeated-CV means also sit below the single split, showing that split (seed 42) was mildly optimistic.

## Reproducibility

Fixed global seed plus per-repeat seeds; results regenerate deterministically. Written in R with `xgboost`, `shapviz`, `pROC`, `ggplot2`, `data.table`. Data is not committed (see [`data/README.md`](data/README.md) for how to obtain it).

```
metabric-survival-ml/
├── R/
│   ├── 01_clinical_vs_npi_vs_pam50.R   # main comparison + ROC + SHAP
│   └── 02_repeated_cv.R                # 25-repeat robustness analysis
├── clinical_pam50/                     # figures from the main run
├── clinical_pam50_cv/                  # figures from the repeated run
└── data/                               # data not included
```

## Skills demonstrated

- End-to-end applied-ML pipeline on real, messy biomedical data (joins, missingness, imbalance).
- Careful problem framing: censored survival → binary classification with correct handling of unknown outcomes.
- Leakage and confounding awareness (excluding outcome-derived and expression-derived features).
- Gradient boosting with sensible regularisation and imbalance handling; principled model selection via cross-validation.
- Correct, threshold-independent evaluation with confidence intervals and an appropriate paired significance test.
- Model interpretability with SHAP.
- A robustness design (repeated CV) and — importantly — the statistical judgement to know what it can and cannot claim.
- Publication-quality visualisation and honest reporting of a partly null result.

---

## Limitations

- ~1,900 patients limits power to detect small AUROC gaps; "genes didn't help" means "no clear help detected," not "never helps."
- Tests one fixed 50-gene signature, not expression in general.
- Single cohort — no external validation (e.g. TCGA-BRCA).
- 5-year dichotomisation discards time-to-event information.

## Sources & credit

METABRIC: Curtis et al. (*Nature*, 2012); Pereira et al. (*Nat Commun*, 2016), via cBioPortal (Cerami et al. 2012; Gao et al. 2013). PAM50: Parker et al. (*JCO*, 2009). NPI: Galea et al. (1992). Tools: XGBoost (Chen & Guestrin, 2016); SHAP (Lundberg & Lee, 2017).

## License & author

Code under the MIT License (see [`LICENSE`](LICENSE)). METABRIC data belongs to its original authors and is not redistributed.

**‹YOUR NAME›** — ‹contact / GitHub / LinkedIn›

*A personal project for learning and curiosity. Not medical advice and not a clinical decision tool.*
