# Dataset Doctor: Product Requirements Document (PRD)

Version: 0.2 | Status: Draft | Owner: You

## 1. One-line summary
Upload a CSV. The tool finds the data problems that quietly ruin ML models, **recommends a fix with reasons and test results, asks you to decide**, applies it safely, and shows the before/after proof.

## 2. Problem
- Most ML projects fail from bad data or bad evaluation, not bad models.
- Common silent mistakes:
  - Balancing classes before the train/test split (inflated score).
  - Judging an imbalanced problem by accuracy only.
  - A column that secretly leaks the answer (data leakage).
  - Duplicates that appear in both train and test.
  - Filling or deleting missing values with no evidence that the choice is good.
- Existing tools (profilers, validators) list problems. They do not prove whether a fix helps, and they do not help you choose between fixes.

## 3. Goal
- Primary: a correct, honest tool that measures the real effect of each data fix.
- Personal goal: learn correct evaluation, leakage detection, and deployment.
- Non-goal: winning a pitch. No deadline.

## 4. Target user
- Students and junior data scientists who train models on CSV files.
- They know basic pandas and scikit-learn, but do not know what they are getting wrong.

## 5. Core principles (non-negotiable)
- **Split first.** Test set is never touched, never resampled, never loses rows, keeps the real class ratio.
- **Fix training data only.** Anything learned from data (fill values, clip bounds, resampling) is fit on train and only applied to test.
- **Never judge by accuracy alone.** For imbalanced targets, report minority-class recall, precision, F1, PR-AUC.
- **Advise, then ask.** For every issue the tool recommends a method, gives a reason, shows measured scores for the alternatives, and the user decides. Nothing is changed silently.
- **Rules are guesses, measurements are evidence.** Show both. Say "close call" when scores are within noise. Say "not enough data" when the dataset is small.
- **Detect, advise and fix are separate.** Each lives in its own file so each can be tested alone.
- **Say when unsure.** Show confidence and limits, do not overclaim.

## 6. The Four Defining Capabilities

Dataset Doctor evolves beyond static profiling into a full data quality and machine learning reliability platform through four foundational capabilities:

### Capability 1: Data-to-Model Quality Assessment
Connects data quality anomalies directly to machine learning failure modes. The platform synthesizes findings into a unified Quality & Risk Matrix covering completeness, uniqueness, type validity, extreme values, class imbalance, and evaluation contamination. Every finding exposes evidence, severity, certainty, and specific model failure risks.

### Capability 2: Intelligent Repair and Verification Loop
Extends the advisor-fixer architecture into a closed-loop system:
`Detect -> Collect Evidence -> Shortlist Heuristics -> Measure 5-Fold CV on Train -> User Decision -> Leak-Free Transform -> Re-Evaluate Model`.
Transformations are learned strictly from training data; test set rows are never dropped and test distributions are never resampled.

### Capability 3: Leakage-Aware, Reproducible ML Training
Enforces strict evaluation discipline:
- `split.py` is the single authoritative source of truth.
- Partitions are locked before any preprocessing.
- Resampling (SMOTE, oversampling) and scaling are isolated strictly to training folds.
- Standardized baseline (RandomForest) and multi-model benchmarking (linear, tree, GBDT) on identical splits.
- Task-appropriate minority metrics (Recall, Precision, F1, PR-AUC).

### Capability 4: Reproducible Project Export
Every accepted decision is logged sequentially. The platform exports:
- Standalone, dependency-free `reproduce.py` script.
- Cleaned CSV with audit headers.
- Machine-readable `manifest.json` capturing dataset hashes, split configs, transformer parameters, and evaluation metrics.
- Production-ready serialized pipeline (`joblib`).

---

## 7. Features Matrix (21-Phase Platform Roadmap)

| ID | Feature | Subsystem / Detail | Phase | Stage |
|:---|:---|:---|:---:|:---:|
| **F0** | Setup & Planted Baseline | Planted data generator, truth files, clean control, accident baseline | 0 | A |
| **F1** | CSV Upload & Profiler | 50 MB limit, streaming chunker, encoding detection, shape & types | 1 | A |
| **F2** | Test Bench Engine | Live interactive test bench, planted vs found, gate runner | 1 | A |
| **F3** | Null Detection | NaNs, hidden tokens (`?`, `N/A`), sentinels (`-999`), target missingness | 2 | A |
| **F4** | Split & Evaluation Engine | Stratified split, locked test partition, 5-fold CV, noise guard | 3 | A |
| **F5** | Null Advisor & Fixer | 5-fold CV on train, simplicity tie-breaker, leak-free imputation | 4 | A |
| **F6** | Duplicate Detect & Fix | Exact duplicates, near duplicates, conflicting targets, pre-balanced alert | 5 | A |
| **F7** | Type Detect & Fix | Text numbers (`$1,200`), ambiguous dates, booleans, leading-zero codes | 6 | A |
| **F8** | Outlier Detection | Domain-impossible values, order-of-magnitude unit mix-ups, MAD skew guard | 7 | A |
| **F9** | Outlier Advisor & Fixer | Rare-class extreme protection, train bounds clipping, test preservation | 8 | A |
| **F10** | Imbalance Detection | Class shares, minority share, imbalance ratio, real-world prompt | 9 | A |
| **F11** | Imbalance Advisor & Fix | Train-only resampling (weights, oversample, SMOTE, undersample), test untouched | 10 | A |
| **F12** | Leakage Detection | Perfect single predictors, target copies, sequential ID leaks, time order | 11 | A |
| **F13** | Leakage Advisor & Fix | Evidence-backed column dropping, chronological time-ordered splits | 12 | A |
| **F14** | End-to-End Pipeline | Causal ordering verification, sequential fit/transform, full diff | 13 | A |
| **F15** | Unified Quality Matrix | Cross-module risk matrix, ML consequence mapping, unresolved issues tracker | 14 | B |
| **F16** | Multi-Model Experiments | Model comparison engine (RandomForest, Logistic, DecisionTree, LightGBM) | 15 | C |
| **F17** | Error Analysis & Diag | Confusion matrix, per-class PR curves, residual histograms, error segments | 16 | D |
| **F18** | Model Explainability | Impurity importance, permutation importance on train folds, non-causal warnings | 17 | D |
| **F19** | Python SDK & CLI | Documented `doctor.*` SDK, terminal CLI (`profile`, `audit`, `evaluate`, `bench`) | 18 | E |
| **F20** | Experiment Manifest | Machine-readable `manifest.json`, bit-for-bit reproducibility test harness | 19 | F |
| **F21** | MCP Server & Agent Tools | Model Context Protocol server (`doctor.mcp`), agent tooling & safety guards | 20 | G |
| **F22** | Production Release | 5 real-world benchmark datasets, performance audit, live cloud deployment | 21 | G |

---

## 8. User Flow

1. **Ingest & Profile**: Upload CSV. Preview shape, memory footprint, column schemas, and distributions.
2. **Lock Evaluation Protocol**: Pick target feature. Auto-detect task type (classification / regression). Stratified train/test split is immediately computed and locked.
3. **Audit Data Quality**: Review Unified Quality & Risk Matrix. Inspect individual diagnostic workbenches (`nulls`, `duplicates`, `types`, `outliers`, `imbalance`, `leakage`).
4. **Evaluate Decisions**: For each detected issue, inspect candidate options tested via 5-fold CV on training data. Review measured scores, standard deviations, and noise labels ("close call", "not enough data").
5. **Approve Remediations**: Accept, customize, or skip recommendations. All decisions are sequentially logged.
6. **Sequential Pipeline Execution**: Execute approved pipeline in strict causal order. Learned parameters are fit on train and transformed onto train and test sets without test leakage.
7. **Model Evaluation & Comparison**: Evaluate baseline model and compare candidate algorithms against the locked test set.
8. **Inspect Diagnostics**: Analyze confusion matrices, residual error distributions, and permutation feature attributions.
9. **Export Artifacts**: Download cleaned CSV, standalone `reproduce.py`, machine-readable `manifest.json`, and executive Markdown report.

---

## 9. Success Criteria & Non-Negotiables

- **Zero Test Set Corruption**: The test partition is never resampled, never loses rows, and never participates in parameter estimation.
- **Evidence Over Rules**: Every recommended remediation provides measured CV scores with standard deviations. Deltas within noise invoke simplicity tie-breakers.
- **100% Planted Verification**: Every diagnostic module achieves $100\%$ recall of planted defects and $0$ false alarms on `clean_control.csv`.
- **Honest Uncertainty**: Small datasets ($N < 500$) are explicitly flagged as noisy. Imbalanced targets ($< 20\%$ minority) never report accuracy as the primary metric.
- **Standalone Reproducibility**: Running exported `reproduce.py` on the raw CSV reproduces the identical cleaned dataset and model evaluation metrics.
- **Performance**: End-to-end quality audit on a 50 MB dataset completes in under 2 minutes on standard hardware.

