# Dataset Doctor: Developer Notes & Competitive Analysis

## Phase 0: Competitive Landscape Analysis (What existing tools do NOT do)

1. **ydata-profiling (pandas-profiling)**: Generates static exploratory reports and summary statistics, but does not benchmark repair strategies against downstream ML metrics and cannot apply fixes.
2. **Great Expectations**: Enforces rule-based schema assertions in data pipelines, but only flags binary pass/fail without evaluating how data issues impact model performance or recommending fixes.
3. **Deepchecks**: Detects data distribution drift, leakage checks, and train-test mismatch, but does not provide interactive decision cards that empirically test and rank competing remediation methods.
4. **General Limitation across all three**: None of them fit data transformations exclusively on the training set and verify that the test set remains untouched with zero row drops and identical class ratios.
5. **The Dataset Doctor Difference**: Dataset Doctor bridges detection and remediation: it detects problems without altering data, tests candidate fixes strictly on training data via cross-validation, presents transparent decision cards to the user, and applies accepted fixes through a leak-proof pipeline.

## Phase 1: GUI Shell, Loader, and Profiler Notes
- **Loader safety**: Standard pandas `read_csv` silently converts string `"NA"` to `NaN`. Using `keep_default_na=False` with chardet encoding fallback ensures clean, uncorrupted ingestion of international datasets.
- **Neo-brutalist shell**: The bold high-contrast UI with 3px black borders, hard offset shadows, and explicit status chips provides an intuitive workbench where all 6 diagnostic modules are discoverable from day one.
- **Zero-loss profiling**: Profile calculations compute memory footprint, semantic type inferences, and missingness counts without altering underlying dataframe arrays or dropping rows.

## Phase 5: Duplicates Detection, Advice, and Fixer Notes
- **Pre-split deduplication rule**: Exact duplicate rows are dropped before dataset partitioning so identical records cannot span train and test sets and cause synthetic test leakage.
- **Conflicting label guard**: Rows that possess identical features but contradictory targets introduce conflicting loss gradients. The advisor recommends dropping these ambiguous instances to ensure consistent learning signals.
- **Pre-balanced file detection**: Files with a 50/50 target ratio containing high minority duplicate counts signal pre-split oversampling. The detector flags this as possible test leakage and instructs splitting the raw data first.

## Phase 6: Data Types Detection, Advice, and Fixer Notes
- **Text number stripping & leading zero code protection**: Strips currencies, commas, percentages, and units while strictly guarding categorical codes with leading zeros (e.g. '001', '007') so codes are not corrupted into numbers.
- **Ambiguous date format user disambiguation**: Dates where both day and month components are 12 or less (e.g. '03/04/2021') cannot be guessed reliably without domain knowledge. The advisor surfaces explicit Day-First (DD/MM) vs Month-First (MM/DD) choices and asks the user rather than guessing silently.
- **Zero-variance and identifier pruning**: Constant columns (zero mutual information) and high-cardinality IDs (leakage and memorization vectors) are flagged for removal across both train and test partitions without altering test row counts.

## Phase 7: Outliers Detection Notes
- **Domain-impossible value enforcement**: Features with hard physical or logical boundaries (e.g. age < 0 or > 120, percentages > 100, negative salaries/counts) are flagged as certain errors (`severity="high"`). They are never treated as legitimate statistical extremes.
- **Unit mix-up identification**: Isolated order-of-magnitude jumps (100x or 1000x the column median, such as 170,000 cm instead of 170 cm) point to data collection unit confusion (e.g. millimeters vs centimeters, grams vs kilograms).
- **Robust Median Absolute Deviation (MAD) for skewed features**: Standard $1.5\times\text{IQR}$ formulas over-flag natural long tails on skewed continuous features (log-normal, Pareto distributions). Using robust MAD and 1%/99% percentile thresholds ensures clean skewed features stay under 5% flagged without discarding valid tail data.
- **Categorical and binary immunity**: Outlier scanning strictly skips non-numeric, binary, and categorical columns to prevent false alarms.

## Phase 8: Outlier Advisor & Fixer Notes
- **Principled remediation hierarchy**: Impossible values are shortlisted for `set_to_null` so they can be imputed cleanly through the Null Advisor. Legitimate extremes are shortlisted for `leave`, `clip_percentile`, `clip_iqr`, or `log_transform`.
- **Rare-class extreme protection**: When the prediction target is rare (< 20% minority share, e.g. severe accidents or fraud), extreme feature values frequently contain the critical signal. Dropping these rows would eliminate minority target instances; the advisor explicitly protects them and recommends `leave`.
- **Simplicity tie-breaker on CV noise**: Tree-based models (RandomForest, XGBoost, LightGBM) are invariant to monotonic scaling and extreme tail values. When 5-fold CV evaluation shows that score deltas between options are smaller than CV standard deviation, the advisor flags a "close call" and recommends `leave` as the simpler, non-destructive tie-breaker.
- **Strict test partition integrity**: All clipping bounds and quantile limits are fitted strictly on `train_df`. When applied to test data, the same train-derived bounds are enforced, and test row counts are strictly preserved (0 test rows dropped).

## Expanded Platform Architecture Notes (Phases 9–21 & Stages A–G)

### 1. Evolution from Single-Model Profiler to Full ML Reliability Platform
- **The Core Problem**: Profile tools report symptoms; they don't test remedies. Auto-cleaning tools mutate data silently without understanding model trade-offs.
- **The Solution**: Dataset Doctor expands from a 11-phase MVP into a 21-phase end-to-end reliability platform structured into 7 distinct stages:
  - **Stage A (Phases 0–13)**: Core Data Quality MVP (Nulls, Duplicates, Types, Outliers, Imbalance, Leakage, and Pipeline Ordering).
  - **Stage B (Phase 14)**: Unified Data-to-Model Quality Assessment (Risk matrix connecting data defects to downstream model failures).
  - **Stage C (Phase 15)**: Multi-Model Experimentation (Comparing RF, GBDT, DecisionTree, and Linear baselines on locked splits).
  - **Stage D (Phases 16–17)**: Error Diagnostics & Attribution (Confusion matrices, error segments, and permutation importance).
  - **Stage E (Phase 18)**: Headless Developer Tooling (Python Core SDK and terminal CLI).
  - **Stage F (Phase 19)**: Comprehensive Experiment Manifest & Bit-for-Bit Reproducibility.
  - **Stage G (Phases 20–21)**: MCP Server, Agent Collaboration & Production Deployment.

### 2. Imbalance Remediation Hierarchy (Phases 9 & 10)
- **Accuracy Illusion**: In 90/10 or 99/1 imbalance, majority prediction scores 90-99% accuracy while catching 0% of minority cases.
- **Train-Only Resampling Invariant**: Resampling (RandomOverSampler, SMOTE, RandomUnderSampler) must NEVER be applied to the test partition. Test sets retain natural real-world class distributions.
- **SMOTE Categorical Guard**: SMOTE interpolates vectors in continuous space. Applying SMOTE to unencoded categoricals generates invalid fractional codes. The fixer refuses unencoded categoricals and prompts one-hot/ordinal encoding first.

### 3. Model Comparison & Diagnostic Safety (Phases 15–17)
- **Zero Test Snooping**: Model selection and hyperparameter exploration must use training-fold cross-validation. The final test set is evaluated only once per experiment to avoid turning the test set into an implicit training set.
- **Non-Causal Explanation Disclosure**: Feature importance and permutation attribution indicate associative predictive power, never real-world causal mechanisms.

