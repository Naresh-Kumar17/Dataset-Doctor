# Gate Log

## 2026-10-19  Phase 0: Setup and evidence   PASS
- pytest -q runs: 1 passed (tests/test_hello.py verified FastAPI hello endpoint)
- FastAPI root endpoint verified: `{"status": "Dataset Doctor is running"}`
- Baseline evidence established on road accident dataset (2,000 rows, 85/15 ratio):
  - Flawed (oversample before split): Accuracy 0.9574, Severe Recall 0.9941, Severe F1 0.9589
  - Honest (split first, no fix): Accuracy 0.8700, Severe Recall 0.2167, Severe F1 0.3333
  - Honest (split first, class weights): Accuracy 0.8375, Severe Recall 0.4000, Severe F1 0.4248
  - Honest (split first, oversample train only): Accuracy 0.8500, Severe Recall 0.3167, Severe F1 0.3878
- Test set preserved clean class distribution: 85% class 0 / 15% class 1 (340 vs 60 rows)
- Documented 3-line explanation of test leakage from pre-split balancing
- Competitive analysis recorded in docs/notes.md

## 2026-10-19  Phase 1: GUI shell + loader + profile   PASS
- Loader verified:
  - Accident dataset matches pandas row count (2,000), column count (9), and dtypes exactly.
  - Latin-1 encoded file decoded and loaded accurately.
  - 60 MB file size limit enforced with clear error message (FileTooLargeError).
  - Empty files (0-byte / whitespace) and header-only files handled with friendly errors (EmptyFileError, NoDataRowsError) with zero crash.
  - keep_default_na=False verified: valid tokens like 'NA' preserved without silent NaN conversion.
- Profile verified:
  - Computes row count, column count, memory usage, missing column counts, dtypes, unique counts, sample values, and 10-row preview.
  - Correctly suggested target column ('severe_accident') and task type ('classification').
- GUI Shell and Routes verified:
  - Neo-brutalist styling, responsive layout (desktop & 375px mobile), sidebar registry navigation.
  - Routes /, /upload, /profile, /target, /module/{id}, and /bench verified via automated tests (17 passed).
  - Test Bench page (/bench) opens and lists all modules with status chips.

## 2026-10-21  Phase 2: Null detection   PASS
- Automated benchmark runner verified via `python -m doctor.bench nulls` and `python -m doctor.bench --all`:
  - 100% of planted missing cells found:
    - 50 true NaNs in 'age' found with exact count.
    - 60 hidden tokens ('N/A') in 'phone' found with exact count.
    - 40 sentinel values ('-999') in 'score' found and classified strictly as 'possible'.
    - 700 NaNs in 'notes' found with 'high' severity (>60% missing threshold).
  - 0 false alarms on clean_control.csv:
    - 0 certain issues on clean control.
    - Tricky country code 'NA' (Namibia) flagged at most as 'possible'.
  - Exact row-loss number confirmed:
    - 796 rows lost (79.6%) exactly matches dropping all true missing values and confirmed sentinels.
    - Row breakdown: 17 rows with 1 null, 66 rows with 2 nulls, 713 rows with 3+ nulls.
  - Missingness pattern statistical tests verified:
    - Planted target-related missingness ('income') detected and reported (Chi-square p < 0.05).
    - Planted random missingness ('noise_col') correctly identified as not related (p >= 0.05).
  - High-performance benchmark verified:
    - 100,000 rows x 5 columns scanned in under 0.35s (threshold < 5.0s).
- Full Phase 2 test suite: 10 passed in `tests/issues/test_nulls_detect.py`.
- Neo-brutalist GUI diagnostic card & table added to `/module/nulls` with summary strip, row-loss impact banner, and evidence badges.

## 2026-10-21  Phase 3: Split + evaluation engine   PASS
- Automated benchmark runner verified via `python -m doctor.bench split_eval` and `python -m doctor.bench --all`:
  - 7 of 7 Gate 3 checks PASSED:
    - [PASS] Stratified split preserves class ratio within 1% on train and test (exact 85/15 ratio preserved).
    - [PASS] Deterministic split verified (same seed yields identical partition; different seed yields different partition).
    - [PASS] Zero partition overlap verified (strict set intersection check).
    - [PASS] Synthetic leakage detected: pre-split balancing inflates recall (0.9941 vs honest 0.3167).
    - [PASS] Always-predict-majority baseline verified (~85% accuracy, 0 minority recall, accuracy not reliable banner).
    - [PASS] Delta smaller than CV std labelled 'within noise'.
    - [PASS] Dataset < 500 rows generates noisy comparison warning.
- Full Phase 3 test suite: 13 passed in `tests/test_split.py` and `tests/test_evaluate.py`.
- Regression check: all 46 tests passed across the entire project.
- Neo-brutalist GUI dashboard integrated at `/module/split_eval` with split overview stat grid, imbalance alert banner, baseline CV table, and synthetic leakage demonstration card.

## 2026-10-23  Phase 4: Null advisor + fixer, interactive   PASS
- Automated benchmark runner verified via `python -m doctor.bench nulls_advisor` and `python -m doctor.bench --all`:
  - 14 of 14 Gate 4 checks PASSED:
    - Gate 4A (Advisor Ground Truth):
      - [PASS] Scenario 1 (Numeric, 3% random missing, symmetric): recommends 'median' or 'mean'.
      - [PASS] Scenario 2 (Numeric, 10% missing, heavily skewed): recommends 'median'.
      - [PASS] Scenario 3 (Categorical, 5% missing): recommends 'mode' or 'missing_label'.
      - [PASS] Scenario 4 (Column 75% missing): recommends 'add_indicator' or 'drop_column'.
      - [PASS] Scenario 5 (Missingness linked to target): recommends 'add_indicator', does NOT drop rows.
      - [PASS] 5 of 5 planted advisor recommendations matched expected ground truth (threshold >= 4/5).
      - [PASS] Every recommendation provides plain-English rationale and measured 5-fold CV scores.
      - [PASS] Close call flag and simplicity tie-breaker trigger when score delta is within CV std.
    - Gate 4B (Fixer Verification):
      - [PASS] After fix: 0 nulls remain in train and test partitions.
      - [PASS] Test row count is strictly unchanged (0 test rows dropped; train-fitted fallback applied).
      - [PASS] Test values changed only in cells that were originally null.
      - [PASS] Learned values derived strictly from train partition (train median 20.0 vs full data 30.0; test imputed with 20.0, zero leakage).
      - [PASS] Decision log persists sequential choices and exported `reproduce.py` builds identical result.
      - [PASS] User override successfully changes applied remediation method.
- Full Phase 4 test suite: 19 passed in `tests/issues/test_nulls_advise.py`, `tests/issues/test_nulls_fix.py`, and `tests/test_decide_routes.py`.
- Regression check: all 65 tests passed across the entire project.
- Neo-brutalist interactive decision cards integrated into `/module/nulls` with Before/After live preview, [APPLY RECOMMENDED], [CHOOSE ANOTHER], [SKIP], batch action [APPLY RECOMMENDED TO ALL], and standalone `reproduce.py` export endpoint.

## 2026-10-26  Phase 5: Duplicates (Detect + Advise + Fix)   PASS
- Automated benchmark runner verified via `python -m doctor.bench duplicates` and `python -m doctor.bench --all`:
  - 9 of 9 Gate 5 checks PASSED:
    - [PASS] 50 of 50 planted exact duplicate rows detected and flagged with certainty='certain'.
    - [PASS] 10 of 10 planted conflicting duplicate pairs (identical features, differing target) detected with severity='high'.
    - [PASS] 20 of 20 planted near-duplicate rows (casing, whitespace, precision fuzz) detected and flagged as certainty='possible'.
    - [PASS] 0 false alarms on clean_control.csv (0 exact or conflicting duplicates flagged).
    - [PASS] Fixer verification: After `keep_first` fix, exactly 0 duplicate rows remain and row count drops by exactly 50 (530 == 530).
    - [PASS] Test set safety: test rows are never dropped during deduplication (`is_test=True` preserves row count).
    - [PASS] Conflicting pairs fixer: `drop_all_conflicting` successfully removes ambiguous contradictory label pairs.
    - [PASS] Pre-balanced warning fires on 50/50 oversampled file (synthetic pre-balancing leakage guard).
    - [PASS] Pre-balanced warning stays quiet on clean control.
- Full project test suite: all 80 tests passed across the entire project (`pytest -v`).
- Neo-brutalist interactive duplicate management integrated into `/module/duplicates` with diagnostic summary cards, conflicting target alert banner, pre-balanced warning card, and quick remediation actions ([DROP EXACT DUPLICATES], [DROP CONFLICTING ROWS], [EXPORT REPRODUCE.PY]).

## 2026-11-02  Phase 6: Data Types (Detect + Advise + Fix)   PASS
- Automated benchmark runner verified via `python -m doctor.bench types` and `python -m doctor.bench --all`:
  - 12 of 12 Gate 6 checks PASSED:
    - [PASS] All planted text numbers found (`price_usd`, `discount_pct`, `weight_kg`).
    - [PASS] Ambiguous dates flagged and user is asked, never guessed (certainty='possible').
    - [PASS] Categorical column with codes like '001' is NOT turned into a number by mistake.
    - [PASS] Unambiguous ISO date timestamp found with certainty='certain'.
    - [PASS] Yes/No boolean representation detected with certainty='certain'.
    - [PASS] Constant zero-variance column detected with severity='high'.
    - [PASS] 0 false alarms on clean_control.csv (no false text numbers, ambiguous dates, or constants).
    - [PASS] Advisor presents explicit Day-First and Month-First options for ambiguous dates.
    - [PASS] After convert_to_numeric, numeric column values match planted truth exactly.
    - [PASS] After encode_boolean, values are standard binary 0 and 1.
    - [PASS] After drop_column, constant column is removed without altering row count.
    - [PASS] Decision log persists and exported reproduce.py code rebuilds identical clean dataset.
- Full project test suite: 15 unit tests in `tests/issues/test_types.py` and 2 integration tests in `tests/test_web_routes.py` passing cleanly.
- Neo-brutalist interactive types workbench integrated into `/module/types` with status chips, ambiguous date alert banner with Day-First and Month-First buttons, type anomaly table, and quick action conversion buttons.

## 2026-11-07  Phase 7: Outlier Detection   PASS
- Automated benchmark runner verified via `python -m doctor.bench outliers` and `python -m doctor.bench --all`:
  - 8 of 8 Gate 7 checks PASSED:
    - [PASS] All 10 planted impossible values found with severity=high (age < 0 or > 120).
    - [PASS] Planted unit mix-up found and flagged with severity=medium (170,000 cm magnitude jump).
    - [PASS] All 20 extreme-but-real values detected as medium/low tail outliers.
    - [PASS] Heavily skewed clean column flags <= 5% as outliers (measured: 1.3%).
    - [PASS] Clean normal Gaussian column stays under 1% flagged (measured: 0.0%).
    - [PASS] 0 outlier flags on binary, categorical, ID, or target columns.
    - [PASS] 0 false alarms on clean_control.csv.
    - [PASS] Plain-English explanation explicitly warns about skewness and IQR distortion.
- Full Phase 7 unit tests: 12 tests passed in `tests/issues/test_outliers.py`.
- Neo-brutalist interactive outliers detection workbench integrated into `/module/outliers` with stat cards for Impossible Values, Unit Errors, Tail Extremes, and Total Anomalies.

## 2026-11-12  Phase 8: Outlier Advisor & Fixer (Interactive)   PASS
- Automated benchmark runner verified via `python -m doctor.bench outliers_advisor` and `python -m doctor.bench --all`:
  - 7 of 7 Gate 8 checks PASSED:
    - [PASS] Impossible values are recommended as set_to_null (non-negotiable domain rule).
    - [PASS] Rare-class protection: leave is recommended when dropping extremes removes minority cases.
    - [PASS] Simplicity tie-breaker: recommends leave when CV score differences are within noise.
    - [PASS] Clipping bounds learned strictly from train and applied identically to test.
    - [PASS] Gate 8: Test set row count is strictly preserved (0 test rows dropped).
    - [PASS] Decision log persists and exported reproduce.py code rebuilds identical clean dataset.
    - [PASS] User override successfully alters applied remediation to log_transform.
- Full project test suite: 12 unit tests in `tests/issues/test_outliers.py`, integration tests in `tests/test_web_routes.py` and `tests/test_decide_routes.py`.
- RemediationPipeline and `reproduce.py` code generator updated with `set_to_null`, `clip_iqr`, `clip_percentile`, `log_transform`, and `drop_rows` transformations.
- Complete regression suite: 100% PASS across all 7 benchmark modules.

---

## 2026-11-10  Phase 9: Class Imbalance Detection   PASS
- Automated benchmark runner verified via `python -m doctor.bench imbalance` and `python -m doctor.bench --all`:
  - 8 of 8 Gate 9 checks PASSED:
    - [PASS] Planted 90/10 target detected with high severity (ratio 9.0, minority 10%).
    - [PASS] Planted 99/1 target detected with high severity (ratio 99.0, minority 1%).
    - [PASS] Planted multiclass target (70/25/5) detected with high severity.
    - [PASS] Balanced control (50/50) produces 0 issues (stays completely quiet).
    - [PASS] 0 certain false alarms on clean_control.csv.
    - [PASS] Surfaces accuracy paradox warning in evidence.
    - [PASS] Surfaces real-world prevalence prompt for domain confirmation.
    - [PASS] Detection executes rapidly (0.005s < 2.0s).
- Full project test suite: 10 unit tests in `tests/issues/test_imbalance.py` passing cleanly.
- Neo-brutalist interactive imbalance workbench integrated into `/module/imbalance` with stat cards for Severe Imbalance, Moderate Imbalance, Target Analyzed, and Train-Only Resampling Guarantee.

## 2026-11-13  Phase 10: Class Imbalance Advisor & Fixer (Interactive)   PASS
- Automated benchmark runner verified via `python -m doctor.bench imbalance_advisor` and `python -m doctor.bench --all`:
  - 9 of 9 Gate 10 checks PASSED:
    - [PASS] Advisor evaluates 5-fold CV on train partition and recommends evidence-backed strategy (`class_weight` or `random_oversample`).
    - [PASS] Advisor provides trade-off visibility across recall, precision, and F1.
    - [PASS] Class weighting preserves 100% of training data rows.
    - [PASS] Random oversampling balances minority and majority classes on train partition.
    - [PASS] Gate 10: Test partition row count is strictly preserved (0 test rows resampled/dropped).
    - [PASS] Gate 10: Test partition class ratio is 100% unchanged.
    - [PASS] SMOTE safety guard: refuses unencoded non-numeric features with clear error.
    - [PASS] RemediationPipeline integration: decision persists and reproduce code exports cleanly.
    - [PASS] User override: successfully applies alternative remediation strategy (`leave`).
- Full project test suite: 130 tests passing across all suites (`pytest`).
- Web API endpoints: `POST /decide/imbalance/apply` with session tracking.
- Interactive decision controls: 1-click remediation actions for Class Weights, Oversample Train Only, SMOTE (Numeric Only), Undersample, and Leave.
- RemediationPipeline and `reproduce.py` code generator updated with `class_weight`, `random_oversample`, `random_undersample`, and `smote` transformations.
- Complete regression suite: 100% PASS across all 9 benchmark modules (Gates 2 through 10).

---

## 2026-11-05  Phase 11: Data Leakage Detection   PASS
- Automated benchmark runner verified via `python -m doctor.bench leakage` and `python -m doctor.bench --all`:
  - 9 of 9 Gate 11 checks PASSED:
    - [PASS] Planted direct target copy (`target_copy`) detected with high severity (r = 1.0).
    - [PASS] Planted linearly scaled target (`target_scaled`) detected with high severity (r = 1.0).
    - [PASS] Planted inverted target (`target_inverted`) detected with high severity (r = -1.0).
    - [PASS] Planted row-index / ID sequence leak (`id_leak`) detected with high severity (r = 0.985).
    - [PASS] Planted post-outcome payout proxy (`payout_amount`) detected with high severity (r = 0.9999).
    - [PASS] Legitimate strong predictor (`legit_strong`, r ~0.65) NOT flagged as high severity leak (immunity guard).
    - [PASS] Temporal column (`timestamp_col`) flagged as time-split risk (recommends time-based split).
    - [PASS] 0 false alarms on `clean_control.csv`.
    - [PASS] Leakage detection executes rapidly (0.031s < 2.0s).
- Full project test suite: 10 unit tests in `tests/issues/test_leakage.py` passing cleanly.
- Neo-brutalist interactive data leakage workbench integrated into `/module/leakage` with summary cards, evidence badges, and leak risk descriptions.

## 2026-11-06  Phase 12: Data Leakage Advisor & Prevention   PASS
- Automated benchmark runner verified via `python -m doctor.bench leakage_advisor` and `python -m doctor.bench --all`:
  - 7 of 7 Gate 12 checks PASSED:
    - [PASS] Advisor recommends 'drop_column' for all high-severity leaking features (`target_copy`, `target_scaled`, `target_inverted`, `id_leak`, `payout_amount`).
    - [PASS] Advisor recommends 'time_split' for time-order risk column (`timestamp_col`).
    - [PASS] Fixer successfully removes leaking column from feature space.
    - [PASS] Gate 12 Invariant: 100% row preservation on train and test partitions (0 rows dropped).
    - [PASS] Gate 12 Invariant: Test set class distribution is 100% preserved.
    - [PASS] RemediationPipeline integration: all leaks removed and reproduce code exported cleanly.
    - [PASS] User override ('keep_with_acknowledgement') successfully preserves column.
- Web API endpoints: `POST /decide/leakage/apply` with session decision tracking.
- Interactive decision controls: 1-click remediation actions for [DROP LEAKING COLUMN] and [ACKNOWLEDGE & KEEP].
- Complete regression suite: 100% PASS across all 11 benchmark modules (Gates 2 through 12).

---

## 2026-11-06  Phase 13: 10-Step Deterministic Pipeline & Causal Ordering   PASS
- Automated benchmark runner verified via `python -m doctor.bench pipeline` and `python -m doctor.bench --all`:
  - 8 of 8 Gate 13 checks PASSED:
    - [PASS] Canonical execution order strictly defined across all 6 diagnostic domains (`types` -> `nulls` -> `duplicates` -> `outliers` -> `imbalance` -> `leakage`).
    - [PASS] Pipeline detects causal ordering violation when downstream step precedes upstream.
    - [PASS] Pipeline successfully reorders arbitrary steps into canonical causal sequence.
    - [PASS] End-to-end pipeline cleans multi-issue dataset across types, nulls, duplicates, outliers, and leakage.
    - [PASS] Gate 13 Invariant: Test partition row count and class ratio are 100% preserved.
    - [PASS] Exports verified, standalone reproduce.py script with learned parameters.
    - [PASS] Side-by-side Before/After comparison report accurately tracks all metric deltas.
    - [PASS] Pipeline verification executes rapidly (< 4.0s).
- Full regression suite: 100% PASS across all 12 benchmark modules (Gates 2 through 13).

---

## 2026-11-07  Phase 14: Unified Data-to-Model Quality Assessment   PASS
- Automated benchmark runner verified via `python -m doctor.bench quality_assessment` and `python -m doctor.bench --all`:
  - 8 of 8 Gate 14 checks PASSED:
    - [PASS] Clean control dataset achieves 100/100 quality score and TRAINING_READY readiness.
    - [PASS] Cross-module risk matrix covers all 6 diagnostic domains.
    - [PASS] Unresolved direct target leakage triggers BLOCKED_CRITICAL readiness.
    - [PASS] Training blocker check strictly blocks model training without override.
    - [PASS] User override explicitly permits proceeding with logged acknowledgement.
    - [PASS] Quality score accurately reflects penalty deductions for unhandled anomalies.
    - [PASS] Applying remediation decisions successfully clears blockers and elevates quality score.
    - [PASS] Quality assessment executes rapidly (< 4.0s).
- Full regression suite: 100% PASS across all 13 benchmark modules (Gates 2 through 14).

---

## 2026-11-08  Phase 15: Multi-Model Experimentation & Comparison   PASS
- Automated benchmark runner verified via `python -m doctor.bench multi_model` and `python -m doctor.bench --all`:
  - 7 of 7 Gate 15 checks PASSED:
    - [PASS] Trains and evaluates 4 distinct model architectures against locked test partition (Logistic Regression, Decision Tree, Random Forest, Gradient Boosting).
    - [PASS] Computes full multi-metric matrix (Accuracy, Balanced Accuracy, Precision, Recall, F1, PR-AUC).
    - [PASS] Surfaces Accuracy Illusion Warning when high accuracy masks low minority recall.
    - [PASS] Identifies top-performing model across F1, PR-AUC, and composite metrics.
    - [PASS] Evaluates train vs test performance delta to flag overfitting generalization gaps.
    - [PASS] Gate 15 Invariant: Test partition row count and class distribution are 100% untouched.
    - [PASS] Multi-model experimentation completes efficiently (< 4.0s).
- Full regression suite: 100% PASS across all 14 benchmark modules (Gates 2 through 15).

---

## 2026-11-09  Phase 16: Model Error Analysis & Diagnostics   PASS
- Automated benchmark runner verified via `python -m doctor.bench error_analysis` and `python -m doctor.bench --all`:
  - 7 of 7 Gate 16 checks PASSED:
    - [PASS] Computes exact confusion matrix (TP, FP, TN, FN) matching test row count.
    - [PASS] Calculates normalized False Positive Rate (FPR) and False Negative Rate (FNR).
    - [PASS] Isolates top high-confidence False Positive prediction errors with context.
    - [PASS] Isolates top False Negative prediction errors with predicted probabilities.
    - [PASS] Slices error rates across subgroup categories to identify slice vulnerabilities.
    - [PASS] Quantifies confidence separation gap between correct predictions and errors.
    - [PASS] Error analysis completes rapidly (< 2.0s).
- Complete regression suite: 100% PASS across all 16 benchmark modules (Gates 2 through 16).

---

## 2026-11-10  Phase 17: Model Explainability & Feature Attribution   PASS
- Automated benchmark runner verified via `python -m doctor.bench explainability` and `python -m doctor.bench --all`:
  - 7 of 7 Gate 17 checks PASSED:
    - [PASS] Attribution computed on validation fold only with zero test partition snooping.
    - [PASS] Planted strong signal features rank top with high validation degradation.
    - [PASS] Detects high-cardinality spurious feature and flags impurity split inflation.
    - [PASS] Feature Pruning Advisor recommends removing pure noise and spurious features.
    - [PASS] Feature Pruning Advisor protects strong predictive signals from removal.
    - [PASS] Attribution profile includes prominent non-causal disclosures and disclaimers.
    - [PASS] Explainability attribution completes rapidly (1.171s < 4.0s).
- Full regression suite: 100% PASS across all 17 benchmark modules (Gates 2 through 17).
- Full test suite: 162 unit & integration tests passing (`pytest`).

---

## 2026-11-18  Phase 18: Python Core SDK & Terminal CLI   PASS
- Automated benchmark runner verified via `python -m doctor.bench sdk_cli` and `python -m doctor.bench --all`:
  - 7 of 7 Gate 18 checks PASSED:
    - [PASS] DatasetDoctor SDK loads CSV and profiles schema cleanly with zero web dependencies.
    - [PASS] SDK audit aggregates 6-domain risk matrix and scores clean dataset 100/100.
    - [PASS] SDK split locks test partition and preserves clean row partition invariants.
    - [PASS] CLI profile command executes cleanly and formats output in JSON / tabular format.
    - [PASS] CLI audit exports structured JSON quality report to disk cleanly.
    - [PASS] CLI check-leaks enforces strict exit code 1 on data leaks and exit code 0 on clean data.
    - [PASS] SDK & CLI operations complete rapidly (0.729s < 4.0s).
- Unified Headless Python SDK in `doctor/api.py` with programmatic methods: `from_csv()`, `from_dataframe()`, `profile()`, `audit()`, `split()`, `evaluate()`, `check_leakage()`, `explain()`, and `auto_remediate()`.
- Production-grade Terminal CLI in `doctor/cli.py` & `doctor/__main__.py` with commands: `profile`, `audit`, `evaluate`, `check-leaks`, `bench`, and `version`.
- 12 unit & integration tests in `tests/test_sdk.py` and `tests/test_cli.py` passing cleanly.

---

## 2026-11-19  Phase 19: Full Experiment Manifest & Bit-for-Bit Reproducibility   PASS
- Automated benchmark runner verified via `python -m doctor.bench manifest` and `python -m doctor.bench --all`:
  - 8 of 8 Gate 19 checks PASSED:
    - [PASS] compute_dataframe_hash generates deterministic 64-character SHA-256 hex digest.
    - [PASS] Cryptographic hash provides bit-for-bit sensitivity against minute data modifications.
    - [PASS] ManifestGenerator compiles complete provenance including split and hash metrics.
    - [PASS] Manifest serializes to standard JSON conforming to cryptographic specification.
    - [PASS] ManifestGenerator.verify_reproducibility verifies bit-for-bit fidelity on identical data.
    - [PASS] verify_reproducibility reliably detects dataset tampering or floating-point drift.
    - [PASS] export_standalone_script outputs 100% self-contained reproduction script with embedded hashes.
    - [PASS] Manifest and reproducibility verification execute rapidly (0.384s < 4.0s).
- Full cryptographic data provenance engine in `doctor/manifest.py` tracking raw dataset, split indices, learned transformer parameters, decision audit trail, evaluation metrics, and system environment metadata.
- Standalone reproduction script generator in `doctor/export.py` exporting zero-dependency, bit-for-bit verifiable `reproduce.py`.
- 4 unit tests in `tests/test_manifest.py` verifying hash determinism, JSON serialization, and reproduction audit.

---

## 2026-11-20  Phase 20: Model Context Protocol (MCP) Server & Agent Safety Contract   PASS
- Automated benchmark runner verified via `python -m doctor.bench mcp` and `python -m doctor.bench --all`:
  - 9 of 9 Gate 20 checks PASSED:
    - [PASS] MCP Server executes JSON-RPC 2.0 protocol handshake (initialize, ping) cleanly.
    - [PASS] tools/list advertises all 6 specialized Dataset Doctor tools with valid schemas.
    - [PASS] doctor_load_dataset profiles dataset and locks test partition cryptographically.
    - [PASS] doctor_audit_quality evaluates cross-module risk matrix and returns health score.
    - [PASS] doctor_get_recommendations produces actionable advice with candidate alternatives.
    - [PASS] doctor_apply_remediation fits on train and updates active pipeline cleanly.
    - [PASS] Agent Safety Contract strictly intercepts and blocks test partition mutation attempts.
    - [PASS] doctor_evaluate_baseline and doctor_export_reproduce output verified artifacts.
    - [PASS] Full MCP workflow execution completes rapidly (3.908s < 10.0s).
- Standard JSON-RPC 2.0 stdio server in `doctor/mcp.py` exposing 6 standardized tools:
  1. `doctor_load_dataset`: dataset loading, schema inference, and train/test partition locking.
  2. `doctor_audit_quality`: multi-domain risk matrix audit and data health scoring.
  3. `doctor_get_recommendations`: cross-validated, evidence-backed remediation advice.
  4. `doctor_apply_remediation`: train-only fitting and pipeline step recording.
  5. `doctor_evaluate_baseline`: Random Forest baseline training and evaluation against locked test partition.
  6. `doctor_export_reproduce`: standalone reproduction script and manifest generation.
- Agent Safety Contract strictly enforced: any agent action attempting to drop test rows, resample test partitions, or mutate test partition indices is intercepted and rejected with an invariant error.
- Full test suite: 184 unit & integration tests passing (`pytest`).
- Benchmark test bench: 20 of 20 gates passing 100% (`python -m doctor.bench --all`).

---

## Future Gates Pipeline (Phases 17–21 Specifications)

### Gate 9: Class Imbalance Detection (Scheduled)
- Planted targets: 90/10, 99/1, multi-class with tiny minority (5%), and 50/50 balanced control.
- Passes when 90/10, 99/1, and multi-class are flagged as high severity.
- Stays strictly silent (0 flags) on balanced control.
- Surfaces accuracy paradox warning and real-world prevalence prompt.

### Gate 10: Class Imbalance Advisor & Remediation (Scheduled)
- Advisor recommends `class_weight` or `random_oversample` based on 5-fold CV.
- Strictly asserts test set row count and class ratio are unchanged by any fix.
- Zero resampled rows appear in the test set.
- SMOTE refuses unencoded categorical features with clear error/warning.

### Gate 11: Data Leakage Detection (Scheduled)
- Detects target copies, scalar multiples ($X \times 2$), ID-like sequence leaks, and post-event columns.
- Legitimate strong features (correlation $\approx 0.70$) are not flagged as high severity.
- 0 false alarms on `clean_control.csv`.
- Time-ordered sequences trigger time-split warning.

### Gate 12: Data Leakage Advisor & Prevention (Scheduled)
- Recommends `drop_column` with explicit evidence.
- Allows user acknowledgement override.
- Column drops persist to `reproduce.py` without test leakage.

### Gate 13: End-to-End Pipeline & Ordering Verification (Scheduled)
- Enforces canonical causal execution order across all diagnostic remediations.
- Inverting step order alters output on planted scenario, proving order enforcement.
- Side-by-side before/after comparison with honest noise flagging.

### Gate 14: Unified Data-to-Model Quality Assessment (Scheduled)
- Cross-module risk matrix aggregates findings across all 6 domains.
- Unresolved critical issues block model export without explicit confirmation.

### Gate 15: Multi-Model Experimentation & Comparison (Scheduled)
- Evaluates Random Forest, Logistic Regression, Decision Tree, and LightGBM against the same locked test split.
- Multi-metric comparison matrix with minority metrics and overfitting flags.

### Gate 16: Model Error Analysis & Diagnostics (Scheduled)
- Confusion matrix and residual distribution derived strictly from model predictions.
- Top prediction errors isolated with original feature contexts.

### Gate 17: Model Explainability (Scheduled)
- Permutation importance calculated on validation folds without test data snooping.
- Non-causal disclaimers displayed on all attribution views.

### Gate 18: Python Core SDK & CLI (Scheduled)
- All CLI commands (`doctor profile`, `audit`, `evaluate`, `bench`) execute cleanly and return consistent JSON/text.
- Headless Python SDK passes unit tests without web dependencies.

### Gate 19: Full Experiment Manifest & Reproducibility (Scheduled)
- `reproduce.py` runs standalone in clean Python environment.
- Reproduces identical output DataFrame and test metrics within float tolerance.

### Gate 20: Model Context Protocol (MCP) Server (Scheduled)
- MCP tools discoverable and callable via standard JSON-RPC protocol.
- Integration test validates full agentic workflow via MCP tools with test protection invariants.

### Gate 21: Full System Integration & Release (Scheduled)
- 100% pass on pytest regression suite across all 21 phases.
- Real-world benchmark validation on Accident, Weather, and Kaggle datasets.
- 50 MB dataset audit completes in under 2 minutes.

