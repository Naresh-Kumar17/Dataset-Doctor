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


