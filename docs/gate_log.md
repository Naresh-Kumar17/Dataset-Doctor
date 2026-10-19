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

