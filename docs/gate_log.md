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
