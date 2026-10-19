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

