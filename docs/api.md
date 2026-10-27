# Dataset Doctor: API Reference Specification

## 1. Overview

Dataset Doctor provides a programmatic interface across three complementary surfaces:
1. **Python Core API (`doctor.*`)**: For headless scripts, Jupyter notebooks, and automated machine learning pipelines.
2. **FastAPI HTTP REST API (`/api/*`, `/decide/*`)**: For web frontends, microservices, and external client integrations.
3. **Command-Line Interface (CLI)**: For terminal-based profiling, diagnostic auditing, benchmark validation, and pipeline export.

---

## 2. Python Core API Reference

### 2.1. Dataset Loading and Profiling
```python
from doctor.loader import load_csv, DatasetLoadConfig
from doctor.profile import profile_dataset

# Load CSV safely with automatic encoding detection and size checks
config = DatasetLoadConfig(max_file_size_mb=50)
df = load_csv("data/raw/customer_churn.csv", config=config)

# Generate lightweight, honest profile
profile = profile_dataset(df, target_column="churn")
print(f"Shape: {profile.n_rows} rows x {profile.n_cols} columns")
print(f"Task type: {profile.task_type}, Minority share: {profile.minority_share:.1%}")
```

### 2.2. Split and Evaluation Engine
```python
from doctor.split import split_dataset, SplitConfig
from doctor.evaluate import evaluate_model, Preprocessor

# Deterministic stratified split (test set strictly locked)
split_cfg = SplitConfig(test_size=0.2, random_state=42, stratify=True)
train_df, test_df = split_dataset(df, target="churn", config=split_cfg)

# Evaluate fixed baseline with 5-fold CV noise estimation
eval_result = evaluate_model(train_df, test_df, target="churn")
print("Train CV F1:", eval_result.train_cv_mean["f1"], "±", eval_result.train_cv_std["f1"])
print("Test Set F1:", eval_result.test_scores["f1"])
```

### 2.3. Running Diagnostic Detectors
```python
from doctor.base import DetectionContext
from doctor.issues.nulls.detect import NullDetector
from doctor.issues.duplicates.detect import DuplicateDetector
from doctor.issues.types.detect import TypeDetector
from doctor.issues.outliers.detect import OutlierDetector

ctx = DetectionContext(target_column="churn", task_type="classification")

# Run pure read-only analyzers
null_issues = NullDetector().detect(train_df, ctx)
dup_issues = DuplicateDetector().detect(train_df, ctx)
type_issues = TypeDetector().detect(train_df, ctx)
outlier_issues = OutlierDetector().detect(train_df, ctx)
```

### 2.4. Remediation Advisor and Pipeline Application
```python
from doctor.base import Decision
from doctor.issues.outliers.advise import OutlierAdvisor
from doctor.pipeline import RemediationPipeline

# Obtain recommendations with CV evidence
advisor = OutlierAdvisor()
recommendation = advisor.advise(outlier_issues[0], train_df, target="churn", ctx=ctx)
print("Recommended:", recommendation.recommended, "| Reason:", recommendation.reason)

# Build pipeline and record approved decision
pipeline = RemediationPipeline()
decision = Decision(
    issue_id=outlier_issues[0].module_id,
    column=outlier_issues[0].column,
    option_id=recommendation.recommended,
    params={},
    decided_by="user",
)
pipeline.record_decision(decision, train_df=train_df)

# Transform train and test (test rows are strictly preserved)
clean_train = pipeline.transform(train_df, is_test=False)
clean_test = pipeline.transform(test_df, is_test=True)

# Export standalone reproduction script
code_str = pipeline.export_reproduce_code(train_path="train.csv", test_path="test.csv")
with open("reproduce.py", "w") as f:
    f.write(code_str)
```

---

## 3. FastAPI REST Endpoints

### 3.1. Upload and Session Management
| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `POST` | `/upload` | Uploads multipart CSV file with MIME and size verification. Initializes session partition. |
| `GET` | `/` | Renders dataset upload workbench or redirects to active session dashboard. |
| `GET` | `/bench` | Displays interactive Test Bench running live synthetic planted benchmark gates. |

### 3.2. Issue Diagnostics & Workbench
| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/module/{module_id}` | Renders dedicated Neo-Brutalist diagnostic workbench for an issue module (`nulls`, `duplicates`, `types`, `outliers`, `imbalance`, `leakage`). |
| `POST` | `/api/issues/diagnose` | Returns JSON diagnostic issues array across all registered modules for the loaded session. |

### 3.3. Remediation Decision API
| Method | Endpoint | Request Body | Description |
| :--- | :--- | :--- | :--- |
| `POST` | `/decide/nulls/apply` | `{"column": "...", "method": "median", "params": {}}` | Fits null transformer on train; records decision. |
| `POST` | `/decide/duplicates/apply` | `{"remediation_type": "keep_first"}` | Fits deduplication transformer on train; records decision. |
| `POST` | `/decide/types/apply` | `{"column": "...", "conversion_type": "convert_to_numeric"}` | Applies type conversion; updates schema. |
| `POST` | `/decide/outliers/apply` | `{"column": "...", "strategy": "clip_iqr", "params": {}}` | Fits bounds on train; clips train & test sets. |
| `POST` | `/decide/imbalance/apply` | `{"strategy": "class_weight", "params": {}}` | Configures train-only resampling / weighting. |

---

## 4. Command-Line Interface (CLI)

Dataset Doctor CLI wraps the core engine for terminal workflows:

```bash
# 1. Profile dataset
python -m doctor profile data/raw/accidents.csv --target severity

# 2. Audit quality across all diagnostic modules
python -m doctor audit data/raw/accidents.csv --target severity --format json

# 3. Run full automated gate benchmark
python -m doctor.bench --all

# 4. Run specific module benchmark gate
python -m doctor.bench outliers
python -m doctor.bench types
python -m doctor.bench nulls
```
