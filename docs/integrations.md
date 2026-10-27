# Dataset Doctor: Ecosystem Integrations Guide

## 1. Core Machine Learning & Data Stack

Dataset Doctor is built on the standard Python scientific and machine learning ecosystem:

| Library | Version Target | Role in Dataset Doctor |
| :--- | :--- | :--- |
| **pandas** | `>=2.2` | Core tabular representation, schema inspection, and vectorized transformations. |
| **numpy** | `>=1.26` | Mathematical operations, array indexing, and sentinel token handling. |
| **scikit-learn** | `>=1.4` | Stratified splitting, baseline models, cross-validation, and metrics (Precision, Recall, F1, PR-AUC). |
| **imbalanced-learn** | `>=0.12` | Sampling algorithms (RandomOverSampler, SMOTE, RandomUnderSampler) restricted strictly to training folds. |
| **scipy** | `>=1.12` | Statistical calculations including skewness, kurtosis, and Median Absolute Deviation (MAD). |
| **FastAPI & Uvicorn** | `>=0.110` / `>=0.29` | High-performance asynchronous web server and REST API endpoints. |
| **Jinja2** | `>=3.1` | Server-rendered HTML templates utilizing the Neo-Brutalist design system. |

---

## 2. Model Baseline & Evaluator Integration

### 2.1. Standardized Baseline
- **RandomForest Model**: 200 trees, `max_depth=None`, `random_state=42`. Serves as an un-tuned, non-linear baseline to measure data quality impacts without hyperparameter confounding.
- **Fast Mode**: For advisor cross-validation loops, tree count is adjusted (e.g. 50 trees) to deliver responsive recommendation scoring within interactive UI time budgets.

### 2.2. Extended Model Comparison Roadmap
When extending the platform to model comparison (Stage C):
- **Baseline Models**: LogisticRegression / Ridge, DecisionTreeClassifier / Regressor, LightGBM / HistGradientBoosting.
- **Fair Comparison Protocol**: All models must be evaluated against the exact same locked train/test split using identical preprocessing pipelines.
- **Multi-Metric Comparison Table**: Prevents single-metric optimization traps by comparing Precision, Recall, F1, PR-AUC, and inference latency concurrently.

---

## 3. Explainability & Diagnostics Integration

### 3.1. Native Feature Importance
- Supported tree models expose Gini / impurity-based feature importances natively.
- **Limitation Warning**: Impurity importance tends to inflate the importance of high-cardinality numerical features.

### 3.2. Permutation Importance
- Computed strictly on training validation folds using `sklearn.inspection.permutation_importance`.
- Prevents test set snooping during exploratory diagnostic inspection.

### 3.3. Optional SHAP Integration
- For deep model explanation, SHAP (TreeExplainer) may be optionally imported if installed.
- **Constraint**: SHAP remains an optional peer dependency; Dataset Doctor core never mandates heavyweight dependencies for standard operation.

---

## 4. Export & Artifact Integrations

### 4.1. Standalone `reproduce.py`
- Emits a clean, standard Python script using standard pandas and scikit-learn.
- Zero runtime dependency on Dataset Doctor itself.

### 4.2. Cleaned Dataset Export
- Exports cleaned CSV with explicit metadata headers noting applied transformations, random seeds, and row-loss statistics.

### 4.3. Serialized Pipeline
- Supports exporting fitted transformers via standard `joblib.dump` for direct deployment into production inference services.
