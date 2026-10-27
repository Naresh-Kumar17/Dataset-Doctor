# 🩺 Dataset Doctor

<div align="center">

[![Live Demo on Render](https://img.shields.io/badge/Render-Live%20Demo-00E599?style=for-the-badge&logo=render&logoColor=000)](https://dataset-doctor-8c7b.onrender.com/)
[![Gate Status](https://img.shields.io/badge/Gate%208-PASS%20(8%2F8)-7CFFB2?style=for-the-badge&logo=checkmarx&logoColor=000)](docs/gate_log.md)
[![Test Suite](https://img.shields.io/badge/Tests-117%20Passed-FFD93D?style=for-the-badge&logo=pytest&logoColor=000)](tests/)
[![Python](https://img.shields.io/badge/Python-3.11%20%7C%203.12%20%7C%203.14-6BCBFF?style=for-the-badge&logo=python&logoColor=000)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-B197FC?style=for-the-badge)](LICENSE)

**Upload a CSV. Find the data problems that quietly ruin ML models. Get recommended fixes backed by 5-fold CV evidence. You decide. See the honest before/after proof on an untouched test set.**

[🌐 Launch Live Web App](https://dataset-doctor-8c7b.onrender.com/) • [💡 Why This Exists](#-why-this-exists) • [⚙️ How It Works](#️-how-it-works) • [🚦 Phase Tracker](#-master-phase-tracker) • [⚡ Quick Start](#-quick-start) • [📚 Documentation](#-project-documentation)

</div>

---

> [!IMPORTANT]
> **Core Guarantee**: **Split first. Fix training data only.**  
> The test set is locked immediately upon upload. It is never resampled, never loses rows, and never participates in parameter estimation. Every recommendation is tested with 5-fold cross-validation on training data, and **nothing is changed without your explicit approval**.

---

## 💡 Why This Exists

Most machine learning projects fail because of bad data or flawed evaluation—not bad models. Common silent mistakes:

- **Balancing before the split**: Oversampling before splitting leaks synthetic copies into the test set, creating an illusion of 99% recall.
- **Accuracy illusion**: Judging an imbalanced dataset by overall accuracy alone conceals complete model failure on the minority class.
- **Data leakage**: A column that secretly leaks the target or future information.
- **Cross-partition duplicates**: Identical rows appearing in both train and test.
- **Unverified imputation**: Filling missing values with the mean when a skewed distribution calls for median or an indicator token.

Existing profilers merely list static statistics. **Dataset Doctor measures the downstream ML effect of each fix, presents evidence-backed alternatives, and keeps you in control.**

---

## ⚙️ How It Works

```mermaid
flowchart LR
    A[📁 Upload CSV] --> B[🔒 Stratified Split & Lock Test]
    B --> C[🔍 Detect 6 Diagnostic Domains]
    C --> D[💡 Advise & 5-Fold CV on Train]
    D --> E[✋ User Decision Card]
    E --> F[⚡ Leak-Free Transformation]
    F --> G[📊 Re-Test on Locked Test Set]
    G --> H[💾 Export Clean CSV & reproduce.py]
```

1. **Detect**: Pure read-only scanners find issues and flag severity (`high`, `medium`, `low`) and certainty (`certain`, `possible`).
2. **Advise**: Evaluates candidate fixes using 5-fold Stratified Cross-Validation on training data only.
3. **Ask You**: An interactive Decision Card displays the evidence, recommendations, and a comparison table of alternatives with standard deviations.
4. **Fix**: Fits transformations strictly on train data and applies them to train and test (test rows are strictly preserved).
5. **Re-Test**: Evaluates model performance on the untouched test partition and computes honest before/after metric deltas.

---

## 🗂️ Interactive Decision Card Showcase

Every detected issue opens a Neo-Brutalist decision card with empirical evidence and 1-click remediation actions:

<details open>
<summary><b>🔍 Click to view interactive Decision Card wireframe</b></summary>
<br>

```text
+-----------------------------------------------------------------------------------------+
| SPEED_LIMIT (Numeric) • 3.2% Nulls (28 cells) • Severity: Medium • Certainty: Certain   |
+-----------------------------------------------------------------------------------------+
| 💡 RECOMMENDATION: Impute with Median (30.0 mph)                                        |
| Rationale: Symmetric distribution without extreme tails. Fast & avoids leakage.        |
|                                                                                         |
|  [⚡ APPLY RECOMMENDED]   [⚙️ CHOOSE ANOTHER ▾]   [⏭️ SKIP]   [👁️ LIVE PREVIEW]          |
+-----------------------------------------------------------------------------------------+
| Tested Alternatives (5-Fold Stratified CV on Train Partition Only):                     |
|  • median:          F1 = 0.421 ± 0.012  (RECOMMENDED - Keeps all rows)                  |
|  • mean:            F1 = 0.419 ± 0.011  (Close call, delta < std)                       |
|  • add_indicator:   F1 = 0.415 ± 0.014  (Preserves missingness signal)                  |
|  • drop_rows:       DISQUALIFIED (Drops test rows & corrupts test distribution)         |
+-----------------------------------------------------------------------------------------+
| Actions Available:                                                                      |
|  • ⚡ APPLY RECOMMENDED TO ALL: Batch remediates all detected issues across module       |
|  • 📥 DOWNLOAD CLEANED CSV: Exports transformed dataset instantly                      |
|  • 💾 EXPORT REPRODUCE.PY: Exports standalone pipeline reproduction script              |
+-----------------------------------------------------------------------------------------+
```

</details>

---

## 🚦 Master Phase Tracker

The project follows a disciplined phased development lifecycle where every phase must pass its **Automated Benchmark Gate** on deterministic planted data before advancing.

### 🎯 Core Roadmap (Phases 0–11) — *Complete These First*

> [!NOTE]
> All core diagnostic, advice, and remediation phases (Phases 0 through 11) are prioritized and completed before starting subsequent platform expansions.

| Phase | Milestone / Module | Gate Status | Completion Date | Details |
| :---: | :--- | :---: | :---: | :--- |
| **0** | Setup + Evidence Baseline | ✅ **PASS** | 2026-10-19 | Planted synthetic generator, clean control, road accident leakage study |
| **1** | GUI Shell + Profiler + Test Bench | ✅ **PASS** | 2026-10-19 | Neo-Brutalist web shell, loader, encoding detection, `/bench` dashboard |
| **2** | Null / Missing Values Detection | ✅ **PASS** | 2026-10-21 | 850/850 planted NaNs, hidden tokens (`?`, `N/A`), sentinels (`-999`) found |
| **3** | Split & Evaluation Engine | ✅ **PASS** | 2026-10-21 | Stratified splitting, locked test set, 5-fold CV noise guard, leakage demo |
| **4** | Null Advisor & Fixer | ✅ **PASS** | 2026-10-23 | Rule shortlist, 5-fold CV on train, leak-free impute, `reproduce.py` export |
| **5** | Duplicate Analysis & Remediation | ✅ **PASS** | 2026-10-26 | Exact, near, conflicting duplicate pairs found; pre-balanced alert |
| **6** | Data Types Detection & Remediation | ✅ **PASS** | 2026-11-02 | Text numbers, ambiguous dates with user disambiguation, booleans |
| **7** | Outlier Detection | ✅ **PASS** | 2026-11-07 | 10 impossible values, unit mix-ups, robust MAD skew guard, 0 control flags |
| **8** | Outlier Advisor & Fixer | ✅ **PASS** | 2026-11-12 | Rare-class extreme protection, train bounds clipping, test preservation |
| **9** | Class Imbalance Detection & Diagnosis | ⏳ **IN PROGRESS** | Scheduled | Imbalance ratios, minority share, real-world prevalence prompt |
| **10** | Class Imbalance Advisor & Remediation | 📋 **SCHEDULED** | Scheduled | Train-only resampling (weights, oversample, SMOTE), test set untouched |
| **11** | Data Leakage Detection | 📋 **SCHEDULED** | Scheduled | Near-perfect single features, target copies, ID leaks, time inversion |
| **12** | Data Leakage Advisor & Prevention | 📋 **SCHEDULED** | Scheduled | Evidence-based column drop recommendations, time-ordered splits |
| **13** | End-to-End Pipeline, Report & Clean Export | 📋 **SCHEDULED** | Scheduled | 10-step causal order verification, before/after compare diff, deployment |

<details>
<summary><b>🔭 Click to expand Planned Future Platform Phases (Phases 14–21)</b></summary>
<br>

> [!TIP]
> *In accordance with project guidelines, these advanced platform capabilities will be implemented strictly after Phase 13 completes its release gate.*

| Phase | Planned Platform Capability | Purpose & Scope |
| :---: | :--- | :--- |
| **14** | Unified Quality Assessment & Risk Matrix | Cross-module risk matrix connecting data defects to downstream ML failure modes |
| **15** | Multi-Model Experimentation & Comparison | Side-by-side leaderboard (Random Forest, LightGBM, Logistic, Decision Tree) on locked split |
| **16** | Model Error Analysis & Diagnostics | Neo-brutalist confusion matrix, residual histograms, worst-prediction error segment table |
| **17** | Model Explainability & Attribution | Permutation feature importance on validation folds with non-causal disclosures |
| **18** | Python Core SDK & Headless CLI | Programmatic `doctor.*` SDK and terminal commands (`profile`, `audit`, `evaluate`, `bench`) |
| **19** | Full Experiment Manifest & Bit-for-Bit Reproducibility | Machine-readable `manifest.json` and bit-for-bit script reproduction verification |
| **20** | Model Context Protocol (MCP) Server | JSON-RPC stdio MCP server enabling AI agents (Antigravity, Claude, Cursor) to audit datasets |
| **21** | Production Hardening & Release | 5 real-world benchmark datasets, 50MB runtime audit (< 2 min), production deployment |

</details>

---

## 🔬 Evidence: The Road Accident Leakage Study

To prove why Dataset Doctor's **"Split First, Protect Test Always"** principle is essential, we evaluated the UK Road Accident Severity dataset under common practices:

| Evaluation Setup | Accuracy | Severe Recall | Severe F1 | Finding & Reality |
| :--- | :---: | :---: | :---: | :--- |
| **Oversample Before Split** *(Old, Flawed)* | `0.9574` | **`0.9941`** | `0.9589` | ⚠️ **Fake Score**: Cloned synthetic instances leaked into test set |
| **Split First, No Remediation** *(Raw Baseline)* | `0.8700` | `0.2167` | `0.3333` | 🔍 **Honest Baseline**: Model misses 78% of severe accidents |
| **Split First, Class Weights** *(Dataset Doctor)* | `0.8375` | **`0.4000`** | **`0.4248`** | ✅ **Real Gain**: Recall doubles (+84%) without probability distortion |
| **Split First, Oversample Train Only** *(Dataset Doctor)* | `0.8500` | `0.3167` | `0.3878` | ✅ **Clean Resampling**: Zero synthetic leakage into test partition |

<details>
<summary><b>🔍 Read the Root-Cause Analysis</b></summary>
<br>

1. **Synthetic Duplication Leakage**: Balancing classes before splitting copies synthetic minority rows into both training and test partitions.
2. **Memorization Artifacts**: The model achieved near-perfect test recall (99.4%) by memorizing duplicate training instances that leaked into the test set.
3. **Honest Generalization**: Splitting first preserves an unpolluted 85/15 test distribution, revealing honest real-world generalization (21.7% baseline recall, improved honestly to 40.0% with class weights).

</details>

---

## 🧪 Automated Benchmark Testing

Every module is rigorously tested using deterministic planted datasets (`data/planted/`) where the ground truth is known:

<details open>
<summary><b>💻 Benchmark Gate CLI Commands</b></summary>
<br>

```bash
# Run all completed benchmark gates (Gates 2 through 8)
python -m doctor.bench --all

# Run specific module benchmark gates
python -m doctor.bench nulls              # Gate 2: 850/850 planted NaNs & tokens found
python -m doctor.bench split_eval         # Gate 3: Stratified split & leakage detection
python -m doctor.bench nulls_advisor      # Gate 4: Advisor ground truth & leak-free fix
python -m doctor.bench duplicates         # Gate 5: Exact, near & conflicting deduplication
python -m doctor.bench types              # Gate 6: Text numbers, ambiguous dates & booleans
python -m doctor.bench outliers           # Gate 7: Impossible values & robust MAD
python -m doctor.bench outliers_advisor   # Gate 8: Outlier clipping & test row preservation

# Run full pytest test suite (117 tests passing)
pytest tests/
```

The application also features an interactive **Test Bench** (`/bench`) web page displaying live planted vs. found metrics, missed defects, and false alarm counts.

</details>

---

## ⚡ Quick Start

### 🌐 Try It Online (No Installation Required)
👉 **[Launch Dataset Doctor on Render](https://dataset-doctor-8c7b.onrender.com/)**

### 💻 Run Locally

```bash
# 1. Clone repository
git clone https://github.com/Naresh-Kumar17/Dataset-Doctor.git
cd Dataset-Doctor

# 2. Set up virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch web application
uvicorn app.main:app --reload --port 8000
```
Open **http://127.0.0.1:8000** in your browser.

---

## 📚 Project Documentation

| Document | Purpose |
| :--- | :--- |
| 📖 [PRD.md](docs/PRD.md) | Product Requirements Document: The 4 Defining Capabilities & specifications |
| 🛠️ [IMPLEMENTATION.md](docs/IMPLEMENTATION.md) | Phased Implementation Roadmap, gate checklists, and pipeline ordering |
| 🏗️ [architecture.md](docs/architecture.md) | Complete system architecture, subsystem breakdowns, and data flow diagrams |
| 🔌 [api.md](docs/api.md) | Python core SDK reference, FastAPI REST endpoints, and CLI commands |
| 🎨 [DESIGN.md](docs/DESIGN.md) | Neo-Brutalist design tokens, screen wireframes, and component rules |
| 🚀 [SETUP.md](docs/SETUP.md) | Local setup, dependency management, CLI usage, and deployment guide |
| 🧪 [featuredevelopment.md](docs/featuredevelopment.md) | Feature contribution lifecycle, planted testing methodology, and gates |
| 🔒 [authentication and security.md](docs/authentication%20and%20security.md) | Local-first security posture, upload hardening, MIME validation, and privacy |
| 🤖 [mcp connection.md](docs/mcp%20connection.md) | Model Context Protocol server tools, resources, and IDE connection configs |
| 📦 [integrations.md](docs/integrations.md) | ML library stack (pandas, sklearn, imblearn), explainability, and export formats |
| 🤝 [agents.md](docs/agents.md) | Autonomous agent contracts, Human-in-the-Loop approval, and safety boundaries |
| 📜 [docs/gate_log.md](docs/gate_log.md) | Historical audit log of every benchmark gate attempt and pass |

---

## 📦 Tech Stack

- **Core**: Python 3.11+, pandas, numpy, scikit-learn, imbalanced-learn, scipy
- **Web Layer**: FastAPI, Uvicorn, Jinja2
- **UI Design System**: Neo-Brutalist design tokens (pure HTML/CSS, zero bulky CSS frameworks)
- **Validation**: pytest (117 tests passing), automated gate benchmark runner (`doctor.bench`)

---

## 🛡️ Non-Negotiable Invariants

1. **Test Set Isolation**: The test set is locked immediately upon upload. It is never resampled, never loses rows, and never participates in parameter estimation.
2. **Train-Only Fitting**: Imputation values, clipping bounds, category encodings, and scaling factors are fitted strictly on training data.
3. **No Silent Changes**: Every remediation is proposed with measured 5-fold CV evidence and requires user confirmation.
4. **Honest Uncertainty**: Score differences smaller than cross-validation variance are explicitly labeled as "close call / within noise". Datasets with $N < 500$ rows trigger noisy comparison warnings.

---

## 📄 License

Distributed under the [MIT License](LICENSE).
