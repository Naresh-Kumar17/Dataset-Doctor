# Dataset Doctor

[![Live Demo on Render](https://img.shields.io/badge/Render-Live%20Demo-brightgreen?style=for-the-badge&logo=render)](https://dataset-doctor-8c7b.onrender.com/)
[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Status](https://img.shields.io/badge/Gate%204-PASS-success?style=for-the-badge)](docs/gate_log.md)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](LICENSE)

Upload a CSV. Find the data problems that quietly ruin ML models. Get a recommended fix with reasons and test results. You decide. See the real before/after result.

> 🌐 **Live Application**: [https://dataset-doctor-8c7b.onrender.com/](https://dataset-doctor-8c7b.onrender.com/)  
> 🚦 **Development Gate**: **Phase 4 PASS** (Interactive Null Advisor, Fixer, Batch Actions, & CSV Export).

---

### Quick Navigation
[🌐 Live Web Demo](https://dataset-doctor-8c7b.onrender.com/) • [💡 Why This Exists](#why-this-exists) • [⚙️ Workflow](#how-it-works) • [📋 Modules](#modules-one-folder-per-issue-three-files-each) • [🚦 Phase Tracker](#phase-tracker) • [⚡ Quick Start](#quick-start)


## Why this exists
Many ML scores are fake. Common causes:
- Balancing classes **before** the train/test split.
- Judging an imbalanced problem by **accuracy** alone.
- A column that **leaks** the answer.
- Duplicate rows sitting in both train and test.
- Missing values filled or deleted with no proof the choice was good.

Existing tools list these problems. Dataset Doctor also **tests which fix works best** on a clean test set and **asks you before changing anything**.

## How it works
```
Upload -> Read -> Detect -> Advise -> Ask you -> Fix -> Re-test
```
- **Detect:** find problems, change nothing.
- **Advise:** pick likely methods by rules, then test each on the training data.
- **Ask you:** a decision card shows the evidence, the recommendation, and the scores of the alternatives.
- **Fix:** applied to training data, with values learned from training data only.
- **Re-test:** same untouched test set, before vs after.

<details>
<summary><b>🔍 Click to view interactive Decision Card preview & features</b></summary>
<br>

```text
+-----------------------------------------------------------------------------------+
| SPEED_LIMIT (Numeric) • 3.2% Nulls (28 cells) • Severity: Medium                  |
+-----------------------------------------------------------------------------------+
| 💡 RECOMMENDATION: Impute with Median (30.0 mph)                                  |
| Rationale: Symmetric distribution without extreme tails. Fast & avoids leakage.  |
|                                                                                   |
|  [⚡ APPLY RECOMMENDED]   [⚙️ CHOOSE ANOTHER]   [⏭️ SKIP]   [👁️ LIVE PREVIEW]      |
+-----------------------------------------------------------------------------------+
| Remediations Comparison (5-Fold Stratified CV on Train Partition):                |
|  • median:          F1 = 0.421 ± 0.012  (RECOMMENDED)                            |
|  • mean:            F1 = 0.419 ± 0.011  (Close call, delta < std)                |
|  • add_indicator:   F1 = 0.415 ± 0.014                                           |
|  • drop_rows:       DISQUALIFIED (Drops test rows & distorts true distribution)   |
+-----------------------------------------------------------------------------------+
| Actions available in UI:                                                          |
|  • ⚡ APPLY RECOMMENDED TO ALL: Batch remediates all detected null issues          |
|  • 📥 DOWNLOAD CLEANED CSV: Exports transformed dataset instantly                |
|  • 💾 EXPORT REPRODUCE.PY: Exports reproducible pipeline script                   |
+-----------------------------------------------------------------------------------+
```

</details>


## Modules (one folder per issue, three files each)
| Issue | Detect | Advise | Fix | Phase |
|-------|--------|--------|-----|-------|
| Null / missing values | `issues/nulls/detect.py` | `advise.py` | `fix.py` | 2 and 4 |
| Duplicates | `issues/duplicates/detect.py` | `advise.py` | `fix.py` | 5 |
| Data types | `issues/types/detect.py` | `advise.py` | `fix.py` | 6 |
| Outliers | `issues/outliers/detect.py` | `advise.py` | `fix.py` | 7 and 8 |
| Class imbalance | `issues/imbalance/detect.py` | `advise.py` | `fix.py` | 9 |
| Leakage | `issues/leakage/detect.py` | `advise.py` | `fix.py` | 10 |

## Rules the tool follows
1. Split first. Fix training data only.
2. Test set keeps the real class ratio and never loses rows.
3. For imbalanced targets, main metrics are minority recall, precision, F1, PR-AUC. Not accuracy.
4. No silent changes. You accept, change, or skip every fix.
5. Rules are guesses. Measured scores are evidence. The tool shows both and says "close call" when scores are within noise.

## How the project is built: phases with gates
Each phase ends with a **gate**: run the module on data with problems planted in advance. Pass, then continue. Fail, then find where it went wrong and run the gate again.

<details>
<summary><b>🧪 Click to expand Automated Benchmark Gate commands</b></summary>
<br>

```bash
# Run individual phase gates
python -m doctor.bench nulls          # Phase 2: 100% planted missing found, 0 false alarms
python -m doctor.bench split_eval     # Phase 3: Train/test split preservation & leakage check
python -m doctor.bench nulls_advisor  # Phase 4: Advisor ground truth & fixer verification

# Re-run all completed gates
python -m doctor.bench --all
```

The GUI also features an interactive **Test Bench** page (`/bench`) that shows the same result: planted vs found, missed, false alarms.
</details>

Full details: [IMPLEMENTATION.md](IMPLEMENTATION.md)

## Phase tracker
| Phase | What | Gate passed | Date |
|-------|------|-------------|------|
| 0 | Setup + accident redo (evidence) | [x] PASS | 2026-10-19 |
| 1 | GUI shell + loader + profile (test bench) | [x] PASS | 2026-10-19 |
| 2 | Null detection | [x] PASS | 2026-10-21 |
| 3 | Split + evaluation engine | [x] PASS | 2026-10-21 |
| 4 | Null advisor + fixer (interactive) | [x] PASS | 2026-10-23 |
| 5 | Duplicates | [ ] | |
| 6 | Data types | [ ] | |
| 7 | Outlier detection | [ ] | |
| 8 | Outlier advisor + fixer | [ ] | |
| 9 | Imbalance | [ ] | |
| 10 | Leakage | [ ] | |
| 11 | Full pipeline, report, deploy | [ ] | |

## Evidence from the accident dataset (fill after Phase 0)
| Setup | Accuracy | Severe recall | Severe F1 |
|-------|----------|---------------|-----------|
| Oversample before split (old, wrong) | 0.9574 | 0.9941 | 0.9589 |
| Split first, no fix | 0.8700 | 0.2167 | 0.3333 |
| Split first, class weights | 0.8375 | 0.4000 | 0.4248 |
| Split first, oversample train only | 0.8500 | 0.3167 | 0.3878 |

<details>
<summary><b>🔍 Click to read the analysis: Why the old score was inflated & how Dataset Doctor prevents it</b></summary>
<br>

1. **Synthetic Duplication Leakage**: Oversampling before splitting cloned synthetic minority rows into both training and test sets.
2. **Memorization Artifacts**: The model achieved near-perfect test recall (99.4%) by memorizing duplicate training instances that leaked into the test set.
3. **Honest Generalization**: Splitting first preserves an unpolluted 85/15 test distribution, revealing honest real-world generalization (21.7% baseline recall, improved honestly to 40.0% with class weights).

</details>



## Tech stack
- Python 3.11+
- pandas, numpy, scikit-learn, imbalanced-learn
- FastAPI + Jinja2 templates
- Plain HTML/CSS (neo-brutalist UI, no CSS framework)
- pytest

## Quick start

> 🚀 **Try it online instantly without installation:**  
> 👉 **[https://dataset-doctor-8c7b.onrender.com/](https://dataset-doctor-8c7b.onrender.com/)**

<details>
<summary><b>💻 Run locally from source (click to expand)</b></summary>
<br>

```bash
git clone https://github.com/Naresh-Kumar17/Dataset-Doctor.git
cd Dataset-Doctor
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

</details>

Full setup and testing steps: [SETUP.md](SETUP.md)

## Project docs
- [PRD.md](PRD.md): what and why
- [IMPLEMENTATION.md](IMPLEMENTATION.md): how it is built, phases and gates
- [SETUP.md](SETUP.md): install, run, test, planted data, deploy
- [DESIGN.md](DESIGN.md): neo-brutalist GUI spec, decision card, test bench
- [GITHUB_INSTRUCTIONS.md](GITHUB_INSTRUCTIONS.md): git branching, phase completion, and PR rules

## Project structure

<details>
<summary><b>📁 Click to expand repository directory map</b></summary>
<br>

```text
dataset-doctor/
  doctor/        core diagnostic and evaluation logic (no web dependencies)
    issues/      one folder per issue: detect.py, advise.py, fix.py
    bench.py     automated benchmark gate runner
    pipeline.py  remediation pipeline & reproduce.py generator
  app/           FastAPI web application, neo-brutalist templates, API routes
  tools/         planted data generator and truth file builder
  data/          samples and planted test datasets with truth files
  tests/         unit & integration test suite (66+ passing tests)
  experiments/   baseline evaluations and accident study
  docs/          gate_log.md, phase specifications, and notes
```

</details>


## Limits
- CSV only, max 50 MB.
- Tabular classification first; regression is basic.
- Rule-based advice is a starting guess; scores are noisy on small data.
- Leakage detection uses simple rules and will miss some cases.
- Results depend on one fixed test model.

## License
MIT (change if you prefer another).
