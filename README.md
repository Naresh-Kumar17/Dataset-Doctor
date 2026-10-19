# Dataset Doctor

Upload a CSV. Find the data problems that quietly ruin ML models. Get a recommended fix with reasons and test results. You decide. See the real before/after result.

> Status: in development (Phase 0).

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

```bash
python -m doctor.bench nulls     # run one gate
python -m doctor.bench --all     # re-run every finished gate
```
The GUI has a **Test Bench** page (`/bench`) that shows the same result: planted vs found, missed, false alarms.

Full details: [IMPLEMENTATION.md](IMPLEMENTATION.md)

## Phase tracker
| Phase | What | Gate passed | Date |
|-------|------|-------------|------|
| 0 | Setup + accident redo (evidence) | [x] PASS | 2026-10-19 |
| 1 | GUI shell + loader + profile (test bench) | [x] PASS | 2026-10-19 |
| 2 | Null detection | [ ] | |
| 3 | Split + evaluation engine | [ ] | |
| 4 | Null advisor + fixer (interactive) | [ ] | |
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

**Why the old score was inflated:**
1. Oversampling before splitting cloned synthetic/duplicated minority rows into both training and test sets.
2. The model achieved near-perfect test recall (99.4%) by memorizing duplicate training instances that leaked into the test set.
3. Splitting first preserves an unpolluted 85/15 test distribution, revealing honest real-world generalization (21.7% baseline recall, improved honestly to 40.0% with class weights).


## Tech stack
- Python 3.11+
- pandas, numpy, scikit-learn, imbalanced-learn
- FastAPI + Jinja2 templates
- Plain HTML/CSS (neo-brutalist UI, no CSS framework)
- pytest

## Quick start
```bash
git clone <your-repo-url>
cd dataset-doctor
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```
Open http://127.0.0.1:8000

Full steps: [SETUP.md](SETUP.md)

## Project docs
- [PRD.md](PRD.md): what and why
- [IMPLEMENTATION.md](IMPLEMENTATION.md): how it is built, phases and gates
- [SETUP.md](SETUP.md): install, run, test, planted data, deploy
- [DESIGN.md](DESIGN.md): neo-brutalist GUI spec, decision card, test bench
- [GITHUB_INSTRUCTIONS.md](GITHUB_INSTRUCTIONS.md): git branching, phase completion, and PR rules

## Project structure
```
dataset-doctor/
  doctor/        core logic (no web code)
    issues/      one folder per issue: detect.py, advise.py, fix.py, explain.py
  app/           FastAPI app, templates, static files
  tools/         planted data generator
  data/          samples and planted test data with truth files
  tests/         unit tests per module
  experiments/   first test, accident redo
  docs/          notes.md, gate_log.md
```

## Limits
- CSV only, max 50 MB.
- Tabular classification first; regression is basic.
- Rule-based advice is a starting guess; scores are noisy on small data.
- Leakage detection uses simple rules and will miss some cases.
- Results depend on one fixed test model.

## License
MIT (change if you prefer another).
