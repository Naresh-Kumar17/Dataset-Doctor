# Dataset Doctor: Setup Guide

## 1. Requirements
- Python 3.11 or newer
- Git
- 4 GB RAM or more
- Any modern browser

Check Python:
```bash
python --version
```

## 2. Create the project
```bash
mkdir dataset-doctor && cd dataset-doctor
git init
python -m venv .venv
```
Activate:
- Mac/Linux: `source .venv/bin/activate`
- Windows: `.venv\Scripts\activate`

## 3. requirements.txt
```
pandas>=2.2
numpy>=1.26
scikit-learn>=1.4
imbalanced-learn>=0.12
scipy>=1.12
fastapi>=0.110
uvicorn[standard]>=0.29
jinja2>=3.1
python-multipart>=0.0.9
chardet>=5.2
pytest>=8.0
httpx>=0.27
```
Install:
```bash
pip install -r requirements.txt
```

## 4. Create folders and empty module files
Mac/Linux (Git Bash on Windows also works):
```bash
mkdir -p doctor/issues app/routes app/templates app/static tools \
         data/samples data/planted tests/issues experiments docs

for m in nulls duplicates types outliers imbalance leakage; do
  mkdir -p doctor/issues/$m
  touch doctor/issues/$m/__init__.py
  for f in detect advise fix explain; do touch doctor/issues/$m/$f.py; done
done

touch doctor/__init__.py doctor/issues/__init__.py \
      doctor/base.py doctor/registry.py doctor/loader.py doctor/profile.py \
      doctor/split.py doctor/evaluate.py doctor/compare.py \
      doctor/pipeline.py doctor/report.py doctor/bench.py

touch app/main.py app/session.py \
      app/routes/upload.py app/routes/issues.py app/routes/decide.py \
      app/routes/bench.py app/routes/report.py

touch tools/make_planted.py docs/notes.md docs/gate_log.md
```
Empty files are fine. Each phase fills in only its own files.

## 5. .gitignore
```
.venv/
__pycache__/
*.pyc
.env
data/uploads/
.pytest_cache/
.DS_Store
bench_results.json
```
Do not commit large datasets. Keep only tiny samples in `data/samples/`. Planted data is small and **should** be committed (`data/planted/`).

## 6. Run the app
```bash
uvicorn app.main:app --reload
```
Open http://127.0.0.1:8000

First file to prove setup works (`app/main.py`):
```python
from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def home():
    return {"status": "Dataset Doctor is running"}
```

## 7. Planted test data
Each module gets a CSV with problems you decided in advance, and a truth file.

```bash
python tools/make_planted.py nulls        # creates data/planted/nulls.csv + nulls.truth.json
python tools/make_planted.py --all
```

`nulls.truth.json` example:
```json
{
  "module": "nulls",
  "rows": 1000,
  "planted": [
    {"column": "age",   "kind": "nan",          "count": 50},
    {"column": "phone", "kind": "hidden_token", "token": "N/A", "count": 60},
    {"column": "score", "kind": "sentinel",     "token": "-999", "count": 40},
    {"column": "notes", "kind": "nan",          "count": 700}
  ],
  "must_not_flag": ["country", "id"]
}
```

Also make `data/planted/clean_control.csv`: no real problems, but include tricky legitimate values (country code "NA", a real 0 value). A good detector stays quiet on it.

Use a fixed random seed in `make_planted.py` so everyone gets the same file.

## 8. Run a gate
```bash
python -m doctor.bench nulls        # one module
python -m doctor.bench --all        # all finished modules
```
Output shape:
```
MODULE nulls (detect)
  planted: 850 cells   found: 850   missed: 0   false alarms: 0
  RESULT: PASS
```
Or open the GUI at http://127.0.0.1:8000/bench.

## 8b. Tests
```bash
pytest -q                       # everything
pytest -q tests/issues          # only issue modules
pytest -q -k nulls              # only null tests
```

## 9. Work loop for every phase
1. Read the phase in IMPLEMENTATION.md.
2. Make the planted data first (`tools/make_planted.py`).
3. Write the gate test first, watch it fail, then write the module.
4. Run the gate.
5. PASS: write the result in `docs/gate_log.md`, tick the tracker in README.md, write 3 lines in `docs/notes.md` in your own words, commit.
6. FAIL: debug loop (below), then run again.
7. Run `python -m doctor.bench --all` before moving on.

### Debug loop (short version)
1. Is it a missed problem or a false alarm?
2. Copy the failing column into a 10-row DataFrame. Reproduce it.
3. Print the evidence the module produced.
4. Check the usual suspects: hidden tokens (`?`, `-`, `N/A`, `-999`), pandas turning "NA" into NaN, wrong threshold, numbers stored as text, another fix changing the data first.
5. Fix. Add a test that fails before and passes after.
6. Run this gate and all earlier gates.
7. Log it. After 3 failed loops, question the gate or the approach. Do not just lower the threshold.

### gate_log.md format
```
## 2026-11-03  Phase 2: nulls detect   FAIL
- Missed: 20 cells with "?" token in column phone
- Cause: token list had "-" but not "?"
- Change: added "?" to tokens, test added
## 2026-11-03  Phase 2: nulls detect   PASS
- planted 850, found 850, false alarms 0
```

## 10. First 2-day test (Phase 0)
Create `experiments/first_test.py`:
1. Load your accident CSV.
2. Stratified 80/20 split (`train_test_split(..., stratify=y, random_state=42)`).
3. Train a RandomForest baseline on train. Print accuracy, plus recall and F1 for the severe class on test.
4. Retrain with `class_weight="balanced"`. Print the same numbers.
5. Oversample **train only**, retrain, print again.
6. Compare with your old oversample-before-split result.

Pass: the numbers differ in a way you can explain in 3 lines.

## 11. Common problems
| Problem | Fix |
|---------|-----|
| `ModuleNotFoundError` | Venv not active. Activate it and reinstall |
| Port already in use | `uvicorn app.main:app --port 8001` |
| CSV fails to read | Try `encoding="latin-1"` or let chardet detect |
| Country "NA" becomes NaN | Read with `keep_default_na=False` and detect tokens yourself |
| Upload rejected | File over 50 MB; trim or sample it |
| SMOTE error on text columns | Encode categoricals first, or use random oversampling |
| Gate passes but real file breaks | Planted data was too easy; add a harder case from the real file |

## 12. Deploy (free options)
- **Render**: connect GitHub repo, start command `uvicorn app.main:app --host 0.0.0.0 --port $PORT`, health check path `/health`
- **Railway**: same start command, health check path `/health`
- **Hugging Face Spaces**: Docker space, expose port 7860

Free hosts have small memory. Lower the file limit to 10 MB on the deployed version. Public `/health` endpoint is available for container health checks and liveness monitoring.

## 13. Weekly routine
- Pick one phase (or part of one) as the week's target.
- Commit at least 3 times a week with clear messages.
- Never start a new phase before the current gate is PASS.
