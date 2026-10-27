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

## 8. Automated Benchmark Gates & Testing

Run all completed benchmark gates:
```bash
python -m doctor.bench --all
```

Run specific module benchmark gates:
```bash
python -m doctor.bench nulls              # Gate 2: Null detection
python -m doctor.bench split_eval         # Gate 3: Split and evaluation engine
python -m doctor.bench nulls_advisor      # Gate 4: Null advisor and fixer
python -m doctor.bench duplicates         # Gate 5: Duplicates detection & fixer
python -m doctor.bench types              # Gate 6: Data types detection & fixer
python -m doctor.bench outliers           # Gate 7: Outlier detection
python -m doctor.bench outliers_advisor   # Gate 8: Outlier advisor and fixer
python -m doctor.bench imbalance          # Gate 9: Imbalance detection & fixer
```

Run pytest test suite:
```bash
pytest -q                       # all 117+ tests
pytest -q tests/issues          # issue-specific tests
pytest -q -k outliers           # filtered tests
```

---

## 8b. Command-Line Interface (CLI) Usage

Dataset Doctor exposes a headless CLI reusing the core engine without starting the web server:
```bash
# Profile a dataset
python -m doctor profile data/samples/road_accident.csv --target severity

# Run complete quality audit
python -m doctor audit data/samples/road_accident.csv --target severity --format json

# Evaluate baseline model
python -m doctor evaluate data/samples/road_accident.csv --target severity
```

---

## 8c. Model Context Protocol (MCP) Server

Connect Dataset Doctor to AI assistants (Antigravity, Claude Desktop, Cursor) via MCP:
```bash
python -m doctor.mcp
```
Add server configuration to your IDE's `mcp_config.json` as detailed in [mcp connection.md](mcp%20connection.md).

---

## 9. Work loop for every phase
1. Read the phase specifications in [IMPLEMENTATION.md](IMPLEMENTATION.md).
2. Make or verify the planted dataset first (`tools/make_planted.py`).
3. Write the gate test first in `doctor/bench.py`, watch it fail, then implement the module.
4. Run the gate: `python -m doctor.bench <module_id>`.
5. PASS: write the result in `docs/gate_log.md`, update [README.md](README.md), log notes in `docs/notes.md`, commit.
6. FAIL: debug loop (below), then run again.
7. Always run `python -m doctor.bench --all` and `pytest tests/` before advancing to the next phase.

---

## 10. Documentation Index
- [Architecture Blueprint](architecture.md): System design, data flow, and isolation invariants.
- [API Reference](api.md): Python SDK, REST endpoints, and CLI commands.
- [Feature Development Guide](featuredevelopment.md): Planted testing methodology and phase lifecycle.
- [Authentication & Security](authentication%20and%20security.md): Local-first security, upload hardening, and privacy.
- [Model Context Protocol Guide](mcp%20connection.md): MCP tool definitions and connection setup.
- [Ecosystem Integrations](integrations.md): ML library stack, SHAP, and export formats.
- [Agentic Workflows](agents.md): AI pair programming rules and collaboration protocols.
- [Product Requirements](PRD.md): 4 defining capabilities and 21-phase roadmap.
- [Implementation Plan](IMPLEMENTATION.md): Detailed phase specs, gate criteria, and tracker.
- [GUI Design Specification](DESIGN.md): Neo-Brutalist tokens and screen wireframes.

