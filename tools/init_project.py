"""
Setup script to initialize the project folder structure and empty files
as defined in SETUP.md and IMPLEMENTATION.md.
"""
from pathlib import Path

root = Path(__file__).resolve().parent.parent

# Directories
dirs = [
    "doctor/issues/nulls",
    "doctor/issues/duplicates",
    "doctor/issues/types",
    "doctor/issues/outliers",
    "doctor/issues/imbalance",
    "doctor/issues/leakage",
    "app/routes",
    "app/templates",
    "app/static",
    "tools",
    "data/samples",
    "data/planted",
    "tests/issues",
    "experiments",
    "docs",
]

for d in dirs:
    (root / d).mkdir(parents=True, exist_ok=True)

# Files to touch
files = [
    "doctor/__init__.py",
    "doctor/issues/__init__.py",
    "doctor/base.py",
    "doctor/registry.py",
    "doctor/loader.py",
    "doctor/profile.py",
    "doctor/split.py",
    "doctor/evaluate.py",
    "doctor/compare.py",
    "doctor/pipeline.py",
    "doctor/report.py",
    "doctor/bench.py",
    "app/__init__.py",
    "app/session.py",
    "app/routes/__init__.py",
    "app/routes/upload.py",
    "app/routes/issues.py",
    "app/routes/decide.py",
    "app/routes/bench.py",
    "app/routes/report.py",
    "tools/__init__.py",
    "tools/make_planted.py",
    "docs/notes.md",
    "docs/gate_log.md",
]

modules = ["nulls", "duplicates", "types", "outliers", "imbalance", "leakage"]
for m in modules:
    files.append(f"doctor/issues/{m}/__init__.py")
    for f in ["detect.py", "advise.py", "fix.py", "explain.py"]:
        files.append(f"doctor/issues/{m}/{f}")

for f in files:
    target = root / f
    if not target.exists():
        target.touch()

print("Folder structure and skeleton files created successfully.")
