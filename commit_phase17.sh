#!/usr/bin/env bash
# Bash script to create scheduled commits for Phase 17 (Model Explainability & Feature Attribution)
# Dates: Strictly scheduled for the next two days after Phase 16 (Nov 11 and Nov 12, 2026)

set -e

make_commit() {
    local date="$1"
    local message="$2"
    shift 2
    for file in "$@"; do
        git add "$file"
    done
    GIT_AUTHOR_DATE="$date" GIT_COMMITTER_DATE="$date" git commit -m "$message"
    echo "Committed: $message ($date)"
}

# 1. Nov 11, 2026 (Day 1 - Morning): Planted Dataset & Truth Generator
make_commit "2026-11-11T11:30:00+05:00" \
    "feat(phase17): add planted explainability dataset and ground-truth generator" \
    "data/planted/explainability.csv" "data/planted/explainability_truth.json" "tools/make_planted.py"

# 2. Nov 11, 2026 (Day 1 - Afternoon): Core Explainability & Attribution Engine
make_commit "2026-11-11T16:45:00+05:00" \
    "feat(phase17): implement validation-only permutation importance and non-causal disclosures" \
    "doctor/explainability.py" "tests/test_explainability.py"

# 3. Nov 12, 2026 (Day 2 - Morning): Web UI & Interactive Workbench
make_commit "2026-11-12T11:15:00+05:00" \
    "feat(web): integrate interactive explainability workbench, pruning advisor, and web routes" \
    "app/routes/issues.py" "app/templates/module.html" "tests/test_web_routes.py"

# 4. Nov 12, 2026 (Day 2 - Afternoon): Registry, Benchmarks, Documentation & Scripts
make_commit "2026-11-12T16:30:00+05:00" \
    "docs: register gate 17, update gate log, progress tracker, and master benchmarks" \
    "doctor/registry.py" "doctor/bench.py" "docs/gate_log.md" "progress.txt" "README.md" "tests/test_registry.py" "commit_phase17.ps1" "commit_phase17.sh"

echo "All 4 Phase 17 scheduled commits completed successfully!"

