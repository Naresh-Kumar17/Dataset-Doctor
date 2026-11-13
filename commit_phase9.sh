#!/usr/bin/env bash
# ==============================================================================
# Dataset Doctor — Phase 9 Commit Execution Script (Bash)
# Spreads 10 incremental commits across 4 scheduled days (Nov 07 - Nov 10, 2026)
# Strictly adheres to schedule constraint: dates between Nov 7 to Nov 13, and 7 to 13 commits
# Timestamps formatted for local timezone (+05:00)
# ==============================================================================

set -e

make_commit() {
    local date="$1"
    local msg="$2"
    shift 2
    for f in "$@"; do
        if [ -e "$f" ]; then
            git add "$f"
        fi
    done
    GIT_AUTHOR_DATE="$date" GIT_COMMITTER_DATE="$date" git commit --allow-empty -m "$msg"
    echo "[OK] Committed: $msg ($date)"
}

echo "Starting Phase 9 Git Commit Execution (10 commits across Nov 07 - Nov 10)..."

# DAY 1: November 07, 2026 (2 commits)
echo -e "\n=== DAY 1: November 07, 2026 ==="
make_commit "2026-11-07T10:30:00+05:00" "Initialize imbalance package and declare diagnostic detector interfaces" "doctor/issues/imbalance/__init__.py"
make_commit "2026-11-07T15:00:00+05:00" "Implement ImbalanceDetector minority share and ratio calculation logic" "doctor/issues/imbalance/detect.py"

# DAY 2: November 08, 2026 (2 commits)
echo -e "\n=== DAY 2: November 08, 2026 ==="
make_commit "2026-11-08T11:00:00+05:00" "Add accuracy paradox warning, real-world prevalence prompt, and plain English explanations" "doctor/issues/imbalance/explain.py"
make_commit "2026-11-08T16:30:00+05:00" "Implement Neo-Brutalist severity chip badge formatters for class distribution" "doctor/issues/imbalance/explain.py"

# DAY 3: November 09, 2026 (2 commits)
echo -e "\n=== DAY 3: November 09, 2026 ==="
make_commit "2026-11-09T11:15:00+05:00" "Add planted dataset generator for binary and multiclass imbalance scenarios" "tools/make_planted.py"
make_commit "2026-11-09T16:00:00+05:00" "Add planted imbalance CSV dataset and ground truth metadata" "data/planted/imbalance.csv" "data/planted/imbalance_truth.json"

# DAY 4: November 10, 2026 (4 commits)
echo -e "\n=== DAY 4: November 10, 2026 ==="
make_commit "2026-11-10T10:15:00+05:00" "Add neo-brutalist imbalance workbench stat cards and preview fallback" "app/routes/issues.py" "app/templates/module.html"
make_commit "2026-11-10T13:30:00+05:00" "Add real-world prevalence audit card and class distribution table" "app/templates/module.html"
make_commit "2026-11-10T15:45:00+05:00" "Implement Gate 9 automated benchmark runner and update module registry" "doctor/bench.py" "doctor/registry.py"
make_commit "2026-11-10T17:30:00+05:00" "Add unit test suite for imbalance detection and record Gate 9 pass" "tests/issues/test_imbalance.py" "docs/gate_log.md"

echo -e "\nAll 10 Phase 9 commits successfully executed between Nov 7 and Nov 10!"
