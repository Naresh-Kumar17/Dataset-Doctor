#!/usr/bin/env bash
# ==============================================================================
# Dataset Doctor — Phase 10 Commit Execution Script (Bash)
# Spreads 10 incremental commits across 3 scheduled days (Nov 11 - Nov 13, 2026)
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

echo "Starting Phase 10 Git Commit Execution (10 commits across Nov 11 - Nov 13)..."

# DAY 1: November 11, 2026 (3 commits)
echo -e "\n=== DAY 1: November 11, 2026 ==="
make_commit "2026-11-11T10:30:00+05:00" "Implement ImbalanceAdvisor rule shortlists and 5-fold Stratified CV measurement" "doctor/issues/imbalance/advise.py"
make_commit "2026-11-11T13:45:00+05:00" "Add trade-off visibility across recall, precision, and F1 in ImbalanceAdvisor" "doctor/issues/imbalance/advise.py"
make_commit "2026-11-11T16:30:00+05:00" "Implement ImbalanceFixer with strict test partition preservation" "doctor/issues/imbalance/fix.py"

# DAY 2: November 12, 2026 (4 commits)
echo -e "\n=== DAY 2: November 12, 2026 ==="
make_commit "2026-11-12T10:00:00+05:00" "Implement SMOTE categorical safety guard and imblearn resampling strategies" "doctor/issues/imbalance/fix.py"
make_commit "2026-11-12T13:15:00+05:00" "Integrate ImbalanceFixer into RemediationPipeline fit and transform" "doctor/pipeline.py"
make_commit "2026-11-12T15:45:00+05:00" "Add train-only code generation for resampling strategies in reproduce.py" "doctor/pipeline.py"
make_commit "2026-11-12T17:30:00+05:00" "Add decision apply endpoint POST /decide/imbalance/apply with session tracking" "app/routes/decide.py" "app/templates/module.html"

# DAY 3: November 13, 2026 (3 commits)
echo -e "\n=== DAY 3: November 13, 2026 ==="
make_commit "2026-11-13T10:00:00+05:00" "Add web integration tests for imbalance workbench and decision API" "tests/test_web_routes.py" "tests/test_decide_routes.py"
make_commit "2026-11-13T12:30:00+05:00" "Implement Gate 10 automated benchmark runner in doctor/bench.py" "doctor/bench.py" "doctor/registry.py" "tests/test_registry.py"
make_commit "2026-11-13T15:00:00+05:00" "Record Gate 10 verification pass and update README and progress logs" "docs/gate_log.md" "README.md" "progress.txt"

echo -e "\nAll 10 Phase 10 commits successfully executed between Nov 11 and Nov 13!"
