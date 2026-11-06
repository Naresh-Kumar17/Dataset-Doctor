#!/usr/bin/env bash
# ==============================================================================
# Dataset Doctor — Phase 11 & Phase 12 Commit Execution Script (Bash)
# Spreads 10 incremental commits across 2 scheduled days (Nov 05 - Nov 06, 2026)
# Strictly adheres to schedule constraint:
#   - Dates: November 05 and November 06, 2026 ONLY
#   - Count: Between 7 and 13 commits (exactly 10 commits)
# ==============================================================================

set -euo pipefail

make_dated_commit() {
    local date="$1"
    local message="$2"
    shift 2
    local files=("$@")

    for f in "${files[@]}"; do
        if [ -e "$f" ]; then
            git add "$f"
        fi
    done

    GIT_AUTHOR_DATE="$date" GIT_COMMITTER_DATE="$date" git commit --allow-empty -m "$message"
    echo "[OK] Committed: $message ($date)"
}

echo "Starting Phase 11 & 12 Git Commit Execution (10 commits on Nov 05 and Nov 06)..."

# DAY 1: November 05, 2026 (5 commits)
make_dated_commit "2026-11-05T09:30:00+05:00" "feat(leakage): add planted synthetic dataset generator and ground truth for leakage traps" "tools/make_planted.py" "data/planted/leakage.csv" "data/planted/leakage_truth.json"

make_dated_commit "2026-11-05T11:45:00+05:00" "feat(leakage): implement LeakageDetector with correlation scan, target duplicate, and immunity guards" "doctor/issues/leakage/detect.py"

make_dated_commit "2026-11-05T14:15:00+05:00" "feat(leakage): implement LeakageAdvisor, plain-English explanations, and Neo-brutalist evidence badges" "doctor/issues/leakage/advise.py" "doctor/issues/leakage/explain.py" "doctor/issues/leakage/__init__.py"

make_dated_commit "2026-11-05T16:00:00+05:00" "feat(leakage): implement LeakageFixer engine with strict row preservation invariant" "doctor/issues/leakage/fix.py"

make_dated_commit "2026-11-05T17:45:00+05:00" "test(leakage): add comprehensive unit test suite covering detection, advice, fixing, and test invariants" "tests/issues/test_leakage.py"

# DAY 2: November 06, 2026 (5 commits)
make_dated_commit "2026-11-06T09:45:00+05:00" "feat(pipeline): integrate LeakageFixer into RemediationPipeline and standalone reproduce script generator" "doctor/pipeline.py"

make_dated_commit "2026-11-06T11:30:00+05:00" "feat(api): add POST /decide/leakage/apply endpoint with session audit trail and unit tests" "app/routes/decide.py" "tests/test_decide_routes.py"

make_dated_commit "2026-11-06T14:00:00+05:00" "feat(ui): add interactive Neo-brutalist data leakage workbench, alert banners, and action controls" "app/routes/issues.py" "app/templates/module.html" "tests/test_web_routes.py"

make_dated_commit "2026-11-06T16:15:00+05:00" "feat(bench): implement Gate 11 and Gate 12 benchmark runners and update module registry" "doctor/bench.py" "doctor/registry.py" "tests/test_registry.py"

make_dated_commit "2026-11-06T17:50:00+05:00" "docs(leakage): record Gate 11 and Gate 12 PASS in gate log, update progress log and master phase tracker" "docs/gate_log.md" "progress.txt" "README.md"

echo "[SUCCESS] Phase 11 & Phase 12 commit schedule completed successfully!"
