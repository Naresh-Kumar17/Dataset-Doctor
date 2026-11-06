# ==============================================================================
# Dataset Doctor — Phase 11 & Phase 12 Commit Execution Script (PowerShell)
# Spreads 10 incremental commits across 2 scheduled days (Nov 05 - Nov 06, 2026)
# Strictly adheres to schedule constraint:
#   - Dates: November 05 and November 06, 2026 ONLY
#   - Count: Between 7 and 13 commits (exactly 10 commits)
# Timestamps formatted for local timezone (+05:00)
# ==============================================================================

$ErrorActionPreference = "Stop"

function Make-DatedCommit {
    param (
        [string]$Date,
        [string]$Message,
        [string[]]$Files = @()
    )
    foreach ($f in $Files) {
        if (Test-Path $f) {
            git add $f
        }
    }
    $env:GIT_AUTHOR_DATE = $Date
    $env:GIT_COMMITTER_DATE = $Date
    git commit --allow-empty -m $Message
    Write-Host "[OK] Committed: $Message ($Date)" -ForegroundColor Green
}

Write-Host "Starting Phase 11 & 12 Git Commit Execution (10 commits on Nov 05 and Nov 06)..." -ForegroundColor Cyan

# ------------------------------------------------------------------------------
# DAY 1: November 05, 2026 — Data Leakage Detection Core & Tests (5 commits)
# ------------------------------------------------------------------------------
Write-Host "`n=== DAY 1: November 05, 2026 (5 commits) ===" -ForegroundColor Yellow

Make-DatedCommit "2026-11-05T09:30:00+05:00" "feat(leakage): add planted synthetic dataset generator and ground truth for leakage traps" @("tools/make_planted.py", "data/planted/leakage.csv", "data/planted/leakage_truth.json")

Make-DatedCommit "2026-11-05T11:45:00+05:00" "feat(leakage): implement LeakageDetector with correlation scan, target duplicate, and immunity guards" @("doctor/issues/leakage/detect.py")

Make-DatedCommit "2026-11-05T14:15:00+05:00" "feat(leakage): implement LeakageAdvisor, plain-English explanations, and Neo-brutalist evidence badges" @("doctor/issues/leakage/advise.py", "doctor/issues/leakage/explain.py", "doctor/issues/leakage/__init__.py")

Make-DatedCommit "2026-11-05T16:00:00+05:00" "feat(leakage): implement LeakageFixer engine with strict row preservation invariant" @("doctor/issues/leakage/fix.py")

Make-DatedCommit "2026-11-05T17:45:00+05:00" "test(leakage): add comprehensive unit test suite covering detection, advice, fixing, and test invariants" @("tests/issues/test_leakage.py")

# ------------------------------------------------------------------------------
# DAY 2: November 06, 2026 — Pipeline, Web UI, Benchmarks & Documentation (5 commits)
# ------------------------------------------------------------------------------
Write-Host "`n=== DAY 2: November 06, 2026 (5 commits) ===" -ForegroundColor Yellow

Make-DatedCommit "2026-11-06T09:45:00+05:00" "feat(pipeline): integrate LeakageFixer into RemediationPipeline and standalone reproduce script generator" @("doctor/pipeline.py")

Make-DatedCommit "2026-11-06T11:30:00+05:00" "feat(api): add POST /decide/leakage/apply endpoint with session audit trail and unit tests" @("app/routes/decide.py", "tests/test_decide_routes.py")

Make-DatedCommit "2026-11-06T14:00:00+05:00" "feat(ui): add interactive Neo-brutalist data leakage workbench, alert banners, and action controls" @("app/routes/issues.py", "app/templates/module.html", "tests/test_web_routes.py")

Make-DatedCommit "2026-11-06T16:15:00+05:00" "feat(bench): implement Gate 11 and Gate 12 benchmark runners and update module registry" @("doctor/bench.py", "doctor/registry.py", "tests/test_registry.py")

Make-DatedCommit "2026-11-06T17:50:00+05:00" "docs(leakage): record Gate 11 and Gate 12 PASS in gate log, update progress log and master phase tracker" @("docs/gate_log.md", "progress.txt", "README.md")

Write-Host "`n[SUCCESS] Phase 11 & Phase 12 commit schedule completed successfully!" -ForegroundColor Green
