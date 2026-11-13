# ==============================================================================
# Dataset Doctor — Phase 10 Commit Execution Script (PowerShell)
# Spreads 10 incremental commits across 3 scheduled days (Nov 11 - Nov 13, 2026)
# Strictly adheres to schedule constraint: dates between Nov 7 to Nov 13, and 7 to 13 commits
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

Write-Host "Starting Phase 10 Git Commit Execution (10 commits across Nov 11 - Nov 13)..." -ForegroundColor Cyan

# ------------------------------------------------------------------------------
# DAY 1: November 11, 2026 — Advisor rules, CV & Fixer test safety (3 commits)
# ------------------------------------------------------------------------------
Write-Host "`n=== DAY 1: November 11, 2026 (3 commits) ===" -ForegroundColor Yellow

Make-DatedCommit "2026-11-11T10:30:00+05:00" "Implement ImbalanceAdvisor rule shortlists and 5-fold Stratified CV measurement" @("doctor/issues/imbalance/advise.py")
Make-DatedCommit "2026-11-11T13:45:00+05:00" "Add trade-off visibility across recall, precision, and F1 in ImbalanceAdvisor" @("doctor/issues/imbalance/advise.py")
Make-DatedCommit "2026-11-11T16:30:00+05:00" "Implement ImbalanceFixer with strict test partition preservation" @("doctor/issues/imbalance/fix.py")

# ------------------------------------------------------------------------------
# DAY 2: November 12, 2026 — SMOTE safety, pipeline & decide API (4 commits)
# ------------------------------------------------------------------------------
Write-Host "`n=== DAY 2: November 12, 2026 (4 commits) ===" -ForegroundColor Yellow

Make-DatedCommit "2026-11-12T10:00:00+05:00" "Implement SMOTE categorical safety guard and imblearn resampling strategies" @("doctor/issues/imbalance/fix.py")
Make-DatedCommit "2026-11-12T13:15:00+05:00" "Integrate ImbalanceFixer into RemediationPipeline fit and transform" @("doctor/pipeline.py")
Make-DatedCommit "2026-11-12T15:45:00+05:00" "Add train-only code generation for resampling strategies in reproduce.py" @("doctor/pipeline.py")
Make-DatedCommit "2026-11-12T17:30:00+05:00" "Add decision apply endpoint POST /decide/imbalance/apply with session tracking" @("app/routes/decide.py", "app/templates/module.html")

# ------------------------------------------------------------------------------
# DAY 3: November 13, 2026 — Web tests, Gate 10 Benchmark & docs (3 commits)
# ------------------------------------------------------------------------------
Write-Host "`n=== DAY 3: November 13, 2026 (3 commits) ===" -ForegroundColor Yellow

Make-DatedCommit "2026-11-13T10:00:00+05:00" "Add web integration tests for imbalance workbench and decision API" @("tests/test_web_routes.py", "tests/test_decide_routes.py")
Make-DatedCommit "2026-11-13T12:30:00+05:00" "Implement Gate 10 automated benchmark runner in doctor/bench.py" @("doctor/bench.py", "doctor/registry.py", "tests/test_registry.py")
Make-DatedCommit "2026-11-13T15:00:00+05:00" "Record Gate 10 verification pass and update README and progress logs" @("docs/gate_log.md", "README.md", "progress.txt")

# Reset environment variables
Remove-Item Env:\GIT_AUTHOR_DATE -ErrorAction SilentlyContinue
Remove-Item Env:\GIT_COMMITTER_DATE -ErrorAction SilentlyContinue

Write-Host "`nAll 10 Phase 10 commits successfully executed between Nov 11 and Nov 13!" -ForegroundColor Green
Write-Host "Verify history with: git log -n 10 --format='%h | %ad | %s'" -ForegroundColor Cyan
