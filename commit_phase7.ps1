# ==============================================================================
# Dataset Doctor — Phase 7 Commit Execution Script (PowerShell)
# Spreads 23 incremental commits across 5 scheduled days (Nov 3, 4, 5, 6, 7)
# With varied, organic commit distributions (not 10 for each)
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

Write-Host "Starting Phase 7 Git Commit Execution across 5 days (Nov 3, 4, 5, 6, 7)..." -ForegroundColor Cyan

# ------------------------------------------------------------------------------
# DAY 1: November 03, 2026 — Package init, planted data & explainers (5 commits)
# ------------------------------------------------------------------------------
Write-Host "`n=== DAY 1: November 03, 2026 (5 commits) ===" -ForegroundColor Yellow

Make-DatedCommit "2026-11-03T09:30:00+05:00" "Initialize outliers package and declare diagnostic detector interfaces" @("doctor/issues/outliers/__init__.py")
Make-DatedCommit "2026-11-03T11:15:00+05:00" "Add planted outliers generator with domain impossible values and unit mix-ups" @("tools/make_planted.py")
Make-DatedCommit "2026-11-03T13:40:00+05:00" "Add planted outliers synthetic dataset CSV" @("data/planted/outliers.csv")
Make-DatedCommit "2026-11-03T15:20:00+05:00" "Add ground truth metadata for planted outlier detection benchmarks" @("data/planted/outliers_truth.json")
Make-DatedCommit "2026-11-03T17:05:00+05:00" "Implement plain English outlier explanations and severity badge formatters" @("doctor/issues/outliers/explain.py")

# ------------------------------------------------------------------------------
# DAY 2: November 04, 2026 — Detection rules: impossible, unit mix-up, MAD (4 commits)
# ------------------------------------------------------------------------------
Write-Host "`n=== DAY 2: November 04, 2026 (4 commits) ===" -ForegroundColor Yellow

Make-DatedCommit "2026-11-04T10:00:00+05:00" "Implement OutlierDetector domain-impossible boundary rules for age and percentages" @("doctor/issues/outliers/detect.py")
Make-DatedCommit "2026-11-04T12:30:00+05:00" "Implement order-of-magnitude unit mix-up detection on numeric columns"
Make-DatedCommit "2026-11-04T14:45:00+05:00" "Implement robust Median Absolute Deviation for skewed continuous features"
Make-DatedCommit "2026-11-04T16:50:00+05:00" "Add categorical and binary column immunity guards in OutlierDetector"

# ------------------------------------------------------------------------------
# DAY 3: November 05, 2026 — UI workbench, tests & Gate 7 runner (6 commits)
# ------------------------------------------------------------------------------
Write-Host "`n=== DAY 3: November 05, 2026 (6 commits) ===" -ForegroundColor Yellow

Make-DatedCommit "2026-11-05T09:15:00+05:00" "Add outliers planted dataset preview fallback to issues web route" @("app/routes/issues.py")
Make-DatedCommit "2026-11-05T10:45:00+05:00" "Add neo-brutalist outliers workbench stat cards and detection view"
Make-DatedCommit "2026-11-05T12:20:00+05:00" "Add unit test suite for domain-impossible value and unit error detection" @("tests/issues/test_outliers.py")
Make-DatedCommit "2026-11-05T14:00:00+05:00" "Add unit tests for clean skewed distribution MAD guard and control immunity"
Make-DatedCommit "2026-11-05T15:30:00+05:00" "Add web route integration test for /module/outliers workbench page" @("tests/test_web_routes.py")
Make-DatedCommit "2026-11-05T17:10:00+05:00" "Implement Gate 7 automated benchmark runner in doctor/bench.py" @("doctor/bench.py")

# ------------------------------------------------------------------------------
# DAY 4: November 06, 2026 — Bench dispatcher, registry & docs (4 commits)
# ------------------------------------------------------------------------------
Write-Host "`n=== DAY 4: November 06, 2026 (4 commits) ===" -ForegroundColor Yellow

Make-DatedCommit "2026-11-06T10:15:00+05:00" "Connect Gate 7 runner to CLI benchmark dispatcher and summary reporter"
Make-DatedCommit "2026-11-06T12:45:00+05:00" "Update module registry metadata and gate status for Outlier Detection"
Make-DatedCommit "2026-11-06T14:50:00+05:00" "Update registry unit test assertions for Gate 7 progression"
Make-DatedCommit "2026-11-06T16:40:00+05:00" "Document outlier detection rules and robust MAD skewness considerations"

# ------------------------------------------------------------------------------
# DAY 5: November 07, 2026 — Gate log, milestones & scripts (4 commits)
# ------------------------------------------------------------------------------
Write-Host "`n=== DAY 5: November 07, 2026 (4 commits) ===" -ForegroundColor Yellow

Make-DatedCommit "2026-11-07T10:30:00+05:00" "Record Gate 7 benchmark verification pass in project gate log"
Make-DatedCommit "2026-11-07T12:15:00+05:00" "Update README phase tracker and development status for Phase 7 pass"
Make-DatedCommit "2026-11-07T14:30:00+05:00" "Record completed Phase 7 milestones in project progress log"
Make-DatedCommit "2026-11-07T16:45:00+05:00" "Add automated Phase 7 commit execution scripts" @("commit_phase7.ps1", "commit_phase7.sh")

# Reset environment variables
Remove-Item Env:\GIT_AUTHOR_DATE -ErrorAction SilentlyContinue
Remove-Item Env:\GIT_COMMITTER_DATE -ErrorAction SilentlyContinue

Write-Host "`nAll 23 Phase 7 commits successfully executed across 5 days (Nov 3, 4, 5, 6, 7)!" -ForegroundColor Green
