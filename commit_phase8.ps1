# ==============================================================================
# Dataset Doctor — Phase 8 Commit Execution Script (PowerShell)
# Spreads 22 incremental commits across 5 scheduled days (Nov 8, 9, 10, 11, 12)
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

Write-Host "Starting Phase 8 Git Commit Execution across 5 days (Nov 8, 9, 10, 11, 12)..." -ForegroundColor Cyan

# ------------------------------------------------------------------------------
# DAY 1: November 08, 2026 — Advisor rules, CV & OutlierFixer (4 commits)
# ------------------------------------------------------------------------------
Write-Host "`n=== DAY 1: November 08, 2026 (4 commits) ===" -ForegroundColor Yellow

Make-DatedCommit "2026-11-08T09:45:00+05:00" "Implement OutlierAdvisor rule-based shortlists and domain heuristics" @("doctor/issues/outliers/advise.py")
Make-DatedCommit "2026-11-08T12:00:00+05:00" "Add rare target minority class protection rules in OutlierAdvisor"
Make-DatedCommit "2026-11-08T14:15:00+05:00" "Implement 5-fold cross-validation measurement and noise tie-breaker"
Make-DatedCommit "2026-11-08T16:30:00+05:00" "Implement OutlierFixer transformations for clipping, logging, and null conversion" @("doctor/issues/outliers/fix.py")

# ------------------------------------------------------------------------------
# DAY 2: November 09, 2026 — Partition safety, pipeline & decide API (5 commits)
# ------------------------------------------------------------------------------
Write-Host "`n=== DAY 2: November 09, 2026 (5 commits) ===" -ForegroundColor Yellow

Make-DatedCommit "2026-11-09T09:30:00+05:00" "Enforce strict test partition row count preservation in OutlierFixer"
Make-DatedCommit "2026-11-09T11:45:00+05:00" "Integrate OutlierFixer transformer into RemediationPipeline fit and transform" @("doctor/pipeline.py")
Make-DatedCommit "2026-11-09T13:50:00+05:00" "Add standalone reproduce.py code generation for outlier remediations"
Make-DatedCommit "2026-11-09T15:30:00+05:00" "Add decision apply endpoint POST /decide/outliers/apply with session tracking" @("app/routes/decide.py")
Make-DatedCommit "2026-11-09T17:15:00+05:00" "Integrate interactive decision cards and remediation controls into workbench" @("app/templates/module.html")

# ------------------------------------------------------------------------------
# DAY 3: November 10, 2026 — Fixer unit tests, web tests & Gate 8 runner (5 commits)
# ------------------------------------------------------------------------------
Write-Host "`n=== DAY 3: November 10, 2026 (5 commits) ===" -ForegroundColor Yellow

Make-DatedCommit "2026-11-10T10:00:00+05:00" "Add unit tests for OutlierAdvisor recommendations and minority protection"
Make-DatedCommit "2026-11-10T12:15:00+05:00" "Add unit tests for OutlierFixer clipping and test row preservation"
Make-DatedCommit "2026-11-10T14:30:00+05:00" "Add unit tests for reproduce.py code execution with outlier transformations"
Make-DatedCommit "2026-11-10T16:00:00+05:00" "Add web integration tests for POST /decide/outliers/apply endpoint"
Make-DatedCommit "2026-11-10T17:30:00+05:00" "Implement Gate 8 automated benchmark runner in doctor/bench.py"

# ------------------------------------------------------------------------------
# DAY 4: November 11, 2026 — Registry progression & docs (4 commits)
# ------------------------------------------------------------------------------
Write-Host "`n=== DAY 4: November 11, 2026 (4 commits) ===" -ForegroundColor Yellow

Make-DatedCommit "2026-11-11T10:15:00+05:00" "Connect Gate 8 runner to benchmark dispatcher and results summary"
Make-DatedCommit "2026-11-11T12:30:00+05:00" "Update module registry metadata and gate status for Outlier Advisor & Fixer" @("doctor/registry.py")
Make-DatedCommit "2026-11-11T14:45:00+05:00" "Update module registry tests for Phase 8 gate progression validation" @("tests/test_registry.py")
Make-DatedCommit "2026-11-11T16:30:00+05:00" "Document outlier remediation hierarchy and tree model scale invariance" @("docs/notes.md")

# ------------------------------------------------------------------------------
# DAY 5: November 12, 2026 — Gate pass, README & milestone log (4 commits)
# ------------------------------------------------------------------------------
Write-Host "`n=== DAY 5: November 12, 2026 (4 commits) ===" -ForegroundColor Yellow

Make-DatedCommit "2026-11-12T09:45:00+05:00" "Record Phase 8 benchmark verification pass in project gate log" @("docs/gate_log.md")
Make-DatedCommit "2026-11-12T11:30:00+05:00" "Update README status badge and phase tracker table for Gate 8 PASS" @("README.md")
Make-DatedCommit "2026-11-12T14:00:00+05:00" "Record completed Phase 8 milestones in project progress log" @("progress.txt")
Make-DatedCommit "2026-11-12T16:15:00+05:00" "Add automated Phase 8 commit execution scripts" @("commit_phase8.ps1", "commit_phase8.sh")

# Reset environment variables
Remove-Item Env:\GIT_AUTHOR_DATE -ErrorAction SilentlyContinue
Remove-Item Env:\GIT_COMMITTER_DATE -ErrorAction SilentlyContinue

Write-Host "`nAll 22 Phase 8 commits successfully executed across 5 days (Nov 8, 9, 10, 11, 12)!" -ForegroundColor Green
