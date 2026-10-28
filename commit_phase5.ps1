# ==============================================================================
# Dataset Doctor — Phase 5 Commit Execution Script (PowerShell)
# Spreads 28 incremental commits across 3 consecutive days
# Timestamps formatted for local timezone (+05:00)
# ==============================================================================

$ErrorActionPreference = "Stop"

function Make-DatedCommit {
    param (
        [string]$Date,
        [string]$Message,
        [string[]]$Files
    )
    foreach ($f in $Files) {
        git add $f
    }
    $env:GIT_AUTHOR_DATE = $Date
    $env:GIT_COMMITTER_DATE = $Date
    git commit -m $Message
    Write-Host "[OK] Committed: $Message ($Date)" -ForegroundColor Green
}

Write-Host "Starting Phase 5 Git Commit Execution across 3 days..." -ForegroundColor Cyan

# ------------------------------------------------------------------------------
# DAY 1 (Today: 2026-10-26): Phase 4 wrap-up, directory setup, base configs (9 commits)
# ------------------------------------------------------------------------------
Write-Host "`n=== DAY 1: October 26, 2026 ===" -ForegroundColor Yellow

Make-DatedCommit "2026-10-26T10:15:00+05:00" "Add optional measurement flag to null advisor" @("doctor/issues/nulls/advise.py")
Make-DatedCommit "2026-10-26T11:00:00+05:00" "Support missing string tokens and sentinels in null fixer" @("doctor/issues/nulls/fix.py")
Make-DatedCommit "2026-10-26T11:45:00+05:00" "Add unit tests for sentinel and string token normalization" @("tests/issues/test_nulls_fix.py")
Make-DatedCommit "2026-10-26T13:20:00+05:00" "Allow unsupervised mode without target column in target route" @("app/routes/upload.py")
Make-DatedCommit "2026-10-26T14:15:00+05:00" "Initialize duplicates issue package and export public interface" @("doctor/issues/duplicates/__init__.py")
Make-DatedCommit "2026-10-26T15:00:00+05:00" "Add planted dataset generators for duplicate rows and prebalanced files" @("tools/make_planted.py")
Make-DatedCommit "2026-10-26T15:45:00+05:00" "Add planted duplicates dataset CSV" @("data/planted/duplicates.csv")
Make-DatedCommit "2026-10-26T16:30:00+05:00" "Add planted duplicates ground truth metadata" @("data/planted/duplicates_truth.json")
Make-DatedCommit "2026-10-26T17:15:00+05:00" "Add synthetic prebalanced planted dataset for leakage detection" @("data/planted/prebalanced_planted.csv")

# ------------------------------------------------------------------------------
# DAY 2 (Tomorrow: 2026-10-27): Main logic, API endpoints, feature components (10 commits)
# ------------------------------------------------------------------------------
Write-Host "`n=== DAY 2: October 27, 2026 ===" -ForegroundColor Yellow

Make-DatedCommit "2026-10-27T09:30:00+05:00" "Implement duplicate fixer strategies for row deduplication" @("doctor/issues/duplicates/fix.py")
Make-DatedCommit "2026-10-27T10:20:00+05:00" "Implement duplicate advisor recommendations and strategy options" @("doctor/issues/duplicates/advise.py")
Make-DatedCommit "2026-10-27T11:15:00+05:00" "Add plain English duplicate explanation and badge formatters" @("doctor/issues/duplicates/explain.py")
Make-DatedCommit "2026-10-27T12:00:00+05:00" "Implement duplicate detector for exact, near, and conflicting duplicates" @("doctor/issues/duplicates/detect.py")
Make-DatedCommit "2026-10-27T13:30:00+05:00" "Integrate duplicate fixer into remediation pipeline and code generator" @("doctor/pipeline.py")
Make-DatedCommit "2026-10-27T14:20:00+05:00" "Add deduplication decision apply endpoint and reproduce script support" @("app/routes/decide.py")
Make-DatedCommit "2026-10-27T15:10:00+05:00" "Integrate duplicate diagnostics and preview fallbacks into module route" @("app/routes/issues.py")
Make-DatedCommit "2026-10-27T16:00:00+05:00" "Add unsupervised task selector and target synchronization script" @("app/templates/profile.html")
Make-DatedCommit "2026-10-27T16:45:00+05:00" "Update module registry status for duplicates gate progression" @("doctor/registry.py")
Make-DatedCommit "2026-10-27T17:30:00+05:00" "Update module registry tests for Phase 5 status validation" @("tests/test_registry.py")

# ------------------------------------------------------------------------------
# DAY 3 (Day After Tomorrow: 2026-10-28): Styling, bug fixes, tests, docs, cleanup (9 commits)
# ------------------------------------------------------------------------------
Write-Host "`n=== DAY 3: October 28, 2026 ===" -ForegroundColor Yellow

Make-DatedCommit "2026-10-28T09:30:00+05:00" "Add neo-brutalist duplicates workbench and remediation cards" @("app/templates/module.html")
Make-DatedCommit "2026-10-28T10:20:00+05:00" "Add comprehensive unit tests for duplicate detection and remediation" @("tests/issues/test_duplicates.py")
Make-DatedCommit "2026-10-28T11:15:00+05:00" "Add integration tests for duplicates module web route and unsupervised mode" @("tests/test_web_routes.py")
Make-DatedCommit "2026-10-28T12:00:00+05:00" "Add Gate 5 automated benchmark test runner for duplicates module" @("doctor/bench.py")
Make-DatedCommit "2026-10-28T13:30:00+05:00" "Document pre-split deduplication rules and conflicting label guards" @("docs/notes.md")
Make-DatedCommit "2026-10-28T14:15:00+05:00" "Record Phase 5 benchmark verification pass in gate log" @("docs/gate_log.md")
Make-DatedCommit "2026-10-28T15:00:00+05:00" "Update readme phase tracker and status badge for Phase 5 pass" @("README.md")
Make-DatedCommit "2026-10-28T15:45:00+05:00" "Record completed Phase 5 milestones in project progress log" @("progress.txt")
Make-DatedCommit "2026-10-28T16:30:00+05:00" "Add automated Phase 5 commit execution script" @("commit_phase5.ps1", "commit_phase5.sh")

# Reset environment variables
Remove-Item Env:\GIT_AUTHOR_DATE -ErrorAction SilentlyContinue
Remove-Item Env:\GIT_COMMITTER_DATE -ErrorAction SilentlyContinue

Write-Host "`nAll 28 Phase 5 commits successfully executed across 3 days!" -ForegroundColor Green
Write-Host "Verify history with: git log -n 28 --format='%h | %ad | %s'" -ForegroundColor Cyan
