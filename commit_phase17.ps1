# PowerShell script to create scheduled commits for Phase 17 (Model Explainability & Feature Attribution)
# Dates: Strictly scheduled for the next two days after Phase 16 (Nov 11 and Nov 12, 2026)
# Total Commits: 4 commits

$ErrorActionPreference = "Stop"

function Make-Commit {
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
    Write-Host "Committed: $Message ($Date)"
}

# 1. Nov 11, 2026 (Day 1 - Morning): Planted Dataset & Truth Generator
Make-Commit -Date "2026-11-11T11:30:00+05:00" `
    -Message "feat(phase17): add planted explainability dataset and ground-truth generator" `
    -Files @("data/planted/explainability.csv", "data/planted/explainability_truth.json", "tools/make_planted.py")

# 2. Nov 11, 2026 (Day 1 - Afternoon): Core Explainability & Attribution Engine
Make-Commit -Date "2026-11-11T16:45:00+05:00" `
    -Message "feat(phase17): implement validation-only permutation importance and non-causal disclosures" `
    -Files @("doctor/explainability.py", "tests/test_explainability.py")

# 3. Nov 12, 2026 (Day 2 - Morning): Web UI & Interactive Workbench
Make-Commit -Date "2026-11-12T11:15:00+05:00" `
    -Message "feat(web): integrate interactive explainability workbench, pruning advisor, and web routes" `
    -Files @("app/routes/issues.py", "app/templates/module.html", "tests/test_web_routes.py")

# 4. Nov 12, 2026 (Day 2 - Afternoon): Registry, Benchmarks, Documentation & Scripts
Make-Commit -Date "2026-11-12T16:30:00+05:00" `
    -Message "docs: register gate 17, update gate log, progress tracker, and master benchmarks" `
    -Files @("doctor/registry.py", "doctor/bench.py", "docs/gate_log.md", "progress.txt", "README.md", "tests/test_registry.py", "commit_phase17.ps1", "commit_phase17.sh")

Write-Host "All 4 Phase 17 scheduled commits completed successfully!"

