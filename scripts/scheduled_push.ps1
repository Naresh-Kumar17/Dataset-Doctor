<#
.SYNOPSIS
  Automated git push scheduler for Dataset Doctor 3-day release rollout.
  - Day 1 (Nov 26, 2026): Pushes first 3 commits (Intake Stepper, ELI5 Explainer, Letter Grades)
  - Day 2 (Nov 27, 2026): Pushes next 3 commits (Cure All Hero, Tour Modal, Tour Tests)
  - Day 3 (Nov 28, 2026): Pushes rest 2 commits (Roadmap Docs, v2.1 Certification)

.PARAMETER Day
  Explicit day to push (1, 2, or 3). If omitted, automatically checks current system date.

.PARAMETER DryRun
  Show what would be pushed without executing the push.
#>
param(
    [ValidateSet(1, 2, 3)]
    [int]$Day = 0,
    [switch]$DryRun
)

$commitDay1 = "263cf22" # Day 1 (3rd commit: Letter Grades)
$commitDay2 = "f75aca2" # Day 2 (6th commit: Tour Tests)
$commitDay3 = "main"    # Day 3 (all commits)

if ($Day -eq 0) {
    $today = Get-Date -Format "yyyy-MM-dd"
    if ($today -le "2026-11-26") {
        $Day = 1
    } elseif ($today -eq "2026-11-27") {
        $Day = 2
    } else {
        $Day = 3
    }
}

Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "  🩺 DATASET DOCTOR: 3-DAY RELEASE PUSH SCHEDULER" -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan

switch ($Day) {
    1 {
        Write-Host "📅 DAY 1 (Nov 26, 2026): Releasing 3 Commits Today" -ForegroundColor Green
        Write-Host "  1. feat(ui): add universal 4-step patient consultation stepper header"
        Write-Host "  2. feat(ux): implement Plain-English ELI5 explainer engine and toggleable UI guides"
        Write-Host "  3. feat(quality): add letter grade classification and layman clinical status badges"
        $target = "$commitDay1`:main"
    }
    2 {
        Write-Host "📅 DAY 2 (Nov 27, 2026): Releasing Next 3 Commits" -ForegroundColor Yellow
        Write-Host "  4. feat(remediation): implement 1-click Prescribe and Cure All engine and UI hero bar"
        Write-Host "  5. feat(onboarding): add interactive How It Works 30-second tour modal carousel"
        Write-Host "  6. test(onboarding): add automated web and integration tests for tour carousel"
        $target = "$commitDay2`:main"
    }
    3 {
        Write-Host "📅 DAY 3 (Nov 28, 2026): Releasing Remaining 2 Commits" -ForegroundColor Magenta
        Write-Host "  7. docs(roadmap): create 3-day scheduled user-friendly release roadmap"
        Write-Host "  8. chore(release): finalize user-friendly v2.1 certification and benchmark pass"
        $target = "main"
    }
}

Write-Host "`nTarget push refspec: git push origin $target" -ForegroundColor Gray

if ($DryRun) {
    Write-Host "[DRY RUN] Executing: git push --dry-run origin $target" -ForegroundColor Yellow
    git push --dry-run origin $target
} else {
    Write-Host "[PUSHING] Executing: git push origin $target ..." -ForegroundColor Green
    git push origin $target
    if ($LASTEXITCODE -eq 0) {
        Write-Host "`nSUCCESS: Day $Day commits successfully pushed to GitHub!" -ForegroundColor Green
    } else {
        Write-Host "`nERROR: Push failed with exit code $LASTEXITCODE" -ForegroundColor Red
    }
}
