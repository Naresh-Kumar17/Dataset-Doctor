# Git Branching, Phase Completion, and Pull Request Rules

Follow this Git workflow throughout the entire project. These rules are mandatory for every development phase.

## 1. Main Branch Protection
- Treat `main` as the stable, working version of the project.
- Never push directly to `main`.
- Never merge any Pull Request into `main` without my explicit approval.
- Do not skip testing, code review, or phase verification.

## 2. Create a New Branch for Every Phase
- Before starting a new phase, ensure the previous phase has been merged into `main`.
- Pull the latest changes from `main`.
- Create a separate branch for the new phase using a clear name, such as:
  - `phase/01-project-setup`
  - `phase/02-core-logic`
  - `phase/03-user-interface`
  - `phase/04-testing`
- Keep each phase's changes isolated from other unfinished work.

## 3. Complete and Test Each Phase
Before declaring a phase complete:
- Finish all requirements for that phase.
- Build the project and run the relevant tests.
- Check for compilation errors, runtime errors, regressions, and integration issues.
- Verify that existing functionality still works.
- Review the changes and ensure no unrelated files or accidental changes are included.
- Report what was implemented, which files changed, what tests were run, and their results.
- Clearly identify any incomplete requirements or known issues. Never claim that a test passed if it was not actually run.

## 4. Push the Branch and Create a Pull Request
After the phase passes its checks:
- Commit the changes with a clear, descriptive commit message.
- Push the phase branch to GitHub.
- Create a Pull Request targeting `main`.
- Include a summary of changes, test results, and a checklist of completed requirements.
- Do not merge the PR automatically.

## 5. Stop and Wait for My Approval
Once the Pull Request is ready:
- Tell me that the phase is ready for review.
- Provide the PR link, a summary of changes, and the test results.
- Stop development on the next phase and wait for me to review and test the current phase.
- Do not merge, close, or bypass the PR without my explicit permission.

I will test the phase and review its implementation. If I request changes, make the corrections on the same branch, rerun the relevant tests, push the updates, and ask me to review again.

## 6. Merge Only After Explicit Approval
Only after I explicitly approve the phase:
- Merge the approved PR into `main`.
- Confirm that the merge succeeded.
- Update the local `main` branch.
- Verify that the merged version builds and works correctly.
- Only then create a new branch for the next phase.

If testing fails, the PR has unresolved issues, or approval has not been given, stop and fix the problem instead of proceeding.

## 7. Important Restrictions
- Never bypass this workflow to save time.
- Never start the next phase on top of unapproved changes.
- Never force-push or rewrite shared branch history without explicit permission.
- Do not modify unrelated modules while completing a phase.
- If a requirement is unclear, a test cannot run, or a Git operation could overwrite work, stop and explain the issue before proceeding.
- Do not claim that a branch was pushed, a PR was created, or a merge was completed unless the operation actually succeeded.

**Core rule: One phase → one branch → implementation and testing → Pull Request → my review and testing → my explicit approval → merge into `main` → next phase.**

Follow this process for every phase until the entire project is complete.
