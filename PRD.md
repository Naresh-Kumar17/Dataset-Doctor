# Dataset Doctor: Product Requirements Document (PRD)

Version: 0.2 | Status: Draft | Owner: You

## 1. One-line summary
Upload a CSV. The tool finds the data problems that quietly ruin ML models, **recommends a fix with reasons and test results, asks you to decide**, applies it safely, and shows the before/after proof.

## 2. Problem
- Most ML projects fail from bad data or bad evaluation, not bad models.
- Common silent mistakes:
  - Balancing classes before the train/test split (inflated score).
  - Judging an imbalanced problem by accuracy only.
  - A column that secretly leaks the answer (data leakage).
  - Duplicates that appear in both train and test.
  - Filling or deleting missing values with no evidence that the choice is good.
- Existing tools (profilers, validators) list problems. They do not prove whether a fix helps, and they do not help you choose between fixes.

## 3. Goal
- Primary: a correct, honest tool that measures the real effect of each data fix.
- Personal goal: learn correct evaluation, leakage detection, and deployment.
- Non-goal: winning a pitch. No deadline.

## 4. Target user
- Students and junior data scientists who train models on CSV files.
- They know basic pandas and scikit-learn, but do not know what they are getting wrong.

## 5. Core principles (non-negotiable)
- **Split first.** Test set is never touched, never resampled, never loses rows, keeps the real class ratio.
- **Fix training data only.** Anything learned from data (fill values, clip bounds, resampling) is fit on train and only applied to test.
- **Never judge by accuracy alone.** For imbalanced targets, report minority-class recall, precision, F1, PR-AUC.
- **Advise, then ask.** For every issue the tool recommends a method, gives a reason, shows measured scores for the alternatives, and the user decides. Nothing is changed silently.
- **Rules are guesses, measurements are evidence.** Show both. Say "close call" when scores are within noise. Say "not enough data" when the dataset is small.
- **Detect, advise and fix are separate.** Each lives in its own file so each can be tested alone.
- **Say when unsure.** Show confidence and limits, do not overclaim.

## 6. How the work is delivered: phases with gates
- The project is built in small phases (see IMPLEMENTATION.md).
- Each phase ends with a **gate**: run the module on data with **planted problems** where the right answer is known.
- Pass: continue to the next phase.
- Fail: run the debug loop (find where and why it was wrong), fix, run the gate again.
- Earlier gates are re-run after every phase to catch breakage.
- A **web GUI test bench is built first** (Phase 1) so you can upload any file and see what each finished module does.

## 7. Features

### MVP (must have)
| ID | Feature | Detail | Phase |
|----|---------|--------|-------|
| F1 | CSV upload and profile | Max 50 MB, common encodings, preview, column types | 1 |
| F2 | Test bench page | Planted vs found table per module, module status | 1 |
| F3 | Null detection | NaN, hidden tokens (`?`, `N/A`, `-`), sentinels (`-999`), row loss estimate, missingness vs target | 2 |
| F4 | Split + evaluation engine | Stratified split, fixed model, CV with noise estimate, minority metrics | 3 |
| F5 | Null advisor + fixer | Recommends drop / mean / median / mode / label / indicator, reasons, measured scores, user chooses | 4 |
| F6 | Duplicate detect + fix | Exact, near, conflicting-label duplicates, pre-balanced file warning | 5 |
| F7 | Type detect + fix | Text numbers, ambiguous dates, mixed types, constant and ID-like columns | 6 |
| F8 | Outlier detect | IQR, MAD, percentile, impossible values, unit mix-ups | 7 |
| F9 | Outlier advisor + fixer | Leave, clip, log, drop train rows, set to null, with measured effect | 8 |
| F10 | Imbalance detect + advise + fix | Class weights, oversample, SMOTE, undersample, threshold | 9 |
| F11 | Leakage detect + advise | Near-perfect single features, target copies, ID leaks, time-split warning | 10 |
| F12 | Full pipeline + before/after | Correct step order, noise label, decision log | 11 |
| F13 | Report + reproduce script + cleaned CSV | Markdown or HTML report and a script that repeats the accepted steps | 11 |
| F14 | Deployment | Free host with lower file limit | 11 |

### Later (nice to have)
- Time-series mode (time-ordered split, future-data leakage).
- KNN and iterative imputation.
- Optional LLM-written plain-language explanations (off by default).
- Model choice beyond the fixed one.

### Out of scope
- Deep learning, image or text datasets.
- User accounts, databases, multi-user storage.
- Auto-pilot cleaning with no confirmation.

## 8. User flow
1. Upload CSV and see the profile.
2. Pick target column (task type auto-detected, editable).
3. See the module list. Each module shows issues found, with severity and certainty.
4. For each issue, open the **decision card**: evidence, recommended method with reason, table of tested alternatives, preview.
5. Accept, change, or skip. Each choice is logged.
6. Apply all decisions in the correct order.
7. See before/after metrics on the untouched test set.
8. Download report, cleaned CSV and `reproduce.py`.

## 9. Example: the null decision
Column `income`, 12% missing, skewed:
- Evidence: 120 of 1000 missing, skew 2.3, missingness not linked to target.
- Recommended: **median**. Reason: the mean is pulled up by a few very high incomes.
- Tested (5-fold CV, train only): median F1 0.71 ± 0.02, mean 0.69 ± 0.02, drop rows 0.70 ± 0.03 (880 rows kept).
- Confidence: close call between median and drop rows; median recommended because it keeps all rows.
- User: Apply recommended / Choose another / Skip.

## 10. Success criteria
- Every module passes its gate on planted data (recall of planted problems at target, zero false alarms on the clean control).
- On the road-accident dataset, shows the gap between the old (leaky) score and the correct score.
- Never shows accuracy as the main metric when the minority share is under 20%.
- Null advisor matches the known best method in at least 4 of 5 planted scenarios.
- No fix ever changes the test set except filling missing cells using train-learned values (asserted in tests).
- Runs end to end on a 50 MB file in under 2 minutes on a normal laptop.
- At least 80% test coverage on the issue modules.

## 11. Test datasets
- Your road accident severity data (85/15, known oversample-before-split mistake).
- Your air/weather forecasting data (time-ordered, leakage risk).
- 3 Kaggle datasets with known problems.
- **Planted datasets** made by `tools/make_planted.py` with a truth file per module, plus a clean control with tricky legitimate values (country code "NA", real zeros).

## 12. Risks
| Risk | Impact | Plan |
|------|--------|------|
| Rule-based advice is wrong | Bad cleaning | Always test alternatives; show both rule and evidence; user decides |
| Measured differences are noise | False confidence | Show std, label "close call", warn on small data |
| Hidden nulls missed (`?`, `-999`) | Silent bad data | Token list plus `possible` flags; tested in Gate 2 |
| Leakage detection false alarms | Loses trust | Show evidence, allow dismiss, never auto-drop |
| Auto-fixes damage data | Harm | Never silent, preview before apply, log every decision |
| Planted tests too easy | Passing gate means little | Add a harder case after each pass; use real datasets in Phase 11 |
| Lowering gate thresholds to pass | Fake progress | Any loosening must be written in `docs/gate_log.md` with a reason |
| Scope grows | Never finishes | MVP only; no new module until the current gate passes |
| Weak web skills | Slow UI | Thin UI, all logic in `doctor/`, GUI shell built first |

## 13. Competition check (Phase 0)
- Look at ydata-profiling, Great Expectations, Deepchecks.
- Write down what Dataset Doctor adds: **a recommendation you can argue with, measured before/after effect of each fix, with correct evaluation, and the user in control.**
- If one of them already does this well, change the angle before Phase 3.

## 14. Milestones
Day estimates are in IMPLEMENTATION.md section 8. They are guesses; replace them with your real pace after Phase 2.
- Phase 0-1: skeleton, evidence, GUI test bench
- Phase 2-4: nulls end to end (detect, evaluate, advise, fix) is the **first full vertical slice**
- Phase 5-10: remaining issue modules, one at a time
- Phase 11: pipeline, report, deploy
