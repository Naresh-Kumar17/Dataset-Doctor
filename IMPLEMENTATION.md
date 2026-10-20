# Dataset Doctor: Implementation Plan (v0.2, phased with gates)

## 1. How to use this document

Work in **phases**. Each phase has a **gate**. The gate is a test you run against data where you already know the right answer.

```
Build the phase  ->  Run the gate  ->  PASS: go to next phase
                                   ->  FAIL: debug loop (section 6), run gate again
```

Rules:
- Never start a phase until the previous gate is PASS.
- When a later phase breaks an earlier gate, fix it first. Run all earlier gates every time (`python -m doctor.bench --all`).
- Write every gate attempt in `docs/gate_log.md` (date, result, what failed, what you changed). This is your plan tracker and your learning record.
- Day estimates are guesses. Replace them with your real pace after Phase 2.

## 2. How the tool works (the flow)

```
Upload CSV
   |
   v
READ     loader.py + profile.py      (encoding, types, size, preview)
   |
   v
DETECT   issues/<name>/detect.py     (find problems, show evidence, change nothing)
   |
   v
ADVISE   issues/<name>/advise.py     (pick likely methods by rules, then TEST each on the training data)
   |
   v
ASK      GUI decision card           (user accepts, picks another, or skips)
   |
   v
FIX      issues/<name>/fix.py        (fit on train only, apply to train and test)
   |
   v
RE-TEST  evaluate.py + compare.py    (same untouched test set, before vs after)
```

Key idea: **detect, advise, fix are three separate files for every issue type.** So you can build and test each one alone.

Why FastAPI + plain HTML/CSS and not Streamlit: neo-brutalism needs full control of CSS, and Streamlit fights custom styling. Trade-off: more web code. Keep pages few and simple.

## 3. Folder structure

```
dataset-doctor/
  doctor/                        # all data science, NO web code
    __init__.py
    base.py                      # Detector / Advisor / Fixer interfaces, Issue, Recommendation, Decision
    registry.py                  # list of modules + status; the GUI reads this
    loader.py                    # read CSV, detect encoding, size limit, keep_default_na=False
    profile.py                   # column types, basic stats, preview
    split.py                     # the ONLY place that splits data
    evaluate.py                  # fixed model, CV, metrics, noise estimate
    compare.py                   # before/after tables
    pipeline.py                  # runs accepted decisions in the correct order
    report.py                    # report + reproduce.py export
    bench.py                     # gate runner: python -m doctor.bench nulls
    issues/
      nulls/
        detect.py                # find missing values and hidden missing values
        advise.py                # recommend drop / mean / median / mode / etc., with reasons and measured scores
        fix.py                   # apply the chosen method
        explain.py               # plain-language text shown in the GUI
      duplicates/                # detect.py  advise.py  fix.py  explain.py
      types/                     # detect.py  advise.py  fix.py  explain.py
      outliers/                  # detect.py  advise.py  fix.py  explain.py
      imbalance/                 # detect.py  advise.py  fix.py  explain.py
      leakage/                   # detect.py  advise.py  fix.py  explain.py
  app/
    main.py
    session.py                   # in-memory sessions
    routes/
      upload.py  issues.py  decide.py  bench.py  report.py
    templates/
      base.html  index.html  shell.html  target.html
      module.html  decision_card.html  bench.html  results.html
    static/
      style.css  app.js
  tools/
    make_planted.py              # creates planted-problem test data + truth files
  data/
    samples/                     # tiny real CSVs
    planted/                     # <name>.csv + <name>.truth.json + clean_control.csv
  tests/
    issues/
      test_nulls_detect.py  test_nulls_advise.py  test_nulls_fix.py
      test_duplicates.py  test_types.py  test_outliers.py
      test_imbalance.py  test_leakage.py
    test_split.py  test_evaluate.py  test_pipeline.py
  experiments/
    first_test.py                # 2-day test
    accident_redo.py             # Phase 0 evidence
  docs/
    notes.md                     # 3 lines per module in your own words
    gate_log.md                  # every gate attempt
  requirements.txt
  README.md  PRD.md  IMPLEMENTATION.md  SETUP.md  DESIGN.md
```

## 4. Interfaces (doctor/base.py)

```python
from dataclasses import dataclass, field

@dataclass
class Issue:
    id: str                  # "nulls", "dup_rows", ...
    column: str | None
    severity: str            # "high" | "medium" | "low"
    certainty: str           # "certain" | "possible"
    title: str
    evidence: dict           # numbers that prove it
    affected_rows: list[int] = field(default_factory=list)

@dataclass
class Option:
    id: str                  # "median", "drop_rows", ...
    label: str
    params: dict
    explanation: str         # plain English, 1-2 lines
    measured: dict | None    # {"f1_mean": .., "f1_std": .., "n_rows_kept": ..}

@dataclass
class Recommendation:
    issue: Issue
    recommended: str                 # option id
    reason: str                      # why, in plain English
    rule_hits: list[str]             # which rules fired
    options: list[Option]            # all tested options, ranked
    confidence: str                  # "clear" | "close call" | "not enough data"

@dataclass
class Decision:
    issue_id: str
    column: str | None
    option_id: str
    params: dict
    decided_by: str                  # "user" | "recommended_accepted"
    timestamp: str
```

Every issue module implements:

```python
class Detector:
    id: str
    def detect(self, df, ctx) -> list[Issue]: ...          # read-only

class Advisor:
    def advise(self, issue, train_df, target, ctx) -> Recommendation: ...   # uses TRAIN data only

class Fixer:
    def fit(self, train_df, decision) -> "FittedFix": ...  # learns values from TRAIN only
    def transform(self, df, fitted) -> "DataFrame": ...    # applies to train or test
    rows_may_be_dropped_from = "train"                     # test rows are never dropped
```

`ctx` holds: target column, task type, minority share, random seed, dataset size.

## 5. Safety rules in code (non-negotiable)

- **Pre-split steps** are only deterministic ones that learn nothing from the data: normalize hidden-null tokens to NaN, parse text numbers, remove exact duplicates.
- **Post-split steps** are everything that learns a number from data: imputation values, clip bounds, scalers, resampling. They are fit on train and only applied to test.
- The test set is never resampled and never loses rows. A row-dropping fix is applied to train only. Test cells that are still missing get the train-fitted fallback value.
- Rows with a missing **target** are dropped before the split (mandatory, shown to the user).
- `split.py` is the only file that splits. Everything else receives `train` and `test`.
- Detect modules may read the whole file because they change nothing. Advise and fix modules never see the test set.
- Same random seed and same test set for every comparison.
- Imbalanced target (minority < 20%): primary metrics are minority recall, precision, F1, PR-AUC. Accuracy is shown with the label "not reliable here".

## 6. The gate system

### 6.1 Planted data
`tools/make_planted.py` builds, for every module, a CSV where **you decide the problems in advance**, plus a truth file:

```json
{
  "module": "nulls",
  "rows": 1000,
  "planted": [
    {"column": "age",    "kind": "nan",          "count": 50},
    {"column": "phone",  "kind": "hidden_token", "token": "N/A", "count": 60},
    {"column": "score",  "kind": "sentinel",     "token": "-999", "count": 40},
    {"column": "notes",  "kind": "nan",          "count": 700}
  ],
  "must_not_flag": ["country", "id"]
}
```

`data/planted/clean_control.csv` has no real problems but has tricky legitimate values (for example country code "NA" for Namibia, a real 0 reading). A good detector stays quiet on it.

### 6.2 Gate runner
```bash
python -m doctor.bench nulls          # one module
python -m doctor.bench --all          # every finished module (regression check)
```
It prints: planted vs found, missed, false alarms, PASS or FAIL. The GUI Test Bench (Phase 1) shows the same table. Results are saved to `bench_results.json`.

### 6.3 Debug loop (when a gate fails)
1. **Classify the failure:** missed problem (false negative), or false alarm (false positive)?
2. **Shrink it:** copy the failing column into a 10-row DataFrame in a test. Reproduce it there.
3. **Print the evidence** the detector produced. Is the logic wrong or the input different from what you assumed?
4. **Common causes:**
   - Hidden null tokens: `"?"`, `"-"`, `"NA"`, `"n/a"`, `"none"`, `"null"`, whitespace-only, `-999`, `9999`.
   - pandas silently turned `"NA"` into NaN on load (use `keep_default_na=False`).
   - Threshold too strict or too loose.
   - Column type misread (numbers stored as text).
   - Check ran after another fix changed the data.
5. **Fix, then add a regression test** that fails before the fix and passes after.
6. **Re-run this gate and all earlier gates.**
7. Log it in `docs/gate_log.md`.
8. If a gate still fails after 3 loops, stop. Ask: is the gate unrealistic, or is the approach wrong? Change one of them deliberately. Do not just lower the threshold.

## 7. Phases

Gate thresholds are targets. Tighten them if too easy. Never loosen them without writing down why.

---

### Phase 0: Setup and evidence (about 5-7 days)

**Goal:** a working project skeleton and one real piece of evidence for the whole project.

**Build**
- Project folders, venv, requirements, `app/main.py` hello route (see SETUP.md).
- `experiments/first_test.py` (2-day test).
- `experiments/accident_redo.py`: redo the road-accident model with **split first, then fix on train only**. Record old vs new recall and F1 for the severe class.
- Competition check: read about ydata-profiling, Great Expectations, Deepchecks. Write 5 lines in `docs/notes.md` on what they do not do.

**Gate 0**
- [ ] `pytest -q` runs (even with zero tests) and the hello route opens in the browser.
- [ ] Old (oversample before split) and new (split first) numbers recorded in README.
- [ ] Test set in the redo keeps the real ratio (about 85/15).
- [ ] You can explain in 3 lines why the old score was inflated.

**If it fails:** if the old vs new numbers are nearly equal, check that your old code really oversampled before the split. If they are equal after checking, write that down too. It is still a result.

---

### Phase 1: GUI shell + loader + profile (about 3-4 days)

**Goal:** a working web page where you upload any CSV and see what the tool understood. This is also your **test bench** for every later phase.

**Build**
- `doctor/loader.py`: read CSV, guess encoding (chardet), 50 MB limit, `keep_default_na=False`, friendly errors.
- `doctor/profile.py`: rows, columns, dtype guess, unique count, sample values, memory.
- `doctor/registry.py`: list of all planned modules with a status: `not_built`, `built`, `gate_passed`.
- `app/main.py` + routes: `/` upload, `/upload`, `/profile`, `/bench`.
- Templates: `base.html`, `index.html`, `shell.html` (left sidebar of modules, main area), `bench.html`.
- Style from DESIGN.md (basic card, button, table only).
- Every module appears in the sidebar as a card with status **NOT BUILT**. Later phases turn them on without changing the shell.

**Gate 1**
- [ ] Upload the accident CSV: row count, column count, dtypes match `pandas` in a notebook exactly.
- [ ] Upload a file with Latin-1 characters: loads or shows a clear error.
- [ ] Upload a 60 MB file: rejected with a clear message.
- [ ] Upload an empty file and a file with only headers: clear message, no crash.
- [ ] `/bench` page opens and lists all modules with status.
- [ ] Works at phone width (375px).

**If it fails:** most failures here are encoding, delimiter (`;` vs `,`) or a header-row problem. Print `df.head()` and `df.dtypes` and compare with a plain pandas read.

---

### Phase 2: Null detection (about 3-4 days)

**Goal:** find every missing value, including hidden ones. Change nothing.

**Files:** `issues/nulls/detect.py`, `issues/nulls/explain.py`, `tests/issues/test_nulls_detect.py`

**Detector must report, per column**
- Count and percent of true NaN.
- Hidden missing tokens: `""`, whitespace-only, `?`, `-`, `NA`, `N/A`, `n/a`, `null`, `none`, `nan`, case-insensitive.
- Sentinel numbers: `-999`, `-1`, `9999`, `0` only as **possible**, never certain (a real 0 reading is common).
- Row-level: how many rows have 1, 2, 3+ missing cells. How many rows would be lost if all rows with any null were dropped.
- Missing in the **target** column (high severity).
- **Missingness pattern:** is the "is missing" flag related to the target? (chi-square for classification, point-biserial or t-test for regression.) Is it related to another column's missingness? Report as evidence only.
- Severity: high if target missing, or column > 60% missing; medium 5-60%; low under 5%.
- `certainty`: `certain` for true NaN and clear tokens, `possible` for ambiguous tokens (`NA`, `0`, `-1`).

**Planted data:** `data/planted/nulls.csv` (see 6.1) and `clean_control.csv`.

**Gate 2**
- [x] 100% of planted missing cells found (hidden tokens and sentinels included, counts exact for certain tokens).
- [x] 0 false alarms on `clean_control.csv`. Country code "NA" is flagged at most as `possible`.
- [x] Row-loss number is exact (compare with `df.dropna().shape`).
- [x] Planted target-related missingness is reported; planted random missingness is not.
- [x] GUI shows the module card with a table: column, missing %, hidden tokens found, severity.
- [x] Runs in under 5 seconds on 100,000 rows.


**If it fails:** go through the debug loop. Most likely problem: hidden tokens, or pandas already converted them. Test each token type alone.

---

### Phase 3: Split + evaluation engine (about 4-5 days)

**Goal:** an honest way to measure. The advisors in Phase 4+ depend on it, so it comes before them.

**Files:** `doctor/split.py`, `doctor/evaluate.py`, `doctor/compare.py`, `tests/test_split.py`, `tests/test_evaluate.py`

**Build**
- `split.py`: stratified 80/20 for classification, plain random for regression, fixed seed, stores the row indexes so the split can be locked and shown in the GUI. Returns `train, test`. Drops rows with missing target first and reports how many.
- `evaluate.py`:
  - Fixed model: RandomForest, 200 trees, fixed seed (plus a fast mode with fewer trees for advisor tests).
  - Simple built-in preprocessing so a raw dataframe can be scored (one-hot encode categoricals, so models run before any user fix).
  - 5-fold stratified CV on **train** (mean and standard deviation) and one final score on the untouched **test**.
  - Metrics: accuracy, precision, recall, F1 (minority class), PR-AUC. Primary metric chosen by minority share.
  - A `is_significant(before, after)` helper: a change smaller than the CV spread is "within noise".
- `compare.py`: before/after table with delta and the noise label.
- The **split-order demo** function: score with "balance before split" vs "split first" on the same data.

**Gate 3**
- [ ] Stratified split keeps the class ratio within 1% on both parts.
- [ ] Same seed gives the same split. Different seed gives a different one.
- [ ] No row appears in both train and test (check with row hashes).
- [ ] Planted check: oversampling before the split gives a **higher** test recall than split-first on the same data. If this ever flips, the engine is broken.
- [ ] Always-predict-majority model on the accident data shows about 85% accuracy and 0 minority recall in the table, and the GUI shows the "accuracy not reliable" banner.
- [ ] A change smaller than the CV std is labelled "within noise".
- [ ] Dataset under 500 rows shows a "noisy comparison" warning.

**If it fails:** if the demo does not show the inflated score, the planted data may have too few duplicated minority rows, or too simple a pattern. Make a harder version before blaming the engine.

---

### Phase 4: Null advisor + fixer, interactive (about 5-6 days)

**Goal:** for each column with missing values, the tool **recommends a method, explains why, tests the alternatives, asks the user, then applies the choice.**

**Files:** `issues/nulls/advise.py`, `issues/nulls/fix.py`, `issues/nulls/explain.py`, `app/routes/decide.py`, `templates/decision_card.html`, tests `test_nulls_advise.py`, `test_nulls_fix.py`

#### 4.1 Options the fixer supports
| Option id | What it does |
|-----------|--------------|
| `drop_rows` | remove rows with a null in this column (train only) |
| `drop_column` | remove the column |
| `mean` | fill with the train mean (numeric) |
| `median` | fill with the train median (numeric) |
| `mode` | fill with the most common train value |
| `constant` | fill with a value the user types (for example "Unknown" or 0) |
| `missing_label` | categorical: new category "MISSING" |
| `ffill_interpolate` | time-ordered data: carry forward or interpolate |
| `knn` | fill from similar rows (later, optional) |
| `add_indicator` | add a 0/1 column "was_missing", can combine with any fill |
| `leave` | do nothing (for models that handle NaN) |

#### 4.2 Step A: rule-based shortlist (the reasoning)
| Situation | Shortlist | Reason shown to the user |
|-----------|-----------|--------------------------|
| Target is missing | `drop_rows` (forced) | cannot learn from a row with no answer |
| Column > 60% missing | `drop_column`, `add_indicator` | too little real data to fill honestly |
| Missing in < 5% of rows, unrelated to target, rows are plentiful | `drop_rows`, `median`/`mode` | little data lost either way |
| Numeric, skewed (abs skew > 1) or has outliers | `median` first, `mean` second | mean gets pulled by extreme values |
| Numeric, roughly symmetric | `mean`, `median` | both give similar values |
| Categorical or text | `mode`, `missing_label` | no average exists |
| Missingness is linked to the target | `add_indicator` + a fill; **do not** drop rows | the fact that it is missing carries signal, and dropping would bias the data |
| Time-ordered data | `ffill_interpolate` | neighbors in time are the best guess |
| ID-like column | none | do not fill IDs |
| Dropping rows would lose > 20% of the data | remove `drop_rows` from shortlist | too costly |

#### 4.3 Step B: measure (the evidence)
- For each shortlisted option, run CV on **train only** and record the primary metric mean and std, plus rows kept.
- Rank by score. If the best two differ by less than the std, confidence is **close call** and the simpler option is recommended (order of simplicity: `leave` < `median`/`mode` < `mean` < `constant`/`missing_label` < `add_indicator` combos < `knn` < `drop_rows` < `drop_column`).
- Dataset under 500 rows: confidence is **not enough data**; fall back to the rule-based choice and say so.
- Row-dropping is scored fairly: the final test set is the same untouched one, so test nulls are filled with the train-fitted fallback value.

#### 4.4 Step C: ask the user (decision card)
The GUI shows, per issue:
- What was found (evidence numbers).
- **Recommended:** option + reason + rules that fired.
- Table of all tested options with score, std and rows kept.
- Buttons: **APPLY RECOMMENDED**, **CHOOSE ANOTHER**, **SKIP**.
- **PREVIEW**: sample of 10 rows before and after.
- Option **"Apply recommended to all columns"** shows the whole plan first and needs one confirmation.
- Every choice is saved to the decision log with who decided.

#### 4.5 Step D: apply
- `fit()` learns values from train only. `transform()` applies them to train and test.
- Decision log feeds the final report and the exported `reproduce.py`.

**Gate 4A: advisor**

Plant 5 scenarios with a known right answer:
| Scenario | Expected recommendation |
|----------|-------------------------|
| Numeric, 3% random missing, symmetric | `mean` or `median` (either accepted) |
| Numeric, 10% missing, heavily skewed | `median` |
| Categorical, 5% missing | `mode` or `missing_label` |
| Column 75% missing | `drop_column` or `add_indicator` |
| Missingness depends on target | includes `add_indicator`, not `drop_rows` |

- [ ] At least 4 of 5 recommendations are as expected.
- [ ] Every recommendation shows a reason and measured scores.
- [ ] "Close call" appears when scores are within noise.

**Gate 4B: fixer**
- [ ] After the fix: 0 nulls remain in train and test in that column (except `leave`).
- [ ] Test row count is unchanged.
- [ ] Test values changed only in cells that were null.
- [ ] Learned values come from train: build data where train median differs from full-data median; the fill value must equal the train median.
- [ ] Decision log contains the choice, and `reproduce.py` rebuilds the same result.
- [ ] A user override changes the applied method.

**If it fails:** if recommendations disagree with the planted truth, first check if the measured scores are noise (print the std). If rules are wrong, change the rule. If the measurement is noise, the dataset is too small or the CV too weak. Do not tune rules until all five pass by luck.

---

### Phase 5: Duplicates, detect + fix (about 3 days)

**Files:** `issues/duplicates/detect.py`, `advise.py`, `fix.py`, `explain.py`

**Detect**
- Exact duplicate rows (count, percent, which rows).
- Near duplicates (differ only by case, spaces or tiny number differences): `possible`.
- **Conflicting duplicates:** same features, different target. High severity.
- Duplicate IDs in an ID-like column.
- **Pre-balanced file warning:** target classes exactly 50/50 and many repeated minority rows. This suggests the file was oversampled before saving, the same mistake from your accident project.

**Fix options (before the split)** `keep_first`, `keep_last`, `drop_all_conflicting`, `leave`
**Advise:** exact duplicates default to `keep_first`. Conflicting duplicates are shown to the user, never auto-removed. Duplicates removed before the split so no row can be in both train and test.

**Gate 5**
- [ ] Planted: 50 exact, 20 near, 10 conflicting pairs. All exact and conflicting found; near ones flagged as `possible`.
- [ ] 0 false alarms on control (rows that look similar but differ in real values).
- [ ] After the fix, no duplicates remain and row count equals original minus planted exact duplicates.
- [ ] The pre-balanced warning fires on a file built by oversampling to 50/50, and does not fire on the control.

---

### Phase 6: Data types (about 4 days)

**Files:** `issues/types/detect.py`, `advise.py`, `fix.py`, `explain.py`

**Detect**
- Numbers stored as text: `"1,200"`, `"$50"`, `"45%"`, `"12 kg"`.
- Dates stored as text (and **ambiguous** formats such as `03/04/2021`, which could be day-first or month-first).
- Mixed type columns (numbers plus words).
- Yes/no style columns (`Y/N`, `yes/no`, `true/false`, `0/1`).
- Constant columns (one value only).
- ID-like columns (unique ratio > 95%).
- High-cardinality categoricals (very many distinct values).

**Fix options** convert to number (strip symbols), parse dates (asks user for day-first or month-first when ambiguous), encode booleans, drop constant, drop ID-like, leave.
**Advise:** rules only for most cases. Date ambiguity is always a question to the user.

**Gate 6**
- [ ] All planted type problems found, 0 false alarms on control.
- [ ] Ambiguous dates are flagged and the user is asked, never guessed.
- [ ] After conversion, a numeric column has the same values as the planted truth.
- [ ] A categorical column with codes like `001` is not turned into a number by mistake.

---

### Phase 7: Outlier detection (about 3 days)

**Files:** `issues/outliers/detect.py`, `explain.py`

**Detect (numeric columns, flag only)**
- IQR rule (1.5x).
- Modified z-score (MAD) for skewed data.
- Percentile view (below 1%, above 99%).
- **Impossible values** by simple rules: negative age, age above 120, percent above 100, plus user-defined min/max per column (asked in the GUI).
- Possible unit mix-ups (a few values 100x or 1000x the typical value).
- Report per column: count, percent, the values, where they sit, and whether the column is skewed (IQR over-flags skewed data, so say so).
- Severity: high for impossible values, medium for extreme but possible, low otherwise.

**Gate 7**
- [ ] Planted: 10 impossible values, 20 extreme-but-real values, 1 unit error. All 10 impossible found as high; extreme ones as medium or low; unit error flagged.
- [ ] A heavily skewed but clean column does not flag more than 5% of its values as outliers (use MAD or log scale).
- [ ] A normal clean column stays under 1% flagged (the tail you would expect).
- [ ] 0 outlier flags on a binary or categorical column.

---

### Phase 8: Outlier advisor + fixer, interactive (about 3 days)

**Files:** `issues/outliers/advise.py`, `fix.py`

**Options** `leave`, `clip_iqr`, `clip_percentile` (1/99), `drop_rows` (train only), `log_transform`, `set_to_null` (then handed to the null module).

**Rule shortlist**
| Situation | Shortlist | Reason |
|-----------|-----------|--------|
| Impossible values | `set_to_null` (then null advisor) | the value is an error, not real data |
| Extreme but possible and target is rare (like severe accidents) | `leave` first | extremes may be the very cases you want to predict |
| Skewed positive column | `log_transform`, `clip_percentile` | tames the tail without deleting data |
| Tree model in use | `leave` first | tree models are mostly not affected by outliers |
| Few rows, clearly errors | `drop_rows` | small loss |

**Measure:** same CV method as Phase 4. Honest outcome: often the measured effect is zero, so the tool should say "leave it" with evidence. That is a correct answer.
**Clip bounds are learned from train only.**

**Gate 8**
- [ ] Impossible values are recommended as `set_to_null` in all planted cases.
- [ ] When the effect is within noise, the tool says so and recommends `leave`.
- [ ] After a clip, train min/max equal the train-derived bounds; test is clipped using the same bounds (not its own).
- [ ] Rare-class planted case: recommending `drop_rows` on extremes is not chosen when it removes minority cases (check minority count before and after).

---

### Phase 9: Imbalance (about 4 days)

**Files:** `issues/imbalance/detect.py`, `advise.py`, `fix.py`, `explain.py`

**Detect:** class shares, minority share, imbalance ratio. Flag high under 10%, medium under 20%. Also checks that the dataset share may not be the real-world share (asks the user: "is 15% the true rate?").

**Fix options (train only)** `class_weight`, `random_oversample`, `smote`, `random_undersample`, `threshold_tuning`, `leave`
**Advise:** test each by CV on train, compare on minority recall, precision, F1, PR-AUC. Warn that oversampling changes predicted probabilities. Show the trade-off (recall up, precision down). The user decides.

**Gate 9**
- [ ] Planted 90/10, 99/1, multi-class with one tiny class, and a balanced control. Flags correct ones, silent on balanced.
- [ ] After any fix, test class ratio is unchanged (assert).
- [ ] No resampled row appears in test.
- [ ] Accident data: tool shows high accuracy but low minority recall at baseline, and an improvement in recall with at least one fix. Numbers match your Phase 0 redo.
- [ ] SMOTE refuses or warns on data with unencoded categoricals instead of producing garbage.

---

### Phase 10: Leakage (about 4 days)

**Files:** `issues/leakage/detect.py`, `advise.py`, `fix.py`, `explain.py`

**Detect**
- Single-feature score: train a tiny model on one column alone; flag a near-perfect score.
- Very high correlation with the target.
- Column identical or a simple transform of the target (copy, x2, relabel).
- ID-like or index-like columns that predict the target by row order.
- Column names that suggest the future (`outcome`, `result`, `status_after`, `final_`): `possible` only.
- Time data: random split used on time-ordered rows (warning, recommends time-ordered split).

**Fix options** `drop_column`, `keep_and_ignore_warning`, `leave`. The user decides. Never auto-drop.

**Gate 10**
- [ ] Planted: target copy, target x2 plus noise, ID-like leak, and an after-the-fact column. All flagged.
- [ ] A legitimately strong feature (correlation about 0.7) is not flagged as high.
- [ ] 0 high flags on the control.
- [ ] Every flag shows the evidence number that triggered it.
- [ ] Time-ordered planted data triggers the time-split warning.

---

### Phase 11: Full pipeline, report, deploy (about 5-6 days)

**Files:** `doctor/pipeline.py`, `doctor/report.py`, `app/routes/report.py`, `templates/results.html`

**Pipeline order (important)**
1. Read and normalize hidden-null tokens to NaN (deterministic).
2. Types (deterministic conversions).
3. Duplicates.
4. Drop rows with a missing target. **Lock the split.**
5. Impossible outlier values set to null.
6. Nulls (user decisions, fit on train).
7. Extreme outlier handling.
8. Leakage decisions (drop columns).
9. Imbalance fix (train only).
10. Final model and metrics on the untouched test set.

**Build**
- Combined before/after with the noise label for the whole pipeline.
- Report (Markdown or HTML): issues found, every decision and who made it, metrics before and after, limits.
- Export: cleaned CSV and `reproduce.py`, a plain script that repeats the accepted decisions.
- Deploy to Render, Railway or Hugging Face Spaces with a lower file limit.

**Gate 11**
- [ ] On the accident data, full run reproduces the Phase 0 numbers within noise.
- [ ] `reproduce.py` run on the original file gives the same cleaned file (compare hashes).
- [ ] Order test: changing the order of steps 5 and 6 changes the result in a planted case, proving the order matters and is enforced.
- [ ] End to end on 5 datasets (accident, weather, 3 Kaggle) with no crash.
- [ ] 50 MB file completes in under 2 minutes.
- [ ] A new person can upload a file and finish a run without help (ask one friend; write down where they got stuck).

---

## 8. Phase tracker (copy to README or keep here)

| Phase | Name | Est. days | Gate result | Date |
|-------|------|-----------|-------------|------|
| 0 | Setup + evidence | 5-7 | PASS | 2026-10-19 |
| 1 | GUI shell + loader + profile | 3-4 | PASS | 2026-10-19 |
| 2 | Null detection | 3-4 | PASS | 2026-10-21 |
| 3 | Split + evaluation engine | 4-5 | | |

| 4 | Null advisor + fixer | 5-6 | | |
| 5 | Duplicates | 3 | | |
| 6 | Data types | 4 | | |
| 7 | Outlier detection | 3 | | |
| 8 | Outlier advisor + fixer | 3 | | |
| 9 | Imbalance | 4 | | |
| 10 | Leakage | 4 | | |
| 11 | Pipeline, report, deploy | 5-6 | | |

## 9. API routes

| Method | Path | Purpose |
|--------|------|---------|
| GET | / | upload page |
| POST | /upload | save CSV, create session |
| GET | /profile | profile and preview |
| POST | /target | set target column and task type |
| GET | /module/{name} | module page: detect results |
| POST | /module/{name}/advise | run advisor, return recommendation and measured table |
| GET | /module/{name}/preview | before/after sample rows |
| POST | /decide | save a decision (accept, change, skip) |
| POST | /apply | apply all decisions in pipeline order |
| GET | /compare | before/after metrics |
| GET | /bench | gate results, planted vs found |
| POST | /bench/run/{name} | run a gate from the GUI |
| GET | /report | download report |
| GET | /cleaned.csv | download cleaned data |
| GET | /reproduce.py | download the reproducible script |

Sessions are in memory with a random ID in a cookie. No database. Uploaded files are deleted after 1 hour.

## 10. Testing

- Unit tests per module in `tests/issues/`.
- Gate runner tests use `data/planted/`.
- Regression tests: oversample-before-split score is higher than split-first (Phase 3). If this flips, the engine is broken.
- Fixer tests assert: the test set is never resampled, never loses rows, never gets values learned from itself.
- Run: `pytest -q` and `python -m doctor.bench --all`

## 11. Known weak points (show them in the UI)

- Rule-based advice is a starting guess. Only the measured scores are evidence, and they are noisy on small data.
- One fixed model does not represent every model. A fix that helps a random forest may not help a linear model.
- Leakage rules miss subtle leaks.
- SMOTE on mixed categorical data can create nonsense rows.
- Planted data only tests problems you thought of. Real files will break the tool in new ways. That is why Phase 11 uses real datasets.

## 12. Definition of done (MVP)

- [ ] Gates 0 to 11 all PASS and logged in `docs/gate_log.md`
- [ ] Every issue type has separate detect, advise and fix files
- [ ] The user is asked before any change is applied
- [ ] Test set never touched by fixes (asserted in tests)
- [ ] Accident result documented in README
- [ ] Deployed link in README
