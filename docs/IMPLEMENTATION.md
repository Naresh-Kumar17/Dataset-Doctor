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

### Phase 9: Class Imbalance Detection & Diagnosis (about 3 days)

**Files:** `doctor/issues/imbalance/detect.py`, `explain.py`, `tools/make_planted.py`

**Detect (Classification Targets Only)**
- Class counts, class distributions, and minority class share.
- Imbalance ratio calculation: `majority_count / minority_count`.
- Severity classification:
  - `high`: Minority class share $< 10\%$ (ratio $\ge 9:1$).
  - `medium`: Minority class share $< 20\%$ (ratio $\ge 4:1$).
  - Stays silent on balanced datasets (minority share $\ge 20\%$).
- Multi-class imbalance detection (identifies any minority class $< 10\%$).
- User domain prompt: "Is $X\%$ the true rate in the real world?"
- Explanations explicitly illustrate the "Accuracy Paradox": a naive majority baseline model achieves high accuracy while completely failing on minority instances.

**Gate 9**
- [ ] Planted scenarios: 90/10, 99/1, multi-class with tiny class (5%), and 50/50 balanced control.
- [ ] Correctly flags 90/10, 99/1, and multi-class as high severity.
- [ ] Strictly silent (0 flags) on balanced control.
- [ ] Plain-English explanation exposes accuracy paradox and real-world prevalence question.

---

### Phase 10: Class Imbalance Advisor & Remediation (about 4 days)

**Files:** `doctor/issues/imbalance/advise.py`, `fix.py`, `doctor/pipeline.py`

**Remediation Options (Train Partition Only)**
- `class_weight`: Uses loss penalty weighting (`class_weight="balanced"`). Preserves true row count and natural calibration.
- `random_oversample`: Samples minority rows with replacement on training partition.
- `smote`: Synthetic minority oversampling via nearest neighbor interpolation. Requires encoded numeric features.
- `random_undersample`: Discards majority rows down to minority ratio. Fast, but drops majority data.
- `threshold_tuning`: Adjusts classification decision threshold below 0.5 without changing rows.
- `leave`: Retains natural class proportions when calibration is paramount.

**Safety Invariants & Measurement**
- 5-fold Stratified CV on training data only. Evaluates minority Recall, Precision, F1, and PR-AUC.
- Trade-off visibility: Highlights when Recall increases come at the expense of Precision.
- Simplicity tie-breaker: Recommends `class_weight` or `leave` when score delta is within noise.
- **SMOTE Guard**: Refuses to run on unencoded categorical/object columns, instructing user to encode first.
- **Test Set Row & Ratio Invariant**: Test set rows and class distributions are strictly unchanged. Resampling never touches test.

**Gate 10**
- [ ] Planted 90/10 and 99/1: Advisor recommends `class_weight` or `random_oversample`.
- [ ] After any fix, test class ratio is unchanged (asserted).
- [ ] 0 resampled rows appear in test partition.
- [ ] SMOTE raises a clear warning on unencoded categorical columns instead of generating garbage.
- [ ] On road-accident data, shows high baseline accuracy but low minority recall, with measured improvement from remediation.

---

### Phase 11: Data Leakage Detection (about 3 days)

**Files:** `doctor/issues/leakage/detect.py`, `explain.py`, `tools/make_planted.py`

**Detection Heuristics**
- Single-feature predictive power: Fast single-column classifier achieving near-perfect score (AUC/F1 $> 0.95$).
- Target duplication: Features identical to target, scalar multiples ($X \times 2$), or trivial monotonic transformations.
- Index/ID leaks: High cardinality identifier columns that correlate with row order.
- Future/Post-event indicators: Column names containing post-event markers (`outcome`, `result`, `status_after`, `payout`, `discharged`).
- Time-order inversion: Timestamp or ordered sequence data evaluated with random shuffle instead of chronological split.

**Gate 11**
- [ ] Planted: target copy, $X \times 2$ transform, sequential ID leak, and post-event column. All 4 detected.
- [ ] Legitimate strong feature (correlation $\approx 0.70$) is not flagged as high severity.
- [ ] 0 false alarms on `clean_control.csv`.
- [ ] Time-ordered synthetic sequence triggers time-split warning.

---

### Phase 12: Data Leakage Advisor & Prevention (about 3 days)

**Files:** `doctor/issues/leakage/advise.py`, `fix.py`, `doctor/pipeline.py`

**Remediation Options**
- `drop_column`: Removes the leaking feature from training and test partitions.
- `time_ordered_split`: Reconfigures split engine to use chronological split instead of random shuffle.
- `keep_and_monitor`: Keeps column with explicit user acknowledgement. Never auto-drops columns silently.

**Gate 12**
- [ ] Recommends `drop_column` for target copies and ID leaks with explicit evidence score.
- [ ] User override allows keeping column with recorded warning in audit log.
- [ ] Exported `reproduce.py` includes column drop statements without test leakage.

---

### Phase 13: End-to-End Remediation Pipeline & Ordering Verification (about 4 days)

**Files:** `doctor/pipeline.py`, `doctor/compare.py`, `app/routes/decide.py`, `templates/compare.html`

**Canonical Causal Execution Order**
1. Read & normalize hidden null tokens to NaN (deterministic).
2. Type conversions (numeric, dates, booleans).
3. Deduplication (exact, conflicting rows).
4. Drop rows with missing target. **Lock train/test split.**
5. Impossible outlier values set to null.
6. Null imputation (train-learned medians/modes).
7. Tail outlier clipping (train-derived bounds).
8. Leakage column drops.
9. Imbalance remediation (train-only class weights or resampling).
10. Final model evaluation on untouched test set.

**Gate 13**
- [ ] Order enforcement test: Inverting order of steps alters output on planted scenario, proving order enforcement.
- [ ] Full pipeline run reproduces baseline scores on accident dataset.
- [ ] `compare.py` produces honest side-by-side metric diff with noise flags.

---

### Phase 14: Unified Data-to-Model Quality Assessment & Risk Matrix (about 4 days)

**Files:** `doctor/quality.py`, `app/routes/quality.py`, `templates/quality.html`

**Unified Assessment Engine**
- Synthesizes findings across all 6 diagnostic domains into a comprehensive Risk Matrix.
- Maps data quality defects directly to machine learning consequences:
  - Nulls $\to$ Data loss & imputation bias.
  - Duplicates $\to$ Optimistic test score inflation & conflicting signals.
  - Types $\to$ Silent categorical misinterpretation & date parsing bugs.
  - Outliers $\to$ Extreme gradient distortion & leverage points.
  - Imbalance $\to$ Accuracy illusion & minority blindspots.
  - Leakage $\to$ Production failure & unrealistic offline metrics.
- Dimension scores: Completeness, Uniqueness, Validity, Distributional Health, Evaluation Integrity.
- Unresolved issues tracker: Warns if user attempts model export with unresolved critical risks.

**Gate 14**
- [ ] Accurately aggregates findings from all modules without redundant detection passes.
- [ ] High-risk issues block production export unless explicitly acknowledged.
- [ ] Tested on 5 diverse datasets with consistent risk matrix generation.

---

### Phase 15: Multi-Model Experimentation & Baseline Comparison (about 5 days)

**Files:** `doctor/models.py`, `doctor/experiment.py`, `app/routes/experiments.py`, `templates/experiments.html`

**Model Comparison Engine**
- Extends fixed RandomForest baseline with standardized benchmark models:
  - Logistic Regression / Ridge Regressor (linear benchmark).
  - Decision Tree (interpretable baseline).
  - Gradient Boosted Trees (LightGBM / HistGradientBoosting).
- Evaluates all models against the **exact same locked train/test split** and fitted preprocessing pipeline.
- Multi-Metric Comparison Matrix: Ranks models by task-appropriate metrics (Recall, Precision, F1, PR-AUC, ROC-AUC, MAE, RMSE, R²).
- Overfitting Detection: Highlights large train-vs-test metric gaps.

**Gate 15**
- [ ] All candidate models trained strictly on train partition; evaluated on locked test set.
- [ ] Imbalance guard enforces minority metrics for all classifiers.
- [ ] Small sample warning triggered when $N < 500$ across all model comparisons.

---

### Phase 16: Model Error Analysis & Diagnostics (about 4 days)

**Files:** `doctor/diagnostics.py`, `app/routes/diagnostics.py`, `templates/diagnostics.html`

**Diagnostic Visualizations & Analysis**
- Classification:
  - Confusion Matrix with raw counts and normalized proportions.
  - Per-class Precision, Recall, F1 breakdown.
  - Precision-Recall curves and ROC curves.
  - Inspection of frequently confused class pairs.
- Regression:
  - Actual vs. Predicted scatter plot.
  - Residual distribution & error histogram.
  - Identification of worst-prediction outlier segments.

**Gate 16**
- [ ] Confusion matrix accurately calculated from true model predictions.
- [ ] Diagnostics derived strictly from validation fold or final test set (no data snooping).
- [ ] Residual outliers identified and linked back to original input features.

---

### Phase 17: Model Explainability & Feature Attribution (about 3 days)

**Files:** `doctor/explainability.py`, `app/routes/explainability.py`, `templates/explainability.html`

**Attribution Engine**
- Native tree feature importance (impurity-based) with high-cardinality inflation warnings.
- Permutation importance computed on training validation folds.
- Plain-English feature explanations with non-causal disclosures ("Correlated features do not establish causation").

**Gate 17**
- [ ] Permutation importance executed on validation folds without test data snooping.
- [ ] Clear disclaimer displayed on all explainability views.
- [ ] Feature rankings match ground-truth planted predictive features.

---

### Phase 18: Python Core SDK & Command-Line Interface (about 4 days)

**Files:** `doctor/api.py`, `doctor/cli.py`, `setup.py`, `tests/test_cli.py`

**Public SDK & CLI**
- Documented Python API (`doctor.load`, `doctor.profile`, `doctor.audit`, `doctor.evaluate`, `doctor.pipeline`).
- Terminal CLI commands:
  - `doctor profile <file.csv>`
  - `doctor audit <file.csv> --target <col>`
  - `doctor evaluate <file.csv> --target <col>`
  - `doctor bench --all`
- Shared engine: CLI and SDK reuse identical logic as web application without code duplication.

**Gate 18**
- [ ] All CLI commands execute cleanly and return consistent JSON / formatted outputs.
- [ ] Python SDK workflows pass integration tests without web server dependencies.

---

### Phase 19: Full Experiment Manifest & Bit-for-Bit Reproducibility (about 4 days)

**Files:** `doctor/manifest.py`, `doctor/export.py`, `app/routes/export.py`

**Comprehensive Artifact Generation**
- `manifest.json`: Machine-readable metadata capturing dataset hash, schema, split configuration, decision log, fitted pipeline parameters, model hyperparameters, and final metrics.
- Extended `reproduce.py`: Standalone script reproducing data cleaning, train/test split, model fitting, and evaluation.
- Cleaned CSV download with transformation audit headers.
- Automated Bit-for-Bit Reproducibility Test: Running `reproduce.py` reproduces identical output DataFrame and test metrics within float tolerance.

**Gate 19**
- [ ] `reproduce.py` executes standalone in clean Python environment without `doctor` package installed.
- [ ] Generated clean dataset hash matches Dataset Doctor in-memory DataFrame hash.
- [ ] `manifest.json` conforms to JSON schema specification.

---

### Phase 20: Model Context Protocol (MCP) Server & Agent Tooling (about 3 days)

**Files:** `doctor/mcp.py`, `tests/test_mcp.py`

**MCP Server Implementation**
- Standard JSON-RPC stdio MCP server for Antigravity, Claude Desktop, and Cursor.
- Declares core tools: `doctor_load_dataset`, `doctor_audit_quality`, `doctor_get_recommendations`, `doctor_apply_remediation`, `doctor_evaluate_baseline`, `doctor_export_reproduce`.
- Enforces agent safety invariants: Test set remains strictly immutable; destructive transformations require explicit approval.

**Gate 20**
- [ ] MCP tools discoverable and callable via standard JSON-RPC protocol.
- [ ] Automated integration test validates full agentic workflow via MCP tools.

---

### Phase 21: Full System Integration, Real-World Validation & Deployment (about 5 days)

**Files:** `docs/gate_log.md`, `README.md`, `Dockerfile`, `render.yaml`

**Production Hardening & Deployment**
- End-to-end testing across 5 real-world datasets (Accident severity, Weather forecasting, and 3 Kaggle benchmarks).
- Performance audit: 50 MB dataset completes full quality audit in under 2 minutes.
- Production containerization & live deployment verification on Render / Railway.
- Comprehensive user documentation and walkthrough guides.

**Gate 21**
- [ ] Full regression test suite passing 100% across all 21 phases.
- [ ] Zero memory leaks on large dataset processing.
- [ ] Live demo operational with active health check endpoint.

---

## 8. Master Phase Tracker

| Phase | Subsystem / Milestone | Stage | Gate Status | Target / Pass Date |
| :---: | :--- | :---: | :---: | :---: |
| **0** | Setup, Evidence Baseline & Planted Data Generator | Stage A | **PASS** | 2026-10-19 |
| **1** | Web GUI Shell, Loader, Profile & Test Bench | Stage A | **PASS** | 2026-10-19 |
| **2** | Null / Missing Values Detection | Stage A | **PASS** | 2026-10-21 |
| **3** | Split & Evaluation Engine | Stage A | **PASS** | 2026-10-21 |
| **4** | Null Advisor & Fixer (Interactive) | Stage A | **PASS** | 2026-10-23 |
| **5** | Duplicates Detection, Advice & Fixer | Stage A | **PASS** | 2026-10-26 |
| **6** | Data Types Detection, Advice & Fixer | Stage A | **PASS** | 2026-11-02 |
| **7** | Outlier Detection (Impossible Values & MAD) | Stage A | **PASS** | 2026-11-07 |
| **8** | Outlier Advisor & Fixer (Clipping & Preservation) | Stage A | **PASS** | 2026-11-12 |
| **9** | Class Imbalance Detection & Diagnosis | Stage A | **PLANNED** | Scheduled |
| **10** | Class Imbalance Advisor & Remediation | Stage A | **PLANNED** | Scheduled |
| **11** | Data Leakage Detection | Stage A | **PLANNED** | Scheduled |
| **12** | Data Leakage Advisor & Prevention | Stage A | **PLANNED** | Scheduled |
| **13** | End-to-End Remediation Pipeline & Ordering | Stage A | **PLANNED** | Scheduled |
| **14** | Unified Quality Assessment & Risk Matrix | Stage B | **PLANNED** | Scheduled |
| **15** | Multi-Model Experimentation & Comparison | Stage C | **PLANNED** | Scheduled |
| **16** | Model Error Analysis & Diagnostics | Stage D | **PLANNED** | Scheduled |
| **17** | Model Explainability (Permutation Importance) | Stage D | **PLANNED** | Scheduled |
| **18** | Python Core SDK & Command-Line Interface (CLI) | Stage E | **PLANNED** | Scheduled |
| **19** | Experiment Manifest & Bit-for-Bit Reproducibility | Stage F | **PLANNED** | Scheduled |
| **20** | Model Context Protocol (MCP) Server | Stage G | **PLANNED** | Scheduled |
| **21** | Full System Integration & Production Release | Stage G | **PLANNED** | Scheduled |

---

## 9. Comprehensive API Routes Matrix

| Method | Path | Subsystem | Description |
| :--- | :--- | :--- | :--- |
| `GET` | `/` | Web Shell | Upload page and landing dashboard |
| `POST` | `/upload` | Ingestion | Save CSV stream, validate MIME/size, initialize session |
| `GET` | `/profile` | Profiler | Profile statistics, schema inspection, and column preview |
| `POST` | `/target` | Split | Lock target column, task type, and stratified partitions |
| `GET` | `/module/{name}` | Workbench | Module diagnostic view with evidence tables and stats |
| `POST` | `/decide/{name}/apply` | Remediation | Fit transformer on train and append step to pipeline |
| `GET` | `/quality` | Unified View | Comprehensive Quality Assessment & Risk Matrix (Phase 14) |
| `GET` | `/experiments` | Models | Multi-model evaluation and comparison matrix (Phase 15) |
| `GET` | `/diagnostics` | Diagnostics | Confusion matrix, residual error plots (Phase 16) |
| `GET` | `/explainability` | Attribution | Feature importance and permutation attribution (Phase 17) |
| `GET` | `/compare` | Comparison | Side-by-side before vs after metrics on locked test set |
| `GET` | `/bench` | Test Bench | Interactive planted vs found gate verification dashboard |
| `POST` | `/bench/run/{name}` | Test Bench | Execute on-demand automated benchmark gate |
| `GET` | `/report` | Artifacts | Download Markdown / HTML executive quality report |
| `GET` | `/cleaned.csv` | Artifacts | Download cleaned CSV with transformation audit headers |
| `GET` | `/reproduce.py` | Artifacts | Download standalone pipeline reproduction script |
| `GET` | `/manifest.json` | Artifacts | Download machine-readable experiment manifest (Phase 19) |

---

## 10. Automated Testing & Reliability Invariants

1. **Test Set Isolation**: The test set is completely shielded from learning. Learned parameters (imputation values, clipping bounds, encoders, scalers) derive strictly from training data.
2. **Zero Test Row Loss**: Data cleaning operations may drop rows from the training set, but test set row counts and class distributions are strictly preserved.
3. **Planted-Data Ground Truth**: Every diagnostic module must pass its gate on deterministic planted data with $100\%$ recall of planted defects and $0$ false alarms on `clean_control.csv`.
4. **Noise Guard**: Evaluation score differences within the cross-validation standard deviation must be explicitly flagged as "close call / within noise".
5. **Execution Verification**: Run full test and benchmark suite via:
   ```bash
   pytest tests/
   python -m doctor.bench --all
   ```

