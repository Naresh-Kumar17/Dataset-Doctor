# Dataset Doctor: GUI Design (Neo-Brutalism)

## 1. Style in one line
Flat bright colors, thick black borders, hard offset shadows, big bold type, no gradients, no blur, no rounded softness.

## 2. Design rules
- Every box has a **3px solid black border**.
- Every box has a **hard shadow** (no blur): `box-shadow: 6px 6px 0 #000`.
- Corners are square (`border-radius: 0`). Optional 4px on small chips only.
- Flat fills only. No gradients.
- Text is black on bright backgrounds. Keep contrast high.
- Hover: button moves toward the shadow (shadow shrinks). Press: shadow disappears.
- Layout is a visible grid. Gaps are generous (24px).

## 3. Color tokens
```css
:root {
  --bg:       #FFF8E7;  /* cream page background */
  --ink:      #000000;  /* text and borders */
  --paper:    #FFFFFF;  /* cards */
  --yellow:   #FFD93D;  /* primary action */
  --pink:     #FF6B9D;  /* high severity, fail */
  --blue:     #6BCBFF;  /* info, low severity */
  --green:    #7CFFB2;  /* success, pass, improved */
  --orange:   #FF9F43;  /* medium severity, close call */
  --purple:   #B197FC;  /* accent, "recommended" */
  --grey:     #D9D4C7;  /* not built */
}
```

Meaning of colors:
- High = pink, Medium = orange, Low = blue
- Improved metric = green, worse metric = pink
- Module status: NOT BUILT = grey, BUILT = yellow, GATE PASSED = green, GATE FAILED = pink
- Recommended option = purple

Never use color alone. Always add a text label (for color-blind users).

## 4. Typography
- Headings: `Space Grotesk` 700, or `Archivo Black`. Fallback: `system-ui`.
- Body: `Space Grotesk` 400/500.
- Numbers and code: `JetBrains Mono`.
- Scale: H1 56px, H2 36px, H3 24px, body 16px, small 13px.
- Headings uppercase, tight line height (1.05).

## 5. Core CSS (static/style.css starter)
```css
* { box-sizing: border-box; }

body {
  margin: 0;
  background: var(--bg);
  color: var(--ink);
  font-family: "Space Grotesk", system-ui, sans-serif;
  font-size: 16px;
  line-height: 1.5;
}

.container { max-width: 1100px; margin: 0 auto; padding: 32px 20px; }

h1, h2, h3 { text-transform: uppercase; line-height: 1.05; margin: 0 0 16px; }
h1 { font-size: 56px; }

.card {
  background: var(--paper);
  border: 3px solid var(--ink);
  box-shadow: 6px 6px 0 var(--ink);
  padding: 24px;
}

.btn {
  display: inline-block;
  font: inherit;
  font-weight: 700;
  text-transform: uppercase;
  background: var(--yellow);
  color: var(--ink);
  border: 3px solid var(--ink);
  box-shadow: 4px 4px 0 var(--ink);
  padding: 12px 20px;
  min-height: 44px;
  cursor: pointer;
  transition: transform .08s, box-shadow .08s;
}
.btn.secondary { background: var(--paper); }
.btn.danger    { background: var(--pink); }
.btn:hover  { transform: translate(2px, 2px); box-shadow: 2px 2px 0 var(--ink); }
.btn:active { transform: translate(4px, 4px); box-shadow: 0 0 0 var(--ink); }
.btn:focus-visible { outline: 3px dashed var(--ink); outline-offset: 3px; }

.chip {
  display: inline-block;
  border: 2px solid var(--ink);
  padding: 2px 10px;
  font-weight: 700;
  font-size: 13px;
  text-transform: uppercase;
}
.chip.high   { background: var(--pink); }
.chip.medium { background: var(--orange); }
.chip.low    { background: var(--blue); }
.chip.pass   { background: var(--green); }
.chip.fail   { background: var(--pink); }
.chip.built  { background: var(--yellow); }
.chip.notbuilt { background: var(--grey); }
.chip.reco   { background: var(--purple); }

table { width: 100%; border-collapse: collapse; border: 3px solid var(--ink); background: var(--paper); }
th { background: var(--ink); color: #fff; text-align: left; padding: 10px; text-transform: uppercase; }
td { border-top: 2px solid var(--ink); padding: 10px; font-family: "JetBrains Mono", monospace; }
tr.reco td { background: var(--purple); font-weight: 700; }
.better { background: var(--green); font-weight: 700; }
.worse  { background: var(--pink);  font-weight: 700; }

.dropzone {
  border: 3px dashed var(--ink);
  background: var(--yellow);
  padding: 48px;
  text-align: center;
  font-weight: 700;
  text-transform: uppercase;
}

.layout  { display: grid; grid-template-columns: 260px 1fr; gap: 24px; }
.sidebar .module { display: block; margin-bottom: 16px; }
.sidebar .module.active { background: var(--yellow); }

.grid { display: grid; gap: 24px; }
.grid.two   { grid-template-columns: repeat(2, 1fr); }
.grid.three { grid-template-columns: repeat(3, 1fr); }
@media (max-width: 800px) {
  .layout, .grid.two, .grid.three { grid-template-columns: 1fr; }
  h1 { font-size: 36px; }
}

@media (prefers-reduced-motion: reduce) { .btn { transition: none; } }
```

## 6. Page layout (shell)
All pages after upload use one shell, built in Phase 1:

```
+-----------------------------------------------------------+
| DATASET DOCTOR           file: accident.csv   [NEW FILE]   |
+--------------+--------------------------------------------+
| MODULES      |  MAIN AREA                                 |
| [x] Profile  |  (profile, module page, decision card,     |
| [ ] Nulls    |   results, bench)                          |
|     BUILT    |                                            |
| [ ] Dups     |                                            |
|     NOT BUILT|                                            |
| ...          |                                            |
| [ ] BENCH    |                                            |
+--------------+--------------------------------------------+
```
- The sidebar reads `registry.py`. Every module is listed from day one with its status chip. When you build a module and its gate passes, the chip turns green. No shell change needed.
- On phone width, the sidebar becomes a block above the main area.

## 7. Screens

### Screen 1: Upload
- Big H1: "DATASET DOCTOR".
- Tagline: "Find the data problems. Test the fix. You decide."
- Yellow dropzone card, "DROP CSV HERE OR CLICK".
- Small note: "Max 50 MB. Your file is deleted after 1 hour."

### Screen 2: Profile
- Stat boxes: rows, columns, memory, columns with any missing.
- Column table: name, detected type, unique count, sample values.
- Preview table (first 10 rows).
- Target picker (dropdown) and task type radio (Classification / Regression, auto-selected, editable).
- Button: "LOCK TARGET".

### Screen 3: Module page (same template for every module)
- Header: module name, status chip, "RUN DETECT" button.
- Summary strip: counts of High / Medium / Low issues.
- Issue table: column, what was found, evidence numbers, severity chip, certainty chip (CERTAIN / POSSIBLE).
- Each row has "OPEN DECISION" (after the advisor module exists).
- Empty state: "NO ISSUES FOUND" in a green card, with what was checked.

### Screen 4: Decision card (the interactive part)
One card per issue. Wireframe for a null issue:

```
+----------------------------------------------------------+
| NULLS  /  income        [HIGH]  [CERTAIN]                |
+----------------------------------------------------------+
| WHAT WE FOUND                                            |
|  120 of 1000 rows missing (12%)   skew 2.3               |
|  Missing is NOT linked to target (p = 0.61)              |
+----------------------------------------------------------+
| RECOMMENDED: MEDIAN            [RECOMMENDED]             |
|  Why: a few very high incomes pull the mean up.          |
|  Rules used: skewed numeric, rows are plentiful          |
|  Confidence: CLOSE CALL                                  |
+----------------------------------------------------------+
| TESTED ON TRAIN DATA (5-fold)                            |
| Option       | F1 (minority) | +/-  | Rows kept          |
| median  *    | 0.71          | 0.02 | 1000               |
| drop rows    | 0.70          | 0.03 | 880                |
| mean         | 0.69          | 0.02 | 1000               |
| mode         | 0.68          | 0.02 | 1000               |
+----------------------------------------------------------+
| PREVIEW (10 rows, before -> after)    [SHOW]             |
+----------------------------------------------------------+
| [APPLY RECOMMENDED] [CHOOSE ANOTHER v] [SKIP]            |
+----------------------------------------------------------+
```
Rules for this card:
- The recommended row is purple and marked with a text label, not just color.
- "Close call" appears in an orange chip when the top scores are within noise.
- "Not enough data" appears (orange) for datasets under 500 rows, and says the rule-based choice is used.
- "Choose another" opens a dropdown of the tested options plus `constant`, where the user types the value.
- Accuracy is never the main column when the minority share is under 20%. Show the banner "ACCURACY IS NOT RELIABLE HERE".
- After the user decides, the card collapses to a one-line summary: "income: MEDIAN (you chose)", with an UNDO button until the pipeline is applied.
- Top of the module page: "REVIEW ALL RECOMMENDATIONS" shows the full plan as a table with one APPLY ALL button, which still needs one confirmation.

### Screen 5: Test Bench (`/bench`)
Built in Phase 1, grows with every phase.
- One row per module and per stage (detect, advise, fix).
- Columns: module, stage, status chip, planted, found, missed, false alarms, last run time.
- "RUN GATE" button per row and "RUN ALL GATES" at the top.
- Click a row: shows planted vs found per column, with missed items in pink and false alarms in orange.
- Link to `docs/gate_log.md` content (read-only) so the history of fails and fixes is visible.
- You can also upload any CSV here and run a single module on it, to see how it behaves on a real file without the gate.

### Screen 6: Before / after
- Left: baseline metrics card. Right: result after all accepted decisions.
- Comparison table (Accuracy, Recall, Precision, F1, PR-AUC), green/pink cells for change.
- Banner when imbalanced: "ACCURACY IS NOT RELIABLE HERE. LOOK AT RECALL AND F1."
- Banner when the gain is within noise: "CHANGE IS WITHIN NOISE. NOT SIGNIFICANT."
- Banner when data is small: "UNDER 500 ROWS. COMPARISON IS NOISY."
- Decision log table: issue, column, choice, who decided (YOU / RECOMMENDED), effect.

### Screen 7: Report
- Summary of decisions and results.
- Buttons: "DOWNLOAD REPORT", "DOWNLOAD CLEANED CSV", "DOWNLOAD REPRODUCE.PY".
- Footer note listing known limits.

## 8. Components list
- Button (primary yellow, secondary white, danger pink)
- Card
- Chip (severity, certainty, status, recommended)
- Stat box
- Table (with highlighted recommended row)
- Dropzone
- Banner (warning)
- Decision card
- Sidebar module card
- Dropdown
- Progress bar: thick border, flat fill, step change only

## 9. Interaction details
- Loading: a block that says "CHECKING..." with a striped flat bar. No spinners. Advisor tests can take several seconds, so show which option is being tested ("TESTING MEDIAN 2 OF 5").
- Errors: pink card with black text, what went wrong, and what to do.
- Empty states: plain sentence plus one button.
- Destructive or large actions (apply all, drop column) show a confirm step with a plain sentence: "THIS REMOVES 120 TRAINING ROWS. TEST DATA IS NOT TOUCHED."

## 10. Accessibility
- Contrast: black on all listed fills passes WCAG AA. Do not put white text on yellow, green, blue, purple or orange.
- Visible keyboard focus (dashed outline, already in CSS).
- Labels on every input. Color is never the only signal.
- Respect `prefers-reduced-motion`.
- Minimum tap target 44px.

## 11. Build order for the UI
1. Phase 1: `base.html`, CSS, shell with sidebar, upload, profile, bench page with all modules listed as NOT BUILT.
2. Phase 2: module page template, with the nulls detect table.
3. Phase 4: decision card, decision log, preview.
4. Each later phase: no new layout, only fill the module template and add its decision card content.
5. Phase 11: before/after and report screens.
6. Test at phone width (375px) and desktop in every phase.
7. Only at the end: polish (stripes, sticker-style labels, small rotated badges, 2 to 3 degrees max).

## 12. Things to avoid
- Soft shadows, gradients, glassmorphism, rounded pills.
- More than 3 colors on one screen apart from status chips.
- Heavy animation. Neo-brutalism looks best when still and loud.
- Long paragraphs. Keep text short.
- Hiding the numbers. Every recommendation must show its evidence and its scores.
