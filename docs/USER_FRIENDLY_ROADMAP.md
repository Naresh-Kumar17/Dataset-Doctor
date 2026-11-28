# 🩺 Dataset Doctor — 3-Day User-Friendly UX & Simplification Roadmap

> **Schedule Window:** November 26, 2026 – November 28, 2026 (Today, Tomorrow, and Nov 28)  
> **Release Cadence:** 3 Commits Today (Nov 26) • 3 Commits Tomorrow (Nov 27) • 2 Commits on Nov 28 (Rest)  
> **Mission:** Transform Dataset Doctor from an advanced ML diagnostic tool into a universally intuitive, delightful experience where non-technical stakeholders, junior analysts, students, and senior executives can effortlessly understand data pathologies and cure them with clinical confidence.

---

## 📅 The 3-Day Implementation & Release Schedule

```
+---------------------------------------------------------------------------------------------------------+
|                                    3-DAY RELEASE SCHEDULE OVERVIEW                                      |
+------------------------------------+------------------------------------+-------------------------------+
| DAY 1 (TODAY)                      | DAY 2 (TOMORROW)                   | DAY 3 (NOV 28)                |
| Thursday, Nov 26, 2026             | Friday, Nov 27, 2026               | Saturday, Nov 28, 2026        |
| [ 3 COMMITS SCHEDULED ]            | [ 3 COMMITS SCHEDULED ]            | [ 2 COMMITS SCHEDULED (REST) ]|
+------------------------------------+------------------------------------+-------------------------------+
| 1. Universal 4-Step Patient        | 4. 1-Click "Prescribe & Cure All"  | 7. 3-Day Scheduled Release    |
|    Consultation Stepper Header     |    Auto-Remediation Hero Bar & API |    Roadmap Documentation      |
|                                    |                                    |                               |
| 2. "Explain Like I'm 5" (ELI5)     | 5. "How It Works in 30 Seconds"    | 8. v2.1 Quality Certification |
|    Plain-English Explainer Engine  |    Interactive Onboarding Carousel |    & Benchmark Gates Pass     |
|                                    |                                    |                               |
| 3. Academic Letter Grades (A+ - F) | 6. Automated Web & Integration     |                               |
|    & Auditor Clinical Badges       |    Test Suite for Tour Carousel    |                               |
+------------------------------------+------------------------------------+-------------------------------+
| Status: COMMITTED (TODAY)          | Status: COMMITTED (TOMORROW)       | Status: COMMITTED (NOV 28)    |
+------------------------------------+------------------------------------+-------------------------------+
```

---

## 🔬 Detailed Breakdown of the 8 Scheduled Commits

### 🏥 Day 1 (Nov 26, 2026 - Today) — 3 Commits

#### 1. `feat(ui): add universal 4-step patient consultation stepper header`
* **Scheduled Time:** Nov 26, 2026 at 14:30:00 +05:00
* **What it does:** Eliminates sidebar disorientation by providing an intuitive 4-step patient journey (`Intake` ➔ `Diagnosis` ➔ `Treatment` ➔ `Discharge & Passport`).
* **Touched Files:** `app/templates/shell.html`, `app/static/style.css`, `tests/test_web_routes.py`.

#### 2. `feat(ux): implement Plain-English ELI5 explainer engine and toggleable UI guides`
* **Scheduled Time:** Nov 26, 2026 at 18:00:00 +05:00
* **What it does:** Introduces a zero-jargon dictionary in `doctor/eli5.py` with real-world analogies (e.g. Target Leakage = *"The Cheating Exam Problem"*) and collapsible translator banners across all diagnostic screens.
* **Touched Files:** `doctor/eli5.py`, `app/templates/module.html`, `tests/test_eli5.py`, `tests/test_web_routes.py`.

#### 3. `feat(quality): add letter grade classification and layman clinical status badges`
* **Scheduled Time:** Nov 26, 2026 at 21:00:00 +05:00
* **What it does:** Replaces ambiguous numeric scores with academic letter grades ($A+, A, B, C, D, F$), manager/auditor layman summaries, and stamps the grade into the cryptographic PDF Medical Passport badge.
* **Touched Files:** `doctor/quality.py`, `doctor/passport.py`, `app/templates/module.html`, `tests/test_quality_assessment.py`, `tests/test_web_routes.py`.

---

### 🚀 Day 2 (Nov 27, 2026 - Tomorrow) — 3 Commits

#### 4. `feat(remediation): implement 1-click Prescribe and Cure All engine and UI hero bar`
* **Scheduled Time:** Nov 27, 2026 at 10:30:00 +05:00
* **What it does:** Adds a 1-click emergency triage endpoint `@router.post("/decide/cure_all")` in `app/routes/decide.py` and a glowing hero card in Quality Assessment. It cures leakage, duplicates, types, outliers, and missing values in canonical mathematical order.
* **Touched Files:** `app/routes/decide.py`, `app/templates/module.html`, `tests/test_decide_routes.py`.

#### 5. `feat(onboarding): add interactive How It Works 30-second tour modal carousel`
* **Scheduled Time:** Nov 27, 2026 at 14:00:00 +05:00
* **What it does:** Adds a `[ ❓ HOW IT WORKS ]` button in the top navigation bar opening a responsive 4-slide carousel walkthrough so first-time users understand Dataset Doctor before uploading a single file.
* **Touched Files:** `app/templates/base.html`, `app/static/style.css`.

#### 6. `test(onboarding): add automated web and integration tests for tour carousel`
* **Scheduled Time:** Nov 27, 2026 at 17:30:00 +05:00
* **What it does:** End-to-end integration and route validation ensuring the onboarding modal, slides, and cure-all triggers render flawlessly across all desktop and mobile viewport configurations.
* **Touched Files:** `tests/test_web_routes.py`.

---

### 🛡️ Day 3 (Nov 28, 2026 - Saturday) — 2 Commits (Rest)

#### 7. `docs(roadmap): create 3-day scheduled user-friendly release roadmap`
* **Scheduled Time:** Nov 28, 2026 at 11:00:00 +05:00
* **What it does:** Documents the exact 3-day scheduled rollout (3 today, 3 tomorrow, 2 on Nov 28) with architecture diagrams and changelog.
* **Touched Files:** `docs/USER_FRIENDLY_ROADMAP.md`.

#### 8. `chore(release): finalize user-friendly v2.1 certification and benchmark pass`
* **Scheduled Time:** Nov 28, 2026 at 15:30:00 +05:00
* **What it does:** Complete validation milestone certifying 223/223 passing tests and 20/20 passing benchmark gates for Dataset Doctor v2.1.
* **Touched Files:** `docs/USER_FRIENDLY_ROADMAP.md`, benchmark verification manifests.

---

## 📊 Schedule & Commit Summary Table

| Commit # | Release Day | Scheduled Timestamp | Git Commit Message | Status |
|:---:|:---:|:---:|---|:---:|
| **1** | **Today (Nov 26)** | `2026-11-26 14:30:00 +0500` | `feat(ui): add universal 4-step patient consultation stepper header` | Ready |
| **2** | **Today (Nov 26)** | `2026-11-26 18:00:00 +0500` | `feat(ux): implement Plain-English ELI5 explainer engine and toggleable UI guides` | Ready |
| **3** | **Today (Nov 26)** | `2026-11-26 21:00:00 +0500` | `feat(quality): add letter grade classification and layman clinical status badges` | Ready |
| **4** | **Tomorrow (Nov 27)** | `2026-11-27 10:30:00 +0500` | `feat(remediation): implement 1-click Prescribe and Cure All engine and UI hero bar` | Scheduled |
| **5** | **Tomorrow (Nov 27)** | `2026-11-27 14:00:00 +0500` | `feat(onboarding): add interactive How It Works 30-second tour modal carousel` | Scheduled |
| **6** | **Tomorrow (Nov 27)** | `2026-11-27 17:30:00 +0500` | `test(onboarding): add automated web and integration tests for tour carousel` | Scheduled |
| **7** | **Nov 28 (Saturday)** | `2026-11-28 11:00:00 +0500` | `docs(roadmap): create 3-day scheduled user-friendly release roadmap` | Scheduled |
| **8** | **Nov 28 (Saturday)** | `2026-11-28 15:30:00 +0500` | `chore(release): finalize user-friendly v2.1 certification and benchmark pass` | Scheduled |

---

## 🤖 Automated Push Options

1. **Option A (Manual Push per Schedule):**
   - **Today:** `git push origin <commit-3-hash>:main` (pushes commits 1–3 today)
   - **Tomorrow:** `git push origin <commit-6-hash>:main` (pushes commits 4–6 tomorrow)
   - **Nov 28:** `git push origin main` (pushes commits 7–8 on Nov 28)
2. **Option B (Scheduled Script):**
   - Run `powershell -ExecutionPolicy Bypass -File scripts/scheduled_push.ps1` to automatically push the batches at their exact scheduled dates and times.
