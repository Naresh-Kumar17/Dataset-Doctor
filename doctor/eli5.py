"""
Dataset Doctor: 'Explain Like I'm 5' (ELI5) Plain-English Explainer Engine.
Translates technical machine learning terminology and statistical pathologies
into everyday, crystal-clear layman analogies for students, product managers,
and domain analysts.
"""
from typing import Any, Dict, Optional


ELI5_CATALOG: Dict[str, Dict[str, str]] = {
    "leakage": {
        "title": "Data Leakage: The Cheating Exam Problem",
        "tagline": "When your machine learning model accidentally looks at the answer sheet.",
        "analogy": (
            "Imagine a student taking a medical certification exam who accidentally gets the answer key "
            "stapled to the back of the test booklet. In the classroom they score 100% effortlessly. "
            "But when they step into a real hospital with real patients and no answer key, they fail completely."
        ),
        "why_it_matters": (
            "Columns that copy the target, or record events that happen AFTER the decision (like 'payout_amount' "
            "or 'doctor_discharge_signature'), fabricate fake 100% accuracy in notebooks. In live production, "
            "those columns don't exist yet, causing the model to immediately crash."
        ),
        "how_we_fix_it": (
            "Dataset Doctor acts as an honest exam proctor: we detect and quarantine all cheating features, "
            "ensuring your model learns genuine predictive patterns rather than memorizing shortcuts."
        ),
    },
    "nulls": {
        "title": "Missing Values: The Missing Puzzle Pieces",
        "tagline": "Blanks in your spreadsheet that confuse machine learning algorithms.",
        "analogy": (
            "Imagine assembling a 500-piece jigsaw puzzle where 40 pieces were left in the factory box. "
            "Computers cannot do math on empty cells (NaN). If you throw away every row with a blank, "
            "you might discard half your dataset! If you fill with zero blindly, you distort the entire picture."
        ),
        "why_it_matters": (
            "Most ML libraries (like Scikit-Learn or XGBoost) crash if fed null values. Furthermore, missing data "
            "is rarely random: healthy patients might skip oxygen tests, while sick patients always get them. "
            "Naively deleting rows deletes your healthiest patients."
        ),
        "how_we_fix_it": (
            "Dataset Doctor tests multiple intelligent imputation recipes (median, mode, missingness indicators) "
            "on training folds only, picking the one that preserves true model accuracy without making stuff up."
        ),
    },
    "duplicates": {
        "title": "Duplicate Rows: The Loud Echo Chamber",
        "tagline": "Identical records that artificially skew your model's perspective.",
        "analogy": (
            "Imagine surveying a room of 100 voters, but 20 people vote twice, and 5 people say 'Yes' and 'No' "
            "at the exact same time. The first group gets unfair double-weight, and the second confuses the pollster."
        ),
        "why_it_matters": (
            "Exact duplicate rows cause models to overfit on identical patients. Conflicting duplicates (identical "
            "symptoms but opposite labels) teach the model contradictory rules, destroying its confidence."
        ),
        "how_we_fix_it": (
            "We deduplicate records strictly on training data while preserving your untouched test partition, "
            "resolving conflicting label pairs safely."
        ),
    },
    "types": {
        "title": "Data Type Mismatches: The Foreign Language Barrier",
        "tagline": "Numbers disguised as text that math engines cannot read.",
        "analogy": (
            "If an accountant writes '$1,500.00' or '98.5%' or '04/05/2023', a human understands it instantly. "
            "To an ML model, the dollar sign '$' or percent '%' turns the whole number into random text words, "
            "making arithmetic impossible."
        ),
        "why_it_matters": (
            "Numeric features formatted as text are either ignored entirely or turned into hundreds of useless "
            "dummy categories, wasting compute and obscuring real trends."
        ),
        "how_we_fix_it": (
            "Dataset Doctor automatically strips currency symbols and commas, parses percentages into clean decimal "
            "floats, encodes booleans to standard 0/1 bits, and clarifies ambiguous calendar dates."
        ),
    },
    "outliers": {
        "title": "Outliers & Extremes: The 999-Year-Old Patient",
        "tagline": "Typographical errors and black-swan spikes that distort the averages.",
        "analogy": (
            "If 9 people in a cafe earn $50,000 and Elon Musk walks in, the 'average salary' jumps to $20 billion! "
            "Similarly, if a nurse types an age of '999' or blood pressure of '-50', traditional linear models "
            "bend their entire decision boundary to fit the typo."
        ),
        "why_it_matters": (
            "Impossible domain values corrupt model weights and gradients. Extreme tail bursts cause predictions "
            "to wildly swing out of control."
        ),
        "how_we_fix_it": (
            "We separate biologically impossible values from legitimate rare events, recommending safe winsorization "
            "(capping) or robust isolation without throwing out valid data."
        ),
    },
    "imbalance": {
        "title": "Class Imbalance: The Needle in a Haystack",
        "tagline": "When one class is so rare that guessing 'Never' wins awards.",
        "analogy": (
            "Suppose only 1 out of 1,000 credit card transactions is fraudulent. A lazy security guard who approves "
            "every single swipe has a 99.9% accuracy score! But they catch zero fraud, making them 100% useless."
        ),
        "why_it_matters": (
            "High accuracy is often an illusion. In fraud, cancer diagnosis, or churn, the minority class is what "
            "actually matters. Standard algorithms default to ignoring the minority entirely."
        ),
        "how_we_fix_it": (
            "Dataset Doctor audits Precision-Recall and Balanced Accuracy. We recommend train-only resampling "
            "(SMOTE or honest class-weighting) without contaminating test evaluations."
        ),
    },
    "quality_assessment": {
        "title": "6-Domain Quality Index: Your Dataset's Medical Physical",
        "tagline": "A single 0 to 100 health score summarizing production readiness.",
        "analogy": (
            "Just like a pre-flight checklist for an airliner before takeoff, our 6-Domain Audit inspects "
            "every vital system: Missingness, Duplicates, Types, Outliers, Imbalance, and Cheating Leakage."
        ),
        "why_it_matters": (
            "Deploying a model trained on un-audited data is like flying a plane without checking fuel levels. "
            "Dataset Doctor enforces strict clinical readiness gates before any model training begins."
        ),
        "how_we_fix_it": (
            "We compute an evidence-weighted Health Score, assign an A+ to F letter grade, and generate an "
            "audit-grade Medical Passport PDF compliant with EU AI Act Article 10 standards."
        ),
    },
    "pipeline": {
        "title": "Treatment Recipe: The Doctor's Written Prescription",
        "tagline": "The exact step-by-step instructions to heal dirty data reproducibly.",
        "analogy": (
            "A physician doesn't just say 'you have a cold'; they write an exact prescription with dosages and order. "
            "Similarly, data cleaning must occur in canonical order (Leakage -> Duplicates -> Types -> Nulls) "
            "to avoid circular errors."
        ),
        "why_it_matters": (
            "Ad-hoc Jupyter notebook cleaning code is rarely reproducible in production pipelines. "
            "Dataset Doctor guarantees mathematical bit-for-bit reproducibility."
        ),
        "how_we_fix_it": (
            "We generate a clean, self-contained Python script ('reproduce.py') with zero proprietary dependencies "
            "that your engineering team can plug directly into airflow or production ETL."
        ),
    },
    "multi_model": {
        "title": "Production Crash & Recovery: The Stress Simulator",
        "tagline": "Testing how your model survives real-world data storms before going live.",
        "analogy": (
            "Automakers don't just drive new cars on sunny test tracks; they crash them into barriers and freeze "
            "them in blizzard chambers. We subject your trained models to brutal upstream shocks: Covariate Drift, "
            "Sensor Outages, Zero-Day Categories, and Tail Anomalies."
        ),
        "why_it_matters": (
            "Models that look great on historical data often fail when next month's production data drifts. "
            "We quantify exact crash velocity and prove your survival lift under stress."
        ),
        "how_we_fix_it": (
            "Our automated stress lab benchmarks baseline vs. remediated models, ensuring your pipeline "
            "has the resilience advantage to stay online when upstream systems glitch."
        ),
    },
}


def get_eli5_guide(topic: Optional[str]) -> Dict[str, str]:
    """Retrieves plain-English ELI5 explainer guide for a given topic or domain."""
    if not topic:
        return ELI5_CATALOG["quality_assessment"]
    clean_topic = str(topic).lower().strip()
    # Normalize aliases
    alias_map = {
        "leakage_advisor": "leakage",
        "nulls_advisor": "nulls",
        "duplicates_advisor": "duplicates",
        "types_advisor": "types",
        "outliers_advisor": "outliers",
        "imbalance_advisor": "imbalance",
        "error_analysis": "multi_model",
        "explainability": "multi_model",
    }
    resolved = alias_map.get(clean_topic, clean_topic)
    return ELI5_CATALOG.get(resolved, ELI5_CATALOG["quality_assessment"])


def get_all_eli5_guides() -> Dict[str, Dict[str, str]]:
    """Returns the full catalog of ELI5 explainers."""
    return dict(ELI5_CATALOG)
