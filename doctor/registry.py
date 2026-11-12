"""
Module Registry for Dataset Doctor.
Tracks all planned detection, advice, and fix modules along with their build and gate status.
Used by the GUI sidebar, module routing, and Test Bench.
"""
from dataclasses import dataclass
from typing import Dict, List, Literal

ModuleStatus = Literal["not_built", "built", "gate_passed", "gate_failed"]


@dataclass
class ModuleInfo:
    id: str
    name: str
    phase: int
    description: str
    status: ModuleStatus = "not_built"
    last_run: str | None = None
    gate_details: dict | None = None


# Initial registry for Phase 1: All modules listed as not_built
PLANNED_MODULES: Dict[str, ModuleInfo] = {
    "nulls": ModuleInfo(
        id="nulls",
        name="Null / Missing Values",
        phase=2,
        description="Find missing values, hidden tokens (?, N/A), sentinels (-999), and missingness patterns.",
        status="gate_passed",
    ),
    "split_eval": ModuleInfo(
        id="split_eval",
        name="Split & Evaluation Engine",
        phase=3,
        description="Stratified splitting, fixed RandomForest baseline, 5-fold CV, noise estimation, and leakage demo.",
        status="gate_passed",
    ),
    "nulls_advisor": ModuleInfo(
        id="nulls_advisor",
        name="Null Advisor & Fixer",
        phase=4,
        description="Rule-based shortlist, 5-fold CV measurement, decision cards, leak-free transformation, and reproduce.py pipeline.",
        status="gate_passed",
    ),

    "duplicates": ModuleInfo(
        id="duplicates",
        name="Duplicates",
        phase=5,
        description="Find exact duplicate rows, near-duplicates, conflicting labels, and pre-balanced files.",
        status="gate_passed",
    ),
    "types": ModuleInfo(
        id="types",
        name="Data Types",
        phase=6,
        description="Identify numbers stored as text, ambiguous date formats, booleans, and constant/ID columns.",
        status="gate_passed",
    ),
    "outliers": ModuleInfo(
        id="outliers",
        name="Outlier Detection",
        phase=7,
        description="Flag domain-impossible values, IQR/MAD extremes, and unit mix-ups.",
        status="gate_passed",
    ),
    "outliers_advisor": ModuleInfo(
        id="outliers_advisor",
        name="Outlier Advisor & Fixer",
        phase=8,
        description="Evidence-backed outlier remediation, test bounds preservation, and reproduce.py pipeline.",
        status="gate_passed",
    ),
    "imbalance": ModuleInfo(
        id="imbalance",
        name="Class Imbalance Detection",
        phase=9,
        description="Measure minority class share, imbalance ratio, and accuracy paradox diagnosis.",
        status="gate_passed",
    ),
    "imbalance_advisor": ModuleInfo(
        id="imbalance_advisor",
        name="Class Imbalance Advisor & Fixer",
        phase=10,
        description="Train-only resampling, loss weighting, SMOTE categorical safety, and test distribution preservation.",
        status="gate_passed",
    ),
    "leakage": ModuleInfo(
        id="leakage",
        name="Data Leakage Detection",
        phase=11,
        description="Catch target copies, perfect predictor features, ID leaks, and time-order leaks.",
        status="gate_passed",
    ),
    "leakage_advisor": ModuleInfo(
        id="leakage_advisor",
        name="Data Leakage Advisor & Prevention",
        phase=12,
        description="Evidence-backed column drop recommendations and leak-free split validation.",
        status="gate_passed",
    ),
    "pipeline": ModuleInfo(
        id="pipeline",
        name="10-Step Deterministic Pipeline",
        phase=13,
        description="Enforce strict causal execution ordering and generate verified reproduce.py clean scripts.",
        status="gate_passed",
    ),
    "quality_assessment": ModuleInfo(
        id="quality_assessment",
        name="Unified Data-to-Model Quality Assessment",
        phase=14,
        description="Cross-module risk matrix, overall data quality score (0-100), and training blocker enforcement.",
        status="gate_passed",
    ),
    "multi_model": ModuleInfo(
        id="multi_model",
        name="Multi-Model Experimentation & Comparison",
        phase=15,
        description="Train and evaluate 4 diverse model families on the locked test partition with overfitting guards.",
        status="gate_passed",
    ),
    "error_analysis": ModuleInfo(
        id="error_analysis",
        name="Model Error Analysis & Diagnostics",
        phase=16,
        description="Confusion matrix rates, top confidence errors, subgroup slicing blind spots, and calibration.",
        status="gate_passed",
    ),
    "explainability": ModuleInfo(
        id="explainability",
        name="Model Explainability & Feature Attribution",
        phase=17,
        description="Validation-fold permutation importance, non-causal disclosures, and feature pruning advisor.",
        status="gate_passed",
    ),
}


def get_all_modules() -> List[ModuleInfo]:
    """Returns a list of all registered modules in phase order."""
    return sorted(PLANNED_MODULES.values(), key=lambda m: m.phase)


def get_module(module_id: str) -> ModuleInfo | None:
    """Retrieve metadata for a specific module by id."""
    return PLANNED_MODULES.get(module_id)


def update_module_status(module_id: str, status: ModuleStatus, gate_details: dict | None = None, timestamp: str | None = None):
    """Updates the runtime status of a module."""
    if module_id in PLANNED_MODULES:
        PLANNED_MODULES[module_id].status = status
        if gate_details is not None:
            PLANNED_MODULES[module_id].gate_details = gate_details
        if timestamp is not None:
            PLANNED_MODULES[module_id].last_run = timestamp
