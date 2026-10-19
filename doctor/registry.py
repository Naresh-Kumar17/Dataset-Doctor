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
        status="not_built",
    ),
    "duplicates": ModuleInfo(
        id="duplicates",
        name="Duplicates",
        phase=5,
        description="Find exact duplicate rows, near-duplicates, conflicting labels, and pre-balanced files.",
        status="not_built",
    ),
    "types": ModuleInfo(
        id="types",
        name="Data Types",
        phase=6,
        description="Identify numbers stored as text, ambiguous date formats, booleans, and constant/ID columns.",
        status="not_built",
    ),
    "outliers": ModuleInfo(
        id="outliers",
        name="Outliers",
        phase=7,
        description="Flag domain-impossible values, IQR/MAD extremes, and unit mix-ups.",
        status="not_built",
    ),
    "imbalance": ModuleInfo(
        id="imbalance",
        name="Class Imbalance",
        phase=9,
        description="Measure minority class share, imbalance ratio, and evaluate honest resampling/weighting.",
        status="not_built",
    ),
    "leakage": ModuleInfo(
        id="leakage",
        name="Data Leakage",
        phase=10,
        description="Catch target copies, perfect predictor features, ID leaks, and time-order leaks.",
        status="not_built",
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
