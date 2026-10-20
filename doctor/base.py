"""
Dataset Doctor: Core Data Structures and Interfaces.
Defines Issue, Recommendation, Option, Decision, and base classes for
Detectors, Advisors, and Fixers across all diagnostic modules.
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional
import pandas as pd

SeverityLevel = Literal["high", "medium", "low"]
CertaintyLevel = Literal["certain", "possible"]


@dataclass
class Issue:
    """Represents a diagnostic issue detected in the dataset."""
    module_id: str
    column: Optional[str]
    severity: SeverityLevel
    certainty: CertaintyLevel
    title: str
    description: str
    evidence: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class Option:
    """A concrete fix or remediation option evaluated for an issue."""
    id: str  # e.g. "drop_rows", "median", "mode", "indicator"
    label: str
    params: Dict[str, Any] = field(default_factory=dict)
    explanation: str = ""
    measured: Optional[Dict[str, Any]] = None  # {"f1_mean": ..., "f1_std": ..., "n_rows_kept": ...}


@dataclass
class Recommendation:
    """Actionable recommendation proposing a ranked list of fix options."""
    issue: Issue
    recommended: str  # option id
    reason: str
    rule_hits: List[str] = field(default_factory=list)
    options: List[Option] = field(default_factory=list)
    confidence: Literal["clear", "close call", "not enough data"] = "clear"


@dataclass
class Decision:
    """User choice accepting, altering, or rejecting a recommendation."""
    issue_id: str
    column: Optional[str]
    option_id: str
    params: Dict[str, Any]
    decided_by: Literal["user", "recommended_accepted", "skipped"]
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class DetectionContext:
    """Contextual metadata passed to detectors."""
    target_column: Optional[str] = None
    task_type: Optional[str] = None  # "classification" | "regression"
    minority_share: Optional[float] = None
    random_seed: int = 42
    dataset_size: Optional[int] = None


class Detector:
    """Base class / interface for diagnostic issue detectors."""
    id: str

    def detect(self, df: pd.DataFrame, ctx: Optional[DetectionContext] = None) -> List[Issue]:
        """Runs read-only detection on dataset, returning a list of Issues."""
        raise NotImplementedError


class Advisor:
    """Base class / interface for issue fix advisors (Phase 4+)."""
    id: str

    def advise(
        self,
        issue: Issue,
        train_df: pd.DataFrame,
        target: Optional[str],
        ctx: Optional[DetectionContext] = None,
    ) -> Recommendation:
        """Evaluates remediation options on training data only."""
        raise NotImplementedError


class Fixer:
    """Base class / interface for issue transformers/fixers (Phase 4+)."""
    id: str
    rows_may_be_dropped_from: str = "train"  # Test rows are never dropped

    def fit(self, train_df: pd.DataFrame, decision: Decision) -> Any:
        """Learns fix parameters from training data only."""
        raise NotImplementedError

    def transform(self, df: pd.DataFrame, fitted: Any) -> pd.DataFrame:
        """Applies fitted transformation to train or test set."""
        raise NotImplementedError

