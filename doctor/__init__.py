"""
Dataset Doctor: Honest, leak-free tabular data preparation and diagnostic engine.
"""
from doctor.api import DatasetDoctor
from doctor.loader import load_csv, DatasetLoadConfig
from doctor.profile import profile_dataset, DatasetProfile
from doctor.split import split_dataset, SplitConfig
from doctor.evaluate import evaluate_model, EvalResult
from doctor.pipeline import RemediationPipeline
from doctor.quality import DataQualityAssessor, QualityAssessment
from doctor.registry import get_all_modules, get_module
from doctor.manifest import ManifestGenerator, compute_dataframe_hash
from doctor.mcp import DatasetDoctorMCPServer, AgentSafetyViolation

__version__ = "1.0.0"

__all__ = [
    "DatasetDoctor",
    "load_csv",
    "DatasetLoadConfig",
    "profile_dataset",
    "DatasetProfile",
    "split_dataset",
    "SplitConfig",
    "evaluate_model",
    "EvalResult",
    "RemediationPipeline",
    "DataQualityAssessor",
    "QualityAssessment",
    "ManifestGenerator",
    "compute_dataframe_hash",
    "DatasetDoctorMCPServer",
    "AgentSafetyViolation",
    "get_all_modules",
    "get_module",
    "__version__",
]

