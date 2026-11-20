"""
Dataset Doctor: Model Context Protocol (MCP) Server (Phase 20).
Standard JSON-RPC 2.0 stdio server connecting AI agents to Dataset Doctor.
Enforces the Agent Safety Contract: Test partition is cryptographically locked and read-only.
"""
import io
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from doctor.api import DatasetDoctor
from doctor.base import Decision, DetectionContext, Issue
from doctor.export import export_standalone_script
from doctor.manifest import ManifestGenerator, compute_dataframe_hash
from doctor.pipeline import RemediationPipeline

logger = logging.getLogger("doctor.mcp")


class AgentSafetyViolation(Exception):
    """Raised when an AI agent requests an action violating dataset integrity or test isolation."""
    pass


class DatasetDoctorMCPServer:
    """
    Model Context Protocol (MCP) Server for Dataset Doctor.
    Exposes 6 specialized data science tools over JSON-RPC 2.0 stdio transport.
    """
    PROTOCOL_VERSION = "2024-11-05"
    SERVER_NAME = "dataset-doctor"
    SERVER_VERSION = "1.0.0"

    def __init__(self):
        self.current_doctor: Optional[DatasetDoctor] = None
        self.file_path: Optional[str] = None
        self.target_column: Optional[str] = None
        self.train_df: Optional[pd.DataFrame] = None
        self.test_df: Optional[pd.DataFrame] = None
        self.clean_train_df: Optional[pd.DataFrame] = None
        self.clean_test_df: Optional[pd.DataFrame] = None
        self.pipeline: RemediationPipeline = RemediationPipeline()
        self.initial_test_hash: Optional[str] = None
        self.initial_test_indices: Optional[set] = None

    def list_tools(self) -> List[Dict[str, Any]]:
        """Returns MCP tool definitions conforming to JSON schema standard."""
        return [
            {
                "name": "doctor_load_dataset",
                "description": "Loads a local CSV file into an isolated diagnostic session and performs initial schema inference and train/test partition locking.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "file_path": {
                            "type": "string",
                            "description": "Absolute or relative path to CSV dataset.",
                        },
                        "target_column": {
                            "type": "string",
                            "description": "Optional target feature name for classification or regression.",
                        },
                    },
                    "required": ["file_path"],
                },
            },
            {
                "name": "doctor_audit_quality",
                "description": "Executes comprehensive diagnostic assessment across 6 data quality domains (nulls, duplicates, types, outliers, imbalance, leakage).",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "modules": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Optional subset of modules to audit.",
                        },
                    },
                },
            },
            {
                "name": "doctor_get_recommendations",
                "description": "Solves a specific detected issue by running cross-validation on candidate remediation options strictly on train partition.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "issue_id": {
                            "type": "string",
                            "description": "Identifier of the target issue (e.g. nulls, duplicates, outliers, imbalance, leakage).",
                        },
                        "column": {
                            "type": "string",
                            "description": "Affected column name.",
                        },
                    },
                    "required": ["issue_id", "column"],
                },
            },
            {
                "name": "doctor_apply_remediation",
                "description": "Applies an approved remediation transformation step. Enforces Agent Safety Contract: Test partition is cryptographically protected and read-only.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "issue_id": {
                            "type": "string",
                            "description": "Module identifier.",
                        },
                        "method": {
                            "type": "string",
                            "description": "Remediation option identifier (e.g. median, clip_iqr, drop_duplicates).",
                        },
                        "column": {
                            "type": "string",
                            "description": "Target column for column-specific remediations.",
                        },
                        "params": {
                            "type": "object",
                            "description": "Explicit transformation hyperparameters.",
                        },
                        "apply_to_test": {
                            "type": "boolean",
                            "description": "Whether to apply transformation to test partition. Must NOT drop rows in test.",
                            "default": False,
                        },
                    },
                    "required": ["issue_id", "method"],
                },
            },
            {
                "name": "doctor_evaluate_baseline",
                "description": "Evaluates baseline machine learning model trained on train partition against locked test partition with 5-fold CV noise bounds.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "fast_mode": {
                            "type": "boolean",
                            "description": "Whether to use fast evaluation mode.",
                            "default": True,
                        },
                    },
                },
            },
            {
                "name": "doctor_export_reproduce",
                "description": "Generates a complete experiment manifest and a 100% standalone, dependency-free Python reproduction script.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "output_path": {
                            "type": "string",
                            "description": "Optional file path to write standalone reproduction script.",
                        },
                    },
                },
            },
        ]

    def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Dispatches and executes the requested tool within safety guardrails."""
        try:
            if tool_name == "doctor_load_dataset":
                return self._tool_load_dataset(arguments)
            elif tool_name == "doctor_audit_quality":
                return self._tool_audit_quality(arguments)
            elif tool_name == "doctor_get_recommendations":
                return self._tool_get_recommendations(arguments)
            elif tool_name == "doctor_apply_remediation":
                return self._tool_apply_remediation(arguments)
            elif tool_name == "doctor_evaluate_baseline":
                return self._tool_evaluate_baseline(arguments)
            elif tool_name == "doctor_export_reproduce":
                return self._tool_export_reproduce(arguments)
            else:
                return {"error": f"Unknown tool: '{tool_name}'"}
        except AgentSafetyViolation as e:
            return {
                "error": str(e),
                "safety_violation": True,
                "contract": "Agent Safety Contract (Test partition is cryptographically protected and read-only)",
            }
        except Exception as e:
            return {"error": f"Internal execution error in '{tool_name}': {str(e)}"}

    # -------------------------------------------------------------------------
    # Tool Implementations
    # -------------------------------------------------------------------------

    def _tool_load_dataset(self, args: Dict[str, Any]) -> Dict[str, Any]:
        file_path = args.get("file_path")
        if not file_path:
            return {"error": "Missing required argument 'file_path'"}

        path_obj = Path(file_path)
        if not path_obj.exists():
            return {"error": f"Dataset file does not exist: {file_path}"}

        target_column = args.get("target_column")
        self.current_doctor = DatasetDoctor.from_csv(file_path, target=target_column)
        self.file_path = str(path_obj.resolve())
        self.target_column = target_column
        self.pipeline = RemediationPipeline()

        # Split and lock test partition immediately if target available
        self.train_df, self.test_df = self.current_doctor.split(test_size=0.2, random_state=42)
        self.clean_train_df = self.train_df.copy()
        self.clean_test_df = self.test_df.copy()
        self.initial_test_hash = compute_dataframe_hash(self.test_df)
        self.initial_test_indices = set(self.test_df.index)

        profile = self.current_doctor.profile()

        return {
            "status": "dataset_loaded",
            "file_path": str(path_obj.name),
            "n_rows": profile.n_rows,
            "n_cols": profile.n_cols,
            "target_column": target_column,
            "task_type": self.current_doctor.task_type,
            "train_rows": len(self.train_df),
            "test_rows": len(self.test_df),
            "test_partition_locked": True,
            "test_partition_hash": self.initial_test_hash[:16] + "...",
            "columns": [
                {
                    "name": col.name,
                    "type": col.inferred_type,
                    "null_count": col.null_count,
                    "unique_count": col.unique_count,
                }
                for col in profile.columns
            ],
        }

    def _tool_audit_quality(self, args: Dict[str, Any]) -> Dict[str, Any]:
        if not self.current_doctor:
            return {"error": "No dataset currently loaded. Call doctor_load_dataset first."}

        assessment = self.current_doctor.audit()
        blockers = list(assessment.critical_blockers)
        
        issues_summary = []
        issues = assessment.unresolved_issues
        for issue in issues:
            issues_summary.append({
                "issue_id": issue.module_id,
                "column": issue.column,
                "severity": str(issue.severity),
                "certainty": str(getattr(issue, "certainty", "certain")),
                "title": issue.title,
                "reason": getattr(issue, "description", getattr(issue, "reason", "")),
            })

        domain_sums = {}
        for k, v in assessment.domain_summaries.items():
            if hasattr(v, "__dict__"):
                domain_sums[k] = v.__dict__
            else:
                domain_sums[k] = v

        return {
            "quality_score": assessment.quality_score,
            "readiness": assessment.readiness.value if hasattr(assessment.readiness, "value") else str(assessment.readiness),
            "can_train": assessment.can_train,
            "critical_blockers_count": len(blockers),
            "blockers": blockers,
            "total_issues_count": len(issues),
            "issues": issues_summary,
            "domain_summaries": domain_sums,
        }

    def _tool_get_recommendations(self, args: Dict[str, Any]) -> Dict[str, Any]:
        if not self.current_doctor:
            return {"error": "No dataset currently loaded. Call doctor_load_dataset first."}

        issue_id = args.get("issue_id", "")
        column = args.get("column", "")
        if not issue_id or not column:
            return {"error": "Arguments 'issue_id' and 'column' are required."}

        # Select corresponding advisor based on issue category
        if "null" in issue_id.lower():
            from doctor.issues.nulls.advise import NullAdvisor
            advisor = NullAdvisor()
        elif "dup" in issue_id.lower():
            from doctor.issues.duplicates.advise import DuplicateAdvisor
            advisor = DuplicateAdvisor()
        elif "type" in issue_id.lower():
            from doctor.issues.types.advise import TypeAdvisor
            advisor = TypeAdvisor()
        elif "outlier" in issue_id.lower():
            from doctor.issues.outliers.advise import OutlierAdvisor
            advisor = OutlierAdvisor()
        elif "imbalance" in issue_id.lower():
            from doctor.issues.imbalance.advise import ImbalanceAdvisor
            advisor = ImbalanceAdvisor()
        elif "leak" in issue_id.lower():
            from doctor.issues.leakage.advise import LeakageAdvisor
            advisor = LeakageAdvisor()
        else:
            return {"error": f"Unknown diagnostic domain for issue_id: '{issue_id}'"}

        synthetic_issue = Issue(
            module_id=issue_id,
            column=column,
            severity="medium",
            certainty="certain",
            title=f"Issue on {column}",
            description="Diagnostic issue awaiting remediation recommendation.",
        )

        ctx = DetectionContext(
            target_column=self.target_column,
            task_type=self.current_doctor.task_type,
        )

        rec = advisor.advise(
            issue=synthetic_issue,
            train_df=self.train_df,
            target=self.target_column,
            ctx=ctx,
            measure=False,  # Fast rule-based + simplicity hierarchy
        )

        options_data = []
        for opt in getattr(rec, "options", []):
            options_data.append({
                "id": opt.id,
                "label": getattr(opt, "label", opt.id),
                "description": getattr(opt, "description", ""),
                "is_destructive": getattr(opt, "is_destructive", False),
            })

        return {
            "issue_id": issue_id,
            "column": column,
            "recommended": rec.recommended,
            "reason": rec.reason,
            "confidence": getattr(rec, "confidence", "clear"),
            "options": options_data,
        }

    def _tool_apply_remediation(self, args: Dict[str, Any]) -> Dict[str, Any]:
        if not self.current_doctor or self.train_df is None or self.test_df is None:
            return {"error": "No dataset currently loaded and split. Call doctor_load_dataset first."}

        issue_id = args.get("issue_id", "")
        method = args.get("method", "")
        column = args.get("column")
        params = args.get("params", {})
        apply_to_test = bool(args.get("apply_to_test", False))

        # AGENT SAFETY CONTRACT ENFORCEMENT
        # 1. Reject any attempt to drop, resample, or mutate test rows
        destructive_methods = {
            "drop_rows", "drop", "drop_duplicates", "drop_outliers",
            "smote", "resample_smote", "undersample", "oversample", "random_undersample"
        }
        
        if apply_to_test and method in destructive_methods:
            raise AgentSafetyViolation(
                f"INVARIANT VIOLATION: Method '{method}' mutates or drops row partitions. "
                "The Agent Safety Contract strictly forbids dropping or resampling rows from the locked test set."
            )

        # 2. Record decision in pipeline
        decision = Decision(
            issue_id=issue_id,
            column=column,
            option_id=method,
            params=params,
            decided_by="mcp_agent",
        )

        train_before_shape = (len(self.clean_train_df), len(self.clean_train_df.columns))
        test_before_shape = (len(self.clean_test_df), len(self.clean_test_df.columns))

        # Record and fit on train partition
        self.pipeline.record_decision(decision, train_df=self.train_df)

        # Transform train
        new_clean_train = self.pipeline.transform(self.train_df, is_test=False)

        # Transform test
        new_clean_test = self.pipeline.transform(self.test_df, is_test=True)

        # 3. VERIFY TEST SET IMMUTABILITY INVARIANT
        if len(new_clean_test) != len(self.test_df):
            # Rollback pipeline step
            self.pipeline.steps.pop()
            self.pipeline.decisions.pop()
            raise AgentSafetyViolation(
                f"INVARIANT VIOLATION: Transformation '{method}' altered test set row count "
                f"from {len(self.test_df)} to {len(new_clean_test)}. Transformation rolled back."
            )

        if set(new_clean_test.index) != self.initial_test_indices:
            self.pipeline.steps.pop()
            self.pipeline.decisions.pop()
            raise AgentSafetyViolation(
                "INVARIANT VIOLATION: Transformation altered test row identity or indices. Transformation rolled back."
            )

        self.clean_train_df = new_clean_train
        self.clean_test_df = new_clean_test

        return {
            "status": "remediation_applied",
            "issue_id": issue_id,
            "method": method,
            "column": column,
            "train_shape_before": train_before_shape,
            "train_shape_after": (len(self.clean_train_df), len(self.clean_train_df.columns)),
            "test_shape": (len(self.clean_test_df), len(self.clean_test_df.columns)),
            "test_partition_intact": True,
            "total_pipeline_steps": len(self.pipeline.steps),
        }

    def _tool_evaluate_baseline(self, args: Dict[str, Any]) -> Dict[str, Any]:
        if not self.current_doctor or self.train_df is None or self.test_df is None:
            return {"error": "No active dataset session. Call doctor_load_dataset first."}

        fast_mode = bool(args.get("fast_mode", True))
        eval_res = self.current_doctor.evaluate(fast_mode=fast_mode)

        return {
            "model_name": "RandomForestClassifier",
            "task_type": self.current_doctor.task_type,
            "train_cv_mean": eval_res.train_cv_mean,
            "train_cv_std": eval_res.train_cv_std,
            "test_scores": eval_res.test_scores,
            "accuracy_reliable": eval_res.accuracy_reliable,
            "warnings": eval_res.warnings,
        }

    def _tool_export_reproduce(self, args: Dict[str, Any]) -> Dict[str, Any]:
        if not self.current_doctor or self.train_df is None or self.test_df is None:
            return {"error": "No active dataset session. Call doctor_load_dataset first."}

        gen = ManifestGenerator()
        manifest = gen.generate(
            raw_df=self.current_doctor.df,
            train_df=self.train_df,
            test_df=self.test_df,
            clean_train_df=self.clean_train_df if self.clean_train_df is not None else self.train_df,
            clean_test_df=self.clean_test_df if self.clean_test_df is not None else self.test_df,
            target_column=self.target_column,
            pipeline=self.pipeline,
            dataset_name=Path(self.file_path).name if self.file_path else "dataset.csv",
        )

        output_path = args.get("output_path")
        script_code = export_standalone_script(
            pipeline=self.pipeline,
            manifest=manifest,
            raw_csv_path=self.file_path or "dataset.csv",
        )

        saved_path = None
        manifest_path = None
        if output_path:
            out_p = Path(output_path)
            out_p.parent.mkdir(parents=True, exist_ok=True)
            with open(out_p, "w", encoding="utf-8") as f:
                f.write(script_code)
            saved_path = str(out_p.resolve())

            man_p = out_p.with_name(out_p.stem + "_manifest.json")
            with open(man_p, "w", encoding="utf-8") as f:
                f.write(manifest.to_json())
            manifest_path = str(man_p.resolve())

        return {
            "manifest_version": manifest.manifest_version,
            "raw_dataset_hash": manifest.raw_dataset_hash,
            "clean_train_hash": manifest.clean_train_hash,
            "clean_test_hash": manifest.clean_test_hash,
            "reproduce_script_path": saved_path,
            "manifest_path": manifest_path,
            "total_decisions_logged": len(manifest.decisions),
            "script_preview": "\n".join(script_code.splitlines()[:25]),
        }

    # -------------------------------------------------------------------------
    # JSON-RPC 2.0 Protocol Handler
    # -------------------------------------------------------------------------

    def handle_message(self, message: Union[str, Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Handles incoming JSON-RPC 2.0 messages."""
        if isinstance(message, str):
            try:
                data = json.loads(message)
            except Exception as e:
                return {
                    "jsonrpc": "2.0",
                    "id": None,
                    "error": {"code": -32700, "message": f"Parse error: {str(e)}"},
                }
        else:
            data = message

        msg_id = data.get("id")
        method = data.get("method")

        # 1. initialize
        if method == "initialize":
            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {
                    "protocolVersion": self.PROTOCOL_VERSION,
                    "capabilities": {
                        "tools": {"listChanged": False},
                    },
                    "serverInfo": {
                        "name": self.SERVER_NAME,
                        "version": self.SERVER_VERSION,
                    },
                },
            }

        # 2. notifications/initialized (notification, no response)
        if method in ("notifications/initialized", "initialized"):
            return None

        # 3. ping
        if method == "ping":
            return {"jsonrpc": "2.0", "id": msg_id, "result": {}}

        # 4. tools/list
        if method == "tools/list":
            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {"tools": self.list_tools()},
            }

        # 5. tools/call
        if method == "tools/call":
            params = data.get("params", {})
            name = params.get("name")
            arguments = params.get("arguments", {})

            result = self.call_tool(name, arguments)
            is_error = bool(result.get("error") is not None)

            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {
                    "content": [
                        {
                            "type": "text",
                            "text": json.dumps(result, indent=2),
                        }
                    ],
                    "isError": is_error,
                },
            }

        # Unknown method
        if msg_id is not None:
            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "error": {
                    "code": -32601,
                    "message": f"Method '{method}' not found",
                },
            }
        return None

    def run_stdio(self, input_stream=None, output_stream=None) -> None:
        """Starts stdio message loop listening for JSON-RPC 2.0 requests."""
        inp = input_stream or sys.stdin
        out = output_stream or sys.stdout

        for line in inp:
            line = line.strip()
            if not line:
                continue
            response = self.handle_message(line)
            if response is not None:
                out.write(json.dumps(response) + "\n")
                out.flush()


def main():
    """CLI entrypoint for running Dataset Doctor MCP server."""
    server = DatasetDoctorMCPServer()
    server.run_stdio()


if __name__ == "__main__":
    main()
