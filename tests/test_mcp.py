"""
Unit tests for Model Context Protocol (MCP) Server & Agent Safety Contract (Phase 20).
"""
import json
from pathlib import Path
import pytest

from doctor.mcp import DatasetDoctorMCPServer, AgentSafetyViolation


@pytest.fixture
def clean_control_csv():
    root = Path(__file__).resolve().parent.parent
    return str(root / "data" / "planted" / "clean_control.csv")


@pytest.fixture
def nulls_csv():
    root = Path(__file__).resolve().parent.parent
    return str(root / "data" / "planted" / "nulls.csv")


def test_mcp_initialize_and_ping():
    server = DatasetDoctorMCPServer()

    # 1. initialize
    init_msg = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "test-agent", "version": "1.0"},
        },
    }
    resp = server.handle_message(init_msg)
    assert resp["id"] == 1
    assert resp["result"]["protocolVersion"] == "2024-11-05"
    assert resp["result"]["serverInfo"]["name"] == "dataset-doctor"

    # 2. notification (returns None)
    notif_msg = {"jsonrpc": "2.0", "method": "notifications/initialized"}
    assert server.handle_message(notif_msg) is None

    # 3. ping
    ping_msg = {"jsonrpc": "2.0", "id": 2, "method": "ping"}
    ping_resp = server.handle_message(ping_msg)
    assert ping_resp["id"] == 2
    assert ping_resp["result"] == {}


def test_mcp_list_tools():
    server = DatasetDoctorMCPServer()
    msg = {"jsonrpc": "2.0", "id": 10, "method": "tools/list"}
    resp = server.handle_message(msg)
    assert resp["id"] == 10
    tools = resp["result"]["tools"]

    tool_names = [t["name"] for t in tools]
    expected_tools = [
        "doctor_load_dataset",
        "doctor_audit_quality",
        "doctor_get_recommendations",
        "doctor_apply_remediation",
        "doctor_evaluate_baseline",
        "doctor_export_reproduce",
    ]
    for exp in expected_tools:
        assert exp in tool_names

    for tool in tools:
        assert "description" in tool
        assert "inputSchema" in tool
        assert tool["inputSchema"]["type"] == "object"


def test_mcp_load_and_audit(clean_control_csv):
    server = DatasetDoctorMCPServer()

    # Load dataset
    res_load = server.call_tool("doctor_load_dataset", {
        "file_path": clean_control_csv,
        "target_column": "target",
    })
    assert res_load["status"] == "dataset_loaded"
    assert res_load["n_rows"] == 500
    assert res_load["n_cols"] == 6
    assert res_load["train_rows"] == 400
    assert res_load["test_rows"] == 100
    assert res_load["test_partition_locked"] is True
    assert len(res_load["columns"]) == 6

    # Audit dataset
    res_audit = server.call_tool("doctor_audit_quality", {})
    assert res_audit["quality_score"] == 100
    assert res_audit["can_train"] is True
    assert "nulls" in res_audit["domain_summaries"]


def test_mcp_recommendations_and_remediation(nulls_csv):
    server = DatasetDoctorMCPServer()
    server.call_tool("doctor_load_dataset", {
        "file_path": nulls_csv,
        "target_column": "target",
    })

    # Get recommendations
    res_rec = server.call_tool("doctor_get_recommendations", {
        "issue_id": "nulls",
        "column": "num_missing_low",
    })
    assert res_rec["issue_id"] == "nulls"
    assert res_rec["column"] == "num_missing_low"
    assert res_rec["recommended"] in ("median", "mean", "leave", "drop_rows")
    assert len(res_rec["options"]) > 0

    # Apply remediation
    res_apply = server.call_tool("doctor_apply_remediation", {
        "issue_id": "nulls",
        "method": "median",
        "column": "num_missing_low",
        "params": {"fill_value": 0.0},
    })
    assert res_apply["status"] == "remediation_applied"
    assert res_apply["test_partition_intact"] is True
    assert res_apply["test_shape"][0] == len(server.test_df)


def test_agent_safety_contract_enforcement(clean_control_csv):
    server = DatasetDoctorMCPServer()
    server.call_tool("doctor_load_dataset", {
        "file_path": clean_control_csv,
        "target_column": "target",
    })

    # Attempting to drop rows from the test partition must be REJECTED immediately
    res_bad = server.call_tool("doctor_apply_remediation", {
        "issue_id": "nulls",
        "method": "drop_rows",
        "column": "country",
        "apply_to_test": True,
    })
    assert res_bad.get("safety_violation") is True
    assert "INVARIANT VIOLATION" in res_bad["error"]
    assert "locked test set" in res_bad["error"]

    # Attempting to apply SMOTE or row resampling on test must also be rejected
    res_smote_test = server.call_tool("doctor_apply_remediation", {
        "issue_id": "imbalance",
        "method": "smote",
        "column": "target",
        "apply_to_test": True,
    })
    assert res_smote_test.get("safety_violation") is True
    assert "INVARIANT VIOLATION" in res_smote_test["error"]


def test_mcp_evaluate_and_export(clean_control_csv, tmp_path):
    server = DatasetDoctorMCPServer()
    server.call_tool("doctor_load_dataset", {
        "file_path": clean_control_csv,
        "target_column": "target",
    })

    # Evaluate baseline
    res_eval = server.call_tool("doctor_evaluate_baseline", {"fast_mode": True})
    assert res_eval["model_name"] == "RandomForestClassifier"
    assert "accuracy" in res_eval["test_scores"]

    # Export reproduction script and manifest
    out_script = tmp_path / "agent_reproduce.py"
    res_exp = server.call_tool("doctor_export_reproduce", {
        "output_path": str(out_script),
    })
    assert res_exp["manifest_version"] == "1.0.0"
    assert out_script.exists()
    assert (tmp_path / "agent_reproduce_manifest.json").exists()
    assert "Standalone Dataset Doctor Reproduction Script" in out_script.read_text(encoding="utf-8")
