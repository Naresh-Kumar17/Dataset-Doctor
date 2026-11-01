"""
Unit tests for doctor/registry.py.
"""
from doctor.registry import get_all_modules, get_module


def test_registry_lists_all_planned_modules():
    modules = get_all_modules()
    module_ids = [m.id for m in modules]

    expected = ["nulls", "duplicates", "types", "outliers", "imbalance", "leakage"]
    for exp in expected:
        assert exp in module_ids

    # In Phase 6, nulls, split_eval, nulls_advisor, duplicates, and types are gate_passed
    for m in modules:
        if m.id in ["nulls", "split_eval", "nulls_advisor", "duplicates", "types"]:
            assert m.status in ["built", "gate_passed"]
        else:
            assert m.status == "not_built"



def test_get_specific_module():
    nulls_mod = get_module("nulls")
    assert nulls_mod is not None
    assert nulls_mod.name == "Null / Missing Values"
    assert nulls_mod.phase == 2
