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

    # In Phase 4, nulls and split_eval are gate_passed, while later modules remain not_built
    for m in modules:
        if m.id in ["nulls", "split_eval"]:
            assert m.status in ["built", "gate_passed"]
        elif m.id == "nulls_advisor":
            assert m.status in ["not_built", "built", "gate_passed"]
        else:
            assert m.status == "not_built"



def test_get_specific_module():
    nulls_mod = get_module("nulls")
    assert nulls_mod is not None
    assert nulls_mod.name == "Null / Missing Values"
    assert nulls_mod.phase == 2
