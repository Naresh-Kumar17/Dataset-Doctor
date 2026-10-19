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

    # In Phase 1, all modules are registered as 'not_built'
    for m in modules:
        assert m.status == "not_built"


def test_get_specific_module():
    nulls_mod = get_module("nulls")
    assert nulls_mod is not None
    assert nulls_mod.name == "Null / Missing Values"
    assert nulls_mod.phase == 2
