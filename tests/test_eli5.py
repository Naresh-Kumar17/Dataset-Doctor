"""
Unit tests for 'Explain Like I'm 5' (ELI5) plain-English explainer engine.
Verifies all diagnostic domains are cataloged with analogies, dangers, and cures.
"""
import pytest
from doctor.eli5 import get_eli5_guide, get_all_eli5_guides, ELI5_CATALOG


def test_eli5_catalog_contains_all_core_domains():
    """Verify all 6 core diagnostic domains and workflow steps exist in catalog."""
    expected = [
        "leakage",
        "nulls",
        "duplicates",
        "types",
        "outliers",
        "imbalance",
        "quality_assessment",
        "pipeline",
        "multi_model",
    ]
    catalog = get_all_eli5_guides()
    for exp in expected:
        assert exp in catalog
        guide = catalog[exp]
        assert "title" in guide
        assert "tagline" in guide
        assert "analogy" in guide
        assert "why_it_matters" in guide
        assert "how_we_fix_it" in guide
        assert len(guide["analogy"]) > 20


def test_get_eli5_guide_resolves_aliases():
    """Verify advisor aliases map cleanly to their parent domain explainer."""
    g_leak = get_eli5_guide("leakage_advisor")
    assert "Cheating" in g_leak["title"]

    g_nulls = get_eli5_guide("nulls_advisor")
    assert "Missing" in g_nulls["title"]

    g_default = get_eli5_guide(None)
    assert "Quality" in g_default["title"]
