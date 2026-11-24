"""
Unit tests for Production Crash & Data Autopsy Simulator.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from doctor.simulator import ProductionStressSimulator, CrashSimulationResult, AutopsyBatteryReport


@pytest.fixture
def sample_data():
    np.random.seed(42)
    n = 200
    df = pd.DataFrame({
        "num_feature": np.random.randn(n) * 10 + 50,
        "cat_feature": np.random.choice(["TierA", "TierB", "TierC"], size=n),
        "target": np.random.choice([0, 1], p=[0.7, 0.3], size=n),
    })
    train_df = df.iloc[:150].copy()
    test_df = df.iloc[150:].copy()
    return train_df, test_df


def test_stress_injections(sample_data):
    _, test_df = sample_data
    sim = ProductionStressSimulator(random_seed=42)

    # 1. Covariate drift
    drift_df = sim.inject_covariate_drift(test_df, "target", drift_std_factor=1.5)
    assert drift_df["num_feature"].mean() > test_df["num_feature"].mean()

    # 2. Outage missingness
    outage_df = sim.inject_outage_missingness(test_df, "target", outage_rate=0.3)
    assert outage_df["num_feature"].isna().sum() > 0

    # 3. Category shock
    shock_df = sim.inject_category_shock(test_df, "target", novelty_rate=0.3)
    assert (shock_df["cat_feature"] == "ZERO_DAY_NOVEL_CATEGORY").sum() > 0

    # 4. Extreme tail surge
    surge_df = sim.inject_extreme_tail_surge(test_df, "target", spike_rate=0.2)
    assert surge_df["num_feature"].max() > test_df["num_feature"].max()


def test_single_stress_vector_execution(sample_data):
    train_df, test_df = sample_data
    sim = ProductionStressSimulator(random_seed=42)

    result = sim.run_stress_vector(
        stress_id="covariate_drift",
        raw_train=train_df,
        raw_test=test_df,
        clean_train=train_df,
        clean_test=test_df,
        target_column="target",
    )

    assert isinstance(result, CrashSimulationResult)
    assert result.stress_id == "covariate_drift"
    assert 0.0 <= result.baseline_normal_acc <= 1.0
    assert 0.0 <= result.baseline_stress_acc <= 1.0
    assert result.verdict in ("RESILIENT", "MODERATE_DEGRADATION", "CRITICAL_COLLAPSE")
    assert len(result.clinical_autopsy) > 0

    # Test dictionary export
    d = result.to_dict()
    assert d["stress_name"] == "Covariate Distribution Drift"
    assert "baseline_drop_pct" in d


def test_full_autopsy_battery(sample_data):
    train_df, test_df = sample_data
    sim = ProductionStressSimulator(random_seed=42)

    battery = sim.run_autopsy_battery(
        raw_train=train_df,
        raw_test=test_df,
        clean_train=train_df,
        clean_test=test_df,
        target_column="target",
    )

    assert isinstance(battery, AutopsyBatteryReport)
    assert len(battery.results) == 4
    assert battery.overall_verdict in ("HIGHLY_RESILIENT", "MODERATELY_STABLE")
    assert len(battery.executive_autopsy_summary) > 0

    d = battery.to_dict()
    assert len(d["results"]) == 4
    assert "overall_resilience_gain_pct" in d
