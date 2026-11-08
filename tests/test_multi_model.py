"""
Unit tests for Phase 15: Multi-Model Experimentation & Comparison.
"""
import numpy as np
import pandas as pd
from doctor.experiment import MultiModelExperiment


def test_multi_model_experiment_run():
    np.random.seed(42)
    n = 200
    df = pd.DataFrame({
        "num1": np.random.randn(n),
        "num2": np.random.randn(n),
        "cat1": np.random.choice(["A", "B"], size=n),
        "target": np.random.choice([0, 1], p=[0.7, 0.3], size=n),
    })

    train_df = df.iloc[:140].copy()
    test_df = df.iloc[140:].copy()
    orig_test_len = len(test_df)
    orig_test_dist = test_df["target"].value_counts().to_dict()

    experimenter = MultiModelExperiment(random_seed=42)
    res = experimenter.run_experiment(train_df, test_df, target_column="target")

    # Evaluates 4 model architectures
    assert len(res.results) == 4
    model_names = {r.model_name for r in res.results}
    assert model_names == {"Logistic Regression", "Decision Tree", "Random Forest", "Gradient Boosting"}

    # Metrics computed
    for r in res.results:
        assert 0.0 <= r.train_accuracy <= 1.0
        assert 0.0 <= r.test_accuracy <= 1.0
        assert 0.0 <= r.test_f1 <= 1.0
        assert 0.0 <= r.test_balanced_accuracy <= 1.0

    # Best model picked
    assert res.best_model_f1 in model_names
    assert res.best_overall_model in model_names

    # Test set strictly unchanged
    assert len(test_df) == orig_test_len
    assert test_df["target"].value_counts().to_dict() == orig_test_dist
