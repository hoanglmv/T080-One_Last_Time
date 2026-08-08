import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from src.credit_scoring.ensemble_pipeline import (
    _prepare_catboost_frame,
    optimize_blending_weights,
    rank_averaging_transform,
)


def test_rank_averaging_is_scale_invariant():
    predictions = {"a": np.array([0.1, 0.4, 0.2]), "b": np.array([10.0, 40.0, 20.0])}
    ranked = rank_averaging_transform(predictions)
    np.testing.assert_allclose(ranked["a"], ranked["b"])


def test_gradient_free_weights_reach_best_component_auc():
    target = np.array([0, 0, 1, 1, 0, 1])
    predictions = {
        "weak": np.array([0.9, 0.7, 0.2, 0.1, 0.8, 0.3]),
        "strong": np.array([0.1, 0.2, 0.8, 0.9, 0.3, 0.7]),
    }
    weights = optimize_blending_weights(predictions, target, seed=7)
    blended = sum(weights[name] * predictions[name] for name in weights)
    assert sum(weights.values()) == pytest.approx(1.0)
    assert min(weights.values()) >= 0.0
    assert roc_auc_score(target, blended) >= max(
        roc_auc_score(target, prediction) for prediction in predictions.values()
    )


def test_catboost_frame_preserves_categories_and_uses_train_medians():
    train = pd.DataFrame({"amount": [1.0, np.nan, 3.0], "kind": ["A", None, "B"]})
    validation = pd.DataFrame({"amount": [np.nan], "kind": [None]})
    train_out, validation_out = _prepare_catboost_frame(
        train, validation, numeric_cols=["amount"], categorical_cols=["kind"]
    )
    assert validation_out.loc[0, "amount"] == pytest.approx(2.0)
    assert validation_out.loc[0, "kind"] == "__MISSING__"
    assert pd.api.types.is_string_dtype(train_out["kind"])
