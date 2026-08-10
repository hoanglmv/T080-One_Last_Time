from scripts.train_ensemble import format_comparison_table


def test_format_comparison_table_uses_four_decimal_places():
    metrics = {
        "LogisticRegression": {
            "roc_auc": 0.7507, "pr_auc": 0.2316, "gini": 0.5014,
            "ks": 0.3752, "ece_10": 0.0014,
        },
        "Ensemble_Calibrated": {
            "roc_auc": 0.7638, "pr_auc": 0.2518, "gini": 0.5276,
            "ks": 0.3988, "ece_10": 0.002,
        },
    }

    table = format_comparison_table(metrics, {"LogisticRegression": 0.25})

    assert "Model Blending Weight ROC-AUC PR-AUC   Gini KS Stat    ECE" in table
    assert "LogisticRegression          0.2500  0.7507 0.2316 0.5014  0.3752 0.0014" in table
    assert "Ensemble_Calibrated          0.0000  0.7638 0.2518 0.5276  0.3988 0.0020" in table
