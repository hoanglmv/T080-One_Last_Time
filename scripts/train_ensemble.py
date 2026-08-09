"""Execution script to train Multi-Model Ensemble and print comparison table."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib
import pandas as pd


def format_comparison_table(metrics: dict, weights: dict) -> str:
    """Format test metrics as a fixed-width, four-decimal comparison table."""
    rows = []
    for model_name, model_metrics in metrics.items():
        rows.append(
            {
                "Model": model_name,
                "Blending Weight": f"{weights.get(model_name, 0.0):.4f}",
                "ROC-AUC": f"{model_metrics['roc_auc']:.4f}",
                "PR-AUC": f"{model_metrics['pr_auc']:.4f}",
                "Gini": f"{model_metrics['gini']:.4f}",
                "KS Stat": f"{model_metrics['ks']:.4f}",
                "ECE": f"{model_metrics['ece_10']:.4f}",
            }
        )
    return pd.DataFrame(rows).to_string(index=False)


def main():
    from src.credit_scoring.ensemble_pipeline import train_ensemble_pipeline

    print("================================================================================")
    print("🚀 HUẤN LUYỆN BỘ MÔ HÌNH ENSEMBLE ĐA MÔ HÌNH (LightGBM + XGBoost + CatBoost + LogReg)")
    print("================================================================================")

    res = train_ensemble_pipeline(
        data_dir="data/raw/home-credit-default-risk",
        feature_set="full",
        sample_size=None,
        seed=42,
    )

    metrics = res["test_metrics"]
    weights = res["optimal_weights"]

    print("\n================================================================================")
    print("📊 BẢNG SO SÁNH HIỆU NĂNG TẬP TEST (MULTI-MODEL COMPARISON TABLE)")
    print("================================================================================")
    print(format_comparison_table(metrics, weights))

    # Save Ensemble artifact
    output_path = Path("artifacts/models/ensemble_model.joblib")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(res, output_path, compress=3)

    print("\n================================================================================")
    print(f"✅ ĐÃ LƯU MÔ HÌNH ENSEMBLE THÀNH CÔNG TẠI: {output_path.resolve()}")
    print("================================================================================")


if __name__ == "__main__":
    main()
