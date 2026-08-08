"""Execution script to train Multi-Model Ensemble and print comparison table."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib
import pandas as pd
from src.credit_scoring.ensemble_pipeline import train_ensemble_pipeline


def main():
    print("================================================================================")
    print("🚀 HUẤN LUYỆN BỘ MÔ HÌNH ENSEMBLE ĐA MÔ HÌNH (LightGBM + XGBoost + CatBoost + LogReg)")
    print("================================================================================")

    res = train_ensemble_pipeline(
        data_dir="data/raw/home-credit-default-risk",
        feature_set="serving",
        sample_size=None,
        seed=42,
    )

    metrics = res["test_metrics"]
    weights = res["optimal_weights"]

    rows = []
    for model_name, m in metrics.items():
        weight_str = f"{weights.get(model_name, 0.0):.4f}" if model_name in weights else "-"
        rows.append({
            "Model": model_name,
            "Blending Weight": weight_str,
            "ROC-AUC": f"{m['roc_auc']:.4f}",
            "PR-AUC": f"{m['pr_auc']:.4f}",
            "Gini": f"{m['gini']:.4f}",
            "KS Stat": f"{m['ks']:.4f}",
            "ECE": f"{m['ece_10']:.4f}",
        })

    df_res = pd.DataFrame(rows)

    print("\n================================================================================")
    print("📊 BẢNG SO SÁNH HIỆU NĂNG TẬP TEST (MULTI-MODEL COMPARISON TABLE)")
    print("================================================================================")
    print(df_res.to_string(index=False))

    # Save Ensemble artifact
    output_path = Path("artifacts/models/ensemble_model.joblib")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(res, output_path, compress=3)

    print("\n================================================================================")
    print(f"✅ ĐÃ LƯU MÔ HÌNH ENSEMBLE THÀNH CÔNG TẠI: {output_path.resolve()}")
    print("================================================================================")


if __name__ == "__main__":
    main()
