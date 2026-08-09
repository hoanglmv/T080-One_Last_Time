"""Execution script to train and compare two ensemble models side-by-side:
1. Hybrid Model (Traditional + Alternative Data)
2. Alternative-Only Model (Chỉ sử dụng dữ liệu thay thế)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib
import pandas as pd
from src.credit_scoring.ensemble_pipeline import train_ensemble_pipeline


def main():
    print("==========================================================================================")
    print("🚀 CHẠY SONG SONG VÀ SO SÁNH 2 MÔ HÌNH: HYBRID MODEL VS ALTERNATIVE-ONLY MODEL")
    print("==========================================================================================")

    data_dir = "data/raw/home-credit-default-risk"

    # 1. Huấn luyện Mô hình 1: Hybrid (Dữ liệu truyền thống + Dữ liệu thay thế)
    print("\n------------------------------------------------------------------------------------------")
    print("1️⃣ HUẤN LUYỆN MÔ HÌNH HYBRID (TRADITIONAL + ALTERNATIVE DATA)...")
    print("------------------------------------------------------------------------------------------")
    res_hybrid = train_ensemble_pipeline(
        data_dir=data_dir,
        feature_set="serving",
        sample_size=30000,
        seed=42,
    )

    # 2. Huấn luyện Mô hình 2: Alternative-Only (Chỉ sử dụng dữ liệu thay thế)
    print("\n------------------------------------------------------------------------------------------")
    print("2️⃣ HUẤN LUYỆN MÔ HÌNH ALTERNATIVE-ONLY (CHỈ DÙNG DỮ LIỆU THAY THẾ)...")
    print("------------------------------------------------------------------------------------------")
    res_alt = train_ensemble_pipeline(
        data_dir=data_dir,
        feature_set="alternative_only",
        sample_size=30000,
        seed=42,
    )

    # 3. Bảng So Sánh Mô hình Ensemble của cả 2
    hybrid_ens_metrics = res_hybrid["test_metrics"]["Ensemble_Calibrated"]
    alt_ens_metrics = res_alt["test_metrics"]["Ensemble_Calibrated"]

    rows_summary = [
        {
            "Mô hình (Model Variant)": "Hybrid Model (Traditional + Alternative)",
            "Số đặc trưng": len(res_hybrid.get("feature_cols", [])),
            "ROC-AUC": f"{hybrid_ens_metrics['roc_auc']:.4f}",
            "PR-AUC": f"{hybrid_ens_metrics['pr_auc']:.4f}",
            "Gini Index": f"{hybrid_ens_metrics['gini']:.4f}",
            "KS Stat": f"{hybrid_ens_metrics['ks']:.4f}",
            "ECE (10)": f"{hybrid_ens_metrics['ece_10']:.4f}",
            "Brier Score": f"{hybrid_ens_metrics['brier']:.4f}",
            "Log-Loss": f"{hybrid_ens_metrics['log_loss']:.4f}",
        },
        {
            "Mô hình (Model Variant)": "Alternative-Only Model (Chỉ dữ liệu thay thế)",
            "Số đặc trưng": len(res_alt.get("feature_cols", [])),
            "ROC-AUC": f"{alt_ens_metrics['roc_auc']:.4f}",
            "PR-AUC": f"{alt_ens_metrics['pr_auc']:.4f}",
            "Gini Index": f"{alt_ens_metrics['gini']:.4f}",
            "KS Stat": f"{alt_ens_metrics['ks']:.4f}",
            "ECE (10)": f"{alt_ens_metrics['ece_10']:.4f}",
            "Brier Score": f"{alt_ens_metrics['brier']:.4f}",
            "Log-Loss": f"{alt_ens_metrics['log_loss']:.4f}",
        },
    ]

    df_summary = pd.DataFrame(rows_summary)

    print("\n==========================================================================================")
    print("📊 BẢNG SO SÁNH TỔNG QUAN HIỆU NĂNG MÔ HÌNH ENSEMBLE TÊN TẬP TEST")
    print("==========================================================================================")
    print(df_summary.to_string(index=False))

    # 4. Bảng Chi Tiết Từng Thuật Toán Của Cả 2 Mô Hình
    rows_detail = []
    for model_name, m in res_hybrid["test_metrics"].items():
        rows_detail.append({
            "Variant": "Hybrid",
            "Model": model_name,
            "Weight": f"{res_hybrid['optimal_weights'].get(model_name, 0.0):.4f}" if model_name in res_hybrid['optimal_weights'] else "-",
            "ROC-AUC": f"{m['roc_auc']:.4f}",
            "PR-AUC": f"{m['pr_auc']:.4f}",
            "Gini": f"{m['gini']:.4f}",
            "KS": f"{m['ks']:.4f}",
            "ECE": f"{m['ece_10']:.4f}",
        })
    for model_name, m in res_alt["test_metrics"].items():
        rows_detail.append({
            "Variant": "Alternative-Only",
            "Model": model_name,
            "Weight": f"{res_alt['optimal_weights'].get(model_name, 0.0):.4f}" if model_name in res_alt['optimal_weights'] else "-",
            "ROC-AUC": f"{m['roc_auc']:.4f}",
            "PR-AUC": f"{m['pr_auc']:.4f}",
            "Gini": f"{m['gini']:.4f}",
            "KS": f"{m['ks']:.4f}",
            "ECE": f"{m['ece_10']:.4f}",
        })

    df_detail = pd.DataFrame(rows_detail)
    print("\n==========================================================================================")
    print("📋 BẢNG CHI TIẾT HIỆU NĂNG TỪNG MÔ HÌNH THÀNH PHẦN (SUB-MODELS)")
    print("==========================================================================================")
    print(df_detail.to_string(index=False))

    # Save artifacts
    artifacts_dir = Path("artifacts/models")
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    hybrid_path = artifacts_dir / "hybrid_ensemble_model.joblib"
    alt_path = artifacts_dir / "alternative_only_model.joblib"

    joblib.dump(res_hybrid, hybrid_path, compress=3)
    joblib.dump(res_alt, alt_path, compress=3)

    print("\n==========================================================================================")
    print(f"✅ ĐÃ LƯU HYBRID ENSEMBLE MODEL TẠI: {hybrid_path.resolve()}")
    print(f"✅ ĐÃ LƯU ALTERNATIVE-ONLY MODEL TẠI: {alt_path.resolve()}")
    print("==========================================================================================")


if __name__ == "__main__":
    main()
