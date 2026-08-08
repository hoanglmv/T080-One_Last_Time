"""Script to generate notebooks/07_alternative_data_only_model_training.ipynb."""

import json
from pathlib import Path

notebook_path = Path("notebooks/07_alternative_data_only_model_training.ipynb")

cells = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# 🏦 Notebook 07: Huấn Luyện Mô Hình Alternative Credit Scoring Chỉ Sử Dụng Dữ Liệu Thay Thế (Alternative-Data-Only Model)\n",
            "\n",
            "## 📌 1. Đặt Vấn Đề & Mục Tiêu Phân Tích\n",
            "Notebook này triển khai quy trình huấn luyện và đánh giá mô hình rủi ro tín dụng **chuyên biệt chỉ sử dụng Dữ Liệu Thay Thế (Alternative Data Only)** cho nhóm khách hàng **Thin-file** và **Unbanked/Underbanked**:\n",
            "1. **Tách biệt dữ liệu**: Loại bỏ 100% các biến tín dụng truyền thống (`EXT_SOURCE_1/2/3`, thông tin dư nợ bureau `bureau.csv`, khoản nợ cũ `previous_application.csv`, hạn mức `AMT_CREDIT`, `AMT_ANNUITY`).\n",
            "2. **Tập đặc trưng thay thế**: Khai thác các đặc trưng hành vi di động/thiết bị (`DAYS_LAST_PHONE_CHANGE`, `FLAG_EMP_PHONE`), nhân khẩu học phi truyền thống (`NAME_HOUSING_TYPE`, `OCCUPATION_TYPE`), chỉ số mạng lưới xã hội (`DEF_30_CNT_SOCIAL_CIRCLE`), và đặc trưng kinh tế - xã hội thay thế (`FE_INCOME_PER_PERSON`).\n",
            "3. **Huấn luyện đa mô hình & Blending Ensemble**: Đào tạo `LogisticRegression`, `LightGBM`, `XGBoost`, `CatBoost` và tối ưu hóa trọng số blending.\n",
            "4. **So sánh Benchmark với Mô hình Hybrid**: Đo lường chỉ số Lift và khoảng cách hiệu năng giữa **Hybrid Model (Traditional + Alternative)** và **Alternative-Only Model**.\n",
            "5. **Chẩn đoán tính công bằng (Fairness Audit)**: Đánh giá sai số và mức độ công bằng giữa các nhóm giới tính (`CODE_GENDER`)."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import sys\n",
            "import warnings\n",
            "from pathlib import Path\n",
            "\n",
            "warnings.filterwarnings('ignore')\n",
            "\n",
            "# Thêm root directory vào sys.path để import các module src\n",
            "root_dir = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()\n",
            "if str(root_dir) not in sys.path:\n",
            "    sys.path.insert(0, str(root_dir))\n",
            "\n",
            "import numpy as np\n",
            "import pandas as pd\n",
            "import matplotlib.pyplot as plt\n",
            "import seaborn as sns\n",
            "import joblib\n",
            "\n",
            "from src.credit_scoring.features import (\n",
            "    build_home_credit_features,\n",
            "    ALTERNATIVE_ONLY_RAW_FEATURES,\n",
            "    ALTERNATIVE_ONLY_ENGINEERED_FEATURES\n",
            ")\n",
            "from src.credit_scoring.ensemble_pipeline import train_ensemble_pipeline\n",
            "from src.credit_scoring.metrics import credit_metrics, fairness_report\n",
            "\n",
            "# Thiết lập style giao diện trực quan\n",
            "plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')\n",
            "plt.rcParams['font.size'] = 11\n",
            "plt.rcParams['figure.titlesize'] = 14\n",
            "pd.set_option('display.max_columns', None)\n",
            "\n",
            "print('✓ Đã nạp thành công các thư viện và module hệ thống.')"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "---\n",
            "## 🔍 2. Kiểm Trực & Tải Dữ Liệu Thay Thế (Alternative Data Loading & Audit)"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "data_dir = root_dir / 'data/raw/home-credit-default-risk'\n",
            "print(f'⏳ Đang tải tập đặc trưng Dữ liệu Thay thế từ: {data_dir}')\n",
            "\n",
            "# Tải dữ liệu với feature_set=\"alternative_only\"\n",
            "df_alt = build_home_credit_features(data_dir, feature_set='alternative_only', sample_size=30000)\n",
            "\n",
            "print(f'✓ Tổng số dòng (applicants): {len(df_alt):,}')\n",
            "print(f'✓ Tổng số cột đặc trưng: {df_alt.shape[1]}')\n",
            "print(f'✓ Tỷ lệ nợ xấu (Default Rate): {df_alt[\"TARGET\"].mean():.2%}')\n",
            "\n",
            "print('\\n=== Danh Sách Các Biến Thay Thế Trong Tập Dữ Liệu ===')\n",
            "print(list(df_alt.columns))\n",
            "\n",
            "# Xác minh không có biến truyền thống rò rỉ\n",
            "excluded_leakage = ['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'AMT_CREDIT', 'AMT_ANNUITY', 'BUREAU_CREDIT_COUNT']\n",
            "leakage_found = [c for c in excluded_leakage if c in df_alt.columns]\n",
            "assert len(leakage_found) == 0, f'⚠️ CẢNH BÁO: Phát hiện biến truyền thống rò rỉ: {leakage_found}'\n",
            "print('✓ XÁC NHẬN: Không chứa bất kỳ biến truyền thống hay điểm bureau nào (Data Leakage Free).')"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "---\n",
            "## 🚀 3. Huấn Luyện Bộ Mô Hình Ensemble Chỉ Dùng Dữ Liệu Thay Thế"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "print('⚡ Khởi chạy Pipeline huấn luyện đa mô hình (Alternative-Only Ensemble)...')\n",
            "\n",
            "# Huấn luyện pipeline đầy đủ trên tập dữ liệu thay thế\n",
            "res_alt = train_ensemble_pipeline(\n",
            "    data_dir=data_dir,\n",
            "    feature_set='alternative_only',\n",
            "    sample_size=50000,  # Có thể bỏ sample_size để huấn luyện toàn bộ dữ liệu\n",
            "    seed=42\n",
            ")\n",
            "\n",
            "alt_metrics = res_alt['test_metrics']\n",
            "alt_weights = res_alt['optimal_weights']\n",
            "\n",
            "# Tạo bảng hiển thị kết quả các mô hình thành phần\n",
            "rows_alt = []\n",
            "for model_name, m in alt_metrics.items():\n",
            "    w_str = f\"{alt_weights.get(model_name, 0.0):.4f}\" if model_name in alt_weights else \"-\"\n",
            "    rows_alt.append({\n",
            "        'Mô hình (Model)': model_name,\n",
            "        'Trọng số (Blending Weight)': w_str,\n",
            "        'ROC-AUC': f\"{m['roc_auc']:.4f}\",\n",
            "        'PR-AUC': f\"{m['pr_auc']:.4f}\",\n",
            "        'Gini Index': f\"{m['gini']:.4f}\",\n",
            "        'KS Stat': f\"{m['ks']:.4f}\",\n",
            "        'ECE (10)': f\"{m['ece_10']:.4f}\",\n",
            "        'Log-Loss': f\"{m['log_loss']:.4f}\"\n",
            "    })\n",
            "\n",
            "df_alt_summary = pd.DataFrame(rows_alt)\n",
            "print('\\n=== BẢNG SO SÁNH CÁC MÔ HÌNH THÀNH PHẦN (ALTERNATIVE-ONLY) ===')\n",
            "df_alt_summary"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "---\n",
            "## 📊 4. So Sánh Benchmark: Alternative-Only Model vs Hybrid Model"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "print('⚡ Huấn luyện Mô hình Hybrid (Traditional + Alternative) làm Baseline Benchmark...')\n",
            "\n",
            "res_hybrid = train_ensemble_pipeline(\n",
            "    data_dir=data_dir,\n",
            "    feature_set='serving',\n",
            "    sample_size=50000,\n",
            "    seed=42\n",
            ")\n",
            "\n",
            "hybrid_ens = res_hybrid['test_metrics']['Ensemble_Calibrated']\n",
            "alt_ens = res_alt['test_metrics']['Ensemble_Calibrated']\n",
            "\n",
            "# Bảng so sánh tổng quan giữa 2 phiên bản mô hình\n",
            "comp_data = [\n",
            "    {\n",
            "        'Phiên bản Mô hình (Model Variant)': 'Hybrid Model (Traditional + Alternative)',\n",
            "        'Số biến (Features)': len(res_hybrid.get('feature_cols', [])),\n",
            "        'ROC-AUC': f\"{hybrid_ens['roc_auc']:.4f}\",\n",
            "        'PR-AUC': f\"{hybrid_ens['pr_auc']:.4f}\",\n",
            "        'Gini Index': f\"{hybrid_ens['gini']:.4f}\",\n",
            "        'KS Stat': f\"{hybrid_ens['ks']:.4f}\",\n",
            "        'ECE (10)': f\"{hybrid_ens['ece_10']:.4f}\",\n",
            "        'Brier Score': f\"{hybrid_ens['brier']:.4f}\"\n",
            "    },\n",
            "    {\n",
            "        'Phiên bản Mô hình (Model Variant)': 'Alternative-Only Model (Chỉ dữ liệu thay thế)',\n",
            "        'Số biến (Features)': len(res_alt.get('feature_cols', [])),\n",
            "        'ROC-AUC': f\"{alt_ens['roc_auc']:.4f}\",\n",
            "        'PR-AUC': f\"{alt_ens['pr_auc']:.4f}\",\n",
            "        'Gini Index': f\"{alt_ens['gini']:.4f}\",\n",
            "        'KS Stat': f\"{alt_ens['ks']:.4f}\",\n",
            "        'ECE (10)': f\"{alt_ens['ece_10']:.4f}\",\n",
            "        'Brier Score': f\"{alt_ens['brier']:.4f}\"\n",
            "    }\n",
            "]\n",
            "\n",
            "df_comp = pd.DataFrame(comp_data)\n",
            "print('\\n=== BẢNG SO SÁNH HIỆU NĂNG ENSEMBLE TÊN TẬP TEST ===')\n",
            "df_comp"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "---\n",
            "## 📈 5. Trực Quan Hóa Đồ Thị So Sánh (Performance Visualization)"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "fig, axes = plt.subplots(1, 2, figsize=(14, 5))\n",
            "\n",
            "# Đồ thị 1: So sánh ROC-AUC, Gini, KS\n",
            "metrics_names = ['ROC-AUC', 'Gini Index', 'KS Stat']\n",
            "hybrid_vals = [hybrid_ens['roc_auc'], hybrid_ens['gini'], hybrid_ens['ks']]\n",
            "alt_vals = [alt_ens['roc_auc'], alt_ens['gini'], alt_ens['ks']]\n",
            "\n",
            "x = np.arange(len(metrics_names))\n",
            "width = 0.35\n",
            "\n",
            "axes[0].bar(x - width/2, hybrid_vals, width, label='Hybrid Model', color='#1f77b4', alpha=0.85)\n",
            "axes[0].bar(x + width/2, alt_vals, width, label='Alternative-Only Model', color='#ff7f0e', alpha=0.85)\n",
            "axes[0].set_ylabel('Score')\n",
            "axes[0].set_title('So Sánh Chỉ Số Phân Tách Rủi Ro (Discrimination Metrics)')\n",
            "axes[0].set_xticks(x)\n",
            "axes[0].set_xticklabels(metrics_names)\n",
            "axes[0].set_ylim(0, 1.0)\n",
            "axes[0].legend()\n",
            "for i in range(len(metrics_names)):\n",
            "    axes[0].text(i - width/2, hybrid_vals[i] + 0.02, f'{hybrid_vals[i]:.3f}', ha='center', fontsize=10)\n",
            "    axes[0].text(i + width/2, alt_vals[i] + 0.02, f'{alt_vals[i]:.3f}', ha='center', fontsize=10)\n",
            "\n",
            "# Đồ thị 2: Trọng số Blending Ensemble của Alternative-Only Model\n",
            "model_names = list(alt_weights.keys())\n",
            "weights_vals = list(alt_weights.values())\n",
            "axes[1].pie(weights_vals, labels=model_names, autopct='%1.1f%%', startangle=140, colors=sns.color_palette('Set2'))\n",
            "axes[1].set_title('Phân Phối Trọng Số Blending (Alternative-Only Ensemble)')\n",
            "\n",
            "plt.tight_layout()\n",
            "plt.show()"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "---\n",
            "## ⚖️ 6. Chẩn Đoán Tính Công Bằng (Fairness Audit across Gender)"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Chẩn đoán tính công bằng giữa Nam và Nữ trên Alternative-Only Model\n",
            "y_test = res_alt['y_test']\n",
            "y_prob_alt = res_alt['test_ensemble_calibrated']\n",
            "# Lấy nhóm giới tính từ tập test\n",
            "data_train = pd.read_csv(data_dir / 'application_train.csv', nrows=50000)\n",
            "gender_groups = data_train['CODE_GENDER'].iloc[res_alt['split'].test].to_numpy()\n",
            "\n",
            "fair_report = fairness_report(y_test, y_prob_alt, gender_groups, threshold=0.10)\n",
            "\n",
            "print('=== KẾT QUẢ CHẨN ĐOÁN TÍNH CÔNG BẰNG (FAIRNESS REPORT) ===')\n",
            "print(f\"Khả dụng: {fair_report['available']}\")\n",
            "print(f\"Chênh lệch Tỷ lệ Báo động nhầm (FPR Gap): {fair_report['gaps'].get('max_min_fpr_gap', 0.0):.4f}\")\n",
            "print(f\"Chênh lệch Tỷ lệ Bắt đúng Nợ xấu (TPR Gap): {fair_report['gaps'].get('max_min_tpr_gap', 0.0):.4f}\")\n",
            "\n",
            "pd.DataFrame(fair_report['groups'])"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "---\n",
            "## 💾 7. Lưu Xuất Mô Hình (Save Artifacts)\n",
            "\n",
            "Lưu trữ mô hình Alternative-Only đã huấn luyện hoàn chỉnh vào thư mục `artifacts/models/`."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "out_path = root_dir / 'artifacts/models/alternative_only_model.joblib'\n",
            "out_path.parent.mkdir(parents=True, exist_ok=True)\n",
            "\n",
            "joblib.dump(res_alt, out_path, compress=3)\n",
            "print(f'✅ Đã lưu mô hình Alternative-Only Ensemble thành công tại: {out_path.resolve()}')"
        ]
    }
]

notebook_json = {
    "cells": cells,
    "metadata": {
        "language_info": {
            "name": "python"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 2
}

with open(notebook_path, "w", encoding="utf-8") as f:
    json.dump(notebook_json, f, ensure_ascii=False, indent=2)

print(f"✅ Đã khởi tạo notebook thành công tại: {notebook_path.resolve()}")
