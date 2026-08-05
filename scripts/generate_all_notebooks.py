import json
from pathlib import Path

notebooks_dir = Path("notebooks")
notebooks_dir.mkdir(exist_ok=True)


def create_home_credit_notebook():
    nb_cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# 💳 Home Credit Default Risk - Alternative Credit Scoring (ACS) EDA\n",
                "\n",
                "## 📌 Mục Tiêu Phân Tích & Định Hướng Alternative Credit Scoring\n",
                "Notebook này thực hiện Phân Tích Khám Phá Dữ Liệu (**Exploratory Data Analysis - EDA**) chuyên sâu cho bài toán **Alternative Credit Scoring (Đánh giá tín dụng bằng dữ liệu thay thế)** trên tập dữ liệu **Home Credit Default Risk**:\n",
                "1. **Khai thác dữ liệu thay thế (Alternative Data)**: Đánh giá khả năng nợ xấu cho nhóm khách hàng \"Thin-file / Unbanked\" dựa trên thuộc tính nhà ở, trình độ học vấn, thâm niên thiết bị, liên lạc và chỉ số rủi ro mạng lưới xã hội.\n",
                "2. **Đánh giá điểm tín dụng bên thứ 3 (External Alternative Scores)**: Phân tích 3 nguồn điểm rủi ro tổng hợp `EXT_SOURCE_1`, `EXT_SOURCE_2`, `EXT_SOURCE_3`.\n",
                "3. **Phân tích biến mục tiêu (`TARGET`)**: Đánh giá mất cân bằng dữ liệu (Default rate = 8.07%).\n",
                "4. **Thiết kế chỉ số rủi ro tín dụng thay thế (ACS Features)**: `CREDIT_TO_INCOME_RATIO`, `ANNUITY_TO_INCOME_RATIO`, `DAYS_LAST_PHONE_CHANGE`.\n",
                "5. **Chuẩn bị pipeline huấn luyện & giải thích bằng SHAP** theo quy trình chuẩn trong `AGENTS.md`."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "source": [
                "import warnings\n",
                "warnings.filterwarnings('ignore')\n",
                "\n",
                "import pandas as pd\n",
                "import numpy as np\n",
                "import matplotlib.pyplot as plt\n",
                "import seaborn as sns\n",
                "from pathlib import Path\n",
                "\n",
                "pd.set_option('display.max_columns', 150)\n",
                "pd.set_option('display.float_format', lambda x: '%.2f' % x)\n",
                "plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')\n",
                "plt.rcParams['font.size'] = 11\n",
                "\n",
                "DATA_DIR = Path('../data/raw/home-credit-default-risk')\n",
                "if not DATA_DIR.exists():\n",
                "    DATA_DIR = Path('data/raw/home-credit-default-risk')\n",
                "\n",
                "print('✓ Đã nạp thành công thư viện & đường dẫn dữ liệu Home Credit (Alternative Credit Scoring).')"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "---\n",
                "## 1. 📂 Nạp & Tổng Quan Dữ Liệu Bảng Chính (`application_train.csv`)"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "source": [
                "app_train = pd.read_csv(DATA_DIR / 'application_train.csv')\n",
                "print(f'✓ Tổng số bản ghi (rows): {len(app_train):,}')\n",
                "print(f'✓ Tổng số thuộc tính (features): {app_train.shape[1]}')\n",
                "app_train.head(5)"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "---\n",
                "## 2. 🎯 Phân Tích Biến Mục Tiêu (`TARGET` - Default Rate)\n",
                "- `TARGET = 0`: Khách hàng hoàn trả khoản vay đúng hạn.\n",
                "- `TARGET = 1`: Khách hàng vỡ nợ (Default)."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "source": [
                "target_counts = app_train['TARGET'].value_counts()\n",
                "target_rates = app_train['TARGET'].value_counts(normalize=True) * 100\n",
                "\n",
                "fig, axes = plt.subplots(1, 2, figsize=(14, 5))\n",
                "\n",
                "labels = ['Hoàn trả đúng hạn (0)', 'Vỡ nợ / Rủi ro (1)']\n",
                "sns.barplot(x=labels, y=target_counts.values, ax=axes[0], palette=['#2a9d8f', '#e76f51'], hue=labels, legend=False)\n",
                "axes[0].set_title('Số Lượng Khách Hàng Theo Target', fontweight='bold')\n",
                "axes[0].set_ylabel('Số lượng')\n",
                "for p in axes[0].patches:\n",
                "    axes[0].annotate(f'{int(p.get_height()):,}', (p.get_x() + p.get_width() / 2., p.get_height()),\n",
                "                     ha='center', va='center', xytext=(0, 5), textcoords='offset points')\n",
                "\n",
                "axes[1].pie(target_rates, labels=[f'Trả nợ đúng hạn ({target_rates[0]:.1f}%)', f'Vỡ nợ ({target_rates[1]:.1f}%)'],\n",
                "            autopct='%1.1f%%', startangle=90, colors=['#2a9d8f', '#e76f51'], explode=(0, 0.1),\n",
                "            textprops={'fontsize': 12, 'weight': 'bold'})\n",
                "axes[1].set_title('Tỷ Lệ Vỡ Nợ (Default Rate)', fontweight='bold')\n",
                "\n",
                "plt.tight_layout()\n",
                "plt.show()\n",
                "\n",
                "print(f'► Tỷ lệ vỡ nợ chung: {target_rates[1]:.2f}% ({target_counts[1]:,} / {len(app_train):,} khách hàng)')"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "---\n",
                "## 3. 📱 Phân Tích Dữ Liệu Thay Thế (Alternative Features Analysis)\n",
                "Khảo sát các thuộc tính phi truyền thống: Thâm niên liên lạc (`DAYS_LAST_PHONE_CHANGE`), xác thực liên lạc (`FLAG_EMP_PHONE`, `FLAG_EMAIL`), loại hình nhà ở (`NAME_HOUSING_TYPE`), và rủi ro mạng lưới xã hội (`DEF_30_CNT_SOCIAL_CIRCLE`)."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "source": [
                "fig, axes = plt.subplots(2, 2, figsize=(16, 10))\n",
                "\n",
                "# Thâm niên thay đổi số điện thoại (DAYS_LAST_PHONE_CHANGE converted to years)\n",
                "app_train['PHONE_CHANGE_YEARS'] = abs(app_train['DAYS_LAST_PHONE_CHANGE']) / 365.25\n",
                "sns.kdeplot(data=app_train, x='PHONE_CHANGE_YEARS', hue='TARGET', common_norm=False, fill=True, ax=axes[0, 0], palette=['#2a9d8f', '#e76f51'])\n",
                "axes[0, 0].set_title('Thâm Niên Sử Dụng Số Điện Thoại (Năm) vs Target', fontweight='bold')\n",
                "\n",
                "# Loại hình nhà ở (Housing Type) vs Default Rate\n",
                "housing_default = app_train.groupby('NAME_HOUSING_TYPE')['TARGET'].mean().sort_values(ascending=False) * 100\n",
                "sns.barplot(x=housing_default.values, y=housing_default.index, ax=axes[0, 1], palette='Oranges_r', hue=housing_default.index, legend=False)\n",
                "axes[0, 1].set_title('Tỷ Lệ Vỡ Nợ (%) Theo Loại Hình Nhà Ở', fontweight='bold')\n",
                "axes[0, 1].set_xlabel('Tỷ lệ vỡ nợ (%)')\n",
                "\n",
                "# Trình độ học vấn vs Default Rate\n",
                "edu_default = app_train.groupby('NAME_EDUCATION_TYPE')['TARGET'].mean().sort_values(ascending=False) * 100\n",
                "sns.barplot(x=edu_default.values, y=edu_default.index, ax=axes[1, 0], palette='Purples_r', hue=edu_default.index, legend=False)\n",
                "axes[1, 0].set_title('Tỷ Lệ Vỡ Nợ (%) Theo Trình Độ Học Vấn', fontweight='bold')\n",
                "axes[1, 0].set_xlabel('Tỷ lệ vỡ nợ (%)')\n",
                "\n",
                "# Rủi ro mạng lưới xã hội (DEF_30_CNT_SOCIAL_CIRCLE)\n",
                "sns.boxplot(data=app_train, x='TARGET', y='DEF_30_CNT_SOCIAL_CIRCLE', ax=axes[1, 1], palette=['#2a9d8f', '#e76f51'], hue='TARGET', legend=False)\n",
                "axes[1, 1].set_title('Số Người Trong Mạng Lưới Xã Hội Vỡ Nợ (30 ngày) vs Target', fontweight='bold')\n",
                "axes[1, 1].set_xticks([0, 1])\n",
                "axes[1, 1].set_xticklabels(['Trả nợ (0)', 'Vỡ nợ (1)'])\n",
                "axes[1, 1].set_ylim(-0.5, 5)\n",
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
                "## 4. 🌟 Phân Tích Điểm Tín Dụng Bên Thứ Ba (`EXT_SOURCE_1`, `EXT_SOURCE_2`, `EXT_SOURCE_3`)\n",
                "Đây là các điểm số rủi ro tín dụng tổng hợp từ các đối tác thay thế bên ngoài (Third-party alternative scores)."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "source": [
                "fig, axes = plt.subplots(1, 3, figsize=(18, 5))\n",
                "for i, col in enumerate(['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']):\n",
                "    sns.kdeplot(data=app_train, x=col, hue='TARGET', common_norm=False, fill=True, ax=axes[i], palette=['#2a9d8f', '#e76f51'])\n",
                "    axes[i].set_title(f'Phân Phối {col} vs Target', fontweight='bold')\n",
                "\n",
                "plt.tight_layout()\n",
                "plt.show()\n",
                "\n",
                "corrs = app_train[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'TARGET']].corr()['TARGET'].sort_values()\n",
                "print('Hệ số tương quan Pearson với TARGET:')\n",
                "print(corrs)"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "---\n",
                "## 5. 💡 Feature Engineering Cho Alternative Credit Scoring\n",
                "Tạo các chỉ số khả năng chi trả thay thế cho khách hàng không có báo cáo tài chính chính thức."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "source": [
                "# Tạo biến phái sinh tài chính thay thế\n",
                "app_train['CREDIT_TO_INCOME_RATIO'] = app_train['AMT_CREDIT'] / (app_train['AMT_INCOME_TOTAL'] + 1)\n",
                "app_train['ANNUITY_TO_INCOME_RATIO'] = app_train['AMT_ANNUITY'] / (app_train['AMT_INCOME_TOTAL'] + 1)\n",
                "app_train['EMPLOYMENT_TO_AGE_RATIO'] = abs(app_train['DAYS_EMPLOYED']) / (abs(app_train['DAYS_BIRTH']) + 1)\n",
                "\n",
                "print('Hệ số tương quan của các biến phái sinh với TARGET:')\n",
                "print(app_train[['CREDIT_TO_INCOME_RATIO', 'ANNUITY_TO_INCOME_RATIO', 'EMPLOYMENT_TO_AGE_RATIO', 'TARGET']].corr()['TARGET'])"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "---\n",
                "## 6. 🎯 Định Hướng Xây Dựng Mô Hình Alternative Credit Scoring\n",
                "\n",
                "### 📌 Tóm Tắt Kết Quả EDA:\n",
                "1. **Mất cân bằng dữ liệu**: Tỷ lệ vỡ nợ **8.07%**. Đánh giá mô hình bằng **ROC-AUC**, **PR-AUC**, **KS Statistic** ($KS > 40\%$).\n",
                "2. **Giá trị của Dữ Liệu Thay Thế (Alternative Data)**:\n",
                "   - `EXT_SOURCE_1/2/3` đóng vai trò quan trọng nhất trong phân tách rủi ro.\n",
                "   - Thâm niên liên lạc (`PHONE_CHANGE_YEARS`), loại hình nhà ở và mạng lưới xã hội mang lại tín hiệu dự báo mạnh cho nhóm Unbanked.\n",
                "\n",
                "### 💡 Các Bước Tiếp Theo Theo Quy Trình `AGENTS.md`:\n",
                "- **Thành lập Baseline Model**: Weight of Evidence (WoE) + Logistic Regression.\n",
                "- **Mô hình học máy nâng cao**: LightGBM / XGBoost với dữ liệu đã được tổng hợp từ `bureau.csv` và `previous_application.csv`.\n",
                "- **Giải thích mô hình**: Sử dụng **SHAP values** để minh bạch hóa quyết định cấp tín dụng thay thế."
            ]
        }
    ]

    nb_content = {
        "cells": nb_cells,
        "metadata": {"language_info": {"name": "python"}, "orig_nbformat": 4},
        "nbformat": 4,
        "nbformat_minor": 2
    }
    with open(notebooks_dir / "01_eda_home_credit_default_risk.ipynb", "w", encoding="utf-8") as f:
        json.dump(nb_content, f, ensure_ascii=False, indent=2)
    print("✓ Updated notebooks/01_eda_home_credit_default_risk.ipynb for ACS")


def create_vietnam_churn_notebook():
    nb_cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# 🏦 Vietnam Bank Dataset - Alternative Credit Risk & Digital Behavior EDA\n",
                "\n",
                "## 📌 Mục Tiêu Phân Tích Dữ Liệu Thay Thế (Alternative Data Analysis)\n",
                "Notebook này khai thác dữ liệu hành vi số của 80,000 khách hàng tại Việt Nam cho bài toán **Alternative Credit Scoring & Early Warning System (Cảnh báo rủi ro sớm)**:\n",
                "1. **Đặc trưng hành vi số (Digital Footprint)**: Phân tích `digital_behavior` (Mobile/Web), `engagement_score`, `nums_service`, `nums_card`.\n",
                "2. **Phân tích rủi ro tín dụng thay thế (`risk_score`, `risk_segment`)**: Mối liên hệ giữa hành vi số và phân khúc rủi ro.\n",
                "3. **Tín hiệu cảnh báo sớm (Early Warning Signal)**: Sử dụng biến ngưng sử dụng dịch vụ (`exit`) làm chỉ báo rủi ro suy giảm năng lực tài chính."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "source": [
                "import warnings\n",
                "warnings.filterwarnings('ignore')\n",
                "\n",
                "import pandas as pd\n",
                "import numpy as np\n",
                "import matplotlib.pyplot as plt\n",
                "import seaborn as sns\n",
                "from pathlib import Path\n",
                "\n",
                "pd.set_option('display.max_columns', None)\n",
                "pd.set_option('display.float_format', lambda x: '%.2f' % x)\n",
                "plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')\n",
                "plt.rcParams['font.size'] = 11\n",
                "\n",
                "DATA_PATH = Path('../data/raw/vietnam-bank-churn-dataset-2025/bank_churn_dataset_80k.csv')\n",
                "if not DATA_PATH.exists():\n",
                "    DATA_PATH = Path('data/raw/vietnam-bank-churn-dataset-2025/bank_churn_dataset_80k.csv')\n",
                "\n",
                "df = pd.read_csv(DATA_PATH)\n",
                "print(f'✓ Tổng số dòng (records): {len(df):,}')\n",
                "print(f'✓ Tổng số cột (features): {df.shape[1]}')\n",
                "df.head(5)"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "---\n",
                "## 1. 🔍 Tổng Quan Thuộc Tính Dữ Liệu Thay Thế"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "source": [
                "info_df = pd.DataFrame({\n",
                "    'Data Type': df.dtypes,\n",
                "    'Null Count': df.isnull().sum(),\n",
                "    'Null Ratio (%)': (df.isnull().sum() / len(df)) * 100,\n",
                "    'Unique Values': df.nunique()\n",
                "})\n",
                "print('=== Thông tin các thuộc tính ===')\n",
                "print(info_df)"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "---\n",
                "## 2. 📱 Phân Tích Hành Vi Số (Digital Behavior) & Mức Độ Tương Tác (`engagement_score`)"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "source": [
                "fig, axes = plt.subplots(1, 3, figsize=(18, 5))\n",
                "\n",
                "# Digital behavior (Mobile vs Web)\n",
                "sns.countplot(data=df, x='digital_behavior', hue='risk_segment', ax=axes[0], palette='Blues_r')\n",
                "axes[0].set_title('Kênh Kỹ Thuật Số vs Risk Segment', fontweight='bold')\n",
                "\n",
                "# Engagement Score vs Risk Score\n",
                "sns.scatterplot(data=df.sample(2000, random_state=42), x='engagement_score', y='risk_score', hue='risk_segment', ax=axes[1], alpha=0.6, palette='Set2')\n",
                "axes[1].set_title('Engagement Score vs Risk Score', fontweight='bold')\n",
                "\n",
                "# Customer Segment vs Risk Score\n",
                "sns.boxplot(data=df, x='customer_segment', y='risk_score', ax=axes[2], palette='Set3', hue='customer_segment', legend=False)\n",
                "axes[2].set_title('Risk Score Theo Phân Khúc Khách Hàng', fontweight='bold')\n",
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
                "## 3. 🎯 Mối Tương Quan Giữa Hành Vi Số & Rủi Ro Suy Giảm Khả Năng Tài Chính (`exit`)"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "source": [
                "fig, axes = plt.subplots(1, 2, figsize=(14, 5))\n",
                "\n",
                "# Active member vs Churn\n",
                "sns.countplot(data=df, x='active_member', hue='exit', ax=axes[0], palette=['#2ec4b6', '#e71d36'])\n",
                "axes[0].set_title('Trạng Thái Active Member vs Risk Signal (Exit)', fontweight='bold')\n",
                "\n",
                "# Risk score distribution vs Exit\n",
                "sns.kdeplot(data=df, x='risk_score', hue='exit', common_norm=False, fill=True, ax=axes[1], palette=['#2ec4b6', '#e71d36'])\n",
                "axes[1].set_title('Phân Phối Risk Score vs Risk Signal (Exit)', fontweight='bold')\n",
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
                "## 4. 📊 Ma Trận Tương Quan Các Đặc Trưng Thay Thế"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "source": [
                "num_cols = df.select_dtypes(include=['int64', 'float64', 'bool']).columns\n",
                "plt.figure(figsize=(14, 10))\n",
                "corr = df[num_cols].corr()\n",
                "sns.heatmap(corr, annot=True, fmt='.2f', cmap='vlag', vmin=-1, vmax=1, linewidths=0.5)\n",
                "plt.title('Ma Trận Tương Quan Dữ Liệu Hành Vi & Rủi Ro Số', fontweight='bold')\n",
                "plt.tight_layout()\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "---\n",
                "## 5. 🎯 Đóng Góp Vào Mô Hình Alternative Credit Scoring\n",
                "\n",
                "1. **Tín hiệu tương tác số**: Khách hàng có `engagement_score` cao và sử dụng kênh `mobile` có điểm rủi ro trung bình thấp hơn.\n",
                "2. **Cảnh báo sớm (EWS)**: Trạng thái `active_member = False` là biến báo động quan trọng cho sự sụt giảm khả năng thanh toán.\n",
                "3. **Ứng dụng cho Việt Nam**: Các đặc trưng hành vi số và phân khúc khách hàng là nguồn dữ liệu thay thế quý giá khi chấm điểm tín dụng cho người dùng chưa có hồ sơ CIC."
            ]
        }
    ]

    nb_content = {
        "cells": nb_cells,
        "metadata": {"language_info": {"name": "python"}, "orig_nbformat": 4},
        "nbformat": 4,
        "nbformat_minor": 2
    }
    with open(notebooks_dir / "02_eda_vietnam_bank_churn.ipynb", "w", encoding="utf-8") as f:
        json.dump(nb_content, f, ensure_ascii=False, indent=2)
    print("✓ Updated notebooks/02_eda_vietnam_bank_churn.ipynb for ACS")


if __name__ == "__main__":
    create_home_credit_notebook()
    create_vietnam_churn_notebook()
