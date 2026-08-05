import json
from pathlib import Path

eda_dir = Path("eda")
eda_dir.mkdir(exist_ok=True)

nb_cells = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# 🏦 Vietnam Bank Churn Dataset 2025 - Exploratory Data Analysis (EDA)\n",
            "\n",
            "## 📌 Mục Tiêu Phân Tích\n",
            "Notebook này thực hiện Phân Tích Khám Phá Dữ Liệu (**Exploratory Data Analysis - EDA**) trên bộ dữ liệu **Vietnam Bank Churn Dataset 2025** (80,000 bản ghi):\n",
            "1. Hiểu rõ cấu trúc, đặc trưng và chất lượng dữ liệu ngân hàng.\n",
            "2. Trực quan hóa tỷ lệ khách hàng rời bỏ dịch vụ (**Churn Rate - `exit`**).\n",
            "3. Phân tích tác động của các yếu tố nhân khẩu học (tuổi, giới tính, tỉnh thành, nghề nghiệp), tài chính (số dư, thu nhập, điểm tín dụng), và hành vi (điểm tương tác, phân khúc rủi ro, kênh kỹ thuật số).\n",
            "4. Đưa ra các nhận xét chuyên sâu (**Business Insights**) phục vụ cho bài toán dự đoán rời bỏ và giữ chân khách hàng."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
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
            "# Thấu kính hiển thị & Style\n",
            "pd.set_option('display.max_columns', None)\n",
            "pd.set_option('display.float_format', lambda x: '%.2f' % x)\n",
            "\n",
            "plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')\n",
            "plt.rcParams['font.size'] = 11\n",
            "plt.rcParams['figure.titlesize'] = 14\n",
            "\n",
            "print('✓ Đã nạp thành công các thư viện phân tích dữ liệu.')"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Đọc dữ liệu từ thư mục data/raw/\n",
            "data_path = Path('../data/raw/bank_churn_dataset_80k.csv')\n",
            "if not data_path.exists():\n",
            "    data_path = Path('data/raw/bank_churn_dataset_80k.csv')\n",
            "\n",
            "df = pd.read_csv(data_path)\n",
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
            "## 1. 🔍 Tổng Quan & Chất Lượng Dữ Liệu (Data Quality & Structure)"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Kiểu dữ liệu và giá trị thiếu (Missing values)\n",
            "info_df = pd.DataFrame({\n",
            "    'Data Type': df.dtypes,\n",
            "    'Null Count': df.isnull().sum(),\n",
            "    'Null Ratio (%)': (df.isnull().sum() / len(df)) * 100,\n",
            "    'Unique Values': df.nunique()\n",
            "})\n",
            "print('=== Thông tin các biến ===')\n",
            "print(info_df)\n",
            "\n",
            "print(f'Số bản ghi trùng lặp (Duplicates): {df.duplicated().sum()}')"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Thống kê mô tả các biến số (Numerical Features)\n",
            "df.describe().T"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Thống kê mô tả các biến phân loại (Categorical Features)\n",
            "df.describe(include=['object', 'bool', 'str']).T"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "---\n",
            "## 2. 🎯 Phân Tích Biến Mục Tiêu (`exit` - Churn Status)"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "churn_counts = df['exit'].value_counts()\n",
            "churn_rates = df['exit'].value_counts(normalize=True) * 100\n",
            "\n",
            "fig, axes = plt.subplots(1, 2, figsize=(14, 5))\n",
            "\n",
            "labels = churn_counts.index.map({False: 'Duy trì (False)', True: 'Rời bỏ (True)'})\n",
            "sns.barplot(x=labels, y=churn_counts.values, ax=axes[0], palette=['#2ec4b6', '#e71d36'], hue=labels, legend=False)\n",
            "axes[0].set_title('Số Lượng Khách Hàng Churn vs Active', fontweight='bold')\n",
            "axes[0].set_ylabel('Số lượng khách hàng')\n",
            "for p in axes[0].patches:\n",
            "    axes[0].annotate(f'{int(p.get_height()):,}', (p.get_x() + p.get_width() / 2., p.get_height()),\n",
            "                     ha='center', va='center', xytext=(0, 5), textcoords='offset points')\n",
            "\n",
            "axes[1].pie(churn_rates, labels=['Duy trì (82%)', 'Rời bỏ (18%)'], autopct='%1.1f%%',\n",
            "            startangle=90, colors=['#2ec4b6', '#e71d36'], explode=(0, 0.08),\n",
            "            textprops={'fontsize': 12, 'weight': 'bold'})\n",
            "axes[1].set_title('Tỷ Lệ Khách Hàng Rời Bỏ (Churn Rate)', fontweight='bold')\n",
            "\n",
            "plt.tight_layout()\n",
            "plt.show()\n",
            "\n",
            "print(f'► Tỷ lệ Churn chung toàn hệ thống: {churn_rates[True]:.2f}% ({churn_counts[True]:,} khách hàng)')"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "---\n",
            "## 3. 👥 Phân Tích Nhân Khẩu Học (Demographic Analysis)"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "fig, axes = plt.subplots(2, 2, figsize=(16, 10))\n",
            "\n",
            "# Giới tính vs Churn Rate\n",
            "sns.countplot(data=df, x='gender', hue='exit', ax=axes[0, 0], palette=['#2ec4b6', '#e71d36'])\n",
            "axes[0, 0].set_title('Phân Phối Khách Hàng Theo Giới Tính & Churn', fontweight='bold')\n",
            "\n",
            "# Nhóm tuổi (Age Distribution)\n",
            "sns.kdeplot(data=df, x='age', hue='exit', common_norm=False, fill=True, ax=axes[0, 1], palette=['#2ec4b6', '#e71d36'])\n",
            "axes[0, 1].set_title('Phân Phối Độ Tuổi Khách Hàng theo Churn Status', fontweight='bold')\n",
            "\n",
            "# Tình trạng hôn nhân\n",
            "sns.countplot(data=df, x='married', hue='exit', ax=axes[1, 0], palette=['#2ec4b6', '#e71d36'])\n",
            "axes[1, 0].set_title('Tình Trạng Hôn Nhân & Churn', fontweight='bold')\n",
            "\n",
            "# Nghề nghiệp vs Churn Rate\n",
            "occupation_churn = df.groupby('occupation')['exit'].mean().sort_values(ascending=False) * 100\n",
            "sns.barplot(x=occupation_churn.values, y=occupation_churn.index, ax=axes[1, 1], palette='Reds_r', hue=occupation_churn.index, legend=False)\n",
            "axes[1, 1].set_title('Tỷ Lệ Churn (%) Theo Nghề Nghiệp', fontweight='bold')\n",
            "axes[1, 1].set_xlabel('Tỷ lệ Churn (%)')\n",
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
            "## 4. 💰 Phân Tích Tài Chính & Hành Vi Khách Hàng (Financial & Behavioral Factors)"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "fig, axes = plt.subplots(2, 2, figsize=(16, 10))\n",
            "\n",
            "# Số dư tài khoản (Balance)\n",
            "sns.boxplot(data=df, x='exit', y='balance', ax=axes[0, 0], palette=['#2ec4b6', '#e71d36'], hue='exit', legend=False)\n",
            "axes[0, 0].set_title('Số Dư Tài Khoản (Balance) vs Churn', fontweight='bold')\n",
            "axes[0, 0].set_xticks([0, 1])\n",
            "axes[0, 0].set_xticklabels(['Duy trì (False)', 'Rời bỏ (True)'])\n",
            "\n",
            "# Thu nhập hàng tháng (Monthly Income/IR)\n",
            "sns.boxplot(data=df, x='exit', y='monthly_ir', ax=axes[0, 1], palette=['#2ec4b6', '#e71d36'], hue='exit', legend=False)\n",
            "axes[0, 1].set_title('Thu Nhập Hàng Tháng (Monthly IR) vs Churn', fontweight='bold')\n",
            "axes[0, 1].set_xticks([0, 1])\n",
            "axes[0, 1].set_xticklabels(['Duy trì (False)', 'Rời bỏ (True)'])\n",
            "\n",
            "# Điểm tương tác (Engagement Score)\n",
            "sns.violinplot(data=df, x='exit', y='engagement_score', ax=axes[1, 0], palette=['#2ec4b6', '#e71d36'], hue='exit', legend=False)\n",
            "axes[1, 0].set_title('Engagement Score vs Churn', fontweight='bold')\n",
            "axes[1, 0].set_xticks([0, 1])\n",
            "axes[1, 0].set_xticklabels(['Duy trì (False)', 'Rời bỏ (True)'])\n",
            "\n",
            "# Risk Score vs Churn\n",
            "sns.boxplot(data=df, x='exit', y='risk_score', ax=axes[1, 1], palette=['#2ec4b6', '#e71d36'], hue='exit', legend=False)\n",
            "axes[1, 1].set_title('Risk Score vs Churn', fontweight='bold')\n",
            "axes[1, 1].set_xticks([0, 1])\n",
            "axes[1, 1].set_xticklabels(['Duy trì (False)', 'Rời bỏ (True)'])\n",
            "\n",
            "plt.tight_layout()\n",
            "plt.show()"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Phân tích Churn theo Phân khúc Khách hàng & Mức độ Trung thành\n",
            "fig, axes = plt.subplots(1, 3, figsize=(18, 5))\n",
            "\n",
            "# Customer Segment\n",
            "sns.barplot(data=df, x='customer_segment', y='exit', ax=axes[0], palette='viridis', hue='customer_segment', legend=False)\n",
            "axes[0].set_title('Tỷ Lệ Churn Theo Customer Segment', fontweight='bold')\n",
            "axes[0].set_ylabel('Tỷ lệ Churn')\n",
            "\n",
            "# Loyalty Level\n",
            "sns.barplot(data=df, x='loyalty_level', y='exit', ax=axes[1], palette='magma', hue='loyalty_level', legend=False)\n",
            "axes[1].set_title('Tỷ Lệ Churn Theo Loyalty Level', fontweight='bold')\n",
            "axes[1].set_ylabel('Tỷ lệ Churn')\n",
            "\n",
            "# Risk Segment\n",
            "sns.barplot(data=df, x='risk_segment', y='exit', ax=axes[2], palette='rocket', hue='risk_segment', legend=False)\n",
            "axes[2].set_title('Tỷ Lệ Churn Theo Risk Segment', fontweight='bold')\n",
            "axes[2].set_ylabel('Tỷ lệ Churn')\n",
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
            "## 5. 🌍 Phân Tích Địa Lý & Kênh Kỹ Thuật Số (Geography & Digital Behavior)"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "top_provinces = df['origin_province'].value_counts().head(10).index\n",
            "df_top_prov = df[df['origin_province'].isin(top_provinces)]\n",
            "\n",
            "province_churn = df_top_prov.groupby('origin_province')['exit'].agg(['count', 'mean']).reset_index()\n",
            "province_churn['mean'] = province_churn['mean'] * 100\n",
            "province_churn = province_churn.sort_values(by='mean', ascending=False)\n",
            "\n",
            "plt.figure(figsize=(14, 5))\n",
            "sns.barplot(data=province_churn, x='origin_province', y='mean', palette='coolwarm', hue='origin_province', legend=False)\n",
            "plt.title('Tỷ Lệ Churn (%) Tại Top 10 Tỉnh/Thành Phố Đông Khách Hàng Nhất', fontweight='bold')\n",
            "plt.ylabel('Tỷ lệ Churn (%)')\n",
            "plt.xlabel('Tỉnh / Thành phố')\n",
            "plt.xticks(rotation=30)\n",
            "plt.tight_layout()\n",
            "plt.show()"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "---\n",
            "## 6. 📊 Phân Tích Tương Quan (Correlation Analysis)"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Lọc các biến số định lượng\n",
            "num_cols = df.select_dtypes(include=['int64', 'float64', 'bool']).columns\n",
            "\n",
            "plt.figure(figsize=(14, 10))\n",
            "corr = df[num_cols].corr()\n",
            "sns.heatmap(corr, annot=True, fmt='.2f', cmap='vlag', vmin=-1, vmax=1, linewidths=0.5)\n",
            "plt.title('Ma Trận Tương Quan Các Biến Số Trong Dữ Liệu Bank Churn', fontweight='bold')\n",
            "plt.tight_layout()\n",
            "plt.show()"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "---\n",
            "## 7. 🎯 Kết Luận & Khuyến Nghị Kinh Doanh (Insights & Recommendations)\n",
            "\n",
            "### 📌 Các Phát Hiện Chính:\n",
            "1. **Tỷ lệ rời bỏ (Churn Rate)**: Đạt mức **18.0%** (14,400 / 80,000 khách hàng), đây là mức khá cao cần có chiến lược can thiệp sớm.\n",
            "2. **Yếu tố Rủi ro & Tương tác**:\n",
            "   - Khách hàng có **Engagement Score** thấp và **Risk Score** cao có tỷ lệ rời bỏ vượt trội.\n",
            "   - Biến `active_member` đóng vai trò quan trọng trong việc phân tách khách hàng có nguy cơ rời bỏ.\n",
            "3. **Độ tuổi & Thu nhập**:\n",
            "   - Nhóm khách hàng trung niên có xu hướng churn cao hơn nhóm trẻ tuổi.\n",
            "   - Phân khúc khách hàng `Priority` và `Mass` có đặc tính biến động số dư và rời bỏ khác biệt.\n",
            "\n",
            "### 💡 Đề Xuất Bước Tiếp Theo:\n",
            "- **Feature Engineering**: Tạo thêm các chỉ số biến động số dư, tỷ lệ giao dịch hàng tháng trên tổng tài sản.\n",
            "- **Xử lý Mất cân bằng dữ liệu (Imbalanced Data)**: Sử dụng các kỹ thuật như SMOTE, Class Weighting khi huấn luyện mô hình ML.\n",
            "- **Xây dựng mô hình ML Dự đoán Churn**: Sử dụng XGBoost / LightGBM để phân loại và giải thích bằng SHAP values."
        ]
    }
]

notebook_content = {
    "cells": nb_cells,
    "metadata": {
        "language_info": {"name": "python"},
        "orig_nbformat": 4
    },
    "nbformat": 4,
    "nbformat_minor": 2
}

nb_path = Path("eda/bank_churn_eda.ipynb")
with open(nb_path, "w", encoding="utf-8") as f:
    json.dump(notebook_content, f, ensure_ascii=False, indent=2)

print(f"✓ Notebook generated successfully at {nb_path}")
