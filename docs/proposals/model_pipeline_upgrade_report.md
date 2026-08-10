# Bao cao nang cap Home Credit ML pipeline

## 1. Muc tieu

Dot cap nhat nay tham khao notebook `notebook_re/home-credit-full-pipeline.ipynb` de cai thien
khau xu ly du lieu lich su, feature engineering va LightGBM cho bai toan Home Credit Default Risk.
Muc tieu chinh la tang kha nang xep hang rui ro (`ROC-AUC`, `Gini`, `KS`) ma khong dua
target vao feature engineering, khong sua du lieu goc va van giu calibration de su dung trong POC.

## 2. Baseline truoc cap nhat

Artifact cu duoc huan luyen tren 50.000 ho so voi `feature_set=full`:

| Metric | Gia tri |
|---|---:|
| ROC-AUC | 0.764619 |
| PR-AUC | 0.264726 |
| Gini | 0.529237 |
| KS | 0.404595 |
| Brier | 0.066668 |
| Log loss | 0.242615 |
| ECE (10 bins) | 0.004716 |

Pipeline cu chu yeu aggregate cac bang lich su bang `min`, `max`, `mean` va `sum` truc tiep
theo `SK_ID_CURR`. Cach nay giu duoc muc do rui ro tong quat nhung lam mat dien bien gan day va
cau truc tung hop dong `SK_ID_PREV`.

## 3. Thay doi ve data va feature engineering

### 3.1 Cac cua so thoi gian

- Cua so theo thang: `3`, `6`, `12`, `24` thang.
- Cua so theo ngay: `30`, `90`, `180`, `365` ngay.
- Chi su dung ban ghi tai hoac truoc moc tham chieu: relative day/month `<= 0`.
- Khong su dung `TARGET` trong bat ky phep aggregate nao.

### 3.2 Bureau va bureau balance

- Ghep `bureau_balance` vao `bureau` qua `SK_ID_BUREAU`.
- Tao late flag tu `STATUS=1..5`.
- Aggregate late rate va numerical status theo cac cua so gan day.
- Chi giu lich su cua applicant trong tap dang duoc build de giam bo nho.

### 3.3 Installment payments

- Mo rong DPD, payment ratio, late rate va payment amount theo `30/90/180/365` ngay.
- Tao `recent-vs-all late rate` de bieu dien hanh vi dang tot len hay xau di.
- Loai payment co `DAYS_ENTRY_PAYMENT > 0` de tranh su dung thong tin sau moc tham chieu.

### 3.4 POS CASH

- Chuyen tu aggregate truc tiep sang hai cap:
  `monthly record -> SK_ID_PREV -> SK_ID_CURR`.
- Tao DPD diff va remaining-installment diff tai `1/3/6` thang.
- Tinh vectorized OLS trend cua `SK_DPD` theo `MONTHS_BALANCE`.
- Them recent DPD, DPD_DEF, late rate va remaining installments.

### 3.5 Credit card

- Chuyen sang aggregate hai cap `SK_ID_PREV -> SK_ID_CURR`.
- Them utilization, balance diff va utilization trend.
- Them `payment-to-minimum`, `underpaid-minimum rate` va `drawings-to-payment`.
- Cap cac ratio cuc doan tai `100` de han che split khong on dinh.
- Them recent utilization, balance, DPD va underpayment theo cua so thang.

## 4. Thay doi model va training

- LightGBM tang gioi han len `5.000` estimators, `learning_rate=0.02`.
- Cau hinh regularization moi: `num_leaves=63`, `max_depth=8`,
  `min_child_samples=90`, `reg_alpha=0.5`, `reg_lambda=2.0`.
- Su dung early stopping `150` rounds, model full-data thuc te dung o iteration `753`.
- Nhanh `feature_set=full` bo Logistic Regression trong moi lan train champion vi median
  imputation tren ma tran lon gay chi phi bo nho va thoi gian khong can thiet.
- Ensemble script chuyen sang `feature_set=full`, sparse one-hot va early stopping cho
  LightGBM, XGBoost, CatBoost.
- Them script `scripts/train_kaggle_submission.py` de build relational features cho ca
  `application_train.csv` va `application_test.csv`.
- Duong Kaggle dung LightGBM native missing/categorical support va numeric `float32`, tranh
  cap phat them hon 1 GiB cho median imputation.

## 5. Ket qua thuc nghiem

### 5.1 So sanh cong bang tren 50.000 ho so

| Metric | Baseline | Pipeline moi | Chenh lech |
|---|---:|---:|---:|
| ROC-AUC | 0.764619 | 0.767920 | +0.003302 |
| PR-AUC | 0.264726 | 0.261229 | -0.003497 |
| Gini | 0.529237 | 0.535841 | +0.006604 |
| KS | 0.404595 | 0.405801 | +0.001206 |
| Brier | 0.066668 | 0.066657 | -0.000011 |
| Log loss | 0.242615 | 0.241994 | -0.000621 |

Ket qua cho thay ranking tong the tang, trong khi PR-AUC giam nhe. Vi vay can tiep tuc danh gia
class weighting va ensemble neu muc tieu uu tien la tim dung nhom default hiem.

### 5.2 Full-data LightGBM

Model cuoi duoc train tu toan bo `307.511` ho so, voi `690` features va holdout `46.127` ho so:

| Metric | Gia tri |
|---|---:|
| ROC-AUC | 0.793472 |
| PR-AUC | 0.297762 |
| Gini | 0.586944 |
| KS | 0.444115 |
| Brier | 0.065099 |
| Log loss | 0.234169 |
| ECE (10 bins) | 0.001786 |

Mean prediction tren holdout la `0.080825`, gan voi default rate `0.080734`, cho thay calibration
khong bi lech lon nhu cau hinh `class_weight=balanced` trong notebook tham khao.

## 6. Kaggle submission

Lenh tai lap:

```bash
uv run python scripts/train_kaggle_submission.py
```

Output:

- `artifacts/submission_full_lightgbm.csv`
- `artifacts/submission_full_lightgbm_report.json`

Kiem tra output:

- `48.744` dong, dung schema `SK_ID_CURR,TARGET`.
- Thu tu ID khop `sample_submission.csv`.
- Khong co duplicate ID hoac missing probability.
- Probability range: `[0.002195, 0.766888]`.
- Mean prediction: `0.074119`.
- SHA-256: `4A4A06F868DB6ACAC6BA6CD7F12BAF4DE7C723BECC7274CF970338D50642231E`.

## 7. Kiem thu

- `uv run pytest -q`: `19 passed`.
- `uv run ruff check src scripts/train_kaggle_submission.py tests`: passed.
- Them test cho dau cua linear trend va aggregate hai cap POS/credit card.
- `mypy` chua pass do project chua cai type stubs cho pandas, scikit-learn va scipy, dong thoi
  con mot so typing issue ton tai truoc dot cap nhat.

## 8. Gioi han va huong tiep theo

- Home Credit 2018 khong co application timestamp thuc, do do chua the khang dinh temporal stability.
- Holdout hien tai la stratified random split; Kaggle public/private leaderboard moi la phep kiem tra
  out-of-sample tiep theo.
- Can chay ablation theo nhom feature de xac dinh temporal feature nao tao uplift on dinh.
- Can thu OOF ensemble va class-imbalance strategies tren cung split neu uu tien PR-AUC.
- Submission phuc vu nghien cuu/Kaggle, khong phai score CIC hay policy tu dong duyet khoan vay.
