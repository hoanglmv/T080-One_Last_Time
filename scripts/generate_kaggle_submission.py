"""Script to generate Kaggle submission file (artifacts/submission.csv)."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import time
import pandas as pd
from src.credit_scoring.model import CreditModelBundle


def main():
    print("⏳ Đang khởi tạo quá trình tạo file submission cho Kaggle Home Credit...")
    start_time = time.time()

    model_path = Path("artifacts/models/credit_model.joblib")
    test_path = Path("data/raw/home-credit-default-risk/application_test.csv")
    output_path = Path("artifacts/submission.csv")

    if not model_path.exists():
        raise FileNotFoundError(f"Không tìm thấy model artifact tại {model_path}")
    if not test_path.exists():
        raise FileNotFoundError(f"Không tìm thấy tập dữ liệu test tại {test_path}")

    print(f"📦 Loading model bundle từ {model_path}...")
    bundle = CreditModelBundle.load(model_path)
    print(f"   Mô hình: {bundle.model_name} ({bundle.model_version})")

    print(f"📄 Đang đọc dữ liệu application_test.csv từ {test_path}...")
    test_df = pd.read_csv(test_path)
    num_rows = len(test_df)
    print(f"   Tổng số hồ sơ test: {num_rows:,} bản ghi.")

    # Batch prediction in chunks for speed and memory safety
    chunk_size = 10000
    all_probs = []

    print("⚡ Đang thực hiện dự đoán xác suất P(Default) theo lô...")
    for i in range(0, num_rows, chunk_size):
        chunk_df = test_df.iloc[i : i + chunk_size]
        records = chunk_df.to_dict(orient="records")
        probs, _, _ = bundle.predict_proba(records)
        all_probs.extend(probs)
        processed = min(i + chunk_size, num_rows)
        print(f"   --> Đã xử lý {processed:,}/{num_rows:,} hồ sơ ({processed/num_rows*100:.1f}%)...")

    # Create Kaggle Submission DataFrame
    submission = pd.DataFrame({
        "SK_ID_CURR": test_df["SK_ID_CURR"],
        "TARGET": all_probs
    })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output_path, index=False)

    elapsed = time.time() - start_time
    print(f"\n✅ ĐÃ XUẤT TỆP SUBMISSION THÀNH CÔNG!")
    print(f"📌 Đường dẫn tệp: {output_path.resolve()}")
    print(f"📊 Kích thước: {output_path.stat().st_size / (1024*1024):.2f} MB")
    print(f"⏱️ Tổng thời gian xử lý: {elapsed:.2f} giây")
    print(f"👀 Xem 5 bản ghi đầu tiên:")
    print(submission.head())


if __name__ == "__main__":
    main()
