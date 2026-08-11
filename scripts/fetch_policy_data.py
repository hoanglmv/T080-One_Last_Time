"""
Script crawl, nạp dữ liệu và kiểm tra chỉ mục RAG Database cho chính sách pháp luật Việt Nam & Ngân hàng.
"""

from pathlib import Path
import sys

# Thêm root vào sys.path
BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.services.policy_rag import PolicyRAGService


def sync_and_build_policy_database():
    """Đọc các file văn bản quy phạm pháp luật, nạp vào thư mục docs/policies và khởi tạo RAG Index."""
    policy_dir = BASE_DIR / "docs" / "policies"
    vector_store_dir = BASE_DIR / "data" / "vector_store"

    policy_dir.mkdir(parents=True, exist_ok=True)
    vector_store_dir.mkdir(parents=True, exist_ok=True)

    print(f"📁 Thư mục lưu trữ văn bản pháp luật: {policy_dir}")
    print(f"💾 Thư mục lưu cơ sở dữ liệu RAG Vector Store: {vector_store_dir}")

    # Khởi tạo RAG Service & Đánh chỉ mục
    rag_service = PolicyRAGService(policy_dir=policy_dir)
    chunks = rag_service.chunks

    print(f"✅ Đã nạp thành công {len(chunks)} đoạn tri thức chính sách (Policy Chunks).")
    for chunk in chunks:
        print(f"  - [{chunk.source_document}] {chunk.title} ({len(chunk.keywords)} từ khóa)")

    # Thử nghiệm truy vấn mẫu
    sample_query = "trần lãi suất bộ luật dân sự hạn mức eKYC"
    retrieved = rag_service.retrieve_relevant_policies(query=sample_query, top_k=2)
    print("\n🔍 Kết quả truy vấn thử nghiệm RAG DB:")
    for res in retrieved:
        print(f"  -> Match: [{res.source_document}] {res.title}")

    print("\n🎉 Hoàn thành kiểm tra và đồng bộ dữ liệu RAG Policy Database.")


if __name__ == "__main__":
    sync_and_build_policy_database()
