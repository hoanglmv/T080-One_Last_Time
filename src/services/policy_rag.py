"""
Policy RAG Retrieval Service.
Reads, indexes, and retrieves relevant legal regulations and bank credit policy chunks for context injection and post-generation policy audit.
"""

from dataclasses import dataclass, field
from pathlib import Path
import re
from typing import Optional


@dataclass
class PolicyChunk:
    chunk_id: str
    source_document: str
    title: str
    content: str
    keywords: list[str] = field(default_factory=list)


class PolicyRAGService:
    """Service chịu trách nhiệm nạp, đánh chỉ mục và truy vấn tri thức chính sách RAG."""

    def __init__(self, policy_dir: Optional[Path] = None) -> None:
        if policy_dir is None:
            # Tìm thư mục docs/policies mặc định
            base_dir = Path(__file__).resolve().parents[2]
            policy_dir = base_dir / "docs" / "policies"
        self.policy_dir = policy_dir
        self.chunks: list[PolicyChunk] = []
        self._load_and_index_policies()

    def _load_and_index_policies(self) -> None:
        """Đọc toàn bộ file Markdown chính sách và chia nhỏ thành các chunks theo mục."""
        self.chunks.clear()
        if not self.policy_dir.exists():
            return

        for policy_file in sorted(self.policy_dir.glob("*.md")):
            text = policy_file.read_text(encoding="utf-8")
            doc_name = policy_file.name
            sections = re.split(r"\n(?=##?\s+)", text)

            for idx, sec in enumerate(sections):
                sec = sec.strip()
                if not sec:
                    continue
                first_line = sec.split("\n")[0]
                title = re.sub(r"^[#\s]+", "", first_line).strip()
                
                # Trích xuất keywords đơn giản
                words = re.findall(r"\w+", sec.lower())
                
                chunk = PolicyChunk(
                    chunk_id=f"{policy_file.stem}_chunk_{idx}",
                    source_document=doc_name,
                    title=title,
                    content=sec,
                    keywords=list(set(words)),
                )
                self.chunks.append(chunk)

    def retrieve_relevant_policies(self, query: str, top_k: int = 3) -> list[PolicyChunk]:
        """Truy vấn các đoạn văn bản chính sách liên quan nhất dựa trên từ khóa và độ khớp."""
        if not self.chunks:
            self._load_and_index_policies()

        query_terms = set(re.findall(r"\w+", query.lower()))
        if not query_terms:
            return self.chunks[:top_k]

        scored_chunks: list[tuple[float, PolicyChunk]] = []
        for chunk in self.chunks:
            # Score dựa trên TF trùng lặp từ khóa
            matches = sum(1 for term in query_terms if term in chunk.keywords)
            title_matches = sum(2 for term in query_terms if term in chunk.title.lower())
            total_score = matches + title_matches

            if total_score > 0:
                scored_chunks.append((float(total_score), chunk))

        # Sắp xếp giảm dần theo điểm số
        scored_chunks.sort(key=lambda x: x[0], reverse=True)
        results = [chunk for _, chunk in scored_chunks[:top_k]]
        
        # Fallback nếu không khớp từ khóa
        if not results:
            results = self.chunks[:top_k]
            
        return results

    def format_retrieved_context_for_llm(self, query: str, top_k: int = 3) -> str:
        """Định dạng ngữ cảnh chính sách RAG để tiêm vào System Prompt của LLM."""
        retrieved = self.retrieve_relevant_policies(query=query, top_k=top_k)
        if not retrieved:
            return "Không tìm thấy tài liệu chính sách liên quan."

        formatted_lines = [
            "### 📜 QUY ĐỊNH PHÁP LUẬT VÀ CHÍNH SÁCH NỔI BẬT (RAG POLICY CONTEXT):",
            "Dưới đây là các chính sách pháp luật Việt Nam và quy định ngân hàng bắt buộc phải tuân thủ:",
            "",
        ]
        for idx, chunk in enumerate(retrieved, start=1):
            formatted_lines.append(f"--- [Tài liệu {idx}: {chunk.source_document} | {chunk.title}] ---")
            formatted_lines.append(chunk.content)
            formatted_lines.append("")

        return "\n".join(formatted_lines)


# Singleton instance
policy_rag_service = PolicyRAGService()
