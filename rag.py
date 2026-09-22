from functools import cache
import hashlib
import json
import os
from pathlib import Path
from threading import Lock

import numpy as np
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parent
MODEL = "intfloat/multilingual-e5-large-instruct"
INSTRUCTION = "Given a question about KV cache optimization, retrieve relevant passages from technical papers that answer the question."
PAPERS = {
    "sw": ("sw_deepseek_v2.pdf", "DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model", "https://arxiv.org/abs/2405.04434"),
    "hw": ("hw_pim_cxl.pdf", "Scalable Processing-Near-Memory for 1M-Token LLM Inference: CXL-Enabled KV-Cache Management Beyond GPU Limits", "https://arxiv.org/abs/2511.00321"),
}


class PaperIndex:
    def __init__(self):
        self.model = SentenceTransformer(MODEL, device=os.getenv("EMBEDDING_DEVICE") or None)
        self.model.max_seq_length = 512
        self.encode_lock = Lock()
        self.chunks = []
        page_count = 0
        for side, (filename, title, url) in PAPERS.items():
            reader = PdfReader(ROOT / "docs" / filename)
            page_count += len(reader.pages)
            if page_count > 200:
                raise ValueError("논문은 합계 200페이지 이내여야 합니다.")
            count_before = len(self.chunks)
            for page_number, page in enumerate(reader.pages, 1):
                text = page.extract_text() or ""
                tokens = self.model.tokenizer(text, add_special_tokens=False, return_offsets_mapping=True, verbose=False)
                offsets = tokens["offset_mapping"]
                for start in range(0, len(offsets), 340):
                    end = min(start + 400, len(offsets))
                    content = text[offsets[start][0]:offsets[end - 1][1]].strip()
                    if content:
                        self.chunks.append(dict(id=f"{side}_p{page_number}_c{start // 340 + 1}", side=side,
                                                page=page_number, title=title, url=url, file=filename, text=content))
                    if end == len(offsets):
                        break
            if len(self.chunks) == count_before:
                raise ValueError(f"텍스트를 추출하지 못했습니다: {filename}")
        # ponytail: 두 논문은 NumPy 전수 검색; 문서가 크게 늘면 벡터 DB로 교체.
        fingerprint = hashlib.sha256(json.dumps([MODEL, self.chunks], ensure_ascii=False).encode()).hexdigest()
        cache = ROOT / ".cache" / f"{fingerprint}.npy"
        if cache.exists():
            self.vectors = np.load(cache, allow_pickle=False)
        else:
            self.vectors = self.model.encode([c["text"] for c in self.chunks], batch_size=8,
                                             normalize_embeddings=True, show_progress_bar=True)
            cache.parent.mkdir(exist_ok=True)
            temporary = cache.with_suffix(".tmp")
            with temporary.open("wb") as handle:
                np.save(handle, self.vectors, allow_pickle=False)
            temporary.replace(cache)
        if self.vectors.shape != (len(self.chunks), self.model.get_embedding_dimension()):
            raise ValueError("임베딩 캐시 크기가 다릅니다. .cache의 해당 파일을 삭제한 뒤 다시 실행하세요.")

    def search(self, query: str, side: str, k: int = 5) -> list[dict]:
        if side not in PAPERS or not query.strip() or not 1 <= k <= 8:
            raise ValueError("검색에는 비어 있지 않은 query, sw/hw, 1~8의 k가 필요합니다.")
        # MPS 커널 캐시는 동시 encode 시 충돌하므로 공유 모델의 추론만 직렬화한다.
        with self.encode_lock:
            vector = self.model.encode(f"Instruct: {INSTRUCTION}\nQuery: {query}", normalize_embeddings=True)
        indices = [i for i, c in enumerate(self.chunks) if c["side"] == side]
        scores = self.vectors[indices] @ vector
        return [dict(self.chunks[indices[i]], score=float(scores[i])) for i in np.argsort(-scores)[:k]]


@cache
def get_paper_index():
    return PaperIndex()
