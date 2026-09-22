"""에이전트 노드가 공유하는 FAISS 검색 서비스."""

import copy
from functools import cache
from threading import Lock

from config.config import Settings
from ingest.embedding.model import DenseEncoder, normalized_vectors
from ingest.embedding.storage import load_bundle

SEARCH_SIDES = ("sw", "hw")


class PaperIndex:
    """공유받은 FAISS 인덱스를 로딩하고 같은 모델로 질의를 검색한다."""

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or Settings()
        # 인덱스와 청크 JSON의 해시 및 행 연결까지 검증한 뒤 검색에 사용한다.
        self.manifest, self.chunks, self.indices = load_bundle(self.settings.faiss_index_dir)
        embedding = self.manifest["embedding"]
        if self.settings.embedding_model != embedding["model"]:
            raise ValueError("설정한 모델과 저장된 인덱스의 모델이 다릅니다.")
        if self.settings.embedding_dimension is not None and self.settings.embedding_dimension != embedding["dimension"]:
            raise ValueError("EMBEDDING_DIMENSION이 저장된 인덱스 차원과 다릅니다.")
        if self.settings.embedding_revision and self.settings.embedding_revision != embedding["revision"]:
            raise ValueError("EMBEDDING_REVISION이 인덱스의 revision과 다릅니다. manifest의 SHA를 사용하세요.")
        self._chunks_by_id = {chunk["id"]: chunk for chunk in self.chunks}
        self.model = None  # 최초 질의에서만 로딩하며 문서 재임베딩은 수행하지 않는다.
        self.encode_lock = Lock()

    def search(self, query: str, side: str, k: int | None = None) -> list[dict]:
        """질의를 임베딩하고 기술 문서 및 공통 문서에서 최종 top-k를 반환한다."""
        k = self.settings.retrieval_top_k if k is None else k
        if side not in SEARCH_SIDES or not isinstance(query, str) or not query.strip() or type(k) is not int or k < 1:
            raise ValueError("검색에는 비어 있지 않은 query, sw/hw, 양의 정수 k가 필요합니다.")
        groups = [group for group in (side, "common") if group in self.indices]
        if not groups:
            return []
        embedding = self.manifest["embedding"]
        # 공유 MPS 모델 로딩과 질의 임베딩 호출만 직렬화한다.
        with self.encode_lock:
            if self.model is None:
                encoder = DenseEncoder(embedding["model"], embedding["revision"], self.settings.embedding_device)
                if encoder.dimension != embedding["dimension"] or encoder.revision != embedding["revision"]:
                    raise ValueError("질의 모델의 차원 또는 revision이 문서 임베딩과 다릅니다.")
                self.model = encoder
            vector = normalized_vectors(self.model.encode([query.strip()], batch_size=1), 1, embedding["dimension"])

        # 각 인덱스의 행 번호를 manifest의 청크 ID로 변환하고 전체 후보를 다시 정렬한다.
        candidates = {}
        for group in groups:
            index = self.indices[group]
            scores, positions = index.search(vector, min(k, index.ntotal))
            mapping = self.manifest["indexes"][group]["chunk_ids"]
            for score, position in zip(scores[0], positions[0]):
                if position >= 0:
                    key = mapping[position]
                    candidates[key] = max(candidates.get(key, float("-inf")), float(score))
        ranked = sorted(candidates.items(), key=lambda item: (-item[1], item[0]))[:k]
        return [dict(copy.deepcopy(self._chunks_by_id[key]), score=score,
                     side=self._chunks_by_id[key]["technology"])
                for key, score in ranked]


_index_lock = Lock()


@cache
def _cached_paper_index() -> PaperIndex:
    return PaperIndex()


def get_paper_index() -> PaperIndex:
    """같은 프로세스의 모든 노드에 하나의 PaperIndex를 제공한다."""
    # functools.cache만 사용하면 최초 동시 호출에서 인스턴스가 중복 생성될 수 있다.
    with _index_lock:
        return _cached_paper_index()
