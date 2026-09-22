"""문서와 질의가 공유하는 BGE-M3 dense 임베딩 설정."""

import re

import numpy as np


class DenseEncoder:
    def __init__(self, model_name: str, revision: str | None = None, device: str | None = None):
        if model_name != "BAAI/bge-m3":
            raise ValueError("현재 임베딩 계약은 BAAI/bge-m3의 prefix 없는 dense 방식입니다.")
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name, revision=revision, device=device or None)
        self.name = model_name
        self.revision = getattr(self.model[0].auto_model.config, "_commit_hash", None)
        if not self.revision and revision and re.fullmatch(r"[0-9a-f]{40}", revision):
            self.revision = revision
        if not self.revision:
            raise ValueError("모델 revision을 확인하지 못했습니다. EMBEDDING_REVISION에 commit SHA를 지정하세요.")
        self.dimension = self.model.get_embedding_dimension()
        self.tokenizer = self.model.tokenizer
        self.max_tokens = self.model.max_seq_length

    def encode(self, texts: list[str], batch_size: int = 8) -> np.ndarray:
        # 입력을 조용히 자르지 않는다. 특히 긴 질의는 호출자가 다시 구성하도록 알린다.
        if any(len(self.tokenizer.encode(text, add_special_tokens=True)) > self.max_tokens for text in texts):
            raise ValueError("임베딩 입력이 모델 최대 토큰 수를 초과합니다.")
        vectors = self.model.encode(texts, batch_size=batch_size, normalize_embeddings=True,
                                    convert_to_numpy=True, show_progress_bar=False)
        return normalized_vectors(vectors, len(texts), self.dimension)


def normalized_vectors(values, count: int, dimension: int) -> np.ndarray:
    vectors = np.array(values, dtype=np.float32, order="C", copy=True)
    if vectors.shape != (count, dimension) or not np.isfinite(vectors).all():
        raise ValueError("임베딩 출력 차원 또는 값이 유효하지 않습니다.")
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if (norms <= 0).any() or not np.isfinite(norms).all():
        raise ValueError("정규화할 수 없는 임베딩 벡터입니다.")
    vectors /= norms
    return vectors
