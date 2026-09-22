"""전처리 문서에서 공유 가능한 FAISS 산출물을 생성한다."""

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import faiss

from config.config import Settings
from ingest.embedding.documents import load_preprocessed_chunks
from ingest.embedding.model import DenseEncoder, normalized_vectors
from ingest.embedding.storage import FORMAT_VERSION, GROUPS, load_bundle, save_bundle


def build_index(settings: Settings, *, encoder=None) -> tuple[dict, bool]:
    # JSONL의 한 줄은 이미 BGE-M3 기준으로 확정된 한 청크이므로 재청킹하지 않는다.
    chunks, documents = load_preprocessed_chunks(settings.preprocessed_chunks, settings.document_manifest)
    encoder = encoder or DenseEncoder(settings.embedding_model, settings.embedding_revision, settings.embedding_device)
    if encoder.name != settings.embedding_model or not encoder.revision:
        raise ValueError("임베딩 모델명 또는 revision이 올바르지 않습니다.")
    if settings.embedding_dimension is not None and settings.embedding_dimension != encoder.dimension:
        raise ValueError("EMBEDDING_DIMENSION이 모델의 실제 출력 차원과 다릅니다.")
    indexable_chunks = [chunk for chunk in chunks if settings.include_review_chunks or not chunk["needs_review"]]
    if not indexable_chunks:
        raise ValueError("색인 가능한 청크가 없습니다. 검토 대상 포함 설정 또는 전처리 결과를 확인하세요.")
    specification = {
        "format_version": FORMAT_VERSION,
        "index_type": "IndexFlatIP",
        "embedding": {"model": encoder.name, "revision": encoder.revision, "dimension": encoder.dimension,
                      "normalize_embeddings": True, "query_prefix": ""},
        "source": {"chunks_file": settings.preprocessed_chunks.name,
                   "document_manifest": settings.document_manifest.name,
                   "pre_chunked": True},
        "documents": documents,
        "source_chunk_count": len(chunks),
        "indexed_chunk_count": len(indexable_chunks),
        "excluded_review_chunk_ids": [chunk["id"] for chunk in chunks if chunk not in indexable_chunks],
    }
    signature = hashlib.sha256(json.dumps([specification, chunks], sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    output = settings.faiss_index_dir
    if (output / "manifest.json").is_file():
        previous = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
        if previous.get("signature") == signature:
            manifest, _, _ = load_bundle(output)
            return manifest, True
    print(f"문서 {len(documents)}개, 청크 {len(indexable_chunks)}개 임베딩", flush=True)
    vectors = normalized_vectors(encoder.encode([chunk["embedding_text"] for chunk in indexable_chunks],
                                               batch_size=settings.embedding_batch_size), len(indexable_chunks), encoder.dimension)
    indexes, mapping = {}, {}
    for group in GROUPS:
        rows = [i for i, chunk in enumerate(indexable_chunks) if chunk["index_group"] == group]
        if not rows:
            continue
        index = faiss.IndexFlatIP(encoder.dimension)
        index.add(vectors[rows])
        indexes[group] = index
        mapping[group] = {"file": f"{group}.faiss", "chunk_ids": [indexable_chunks[i]["id"] for i in rows]}
    manifest = dict(specification, signature=signature, indexes=mapping,
                    created_at=datetime.now(timezone.utc).isoformat())
    save_bundle(output, manifest, chunks, indexes)
    load_bundle(output)
    return manifest, False


def main():
    parser = argparse.ArgumentParser(description="전처리 문서 청킹, BGE-M3 임베딩 및 FAISS 저장")
    parser.add_argument("--env-file", type=Path, help="환경 설정 파일")
    parser.add_argument("--chunks", type=Path, help="전처리 완료 chunks.jsonl")
    parser.add_argument("--document-manifest", type=Path, help="문서 출처 manifest.json")
    parser.add_argument("--output", type=Path, help="FAISS 저장 폴더")
    parser.add_argument("--include-review-chunks", action="store_true", default=None,
                        help="원본 확인 대상 청크도 검색 인덱스에 포함")
    args = parser.parse_args()
    overrides = {key: value for key, value in {
        "preprocessed_chunks": args.chunks, "document_manifest": args.document_manifest,
        "faiss_index_dir": args.output, "include_review_chunks": args.include_review_chunks,
    }.items() if value is not None}
    try:
        cfg = Settings(**overrides, **({"_env_file": args.env_file} if args.env_file else {}))
        manifest, reused = build_index(cfg)
    except (ValueError, OSError) as exc:
        parser.exit(1, f"인덱스 생성 실패: {exc}\n")
    print(f"{'재사용' if reused else '생성 완료'}: {cfg.faiss_index_dir} ({manifest['indexed_chunk_count']} indexed / {manifest['source_chunk_count']} source chunks, {manifest['embedding']['dimension']} dimensions)")


if __name__ == "__main__":
    main()
