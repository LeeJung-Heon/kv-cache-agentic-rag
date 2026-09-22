"""전처리 문서에서 공유 가능한 FAISS 산출물을 생성한다."""

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import faiss

from config.config import Settings
from ingest.embedding.chunking import chunk_document
from ingest.embedding.documents import load_documents
from ingest.embedding.model import DenseEncoder, normalized_vectors
from ingest.embedding.storage import FORMAT_VERSION, GROUPS, load_bundle, save_bundle


def tokenizer_metadata(tokenizer) -> dict:
    vocabulary = json.dumps(tokenizer.get_vocab(), sort_keys=True, ensure_ascii=False)
    backend = getattr(tokenizer, "backend_tokenizer", None)
    configuration = backend.to_str() if backend is not None else vocabulary
    return {"model": tokenizer.name_or_path,
            "revision": getattr(tokenizer, "init_kwargs", {}).get("_commit_hash"),
            "sha256": hashlib.sha256(configuration.encode()).hexdigest()}


def build_index(settings: Settings, *, encoder=None, comparison_tokenizers=None) -> tuple[dict, bool]:
    documents = load_documents(settings.processed_manifest)  # 모델 다운로드보다 입력 검증을 먼저 수행
    encoder = encoder or DenseEncoder(settings.embedding_model, settings.embedding_revision, settings.embedding_device)
    if encoder.name != settings.embedding_model or not encoder.revision:
        raise ValueError("임베딩 모델명 또는 revision이 올바르지 않습니다.")
    if settings.embedding_dimension is not None and settings.embedding_dimension != encoder.dimension:
        raise ValueError("EMBEDDING_DIMENSION이 모델의 실제 출력 차원과 다릅니다.")
    if settings.chunk_size > encoder.max_tokens:
        raise ValueError("CHUNK_SIZE가 모델의 최대 입력 길이를 초과합니다.")
    if comparison_tokenizers is None:
        from transformers import AutoTokenizer
        comparison_tokenizers = [AutoTokenizer.from_pretrained(name, use_fast=True)
                                 for name in settings.chunk_tokenizer_models if name != encoder.name]
    tokenizers = [encoder.tokenizer, *comparison_tokenizers]
    chunks = []
    for document, pages, _ in documents:
        chunks.extend(chunk_document(document, pages, tokenizers, size=settings.chunk_size,
                                     overlap=settings.chunk_overlap, include_section=settings.chunk_include_section))
    specification = {
        "format_version": FORMAT_VERSION,
        "index_type": "IndexFlatIP",
        "embedding": {"model": encoder.name, "revision": encoder.revision, "dimension": encoder.dimension,
                      "normalize_embeddings": True, "query_prefix": ""},
        "chunking": {"size": settings.chunk_size, "overlap": settings.chunk_overlap,
                     "include_title": True, "include_section": settings.chunk_include_section,
                     "page_boundary": "preserve", "tokenizers": [tokenizer_metadata(t) for t in tokenizers]},
        "documents": [dict(doc.model_dump(), file_sha256=digest) for doc, _, digest in documents],
        "chunk_count": len(chunks),
    }
    signature = hashlib.sha256(json.dumps([specification, chunks], sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    output = settings.faiss_index_dir
    if (output / "manifest.json").is_file():
        previous = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
        if previous.get("signature") == signature:
            manifest, _, _ = load_bundle(output)
            return manifest, True
    print(f"문서 {len(documents)}개, 청크 {len(chunks)}개 임베딩", flush=True)
    vectors = normalized_vectors(encoder.encode([chunk["embedding_text"] for chunk in chunks],
                                               batch_size=settings.embedding_batch_size), len(chunks), encoder.dimension)
    indexes, mapping = {}, {}
    for group in GROUPS:
        rows = [i for i, chunk in enumerate(chunks) if chunk["technology"] == group]
        if not rows:
            continue
        index = faiss.IndexFlatIP(encoder.dimension)
        index.add(vectors[rows])
        indexes[group] = index
        mapping[group] = {"file": f"{group}.faiss", "chunk_ids": [chunks[i]["id"] for i in rows]}
    manifest = dict(specification, signature=signature, indexes=mapping,
                    created_at=datetime.now(timezone.utc).isoformat())
    save_bundle(output, manifest, chunks, indexes)
    load_bundle(output)
    return manifest, False


def main():
    parser = argparse.ArgumentParser(description="전처리 문서 청킹, BGE-M3 임베딩 및 FAISS 저장")
    parser.add_argument("--env-file", type=Path, help="환경 설정 파일")
    parser.add_argument("--manifest", type=Path, help="전처리 문서 목록 JSON")
    parser.add_argument("--output", type=Path, help="FAISS 저장 폴더")
    parser.add_argument("--chunk-size", type=int)
    parser.add_argument("--chunk-overlap", type=int)
    args = parser.parse_args()
    overrides = {key: value for key, value in {
        "processed_manifest": args.manifest, "faiss_index_dir": args.output,
        "chunk_size": args.chunk_size, "chunk_overlap": args.chunk_overlap,
    }.items() if value is not None}
    try:
        cfg = Settings(**overrides, **({"_env_file": args.env_file} if args.env_file else {}))
        manifest, reused = build_index(cfg)
    except (ValueError, OSError) as exc:
        parser.exit(1, f"인덱스 생성 실패: {exc}\n")
    print(f"{'재사용' if reused else '생성 완료'}: {cfg.faiss_index_dir} ({manifest['chunk_count']} chunks, {manifest['embedding']['dimension']} dimensions)")


if __name__ == "__main__":
    main()
