"""FAISS 파일과 JSON의 연결 및 무결성 검사. pickle을 사용하지 않는다."""

import hashlib
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory

import faiss

GROUPS = ("sw", "hw", "common")
FORMAT_VERSION = 1


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_bundle(output: Path, manifest: dict, chunks: list[dict], indexes: dict) -> None:
    """인덱스 묶음을 임시 폴더에서 완성한 뒤 원자적으로 교체한다."""
    output.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".build-", dir=output.parent) as temporary:
        stage = Path(temporary)
        (stage / "chunks.json").write_text(json.dumps(chunks, ensure_ascii=False, indent=2), encoding="utf-8")
        manifest["chunks_sha256"] = file_hash(stage / "chunks.json")
        for group, index in indexes.items():
            path = stage / f"{group}.faiss"
            faiss.write_index(index, str(path))
            manifest["indexes"][group]["sha256"] = file_hash(path)
        (stage / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        # 모든 파일을 완성한 뒤 교체하며 manifest를 마지막에 공개한다.
        # 중간 상태를 읽으면 아래 해시 검증에서 실패하므로 잘못된 근거가 반환되지 않는다.
        for name in ["chunks.json", *[f"{group}.faiss" for group in indexes], "manifest.json"]:
            os.replace(stage / name, output / name)


def load_bundle(directory: Path) -> tuple[dict, list[dict], dict]:
    """공유받은 인덱스 묶음의 해시와 행 매핑을 검증해 로드한다."""
    path = directory / "manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"FAISS 인덱스가 없습니다: {directory}. 먼저 uv run -m ingest.embedding.build_index를 실행하거나 공유 폴더를 복사하세요.")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("format_version") != FORMAT_VERSION:
        raise ValueError("지원하지 않는 FAISS 산출물 버전입니다.")
    if file_hash(directory / "chunks.json") != manifest["chunks_sha256"]:
        raise ValueError("chunks.json이 인덱스 생성 당시 파일과 다릅니다. 같은 버전의 폴더 전체를 사용하세요.")
    chunks = json.loads((directory / "chunks.json").read_text(encoding="utf-8"))
    ids = [chunk["id"] for chunk in chunks]
    if not chunks or len(set(ids)) != len(ids) or len(ids) != manifest["source_chunk_count"]:
        raise ValueError("청크 ID가 중복되거나 청크 수가 다릅니다.")
    if any(chunk.get("index_group") not in GROUPS for chunk in chunks):
        raise ValueError("지원하지 않는 기술 구분입니다.")
    excluded = manifest.get("excluded_review_chunk_ids", [])
    if len(excluded) != len(set(excluded)) or any(chunk_id not in ids for chunk_id in excluded):
        raise ValueError("검토 제외 청크 ID가 올바르지 않습니다.")
    if any(not next(chunk for chunk in chunks if chunk["id"] == chunk_id)["needs_review"] for chunk_id in excluded):
        raise ValueError("검토 제외 목록에는 needs_review 청크만 포함할 수 있습니다.")
    if manifest.get("indexed_chunk_count") != len(chunks) - len(excluded):
        raise ValueError("색인 청크 수가 검토 제외 목록과 다릅니다.")
    embedding = manifest["embedding"]
    if (embedding["model"] != "BAAI/bge-m3" or not embedding["revision"] or
            embedding["normalize_embeddings"] is not True or embedding["query_prefix"] != "" or
            manifest["index_type"] != "IndexFlatIP"):
        raise ValueError("지원하지 않는 임베딩 및 검색 설정입니다.")
    # manifest의 청크 순서는 FAISS 행 번호를 Evidence ID로 되돌리는 계약이다.
    indexed_chunks = [chunk for chunk in chunks if chunk["id"] not in set(excluded)]
    if set(manifest["indexes"]) != {chunk["index_group"] for chunk in indexed_chunks}:
        raise ValueError("청크 그룹과 인덱스 목록이 다릅니다.")
    indexes = {}
    for group, record in manifest["indexes"].items():
        filename = f"{group}.faiss"
        if record["file"] != filename or file_hash(directory / filename) != record["sha256"]:
            raise ValueError(f"{group}: FAISS 파일이 manifest와 다릅니다.")
        expected = [chunk["id"] for chunk in indexed_chunks if chunk["index_group"] == group]
        if record["chunk_ids"] != expected:
            raise ValueError(f"{group}: FAISS 행과 청크 ID 연결이 다릅니다.")
        index = faiss.read_index(str(directory / filename))
        # 파일 해시 외에도 검색 방식, 벡터 차원, 행 수를 확인한다.
        if (not isinstance(index, faiss.IndexFlatIP) or index.d != embedding["dimension"] or
                index.ntotal != len(expected) or index.metric_type != faiss.METRIC_INNER_PRODUCT):
            raise ValueError(f"{group}: FAISS 차원, 행 수 또는 검색 방식이 다릅니다.")
        indexes[group] = index
    return manifest, chunks, indexes
