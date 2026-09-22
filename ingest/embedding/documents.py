"""전처리 담당자가 생성한 JSONL 청크를 FAISS 입력으로 정규화한다."""

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, HttpUrl


TECHNOLOGY_GROUPS = {"MLA": "sw", "CXL-PNM": "hw", "COMMON": "common"}


class SourceDocument(BaseModel):
    """전처리 manifest의 문서 단위 출처 정보."""

    model_config = ConfigDict(extra="forbid")

    doc_id: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    file_name: str = Field(min_length=1)
    technology: Literal["MLA", "CXL-PNM", "COMMON"]
    role: str = Field(min_length=1)
    title: str = Field(min_length=1)
    url: HttpUrl
    published_at: str | None = None
    allowed: bool
    page_count: int = Field(gt=0)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class SourceChunk(BaseModel):
    """database/chunks.jsonl의 한 줄, 즉 이미 확정된 한 청크."""

    model_config = ConfigDict(extra="forbid")

    chunk_id: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    doc_id: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    document_title: str = Field(min_length=1)
    technology: Literal["MLA", "CXL-PNM", "COMMON"]
    role: str = Field(min_length=1)
    section: str = Field(min_length=1)
    section_level: int | None = Field(default=None, gt=0)
    section_path: list[str]
    pdf_pages: list[int] = Field(min_length=1)
    text: str = Field(min_length=1)
    token_count: int = Field(gt=0)
    needs_review: bool
    review_reason: str | None = None


def load_preprocessed_chunks(chunks_path: Path, manifest_path: Path) -> tuple[list[dict], list[dict]]:
    """원본 청킹을 변경하지 않고 검색 메타데이터만 추가한다."""
    if not manifest_path.is_file():
        raise FileNotFoundError(f"문서 manifest가 없습니다: {manifest_path}")
    if not chunks_path.is_file():
        raise FileNotFoundError(f"전처리 청크 파일이 없습니다: {chunks_path}")

    try:
        raw_documents = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"문서 manifest JSON 형식이 올바르지 않습니다: {exc.msg}") from exc
    if not isinstance(raw_documents, list) or not raw_documents:
        raise ValueError("문서 manifest는 비어 있지 않은 JSON 배열이어야 합니다.")
    # Pydantic 검증으로 출처 정보가 누락되거나 예상하지 않은 필드가 섞이는 것을 막는다.
    documents = [SourceDocument.model_validate(record) for record in raw_documents]
    if len({document.doc_id for document in documents}) != len(documents):
        raise ValueError("문서 manifest의 doc_id가 중복됩니다.")
    if any(not document.allowed for document in documents):
        raise ValueError("허용 여부가 확인된 문서만 색인할 수 있습니다.")
    if sum(document.page_count for document in documents) > 200:
        raise ValueError("등록한 원본 PDF 전체 페이지 합계가 200쪽을 초과합니다.")
    documents_by_id = {document.doc_id: document for document in documents}

    chunks, seen_ids = [], set()
    with chunks_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                chunk = SourceChunk.model_validate_json(line)
            except ValueError as exc:
                raise ValueError(f"{chunks_path.name}:{line_number}: 청크 형식이 올바르지 않습니다: {exc}") from exc
            if chunk.chunk_id in seen_ids:
                raise ValueError(f"{chunks_path.name}:{line_number}: chunk_id가 중복됩니다: {chunk.chunk_id}")
            seen_ids.add(chunk.chunk_id)
            # 청크 메타데이터를 manifest와 대조해 서로 다른 전처리 버전의 혼용을 차단한다.
            document = documents_by_id.get(chunk.doc_id)
            if document is None:
                raise ValueError(f"{chunks_path.name}:{line_number}: manifest에 없는 doc_id입니다: {chunk.doc_id}")
            if chunk.technology != document.technology or chunk.role != document.role:
                raise ValueError(f"{chunks_path.name}:{line_number}: 문서 분류가 manifest와 다릅니다.")
            if chunk.document_title != document.title:
                raise ValueError(f"{chunks_path.name}:{line_number}: 문서 제목이 manifest와 다릅니다.")
            if chunk.pdf_pages != sorted(set(chunk.pdf_pages)) or chunk.pdf_pages[-1] > document.page_count:
                raise ValueError(f"{chunks_path.name}:{line_number}: 원본 PDF 페이지 정보가 올바르지 않습니다.")
            if chunk.needs_review and not chunk.review_reason:
                raise ValueError(f"{chunks_path.name}:{line_number}: needs_review 청크에는 review_reason이 필요합니다.")
            if not chunk.needs_review and chunk.review_reason is not None:
                raise ValueError(f"{chunks_path.name}:{line_number}: 검토 대상이 아닌 청크에는 review_reason이 없어야 합니다.")

            # text는 전처리 단계에서 제목과 소절, overlap까지 확정된 임베딩 입력이다.
            # 여기서는 필드명과 검색 그룹만 정규화하고 본문을 다시 자르지 않는다.
            chunks.append({
                "id": chunk.chunk_id,
                "doc_id": chunk.doc_id,
                "title": chunk.document_title,
                "technology": chunk.technology,
                "index_group": TECHNOLOGY_GROUPS[chunk.technology],
                "role": chunk.role,
                "section": chunk.section,
                "section_level": chunk.section_level,
                "section_path": chunk.section_path,
                # 공통 Evidence.page는 대표 첫 페이지를 쓰고 전체 범위는 pdf_pages에 보존한다.
                "page": chunk.pdf_pages[0],
                "pdf_pages": chunk.pdf_pages,
                "url": str(document.url),
                "published_at": document.published_at,
                "text": chunk.text,
                "embedding_text": chunk.text,
                "token_count": chunk.token_count,
                "needs_review": chunk.needs_review,
                "review_reason": chunk.review_reason,
            })
    if not chunks:
        raise ValueError("전처리 청크 파일이 비어 있습니다.")
    return chunks, [document.model_dump(mode="json") for document in documents]
