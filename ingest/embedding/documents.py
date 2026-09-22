"""전처리 결과의 형식 변환 경계. PDF 추출은 전처리 담당 단계에서 수행한다."""

import hashlib
import json
import re
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class Page(BaseModel):
    model_config = ConfigDict(extra="forbid")
    page: int = Field(gt=0)
    text: str


class Document(BaseModel):
    model_config = ConfigDict(extra="forbid")
    doc_id: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    path: str = Field(min_length=1)
    title: str = Field(min_length=1)
    technology: Literal["sw", "hw", "common"]
    url: str = ""
    published_at: str | None = None
    version: str | None = None
    original_pages: int = Field(gt=0)
    allowed: bool


def read_pages(path: Path) -> list[Page]:
    """임시 입력 계약: pages JSON 또는 PAGE 마커가 있는 UTF-8 TXT/Markdown."""
    text = path.read_text(encoding="utf-8-sig")
    if path.suffix.lower() == ".json":
        raw = json.loads(text)
        if not isinstance(raw, dict) or not isinstance(raw.get("pages"), list):
            raise ValueError(f"{path.name}: JSON에는 pages 목록이 필요합니다.")
        pages = [Page.model_validate(page) for page in raw["pages"]]
    elif path.suffix.lower() in {".txt", ".md"}:
        markers = list(re.finditer(r"(?m)^\s*<!--\s*PAGE:\s*([1-9]\d*)\s*-->\s*$", text))
        if not markers or text[:markers[0].start()].strip():
            raise ValueError(f"{path.name}: 본문 앞에 <!-- PAGE: 1 --> 형식의 원본 페이지 마커가 필요합니다.")
        pages = [Page(page=int(marker[1]), text=text[marker.end():markers[i + 1].start() if i + 1 < len(markers) else len(text)])
                 for i, marker in enumerate(markers)]
    else:
        raise ValueError(f"{path.name}: 전처리된 .json, .txt 또는 .md 파일을 입력하세요.")
    numbers = [page.page for page in pages]
    if not numbers or numbers != sorted(set(numbers)):
        raise ValueError(f"{path.name}: 원본 페이지 번호는 중복 없이 오름차순이어야 합니다.")
    if not any(page.text.strip() for page in pages):
        raise ValueError(f"{path.name}: 임베딩할 본문이 없습니다.")
    return pages


def load_documents(manifest_path: Path) -> list[tuple[Document, list[Page], str]]:
    if not manifest_path.is_file():
        raise FileNotFoundError(f"전처리 문서 목록이 없습니다: {manifest_path}. docs/EMBEDDING_PIPELINE.md를 확인하세요.")
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(raw, list) or not raw:
        raise ValueError("문서 목록은 비어 있지 않은 JSON 배열이어야 합니다.")
    documents = [Document.model_validate(item) for item in raw]
    if len({doc.doc_id for doc in documents}) != len(documents):
        raise ValueError("doc_id가 중복됩니다.")
    if sum(doc.original_pages for doc in documents) > 200:
        raise ValueError("등록한 원본 PDF 전체 페이지 합계가 200쪽을 초과합니다.")
    loaded = []
    paths = set()
    for doc in documents:
        if not doc.allowed:
            raise ValueError(f"{doc.doc_id}: 허용 여부가 확인된 문서만 색인할 수 있습니다.")
        path = (manifest_path.parent / doc.path).resolve()
        if path in paths:
            raise ValueError("동일한 전처리 파일이 여러 번 등록되었습니다. 공통 문서는 common으로 등록하세요.")
        paths.add(path)
        pages = read_pages(path)
        if pages[-1].page > doc.original_pages:
            raise ValueError(f"{doc.doc_id}: 본문의 페이지 번호가 원본 전체 쪽수를 초과합니다.")
        loaded.append((doc, pages, hashlib.sha256(path.read_bytes()).hexdigest()))
    return loaded
