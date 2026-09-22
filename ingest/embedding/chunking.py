"""페이지와 Markdown 소절 경계를 보존하는 토큰 기준 청킹."""

import re

from ingest.embedding.documents import Document, Page


def embedding_text(title: str, section: str, text: str, include_section: bool) -> str:
    parts = [title.strip()]
    if include_section and section.strip():
        parts.append(section.strip())
    parts.append(text.strip())
    return "\n\n".join(parts)


def sections(text: str):
    heading, lines = "", []
    for line in text.splitlines():
        match = re.match(r"^#{1,6}\s+(.+?)\s*$", line)
        if match:
            if "\n".join(lines).strip():
                yield heading, "\n".join(lines).strip()
            heading, lines = match[1], []
        else:
            lines.append(line)
    if "\n".join(lines).strip():
        yield heading, "\n".join(lines).strip()


def split_text(text: str, title: str, section: str, tokenizers: list, *,
               size: int, overlap: int, include_section: bool):
    def fits(body):
        value = embedding_text(title, section, body, include_section)
        return all(len(tokenizer.encode(value, add_special_tokens=True)) <= size for tokenizer in tokenizers)

    if not fits(""):
        raise ValueError("제목과 소절명만으로 CHUNK_SIZE를 초과합니다. 설정을 조정하세요.")
    start = 0
    while start < len(text):
        rest = text[start:]
        if not rest.strip():
            break
        if fits(rest):
            yield rest.strip()
            break
        low, high = 0, len(rest)
        while low < high:
            middle = (low + high + 1) // 2
            if fits(rest[:middle]):
                low = middle
            else:
                high = middle - 1
        # 문단을 우선 유지. 긴 문단에서는 문장 또는 단어 경계를 사용한다.
        end = low
        for pattern in (r"\n\s*\n", r"[.!?。]\s+", r"\s+"):
            boundaries = [match.end() for match in re.finditer(pattern, rest[:low]) if match.end() >= low // 2]
            if boundaries:
                end = boundaries[-1]
                break
        piece = rest[:end].strip()
        # 토큰 수는 문자열 길이에 완전히 비례하지 않으므로 결과를 다시 검사한다.
        while piece and not fits(piece):
            end -= 1
            piece = rest[:end].strip()
        if not piece:
            raise ValueError("CHUNK_SIZE 안에 본문을 넣을 수 없습니다.")
        yield piece
        advance = end
        if overlap:
            offsets = tokenizers[0](rest[:end].rstrip(), add_special_tokens=False, return_offsets_mapping=True)["offset_mapping"]
            if len(offsets) <= overlap:
                raise ValueError("청크 본문보다 CHUNK_OVERLAP이 큽니다. overlap을 줄이세요.")
            advance = offsets[-overlap][0]
            if advance <= 0:
                raise ValueError("overlap 때문에 청킹이 진행되지 않습니다.")
        start += advance


def chunk_document(doc: Document, pages: list[Page], tokenizers: list, *,
                   size: int, overlap: int, include_section: bool) -> list[dict]:
    if not tokenizers or not 0 <= overlap < size:
        raise ValueError("토크나이저와 유효한 청킹 설정이 필요합니다.")
    chunks = []
    for page in pages:
        number = 0
        for section, text in sections(page.text):
            for piece in split_text(text, doc.title, section, tokenizers, size=size,
                                    overlap=overlap, include_section=include_section):
                number += 1
                key = f"{doc.technology}_{doc.doc_id}_p{page.page}_c{number}"
                chunks.append({
                    "id": key, "doc_id": doc.doc_id, "technology": doc.technology,
                    "title": doc.title, "url": doc.url, "file": doc.path,
                    "page": page.page, "pdf_pages": [page.page], "section": section,
                    "published_at": doc.published_at, "text": piece,
                    "embedding_text": embedding_text(doc.title, section, piece, include_section),
                })
    if not chunks:
        raise ValueError(f"{doc.doc_id}: 청킹할 본문이 없습니다.")
    return chunks
