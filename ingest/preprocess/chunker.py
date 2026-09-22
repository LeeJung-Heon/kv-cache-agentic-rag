import json
import re
from pathlib import Path

from transformers import AutoTokenizer


ROOT_DIR = Path(__file__).resolve().parents[2]
DATABASE_DIR = ROOT_DIR / "database"

SECTIONS_PATH = DATABASE_DIR / "sections.jsonl"
PAGES_PATH = DATABASE_DIR / "pages.jsonl"
MANIFEST_PATH = DATABASE_DIR / "manifest.json"
CHUNKS_PATH = DATABASE_DIR / "chunks.jsonl"


TOKENIZER_NAME = "BAAI/bge-m3"

MAX_TOKENS = 512
OVERLAP = 64  # 512의 12.5%


def load_jsonl(path: Path):
    items = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))

    return items


def load_manifest():
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        documents = data
    else:
        documents = data["documents"]

    return {
        document["doc_id"]: document
        for document in documents
    }


def count_tokens(tokenizer, text: str):
    """BGE-M3 tokenizer 기준 token 수. special token까지 포함한다."""
    return len(
        tokenizer.encode(
            text,
            add_special_tokens=True,
        )
    )


def make_prefix(document, section):
    """
    모든 chunk 앞에 문서 제목 + section 제목을 넣는다.
    prefix까지 포함한 최종 text가 MAX_TOKENS 이하여야 한다.
    """
    document_title = document["title"].strip()
    section_title = section["section"].strip()

    if section_title == "Front Matter":
        return f"{document_title}\n\n"

    return (
        f"{document_title}\n"
        f"{section_title}\n\n"
    )


def validate_chunk_settings():
    if not 0 <= OVERLAP < MAX_TOKENS:
        raise ValueError(
            "OVERLAP은 0 이상 MAX_TOKENS 미만이어야 합니다."
        )

    if OVERLAP:
        ratio = OVERLAP / MAX_TOKENS
        if not 0.10 <= ratio <= 0.20:
            raise ValueError(
                "OVERLAP은 MAX_TOKENS의 10~20% 범위로 설정하세요. "
                f"현재: {ratio:.1%}"
            )


def build_section_body(section):
    """
    section의 page_texts를 하나의 body로 연결하면서
    각 문자 구간이 어느 PDF page에서 왔는지 기록한다.
    """
    if "page_texts" not in section:
        raise ValueError(
            f"{section['doc_id']} / {section['section']} has no 'page_texts'. "
            "Run section_parser.py again first."
        )

    parts = []
    page_spans = []
    cursor = 0

    for page_data in section["page_texts"]:
        pdf_page = page_data["pdf_page"]
        page_text = page_data["text"].strip()

        if not page_text:
            continue

        if parts:
            separator = "\n\n"
            parts.append(separator)
            cursor += len(separator)

        start = cursor
        parts.append(page_text)
        cursor += len(page_text)
        end = cursor

        page_spans.append(
            {
                "start": start,
                "end": end,
                "pdf_page": pdf_page,
            }
        )

    return "".join(parts), page_spans


def pages_for_range(page_spans, start: int, end: int):
    """문자 범위 [start, end)와 실제로 겹치는 PDF page만 반환한다."""
    pages = []

    for span in page_spans:
        overlaps = (
            start < span["end"]
            and end > span["start"]
        )

        if overlaps:
            pages.append(span["pdf_page"])

    return sorted(set(pages))


def max_fitting_end(tokenizer, prefix: str, rest: str):
    """
    prefix + rest의 앞부분을 MAX_TOKENS까지 tokenizer로 잘라,
    본문에서 사용할 수 있는 최대 문자 끝 위치를 구한다.
    """
    encoded = tokenizer(
        prefix + rest,
        add_special_tokens=True,
        truncation=True,
        max_length=MAX_TOKENS,
        return_offsets_mapping=True,
    )

    prefix_len = len(prefix)
    candidate_ends = []

    for start_offset, end_offset in encoded["offset_mapping"]:
        if end_offset > prefix_len:
            candidate_ends.append(
                min(len(rest), end_offset - prefix_len)
            )

    if not candidate_ends:
        return 0

    end = max(candidate_ends)

    # tokenizer 경계 재계산에 따른 오차가 생겨도 최종 512 이하가 되도록 보정
    while end > 0 and count_tokens(
        tokenizer,
        prefix + rest[:end],
    ) > MAX_TOKENS:
        end -= 1

    return end


def find_natural_boundary(text: str, max_end: int):
    """
    max_end보다 앞에서 자연스러운 경계를 찾는다.
    문단 -> 문장 -> 공백 순으로 우선한다.
    """
    if max_end <= 0:
        return 0

    search_text = text[:max_end]
    minimum = max_end // 2

    patterns = (
        r"\n\s*\n",
        r"[.!?。]\s+",
        r"\s+",
    )

    for pattern in patterns:
        boundaries = [
            match.end()
            for match in re.finditer(pattern, search_text)
            if match.end() >= minimum
        ]

        if boundaries:
            return boundaries[-1]

    return max_end


def overlap_start_offset(tokenizer, piece: str):
    """
    piece의 마지막 OVERLAP tokens를 다음 chunk에 다시 포함하기 위해
    piece 내부에서 다음 chunk가 시작할 문자 offset을 반환한다.
    """
    if OVERLAP <= 0:
        return len(piece)

    encoded = tokenizer(
        piece,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )

    offsets = encoded["offset_mapping"]

    # 분할된 chunk가 overlap보다 짧으면 무한 반복 방지를 위해 overlap 미적용
    if len(offsets) <= OVERLAP:
        return len(piece)

    start = offsets[-OVERLAP][0]

    if start <= 0:
        return len(piece)

    return start


def split_section(tokenizer, section, document):
    """
    section/paragraph 경계를 우선 고려하여 최대 MAX_TOKENS로 청킹한다.

    - 문서 제목 + section 제목 + 본문 전체가 MAX_TOKENS 이하
    - 인접 chunk 사이 본문 64-token overlap
    - 실제 chunk가 참조하는 PDF page provenance 유지
    """
    prefix = make_prefix(document, section)

    if count_tokens(tokenizer, prefix) >= MAX_TOKENS:
        raise ValueError(
            f"{section['doc_id']} / {section['section']}: "
            "prefix itself exceeds token limit."
        )

    body, page_spans = build_section_body(section)

    if not body.strip():
        return []

    chunks = []
    start = 0

    while start < len(body):
        # chunk 시작의 불필요한 공백 제거
        while start < len(body) and body[start].isspace():
            start += 1

        if start >= len(body):
            break

        rest = body[start:]

        # 남은 section 전체가 들어가면 마지막 chunk
        if count_tokens(tokenizer, prefix + rest) <= MAX_TOKENS:
            piece = rest.rstrip()
            chunk_end = start + len(piece)

            chunks.append(
                {
                    "text": prefix + piece,
                    "pdf_pages": pages_for_range(
                        page_spans,
                        start,
                        chunk_end,
                    ),
                }
            )
            break

        max_end = max_fitting_end(
            tokenizer,
            prefix,
            rest,
        )

        if max_end <= 0:
            raise ValueError(
                f"{section['doc_id']} / {section['section']}: "
                "본문을 MAX_TOKENS 안에 넣을 수 없습니다."
            )

        boundary = find_natural_boundary(
            rest,
            max_end,
        )

        piece = rest[:boundary].rstrip()

        if not piece:
            piece = rest[:max_end].rstrip()

        if not piece:
            raise ValueError(
                f"{section['doc_id']} / {section['section']}: "
                "empty chunk detected."
            )

        chunk_text = prefix + piece
        token_count = count_tokens(tokenizer, chunk_text)

        if token_count > MAX_TOKENS:
            raise ValueError(
                f"{section['doc_id']} / {section['section']}: "
                f"{token_count} > {MAX_TOKENS}"
            )

        chunk_end = start + len(piece)

        chunks.append(
            {
                "text": chunk_text,
                "pdf_pages": pages_for_range(
                    page_spans,
                    start,
                    chunk_end,
                ),
            }
        )

        # 현재 chunk의 마지막 64 body tokens를 다음 chunk에 재사용
        overlap_offset = overlap_start_offset(
            tokenizer,
            piece,
        )

        next_start = start + overlap_offset

        # 안전장치: 반드시 앞으로 진행
        if next_start <= start:
            next_start = chunk_end

        start = next_start

    return chunks


def build_review_lookup(pages):
    """(doc_id, pdf_page) -> review_reason"""
    lookup = {}

    for page in pages:
        if not page.get("needs_review", False):
            continue

        key = (
            page["doc_id"],
            page["pdf_page"],
        )

        lookup[key] = page.get("review_reason")

    return lookup


def get_review_info(doc_id, pdf_pages, review_lookup):
    """
    실제 chunk가 포함하는 pdf_pages만 기준으로 review 여부를 판단한다.
    section 전체 page 범위를 사용하지 않는다.
    """
    reasons = []

    for pdf_page in pdf_pages:
        key = (doc_id, pdf_page)

        if key not in review_lookup:
            continue

        reason = review_lookup[key]

        if reason:
            reasons.append(
                f"p.{pdf_page}: {reason}"
            )

    reasons = list(dict.fromkeys(reasons))

    if reasons:
        return True, "; ".join(reasons)

    return False, None


def build_chunks():
    validate_chunk_settings()

    tokenizer = AutoTokenizer.from_pretrained(
        TOKENIZER_NAME,
        use_fast=True,
    )

    if OVERLAP > 0 and not getattr(tokenizer, "is_fast", False):
        raise ValueError(
            "Overlap 계산을 위해 fast tokenizer가 필요합니다."
        )

    sections = load_jsonl(SECTIONS_PATH)
    pages = load_jsonl(PAGES_PATH)
    manifest = load_manifest()

    review_lookup = build_review_lookup(pages)

    chunks = []
    doc_counters = {}

    for section in sections:
        doc_id = section["doc_id"]
        document = manifest[doc_id]

        if not document.get("allowed", True):
            continue

        chunk_items = split_section(
            tokenizer=tokenizer,
            section=section,
            document=document,
        )

        if doc_id not in doc_counters:
            doc_counters[doc_id] = 0

        for chunk_data in chunk_items:
            text = chunk_data["text"]
            pdf_pages = chunk_data["pdf_pages"]

            needs_review, review_reason = get_review_info(
                doc_id=doc_id,
                pdf_pages=pdf_pages,
                review_lookup=review_lookup,
            )

            doc_counters[doc_id] += 1

            chunk_id = (
                f"{doc_id}_chunk_"
                f"{doc_counters[doc_id]:04d}"
            )

            token_count = count_tokens(
                tokenizer,
                text,
            )

            if token_count > MAX_TOKENS:
                raise ValueError(
                    f"{chunk_id} exceeds {MAX_TOKENS} tokens: "
                    f"{token_count}"
                )

            chunk = {
                "chunk_id": chunk_id,
                "doc_id": doc_id,
                "document_title": document["title"],
                "technology": document["technology"],
                "role": document["role"],
                "section": section["section"],
                "section_level": section.get("section_level"),
                "section_path": section.get("section_path", []),
                "pdf_pages": pdf_pages,
                "text": text,
                "token_count": token_count,
                "needs_review": needs_review,
                "review_reason": review_reason,
            }

            chunks.append(chunk)

    return chunks


def save_chunks(chunks):
    with open(CHUNKS_PATH, "w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(
                json.dumps(
                    chunk,
                    ensure_ascii=False,
                )
                + "\n"
            )


def validate_chunks(chunks):
    """생성 결과에 대한 마지막 validation."""
    if not chunks:
        raise ValueError("No chunks were generated.")

    chunk_ids = [
        chunk["chunk_id"]
        for chunk in chunks
    ]

    if len(chunk_ids) != len(set(chunk_ids)):
        raise ValueError("Duplicate chunk_id detected.")

    oversized = [
        chunk
        for chunk in chunks
        if chunk["token_count"] > MAX_TOKENS
    ]

    if oversized:
        raise ValueError(
            "Chunk exceeding token limit detected."
        )

    missing_pages = [
        chunk
        for chunk in chunks
        if not chunk["pdf_pages"]
    ]

    if missing_pages:
        raise ValueError(
            "Chunk without pdf_pages detected."
        )


if __name__ == "__main__":
    chunks = build_chunks()

    validate_chunks(chunks)
    save_chunks(chunks)

    review_chunks = [
        chunk
        for chunk in chunks
        if chunk["needs_review"]
    ]

    max_token_count = max(
        chunk["token_count"]
        for chunk in chunks
    )

    min_token_count = min(
        chunk["token_count"]
        for chunk in chunks
    )

    print()
    print(f"Chunks saved: {CHUNKS_PATH}")
    print(f"Total chunks: {len(chunks)}")
    print(f"Max token count: {max_token_count}")
    print(f"Min token count: {min_token_count}")
    print(f"Review chunks: {len(review_chunks)}")
    print(
        f"Overlap: {OVERLAP} tokens "
        f"({OVERLAP / MAX_TOKENS:.1%})"
    )
    print()

    for chunk in review_chunks:
        print(
            f"- {chunk['chunk_id']} | "
            f"{chunk['doc_id']} | "
            f"{chunk['pdf_pages']} | "
            f"{chunk['review_reason']}"
        )
