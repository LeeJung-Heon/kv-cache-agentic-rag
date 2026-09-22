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

MAX_TOKENS = 400
OVERLAP = 0


def load_jsonl(path: Path):
    items = []

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as f:
        for line in f:
            if line.strip():
                items.append(
                    json.loads(line)
                )

    return items


def load_manifest():
    with open(
        MANIFEST_PATH,
        "r",
        encoding="utf-8",
    ) as f:
        data = json.load(f)

    if isinstance(data, list):
        documents = data
    else:
        documents = data["documents"]

    return {
        document["doc_id"]: document
        for document in documents
    }


def count_tokens(
    tokenizer,
    text: str,
):
    """
    BGE-M3 tokenizer 기준 token 수.
    special token까지 포함한다.
    """

    return len(
        tokenizer.encode(
            text,
            add_special_tokens=True,
        )
    )


def make_prefix(
    document,
    section,
):
    """
    모든 chunk 앞에
    문서 제목 + section 제목을 넣는다.

    이 prefix까지 포함해서
    최대 400 token으로 제한한다.
    """

    document_title = document["title"]
    section_title = section["section"]

    if section_title == "Front Matter":
        return (
            f"{document_title}\n\n"
        )

    return (
        f"{document_title}\n"
        f"{section_title}\n\n"
    )


def hard_split_text(
    tokenizer,
    text: str,
    prefix: str,
):
    """
    하나의 paragraph 자체가 너무 길 경우
    tokenizer 기준으로 강제 분할한다.

    overlap = 0
    """

    prefix_ids = tokenizer.encode(
        prefix,
        add_special_tokens=False,
    )

    special_tokens = (
        tokenizer.num_special_tokens_to_add(
            pair=False
        )
    )

    available_tokens = (
        MAX_TOKENS
        - len(prefix_ids)
        - special_tokens
    )

    if available_tokens <= 0:
        raise ValueError(
            "Prefix itself exceeds "
            f"{MAX_TOKENS} token limit."
        )

    text_ids = tokenizer.encode(
        text,
        add_special_tokens=False,
    )

    pieces = []

    for start in range(
        0,
        len(text_ids),
        available_tokens,
    ):
        end = (
            start
            + available_tokens
        )

        piece_ids = text_ids[
            start:end
        ]

        piece = tokenizer.decode(
            piece_ids,
            skip_special_tokens=True,
        ).strip()

        if piece:
            pieces.append(piece)

    return pieces


def split_oversized_unit(
    tokenizer,
    unit: str,
    prefix: str,
):
    """
    긴 paragraph/table은 먼저 line 단위로
    나누고, 그래도 긴 line은 tokenizer
    기준으로 분할한다.
    """

    full_text = (
        prefix + unit
    )

    if (
        count_tokens(
            tokenizer,
            full_text,
        )
        <= MAX_TOKENS
    ):
        return [unit]

    lines = [
        line.strip()
        for line in unit.splitlines()
        if line.strip()
    ]

    # 일반 paragraph처럼 한 줄이면
    # tokenizer 기준으로 직접 분할
    if len(lines) <= 1:
        return hard_split_text(
            tokenizer,
            unit,
            prefix,
        )

    pieces = []
    current_lines = []

    for line in lines:

        # line 자체가 너무 긴 경우
        if (
            count_tokens(
                tokenizer,
                prefix + line,
            )
            > MAX_TOKENS
        ):
            if current_lines:
                pieces.append(
                    "\n".join(
                        current_lines
                    )
                )

                current_lines = []

            pieces.extend(
                hard_split_text(
                    tokenizer,
                    line,
                    prefix,
                )
            )

            continue

        candidate_lines = (
            current_lines
            + [line]
        )

        candidate = "\n".join(
            candidate_lines
        )

        if (
            count_tokens(
                tokenizer,
                prefix + candidate,
            )
            <= MAX_TOKENS
        ):
            current_lines.append(
                line
            )

        else:
            if current_lines:
                pieces.append(
                    "\n".join(
                        current_lines
                    )
                )

            current_lines = [line]

    if current_lines:
        pieces.append(
            "\n".join(
                current_lines
            )
        )

    return pieces


def split_section(
    tokenizer,
    section,
    document,
):
    """
    PDF page provenance를 유지하면서
    paragraph 단위로 최대 400 token
    chunk를 생성한다.

    반환 예:
    {
        "text": "...",
        "pdf_pages": [5, 6]
    }
    """

    prefix = make_prefix(
        document,
        section,
    )

    # 새 section_parser가 실행됐는지 확인
    if "page_texts" not in section:
        raise ValueError(
            f"{section['doc_id']} / "
            f"{section['section']} has no "
            "'page_texts'. "
            "Run section_parser.py again first."
        )

    units = []

    for page_data in section[
        "page_texts"
    ]:

        pdf_page = page_data[
            "pdf_page"
        ]

        page_text = page_data[
            "text"
        ]

        paragraphs = [
            paragraph.strip()
            for paragraph in re.split(
                r"\n\s*\n",
                page_text,
            )
            if paragraph.strip()
        ]

        for paragraph in paragraphs:

            pieces = (
                split_oversized_unit(
                    tokenizer,
                    paragraph,
                    prefix,
                )
            )

            for piece in pieces:
                units.append(
                    {
                        "text": piece,
                        "pdf_pages": {
                            pdf_page
                        },
                    }
                )

    chunks = []

    current_units = []
    current_pages = set()

    for unit in units:

        candidate_units = (
            current_units
            + [unit]
        )

        candidate_body = (
            "\n\n".join(
                item["text"]
                for item
                in candidate_units
            )
        )

        candidate_text = (
            prefix
            + candidate_body
        )

        if (
            count_tokens(
                tokenizer,
                candidate_text,
            )
            <= MAX_TOKENS
        ):
            current_units.append(
                unit
            )

            current_pages.update(
                unit["pdf_pages"]
            )

        else:
            if current_units:
                chunks.append(
                    {
                        "text": (
                            prefix
                            + "\n\n".join(
                                item["text"]
                                for item
                                in current_units
                            )
                        ),
                        "pdf_pages": (
                            sorted(
                                current_pages
                            )
                        ),
                    }
                )

            current_units = [
                unit
            ]

            current_pages = set(
                unit["pdf_pages"]
            )

    if current_units:
        chunks.append(
            {
                "text": (
                    prefix
                    + "\n\n".join(
                        item["text"]
                        for item
                        in current_units
                    )
                ),
                "pdf_pages": sorted(
                    current_pages
                ),
            }
        )

    return chunks


def build_review_lookup(
    pages,
):
    """
    (doc_id, pdf_page)
    -> review_reason
    """

    lookup = {}

    for page in pages:

        if not page.get(
            "needs_review",
            False,
        ):
            continue

        key = (
            page["doc_id"],
            page["pdf_page"],
        )

        lookup[key] = page.get(
            "review_reason"
        )

    return lookup


def get_review_info(
    doc_id,
    pdf_pages,
    review_lookup,
):
    """
    실제 chunk가 포함하는 pdf_pages만
    기준으로 review 여부를 판단한다.

    section 전체 page 범위를 사용하지 않는다.
    """

    reasons = []

    for pdf_page in pdf_pages:

        key = (
            doc_id,
            pdf_page,
        )

        if key not in review_lookup:
            continue

        reason = review_lookup[
            key
        ]

        if reason:
            reasons.append(
                f"p.{pdf_page}: "
                f"{reason}"
            )

    # 같은 reason 중복 제거
    reasons = list(
        dict.fromkeys(reasons)
    )

    if reasons:
        return (
            True,
            "; ".join(reasons),
        )

    return False, None


def build_chunks():

    tokenizer = (
        AutoTokenizer
        .from_pretrained(
            TOKENIZER_NAME
        )
    )

    sections = load_jsonl(
        SECTIONS_PATH
    )

    pages = load_jsonl(
        PAGES_PATH
    )

    manifest = load_manifest()

    review_lookup = (
        build_review_lookup(
            pages
        )
    )

    chunks = []

    # doc별 chunk 번호
    doc_counters = {}

    for section in sections:

        doc_id = section[
            "doc_id"
        ]

        document = manifest[
            doc_id
        ]

        if not document.get(
            "allowed",
            True,
        ):
            continue

        chunk_items = split_section(
            tokenizer=tokenizer,
            section=section,
            document=document,
        )

        if doc_id not in doc_counters:
            doc_counters[
                doc_id
            ] = 0

        for chunk_data in chunk_items:

            text = chunk_data[
                "text"
            ]

            pdf_pages = chunk_data[
                "pdf_pages"
            ]

            # review 여부도 chunk의
            # 실제 page 범위 기준으로 판정
            (
                needs_review,
                review_reason,
            ) = get_review_info(
                doc_id=doc_id,
                pdf_pages=pdf_pages,
                review_lookup=(
                    review_lookup
                ),
            )

            doc_counters[
                doc_id
            ] += 1

            chunk_id = (
                f"{doc_id}_chunk_"
                f"{doc_counters[doc_id]:04d}"
            )

            token_count = (
                count_tokens(
                    tokenizer,
                    text,
                )
            )

            # 최종 안전 검증
            if (
                token_count
                > MAX_TOKENS
            ):
                raise ValueError(
                    f"{chunk_id} exceeds "
                    f"{MAX_TOKENS} tokens: "
                    f"{token_count}"
                )

            chunk = {
                "chunk_id": (
                    chunk_id
                ),

                "doc_id": doc_id,

                "document_title": (
                    document["title"]
                ),

                "technology": (
                    document[
                        "technology"
                    ]
                ),

                "role": (
                    document["role"]
                ),

                "section": (
                    section[
                        "section"
                    ]
                ),

                "section_level": (
                    section.get(
                        "section_level"
                    )
                ),

                "section_path": (
                    section.get(
                        "section_path",
                        [],
                    )
                ),

                # 실제 chunk가 사용한
                # PDF page만 저장
                "pdf_pages": (
                    pdf_pages
                ),

                # embedding 대상
                "text": text,

                "token_count": (
                    token_count
                ),

                # QA / 원본 확인용
                "needs_review": (
                    needs_review
                ),

                "review_reason": (
                    review_reason
                ),
            }

            chunks.append(
                chunk
            )

    return chunks


def save_chunks(
    chunks,
):

    with open(
        CHUNKS_PATH,
        "w",
        encoding="utf-8",
    ) as f:

        for chunk in chunks:

            f.write(
                json.dumps(
                    chunk,
                    ensure_ascii=False,
                )
                + "\n"
            )


def validate_chunks(
    chunks,
):
    """
    생성 결과에 대한 마지막 validation.
    """

    if not chunks:
        raise ValueError(
            "No chunks were generated."
        )

    chunk_ids = [
        chunk["chunk_id"]
        for chunk in chunks
    ]

    if (
        len(chunk_ids)
        != len(set(chunk_ids))
    ):
        raise ValueError(
            "Duplicate chunk_id detected."
        )

    oversized = [
        chunk
        for chunk in chunks
        if chunk["token_count"]
        > MAX_TOKENS
    ]

    if oversized:
        raise ValueError(
            "Chunk exceeding token "
            "limit detected."
        )

    missing_pages = [
        chunk
        for chunk in chunks
        if not chunk["pdf_pages"]
    ]

    if missing_pages:
        raise ValueError(
            "Chunk without pdf_pages "
            "detected."
        )


if __name__ == "__main__":

    chunks = build_chunks()

    validate_chunks(
        chunks
    )

    save_chunks(
        chunks
    )

    review_chunks = [
        chunk
        for chunk in chunks
        if chunk[
            "needs_review"
        ]
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
    print(
        f"Chunks saved: "
        f"{CHUNKS_PATH}"
    )

    print(
        f"Total chunks: "
        f"{len(chunks)}"
    )

    print(
        f"Max token count: "
        f"{max_token_count}"
    )

    print(
        f"Min token count: "
        f"{min_token_count}"
    )

    print(
        f"Review chunks: "
        f"{len(review_chunks)}"
    )

    print(
        f"Overlap: "
        f"{OVERLAP}"
    )

    print()

    for chunk in review_chunks:

        print(
            f"- {chunk['chunk_id']} | "
            f"{chunk['doc_id']} | "
            f"{chunk['pdf_pages']} | "
            f"{chunk['review_reason']}"
        )