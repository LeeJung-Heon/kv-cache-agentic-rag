import json
import re
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[2]

DATABASE_DIR = ROOT_DIR / "database"
PAGES_PATH = DATABASE_DIR / "pages.jsonl"
MANIFEST_PATH = DATABASE_DIR / "manifest.json"
SECTIONS_PATH = DATABASE_DIR / "sections.jsonl"


# Markdown heading
# 예: ## **3 MHA2MLA**
HEADING_RE = re.compile(
    r"^(#{1,6})\s+(.+?)\s*$"
)

# Section number
# 예: 2 / 2.1 / 3.2.1
SECTION_NUMBER_RE = re.compile(
    r"(?<![\w.])(\d+(?:\.\d+)*)(?=\s+[A-Za-z])"
)


def clean_heading_text(text: str) -> str:
    """
    heading 안의 Markdown 장식만 제거한다.
    """

    text = text.replace("\\*", "*")
    text = text.replace("\\_", "_")
    text = text.replace("**", "")

    return text.strip()


def section_number_tuple(section_number: str):
    """
    '3.2.1' -> (3, 2, 1)
    """

    return tuple(
        int(part)
        for part in section_number.split(".")
    )


def split_compound_heading(text: str):
    """
    한 줄에 여러 section heading이 붙은 경우 분리한다.

    예:
    2 Preliminary 2.1 Multi-Head Attention (MHA)

    ->
    2 Preliminary
    2.1 Multi-Head Attention (MHA)
    """

    text = clean_heading_text(text)

    matches = list(
        SECTION_NUMBER_RE.finditer(text)
    )

    if not matches:
        return []

    headings = []

    for i, match in enumerate(matches):
        start = match.start()

        if i + 1 < len(matches):
            end = matches[i + 1].start()
        else:
            end = len(text)

        heading = text[start:end].strip()

        section_number = match.group(1)

        level = (
            section_number.count(".") + 1
        )

        headings.append(
            (
                heading,
                level,
                section_number_tuple(
                    section_number
                ),
            )
        )

    return headings


def extract_heading(line: str):
    """
    Markdown heading 한 줄을 section 정보로 변환한다.
    """

    match = HEADING_RE.match(
        line.strip()
    )

    if not match:
        return []

    markdown_level = len(
        match.group(1)
    )

    heading_text = clean_heading_text(
        match.group(2)
    )

    numbered = split_compound_heading(
        heading_text
    )

    if numbered:
        return [
            {
                "title": title,
                "level": level,
                "number": number,
            }
            for title, level, number
            in numbered
        ]

    # Abstract / References 등 번호 없는 heading
    return [
        {
            "title": heading_text,
            "level": markdown_level,
            "number": None,
        }
    ]


def load_pages():
    pages = []

    with open(
        PAGES_PATH,
        "r",
        encoding="utf-8",
    ) as f:

        for line in f:
            if line.strip():
                pages.append(
                    json.loads(line)
                )

    return pages


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


def clear_old_section_review(page):
    """
    section_parser를 다시 실행해도
    non_monotonic review 사유가 중복되지 않도록
    이전 section-order review만 제거한다.
    """

    reason = page.get(
        "review_reason"
    )

    if not reason:
        return

    reasons = [
        item.strip()
        for item in reason.split(";")
        if item.strip()
    ]

    reasons = [
        item
        for item in reasons
        if not item.startswith(
            "non_monotonic_section_order:"
        )
    ]

    if reasons:
        page["review_reason"] = (
            "; ".join(reasons)
        )
        page["needs_review"] = True
    else:
        page["review_reason"] = None
        page["needs_review"] = False


def append_review_reason(
    page,
    reason: str,
):
    page["needs_review"] = True

    old_reason = page.get(
        "review_reason"
    )

    if old_reason:
        page["review_reason"] = (
            f"{old_reason}; {reason}"
        )
    else:
        page["review_reason"] = reason


def flush_buffer(
    sections,
    buffer,
    doc_id,
    technology,
    current_section,
    current_level,
    current_path,
):
    """
    section 본문을 저장하면서
    실제 PDF page별 text도 함께 보존한다.
    """

    if not buffer:
        return

    page_map = {}

    for item in buffer:
        pdf_page = item["pdf_page"]
        line = item["text"]

        if pdf_page not in page_map:
            page_map[pdf_page] = []

        page_map[pdf_page].append(
            line
        )

    page_texts = []

    for pdf_page in sorted(
        page_map.keys()
    ):
        page_text = "\n".join(
            page_map[pdf_page]
        ).strip()

        if page_text:
            page_texts.append(
                {
                    "pdf_page": pdf_page,
                    "text": page_text,
                }
            )

    if not page_texts:
        return

    full_text = "\n\n".join(
        item["text"]
        for item in page_texts
    )

    pdf_pages = [
        item["pdf_page"]
        for item in page_texts
    ]

    sections.append(
        {
            "doc_id": doc_id,
            "technology": technology,

            "section": current_section,
            "section_level": current_level,
            "section_path": (
                current_path.copy()
            ),

            "pdf_pages": pdf_pages,

            # chunk에서 정확한 page provenance를
            # 계산하기 위해 유지
            "page_texts": page_texts,

            "text": full_text,
        }
    )


def parse_document(
    pages,
    document,
):
    sections = []

    doc_id = document["doc_id"]
    technology = document["technology"]

    current_section = "Front Matter"
    current_level = 0
    current_path = []

    # level -> heading
    hierarchy = {}

    buffer = []

    last_number = None

    for page in pages:
        pdf_page = page["pdf_page"]

        text = page.get(
            "clean_text",
            "",
        )

        for line in text.splitlines():

            headings = extract_heading(
                line
            )

            # 일반 본문
            if not headings:
                buffer.append(
                    {
                        "pdf_page": pdf_page,
                        "text": line,
                    }
                )
                continue

            # 새 heading이 나왔으므로
            # 직전 section의 본문 저장
            flush_buffer(
                sections=sections,
                buffer=buffer,
                doc_id=doc_id,
                technology=technology,
                current_section=(
                    current_section
                ),
                current_level=(
                    current_level
                ),
                current_path=(
                    current_path
                ),
            )

            buffer = []

            # 같은 line에
            # 2 Preliminary + 2.1 MHA
            # 같이 들어올 수 있음
            for heading in headings:

                title = heading["title"]
                level = heading["level"]
                number = heading["number"]

                # section 번호가 역행하면
                # 원본 PDF review 대상
                if (
                    number is not None
                    and last_number
                    is not None
                    and number
                    < last_number
                ):
                    append_review_reason(
                        page,
                        (
                            "non_monotonic_section_order: "
                            f"{'.'.join(map(str, last_number))} "
                            "-> "
                            f"{'.'.join(map(str, number))}"
                        ),
                    )

                if number is not None:
                    last_number = number

                hierarchy[level] = title

                # 현재 level보다 아래의
                # 기존 hierarchy 제거
                for existing_level in list(
                    hierarchy.keys()
                ):
                    if (
                        existing_level
                        > level
                    ):
                        del hierarchy[
                            existing_level
                        ]

                current_section = title
                current_level = level

                current_path = [
                    hierarchy[key]
                    for key in sorted(
                        hierarchy.keys()
                    )
                ]

    # 문서 마지막 section 저장
    flush_buffer(
        sections=sections,
        buffer=buffer,
        doc_id=doc_id,
        technology=technology,
        current_section=current_section,
        current_level=current_level,
        current_path=current_path,
    )

    return sections


def save_pages(pages):
    """
    section 순서 이상 여부를
    pages.jsonl에도 반영한다.
    """

    with open(
        PAGES_PATH,
        "w",
        encoding="utf-8",
    ) as f:

        for page in pages:
            f.write(
                json.dumps(
                    page,
                    ensure_ascii=False,
                )
                + "\n"
            )


def save_sections(sections):
    with open(
        SECTIONS_PATH,
        "w",
        encoding="utf-8",
    ) as f:

        for section in sections:
            f.write(
                json.dumps(
                    section,
                    ensure_ascii=False,
                )
                + "\n"
            )


def main():
    pages = load_pages()
    manifest = load_manifest()

    # 재실행 시 동일 review reason이
    # 계속 누적되는 것 방지
    for page in pages:
        clear_old_section_review(
            page
        )

    all_sections = []

    for doc_id, document in (
        manifest.items()
    ):

        if not document.get(
            "allowed",
            True,
        ):
            continue

        doc_pages = [
            page
            for page in pages
            if page["doc_id"]
            == doc_id
        ]

        doc_pages.sort(
            key=lambda x: x["pdf_page"]
        )

        sections = parse_document(
            pages=doc_pages,
            document=document,
        )

        all_sections.extend(
            sections
        )

        print(
            f"{doc_id}: "
            f"{len(sections)} sections"
        )

    save_pages(pages)
    save_sections(
        all_sections
    )

    review_pages = [
        page
        for page in pages
        if page.get(
            "needs_review"
        )
    ]

    print()
    print(
        f"Sections saved: "
        f"{SECTIONS_PATH}"
    )
    print(
        f"Total sections: "
        f"{len(all_sections)}"
    )
    print(
        f"Pages needing review: "
        f"{len(review_pages)}"
    )

    for page in review_pages:
        print(
            f"- {page['doc_id']} "
            f"p.{page['pdf_page']}: "
            f"{page['review_reason']}"
        )


if __name__ == "__main__":
    main()