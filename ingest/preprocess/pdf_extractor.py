import json
import re
from pathlib import Path

import pymupdf
import pymupdf4llm

from ingest.preprocess.manifest import DOCUMENTS


ROOT_DIR = Path(__file__).resolve().parents[2]
DOCS_DIR = ROOT_DIR / "docs"
DATABASE_DIR = ROOT_DIR / "database"
PAGES_PATH = DATABASE_DIR / "pages.jsonl"


def remove_picture_text(markdown_text: str) -> str:
    """
    PyMuPDF4LLM이 그림 내부에서 추출한 텍스트를 제거한다.

    제거 범위:
    <!-- Start of picture text -->
    ...
    <!-- End of picture text -->

    Figure caption 등 picture block 바깥의 텍스트는 유지한다.
    """

    pattern = r"<!-- Start of picture text -->.*?<!-- End of picture text -->"

    cleaned_text = re.sub(
        pattern,
        "",
        markdown_text,
        flags=re.DOTALL,
    )

    return cleaned_text.strip()


def extract_pdf(file_path: Path, doc_id: str):
    pages = []

    # raster image 개수 확인용
    with pymupdf.open(file_path) as pdf:
        image_counts = [
            len(page.get_images(full=True))
            for page in pdf
        ]

    # layout-aware Markdown 추출
    md_pages = pymupdf4llm.to_markdown(
        str(file_path),
        page_chunks=True,
        header=False,
        footer=False,
        use_ocr=False,
        show_progress=False,
    )

    for page_index, page_data in enumerate(md_pages):
        pdf_page = page_index + 1

        raw_text = page_data["text"]

        # Figure 내부의 깨진 텍스트 제거
        cleaned_picture_text = remove_picture_text(raw_text)

        page = {
            "doc_id": doc_id,
            "pdf_page": pdf_page,
            "raw_text": cleaned_picture_text,
            "image_count": image_counts[page_index],

            "needs_review": False,
            "review_reason": None,
        }

        pages.append(page)

    return pages


def extract_all_documents():
    all_pages = []

    for document in DOCUMENTS:
        file_path = DOCS_DIR / document["file_name"]

        print(f"Extracting: {document['doc_id']}")

        pages = extract_pdf(
            file_path=file_path,
            doc_id=document["doc_id"],
        )

        all_pages.extend(pages)

        print(
            f"  pages={len(pages)}, "
            f"images={sum(page['image_count'] for page in pages)}"
        )

    return all_pages


def save_pages(pages):
    DATABASE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

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


if __name__ == "__main__":
    pages = extract_all_documents()
    save_pages(pages)

    print()
    print(f"Pages saved: {PAGES_PATH}")
    print(f"Total pages: {len(pages)}")