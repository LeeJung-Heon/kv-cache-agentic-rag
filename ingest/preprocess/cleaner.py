import json
import re
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[2]
PAGES_PATH = ROOT_DIR / "database" / "pages.jsonl"


def clean_text(text: str) -> str:
    # 1. Markdown/HTML 줄바꿈
    text = re.sub(r"<br\s*/?>", "\n", text, flags=re.IGNORECASE)

    # 2. 강조용 HTML 태그는 내용만 남김
    text = re.sub(r"</?mark>", "", text, flags=re.IGNORECASE)

    # 3. 위첨자 / 아래첨자는 의미를 어느 정도 보존
    text = re.sub(
        r"<sup>(.*?)</sup>",
        r"^(\1)",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )

    text = re.sub(
        r"<sub>(.*?)</sub>",
        r"_(\1)",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )

    # 4. 기타 남은 HTML 태그 제거
    text = re.sub(r"<[^>]+>", "", text)

    # 5. 불필요한 Markdown picture comment가 혹시 남아 있으면 제거
    text = re.sub(
        r"<!--.*?-->",
        "",
        text,
        flags=re.DOTALL,
    )

    # 6. 줄 끝 불필요한 공백 제거
    lines = [line.rstrip() for line in text.splitlines()]

    # 7. 페이지 맨 앞/뒤의 단독 숫자 페이지 번호 제거
    while lines and re.fullmatch(r"\d{1,5}", lines[0].strip()):
        lines.pop(0)

    while lines and re.fullmatch(r"\d{1,5}", lines[-1].strip()):
        lines.pop()

    text = "\n".join(lines)

    # 8. 지나치게 많은 빈 줄 정리
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def clean_pages():
    pages = []

    with open(PAGES_PATH, "r", encoding="utf-8") as f:
        for line in f:
            page = json.loads(line)

            page["clean_text"] = clean_text(
                page["raw_text"]
            )

            pages.append(page)

    return pages


def save_pages(pages):
    with open(PAGES_PATH, "w", encoding="utf-8") as f:
        for page in pages:
            f.write(
                json.dumps(
                    page,
                    ensure_ascii=False,
                )
                + "\n"
            )


if __name__ == "__main__":
    pages = clean_pages()
    save_pages(pages)

    print(f"Cleaned pages: {len(pages)}")
    print(f"Saved: {PAGES_PATH}")