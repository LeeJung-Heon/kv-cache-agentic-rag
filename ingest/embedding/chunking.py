"""페이지와 Markdown 소절 경계를 보존하는 토큰 기준 청킹."""

import re

from ingest.embedding.documents import Document, Page


# 청킹 설정
CHUNK_SIZE = 512
CHUNK_OVERLAP = 64  # 12.5%


def embedding_text(
    title: str,
    section: str,
    text: str,
    include_section: bool,
) -> str:
    """
    실제 임베딩에 사용할 문자열을 만든다.

    title
    + section
    + body text
    """

    parts = [title.strip()]

    if include_section and section.strip():
        parts.append(section.strip())

    parts.append(text.strip())

    return "\n\n".join(parts)


def sections(text: str):
    """
    Markdown heading을 기준으로 section을 분리한다.

    예:
    ## 3 MHA2MLA
    ### 3.1 Partial-RoPE
    """

    heading = ""
    lines = []

    for line in text.splitlines():

        match = re.match(
            r"^#{1,6}\s+(.+?)\s*$",
            line,
        )

        if match:

            body = "\n".join(lines).strip()

            if body:
                yield heading, body

            heading = match.group(1).strip()

            # **3 MHA2MLA** 같은 Markdown bold 제거
            heading = re.sub(
                r"^\*+|\*+$",
                "",
                heading,
            ).strip()

            lines = []

        else:
            lines.append(line)

    body = "\n".join(lines).strip()

    if body:
        yield heading, body


def split_text(
    text: str,
    title: str,
    section: str,
    tokenizers: list,
    *,
    size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
    include_section: bool = True,
):
    """
    title + section + body 전체가 size 이하가 되도록
    text를 나눈다.

    이전 chunk의 마지막 overlap token을
    다음 chunk에 다시 포함한다.
    """

    if not tokenizers:
        raise ValueError(
            "토크나이저가 필요합니다."
        )

    if not 0 <= overlap < size:
        raise ValueError(
            "CHUNK_OVERLAP은 0 이상 "
            "CHUNK_SIZE 미만이어야 합니다."
        )

    # overlap 사용 시 10~20% 범위 검증
    if overlap:
        overlap_ratio = overlap / size

        if not 0.10 <= overlap_ratio <= 0.20:
            raise ValueError(
                "CHUNK_OVERLAP은 CHUNK_SIZE의 "
                "10~20% 범위로 설정하세요. "
                f"현재: {overlap_ratio:.1%}"
            )

    # offset_mapping은 fast tokenizer 필요
    if (
        overlap
        and not getattr(
            tokenizers[0],
            "is_fast",
            False,
        )
    ):
        raise ValueError(
            "Overlap 계산을 위해 첫 번째 tokenizer는 "
            "fast tokenizer여야 합니다."
        )

    def fits(body: str) -> bool:
        """
        title + section + body를 합친 최종 입력이
        모든 tokenizer에서 size 이하인지 확인한다.
        """

        value = embedding_text(
            title,
            section,
            body,
            include_section,
        )

        return all(
            len(
                tokenizer.encode(
                    value,
                    add_special_tokens=True,
                )
            )
            <= size
            for tokenizer in tokenizers
        )

    # 제목 + section만으로 제한 초과 여부
    if not fits(""):
        raise ValueError(
            "제목과 소절명만으로 "
            "CHUNK_SIZE를 초과합니다. "
            "설정을 조정하세요."
        )

    start = 0

    while start < len(text):

        rest = text[start:]

        if not rest.strip():
            break

        # 남은 본문 전체가 들어가면 마지막 chunk
        if fits(rest):
            yield rest.strip()
            break

        # ---------------------------------
        # 최대한 많이 들어가는 문자 위치 탐색
        # ---------------------------------

        low = 0
        high = len(rest)

        while low < high:

            middle = (
                low + high + 1
            ) // 2

            if fits(rest[:middle]):
                low = middle
            else:
                high = middle - 1

        # ---------------------------------
        # 자연스러운 경계 우선
        #
        # 1. 문단
        # 2. 문장
        # 3. 단어
        # ---------------------------------

        end = low

        boundary_patterns = (
            r"\n\s*\n",
            r"[.!?。]\s+",
            r"\s+",
        )

        for pattern in boundary_patterns:

            boundaries = [
                match.end()
                for match in re.finditer(
                    pattern,
                    rest[:low],
                )
                if match.end()
                >= low // 2
            ]

            if boundaries:
                end = boundaries[-1]
                break

        piece = rest[:end].strip()

        # tokenizer 특성상 문자 수와
        # token 수가 정확히 비례하지 않으므로 재검사
        while (
            piece
            and not fits(piece)
        ):
            end -= 1

            piece = (
                rest[:end]
                .strip()
            )

        if not piece:
            raise ValueError(
                "CHUNK_SIZE 안에 "
                "본문을 넣을 수 없습니다."
            )

        yield piece

        # ---------------------------------
        # 다음 chunk 시작 위치 계산
        # ---------------------------------

        advance = end

        if overlap:

            raw_piece = (
                rest[:end]
                .rstrip()
            )

            encoding = tokenizers[0](
                raw_piece,
                add_special_tokens=False,
                return_offsets_mapping=True,
            )

            offsets = encoding[
                "offset_mapping"
            ]

            if not offsets:
                raise ValueError(
                    "Overlap 계산을 위한 "
                    "token offset이 없습니다."
                )

            # chunk가 overlap보다 작은 예외 방지
            actual_overlap = min(
                overlap,
                len(offsets) - 1,
            )

            if actual_overlap <= 0:
                raise ValueError(
                    "청크 본문이 너무 짧아 "
                    "overlap을 적용할 수 없습니다."
                )

            # 마지막 64 token을 다음 chunk에 다시 포함
            advance = offsets[
                -actual_overlap
            ][0]

            if advance <= 0:
                raise ValueError(
                    "Overlap 때문에 청킹이 "
                    "진행되지 않습니다."
                )

        start += advance


def chunk_document(
    doc: Document,
    pages: list[Page],
    tokenizers: list,
    *,
    size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
    include_section: bool = True,
) -> list[dict]:
    """
    문서를 page → section → chunk 순서로 분할한다.

    페이지 경계는 넘지 않는다.
    """

    if not tokenizers:
        raise ValueError(
            "토크나이저가 필요합니다."
        )

    if not 0 <= overlap < size:
        raise ValueError(
            "유효한 청킹 설정이 필요합니다."
        )

    chunks = []

    for page in pages:

        number = 0

        for section, text in sections(
            page.text
        ):

            for piece in split_text(
                text,
                doc.title,
                section,
                tokenizers,
                size=size,
                overlap=overlap,
                include_section=include_section,
            ):

                number += 1

                key = (
                    f"{doc.technology}_"
                    f"{doc.doc_id}_"
                    f"p{page.page}_"
                    f"c{number}"
                )

                embed_text = embedding_text(
                    doc.title,
                    section,
                    piece,
                    include_section,
                )

                # 첫 번째 tokenizer 기준
                # 실제 embedding token 수 기록
                token_count = len(
                    tokenizers[0].encode(
                        embed_text,
                        add_special_tokens=True,
                    )
                )

                if token_count > size:
                    raise ValueError(
                        f"{key}: "
                        f"{token_count} tokens > "
                        f"{size}"
                    )

                chunks.append(
                    {
                        "id": key,
                        "doc_id": doc.doc_id,
                        "technology": doc.technology,

                        "title": doc.title,
                        "url": doc.url,
                        "file": doc.path,

                        "page": page.page,
                        "pdf_pages": [
                            page.page
                        ],

                        "section": section,

                        # 본문 자체
                        "text": piece, #본문만 !!

                        # 실제 임베딩 입력
                        "embedding_text": (   # 실제 임베딩 대상
                            embed_text
                        ),

                        "token_count": (
                            token_count
                        ),
                    }
                )

    if not chunks:
        raise ValueError(
            f"{doc.doc_id}: "
            "청킹할 본문이 없습니다."
        )

    return chunks
