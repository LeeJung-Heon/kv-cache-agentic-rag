# 문서 임베딩과 FAISS 검색

전처리 담당자가 만든 `database/chunks.jsonl`을 그대로 BGE-M3로 임베딩한다. 한 줄이 한 청크이며, 임베딩 단계는 PDF 추출, 텍스트 정리, 청킹, overlap 처리를 다시 수행하지 않는다. 따라서 전처리 담당자가 청크 크기나 overlap을 변경해 JSONL을 다시 생성하면, 이 명령으로 FAISS 인덱스 전체를 다시 생성하면 된다.

## 입력

```text
database/
  chunks.jsonl      # 한 줄 = 확정된 청크
  manifest.json     # 문서별 출처, 허용 여부, URL, 공개일
```

`chunks.jsonl`의 `chunk_id`, `document_title`, `pdf_pages`, `section`, `section_path`, `role`, `needs_review`, `review_reason`은 `artifacts/faiss/chunks.json`에 보존된다. 검색을 위한 그룹은 별도 `index_group` 필드로 추가한다.

| 원본 technology | index_group |
|---|---|
| `MLA` | `sw` |
| `CXL-PNM` | `hw` |
| `COMMON` | `common` |

원래 `technology` 값은 결과 메타데이터에 그대로 남는다. `pdf_pages`의 첫 페이지는 기존 Evidence 형식과 호환되도록 `page`에도 저장한다.

문서별 URL과 공개일은 `database/manifest.json`에서 가져온다. 해당 파일의 URL, 공개일을 수정한 뒤에는 인덱스를 다시 생성해야 한다.

`needs_review=true` 청크는 기본적으로 원본 확인 전 검색 인덱스에서 제외한다. 해당 청크와 `review_reason`은 `chunks.json`에는 보존된다. 검토 대상까지 임시로 검색하려면 `INCLUDE_REVIEW_CHUNKS=true` 또는 `--include-review-chunks`를 사용한다.

## 생성

```sh
uv sync --locked
uv run -m ingest.embedding.build_index
```

출력은 다음과 같이 생성된다.

```text
artifacts/faiss/
  sw.faiss
  hw.faiss
  common.faiss
  chunks.json
  manifest.json
```

`manifest.json`에는 BGE-M3 revision, 차원, 입력 문서, 전체 청크 수, 실제 색인 수, 제외된 검토 청크 ID와 인덱스 행 연결을 기록한다. 입력과 설정이 같으면 기존 산출물을 재사용하며, JSONL 또는 문서 manifest가 바뀌면 다시 임베딩한다.

## 검색

```python
from service.retrieval.paper_index import get_paper_index

index = get_paper_index()
results = index.search("KV cache 메모리를 줄이는 원리", side="sw", k=5)
```

`side="sw"`는 SW와 공통 인덱스를, `side="hw"`는 HW와 공통 인덱스를 함께 검색한다. 결과에는 `id`, `title`, `url`, `published_at`, `page`, `pdf_pages`, `section`, `section_path`, `role`, `needs_review`, `review_reason`, `text`, `score`가 포함된다.

`artifacts/faiss/` 전체를 공유하면 다른 팀원은 원본 PDF와 JSONL 없이 검색할 수 있다. 모델 파일은 로컬에 있거나 다운로드 가능해야 한다.

## 검증

```sh
uv run --locked python -m unittest tests.test_embedding -v
uv run --locked check_domain.py
uv run --locked check_technical.py
```
