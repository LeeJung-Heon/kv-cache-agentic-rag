# 문서 임베딩과 FAISS 검색

담당 범위는 전처리 문서 로딩, 청킹, BGE-M3 문서 임베딩, FAISS 산출물 저장 및 공유, 질의 임베딩과 검색이다. `ingest/embedding/build_index.py`가 문서 색인을 만들고 `service/retrieval/paper_index.py`가 저장된 결과로 검색한다. LangGraph의 노드 및 평가 흐름은 별도 담당 범위다.

## 현재 구현과 확정할 사항

실제 전처리 문서는 아직 제공되지 않았다. 입력 로더는 아래 임시 계약의 JSON, TXT, Markdown을 지원하며 실제 형식이 다르면 `ingest/embedding/documents.py`에서 변환한다. PDF 추출이나 줄바꿈 정리를 다시 수행하지 않는다.

| 항목 | 초기 설정 | 문서 수령 후 확인 |
|---|---|---|
| 모델 | `BAAI/bge-m3`, dense | 설계서의 채택 모델 유지 여부 |
| 차원 | 모델에서 자동 확인 | BGE-M3는 통상 1024차원. `EMBEDDING_DIMENSION`은 차원 변경이 아니라 일치 검사용 |
| 청크 크기 | 제목과 소절명 및 특수 토큰 포함 최대 400 | 문단과 표 길이, 근거 경계에 맞게 조정 |
| overlap | 0 | 필요하면 BGE-M3 본문 토큰 기준으로 조정 |
| 소절명 | 임베딩 입력에 포함 | Markdown 제목 보존 여부 확인 |
| 배치 크기 | 8 | 장비 메모리에 맞게 조정 |
| 검색 개수 | 기술별 최종 top-5 | 한국어 평가 질문으로 검토 |

차원은 문서 내용에 따라 정하지 않는다. 실제 모델 출력으로 FAISS를 생성하고, 값을 지정한 경우 해당 차원과 다른 출력은 거부한다. 모델 출력의 차원을 임의로 잘라내지 않는다.

설계서 기준으로 BGE-M3 외에 E5와 Qwen3의 **토크나이저만** 로딩하여 모든 비교 모델에서 400토큰 한도를 검사한다. 세 모델의 가중치를 모두 로딩하는 것은 아니다. `CHUNK_TOKENIZER_MODELS=[]`로 두면 BGE-M3만 검사하며, 이 경우 원래 모델 비교 실험과 동일한 조건은 아니다. 실제 적용한 토크나이저 정보와 해시는 산출물에 기록한다.

문서에는 제목과 선택한 소절명을 포함한다. 질의에는 BGE-M3에 불필요한 E5 instruction을 붙이지 않는다. 문서와 질의 모두 같은 모델 revision 및 L2 정규화를 사용한다. BM25와 하이브리드 검색은 포함하지 않는다.

## 디렉터리

```text
docs/                              # 기존 원본 PDF 및 설명 문서
data/processed/
    documents.json                 # 전달받은 전처리 문서 목록
    sw/...
    hw/...
    common/...
ingest/embedding/
    documents.py                   # 입력 형식 변환 및 페이지 검증
    chunking.py                    # 페이지와 소절 경계를 보존하는 청킹
    model.py                       # 문서와 질의의 공통 임베딩 방식
    storage.py                     # FAISS 및 JSON 저장과 무결성 검사
    build_index.py                 # 색인 생성 명령
artifacts/faiss/
    sw.faiss                       # 해당 문서가 있는 그룹만 생성
    hw.faiss
    common.faiss
    chunks.json                    # 청크 본문과 출처
    manifest.json                  # 모델, 설정, 입력 해시, 인덱스 행 연결
service/retrieval/paper_index.py    # 공통 인덱스 로딩 및 질의 임베딩과 검색
```

입력 및 출력 설정의 상대 경로는 프로젝트 루트 기준이다. 문서 목록 안의 `path`는 `documents.json`이 있는 폴더 기준이다. `artifacts/faiss/`와 `data/processed/`의 실제 데이터는 Git에서 제외한다.

## 입력 계약

`docs/processed_manifest.example.json`을 `data/processed/documents.json`에 복사하고 실제 문서에 맞게 작성한다. 예시의 제목, 파일 경로, 쪽수는 실물과 대조해야 하며 제공 문서 풀 허용 여부를 확인한 뒤 `allowed`를 `true`로 바꾼다. 설정 예시는 실제 전처리 문서가 아니다.

```json
[
  {
    "doc_id": "deepseek_v2",
    "path": "sw/deepseek_v2.md",
    "title": "DeepSeek-V2",
    "technology": "sw",
    "url": "https://arxiv.org/abs/2405.04434",
    "published_at": null,
    "version": null,
    "original_pages": 52,
    "allowed": true
  }
]
```

- `doc_id`는 전체 문서에서 고유한 영문 소문자, 숫자, 밑줄 식별자다. 첫 글자는 영문이다.
- `technology`는 `sw`, `hw`, `common` 중 하나다. 두 기술에 공통인 문서는 한 번만 `common`으로 등록한다.
- `original_pages`는 추출한 페이지 수가 아닌 원본 PDF 전체 쪽수다. 등록된 원본 쪽수 합계가 200을 초과하면 색인 생성을 중단한다. 이 값의 원본 대조는 입력 제공자가 수행해야 한다.
- `url`, `published_at`, `version`은 확인된 값만 기록한다. 알 수 없는 날짜와 버전은 `null`로 둔다.
- 누락된 파일, 허용되지 않은 문서, 중복 문서 ID, 중복 등록 파일, 빈 본문은 오류로 알린다.

TXT/Markdown은 원본 PDF 페이지 마커가 필요하다. 번호는 1부터 세는 원본 PDF 페이지이며, 선택한 페이지만 포함할 수 있다.

```markdown
<!-- PAGE: 7 -->
## Multi-head Latent Attention

전처리된 첫 번째 문단.

전처리된 두 번째 문단.

<!-- PAGE: 8 -->
다음 페이지 본문.
```

JSON의 경우 다음 형식으로 동일한 정보를 전달한다.

```json
{
  "pages": [
    {"page": 7, "text": "## Multi-head Latent Attention\n\n전처리된 본문"},
    {"page": 8, "text": "다음 페이지 본문"}
  ]
}
```

페이지 정보가 없는 텍스트에서 PDF 페이지를 추측하지 않는다. 다른 형식으로 페이지 정보가 전달되면 입력 로더를 맞추면 된다. 표, 캡션, 수식의 추출 정확성은 원문 대조가 필요하다.

초기 구현은 PDF 페이지와 Markdown 소절 경계를 넘지 않으며 문단을 우선 묶는다. 긴 문단은 문장 또는 단어 경계에서 나눈다. 소절명은 Markdown 제목에서만 가져온다. 페이지 경계를 유지하므로 기존 보고서의 단일 `page` 필드에도 정확한 출처를 전달할 수 있다. `pdf_pages` 목록도 함께 저장한다. 여러 페이지를 하나의 청크로 합치는 변경은 인용 처리까지 함께 맞춘 뒤 적용해야 한다.

## 생성과 검색

```sh
uv sync --locked

# 실제 전처리 문서와 documents.json을 준비한 후 실행
uv run -m ingest.embedding.build_index

# 문서 수령 후 다른 설정으로 생성하는 예시
uv run -m ingest.embedding.build_index --manifest data/processed/documents.json --output artifacts/faiss --chunk-size 400 --chunk-overlap 0

# 저장된 인덱스 로딩 및 질의 임베딩과 검색 확인
uv run pipeline.py --index-only
```

생성 단계는 최초 실행 시 모델과 토크나이저를 다운로드할 수 있다. 모델 revision을 지정하지 않으면 실제로 로딩된 commit SHA를 기록한다. 문서, 모델 revision, 청크 및 청킹 조건이 같으면 검증된 기존 인덱스를 재사용하고 문서 임베딩을 생략한다. 재생성 시 완성한 파일을 교체하고 manifest를 마지막에 기록한다. 장기 보관할 버전은 별도 출력 폴더로 지정한다.

```python
from service.retrieval.paper_index import get_paper_index

index = get_paper_index()
results = index.search("KV-cache 메모리를 줄이는 원리는 무엇인가요?", side="sw", k=5)
```

`get_paper_index()`는 같은 프로세스에서 하나의 `PaperIndex`를 공유한다. 최초 호출에서는 인덱스와 JSON만 읽으며 여러 노드가 동시에 처음 호출해도 중복 생성하지 않는다. 첫 검색에서 manifest에 기록된 모델과 revision으로 질의 모델을 로딩하고 이후 재사용한다. 실행 시 원본 또는 전처리 파일은 필요하지 않다. 모델 파일은 로컬에 있거나 다운로드 가능해야 한다.

각 노드의 `retrieval.py`에서는 이 함수를 가져와 `index.search(query, side="sw", k=5)`를 호출한다. `side`는 `sw` 또는 `hw`이며, `k` 생략 시 `RETRIEVAL_TOP_K` 설정을 따른다. 인덱스나 설정을 교체한 경우 실행 중인 프로세스를 재시작하여 새 파일을 로딩한다.

SW는 SW와 공통 인덱스에서, HW는 HW와 공통 인덱스에서 각각 후보를 가져온다. 동일한 정규화 방식의 내적 점수로 후보를 합치고 청크 ID 중복을 제거한 뒤 최종 k개만 반환한다. 해당 그룹에 근거가 없으면 빈 목록을 반환하며 다른 기술 문서로 채우지 않는다. 점수는 정답 확률이 아니다.

반환값의 `id`, `title`, `url`, `page`, `text`, `score`와 `index.chunks`를 유지한다. 추가로 `doc_id`, `technology`, `section`, `pdf_pages`, `published_at` 등을 제공한다. 새 ID는 `sw_deepseek_v2_p7_c1` 형식이며 기존 `sw_p7_c1` 형식도 파이프라인의 인용 검사에서 계속 지원한다. 공통 문서만으로 특정 기술의 조사 항목을 충족했다고 계산하지 않는다.

## 공유와 검증

`artifacts/faiss/` 폴더 전체를 압축해 전달한다. `.faiss` 파일만 복사하면 본문과 출처를 찾을 수 없다. 팀원은 폴더를 같은 경로에 풀고 `uv sync --locked` 후 `get_paper_index()`를 사용한다. 원본 및 전처리 문서는 재현과 검토가 필요할 때 별도로 전달한다. 서로 다른 생성 버전의 파일을 섞으면 해시와 행 번호 검증에서 오류가 발생한다.

```sh
uv run --locked python -m unittest discover -s tests -v
uv run --locked check_graph.py
```

자동 검증은 테스트용 임베딩과 실제 FAISS를 사용하므로 모델 다운로드나 LLM/Tavily 호출이 없다. 청킹, 기술 필터와 공통 문서 결합, NumPy 내적과의 검색 일치, 저장 및 재로딩, 문서 없이 공유본 검색, 메타데이터 불일치와 동시 질의 모델 로딩을 검사한다.

실제 문서 수령 후에는 페이지 인용과 토큰 길이를 원문에 대조하고 한국어 평가 질문으로 검색 품질을 확인해야 한다. 설계서의 123개 청크 및 기존 임베딩 지표는 이번 결과로 간주하지 않는다. 모델 revision, 청크 수와 실제 측정값은 새로 기록한다.
