# KV cache 다관점 평가

DeepSeek-V2의 MLA(SW)와 CXL-PNM(HW)을 두 논문 및 공개 웹 자료로 비교하는 LangGraph 파이프라인입니다. 특정 기술의 우열이나 추천 대신 관점별 평가 차이와 불확실성을 정리합니다.

```text
공통 입력 → 기술 선정 → 기술 조사 (논문 RAG + TRL)
                          ├─ 근거 부족 + 잔여 횟수 → 검색어 수정 → 기술 조사
                          └─ 완료 또는 한도 도달
                               ├─ 시장 평가 (Tavily) ───────┐
                               ├─ 이해관계자 평가 (Tavily) ─┼→ 평가 종합 → 보고서 → Markdown / PDF
                               └─ 도메인 평가 (논문 RAG) ──┘
```

기술·도메인은 [선정 문서](docs/TECHNOLOGY_DOMAIN_SELECTION.md)에 따라 사람이 확정했습니다. 기술 선정 노드는 LLM으로 다시 고르지 않고 기술 ID·이름·접근 방식·선정 이유를 State에 기록합니다. 기본 도메인은 **데이터센터·클라우드 LLM 서빙**입니다.

각 평가 노드는 근거를 검색하고 구조화된 분석을 생성합니다. 기술 조사에서는 모델이 부족한 근거에 대한 수정 질의를 제안하며, 그래프가 재검색 횟수를 제한합니다. 시장·이해관계자·도메인 평가는 모두 끝난 뒤 한 번만 종합합니다.

## 실행

```sh
git clone https://github.com/chaeeunwang/kv-cache-agentic-rag.git
cd kv-cache-agentic-rag
uv sync --locked
cp .env.example .env
# .env의 OPENAI_API_KEY와 TAVILY_API_KEY 입력
uv run pipeline.py --max-technical-retries 2
```

기존 설정 파일을 명시해 사용할 수도 있습니다. 키는 복사하거나 결과물에 저장하지 않습니다.

```sh
uv run pipeline.py --env-file /path/to/your/.env
```

지정한 `.env` 값이 프로세스 환경변수보다 우선합니다. 기본 생성 모델은 `gpt-4.1-mini`이며 `OPENAI_MODEL`로 변경합니다. OpenAI 호환 엔드포인트는 `OPENAI_BASE_URL`로 지정합니다. 실제 실행에는 LLM 및 Tavily API 호출 비용이 발생할 수 있습니다.

`--domain`은 실행 도메인을 변경하고 `--selection-reason`은 첨부 문서에 실행별 선정 이유를 추가합니다. 도메인을 변경한 경우 원래 문서의 선정 이유를 새 도메인의 이유로 단정하지 않도록 입력에 표시합니다.

## 파일과 State

| 파일 | 역할 |
|---|---|
| `state.py` | Notion 설계의 GraphState, Technology, Evidence, Finding, AgentResult |
| `pipeline.py` | 기술 선정, 역할별 분석, 병렬 합류, 보고서, CLI |
| `service/agent/node/technical/` | 기술 조사 노드 묶음: `schema.py`(서브그래프 State·초안 스키마), `retrieval.py`(논문 검색 도구·근거 수집), `model.py`(전용 모델 생성·설정·호출), `prompts.py`(역할·TRL 판정표), `core.py`(순수 검증 규칙), `node.py` |
| `service/agent/graph/technical.py` | 기술 조사 서브그래프 `build_technical_research_graph(index, model=None, max_retries=2, rules=)` |
| `check_technical.py` | 기술 조사 서브그래프만 Mock 인덱스·모델로 점검 |
| `tests/` | API·모델 없이 도는 pytest (`uv run pytest -q`) |
| `rag.py` | PDF 로딩, E5 토큰 청킹, 임베딩 캐시, cosine 검색 |
| `report.py` | Markdown 저장 및 한국어 PDF 생성 |
| `check_graph.py` | API 없이 재검색·합류·오류·근거 참조 경로 검사 |
| `docs/` | 원본 논문 및 기술·도메인 선정 문서 |
| `outputs/<실행시각>/` | State, 보고서, 실패 시 오류 종류 (Git 제외) |

공통 입력은 `request`, `target_domain`, `evaluation_criteria`, `technical_retry_count=0`입니다. 결과 필드는 생성 전에는 생략합니다.

| 노드 | 주요 읽기 필드 | 갱신 필드 |
|---|---|---|
| 기술 선정 | 사람의 선정 문서와 공통 입력 | `technologies` |
| 기술 조사 (서브그래프) | 공통 입력, technologies | `technical_result`, `technical_retry_count`, `technical_queries`, `technical_missing_items` |
| 시장 평가 | 공통 입력, technologies, technical_result | `market_result` |
| 이해관계자 평가 | 공통 입력, technologies, technical_result | `stakeholder_result` |
| 도메인 평가 | 공통 입력, technologies, technical_result | `domain_result` |
| 평가 종합 | 네 평가 결과 | `synthesis_result` |
| 보고서 | 선정 문서, 네 평가 및 종합 | `report_markdown`, `report_evidence_ids` |

세 병렬 노드는 서로 다른 필드만 갱신하므로 reducer 없이 기본 덮어쓰기를 사용합니다. 도메인 평가는 동시에 생성 중인 시장 평가를 읽지 않습니다.

`AgentResult`는 `status`, `summary`, `findings`, `evidence`, `limitations`로 통일합니다. 각 Finding은 `technology_ids`, `claim`, `evidence_ids`, `is_inference`를 가지며, Evidence는 `id`, `source_type`, `title`, `url`, `page`, `published_at`, `excerpt`를 보존합니다. 근거 메타데이터와 원문은 검색 결과에서 코드가 생성하며 모델이 만들지 않습니다. LLM 응답의 내부 `criterion`, `next_queries`는 누락 판단과 재검색에만 사용하고 AgentResult에는 넣지 않습니다.

## 재검색 및 오류 정책

- `complete`: SW/HW 각각의 지정 평가 항목에 근거가 연결됨. 의미적 정확성이 자동 검증됐다는 뜻은 아닙니다.
- `partial`: 항목 누락 또는 모델이 판단한 근거 부족. 기술 조사의 필수 항목은 각 기술의 원리·성능·한계·TRL입니다.
- 기술 조사만 `partial`일 때 서브그래프 안에서 수정 질의로 재검색합니다. 재검색은 미충족 항목이 있는 기술의 논문만 다시 검색하며 질의는 최대 4개·500자입니다. `technical_retry_count`는 **추가 조사 횟수**이며 기본 2회, `--max-technical-retries 0`으로 비활성화할 수 있습니다. 설정 범위는 0~5입니다.
- 기술 조사의 근거 ID·기준 검증에 실패하면 실패 사유와 사용 가능한 ID를 피드백으로 넣어 한 번만 다시 생성하고, 두 번째도 실패하면 `error`로 끝냅니다. 논문 검색 자체가 실패하면 확보한 근거를 보존한 `error`를 반환합니다.
- 기술 조사의 TRL Finding은 선택 필드 `trl_assessment`(`level_or_range` 1~9 또는 범위, `as_of`=2026-09-21, `confidence`, `unverified_conditions`, `basis`="공개 정보 기반 추정")를 가지며, 판단 불가는 `level_or_range=null`과 한계 항목으로 남깁니다. 기술 조사 모델은 `service/agent/node/technical/model.py`의 `TECHNICAL_MODEL` 상수로 정하고 키·엔드포인트는 `config.settings`를 따릅니다.
- 재검색 후에도 부족하면 `partial`을 유지하고 `technical_missing_items`와 `limitations`에 남겨 병렬 평가로 진행합니다. 기술 조사 결과가 갱신될 때 기존 충족 항목도 포함하도록 요청합니다.
- 시장·이해관계자·도메인의 `partial`은 그대로 종합합니다. 상위 평가의 부족한 근거와 한계를 종합·보고서에서도 유지합니다.
- API·구조화 응답·근거 ID 오류는 해당 결과를 `error`로 기록합니다. 기술 조사 오류는 즉시 그래프를 끝냅니다. 병렬 평가 오류는 합류 후 종합을 `error`로 기록하고 보고서를 생성하지 않습니다. 오류에는 자동 재시도를 하지 않습니다.
- 보고서 형식·인용 오류 또는 PDF 생성 실패는 실행을 중단하고 `FAILED.txt`를 남깁니다. 부분 State는 보존하며 오류 메시지에 API 요청 원문이나 인증값을 저장하지 않습니다.

## RAG와 출처

- 사용자 선택 모델: `intfloat/multilingual-e5-large-instruct`. 다른 모델보다 우수하다는 비교 결과를 뜻하지 않습니다.
- [임베딩 모델 선정 설계](EMBEDDING_MODEL_SELECTION.md)는 BGE-M3를 잠정 선정합니다. 현재 실행 코드는 E5이며, 설계의 모델 및 청킹 설정은 아직 코드에 반영하지 않았습니다.
- 두 PDF 합계 200페이지 제한, 페이지별 400토큰 청크, 60토큰 overlap.
- 질의에 E5 instruction 적용, 문서는 instruction 없이 임베딩.
- L2 정규화한 벡터의 내적으로 cosine 검색, 기술별 상위 5개 반환.
- 공유 임베딩 모델의 추론 호출은 잠금으로 직렬화하여 MPS 동시 호출 충돌을 방지합니다. 평가 노드와 웹 검색은 병렬로 실행됩니다.
- 두 논문 규모에는 별도 벡터 DB 없이 NumPy를 사용합니다.
- `.cache`에 문서·청크·모델명 기반 캐시를 저장합니다. 원문 변경 시 새 캐시를 만듭니다.
- PDF 근거에는 페이지·청크 ID·논문 URL·원문 발췌, 웹 근거에는 제목·URL·발췌·제공되는 경우 발행일을 보존합니다. 미확인 날짜나 웹 페이지 번호는 null입니다.
- 보고서 인용 ID로 REFERENCE를 생성합니다. Finding의 기술 ID와 근거 ID는 실제 목록에 있어야 합니다. 알 수 없는 ID는 오류로 처리합니다. 최종 사용 근거는 report_evidence_ids에 기록합니다. 이는 인용 문장의 사실성 검증을 대신하지 않습니다.
- 웹 검색 스니펫은 원문 전체 검증이 아니므로 확정 사실과 해석을 구별하도록 지시합니다.

## 출력

`report.md`를 원본으로 `report.pdf`를 만듭니다. 목차는 SUMMARY → 분석 배경 → 대상 기술 선정 → 기술 개요 → 관점별 평가 → 종합 평가 및 시사점 → 분석의 한계 → REFERENCE입니다.

매 실행마다 새 폴더를 만들어 기존 결과를 보존합니다. 도구/API 오류는 숨기고 계속 진행하지 않으며 실행을 중단합니다. 완료한 단계는 `state.json`에 남습니다. PDF 생성에 실패해도 Markdown은 남습니다. 자동 재개 기능은 없습니다.

PDF에는 한국어 TTF가 필요합니다. macOS의 Arial Unicode 또는 Linux의 NanumGothic을 자동 사용하며, 다른 환경에서는 `.env`의 `PDF_FONT`에 TTF 파일 경로를 지정합니다. 출력은 문단·제목·목록을 대상으로 하며 표·수식 렌더러는 포함하지 않습니다.

시장·이해관계자·도메인 본문은 원래 분석의 인용이 요약 중 사라지지 않도록 보고서에 그대로 삽입합니다. 주장별 근거와 사실/추론 구분을 유지하며 모든 평가의 한계를 보고서에 명시적으로 포함합니다. 생성물은 제출 완료본이 아닌 검토용 초안입니다.

## 최소 확인

```sh
uv run pipeline.py --index-only  # API 없이 실제 PDF 임베딩 및 검색, 최초 모델 다운로드 가능
uv run check_graph.py           # 변경한 제어 흐름과 근거 연결만 확인
uv run check_technical.py       # 기술 조사 서브그래프만 Mock으로 점검
uv run pytest -q                # 기술 조사 규칙·재검색·오류 처리 (외부 API 없음)
```

개별 수정마다 전체 테스트나 모델 비교 평가를 실행할 필요는 없습니다. 생성된 보고서의 수치·인용 의미·공개정보 기반 TRL은 제출 전에 사람이 검토해야 합니다.

기존 실행 결과의 State는 이전 형식 그대로 보존합니다. 새 State로 자동 변환하거나 이전 결과에서 실행을 재개하지 않습니다. 임베딩 모델·청킹 설정은 이번 State 변경에서 수정하지 않았습니다.
