# 시장성·이해관계자 평가 에이전트 작업 기록

담당: 박현준 — 시장성·이해관계자 평가 에이전트 (Tavily 기반). 담당 State 필드는 `market_result`, `stakeholder_result`.

| 팀원 | 역할 | 이 에이전트와 맞물리는 부분 |
|---|---|---|
| 박지원 | 전처리 | 없음 |
| 문진영 | 임베딩 | 없음 |
| 이중헌 | 기술 agent | `technical_result.evidence` 재인용 (스키마 확인 필요) |
| 왕채은 | 도메인 agent | LLM 출력에 확장 필드 추가됨 (코드 수정 불필요) |
| 강용현 | 종합 agent | 확장 필드명·값, 오류 정책 차이, 종합도 확장 필드 출력, **summary에 인용 없는 수치가 들어올 수 있음** |

설계 기준: RAG 기반 SW·HW 기술 평가 에이전트 설계서 (평가 기준일 2026-09-21)

## 작업 순서

| 순서 | 작업 | 브랜치 | 상태 |
|---|---|---|---|
| 1 | Finding 확장 필드 추가 | `feat/finding-extension` | 완료 (팀 공유 필요) |
| 2 | `evidence_schema.py`, `query_templates.py` | `feat/tavily-agent` | 완료 |
| 3 | `tavily_client.py` + 로직 테스트 | `feat/tavily-agent` | 완료 |
| 4 | 시장성 에이전트 | `feat/tavily-agent` | 완료 (technical_result는 state.AgentResult 계약 기준, 실제 출력과 대조 필요) |
| 5 | 이해관계자 에이전트 | `feat/tavily-agent` | 완료 |
| 6 | 실제 Tavily API 통합 테스트 | `feat/tavily-agent` | 진행 중 (hw_01 시장성 6회 실측 완료, 이해관계자·sw_01 남음) |
| 7 | 팀 그래프에 노드 연결 | 미정 | 대기 (기술 조사·평가 종합 노드 완성 후) |

## 전제

- 설계서 2.2절은 시장성·이해관계자를 RAG X로, 2.3·3.5절은 문서 근거 재인용 O로 적고 있다. 이를
  "기술 조사 에이전트가 만든 `technical_result.evidence`에서 시장·운영 성격 근거를 재인용한다"로 해석한다.
  해석이 틀리면 프롬프트 설계만 수정한다.
- 기준 문서는 설계서다. 레포 `pipeline.py`는 빠른 목업이므로, 둘이 다르면 설계서를 따른다.

## 1단계: Finding 확장 필드

### 결정

| 필드 | 값 | 적용 | 설계서 근거 |
|---|---|---|---|
| `claim_type` | `fact` / `opinion` / `forecast` / `None` | 모든 에이전트 (시장성·이해관계자는 필수로 검증) | 3.2 "사실·전망·간접 지표 분리", 5.2 |
| `scope` | `direct` / `adjacent` / `None` | 시장성 | 3.4 "직접 시장과 연관 시장 분리" |
| `stage` | `announced` / `pilot` / `production` / `None` | 상용화·채택 | 3.4 "계획 발표, 실증, 실제 운영 구분" |
| `stance` | `positive` / `negative` / `mixed` / `unknown` / `None` | 입장 판단 | 3.2 "긍정·부정·혼합·확인 불가" |

- `None`은 해당 관점에 적용되지 않는 항목이라는 뜻이다.
- `claim_type`은 원문 진술의 성격이고, `is_inference`는 에이전트가 해석했는지 여부다. 두 필드는 서로 독립이다.
  예: "Gartner가 X를 전망" → `claim_type=forecast`, `is_inference=false`.
- `DraftFinding`의 네 필드는 `default=None`인 nullable 필드다. OpenAI SDK의 strict 변환이 `default=None`을
  제거하고 모든 필드를 required로 보내므로 LLM은 네 필드를 항상 출력한다. 기본값 덕분에 다른 팀원의 코드와
  fixture(`check_graph.py`, `check_domain.py`)는 수정 없이 그대로 동작한다. 충돌을 줄이기 위한 선택이다.
- `stance`는 Finding에만 둔다. 어느 질의(긍정/부정)로 수집했는지는 Evidence에 넣지 않고 에이전트 내부에서만
  관리한다. 같은 Evidence ID에 에이전트마다 다른 값이 들어가면 `collect_sources`가 EvidenceError를 낸다.
- 루트 `state.py`(현재 `pipeline.py`가 사용)와 `service/schema/state.py`(신규 구조)를 동기화했다.
- 모듈 위치는 팀 컨벤션(`service/agent/node/domain.py`)을 따른다. `agents/` 폴더는 만들지 않는다.

### 변경 파일

- `state.py`: Literal 타입 4개, `Finding`·`DraftFinding` 필드 추가 (추가만, 기존 줄 변경 없음)
- `service/schema/state.py`: `Finding` 필드 추가 (추가만)

### 검증 (API 호출 없음)

- `uv run check_graph.py`, `uv run check_domain.py` PASS (fixture 수정 없음)
- OpenAI strict 스키마 변환 결과 4개 필드 모두 required·null 허용, 기본값 제거됨
- 잘못된 enum 값은 pydantic 검증에서 거부
- `normalize_result` 결과의 공개 Finding에 4개 필드 포함

### 팀 공유 필요

1. `AnalysisDraft`를 모든 에이전트가 공유하므로, 기술 조사·도메인·종합 에이전트의 LLM 출력에도 4개 필드가
   생긴다(해당 없으면 `null`). 코드 수정은 필요 없다.
2. 공용 `Finding` 결과에 4개 키가 항상 포함된다. 결과를 파싱하는 종합·보고서 쪽에서 알아야 한다.
3. 루트 `state.py`와 `service/schema/state.py` 중 어느 쪽을 기준으로 삼을지 결정이 필요하다.
4. 오류 정책 차이: 레포는 부분 실패를 `error`로 처리하고 중단하지만, 이 에이전트는 설계서 기준으로
   한쪽 질의 실패를 `limitations`에 기록하고 계속 진행한다.
5. 레포 `pipeline.CRITERIA`의 market(4개)·stakeholder(5개) 목업 기준을 설계서 3.4절 기준(각 3개)으로
   바꿔야 한다. 이 에이전트는 `service/agent/tavily/query_templates.CRITERIA`에 설계서 기준을 정의해 사용한다.

## 2단계: 질의 템플릿과 근거 메타데이터

### 결정

- 위치: `service/agent/tavily/`. 공용 파일(`pipeline.py`, `pyproject.toml`)은 수정하지 않았다.
- 평가 기준(설계서 3.4절)
  - 시장성: 시장 규모·성장성 / 상용화·채택 / 생태계
  - 이해관계자: 경쟁 기술 진영 / 도입사·개발자 / 투자 업계
- 기준마다 긍정·부정 질의 한 쌍과 Tavily `topic`(general/news)을 둔다. 1차 검색 예산은
  기준 3 × 기술 2 × 관점 2 × 2 = 24회다.
- `TECH_ALIASES`의 첫 항목은 1차 검색명이고, 나머지는 결과가 빈약할 때만 쓰는 보강 검색 별칭이다.
- 기준일 `END_DATE = "2026-09-21"`.
- `RetrievalRecord`: 공용 Evidence 대신 기준·기술·질의 방향·score·별칭 여부·재인용 여부를 내부에서 기록한다.
- 웹 근거 ID는 `pipeline.search_web`과 같은 `web_` + sha256 16자리 규칙을 쓴다.
- 인용 표기: `[시장 자료, {도메인} 발행 {YYYY-MM-DD}]`, 이해관계자는 `[이해관계자 자료, ...]`.
  발행일이 없으면 `발행일 미확인`이다.
- 발행일은 ISO·RFC 2822 형식을 YYYY-MM-DD로 정규화한다. 해석할 수 없으면 None이다.
  발행일 미확인 자료는 기준일 필터에서 제외하지 않는다.

### 검증

- `uv run python -m unittest discover -s tests -t .` 10건 통과. pytest를 의존성에 추가하지 않도록 unittest를 사용했다.
- `uv run check_graph.py` PASS

## 3단계: Tavily 검색 래퍼 (`service/agent/tavily/client.py`)

### 동작

| 상황 | 처리 |
|---|---|
| 한쪽 질의 실패 (HTTP 오류, 타임아웃, 잘못된 응답) | 성공한 쪽만 사용, limitations에 `부정 근거 수집 실패 (HTTP 429)`와 `한쪽 수집 실패로 한쪽 방향 근거만 사용` 기록 |
| 양쪽 질의 실패 | limitations에 `긍정·부정 질의 모두 실패` 기록 |
| 보강 검색까지 마친 뒤 근거 0건 | limitations에 `{기술} / {기준}: 웹 근거 없음, 판단 유보` 기록 |
| 중복 URL | scheme·`www.`·끝 슬래시·fragment를 정규화해 dedup, 먼저 실행한 긍정 질의 방향 유지 |
| 발행일 없음 | `published_at=None`, 인용에 `발행일 미확인` |
| 기준일 이후 자료 | Tavily `end_date`로 1차 차단, 후처리 필터로 한 번 더 제외하고 건수만 기록 |
| API 키 누락 | 판단 유보로 숨기지 않고 RuntimeError로 올림 |

- 판단 유보는 질의 실패 여부가 아니라 **보강 검색까지 마친 뒤 근거가 없을 때** 기록한다. 1차 질의가 모두 실패해도
  별칭 검색으로 근거를 찾으면 판단 유보가 아니다. 반대로 질의는 성공했지만 결과가 0건이어도 판단 유보다.
- 오류 기록에는 예외 메시지 대신 HTTP 상태 코드나 예외 타입명만 남긴다. 요청 본문·인증값이 섞이지 않게 하기 위함이다.
- 빈약함 기준: 결과 0건 또는 score 상위 3개 평균 < 0.5. 이때만 별칭으로 보강 검색한다. 기본 `max_fallbacks=1`이므로
  검색 상한은 기준별 4회, 전체 48회(1차 24회 + 보강 24회)다. 임계값 0.5는 임시값이며 6단계에서 실제 score 분포를 보고 조정한다.

### Tavily API 확인 (공식 문서, 2026-09-22 확인)

- `end_date`는 `YYYY-MM-DD` 형식으로 지원된다.
- `published_date`는 `topic="news"`일 때만 자동으로 포함된다. `general`에서는 `include_published_date: true`가
  필요하므로 항상 보낸다. 이 값은 Tavily가 추정한 발행일 또는 최종 수정일이며 null일 수 있다.

### 테스트

- `tests/fixtures/tavily_responses/`: Tavily 응답 형식을 본떠 **직접 작성한 합성 fixture**다. 실제 녹화 응답이 아니며
  6단계에서 녹화 응답으로 교체·추가한다.
- `tests/test_tavily_client.py` 18건: 파싱·필터, 한쪽/양쪽 실패, 타임아웃, 중복 URL, 오류 메시지 비노출,
  빈약 판정, 보강 검색 여부와 상한, 실패 후 별칭 복구, API 키 누락, 요청 payload, 비JSON 응답
- 전체 unittest 28건, `check_graph.py`, `check_domain.py` PASS

## 4단계: 시장성 에이전트

### 구조

- `service/agent/tavily/evaluation.py`: 시장성·이해관계자 공통 흐름 (`make_evaluation_node`, `PerspectiveSpec`)
- `service/agent/node/market.py`: 시장성 프롬프트와 전용 검증 규칙, `make_market_node(analyst, *, rules, normalize_result, error_result)`
- 채은의 도메인 노드처럼 `pipeline`을 import하지 않고 필요한 함수를 인자로 받는다. 7단계에서 `pipeline.make_nodes`에 연결한다.

### 결정

| 항목 | 결정 | 이유 |
|---|---|---|
| 평가 기준 | `state["evaluation_criteria"]` 대신 설계서 기준 `query_templates.CRITERIA` 사용 | 레포 목업 기준과 다름. 팀 공유 5번 |
| 재인용 후보 | `technical_result.evidence` 중 excerpt에 `REUSE_KEYWORDS`(cost, power, deployment, cloud 등)가 있는 것만 | 논문 성능 수치를 시장성 근거로 반복하지 않게 함. 규칙이 코드에 남아 재현 가능 |
| 재인용 근거의 기술 연결 | ID 접두어(`sw_`/`hw_`)와 기술의 `approach`로 연결 | SW 논문으로 HW 시장성을 주장하지 못하게 함 |
| LLM 입력 | `reused_evidence`와 `web_evidence`를 분리해 전달. 질의 방향(긍정/부정)은 전달하지 않음 | stance를 질의 방향이 아니라 원문 내용으로 판단하게 함 |
| 잘못된 Finding | 해당 Finding만 제외하고 `검증 실패로 제외: ...`를 limitations에 기록 | 도메인 노드와 같은 방식. 하나 때문에 전체가 error가 되지 않음 |
| 공통 검증 | `normalize_result`를 Finding 한 건 단위로 호출해 기술 ID·기준·근거 ID·인용 일치를 재사용 | pipeline 규칙과 어긋나지 않게 함 |
| 시장성 전용 검증 | `claim_type`·`scope`·`stance` 필수, 상용화·채택은 `stage` 필수 | 설계서 3.4 판단 규칙을 코드로 강제 |
| status | 기술 2 × 기준 3 = 6칸이 모두 검증된 Finding으로 채워지고 LLM도 complete면 complete, 아니면 partial | 설계서 4.4: 근거 부족은 partial |
| error | 모든 Tavily 질의 실패(LLM 호출 안 함), LLM 호출 실패, 설정 오류 | 설계서 4.4: 모델·도구 오류는 error |
| 요약·한계의 알 수 없는 인용 | 제거 | normalize_result가 결과 전체를 error로 만드는 것을 방지 |
| 재작업 피드백 | `quality_feedback` 중 "시장"/"market"이 들어간 항목만 `revision_feedback`으로 전달 | 다른 관점 피드백이 섞이지 않게 함 |

- 웹 인용은 Finding claim 안에서 pipeline 규칙대로 `[web_...]` ID를 쓴다. `[시장 자료, 도메인 발행 날짜]` 표기(`web_citation`)는
  보고서 조립 단계에서 쓸 수 있도록 제공하며, 적용 여부는 보고서 담당과 맞춰야 한다.
- 재작업 시 "검색 방향 조정"은 현재 LLM 입력에 피드백을 전달하는 데까지만 구현했다. 피드백으로 검색 질의를 바꾸는 것은 아직 없다.

### 테스트

- `tests/test_market_evaluation.py` 13건 (가짜 LLM·가짜 Tavily, API 호출 없음): 6칸 complete, 설계서 기준 사용,
  재인용 필터·분리, 재인용 근거의 기술 연결, 일방적 근거 기록, 잘못된 Finding 5종 제외, 알 수 없는 인용 제거,
  결과 0건 partial·판단 유보 6칸, 한쪽 실패 흡수, 전체 실패 error(LLM 미호출·메시지 비노출), LLM 실패 error,
  피드백 필터, 결과 evidence가 인용한 근거만 포함
- 전체 unittest 41건, `check_graph.py`, `check_domain.py` PASS

## 5단계: 이해관계자 에이전트

- `service/agent/node/stakeholder.py`: 4단계 공통 흐름을 그대로 쓰고 프롬프트와 검증 규칙만 다르다.
- 프롬프트(설계서 3.4): 발언·행동 주체와 형식(공식 발표, 제품 문서, 기사, 개인 블로그·포럼)을 claim에 명시,
  경쟁 진영은 직접 발표와 평가자 해석 분리, 도입사·개발자는 공식 사례와 개인 의견 구분·소수 의견 일반화 금지,
  투자 업계는 기업 전체 투자와 특정 기술 투자 구분·전망을 실증 근거로 쓰지 않음, 관측 반응과 예상 이해관계 구별
- 검증: `claim_type`·`stance` 필수. `scope`는 시장성 전용이라 값이 와도 주장은 유지하고 필드만 비운다.
  시장성 전용 규칙(scope·stage 필수)은 적용하지 않는다.
- 재인용 키워드는 이해관계자용(`vendor`, `developer`, `open-source`, `vllm` 등)을 따로 쓴다.
- Finding에 발언 주체 필드는 없다. 설계서 4.1의 "발언 주체"는 claim 본문으로 표현한다. 필드가 필요하면 공용 스키마 변경이라 팀 논의가 필요하다.
- 테스트 `tests/test_stakeholder_evaluation.py` 5건. 전체 unittest 46건, `check_graph.py`, `check_domain.py` PASS

## 6단계: 실제 API 실측 (2026-09-22, CXL-PNM 시장성, gpt-4.1-mini)

실행: `uv run python scripts/tavily_live_check.py --technology hw_01 --perspective market`
(Tavily 응답 축약본은 `tests/fixtures/tavily_responses/recorded/`, 원본·LLM 출력은 `outputs/tavily_live/`)

| 회차 | Tavily | 결과 | 발견한 문제 | 조치 |
|---|---|---|---|---|
| 1 | 12회 | partial, finding 2 | "CXL-PNM" 1차 검색 score 낮음(최고 0.30), CXL 전체 시장 전망을 `scope=direct`·`claim_type=fact`로 표시, finding당 근거 19·13개 통째 연결 | 1차 검색명을 "CXL processing-near-memory"로 교체, scope·claim_type 라벨 교정 규칙, 근거 5개 상한, 본문 인용 필수 |
| 2 | 8회 | partial, finding 0 | LLM이 claim 본문에 인용을 넣지 않아 전부 제외 | 본문 인용 필수 규칙 철회. `pipeline.result_markdown`이 evidence_ids로 인용을 붙이므로 팀 컨벤션과도 맞음 |
| 3 | 8회 | partial, finding 1 | scope 교정은 동작. LLM이 16자리 해시 ID를 잘못 옮겨 적어(`web_793fbf3c...`→`web_793fbf3d...`) finding 2개 제외 | LLM 입력에서 근거 ID를 `E1`, `E2` 참조키로 바꾸고 응답 후 실제 ID로 복원 |
| 4 | 8회 | partial, finding 0 | 참조키가 짧아지자 기준당 finding 1개에 근거 10~13개를 통째 연결해 상한에 걸림. 요약에 인용 없는 수치 | **칸 단위 LLM 호출**로 전환(기술×기준마다 1회, 그 칸 근거만 전달). 요약은 그대로 두고 보고서 담당에 공유 |
| 5 | 8회, LLM 3회 | complete, finding 16 | 근거 개수는 해결. 그러나 상용화·채택에 "논문에서 21.9배 처리량" finding이 `stage=production`·`direct`로 들어감. 인용 근거는 무관한 USENIX 논문이었고 수치는 `request`(선정 문서)에서 가져옴 | 칸 호출 입력에서 `request` 제거, 학술 도메인 근거만으로 된 상용화·채택 finding 제외, 학술 근거의 stage 비움, "근거 title·excerpt에 있는 내용만" 지시 |
| 6 | 8회, LLM 3회 | complete, finding 12 | 모든 finding이 실제 웹 출처 기반. 논문 기반 상용화 finding 제외됨. CXL 전체 시장 수치는 모두 adjacent·forecast로 교정. MLA 관련 잡음 한계 사라짐 | 남은 과제는 아래 |

확인된 사실
- `include_published_date: true` 이후 모든 결과에 발행일이 온다.
- news 토픽 score는 general보다 낮다(최고 0.36). 임계값 0.5에서는 news 기준이 항상 보강 검색된다.
- 1차 검색명 교체로 같은 조건의 Tavily 호출이 12회에서 8회로 줄었다(`tests/test_recorded_replay.py`로 재현).
- 단일 기술로 실행하면 request 문서에 두 기술이 모두 있어서 LLM이 평가하지 않은 기술(MLA)에 대한 한계를 쓴다. 두 기술을 함께 실행하면 생기지 않는 테스트 범위의 부작용이다.

남은 과제 (6회차 기준)
- 요약(summary)에 인용 없는 수치가 들어간다(예: "2028년까지 약 160억 달러"). 결정: 이 에이전트에서는 손대지 않고 보고서 담당(강용현)에 공유.
- 출처 신뢰도 편차: 시장조사 기관·IR 외에 facebook.com, linkedin.com, 개인 블로그·substack이 근거로 쓰인다. 출처 등급 규칙은 아직 없다.
- 시장 규모·성장성은 부정 방향 근거를 찾지 못해 "일방적 근거"로 기록된다.
- 이해관계자 관점과 sw_01(MLA)은 아직 실측하지 않았다.

추가한 규칙 (모두 unittest로 고정, 전체 66건 통과)
- 칸 단위 LLM 호출: 기술×기준마다 1회, 그 칸의 재인용·웹 근거만 전달. 요청한 칸과 다른 기술·기준의 finding은 제외.
  칸 하나의 LLM 실패는 그 칸만 판단 유보, 모든 칸 실패는 error. 근거 없는 칸은 LLM을 호출하지 않음
- 칸 호출 입력에서 `request`(선정 문서 전문) 제외
- 학술 도메인(`ACADEMIC_DOMAINS`, `.edu`, 논문 재인용)만 인용한 상용화·채택 finding 제외, 학술 근거의 stage는 null로 교정
- finding당 근거 최대 5개, 초과 시 제외. 본문 인용이 있으면 evidence_ids를 인용한 근거로 좁힘
- `scope=direct`인데 인용 근거에 기술 고유어(`TECH_TERMS`)가 없으면 adjacent로 교정하고 기록
- `claim_type=fact`인데 기준일 이후 연도나 전망 표현(`FORECAST_TERMS`)이 있으면 forecast로 교정하고 기록
- 수집하지 않은 근거 ID로 제외될 때 해당 ID를 limitations에 명시
- LLM 입력 근거 ID는 참조키(`E1`…)로 전달하고 응답 후 복원

## 설계서 외 자체 안전장치 (구현 후 README에 "확증편향 방지 조치"로 기록)

- 기준별 stance 분포 확인: positive/negative 중 한쪽만 있으면 limitations에 "일방적 근거"로 기록 (4단계 구현, `one_sided_cells`)
- 긍정·부정 질의 쌍 검색 (2·3단계 구현)
- 재작업 진입 시 quality_feedback 중 자기 관점 항목만 반영 (4단계는 LLM 입력 전달까지, 검색 질의 조정은 미구현)
