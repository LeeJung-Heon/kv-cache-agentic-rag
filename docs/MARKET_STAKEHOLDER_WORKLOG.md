# 시장성·이해관계자 평가 에이전트 작업 기록

담당: 박현준 — 시장성·이해관계자 평가 에이전트 (Tavily 기반). 담당 State 필드는 `market_result`, `stakeholder_result`.

| 팀원 | 역할 | 이 에이전트와 맞물리는 부분 |
|---|---|---|
| 박지원 | 전처리 | 없음 |
| 문진영 | 임베딩 | 없음 |
| 이중헌 | 기술 agent | `technical_result.evidence` 재인용 (스키마 확인 필요) |
| 왕채은 | 도메인 agent | LLM 출력에 확장 필드 추가됨 (코드 수정 불필요) |
| 강용현 | 종합 agent | 확장 필드명·값, 오류 정책 차이, 종합도 확장 필드 출력 |

설계 기준: RAG 기반 SW·HW 기술 평가 에이전트 설계서 (평가 기준일 2026-09-21)

## 작업 순서

| 순서 | 작업 | 브랜치 | 상태 |
|---|---|---|---|
| 1 | Finding 확장 필드 추가 | `feat/finding-extension` | 완료 (팀 공유 필요) |
| 2 | `evidence_schema.py`, `query_templates.py` | `feat/tavily-agent` | 완료 |
| 3 | `tavily_client.py` + 로직 테스트 | `feat/tavily-agent` | 다음 작업 |
| 4 | 시장성 에이전트 | 미정 | 대기 (technical_result 실제 스키마 확인 필요) |
| 5 | 이해관계자 에이전트 | 미정 | 대기 |
| 6 | 실제 Tavily API 통합 테스트 (1~2회) | 미정 | 대기 (API 키 필요) |
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

## 설계서 외 자체 안전장치 (구현 후 README에 "확증편향 방지 조치"로 기록)

- 기준별 stance 분포 확인: positive/negative 중 한쪽만 있으면 limitations에 "일방적 근거"로 기록
- 재작업 진입 시 quality_feedback 중 자기 관점 항목만 반영해 검색 방향 조정
