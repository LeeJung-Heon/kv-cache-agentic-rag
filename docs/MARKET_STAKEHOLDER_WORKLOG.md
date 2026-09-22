# 시장성·이해관계자 평가 에이전트 작업 기록

담당: 시장성·이해관계자 평가 에이전트 (Tavily 기반)
설계 기준: RAG 기반 SW·HW 기술 평가 에이전트 설계서 (평가 기준일 2026-09-21)

## 작업 순서

| 순서 | 작업 | 브랜치 | 상태 |
|---|---|---|---|
| 1 | Finding 확장 필드 추가 | `feat/finding-extension` | 완료 (팀 공유 필요) |
| 2 | `evidence_schema.py`, `query_templates.py` | 미정 | 대기 |
| 3 | `tavily_client.py` + 로직 테스트 | 미정 | 대기 |
| 4 | 시장성 에이전트 | 미정 | 대기 (technical_result 실제 스키마 확인 필요) |
| 5 | 이해관계자 에이전트 | 미정 | 대기 |
| 6 | 실제 Tavily API 통합 테스트 (1~2회) | 미정 | 대기 (API 키 필요) |
| 7 | 팀 그래프에 노드 연결 | 미정 | 대기 (기술 조사·평가 종합 노드 완성 후) |

## 전제

- 설계서 2.2절은 시장성·이해관계자를 RAG X로, 2.3·3.5절은 문서 근거 재인용 O로 적고 있다. 이를
  "기술 조사 에이전트가 만든 `technical_result.evidence`에서 시장·운영 성격 근거를 재인용한다"로 해석한다.
  해석이 틀리면 프롬프트 설계만 수정한다.

## 1단계: Finding 확장 필드

### 결정

| 필드 | 값 | 적용 | 설계서 근거 |
|---|---|---|---|
| `claim_type` | `fact` / `opinion` / `forecast` | 모든 에이전트 필수 | 3.2 "사실·전망·간접 지표 분리", 5.2 |
| `scope` | `direct` / `adjacent` / `None` | 시장성 | 3.4 "직접 시장과 연관 시장 분리" |
| `stage` | `announced` / `pilot` / `production` / `None` | 상용화·채택 | 3.4 "계획 발표, 실증, 실제 운영 구분" |
| `stance` | `positive` / `negative` / `mixed` / `unknown` / `None` | 입장 판단 | 3.2 "긍정·부정·혼합·확인 불가" |

- `None`은 해당 관점에 적용되지 않는 항목이라는 뜻이다.
- `claim_type`은 원문 진술의 성격이고, `is_inference`는 에이전트가 해석했는지 여부다. 두 필드는 서로 독립이다.
  예: "Gartner가 X를 전망" → `claim_type=forecast`, `is_inference=false`.
- `DraftFinding`에서는 기본값 없이 nullable required 필드로 정의했다. OpenAI strict json_schema가 모든 필드를
  required로 요구하기 때문이다.
- `stance`는 Finding에만 둔다. 어느 질의(긍정/부정)로 수집했는지는 Evidence에 넣지 않고 에이전트 내부에서만
  관리한다. 같은 Evidence ID에 에이전트마다 다른 값이 들어가면 `collect_sources`가 EvidenceError를 낸다.
- 루트 `state.py`(현재 `pipeline.py`가 사용)와 `service/schema/state.py`(신규 구조)를 동기화했다.
- 모듈 위치는 팀 컨벤션(`service/agent/node/domain.py`)을 따른다. `agents/` 폴더는 만들지 않는다.

### 변경 파일

- `state.py`: Literal 타입 4개, `Finding`·`DraftFinding` 필드 추가
- `service/schema/state.py`: `Finding` 필드 추가
- `check_graph.py`: fixture에 새 필드 반영

### 검증 (API 호출 없음)

- `uv run check_graph.py` PASS
- OpenAI strict 스키마 변환 결과 4개 필드 모두 required, scope·stage·stance는 null 허용
- 잘못된 enum 값과 필드 누락은 pydantic 검증에서 거부
- `normalize_result` 결과의 공개 Finding에 4개 필드 포함

### 팀 공유 필요

1. `AnalysisDraft`를 모든 에이전트가 공유하므로, 기술 조사·도메인·종합 에이전트도 4개 필드를 출력해야 한다.
   해당 없는 필드는 `null`이다.
2. `feat/domain-rag` 머지 시 `check_domain.py` 31·36행 fixture에 새 필드를 추가해야 한다.
3. 루트 `state.py`와 `service/schema/state.py` 중 어느 쪽을 기준으로 삼을지 결정이 필요하다.
4. 오류 정책 차이: 레포는 부분 실패를 `error`로 처리하고 중단하지만, 이 에이전트는 설계서 기준으로
   한쪽 질의 실패를 `limitations`에 기록하고 계속 진행한다.

## 설계서 외 자체 안전장치 (구현 후 README에 "확증편향 방지 조치"로 기록)

- 기준별 stance 분포 확인: positive/negative 중 한쪽만 있으면 limitations에 "일방적 근거"로 기록
- 재작업 진입 시 quality_feedback 중 자기 관점 항목만 반영해 검색 방향 조정
