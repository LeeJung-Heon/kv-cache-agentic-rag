# 시장성·이해관계자 평가 에이전트 (Tavily)

설계서 3.2·3.4절의 시장성·이해관계자 관점을 Tavily 웹 검색으로 평가합니다. 결정 과정과 실측 기록은
[작업 기록](MARKET_STAKEHOLDER_WORKLOG.md)에 있습니다.

## 사용법

| 항목 | 내용 |
|---|---|
| 그래프 노드 | `service.agent.node.market.market_node(state)`, `service.agent.node.stakeholder.stakeholder_node(state)` |
| 읽는 State | `technologies`, `target_domain`, `technical_result`(선택, 재인용), `quality_feedback`(선택, 재작업) |
| 갱신하는 State | `market_result`, `stakeholder_result` (각 노드는 자기 필드만 갱신하므로 병렬 실행에 reducer가 필요 없습니다) |
| 설정 | `config.Settings`의 `OPENAI_API_KEY`, `OPENAI_MODEL`, `OPENAI_BASE_URL`, `TAVILY_API_KEY` |
| 테스트용 생성 | `make_market_node(analyst, search=...)`, `make_stakeholder_node(analyst, search=...)` |

`request`와 `evaluation_criteria`는 읽지 않습니다. 평가 기준은 설계서 3.4절 기준을 `service/agent/tavily/query_templates.py`의
`CRITERIA`에 두고 사용합니다. `request`의 선정 문서는 근거가 아니므로 LLM 입력에 넣지 않습니다(아래 근거 규칙 참조).

```sh
uv run pytest -q tests/test_tavily_client.py tests/test_market_evaluation.py tests/test_stakeholder_evaluation.py \
  tests/test_recorded_replay.py tests/test_query_templates.py tests/test_evidence_schema.py   # API 호출 없음
uv run python scripts/tavily_live_check.py --technology hw_01 --perspective market          # 실제 API, 비용 발생
```

실측 스크립트는 기술 1개 × 관점 1개를 실행합니다(Tavily 6~10회, LLM 3회). 원본 응답과 LLM 원본 출력은 `outputs/tavily_live/`
(Git 제외)에 저장합니다. `--record`를 주면 content를 200자로 자른 응답 축약본을 `tests/tavily_fixtures/recorded/`에 저장합니다.
레포가 public이므로 제3자 웹 발췌 원문은 커밋하지 않습니다.

## 파일

| 파일 | 역할 |
|---|---|
| `service/agent/tavily/query_templates.py` | 평가 기준, 긍정·부정 질의 템플릿, 기술 별칭·고유어, 재인용 키워드, 학술·저신뢰 도메인 목록, 기준일 |
| `service/agent/tavily/client.py` | Tavily 호출, 응답 파싱·기준일 필터, URL 중복 제거, 별칭·방향별 보강 검색 |
| `service/agent/tavily/evidence_schema.py` | 웹 근거 ID, 발행일 정규화, 내부 수집 기록(`RetrievalRecord`), 웹 인용 표기 |
| `service/agent/tavily/evaluation.py` | 두 관점 공통 흐름: 재인용, 칸 단위 LLM 호출, 검증·교정, status, 요약 |
| `service/agent/node/market.py`, `stakeholder.py` | 관점별 프롬프트와 검증·교정 규칙, 노드 진입점 |
| `scripts/tavily_live_check.py` | 실측 실행과 응답 녹화 |
| `tests/tavily_fixtures/` | 합성 응답 fixture와 실측 응답 축약본(`recorded/`) |

## 처리 흐름

1. `technical_result.evidence` 중 excerpt에 시장·운영 키워드(cost, power, deployment, cloud 등)가 있는 청크를 재인용 후보로 고릅니다.
   `sw_`/`hw_` 청크는 해당 기술에만, `common_` 공통 운영 문서는 두 기술 모두에 연결합니다.
2. 기술 × 기준(2 × 3 = 6칸)마다 긍정·부정 질의 쌍으로 검색합니다. 1차 검색 12회, 보강 검색 포함 최대 칸당 5회입니다.
3. 칸마다 LLM을 한 번 호출하고 그 칸의 재인용 근거와 웹 근거만 전달합니다. 근거 ID는 `E1`, `E2` 참조키로 바꿔 전달하고 응답 후 복원합니다.
4. Finding을 검증합니다. 걸린 Finding만 제외하고 사유를 limitations에 남기며, 라벨이 근거와 맞지 않으면 교정하고 기록합니다.
5. 6칸이 모두 검증된 Finding으로 채워지면 `complete`, 아니면 `partial`입니다.
6. 요약(`summary`)은 코드가 검증 결과 현황으로 만듭니다.

## Finding 확장 필드

`service.schema.state`의 `Finding`·`DraftFinding`에 추가된 선택 필드입니다. `None`은 해당 관점에 적용되지 않는다는 뜻입니다.

| 필드 | 값 | 의미 |
|---|---|---|
| `claim_type` | `fact` / `opinion` / `forecast` | 원문 진술의 성격. `is_inference`(에이전트 해석 여부)와 독립 |
| `scope` | `direct` / `adjacent` | 대상 기술 자체 / 상위 기술·연관 시장(CXL 전체, DeepSeek 기업·모델 전체 등) |
| `stage` | `announced` / `pilot` / `production` | 계획 발표 / 실증 / 실제 운영 |
| `stance` | `positive` / `negative` / `mixed` / `unknown` | 대상 기술에 대한 입장 |

## status와 오류 정책

설계서 4.4절 기준입니다. 레포 목업의 "부분 실패도 error" 정책보다 관대합니다.

| 상황 | 처리 |
|---|---|
| 질의 한쪽 실패(HTTP 오류, 타임아웃, 잘못된 응답) | 성공한 쪽만 사용하고 limitations에 기록 |
| 한 칸의 웹·재인용 근거가 모두 없음 | LLM을 호출하지 않고 "판단 유보"로 기록, `partial` |
| 한 칸의 LLM 호출 실패 | 그 칸만 판단 유보, `partial` |
| 모든 Tavily 질의 실패, 모든 칸의 LLM 실패, 설정 오류 | `error` (모델·도구 오류) |

오류 기록에는 예외 메시지 대신 HTTP 상태 코드나 예외 타입명만 남겨 요청 본문·인증값이 섞이지 않게 합니다.

## 확증편향 방지 조치

설계서 5.3 체크리스트("시장성·이해관계자 근거가 없으면 판단을 유보한다")와 6.5 "확증편향을 줄이기 위해 적용한 방법"에 대응합니다.
설계서에 없는 자체 안전장치는 **(자체)**로 표시합니다.

| 조치 | 방법 | 위치 |
|---|---|---|
| 긍정·부정 질의 쌍 | 모든 기준을 긍정 질의와 부정 질의로 함께 검색합니다 | `query_templates.QUERY_TEMPLATES` |
| 반대 방향 보강 검색 **(자체)** | 한쪽 방향 질의 결과만 빈약하면 그 방향만 별칭으로 1회 더 검색합니다 | `client.search_criterion` |
| 일방적 근거 기록 **(자체)** | 한 칸의 Finding stance가 긍정 또는 부정 한쪽뿐이면 "일방적 근거"로 limitations에 기록합니다 | `evaluation.one_sided_cells` |
| 질의 방향 비공개 | LLM에 어느 질의로 수집했는지 알리지 않아, stance를 질의 방향이 아니라 원문 내용으로 판단하게 합니다 | `evaluation.cell_context` |
| 판단 유보 | 근거가 없는 칸은 지어내지 않고 판단 유보로 기록하며 `complete`로 처리하지 않습니다 | `evaluation.make_evaluation_node` |
| 직접·연관 구분 | claim과 인용 근거 양쪽에 기술 고유어가 없으면 `scope=adjacent`로 교정합니다. 연관 시장·상위 기술 반응을 대상 기술의 근거로 확대하지 않습니다 | `evaluation.missing_tech_terms` |
| 사실·전망 구분 | 기준일 이후 연도나 전망 표현이 있는 `fact`는 `forecast`로 교정합니다 | `evaluation.correct_forecast` |
| 공식·개인 의견 구분 | 소셜미디어·개인 블로그만 인용한 근거는 시장성 칸 충족에 세지 않고, 이해관계자에서는 `opinion`으로 교정합니다 | `LOW_TRUST_DOMAINS`, `is_low_trust` |
| 재작업 피드백 필터 **(자체)** | `quality_feedback` 중 자기 관점 항목만 LLM에 전달합니다 | `evaluation.relevant_feedback` |

## 근거 규칙

실측(2026-09-22, gpt-4.1-mini)에서 발견한 문제를 코드로 막은 규칙입니다.

- Finding당 근거는 최대 5개입니다. 본문 인용이 있으면 인용한 근거로 먼저 좁히고, 그래도 넘으면 제외합니다.
- 학술 도메인(arxiv, usenix, acm, ieee, `.edu` 등)이나 논문 재인용만 인용한 경우:
  시장성 상용화·채택과 이해관계자 도입사·개발자·투자 업계에서는 제외하고, 그 밖에서는 `stage`를 비웁니다.
- 요청한 칸과 다른 기술·기준의 Finding, 수집하지 않은 근거 ID를 참조한 Finding은 제외합니다.
- `request`(선정 문서)는 LLM 입력에 넣지 않습니다. LLM이 선정 문서의 논문 수치를 무관한 웹 근거에 붙인 사례가 있었습니다.
- `summary`는 LLM이 아니라 코드가 만듭니다. 칸 충족 현황, 근거 미확보 칸, Finding 수(사실·의견·전망), 직접·연관 근거 수,
  일방적 칸·라벨 교정·저신뢰 건수만 쓰고 사실 주장이나 수치는 넣지 않습니다.

## 알려진 한계

- 기술 설명·성능 수치가 이해관계자 반응이나 생태계 근거로 쓰이는 경우가 남아 있습니다. 대부분 `adjacent`로 표시되지만
  코드로 완전히 판별할 수 없습니다.
- MLA는 알고리즘이라 시장 규모·투자 업계 기준의 직접 근거가 드물고, 연관 근거(DeepSeek 기업, LLM 추론 시장)로 채워집니다.
- 빈약함 임계값(상위 3개 score 평균 0.5), 저신뢰·학술 도메인 목록, 재인용 키워드는 실측 몇 회로 정한 값입니다.
- claim의 수치가 인용 근거에 있는지는 검사하지 않습니다. 번역 중 단위 변환("$3.25B" → "32억 5,000만 달러")으로 오탐이 납니다.
- 재작업 시 피드백은 LLM 입력으로만 전달하며 검색 질의 자체는 바꾸지 않습니다.
- 웹 스니펫은 원문 전체 검증이 아니며, Tavily의 발행일은 추정 발행일 또는 최종 수정일입니다.

## 종합·보고서 담당 참고

- `summary`에는 사실 주장이 없습니다. 보고서의 시장성·이해관계자 본문은 `findings`와 `limitations`로 구성해야 합니다.
- `scope=adjacent`인 Finding은 대상 기술 자체의 근거가 아닙니다. 대상 기술의 판단으로 옮길 때 연관 근거임을 밝혀야 합니다.
- "일방적 근거", "판단 유보", "저신뢰 출처만 인용", "라벨 교정" limitations는 품질 점검(`quality_feedback`)의 입력으로 쓸 수 있습니다.
- 웹 근거 인용 표기 `[시장 자료, {도메인} 발행 {날짜}]`는 `evidence_schema.web_citation`으로 만들 수 있습니다.
