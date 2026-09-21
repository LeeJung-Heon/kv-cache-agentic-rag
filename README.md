# KV cache 다관점 평가

DeepSeek-V2의 MLA(SW)와 CXL-PNM(HW)을 두 논문 및 공개 웹 자료로 비교하는 LangGraph 파이프라인입니다. 특정 기술의 우열이나 추천 대신 관점별 평가 차이와 불확실성을 정리합니다.

```text
Input → 기술 조사 (논문 RAG + TRL)
           ├─ 시장 평가 (Tavily) ───────┐
           ├─ 이해관계자 평가 (Tavily) ─┼→ 평가 종합 → 보고서 → Markdown / PDF
           └─ 도메인 평가 (논문 RAG) ──┘
```

세 평가가 모두 끝난 뒤 종합 노드를 한 번 실행합니다. 별도의 검증 노드나 재평가 분기는 없습니다. 각 분석에는 양쪽 기술의 1차 검색 근거를 먼저 제공하고, 에이전트가 필요할 때 추가 검색 도구를 호출합니다. 도구 호출 루프는 LangGraph recursion limit으로 제한합니다.

## 실행

```sh
git clone https://github.com/chaeeunwang/kv-cache-agentic-rag.git
cd kv-cache-agentic-rag
uv sync --locked
cp .env.example .env
# .env의 OPENAI_API_KEY와 TAVILY_API_KEY 입력
uv run pipeline.py --domain '데이터센터의 장문맥 LLM 추론'
```

기존 설정 파일을 명시해 사용할 수도 있습니다. 키는 복사하거나 결과물에 저장하지 않습니다.

```sh
uv run pipeline.py --env-file /path/to/your/.env
```

지정한 `.env` 값이 프로세스 환경변수보다 우선합니다. 기본 생성 모델은 `gpt-4.1-mini`이며 `OPENAI_MODEL`로 변경합니다. OpenAI 호환 엔드포인트는 `OPENAI_BASE_URL`로 지정합니다. 실제 실행에는 LLM 및 Tavily API 호출 비용이 발생할 수 있습니다.

도메인은 아직 사용자 확정값이 아니므로 실행 가능한 기본 예시로 데이터센터 장문맥 추론을 사용합니다. `--domain`으로 변경하고, 실제 기술 선정 이유는 `--selection-reason`으로 전달합니다. 제공하지 않은 사람의 선정 이유를 생성하지 않도록 프롬프트에 명시했습니다.

## 파일과 State

| 파일 | 역할 |
|---|---|
| `pipeline.py` | State, 6개 역할, 검색 도구, 병렬 그래프, CLI |
| `rag.py` | PDF 로딩, E5 토큰 청킹, 임베딩 캐시, cosine 검색 |
| `report.py` | Markdown 원본 저장, 한국어 PDF 생성 |
| `check_graph.py` | 외부 API 없이 병렬 합류 구조만 확인 |
| `docs/` | 사용자가 선정한 SW/HW 원본 PDF |
| `outputs/<실행시각>/` | 완료 단계의 State, 보고서, 실패 시 오류 종류 |

State는 `input`, `technical_analysis`, `market_evaluation`, `stakeholder_evaluation`, `domain_evaluation`, `synthesis`, `final_report`로 구성합니다. 병렬 노드는 각자 자신의 필드만 갱신합니다. 각 평가 결과에는 `text`, 실제 인용된 `sources`, 도구 호출 내역 `searches`가 들어갑니다.

## RAG와 출처

- 사용자 선택 모델: `intfloat/multilingual-e5-large-instruct`. 다른 모델보다 우수하다는 비교 결과를 뜻하지 않습니다.
- [임베딩 모델 선정 설계](EMBEDDING_MODEL_SELECTION.md)는 BGE-M3를 잠정 선정합니다. 현재 실행 코드는 E5이며, 설계의 모델 및 청킹 설정은 아직 코드에 반영하지 않았습니다.
- 두 PDF 합계 200페이지 제한, 페이지별 400토큰 청크, 60토큰 overlap.
- 질의에 E5 instruction 적용, 문서는 instruction 없이 임베딩.
- L2 정규화한 벡터의 내적으로 cosine 검색, 기술별 상위 5개 반환.
- 공유 임베딩 모델의 추론 호출은 잠금으로 직렬화하여 MPS 동시 호출 충돌을 방지합니다. 평가 노드와 웹 검색은 병렬로 실행됩니다.
- 두 논문 규모에는 별도 벡터 DB 없이 NumPy를 사용합니다.
- `.cache`에 문서·청크·모델명 기반 캐시를 저장합니다. 원문 변경 시 새 캐시를 만듭니다.
- PDF 출처에는 파일·페이지·청크 ID·논문 URL, 웹 출처에는 제목·URL·접근일·제공되는 경우 발행일을 보존합니다.
- 보고서 인용 ID로 REFERENCE를 생성합니다. 수집하지 않은 ID나 인용이 전혀 없는 결과는 오류로 중단하며 재시도 분기는 없습니다. 이는 인용 문장의 사실성 검증을 대신하지 않습니다.
- 웹 검색 스니펫은 원문 전체 검증이 아니므로 확정 사실과 해석을 구별하도록 지시합니다.

## 출력

`report.md`를 원본으로 `report.pdf`를 만듭니다. 목차는 SUMMARY → 분석 배경 → 대상 기술 선정 → 기술 개요 → 관점별 평가 → 종합 평가 및 시사점 → 분석의 한계 → REFERENCE입니다.

매 실행마다 새 폴더를 만들어 기존 결과를 보존합니다. 도구/API 오류는 숨기고 계속 진행하지 않으며 실행을 중단합니다. 완료한 단계는 `state.json`에 남습니다. PDF 생성에 실패해도 Markdown은 남습니다. 자동 재개 기능은 없습니다.

PDF에는 한국어 TTF가 필요합니다. macOS의 Arial Unicode 또는 Linux의 NanumGothic을 자동 사용하며, 다른 환경에서는 `.env`의 `PDF_FONT`에 TTF 파일 경로를 지정합니다. 출력은 문단·제목·목록을 대상으로 하며 표·수식 렌더러는 포함하지 않습니다.

시장·이해관계자·도메인 본문은 원래 분석의 인용이 요약 중 사라지지 않도록 보고서에 그대로 삽입합니다. 따라서 에이전트가 기술 개요를 반복하면 보고서에도 중복이 남을 수 있습니다. 생성물은 제출 완료본이 아닌 검토용 초안입니다.

## 최소 확인

```sh
uv run pipeline.py --index-only  # API 없이 실제 PDF 임베딩 및 검색, 최초 모델 다운로드 가능
uv run check_graph.py           # 필요할 때 병렬 합류만 확인
```

개별 수정마다 전체 테스트나 모델 비교 평가를 실행할 필요는 없습니다. 생성된 보고서의 수치·인용 의미·공개정보 기반 TRL은 제출 전에 사람이 검토해야 합니다.
