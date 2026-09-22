"""실제 Tavily·LLM으로 시장성/이해관계자 노드를 소규모 실행하고 Tavily 응답을 녹화한다.

비용이 발생한다. 기본값은 기술 1개 × 관점 1개(Tavily 6~12회, LLM 1회)다.
    uv run python scripts/tavily_live_check.py --technology hw_01 --perspective market [--record]
녹화 축약본: --record일 때만 tests/tavily_fixtures/recorded/에 저장 (content 200자로 절단, 커밋 대상)
             같은 질의를 반복 녹화하면 레포에 중복이 쌓이므로 새 질의·새 기술을 처음 실행할 때만 쓴다.
원본 응답·노드 결과: outputs/tavily_live/<시각>/  (Git 제외)
팀 레포가 public이므로 제3자 웹 발췌 원문은 커밋하지 않고 축약본만 남긴다. API 키는 응답에 포함되지 않는다.
"""

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config.config import Settings  # noqa: E402
from service.agent.node.market import make_market_node  # noqa: E402
from service.agent.node.stakeholder import make_stakeholder_node  # noqa: E402
from service.agent.tavily.client import tavily_search  # noqa: E402
from service.agent.tavily.evaluation import get_analyst  # noqa: E402

RECORDED = ROOT / "tests" / "tavily_fixtures" / "recorded"
# pipeline.select_technologies와 같은 사람 확정 기술 목록
TECHNOLOGIES = [
    {"id": "sw_01", "name": "DeepSeek-V2 MLA", "approach": "SW",
     "selection_reason": "KV 표현을 저차원 잠재 공간으로 바꾸는 구조적 접근으로, 공개 모델·논문의 성과와 모델 변경·서빙 호환성 부담을 함께 평가할 수 있다."},
    {"id": "hw_01", "name": "CXL-PNM", "approach": "HW",
     "selection_reason": "저장 공간 확장뿐 아니라 메모리 가까이로 연산 위치를 바꾸며, 장문맥·대규모 모델에서 성능·비용·전력 효과와 불리한 조건을 함께 비교할 수 있다."},
]
EXCERPT_LIMIT = 200


def trimmed(payload: dict) -> dict:
    keep = ("url", "title", "score", "published_date")
    return {"results": [{**{k: row.get(k) for k in keep}, "content": (row.get("content") or "")[:EXCERPT_LIMIT]}
                        for row in payload.get("results", [])]}


NODES = {"market": (make_market_node, "market_result"), "stakeholder": (make_stakeholder_node, "stakeholder_result")}


def recording_search(stamp: str, log: list, raw_dir: Path, record: bool):
    def search(query, **kwargs):
        payload = tavily_search(query, **kwargs)
        name = re.sub(r"[^a-z0-9]+", "_", query.lower()).strip("_")[:80]
        request = {"query": query, **kwargs}
        raw_dir.mkdir(parents=True, exist_ok=True)
        path = RECORDED / f"{stamp}_{name}.json"
        if record:
            RECORDED.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({"request": request, "response": trimmed(payload)}, ensure_ascii=False,
                                       indent=2), encoding="utf-8")
        (raw_dir / path.name).write_text(json.dumps({"request": request, "response": payload}, ensure_ascii=False,
                                                    indent=2), encoding="utf-8")
        rows = payload.get("results", [])
        log.append({"query": query, "topic": kwargs.get("topic"), "results": len(rows),
                    "scores": [round(r.get("score", 0), 3) for r in rows],
                    "published_date": sum(1 for r in rows if r.get("published_date")), "file": path.name})
        return payload
    return search


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--technology", choices=["sw_01", "hw_01"], default="hw_01")
    parser.add_argument("--perspective", choices=list(NODES), default="market")
    parser.add_argument("--record", action="store_true", help="응답 축약본을 fixture로 저장")
    args = parser.parse_args()
    settings = Settings()
    missing = [name for name, value in (("TAVILY_API_KEY", settings.tavily_api_key),
                                        ("OPENAI_API_KEY", settings.openai_api_key)) if not value]
    if missing:
        raise SystemExit("환경변수 없음: " + ", ".join(missing))

    state = {"request": "두 기술을 다관점으로 비교하라", "target_domain": "데이터센터·클라우드 LLM 서빙",
             "evaluation_criteria": {}, "technologies": [t for t in TECHNOLOGIES if t["id"] == args.technology],
             "quality_feedback": [], "revision_count": 0}
    structured = get_analyst()
    drafts: list = []

    class RecordingAnalyst:
        # 검증 전 LLM 원본 출력을 남겨 제외·교정 사유를 사람이 확인할 수 있게 한다.
        def invoke(self, messages):
            draft = structured.invoke(messages)
            drafts.append(draft.model_dump())
            return draft

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output = ROOT / "outputs" / "tavily_live" / stamp
    log: list = []
    make_node, field = NODES[args.perspective]
    node = make_node(RecordingAnalyst(), search=recording_search(stamp, log, output / "raw", args.record))
    result = node(state)[field]

    output.mkdir(parents=True, exist_ok=True)
    (output / "result.json").write_text(json.dumps({"search_log": log, "llm_draft": drafts, field: result},
                                                   ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Tavily 호출 {len(log)}회")
    for row in log:
        print(f"- [{row['topic']}] {row['query']} → {row['results']}건, score={row['scores']}, 발행일 {row['published_date']}건")
    print(f"\nstatus={result['status']}, findings={len(result['findings'])}, evidence={len(result['evidence'])}")
    print("limitations:", *[f"  - {item}" for item in result["limitations"]], sep="\n")
    print(f"\n결과: {output / 'result.json'}")


if __name__ == "__main__":
    main()
