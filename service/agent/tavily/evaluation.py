"""시장성·이해관계자 평가의 공통 실행 흐름.

1. technical_result.evidence에서 시장·운영 성격 근거를 재인용 후보로 고른다.
2. 기술 × 기준마다 긍정/부정 질의 쌍으로 웹 근거를 수집한다.
3. 재인용 근거와 웹 근거를 구분해 LLM에 전달하고 AnalysisDraft를 받는다.
4. 검증에 걸린 Finding만 제외하고 limitations에 사유를 남긴다.
5. 기술 × 기준 칸의 충족 여부로 status를 판정한다.
6. 칸별 stance가 한쪽 방향만 있으면 일방적 근거로 기록한다.

관점별 차이(프롬프트, 추가 검증 규칙)는 PerspectiveSpec으로만 주입한다.
pipeline을 import하지 않도록 rules·normalize_result·error_result는 인자로 받는다.
"""

import json
import re
from collections.abc import Callable
from dataclasses import dataclass

from service.agent.tavily.client import SearchFn, search_criterion, tavily_search
from service.agent.tavily.query_templates import CRITERIA, END_DATE, REUSE_KEYWORDS, TECH_ALIASES, Perspective
from state import AnalysisDraft, DraftFinding, Evidence, GraphState

# pipeline.citation_ids와 같은 인용 ID 패턴
CITATION = re.compile(r"\[((?:sw|hw)_p\d+_c\d+|web_[a-f0-9]+)\]")
FEEDBACK_KEYWORDS: dict[Perspective, tuple[str, ...]] = {
    "market": ("market", "시장"),
    "stakeholder": ("stakeholder", "이해관계자"),
}
STANCE_LABELS = {"positive": "긍정", "negative": "부정"}


@dataclass(frozen=True)
class PerspectiveSpec:
    perspective: Perspective
    field: str  # market_result / stakeholder_result
    prompt: str
    # 관점 고유 규칙. 문제가 있으면 사유 문자열, 없으면 None을 반환한다.
    check_finding: Callable[[DraftFinding], str | None]


def reusable_evidence(state: GraphState, perspective: Perspective) -> list[Evidence]:
    keywords = REUSE_KEYWORDS[perspective]
    evidence = (state.get("technical_result") or {}).get("evidence", [])
    return [e for e in evidence if any(k in e["excerpt"].lower() for k in keywords)]


def relevant_feedback(state: GraphState, perspective: Perspective) -> list[str]:
    keywords = FEEDBACK_KEYWORDS[perspective]
    return [item for item in state.get("quality_feedback", []) if any(k in item.lower() for k in keywords)]


def strip_unknown_citations(text: str, known: set[str]) -> str:
    # 요약·한계 문장의 알 수 없는 인용 하나 때문에 결과 전체가 error가 되지 않게 한다.
    return re.sub(r" {2,}", " ", CITATION.sub(lambda m: m[0] if m[1] in known else "", text)).strip()


def one_sided_cells(findings: list[DraftFinding]) -> list[str]:
    stances: dict[tuple[str, str], set[str]] = {}
    for finding in findings:
        for technology_id in finding.technology_ids:
            stances.setdefault((technology_id, finding.criterion), set()).add(finding.stance)
    messages = []
    for (technology_id, criterion), values in sorted(stances.items()):
        directional = values & {"positive", "negative"}
        if len(directional) == 1 and "mixed" not in values:
            only = STANCE_LABELS[directional.pop()]
            messages.append(f"일방적 근거: {technology_id} / {criterion}에서 {only} 방향 근거만 확인됨, 반대 방향 근거 미확보")
    return messages


def make_evaluation_node(spec: PerspectiveSpec, analyst, *, rules: str, normalize_result, error_result,
                         search: SearchFn = tavily_search, end_date: str = END_DATE, max_fallbacks: int = 1):
    perspective, field = spec.perspective, spec.field

    def run(state: GraphState):
        try:
            criteria = CRITERIA[perspective]
            technologies = state["technologies"]
            unknown = [t["id"] for t in technologies if t["id"] not in TECH_ALIASES]
            if not technologies or unknown:
                raise ValueError(f"검색명이 등록되지 않은 기술 ID: {', '.join(unknown) or '없음'}")

            sources: dict[str, Evidence] = {}
            evidence_by_technology = {t["id"]: set() for t in technologies}
            reused = reusable_evidence(state, perspective)
            for evidence in reused:
                sources[evidence["id"]] = evidence
                for technology in technologies:
                    if evidence["id"].startswith(technology["approach"].lower() + "_"):
                        evidence_by_technology[technology["id"]].add(evidence["id"])

            search_limitations, search_results = [], []
            attempted = failed = 0
            for technology in technologies:
                for criterion in criteria:
                    found = search_criterion(perspective, criterion, technology["id"], end_date=end_date,
                                             max_fallbacks=max_fallbacks, search=search)
                    attempted += found.attempted
                    failed += found.failed_count
                    search_limitations.extend(found.limitations)
                    for evidence in found.evidence:
                        sources[evidence["id"]] = evidence
                        evidence_by_technology[technology["id"]].add(evidence["id"])
                    search_results.append({"technology_id": technology["id"], "criterion": criterion,
                                           "evidence_ids": [e["id"] for e in found.evidence]})
            if attempted and failed == attempted:
                # 설계서 4.4: 모델·도구 오류는 error. 근거 부족(partial)과 구분한다.
                raise RuntimeError("Tavily 질의가 모두 실패했습니다: " + "; ".join(search_limitations[:3]))

            web_ids = {key for row in search_results for key in row["evidence_ids"]}
            context = {
                "request": state["request"], "target_domain": state["target_domain"],
                "technologies": technologies, "evaluation_criteria": criteria, "evaluation_base_date": end_date,
                "reused_evidence": [sources[e["id"]] for e in reused],
                "web_evidence": [sources[key] for key in sorted(web_ids)],
                "search_results": search_results,
                "revision_feedback": relevant_feedback(state, perspective),
            }
            draft = analyst.invoke([("system", rules + spec.prompt),
                                    ("human", json.dumps(context, ensure_ascii=False))])

            valid, dropped = [], []
            for finding in draft.findings:
                reason = finding_problem(finding, spec, sources, state, criteria, evidence_by_technology,
                                         normalize_result)
                if reason:
                    dropped.append(f"검증 실패로 제외: {', '.join(finding.technology_ids)} / {finding.criterion}: {reason}")
                else:
                    valid.append(finding)

            covered = {(tid, f.criterion) for f in valid for tid in f.technology_ids}
            cells = {(t["id"], c) for t in technologies for c in criteria}
            status = "complete" if cells <= covered and draft.status == "complete" else "partial"
            known = set(sources)
            limitations = [strip_unknown_citations(item, known) for item in draft.limitations]
            limitations = list(dict.fromkeys(search_limitations + limitations + dropped + one_sided_cells(valid)))
            final = AnalysisDraft(status=status, summary=strip_unknown_citations(draft.summary, known),
                                  findings=valid, limitations=limitations, next_queries=[])
            result, _ = normalize_result(final, sources, state, criteria)
            return {field: result}
        except Exception as exc:
            return {field: error_result(field, exc)}

    return run


def finding_problem(finding: DraftFinding, spec: PerspectiveSpec, sources, state, criteria,
                    evidence_by_technology, normalize_result) -> str | None:
    # 기술 ID·기준·근거 ID·인용 일치는 pipeline 공통 규칙을 한 건 단위로 재사용한다.
    try:
        normalize_result(AnalysisDraft(status="complete", summary="", findings=[finding], limitations=[],
                                       next_queries=[]), sources, state, criteria)
    except ValueError as exc:
        return str(exc)
    unsupported = [tid for tid in finding.technology_ids if not set(finding.evidence_ids) & evidence_by_technology[tid]]
    if unsupported:
        return f"{', '.join(unsupported)}의 근거가 연결되지 않음"
    if finding.claim_type is None:
        return "claim_type 누락"
    return spec.check_finding(finding)
