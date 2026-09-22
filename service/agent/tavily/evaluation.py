"""시장성·이해관계자 평가의 공통 실행 흐름.

1. technical_result.evidence에서 시장·운영 성격 근거를 재인용 후보로 고른다.
2. 기술 × 기준마다 긍정/부정 질의 쌍으로 웹 근거를 수집한다.
3. 기술 × 기준 칸마다 LLM을 한 번 호출한다. 그 칸의 재인용 근거와 웹 근거만 구분해 전달한다.
4. 검증에 걸린 Finding만 제외하고 limitations에 사유를 남긴다.
   검증을 통과한 Finding도 라벨(claim_type, scope)이 근거와 맞지 않으면 교정하고 limitations에 기록한다.
5. 기술 × 기준 칸의 충족 여부로 status를 판정한다.
6. 칸별 stance가 한쪽 방향만 있으면 일방적 근거로 기록한다.

칸 단위 호출은 2026-09-22 실측에서 한 번에 전체를 요청하면 finding 하나에 근거 10~19개를 통째로 붙이고
16자리 해시 ID를 잘못 옮겨 적는 문제가 있어 도입했다. 칸마다 근거가 적어 대응 관계가 분명해진다.

관점별 차이(프롬프트, 추가 검증 규칙)는 PerspectiveSpec으로만 주입한다.
pipeline을 import하지 않도록 rules·normalize_result·error_result는 인자로 받는다.
"""

import json
import re
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field as dataclass_field

from service.agent.tavily.client import SearchFn, search_criterion, tavily_search
from service.agent.tavily.evidence_schema import source_domain
from service.agent.tavily.query_templates import (ACADEMIC_DOMAINS, CRITERIA, END_DATE, FORECAST_TERMS,
                                                  LOW_TRUST_DOMAINS, REUSE_KEYWORDS, TECH_ALIASES, TECH_TERMS,
                                                  Perspective)
from state import AnalysisDraft, DraftFinding, Evidence, GraphState, Technology

# pipeline.citation_ids와 같은 인용 ID 패턴
CITATION = re.compile(r"\[((?:sw|hw)_p\d+_c\d+|web_[a-f0-9]+)\]")
# LLM에게 주는 짧은 참조키. 실측에서 LLM이 16자리 해시 ID를 잘못 옮겨 적었다.
REF = re.compile(r"\[(E\d+)\]")
FEEDBACK_KEYWORDS: dict[Perspective, tuple[str, ...]] = {
    "market": ("market", "시장"),
    "stakeholder": ("stakeholder", "이해관계자"),
}
STANCE_LABELS = {"positive": "긍정", "negative": "부정"}
MAX_EVIDENCE_PER_FINDING = 5
YEAR = re.compile(r"(?<!\d)(20\d{2})(?!\d)")

COMMON_PROMPT = f"""이번 호출은 target_technology 하나와 target_criterion 하나만 평가한다.
claim에는 인용한 근거의 title과 excerpt에 실제로 있는 내용만 쓴다. target_technology의 설명이나 selection_reason은
검색 방향을 정하는 입력이며 근거가 아니다. 근거에 없는 수치·모델명·결과를 쓰지 않는다.
모든 finding의 technology_ids는 [target_technology.id], criterion은 target_criterion이다. 다른 기술·기준은 언급하지 않는다.
주어진 근거가 직접 말하는 내용을 finding 하나에 하나씩 쓴다. 긍정과 부정 근거가 모두 있으면 각각 별도 finding으로 작성한다.
각 finding의 evidence_ids에는 그 claim을 직접 뒷받침하는 근거만 1~{MAX_EVIDENCE_PER_FINDING}개 넣는다.
관련 근거를 모두 붙이지 않는다. 근거가 {MAX_EVIDENCE_PER_FINDING}개를 넘는 finding은 제외된다.
근거 id는 E1, E2 같은 참조키다. evidence_ids와 claim 인용에는 참조키를 그대로 쓴다.
claim 본문에 [참조키]로 인용하면 그 참조키는 evidence_ids에 있어야 한다.
claim 본문에 claim_type·scope·stage·stance 같은 필드 값을 적지 않는다.
기준일 이후 연도의 수치나 전망·예상 표현은 claim_type=forecast다.
근거가 이 기준을 뒷받침하지 않으면 findings를 비우고 limitations에 이유를 적는다.
summary는 빈 문자열로 둔다. 요약은 프로그램이 검증된 findings로 만든다.
"""


@dataclass(frozen=True)
class PerspectiveSpec:
    perspective: Perspective
    field: str  # market_result / stakeholder_result
    label: str  # 요약에 쓰는 관점 이름
    prompt: str
    # 관점 고유 규칙. 인자는 finding과 인용 근거 목록이다. 문제가 있으면 사유 문자열, 없으면 None을 반환한다.
    check_finding: Callable[[DraftFinding, list[Evidence]], str | None]
    # 관점 고유 라벨 교정. finding을 고치고 교정 내용을 반환한다. 인자는 finding과 인용 근거 목록이다.
    correct_finding: Callable[[DraftFinding, list[Evidence]], list[str]] = dataclass_field(
        default=lambda finding, cited: [])
    # False면 저신뢰 출처만 인용한 finding을 칸 충족에 세지 않는다(시장성).
    low_trust_counts: bool = True


def reusable_evidence(state: GraphState, perspective: Perspective) -> list[Evidence]:
    keywords = REUSE_KEYWORDS[perspective]
    evidence = (state.get("technical_result") or {}).get("evidence", [])
    return [e for e in evidence if any(k in e["excerpt"].lower() for k in keywords)]


def reused_for(evidence: list[Evidence], technology: Technology) -> list[Evidence]:
    # 재인용 근거는 ID 접두어(sw_/hw_)로 기술에 연결한다. SW 논문으로 HW를 주장하지 못하게 한다.
    return [e for e in evidence if e["id"].startswith(technology["approach"].lower() + "_")]


def relevant_feedback(state: GraphState, perspective: Perspective) -> list[str]:
    keywords = FEEDBACK_KEYWORDS[perspective]
    return [item for item in state.get("quality_feedback", []) if any(k in item.lower() for k in keywords)]


def strip_unknown_citations(text: str, known: set[str]) -> str:
    # 요약·한계 문장의 알 수 없는 인용 하나 때문에 결과 전체가 error가 되지 않게 한다.
    return re.sub(r" {2,}", " ", CITATION.sub(lambda m: m[0] if m[1] in known else "", text)).strip()


def restore_refs(text: str, refs: dict[str, str]) -> str:
    # 알 수 없는 참조키는 지운다. 실제 ID로 쓴 인용은 그대로 둔다.
    return REF.sub(lambda m: f"[{refs[m[1]]}]" if m[1] in refs else "", text)


def domain_in(evidence: Evidence, domains: list[str]) -> bool:
    domain = source_domain(evidence["url"])
    return any(domain == d or domain.endswith("." + d) for d in domains)


def is_low_trust(evidence: Evidence) -> bool:
    return evidence["source_type"] == "web" and domain_in(evidence, LOW_TRUST_DOMAINS)


def is_academic(evidence: Evidence) -> bool:
    if evidence["source_type"] == "paper":
        return True
    return source_domain(evidence["url"]).endswith(".edu") or domain_in(evidence, ACADEMIC_DOMAINS)


def missing_tech_terms(finding: DraftFinding, cited: list[Evidence]) -> list[str]:
    """claim과 인용 근거 양쪽에 기술 고유어가 있지 않은 기술 ID.

    고유어가 없으면 상위 기술(CXL 전체)이나 기업·모델 전체(DeepSeek)에 대한 주장이다. 2026-09-22 실측에서
    "DeepSeek 앱 다운로드 1위"가 근거 본문의 MLA 언급만으로 MLA 시장의 direct 근거가 되어 claim도 검사한다.
    """
    evidence = " ".join(f"{e['title']} {e['excerpt']}" for e in cited).lower()
    claim = CITATION.sub("", finding.claim).lower()
    return [tid for tid in finding.technology_ids
            if not all(any(re.search(term, text) for term in TECH_TERMS.get(tid, [])) for text in (evidence, claim))]


def correct_academic_stage(finding: DraftFinding, cited: list[Evidence]) -> list[str]:
    if finding.stage is not None and cited and all(is_academic(e) for e in cited):
        previous, finding.stage = finding.stage, None
        return [f"stage {previous}→null (학술 자료만으로 상용화 단계 판단 불가)"]
    return []


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


def build_summary(label: str, cells: set, covered: set, valid: list[DraftFinding], one_sided: list[str],
                  corrected: list[str], low_trust_only: list[DraftFinding]) -> str:
    """검증된 결과의 현황만 요약한다. 사실 주장·수치를 넣지 않아 인용 없는 내용이 보고서에 들어가지 않게 한다.

    2026-09-22 실측에서 LLM 요약에 인용 없는 시장 수치와 "상용화 단계 진입" 같은 과장이 들어갔다.
    """
    technologies = sorted({tid for tid, _ in cells})
    parts = [f"{label} 평가: 기술 {len(technologies)}개 × 기준 {len(cells) // max(len(technologies), 1)}개 = "
             f"{len(cells)}칸 중 {len(cells & covered)}칸에서 검증된 근거를 확보했다."]
    missing = sorted(cells - covered)
    if missing:
        parts.append("근거 미확보 칸: " + ", ".join(f"{tid} / {criterion}" for tid, criterion in missing) + ".")
    types = Counter(f.claim_type for f in valid)
    parts.append(f"finding {len(valid)}건(사실 {types['fact']}, 의견 {types['opinion']}, 전망 {types['forecast']}).")
    scopes = Counter(f.scope for f in valid if f.scope)
    if scopes:
        parts.append(f"대상 기술 직접 근거 {scopes['direct']}건, 연관 기술·시장 근거 {scopes['adjacent']}건.")
    if one_sided:
        parts.append(f"한쪽 방향 근거만 있는 칸 {len(one_sided)}개.")
    if corrected:
        parts.append(f"라벨 교정 {len(corrected)}건.")
    if low_trust_only:
        parts.append(f"저신뢰 출처만 인용한 finding {len(low_trust_only)}건.")
    parts.append("세부 주장과 출처는 findings, 확인되지 않은 사항은 limitations를 따른다.")
    return " ".join(parts)


def cell_context(state: GraphState, technology: Technology, criterion: str, reused: list[Evidence],
                 web: list[Evidence], end_date: str, feedback: list[str]) -> tuple[dict, dict[str, str]]:
    ordered = reused + web
    refs = {f"E{i}": e["id"] for i, e in enumerate(ordered, 1)}
    alias = {real: key for key, real in refs.items()}
    # request에는 사람이 작성한 기술 선정 문서 전문이 들어 있다. 2026-09-22 실측에서 LLM이 그 문서의 논문 수치를
    # 가져와 무관한 웹 근거를 붙였으므로, 칸 호출에는 넣지 않는다.
    context = {
        "target_domain": state["target_domain"], "evaluation_base_date": end_date,
        "target_technology": technology, "target_criterion": criterion,
        "reused_evidence": [{**e, "id": alias[e["id"]]} for e in reused],
        "web_evidence": [{**e, "id": alias[e["id"]]} for e in web],
        "revision_feedback": feedback,
    }
    return context, refs


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
                evidence_by_technology[technology["id"]].update(e["id"] for e in reused_for(reused, technology))

            search_limitations, cell_web = [], {}
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
                    cell_web[(technology["id"], criterion)] = found.evidence
            if attempted and failed == attempted:
                # 설계서 4.4: 모델·도구 오류는 error. 근거 부족(partial)과 구분한다.
                raise RuntimeError("Tavily 질의가 모두 실패했습니다: " + "; ".join(search_limitations[:3]))

            system = rules + spec.prompt + COMMON_PROMPT
            feedback = relevant_feedback(state, perspective)
            drafts, llm_limitations, llm_failures, called = [], [], [], 0
            for technology in technologies:
                cell_reused = reused_for(reused, technology)
                for criterion in criteria:
                    web = cell_web[(technology["id"], criterion)]
                    if not web and not cell_reused:
                        continue  # 근거가 없는 칸은 search_criterion이 이미 판단 유보로 기록했다.
                    context, refs = cell_context(state, technology, criterion, cell_reused, web, end_date, feedback)
                    called += 1
                    try:
                        draft = analyst.invoke([("system", system), ("human", json.dumps(context, ensure_ascii=False))])
                    except Exception as exc:
                        # 칸 하나의 LLM 실패는 그 칸만 판단 유보로 기록한다. 모든 칸이 실패하면 error다.
                        llm_failures.append(f"{technology['id']} / {criterion}: LLM 분석 실패 ({type(exc).__name__}), 판단 유보")
                        continue
                    for item in draft.findings:
                        item.evidence_ids = [refs.get(key, key) for key in item.evidence_ids]
                        item.claim = restore_refs(item.claim, refs)
                    draft.summary = restore_refs(draft.summary, refs)
                    llm_limitations.extend(restore_refs(text, refs) for text in draft.limitations)
                    drafts.append(((technology["id"], criterion), draft))
            if called and len(llm_failures) == called:
                raise RuntimeError("모든 칸의 LLM 분석이 실패했습니다.")

            valid, dropped, corrected = [], [], []
            for (technology_id, criterion), draft in drafts:
                for finding in draft.findings:
                    target = f"{', '.join(finding.technology_ids)} / {finding.criterion}"
                    if finding.technology_ids != [technology_id] or finding.criterion != criterion:
                        reason = f"요청한 칸({technology_id} / {criterion})과 다른 기술·기준"
                    else:
                        reason = finding_problem(finding, spec, sources, state, criteria, evidence_by_technology,
                                                 normalize_result)
                    if reason:
                        dropped.append(f"검증 실패로 제외: {target}: {reason}")
                        continue
                    cited = [sources[key] for key in finding.evidence_ids]
                    notes = (correct_forecast(finding, end_date) + correct_academic_stage(finding, cited)
                             + spec.correct_finding(finding, cited))
                    corrected.extend(f"라벨 교정: {target}: {note}" for note in notes)
                    valid.append(finding)

            low_trust_only = [f for f in valid if all(is_low_trust(sources[key]) for key in f.evidence_ids)]
            counted = valid if spec.low_trust_counts else [f for f in valid if f not in low_trust_only]
            low_trust_notes = [] if spec.low_trust_counts else [
                f"저신뢰 출처만 인용: {', '.join(f.technology_ids)} / {f.criterion} finding은 유지하되 칸 충족에서 제외 "
                f"({', '.join(sorted({source_domain(sources[k]['url']) for k in f.evidence_ids}))})"
                for f in low_trust_only]
            covered = {(tid, f.criterion) for f in counted for tid in f.technology_ids}
            cells = {(t["id"], c) for t in technologies for c in criteria}
            all_complete = len(drafts) == len(cells) and all(d.status == "complete" for _, d in drafts)
            status = "complete" if cells <= covered and all_complete else "partial"
            known = set(sources)
            one_sided = one_sided_cells(valid)
            limitations = [strip_unknown_citations(item, known) for item in llm_limitations]
            limitations = list(dict.fromkeys(search_limitations + llm_failures + limitations + dropped + corrected
                                             + low_trust_notes + one_sided))
            summary = build_summary(spec.label, cells, covered, valid, one_sided, corrected, low_trust_only)
            final = AnalysisDraft(status=status, summary=summary, findings=valid, limitations=limitations,
                                  next_queries=[])
            result, _ = normalize_result(final, sources, state, criteria)
            return {field: result}
        except Exception as exc:
            return {field: error_result(field, exc)}

    return run


def finding_problem(finding: DraftFinding, spec: PerspectiveSpec, sources, state, criteria,
                    evidence_by_technology, normalize_result) -> str | None:
    unknown = [key for key in finding.evidence_ids if key not in sources]
    if unknown:
        return f"수집하지 않은 근거 ID {', '.join(unknown[:3])}"
    # 기술 ID·기준·근거 ID·인용 일치는 pipeline 공통 규칙을 한 건 단위로 재사용한다.
    try:
        normalize_result(AnalysisDraft(status="complete", summary="", findings=[finding], limitations=[],
                                       next_queries=[]), sources, state, criteria)
    except ValueError as exc:
        return str(exc)
    # 본문 인용은 필수가 아니다. pipeline.result_markdown이 evidence_ids로 인용을 붙인다.
    # 실측에서 본문 인용 필수 규칙은 LLM이 따르지 않아 모든 finding이 제외됐다.
    if len(finding.evidence_ids) > MAX_EVIDENCE_PER_FINDING:
        return f"근거 {len(finding.evidence_ids)}개로 상한 {MAX_EVIDENCE_PER_FINDING}개 초과"
    # 본문에 인용이 있으면 인용한 근거로 좁힌다. 인용하지 않은 근거를 함께 붙이는 것을 막는다.
    inline = list(dict.fromkeys(CITATION.findall(finding.claim)))
    if inline:
        finding.evidence_ids = inline
    unsupported = [tid for tid in finding.technology_ids if not set(finding.evidence_ids) & evidence_by_technology[tid]]
    if unsupported:
        return f"{', '.join(unsupported)}의 근거가 연결되지 않음"
    if finding.claim_type is None:
        return "claim_type 누락"
    return spec.check_finding(finding, [sources[key] for key in finding.evidence_ids])


def correct_forecast(finding: DraftFinding, end_date: str) -> list[str]:
    if finding.claim_type != "fact":
        return []
    base_year = int(end_date[:4])
    # 웹 근거 ID(16진수 해시)에 연도처럼 보이는 숫자가 있을 수 있어 인용을 먼저 지운다.
    claim = CITATION.sub("", finding.claim).lower()
    if any(int(year) > base_year for year in YEAR.findall(claim)) or any(t in claim for t in FORECAST_TERMS):
        finding.claim_type = "forecast"
        return ["claim_type fact→forecast (기준일 이후 연도 또는 전망 표현)"]
    return []
