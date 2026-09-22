"""기술 조사 결과의 순수 검증·정규화 규칙. 모델·파일·API에 의존하지 않는다."""
import re

from service.agent.node.technical.prompts import ASSESSMENT_DATE, TRL_BASIS
from service.agent.node.technical.schema import TechnicalDraft
from service.schema.state import AgentResult, Evidence, Technology

# pipeline.py의 인용 규칙과 같은 형식을 유지한다. 청크 ID 형식이 바뀌면 두 곳을 함께 고친다.
CITATION = re.compile(r"\[((?:sw|hw)_p\d+_c\d+|web_[a-f0-9]+)\]")
TRL_LEVEL = re.compile(r"TRL ([1-9])(?:~([1-9]))?")


class EvidenceError(ValueError):
    pass


def citation_ids(text: str) -> set[str]:
    return set(CITATION.findall(text))


def paper_prefix(technology_id: str) -> str:
    return technology_id.split("_")[0] + "_p"


def default_queries(technologies: list[Technology], target_domain: str, criteria: list[str]) -> dict[str, str]:
    return {t["id"]: f"{t['name']} {target_domain} {' '.join(criteria)} experimental conditions limitations"
            for t in technologies}


def normalize_queries(queries: list[str]) -> list[str]:
    return [query.strip()[:500] for query in queries if query.strip()][:4]


def retry_queries(missing_items: list[str]) -> list[str]:
    return normalize_queries(["Find evidence and experimental conditions for " + item for item in missing_items])


def error_result(exc: Exception) -> AgentResult:
    # 인증 헤더나 API 요청 본문이 오류 문구에 남지 않도록 EvidenceError 외에는 타입 이름만 남긴다.
    detail = str(exc) if isinstance(exc, EvidenceError) else type(exc).__name__
    return {"status": "error", "summary": "technical_result 실패", "findings": [], "evidence": [],
            "limitations": [f"technical_result: {detail}"]}


def normalize_technical_result(draft: TechnicalDraft, sources: dict[str, Evidence], technologies: list[Technology],
                               criteria: list[str]) -> tuple[AgentResult, list[str]]:
    technology_ids = {t["id"] for t in technologies}
    covered: set[tuple[str, str]] = set()
    findings = []
    limitations = list(draft.limitations)
    for finding in draft.findings:
        if not finding.claim.strip() or not finding.technology_ids or not set(finding.technology_ids) <= technology_ids:
            raise EvidenceError("분석 항목의 기술 ID 또는 주장이 유효하지 않습니다.")
        if finding.criterion not in criteria:
            raise EvidenceError("분석 항목이 지정된 평가 기준을 참조하지 않습니다.")
        if not finding.evidence_ids or not set(finding.evidence_ids) <= sources.keys():
            raise EvidenceError("Finding.evidence_ids가 실제 수집한 Evidence.id를 참조하지 않습니다.")
        if not citation_ids(finding.claim) <= set(finding.evidence_ids):
            raise EvidenceError("주장 본문의 인용과 Finding.evidence_ids가 일치하지 않습니다.")
        record = finding.model_dump(exclude={"criterion", "trl_assessment"})
        if finding.criterion == "TRL":
            if finding.trl_assessment is None:
                raise EvidenceError("TRL 항목에 trl_assessment가 없습니다.")
            if finding.trl_assessment.as_of != ASSESSMENT_DATE:
                raise EvidenceError("TRL 평가 기준일이 고정 기준일과 다릅니다.")
            level = finding.trl_assessment.level_or_range
            if level is not None:
                match = TRL_LEVEL.fullmatch(level)
                if not match or int(match[1]) > int(match[2] or match[1]):
                    raise EvidenceError("TRL 단계는 1~9이며 범위의 하한은 상한 이하여야 합니다.")
            record["trl_assessment"] = {**finding.trl_assessment.model_dump(), "basis": TRL_BASIS}
            if level is None:
                limitations.extend(f"TRL 추정 불가: {tid}" for tid in finding.technology_ids)
        # 한 기술의 논문만으로 양쪽 기술을 조사 완료한 것으로 계산하지 않는다.
        for tid in finding.technology_ids:
            if any(key.startswith(paper_prefix(tid)) for key in finding.evidence_ids):
                covered.add((tid, finding.criterion))
        findings.append(record)
    missing = [f"{tid}: {criterion}" for tid in sorted(technology_ids) for criterion in criteria
               if (tid, criterion) not in covered]
    status = draft.status
    if status == "complete" and missing:
        status = "partial"
    limitations = list(dict.fromkeys(limitations + [f"근거 부족: {item}" for item in missing]))
    if status == "partial" and not limitations:
        limitations = ["일부 근거 부족: 평가자가 complete로 판정하지 않음"]
    text = "\n".join([draft.summary, *[f["claim"] for f in findings], *limitations])
    cited = citation_ids(text)
    if not cited <= sources.keys():
        raise EvidenceError("분석 본문에 수집하지 않은 인용 ID가 있습니다.")
    used = {key for f in findings for key in f["evidence_ids"]} | cited
    return {"status": status, "summary": draft.summary, "findings": findings,
            "evidence": [sources[key] for key in sorted(used)], "limitations": limitations}, missing
