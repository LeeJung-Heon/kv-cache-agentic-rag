import json

import pytest

from service.agent.node.technical.core import (EvidenceError, citation_ids, default_queries, error_result,
                                     normalize_technical_result, retry_queries)
from service.agent.node.technical.prompts import ASSESSMENT_DATE, TRL_BASIS
from service.agent.node.technical.schema import TechnicalDraft, TechnicalFindingDraft, TrlDraft
from tests.fixtures import sample_technologies

CRITERIA = ["원리", "성능", "한계", "TRL"]


def sources():
    return {f"{side}_p1_c1": {"id": f"{side}_p1_c1", "source_type": "paper", "title": f"{side} paper", "url": "",
                              "page": 1, "published_at": None, "excerpt": f"{side} evidence"} for side in ("sw", "hw")}


def finding(tid, criterion, evidence=None, trl=None, claim=None):
    return TechnicalFindingDraft(technology_ids=[tid], criterion=criterion, claim=claim or f"{criterion} 확인",
                                 evidence_ids=evidence or [tid[:2] + "_p1_c1"], is_inference=False, trl_assessment=trl)


def trl(level="TRL 3~4", as_of=ASSESSMENT_DATE):
    return TrlDraft(level_or_range=level, as_of=as_of, confidence="중간", unverified_conditions=["운용 근거 미확인"])


def full_draft(status="complete"):
    findings = [finding(tid, c, trl=trl() if c == "TRL" else None) for tid in ("sw_01", "hw_01") for c in CRITERIA]
    return TechnicalDraft(status=status, summary="요약", findings=findings, limitations=[], next_queries=[])


def test_citation_ids_only_known_patterns():
    assert citation_ids("a [sw_p3_c2] b [web_0123abcd] c [foo]") == {"sw_p3_c2", "web_0123abcd"}


def test_default_queries_one_per_technology():
    queries = default_queries(sample_technologies(), "도메인", CRITERIA)
    assert set(queries) == {"sw_01", "hw_01"}
    assert queries["sw_01"].startswith("DeepSeek-V2 MLA 도메인 원리 성능 한계 TRL")


def test_retry_queries_from_missing_items():
    assert retry_queries(["hw_01: TRL"]) == ["Find evidence and experimental conditions for hw_01: TRL"]


def test_fallback_queries_limit_count_and_length():
    queries = retry_queries(["hw_01: " + "가" * 600] * 8)
    assert len(queries) == 4
    assert all(len(query) == 500 for query in queries)


def test_complete_when_every_pair_covered():
    result, missing = normalize_technical_result(full_draft(), sources(), sample_technologies(), CRITERIA)
    assert result["status"] == "complete" and missing == []
    assert [e["id"] for e in result["evidence"]] == ["hw_p1_c1", "sw_p1_c1"]
    trl_findings = [f for f in result["findings"] if "trl_assessment" in f]
    assert len(trl_findings) == 2 and trl_findings[0]["trl_assessment"]["basis"] == TRL_BASIS
    assert all("criterion" not in f for f in result["findings"])


def test_missing_pair_forces_partial():
    draft = full_draft()
    draft.findings = [f for f in draft.findings if not (f.technology_ids == ["hw_01"] and f.criterion == "TRL")]
    result, missing = normalize_technical_result(draft, sources(), sample_technologies(), CRITERIA)
    assert result["status"] == "partial" and missing == ["hw_01: TRL"]
    assert "근거 부족: hw_01: TRL" in result["limitations"]


def test_other_paper_evidence_does_not_cover_technology():
    draft = full_draft()
    for f in draft.findings:
        if f.technology_ids == ["hw_01"] and f.criterion == "성능":
            f.evidence_ids = ["sw_p1_c1"]
    _, missing = normalize_technical_result(draft, sources(), sample_technologies(), CRITERIA)
    assert missing == ["hw_01: 성능"]


def test_trl_finding_requires_assessment():
    draft = full_draft()
    draft.findings[3].trl_assessment = None
    with pytest.raises(EvidenceError):
        normalize_technical_result(draft, sources(), sample_technologies(), CRITERIA)


def test_trl_as_of_must_match_assessment_date():
    draft = full_draft()
    draft.findings[3].trl_assessment = trl(as_of="2020-01-01")
    with pytest.raises(EvidenceError):
        normalize_technical_result(draft, sources(), sample_technologies(), CRITERIA)


def test_trl_none_level_adds_limitation():
    draft = full_draft()
    draft.findings[3].trl_assessment = trl(level=None)
    result, _ = normalize_technical_result(draft, sources(), sample_technologies(), CRITERIA)
    assert "TRL 추정 불가: sw_01" in result["limitations"]
    assert result["findings"][3]["trl_assessment"]["level_or_range"] is None


def test_non_trl_finding_drops_assessment():
    draft = full_draft()
    draft.findings[0].trl_assessment = trl()
    result, _ = normalize_technical_result(draft, sources(), sample_technologies(), CRITERIA)
    assert "trl_assessment" not in result["findings"][0]


def test_unknown_evidence_id_rejected():
    draft = full_draft()
    draft.findings[0].evidence_ids = ["missing"]
    with pytest.raises(EvidenceError):
        normalize_technical_result(draft, sources(), sample_technologies(), CRITERIA)


def test_claim_citation_must_be_in_evidence_ids():
    draft = full_draft()
    draft.findings[0].claim = "원리 [hw_p1_c1]"
    with pytest.raises(EvidenceError):
        normalize_technical_result(draft, sources(), sample_technologies(), CRITERIA)


def test_unknown_criterion_rejected():
    draft = full_draft()
    draft.findings[0].criterion = "시장"
    with pytest.raises(EvidenceError):
        normalize_technical_result(draft, sources(), sample_technologies(), CRITERIA)


def test_error_result_hides_non_evidence_messages():
    hidden = error_result(RuntimeError("SECRET_SENTINEL"))
    shown = error_result(EvidenceError("근거 없음"))
    assert hidden["status"] == "error" and "SECRET_SENTINEL" not in json.dumps(hidden)
    assert "근거 없음" in shown["limitations"][0]
