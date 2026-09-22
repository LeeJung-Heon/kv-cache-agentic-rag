"""외부 API·모델 없이 서브그래프를 돌리는 픽스처."""
from service.schema.state import Technology


class PaperFixture:
    """side별 청크 하나를 돌려주는 논문 인덱스 대역이며 호출 질의를 기록한다."""

    def __init__(self):
        self.calls: list[tuple[str, str]] = []

    def search(self, query: str, side: str, k: int = 5) -> list[dict]:
        self.calls.append((query, side))
        return [{"id": f"{side}_p1_c1", "side": side, "page": 1, "title": f"{side} paper",
                 "url": f"https://example.org/{side}", "file": f"{side}.pdf", "text": f"{side} paper evidence", "score": 0.9}]


class FailingPaperFixture(PaperFixture):
    """지정한 검색 호출에서 실패하여 초기 검색과 재검색 오류를 재현한다."""

    def __init__(self, fail_at):
        super().__init__()
        self.fail_at = fail_at

    def search(self, query, side, k=5):
        rows = super().search(query, side, k)
        if len(self.calls) == self.fail_at:
            raise RuntimeError("SECRET_SENTINEL")
        return rows


def sample_technologies() -> list[Technology]:
    return [{"id": "sw_01", "name": "DeepSeek-V2 MLA", "approach": "SW", "selection_reason": "구조적 접근"},
            {"id": "hw_01", "name": "CXL-PNM", "approach": "HW", "selection_reason": "연산 위치 변경"}]


def sample_state() -> dict:
    return {"request": "두 기술을 비교하라", "target_domain": "데이터센터·클라우드 LLM 서빙",
            "evaluation_criteria": {"technical": ["원리", "성능", "한계", "TRL"]},
            "technologies": sample_technologies(), "technical_retry_count": 0}


import json

from service.agent.node.technical.prompts import ASSESSMENT_DATE
from service.agent.node.technical.schema import TechnicalDraft, TechnicalFindingDraft, TrlDraft


class ModelFixture:
    """첫 호출은 hw_01 TRL을 빠뜨린 partial, 이후는 complete를 돌려주는 구조화 출력 모델 대역이다."""

    def __init__(self, always_partial=False, fail=False):
        self.calls = 0
        self.always_partial = always_partial
        self.fail = fail
        self.contexts: list[dict] = []

    def with_structured_output(self, *args, **kwargs):
        return self

    def invoke(self, messages):
        self.calls += 1
        if self.fail:
            raise RuntimeError("SECRET_SENTINEL")
        context = json.loads(messages[-1][1])
        self.contexts.append(context)
        partial = self.always_partial or self.calls == 1
        findings = []
        for technology in context["technologies"]:
            for criterion in context["evaluation_criteria"]:
                if partial and technology["id"] == "hw_01" and criterion == "TRL":
                    continue
                trl = TrlDraft(level_or_range="TRL 3~4", as_of=ASSESSMENT_DATE, confidence="중간",
                               unverified_conditions=["실제 운용 근거 미확인"]) if criterion == "TRL" else None
                findings.append(TechnicalFindingDraft(technology_ids=[technology["id"]], criterion=criterion,
                                                      claim=f"{criterion} 근거 확인 [{technology['id'][:2]}_p1_c1]",
                                                      evidence_ids=[technology["id"][:2] + "_p1_c1"],
                                                      is_inference=criterion == "TRL", trl_assessment=trl))
        return TechnicalDraft(status="partial" if partial else "complete", summary="기술 조사 요약", findings=findings,
                              limitations=["HW TRL 근거 부족"] if partial else [],
                              next_queries=["CXL-PNM prototype maturity TRL validation"] if partial else [])


class EmptyFirstDraftModelFixture(ModelFixture):
    """첫 결과는 모든 항목이 미충족이며 지정한 수정 질의를 반환한다."""

    def __init__(self, next_queries):
        super().__init__()
        self.next_queries = next_queries

    def invoke(self, messages):
        draft = super().invoke(messages)
        if self.calls == 1:
            draft.findings = []
            draft.next_queries = list(self.next_queries)
        return draft


class BadEvidenceFirstModelFixture(ModelFixture):
    """첫 호출은 존재하지 않는 근거 ID를 섞어 돌려주고, 수정 요청을 받은 두 번째 호출부터 정상 응답한다."""

    def __init__(self, always_bad=False):
        super().__init__()
        self.always_bad = always_bad
        self.feedback_seen: list[list[str]] = []

    def invoke(self, messages):
        draft = super().invoke(messages)
        context = json.loads(messages[-1][1])
        self.feedback_seen.append(context.get("validation_feedback", []))
        if self.always_bad or self.calls == 1:
            draft.findings[0].evidence_ids = ["sw_p9_c9"]
        return draft
