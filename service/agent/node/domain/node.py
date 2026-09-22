from state import GraphState

from .model import evaluate_domain
from .retrieval import retrieve_domain


def make_domain_node(index, analyst, *, rules, normalize_result, error_result):
    def run(state: GraphState):
        try:
            criteria = state["evaluation_criteria"]["domain"]
            technologies = state["technologies"]
            if not technologies or len({t["id"] for t in technologies}) != len(technologies):
                raise ValueError("기술 목록은 비어 있지 않고 ID가 고유해야 합니다.")
            if any(not t["id"].strip() or not t["name"].strip() for t in technologies):
                raise ValueError("기술 ID와 이름은 비어 있을 수 없습니다.")
            # 현재 공유 인덱스는 SW/HW별 논문 한 개를 구분한다. 같은 진영을 중복 연결하지 않는다.
            if any(t["approach"] not in {"SW", "HW"} for t in technologies) or len({t["approach"] for t in technologies}) != len(technologies):
                raise ValueError("현재 인덱스에는 접근 방식별 기술을 하나씩 연결해야 합니다.")
            if not criteria or any(not isinstance(c, str) or not c.strip() for c in criteria):
                raise ValueError("비어 있지 않은 평가 기준이 필요합니다.")
            sources, search_results, evidence_by_technology = retrieve_domain(index, state)
            draft = evaluate_domain(analyst, state, sources, search_results, rules=rules)
            normalize_result(draft, sources, state, criteria)
            # ID의 존재뿐 아니라 주장 대상별 논문 연결도 확인한 뒤 공통 검사로 넘긴다.
            valid_findings = []
            for finding in draft.findings:
                unsupported = [tid for tid in finding.technology_ids
                               if not set(finding.evidence_ids) & evidence_by_technology[tid]]
                if unsupported:
                    draft.status = "partial" if draft.status != "error" else "error"
                    draft.limitations.append(f"{', '.join(unsupported)} / {finding.criterion}: 해당 기술의 논문 근거가 연결되지 않아 주장을 제외함")
                    draft.summary = "기술별 논문 근거가 연결되지 않은 주장을 제외했습니다. 확인된 평가 항목과 근거 한계를 함께 참고하세요."
                else:
                    valid_findings.append(finding)
            draft.findings = valid_findings
            result, _ = normalize_result(draft, sources, state, criteria)
            return {"domain_result": result}
        except Exception as exc:
            return {"domain_result": error_result("domain_result", exc)}

    return run
