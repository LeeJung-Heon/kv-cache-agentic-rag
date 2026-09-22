from service.agent.tavily.client import SearchFn, tavily_search
from service.agent.tavily.evaluation import (PerspectiveSpec, is_academic, is_low_trust, make_evaluation_node,
                                             missing_tech_terms)
from state import DraftFinding

STAKEHOLDER_PROMPT = """이해관계자 평가: 기준은 이해관계자 집단(경쟁 기술 진영, 도입사·개발자, 투자 업계)이며
호출마다 그중 하나(target_criterion)의 반응을 평가한다.
근거는 두 종류다. reused_evidence는 기술 조사 에이전트가 수집한 논문 근거를 재인용한 것이고,
web_evidence는 이번에 Tavily로 수집한 웹 근거다. claim에 어느 종류의 근거인지 드러나게 쓴다.
claim에는 발언·행동 주체(기업명, 조직, 개인의 역할)와 그 형식(공식 발표, 제품 문서, 기사, 개인 블로그·포럼)을 명시한다.
기술 설명이나 성능 수치는 이해관계자 반응이 아니다. 발언·행동 주체와 그 입장(채택, 투자, 비판, 경쟁 대응)이
드러나지 않는 근거는 finding으로 쓰지 않는다.
상위 기술 전반(예: CXL 전체)에 대한 반응을 대상 기술 자체에 대한 반응으로 쓰지 않는다. 연관 반응이면 그렇다고 claim에 밝힌다.
경쟁 기술 진영은 당사자의 직접 발표와 제3자 평가자의 해석을 분리한다.
도입사·개발자는 공식 도입 사례와 개인 의견을 구분하고, 한두 개의 개인 의견·소수 의견을 집단 전체의 반응으로 일반화하지 않는다.
투자 업계는 기업 전체에 대한 투자와 특정 기술에 대한 투자를 구분하고, 전망을 실증 근거로 사용하지 않는다.
관측된 반응(실제 발언·도입·투자)과 평가자가 예상하는 이해관계를 구별한다. 예상은 is_inference=true다.
claim_type은 원문 진술의 성격이다: fact(확인된 사실·행동), opinion(의견·평가), forecast(전망·예측).
stance는 해당 집단이 대상 기술에 긍정(positive)·부정(negative)·혼합(mixed)·확인 불가(unknown)인지 원문 내용으로 판단한다.
검색 질의의 방향이나 search_results 연결은 근거 내용의 입장을 뜻하지 않는다.
scope는 프로그램이 근거의 기술 고유어로 정하므로 null로 둬도 된다. stage는 도입 단계가 확인될 때만 채운다.
근거가 없는 집단은 지어내지 말고 limitations에 기술·기준을 명시해 누락으로 기록한다.
웹 스니펫은 원문 전체 검증이 아니므로 '이번 검색에서 확인됨'의 범위로 표현한다.
revision_feedback이 있으면 그 보완 항목을 우선 다룬다. next_queries는 빈 목록이다.
"""


# 논문은 도입·투자 반응이 아니다. 경쟁 기술 진영에서는 제3자 평가로 인정한다.
ACADEMIC_EXCLUDED = ("도입사·개발자", "투자 업계")


def check_stakeholder_finding(finding: DraftFinding, cited) -> str | None:
    if finding.criterion in ACADEMIC_EXCLUDED and cited and all(is_academic(e) for e in cited):
        return f"학술 자료만으로 {finding.criterion} 반응 판단 불가"
    if finding.stance is None:
        return "stance 누락"
    return None


def correct_stakeholder_finding(finding: DraftFinding, cited) -> list[str]:
    # 대상 기술 자체에 대한 반응(direct)과 상위 기술 전반에 대한 반응(adjacent)을 근거의 고유어로 정한다.
    # 2026-09-22 실측에서 "Marvell의 CXL 스위칭 기업 인수"가 CXL-PNM 투자 업계 긍정 근거로 쓰였다.
    finding.scope = "adjacent" if missing_tech_terms(finding, cited) else "direct"
    # 소셜미디어·개인 블로그는 개인 의견 근거로만 쓴다(설계서 3.4 "공식 사례와 개인 의견 구분").
    if finding.claim_type == "fact" and cited and all(is_low_trust(e) for e in cited):
        finding.claim_type = "opinion"
        return ["claim_type fact→opinion (소셜미디어·개인 블로그 출처만 인용)"]
    return []


STAKEHOLDER_SPEC = PerspectiveSpec(perspective="stakeholder", field="stakeholder_result", label="이해관계자",
                                   prompt=STAKEHOLDER_PROMPT, check_finding=check_stakeholder_finding,
                                   correct_finding=correct_stakeholder_finding)


def make_stakeholder_node(analyst, *, rules: str, normalize_result, error_result, search: SearchFn = tavily_search):
    return make_evaluation_node(STAKEHOLDER_SPEC, analyst, rules=rules, normalize_result=normalize_result,
                                error_result=error_result, search=search)
