from service.agent.tavily.client import SearchFn, tavily_search
from service.agent.tavily.evaluation import PerspectiveSpec, make_evaluation_node
from state import DraftFinding

MARKET_PROMPT = """시장성 평가: 입력 technologies 각각의 시장 규모·성장성, 상용화·채택, 생태계를 평가한다.
evaluation_criteria의 각 기준을 기술마다 따로 판단하고, 각 finding의 criterion은 그중 하나다.
근거는 두 종류다. reused_evidence는 기술 조사 에이전트가 수집한 논문 근거를 재인용한 것이고,
web_evidence는 이번에 Tavily로 수집한 웹 근거다. claim에 어느 종류의 근거인지 드러나게 쓴다.
재인용 논문 근거는 운영 데이터·비용 조건으로만 사용하며, 논문 성능을 시장 수요나 채택의 증거로 확대하지 않는다.
시장 규모·성장성은 연도·지역·시장 정의를 명시한다. 해당 기술 자체의 시장이면 scope=direct,
CXL 메모리 전체·LLM 추론 인프라 전체처럼 연관 시장이면 scope=adjacent로 표시하고, 연관 시장 수치를 기술 자체 시장으로 쓰지 않는다.
상용화·채택은 제품명·버전·주체를 확인하고 stage를 announced(계획 발표)·pilot(실증)·production(실제 운영)으로 구분한다.
다른 기준에서도 단계가 확인되면 stage를 채우고, 아니면 null로 둔다.
생태계는 지원 프레임워크·표준의 실제 버전과 범위를 확인한다.
claim_type은 원문 진술의 성격이다: fact(확인된 사실), opinion(의견·평가), forecast(전망·예측).
is_inference는 에이전트가 근거를 해석했는지 여부이며 claim_type과 별개다. 전망 기사를 그대로 옮기면 forecast이면서 is_inference=false다.
stance는 해당 finding이 대상 기술의 시장성에 긍정(positive)·부정(negative)·혼합(mixed)·확인 불가(unknown)인지 원문 내용으로 판단한다.
검색 질의의 방향이나 search_results 연결은 근거 내용의 입장을 뜻하지 않는다.
근거가 없는 기준은 지어내지 말고 limitations에 기술·기준을 명시해 누락으로 기록한다.
웹 스니펫은 원문 전체 검증이 아니므로 '이번 검색에서 확인됨'의 범위로 표현한다.
revision_feedback이 있으면 그 보완 항목을 우선 다룬다. next_queries는 빈 목록이다.
"""


def check_market_finding(finding: DraftFinding) -> str | None:
    if finding.scope is None:
        return "시장성 finding의 scope 누락"
    if finding.criterion == "상용화·채택" and finding.stage is None:
        return "상용화·채택 finding의 stage 누락"
    if finding.stance is None:
        return "stance 누락"
    return None


MARKET_SPEC = PerspectiveSpec(perspective="market", field="market_result", prompt=MARKET_PROMPT,
                              check_finding=check_market_finding)


def make_market_node(analyst, *, rules: str, normalize_result, error_result, search: SearchFn = tavily_search):
    return make_evaluation_node(MARKET_SPEC, analyst, rules=rules, normalize_result=normalize_result,
                                error_result=error_result, search=search)
