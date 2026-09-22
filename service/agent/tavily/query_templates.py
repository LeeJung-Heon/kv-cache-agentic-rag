"""시장성·이해관계자 평가의 Tavily 질의 템플릿.

관점별 차이는 이 파일의 데이터로만 표현한다. 검색 로직은 두 에이전트가 공유한다.
평가 기준은 설계서 3.4절을 따른다. 레포 pipeline.CRITERIA의 목업 기준과는 다르다.
"""

from typing import Literal, NamedTuple

Perspective = Literal["market", "stakeholder"]
Topic = Literal["general", "news"]

# 설계서 평가 기준일. 이후 발행 자료는 검색·후처리 양쪽에서 제외한다.
END_DATE = "2026-09-21"

# 긍정·부정 질의를 쌍으로 두어 한쪽 방향 근거만 수집하는 확증편향을 줄인다.
QUERY_TEMPLATES: dict[Perspective, dict[str, dict[str, str]]] = {
    "market": {
        "시장 규모·성장성": {
            "positive": "{tech} market size growth demand forecast",
            "negative": "{tech} market uncertainty slowdown limited demand",
            "topic": "general",
        },
        "상용화·채택": {
            "positive": "{tech} commercial deployment production adoption",
            "negative": "{tech} adoption barriers delay not deployed",
            "topic": "news",
        },
        "생태계": {
            "positive": "{tech} framework support standard ecosystem",
            "negative": "{tech} compatibility issues lack of support",
            "topic": "general",
        },
    },
    "stakeholder": {
        "경쟁 기술 진영": {
            "positive": "{tech} advantages compared to alternatives",
            "negative": "{tech} criticism drawbacks versus competing approaches",
            "topic": "general",
        },
        "도입사·개발자": {
            "positive": "{tech} developers adopt production experience",
            "negative": "{tech} developer complaints migration problems",
            "topic": "general",
        },
        "투자 업계": {
            "positive": "{tech} investment funding analyst outlook",
            "negative": "{tech} investment risk analyst skepticism",
            "topic": "news",
        },
    },
}

CRITERIA: dict[Perspective, list[str]] = {
    perspective: list(templates) for perspective, templates in QUERY_TEMPLATES.items()
}

# 첫 항목은 1차 검색명, 나머지는 결과가 빈약할 때만 쓰는 보강 검색 별칭이다.
# hw_01은 2026-09-22 실측에서 "CXL-PNM"보다 "CXL processing-near-memory"의 score가 높아 순서를 바꿨다.
TECH_ALIASES: dict[str, list[str]] = {
    "sw_01": ["DeepSeek MLA", "Multi-head Latent Attention", "MLA KV cache compression"],
    "hw_01": ["CXL processing-near-memory", "CXL-PNM", "CXL PNM LLM inference"],
}

# scope=direct로 인정하려면 인용 근거의 제목·발췌에 있어야 하는 기술 고유어(대소문자 무시 정규식).
# "CXL"만 있는 근거는 CXL 전체 시장이므로 연관 시장(adjacent)이다.
TECH_TERMS: dict[str, list[str]] = {
    "sw_01": [r"\bmla\b", r"multi-head latent attention", r"latent attention"],
    "hw_01": [r"\bpnm\b", r"processing[- ]near[- ]memory"],
}

# claim_type=fact인데 이 표현이 있으면 forecast로 교정한다. 기준일 이후 연도는 evaluation에서 따로 검사한다.
FORECAST_TERMS = ["전망", "예상", "예측", "cagr", "연평균", "forecast", "projected", "expected to"]


class QueryPair(NamedTuple):
    positive: str
    negative: str
    topic: Topic


def search_names(technology_id: str) -> list[str]:
    if technology_id not in TECH_ALIASES:
        raise KeyError(f"검색명이 등록되지 않은 기술 ID: {technology_id}")
    return TECH_ALIASES[technology_id]


def build_query_pair(perspective: Perspective, criterion: str, technology_id: str,
                     alias_index: int = 0) -> QueryPair:
    """alias_index=0은 1차 검색, 1 이상은 별칭 보강 검색이다."""
    templates = QUERY_TEMPLATES.get(perspective)
    if templates is None:
        raise KeyError(f"알 수 없는 관점: {perspective}")
    if criterion not in templates:
        raise KeyError(f"{perspective} 관점에 없는 평가 기준: {criterion}")
    names = search_names(technology_id)
    if not 0 <= alias_index < len(names):
        raise IndexError(f"{technology_id}의 별칭 범위를 벗어남: {alias_index}")
    template, tech = templates[criterion], names[alias_index]
    return QueryPair(template["positive"].format(tech=tech), template["negative"].format(tech=tech),
                     template["topic"])


# technical_result.evidence 중 시장·운영 성격 근거만 재인용 후보로 고르는 키워드(소문자 부분 일치).
# 논문 성능 수치를 시장성 근거로 반복하지 않도록 비용·운영·채택 맥락이 있는 청크만 넘긴다.
REUSE_KEYWORDS: dict[Perspective, list[str]] = {
    "market": ["cost", "tco", "dollar", "price", "power", "energy", "watt", "deployment", "deployed",
               "production", "commercial", "cloud", "azure", "datacenter", "data center", "operator",
               "adoption", "market"],
    "stakeholder": ["vendor", "operator", "cloud provider", "developer", "open-source", "open source",
                    "framework", "vllm", "sglang", "hugging face", "adoption", "industry", "azure"],
}
