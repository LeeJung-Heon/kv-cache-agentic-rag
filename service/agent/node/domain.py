import json

from state import GraphState


DOMAIN_PROMPT = """입력 technologies와 target_domain을 기준으로 각 기술의 도메인 적용성을 평가한다.
각 evaluation_criteria 항목에 대해 근거, 실험 조건, 적용 전제, 직접 비교 가능 여부를 claim에 명시한다.
기술마다 변경 대상과 도입 조건이 다를 수 있다. 시스템 전체의 성과를 특정 구성 요소만의 효과로 귀속하지 않는다.
유리한 최대 결과뿐 아니라 구성·운영 조건에 따른 불리한 결과도 함께 검토한다.
모델·문맥 길이·배치·하드웨어·기준선이 다르거나 미확인이면 직접 수치 비교 불가로 명시한다.
직접 측정이 없으면 다른 지표의 개선을 해당 지표의 실측 개선으로 바꾸지 않는다.
선정 이유는 검색 방향을 정하는 입력이며 기술 주장의 증거가 아니다. 기술 개요를 반복하거나 시장 평가 결과를 가정하지 않는다.
각 finding은 대상 기술의 논문 근거를 참조한다. 복수 기술을 대상으로 하는 finding에는 각 기술의 근거가 필요하다.
search_results의 technology_id와 evidence_ids는 검색 대상 연결이며, 내용이 주장을 뒷받침하는지는 원문으로 판단한다.
자료가 충분하지 않은 항목은 findings를 만들어 채우지 말고 limitations에 기술·기준을 명시하며 partial로 반환한다.
검색은 논문 전체를 정밀 검토한 것이 아니다. '논문에 근거 없음' 대신 '이번 검색에서 근거 미확보'라고 표현한다.
조건과 실측값은 원문 보고, 도입 적합성·효과의 해석은 is_inference=true로 구별한다.
summary는 근거 있는 findings만 요약한다. next_queries는 빈 목록으로 반환한다.
"""

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
            previous = state.get("technical_result")
            sources = {e["id"]: e for e in (previous or {}).get("evidence", [])}
            search_results = []
            evidence_by_technology = {t["id"]: set() for t in technologies}
            for technology in technologies:
                side = technology["approach"].lower()
                for criterion in criteria:
                    query = (
                        f"{technology['name']} | {criterion} | {state['target_domain']} | "
                        f"{technology['selection_reason'][:300]} | "
                        "measured results baseline experimental conditions limitations tradeoffs"
                    )
                    evidence_ids = []
                    for row in index.search(query, side):
                        sources[row["id"]] = {
                            "id": row["id"], "source_type": "paper", "title": row["title"],
                            "url": row["url"], "page": row["page"], "published_at": None,
                            "excerpt": row["text"],
                        }
                        evidence_ids.append(row["id"])
                        evidence_by_technology[technology["id"]].add(row["id"])
                    search_results.append({"technology_id": technology["id"], "criterion": criterion,
                                           "query": query, "evidence_ids": evidence_ids})

            context = {
                "request": state["request"], "target_domain": state["target_domain"],
                "technologies": technologies, "evaluation_criteria": criteria,
                "evidence": list(sources.values()), "search_results": search_results,
                "technical_result": previous,
            }
            draft = analyst.invoke([
                ("system", rules + DOMAIN_PROMPT),
                ("human", json.dumps(context, ensure_ascii=False)),
            ])
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
