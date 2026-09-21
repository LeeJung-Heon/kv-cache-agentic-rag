from pipeline import build_graph


def main():
    seen = []
    names = ["technical_analysis", "market_evaluation", "stakeholder_evaluation",
             "domain_evaluation", "synthesis", "final_report"]

    def node(name):
        def run(state):
            if name == "synthesis":
                assert all(key in state for key in names[1:4]), "병렬 평가가 끝나기 전에 종합됨"
            seen.append(name)
            return {name: name if name in names[-2:] else {"text": name}}
        return run

    result = build_graph({name: node(name) for name in names}).invoke({"input": {}})
    assert result["final_report"] == "final_report"
    assert sorted(seen) == sorted(names), "각 노드는 정확히 한 번 실행되어야 함"
    assert seen[0] == names[0] and seen[-2:] == names[-2:]
    print("fan-out/fan-in: PASS (외부 API 호출 없음)")


if __name__ == "__main__":
    main()
