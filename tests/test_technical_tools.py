from service.agent.node.technical.retrieval import make_paper_search_tool
from tests.fixtures import PaperFixture


def test_paper_search_returns_evidence_records():
    index = PaperFixture()
    tool = make_paper_search_tool(index)
    rows = tool.invoke({"query": "MLA KV cache", "side": "sw"})
    assert tool.name == "paper_search"
    assert index.calls == [("MLA KV cache", "sw")]
    assert rows == [{"id": "sw_p1_c1", "source_type": "paper", "title": "sw paper", "url": "https://example.org/sw",
                     "page": 1, "published_at": None, "excerpt": "sw paper evidence"}]
