import pytest

from pipeline import initial_state, make_nodes, select_technologies
from service.agent.graph.technical import build_technical_research_graph
from service.agent.node.technical import model as technical_model
from tests.fixtures import ModelFixture, PaperFixture, sample_state


def test_technical_model_is_declared_in_module_and_uses_team_settings(monkeypatch):
    monkeypatch.setenv("OPENAI_MODEL", "shared-model-a")
    monkeypatch.setenv("TECHNICAL_MODEL", "must-be-ignored")
    monkeypatch.setattr(technical_model.settings, "openai_api_key", "test-key-not-used-for-network")
    monkeypatch.setattr(technical_model.settings, "openai_base_url", "https://example.test/v1")
    model = technical_model.create_technical_model()
    assert model.model_name == technical_model.TECHNICAL_MODEL == "gpt-4.1-mini"
    assert str(model.root_client.base_url) == "https://example.test/v1/"
    assert model.temperature == 0 and model.max_retries == 0 and model.max_tokens == 12000


def test_empty_settings_key_falls_back_to_process_env(monkeypatch):
    monkeypatch.setattr(technical_model.settings, "openai_api_key", "")
    monkeypatch.setattr(technical_model.settings, "openai_base_url", None)
    monkeypatch.setenv("OPENAI_API_KEY", "env-key-not-used-for-network")
    model = technical_model.create_technical_model()
    assert model.openai_api_key.get_secret_value() == "env-key-not-used-for-network"


def test_graph_without_model_uses_internal_factory(monkeypatch):
    monkeypatch.setattr(technical_model, "create_technical_model", lambda: ModelFixture(), raising=False)
    output = build_technical_research_graph(PaperFixture()).invoke(sample_state())
    assert output["technical_result"]["status"] == "complete"
    assert len(output["technical_result"]["findings"]) == 8


def test_explicit_model_skips_internal_creation_and_credentials(monkeypatch):
    def unexpected_factory():
        pytest.fail("명시적 모델을 주입했는데 내부 모델을 생성함")

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(technical_model, "create_technical_model", unexpected_factory, raising=False)
    output = build_technical_research_graph(PaperFixture(), model=ModelFixture()).invoke(sample_state())
    assert output["technical_result"]["status"] == "complete"


def test_pipeline_technical_node_does_not_use_shared_model(monkeypatch):
    monkeypatch.setattr(technical_model, "create_technical_model", lambda: ModelFixture(), raising=False)
    nodes = make_nodes(PaperFixture(), ModelFixture(fail=True))
    state = initial_state()
    state.update(select_technologies(state))
    output = nodes["technical_research"].invoke(state)
    assert output["technical_result"]["status"] == "complete"
    assert len(output["technical_result"]["findings"]) == 8
