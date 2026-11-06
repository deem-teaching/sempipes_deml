import pytest

import sempipes
from sempipes.config import LLM_PROVIDER_VARIABLE, ensure_default_config


@pytest.fixture(name="restore_environment_and_config")
def fixture_restore_environment_and_config(monkeypatch):
    """Yield monkeypatch; afterwards undo its environment changes and reset the config, so nothing leaks into other tests."""
    yield monkeypatch
    monkeypatch.undo()
    ensure_default_config()


def test_updates(restore_environment_and_config):
    # Check the built-in Gemini defaults, independent of SEMPIPES_LLM_PROVIDER in the user's .env
    restore_environment_and_config.delenv(LLM_PROVIDER_VARIABLE, raising=False)
    ensure_default_config()

    # Get default config and check its values
    default_config = sempipes.get_config()

    assert isinstance(default_config.llm_for_code_generation, sempipes.LLM)
    assert isinstance(default_config.llm_for_batch_processing, sempipes.LLM)

    assert default_config.llm_for_code_generation.name == "gemini/gemini-2.5-flash"
    assert default_config.llm_for_code_generation.parameters["temperature"] == 0.0

    assert default_config.llm_for_batch_processing.name == "gemini/gemini-2.5-flash"
    assert default_config.llm_for_batch_processing.parameters["temperature"] == 0.0

    assert default_config.batch_size_for_batch_processing == 20
    assert default_config.max_retries_for_code_gen == 5

    sempipes.update_config(
        llm_for_code_generation=sempipes.LLM(
            name="ollama/gpt-oss:20b",
            parameters={"api_base": "http://localhost:11434", "temperature": 0.5},
        )
    )

    current_config = sempipes.get_config()
    assert current_config.llm_for_code_generation.name == "ollama/gpt-oss:20b"
    assert current_config.llm_for_code_generation.parameters["api_base"] == "http://localhost:11434"
    assert current_config.llm_for_code_generation.parameters["temperature"] == 0.5
    # Fields that were not passed keep their previous values
    assert current_config.llm_for_batch_processing.name == "gemini/gemini-2.5-flash"

    sempipes.update_config(
        llm_for_batch_processing=sempipes.LLM(
            name="ollama/gemma3:1b",
            parameters={"api_base": "http://localhost:11434", "temperature": 0.0},
        ),
        batch_size_for_batch_processing=5,
    )

    current_config = sempipes.get_config()
    assert current_config.llm_for_batch_processing.name == "ollama/gemma3:1b"
    assert current_config.llm_for_batch_processing.parameters["api_base"] == "http://localhost:11434"
    assert current_config.llm_for_batch_processing.parameters["temperature"] == 0.0
    assert current_config.batch_size_for_batch_processing == 5
    assert current_config.llm_for_code_generation.name == "ollama/gpt-oss:20b"


def test_default_provider_from_environment(restore_environment_and_config):
    environment = restore_environment_and_config
    environment.setenv("SAIA_API_KEY", "test-key")
    environment.setenv("SAIA_BASE_URL", "https://saia.example/v1")
    environment.delenv("SAIA_CODE_GENERATION_MODEL", raising=False)
    environment.delenv("SAIA_BATCH_PROCESSING_MODEL", raising=False)

    environment.setenv(LLM_PROVIDER_VARIABLE, "saia")
    ensure_default_config()
    saia_config = sempipes.get_config()
    assert saia_config.llm_for_code_generation.name == "openai/qwen3-coder-next"
    assert saia_config.llm_for_batch_processing.name == "openai/qwen3.6-35b-a3b"
    assert saia_config.llm_for_code_generation.parameters["api_base"] == "https://saia.example/v1"
    assert saia_config.batch_size_for_batch_processing == 10

    environment.setenv("SAIA_CODE_GENERATION_MODEL", "some-coder-model")
    ensure_default_config()
    assert sempipes.get_config().llm_for_code_generation.name == "openai/some-coder-model"
    # An explicit argument wins over the environment variable
    explicit_config = sempipes.get_default_saia_config(code_generation_model="another-model")
    assert explicit_config.llm_for_code_generation.name == "openai/another-model"

    environment.setenv(LLM_PROVIDER_VARIABLE, "Gemini")
    ensure_default_config()
    assert sempipes.get_config().llm_for_code_generation.name == "gemini/gemini-2.5-flash"

    environment.setenv(LLM_PROVIDER_VARIABLE, "unknown-provider")
    with pytest.raises(ValueError, match="SEMPIPES_LLM_PROVIDER"):
        ensure_default_config()


def test_saia_rejects_commercial_lookalike_url(restore_environment_and_config):
    restore_environment_and_config.setenv("SAIA_API_KEY", "test-key")
    restore_environment_and_config.setenv("SAIA_BASE_URL", "https://api.saia.ai/v1")
    with pytest.raises(ValueError, match="unrelated commercial service"):
        sempipes.get_default_saia_config()
