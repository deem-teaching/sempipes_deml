import os
from pathlib import Path

import pytest

import sempipes  # noqa: F401  # loads .env and registers the semantic operators

REPO_ROOT = Path(__file__).resolve().parent.parent
TEST_DATA_DIR = REPO_ROOT / "tests" / "data"

# API key environment variable needed per litellm provider prefix (local providers need none)
_PROVIDER_API_KEYS = {"gemini": "GEMINI_API_KEY", "openai": "OPENAI_API_KEY", "anthropic": "ANTHROPIC_API_KEY"}


def _has_api_key_for_default_llms() -> bool:
    config = sempipes.get_config()
    for llm in (config.llm_for_code_generation, config.llm_for_batch_processing):
        if llm.parameters.get("api_key"):
            continue  # the key is passed explicitly, e.g. by get_default_saia_config()
        key = _PROVIDER_API_KEYS.get(llm.name.split("/")[0])
        if key is not None and not os.environ.get(key):
            return False
    return True


HAS_LLM_API_KEY = _has_api_key_for_default_llms()

requires_llm_api_key = pytest.mark.skipif(
    not HAS_LLM_API_KEY,
    reason="Needs an API key for the configured LLM provider (default: GEMINI_API_KEY) in the environment or .env",
)


def requires_test_data(relative_path: str):
    """Skip a test if a (large, not distributed) file under tests/data is missing."""
    return pytest.mark.skipif(
        not (TEST_DATA_DIR / relative_path).exists(),
        reason=f"Test data tests/data/{relative_path} is not available",
    )
