from __future__ import annotations

import os
from contextvars import ContextVar
from dataclasses import dataclass, field

from dataclasses_json import dataclass_json


@dataclass_json
@dataclass(frozen=True)
class LLM:
    name: str
    parameters: dict = field(default_factory=dict)


@dataclass_json
@dataclass(frozen=True)
class Config:
    llm_for_code_generation: LLM = LLM(
        name="gemini/gemini-2.5-flash",
        parameters={"temperature": 0.0},
    )
    llm_for_batch_processing: LLM = LLM(name="gemini/gemini-2.5-flash", parameters={"temperature": 0.0})
    batch_size_for_batch_processing: int = 20
    max_retries_for_code_gen: int = 5
    verbose_code_synthesis: bool = False


# Environment variable that selects the provider of the default config: "gemini" (the default) or "saia"
LLM_PROVIDER_VARIABLE = "SEMPIPES_LLM_PROVIDER"
_SAIA_BASE_URL = "https://chat-ai.academiccloud.de/v1"
_DEFAULT_SAIA_CODE_GENERATION_MODEL = "qwen3-coder-next"
_DEFAULT_SAIA_BATCH_PROCESSING_MODEL = "qwen3.6-35b-a3b"

# Holds the current config; ContextVar makes it safe for threads/async tasks. None means "not chosen yet": the default
# is then created on first use, so that it reflects the environment (and .env) at that time, also in worker processes.
_CONFIG: ContextVar[Config | None] = ContextVar("_CONFIG", default=None)


def get_config() -> Config:
    """Read-only access to the current configuration."""
    config = _CONFIG.get()
    if config is None:
        config = default_config()
        _CONFIG.set(config)
    return config


def _set_config(config: Config) -> None:
    """Replace the current config, e.g. with `_set_config(get_default_saia_config())`."""
    _CONFIG.set(config)


def default_config() -> Config:
    """Return the config SemPipes starts with, chosen by the environment variable SEMPIPES_LLM_PROVIDER.

    * unset, empty or "gemini": Gemini 2.5 Flash for everything (`Config()`).
    * "saia": SAIA (GWDG Academic Cloud), see `get_default_saia_config`.

    Put e.g. `SEMPIPES_LLM_PROVIDER=saia` into .env to make SAIA the default for scripts, notebooks and tests.
    """
    provider = os.environ.get(LLM_PROVIDER_VARIABLE, "").strip().lower()
    if provider != "saia":
        return Config()
    if provider == "saia":
        return get_default_saia_config()
    raise ValueError(f"Unknown {LLM_PROVIDER_VARIABLE}={provider!r}. Use 'gemini' or 'saia'.")


def ensure_default_config() -> None:
    """Ensure that the default config is set (useful for tests)."""
    _set_config(default_config())


def get_default_saia_config(
    code_generation_model: str | None = None,
    batch_processing_model: str | None = None,
) -> Config:
    """Return a config that sends all LLM requests to SAIA (GWDG Academic Cloud) instead of Gemini.

    SAIA is the free, OpenAI-compatible LLM service of GWDG for German universities. The API key and the base URL
    (https://chat-ai.academiccloud.de/v1) are read from the environment variables SAIA_API_KEY and SAIA_BASE_URL
    (put them into .env). Model names change over time; list the current ones with `GET $SAIA_BASE_URL/models`. The
    batch processing model must understand images if you use sem_extract_features on images (qwen3.6-35b-a3b does).

    The models are, in this order of precedence: the arguments, the environment variables SAIA_CODE_GENERATION_MODEL
    and SAIA_BATCH_PROCESSING_MODEL, and the defaults qwen3-coder-next and qwen3.6-35b-a3b.

    Usage:
        sempipes._set_config(sempipes.get_default_saia_config())
    """
    code_generation_model = code_generation_model or os.environ.get(
        "SAIA_CODE_GENERATION_MODEL", _DEFAULT_SAIA_CODE_GENERATION_MODEL
    )
    batch_processing_model = batch_processing_model or os.environ.get(
        "SAIA_BATCH_PROCESSING_MODEL", _DEFAULT_SAIA_BATCH_PROCESSING_MODEL
    )
    api_key = os.environ.get("SAIA_API_KEY")
    base_url = os.environ.get("SAIA_BASE_URL")
    if not api_key or not base_url:
        raise ValueError(
            "SAIA_API_KEY and SAIA_BASE_URL must be set. Add them to the .env file in the repository root "
            "(see .env.example)."
        )
    if "api.saia.ai" in base_url:
        # A commercial service with the same name; sending the GWDG key there leaks it to a third party
        raise ValueError(
            "SAIA_BASE_URL points to api.saia.ai, an unrelated commercial service. "
            f"Use {_SAIA_BASE_URL} for SAIA of GWDG Academic Cloud."
        )

    def saia_llm(model: str, **extra_parameters) -> LLM:
        # "openai/" tells litellm to talk to the OpenAI-compatible API at api_base
        return LLM(
            name=f"openai/{model}",
            parameters={
                "api_base": base_url,
                "api_key": api_key,
                # Qwen models repeat themselves endlessly with temperature 0 (greedy decoding), so we use the sampling
                # settings that Qwen recommends. Generated code therefore varies a bit more between runs.
                "temperature": 0.7,
                "top_p": 0.8,
                "max_tokens": 8192,
                "timeout": 120,
                **extra_parameters,
            },
        )

    return Config(
       llm_for_code_generation=saia_llm(code_generation_model, presence_penalty=1.0),
       llm_for_batch_processing=saia_llm(batch_processing_model),
       batch_size_for_batch_processing=10,
    )


def update_config(**kwargs) -> None:
    """Update specific attributes of the current config by creating a new Config instance."""
    current_config = get_config()

    # Create a new config with updated attributes
    updated_config = Config(
        llm_for_code_generation=kwargs.get("llm_for_code_generation", current_config.llm_for_code_generation),
        llm_for_batch_processing=kwargs.get("llm_for_batch_processing", current_config.llm_for_batch_processing),
        batch_size_for_batch_processing=kwargs.get(
            "batch_size_for_batch_processing", current_config.batch_size_for_batch_processing
        ),
        max_retries_for_code_gen=kwargs.get("max_retries_for_code_gen", current_config.max_retries_for_code_gen),
        verbose_code_synthesis=kwargs.get("verbose_code_synthesis", current_config.verbose_code_synthesis),
    )

    _set_config(updated_config)
