# Configuration

All settings live in one immutable `Config` object (`sempipes/config.py`). You read it with `sempipes.get_config()` and change it with `sempipes.update_config(...)`.

```python
import sempipes

print(sempipes.get_config())
# Config(llm_for_code_generation=LLM(name='gemini/gemini-2.5-flash', parameters={'temperature': 0.0}),
#        llm_for_batch_processing=LLM(name='gemini/gemini-2.5-flash', parameters={'temperature': 0.0}),
#        batch_size_for_batch_processing=20, max_retries_for_code_gen=5, verbose_code_synthesis=False)

sempipes.update_config(
    llm_for_code_generation=sempipes.LLM(name="gemini/gemini-2.5-pro", parameters={"temperature": 0.0}),
    verbose_code_synthesis=True,
)
```

`update_config` builds a new `Config` and keeps every field you don't pass. The config is a frozen dataclass, so `get_config().batch_size_for_batch_processing = 5` raises an error. Use `update_config` instead.

> **Typos are silently ignored.** `update_config` only looks at the five known keyword names. `update_config(batch_size=5)` does nothing and raises no error. Print `get_config()` afterwards if in doubt.

To reset everything to the defaults, call `sempipes.config.ensure_default_config()`.

## `sempipes.LLM`

```python
sempipes.LLM(name: str, parameters: dict = {})
```

| Field | Meaning |
|---|---|
| `name` | A [litellm](https://docs.litellm.ai/docs/providers) model string of the form `provider/model`, e.g. `gemini/gemini-2.5-flash`, `openai/gpt-4.1`, `ollama/gpt-oss:20b` |
| `parameters` | Keyword arguments passed unchanged to `litellm.completion` / `litellm.batch_completion`, e.g. `temperature` or `api_base` |

## All configuration fields

| Field | Type | Default | Used for |
|---|---|---|---|
| `llm_for_code_generation` | `LLM` | `LLM("gemini/gemini-2.5-flash", {"temperature": 0.0})` | Every operator that writes code (`sem_clean`, `sem_refine`, `sem_fillna` default mode, `sem_extract_features` code mode, `sem_gen_features`, `sem_agg_features`, `sem_augment` code mode, `sem_distill`), plus `sem_select`'s per-column decisions, `sem_choose`, and the feature-proposal step of `sem_extract_features` (`output_columns=None`) |
| `llm_for_batch_processing` | `LLM` | `LLM("gemini/gemini-2.5-flash", {"temperature": 0.0})` | Per-row work: `sem_fillna(with_llm_only=True)`, `sem_extract_features(generate_via_code=False)`, `sem_augment(generate_via_code=False)` |
| `batch_size_for_batch_processing` | `int` | `20` | How many per-row requests are sent **concurrently** in one chunk via `litellm.batch_completion` |
| `max_retries_for_code_gen` | `int` | `5` | Total number of attempts to get working code: the first try plus retries with the error fed back. `sem_choose` applies it per parameter. |
| `verbose_code_synthesis` | `bool` | `False` | Log every accepted piece of code and full tracebacks of failed attempts. Applies to the [optimisable operators](concepts.md#optimisable-operators-and-hidden-variables) (`sem_gen_features`, `sem_agg_features`, code-mode `sem_extract_features` and `sem_augment`). The other operators ignore it: they always print their own progress and error messages, and `sem_fillna` (default mode) and `sem_distill` also print the code. |

Some limits are hard-coded and cannot be configured. `_MAX_RETRIES = 5` in `sempipes/llm/llm.py` sets how often an empty LLM answer is re-requested and how many rounds failed or unparsable per-row requests are retried.

## Provider examples

API keys are read from environment variables, usually through the `.env` file (see [installation.md](installation.md#4-api-keys)).

**Gemini (default, needs `GEMINI_API_KEY`):**

```python
sempipes.update_config(
    llm_for_code_generation=sempipes.LLM(name="gemini/gemini-2.5-flash", parameters={"temperature": 0.0}),
)
```

**OpenAI (needs `OPENAI_API_KEY`):**

```python
sempipes.update_config(
    llm_for_code_generation=sempipes.LLM(name="openai/gpt-4.1", parameters={"temperature": 0.0}),
    llm_for_batch_processing=sempipes.LLM(name="openai/gpt-4.1", parameters={"temperature": 0.0}),
)
```

**Ollama (local, no key; start the Ollama app or `ollama serve`, then `ollama pull <model>`):**

```python
sempipes.update_config(
    llm_for_code_generation=sempipes.LLM(
        name="ollama/gpt-oss:20b",
        parameters={"api_base": "http://localhost:11434", "temperature": 0.0},
    ),
    llm_for_batch_processing=sempipes.LLM(
        name="ollama/gemma3:1b",
        parameters={"api_base": "http://localhost:11434", "temperature": 0.0},
    ),
)
```

`http://localhost:11434` is Ollama's default address.

## Using SAIA (GWDG Academic Cloud)

[SAIA](https://docs.hpc.gwdg.de/services/ai-services/saia/) is the free LLM service of GWDG for German universities. It offers open models through an OpenAI-compatible API, which litellm reaches with the `openai/` prefix and an `api_base`. Request an API key as described in the SAIA documentation and put it, together with the base URL, into `.env`:

```
SAIA_API_KEY=...
SAIA_BASE_URL=https://chat-ai.academiccloud.de/v1
```

Use exactly this URL. `api.saia.ai` is an unrelated commercial service with the same name.

Then switch SemPipes to SAIA at the top of your script or notebook, before you build the pipeline:

```python
import sempipes

sempipes._set_config(sempipes.get_default_saia_config())
```

**Make SAIA the default.** Add this line to `.env`:

```
SEMPIPES_LLM_PROVIDER=saia
```

From then on every script, notebook **and test** starts with the SAIA config. That includes the tests, which call `ensure_default_config()` and would otherwise switch back to Gemini, and worker processes (see the caveat below). Leave the variable out, or set it to `gemini`, for the Gemini default. For a single run you can also set it in the shell, which takes precedence over `.env`:

```bash
SEMPIPES_LLM_PROVIDER=saia poetry run pytest -q tests/test_sem_refine.py
```

`get_default_saia_config()` uses `qwen3-coder-next` for code generation and `qwen3.6-35b-a3b` (which also reads images) for per-row processing, and sets `batch_size_for_batch_processing=10`. It also limits every request to 2 minutes and 8192 output tokens, because open models occasionally repeat themselves endlessly or a request is never answered, and litellm would otherwise wait 10 minutes. It is equivalent to:

```python
SAIA = {
    "api_base": os.environ["SAIA_BASE_URL"],
    "api_key": os.environ["SAIA_API_KEY"],
    # Qwen models repeat themselves with greedy decoding (temperature 0), so Qwen's recommended sampling is used
    "temperature": 0.7,
    "top_p": 0.8,
    "max_tokens": 8192,
    "timeout": 120,
}
sempipes.update_config(
    llm_for_code_generation=sempipes.LLM(name="openai/qwen3-coder-next", parameters={**SAIA, "presence_penalty": 1.0}),
    llm_for_batch_processing=sempipes.LLM(name="openai/qwen3.6-35b-a3b", parameters=SAIA),
    batch_size_for_batch_processing=10,
)
```

**Other models.** SAIA's model list changes over time. List the current models, including which inputs (text, image) each one accepts:

```bash
curl -s -H "Authorization: Bearer $SAIA_API_KEY" "$SAIA_BASE_URL/models"
```

Then choose the models in `.env`, which also works together with `SEMPIPES_LLM_PROVIDER=saia`:

```
SAIA_CODE_GENERATION_MODEL=<model id>
SAIA_BATCH_PROCESSING_MODEL=<model id>
```

or in code: `get_default_saia_config(code_generation_model="...", batch_processing_model="...")`. Arguments take precedence over the environment variables. For code generation, prefer a code-specialised instruct model (such as the default `qwen3-coder-next`) and avoid models with `"thought"` in their output list: reasoning models are slower and their thinking text can confuse the code extraction.

**Rate limits.** The free tier allows **10 requests per minute, 200 per hour and 400 per day**. The code-generating examples need 1-15 requests each and run fine. The per-row examples (`04_sem_fillna.py` part A, `05`, `06`) send 12-100 requests and will hit the per-minute limit: run them on fewer rows or with Gemini.

**Keep your key private.** The key is stored in the config, so `print(sempipes.get_config())` shows it. Don't print the config in notebooks you share. Saved optimiser trajectories (`.sempipes_trajectories/`) leave the key out automatically.

**Worker processes** don't see `_set_config` (see [the caveat below](#caveat-the-config-is-per-thread--per-context)), but they do pick up `SEMPIPES_LLM_PROVIDER=saia` from `.env`. With `_set_config`, use `n_jobs=1` when the folds of a cross-validation call the LLM.

## Different models for code generation and per-row processing

The two roles have different needs, which is why they are configured separately:

* **Code generation** happens rarely (once per fit) and the code has to work. Use a strong model here. Weak models burn through `max_retries_for_code_gen` and then fall back or fail.
* **Per-row processing** happens for every row, so cost and speed matter. A smaller, cheaper model is often good enough.
* **Images and audio** need a **multimodal** model. In per-row `sem_extract_features` that is the batch model. The feature-proposal step (`output_columns=None`) also sends sample images and audio to the **code-generation** model. The default `gemini/gemini-2.5-flash` accepts text, images and audio.

## Rate limits and batch size

Per-row operators send `batch_size_for_batch_processing` requests at the same time. On free tiers (e.g. Gemini) this quickly produces `RateLimitError` / HTTP 429. Fixes:

```python
sempipes.update_config(batch_size_for_batch_processing=5)
```

* Work on fewer rows while developing: `df.head(100)` or `.skb.subsample(n=100)` for previews.
* Failed requests are handled differently per operator. `sem_extract_features` (per-row) and `sem_augment` (direct mode) retry failed rows up to 5 rounds. `sem_fillna(with_llm_only=True)` does **not** retry: a single failed request makes the whole step raise.

## Retries

`max_retries_for_code_gen` bounds the cost of a failing code-generation step. Each failed attempt adds the previous code and the error message to the conversation, so later attempts are longer (and more expensive) requests.

```python
sempipes.update_config(max_retries_for_code_gen=3)
```

## Verbose output

```python
sempipes.update_config(verbose_code_synthesis=True)
```

This logs the code that was accepted and the full traceback of each failed attempt for the optimisable operators. Without it, a failed attempt is logged as a compact line `SYNTHESIS_ERROR,<attempt>,<exception type>,<message>`. For `sem_extract_features` in code mode, the code is printed before each validation run either way.

## Logging

`import sempipes` configures Python's root logger at level `INFO` with the format

```
2026-01-01 12:00:00,000 - INFO - SEMPIPES> <message>
```

It also mutes the noisy `litellm`, `LiteLLM`, `httpx` and `sdv` loggers (level `ERROR`) and sets `LITELLM_LOGS=error`, unless you already set that variable. Useful lines to grep for:

| Prefix | Meaning |
|---|---|
| `TOKEN_USAGE,<in>,<out>` | Input and output tokens of one LLM response. Sum these to estimate cost. |
| `SYNTHESIS_ERROR,<attempt>,<type>,<msg>` | A code-generation attempt failed and will be retried |
| `Querying '<model>' with …` | An LLM request is about to be sent |
| `COLOPRO>`, `OP-SELECTION>`, `TREE_SEARCH>`, `MCT_SEARCH>` | Progress of the [optimiser](optimizer.md) |

Many operators also `print(...)` progress messages (e.g. `--- sempipes.sem_clean(...)`), which are not affected by logging levels. To quieten SemPipes' own log messages:

```python
import logging
logging.getLogger("sempipes.logging").setLevel(logging.WARNING)
```

## Caveat: the config is per thread / per context

The config is stored in a `contextvars.ContextVar`. This has a few consequences:

* In a script or a Jupyter notebook, `update_config(...)` affects everything that runs afterwards **in the same thread**. A change made in one notebook cell is still visible in later cells.
* **New threads start with the default config**, not with your changes.
* **Worker processes** (e.g. `n_jobs=-1` in `skrub.cross_validate`, randomized search, or the optimiser's cross-validation) import SemPipes from scratch. They see the default config (including the provider chosen with `SEMPIPES_LLM_PROVIDER` in `.env`) and your `.env` keys, but **not** your `update_config` changes. If an LLM call happens inside such a worker, it uses the default model. This affects every non-optimisable semantic operator that is refitted in each fold. Use `n_jobs=1` when you rely on a non-default config and the folds call the LLM.

In practice: call `update_config` once at the top of your script or notebook, before building the pipeline, and avoid parallel cross-validation when the folds call the LLM.
